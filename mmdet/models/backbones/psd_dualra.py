from mmengine import MessageHub
from torch import nn
import torch
import torch.nn.functional as F
import os
from torchvision.utils import save_image
from mmdet.registry import MODELS


# =========================
# 0) 轻量 high-pass（用于空域一致性）
# =========================
def _avg_blur(x: torch.Tensor, k: int = 5) -> torch.Tensor:
    if k <= 1:
        return x
    return F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)

def _highpass(x: torch.Tensor, blur_k: int = 5) -> torch.Tensor:
    return x - _avg_blur(x, k=blur_k)


# =========================
# 1) PSD 工具 + 高频mask + bound
# =========================
def _psd_gray_rfft(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B,3,H,W] -> PSD_gray: [B,1,H,W2]
    """
    x32 = x.to(torch.float32)
    X = torch.fft.rfft2(x32, norm='ortho')         # [B,C,H,W2]
    phi = X.abs().pow(2)                           # power
    return phi.mean(dim=1, keepdim=True)           # [B,1,H,W2]

def _psd_shape_norm(phi: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    归一化到 mean=1，强调“形状一致性”（scale 不敏感）
    phi: [B,1,H,W2]
    """
    return phi / (phi.mean(dim=(2, 3), keepdim=True) + eps)

def _radial_hf_mask(H: int, W2: int, device, r0: float = 0.4) -> torch.Tensor:
    """
    rfft2 半谱网格：y[-1,1], x[0,1]
    返回: [1,1,H,W2]，高频区域=1
    """
    yy = torch.linspace(-1.0, 1.0, H, device=device).view(H, 1)
    xx = torch.linspace(0.0, 1.0, W2, device=device).view(1, W2)
    rr = torch.sqrt(yy ** 2 + xx ** 2)
    m = (rr >= float(r0)).float()
    return m.view(1, 1, H, W2)

def psd_upper_bound_hf_loss(phi_n: torch.Tensor,
                            phi_x_gray: torch.Tensor,
                            rho: float = 1.02,
                            r0: float = 0.4,
                            use_log: bool = True,
                            eps: float = 1e-6) -> torch.Tensor:
    """
    只在高频区域惩罚 phi_n > rho * phi_x_gray
    """
    H, W2 = phi_n.shape[2], phi_n.shape[3]
    m = _radial_hf_mask(H, W2, phi_n.device, r0=r0)

    if use_log:
        diff = (torch.log(phi_n + eps) - torch.log(rho * phi_x_gray + eps)).relu()
        return (diff * m).sum() / (m.sum() + eps)
    else:
        excess = (phi_n - rho * phi_x_gray).relu()
        return (excess * m).sum() / (m.sum() + eps)


# =========================
# 2) 论文 Neighboring Masking 两视图（替代你之前的 quarter split）
#    2x2 patch 中随机选一对“相邻像素”，分别放到 D1/D2（H/2, W/2）
# =========================
def make_neighboring_mask_views(x: torch.Tensor, mode: str = "random"):
    """
    x: [B,C,H,W] -> D1,D2: [B,C,H/2,W/2]
    mode:
      - "random": 每个 2x2 patch 随机选 4 种相邻对之一
      - "fixed" : 固定选 (TL,TR) 对，推理时更稳定
    """
    B, C, H, W = x.shape
    H2, W2 = (H // 2) * 2, (W // 2) * 2
    x = x[:, :, :H2, :W2]

    xu = F.pixel_unshuffle(x, 2)                      # [B,4C,H/2,W/2]
    xu = xu.view(B, 4, C, H2 // 2, W2 // 2)           # [B,4,C,h,w]
    h, w = H2 // 2, W2 // 2

    # 0:TL, 1:TR, 2:BL, 3:BR
    pairs = torch.tensor([[0, 1], [0, 2], [1, 3], [2, 3]], device=x.device)  # 4x2

    if mode == "fixed":
        choice = torch.zeros((B, h, w), device=x.device, dtype=torch.long)  # always [0,1]
    else:
        choice = torch.randint(0, 4, (B, h, w), device=x.device)

    pair_idx = pairs[choice]  # [B,h,w,2]
    idx1 = pair_idx[..., 0].view(B, 1, 1, h, w).expand(-1, 1, C, -1, -1)
    idx2 = pair_idx[..., 1].view(B, 1, 1, h, w).expand(-1, 1, C, -1, -1)

    d1 = xu.gather(dim=1, index=idx1).squeeze(1)  # [B,C,h,w]
    d2 = xu.gather(dim=1, index=idx2).squeeze(1)
    return d1, d2


# =========================
# 3) N2N / 两视图一致性损失（默认不加，给你先检测器驱动）
# =========================
def n2n_spatial_loss_hp(pred_clean: torch.Tensor, target_noisy: torch.Tensor, hp_blur_k: int = 5):
    """
    用 high-pass 做 N2N：HP(clean_pred) ~ HP(noisy_target)
    更稳：减少低频亮度/色彩偏差导致的误导
    """
    hp_p = _highpass(pred_clean, blur_k=hp_blur_k)
    hp_t = _highpass(target_noisy, blur_k=hp_blur_k)
    return F.l1_loss(hp_p, hp_t)

def psd_log_l1_hf(phi_a: torch.Tensor, phi_b: torch.Tensor, r0: float = 0.4, eps: float = 1e-6) -> torch.Tensor:
    H, W2 = phi_a.shape[2], phi_a.shape[3]
    m = _radial_hf_mask(H, W2, phi_a.device, r0=r0)
    diff = (torch.log(phi_a + eps) - torch.log(phi_b + eps)).abs()
    return (diff * m).sum() / (m.sum() + eps)


# =========================
# 4) 二分支防坍缩正则（保留你的）
# =========================
def two_branch_nocollapse_reg(w_s: torch.Tensor, w_f: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    ws = w_s.clamp(min=eps, max=1.0)
    wf = w_f.clamp(min=eps, max=1.0)
    return (-(ws.log() + wf.log())).mean()


# =========================
# 5) Denoise：空域 + 频域 + 融合（aux 返回你要的量）
#    主路：保持你原来用 full-res x 提特征的写法
#    新增：训练时可选 N2N 两视图 consistency（默认权重=0，不会进总loss）
#    新增：保存 HP(d1)、HP(d2)、HP(clean1_s) 三张图
# =========================
class Denoise(nn.Module):
    def __init__(self,
                 chan_embed: int = 64,
                 depth: int = 3,
                 groups: int = 1,
                 psd_coarse_div: int = 8,
                 return_aux: bool = True,
                 detach_sigma_in_psd_loss: bool = True,

                 # --- N2N 两视图开关 ---
                 use_n2n_views: bool = False,
                 n2n_mode: str = "neighbor",          # neighbor / (你要保留旧版也可扩展)
                 n2n_view_fixed_in_eval: bool = True, # eval 时用 fixed 更稳
                 n2n_hp_blur_k: int = 5,
                 n2n_psd_r0: float = 0.4,

                 # --- debug：保存 N2N 的 high-pass 对齐情况 ---
                 debug_n2n_hp_vis: bool = False,
                 debug_n2n_hp_interval: int = 200,    # 每隔多少次 N2N 计算保存一次
                 debug_n2n_hp_max_b: int = 1,         # 每次最多存几个 batch 样本
                 ):
        super().__init__()
        c = chan_embed
        g = max(1, groups)
        self.psd_coarse_div = int(psd_coarse_div)
        self.return_aux = bool(return_aux)
        self.detach_sigma_in_psd_loss = bool(detach_sigma_in_psd_loss)

        self.use_n2n_views = bool(use_n2n_views)
        self.n2n_mode = str(n2n_mode)
        self.n2n_view_fixed_in_eval = bool(n2n_view_fixed_in_eval)
        self.n2n_hp_blur_k = int(n2n_hp_blur_k)
        self.n2n_psd_r0 = float(n2n_psd_r0)

        self.debug_n2n_hp_vis = bool(debug_n2n_hp_vis)
        self.debug_n2n_hp_interval = int(debug_n2n_hp_interval)
        self.debug_n2n_hp_max_b = int(debug_n2n_hp_max_b)
        self._n2n_hp_vis_step = 0  # 内部计数器（按 N2N 执行次数计）

        # spatial branch
        layers = [
            nn.Conv2d(3, c, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=False),
        ]
        for _ in range(depth):
            layers += [
                nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False),
                nn.Conv2d(c, c, 1, 1, 0, groups=g, bias=True),
                nn.LeakyReLU(0.2, inplace=False),
            ]
        self.spatial = nn.Sequential(*layers)
        self.s_out   = nn.Conv2d(c, 3, 1, 1, 0, bias=True)

        self.u_head   = nn.Sequential(nn.Conv2d(c, 1, 1, 1, 0, bias=True), nn.Sigmoid())
        self.gap      = nn.AdaptiveAvgPool2d(1)
        self.sigma_hd = nn.Sequential(nn.Conv2d(c, 1, 1, 1, 0, bias=True), nn.Softplus())
        self.w_head   = nn.Conv2d(c, 2, 1, 1, 0, bias=True)

        self.psd_head = nn.Sequential(
            nn.Conv2d(c, 32, 1, 1, 0, bias=True),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(32, 1, 1, 1, 0, bias=True),
            nn.Softplus()
        )
        self.psd_scale = nn.Parameter(torch.tensor(-3.0))  # softplus(-3)≈0.048

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def _save_noise_debug(self, noise_s, noise_f, U_map, res):
        # ===== 可视化（保持你原样）=====
        B = noise_s.shape[0]
        if not hasattr(self, 'vis_idx'):
            self.vis_idx = 0
        self.vis_idx += 1
        out_dir = 'work_dirs/ram_noise'
        os.makedirs(out_dir, exist_ok=True)

        def norm_minmax(t, eps_=1e-6):
            B_ = t.shape[0]
            t_flat = t.view(B_, -1)
            minv = t_flat.min(dim=1, keepdim=True).values.view(B_, 1, 1, 1)
            maxv = t_flat.max(dim=1, keepdim=True).values.view(B_, 1, 1, 1)
            return ((t - minv) / (maxv - minv + eps_)).clamp(0., 1.)

        # ns = norm_minmax(noise_s.abs().mean(dim=1, keepdim=True))

        ns_raw = noise_s.mean(dim=1, keepdim=True)  # 保留正负（更像噪声）
        flat = ns_raw.view(B, -1)
        # robust 尺度：用 99% 分位数的绝对值当 m（防 outlier）
        m = flat.abs().quantile(0.99, dim=1, keepdim=True).view(B,1,1,1).clamp_min(1e-6)
        ns = (0.5 + 0.5 * (ns_raw / m)).clamp(0.0, 1.0)

        nf = norm_minmax(noise_f.abs().mean(dim=1, keepdim=True))
        u  = norm_minmax(U_map)
        rr = norm_minmax(res.abs().mean(dim=1, keepdim=True))

        for b in range(B):
            base = f'{self.vis_idx:06d}_b{b:02d}'
            save_image(ns[b], os.path.join(out_dir, base + '_noiseS.png'))
            save_image(nf[b], os.path.join(out_dir, base + '_noiseF_wiener.png'))
            save_image(u[b],  os.path.join(out_dir, base + '_Umap.png'))
            save_image(rr[b], os.path.join(out_dir, base + '_noiseRes.png'))

    @torch.no_grad()
    def _save_n2n_hp_debug(self, d1, d2, clean1_s, hp_blur_k: int = 5):
        """
        保存三张图：
          - HP(d1)
          - HP(d2)
          - HP(clean1_s)
        为了可视化，把 high-pass 做对称归一化到 [0,1]
        """
        out_dir = 'work_dirs/ram_noise/n2n_hp'
        os.makedirs(out_dir, exist_ok=True)

        hp_d1 = _highpass(d1, blur_k=hp_blur_k)
        hp_d2 = _highpass(d2, blur_k=hp_blur_k)
        hp_c1 = _highpass(clean1_s, blur_k=hp_blur_k)

        B = d1.shape[0]
        nb = min(B, self.debug_n2n_hp_max_b)
        eps = 1e-6

        for b in range(nb):
            # 用同一个尺度归一化三张图，方便对比
            m = torch.stack([
                hp_d1[b].abs().view(-1).quantile(0.99),
                hp_d2[b].abs().view(-1).quantile(0.99),
                hp_c1[b].abs().view(-1).quantile(0.99),
            ]).amax().clamp_min(eps)

            def vis01(t):
                # [-m, m] -> [0,1]
                return (0.5 + 0.5 * (t / m)).clamp(0.0, 1.0)

            tag = f'{self._n2n_hp_vis_step:06d}_b{b:02d}'
            save_image(vis01(hp_d1[b]), os.path.join(out_dir, tag + '_HP_d1.png'))
            save_image(vis01(hp_d2[b]), os.path.join(out_dir, tag + '_HP_d2.png'))
            save_image(vis01(hp_c1[b]), os.path.join(out_dir, tag + '_HP_clean1s.png'))

    def _forward_core(self, x: torch.Tensor):
        """
        把你原 forward 主路封装一下，方便在 N2N 两视图上复用
        返回：
          res(noise_pred), aux（包含 x_wiener/phi_shape/phi_n/phi_x_gray/w_s/w_f/l_nocollapse/U_map/noise_s/noise_f）
        """
        B, C, H, W = x.shape
        W2 = W // 2 + 1
        dtype = x.dtype
        eps = 1e-6

        # ===== 空域 =====
        feat    = self.spatial(x)
        noise_s = self.s_out(feat)

        U_map  = self.u_head(feat)
        U_mean = U_map.mean(dim=(2, 3), keepdim=True)

        sigma     = self.sigma_hd(self.gap(feat))
        sigma_eff = (sigma + 0.06) * (1.0 - 0.6 * U_mean) + 1e-3

        # ===== 频域：PSD + Wiener =====
        Hc = max(8, H // self.psd_coarse_div)
        Wc = max(8, W2 // self.psd_coarse_div)
        feat_c = F.adaptive_avg_pool2d(feat, (Hc, Wc))

        phi_shape_c = self.psd_head(feat_c)  # >=0
        phi_shape_c = phi_shape_c / (phi_shape_c.mean(dim=(2, 3), keepdim=True) + eps)
        phi_shape = F.interpolate(phi_shape_c, size=(H, W2), mode='bilinear', align_corners=False)

        scale = F.softplus(self.psd_scale)
        sigma_g = sigma_eff.to(torch.float32)

        # Wiener 用的 phi_n
        phi_n_wiener = phi_shape.to(torch.float32) * (sigma_g ** 2) * scale + eps

        # loss 用的 phi_n（可 detach sigma 防爆）
        sigma_g_loss = sigma_g.detach() if self.detach_sigma_in_psd_loss else sigma_g
        phi_n_loss = phi_shape.to(torch.float32) * (sigma_g_loss ** 2) * scale + eps

        x32 = x.to(torch.float32)
        X = torch.fft.rfft2(x32, norm='ortho')
        phi_x = X.abs().pow(2)
        phi_x_gray = phi_x.mean(dim=1, keepdim=True)

        phi_s = (phi_x_gray - phi_n_wiener).clamp_min(0.0)
        Hgain = phi_s / (phi_s + phi_n_wiener + eps)

        S_hat = X * Hgain
        x_wiener = torch.fft.irfft2(S_hat, s=(H, W), norm='ortho')        # float32
        noise_f = (x32 - x_wiener).to(dtype)

        # ===== 融合 =====
        strength = 0.1 + 0.2 * U_map
        # w  = torch.softmax(self.w_head(self.gap(feat)), dim=1)            # [B,2,1,1]
        # w_s, w_f = w[:, :1], w[:, 1:]

        logits = self.w_head(self.gap(feat))   # [B,2,1,1]

        # 1) softmax temperature：T>1 会更“均匀”，不容易塌到 0/1
        T = 2.0
        w = torch.softmax(logits / T, dim=1)

        # 2) weight floor：保证每个分支至少占一点比重（不然空域分支永远学不动）
        alpha = 0.1   # 0.02~0.1 都可以试，先 0.05
        w = w * (1.0 - 2.0 * alpha) + alpha

        w_s, w_f = w[:, :1], w[:, 1:]

        res = (w_s * noise_s + w_f * noise_f) * strength                 # noise_pred

        l_nocollapse = two_branch_nocollapse_reg(w_s, w_f, eps=1e-6)

        aux = dict(
            x_wiener=x_wiener.to(dtype),
            phi_n=phi_n_loss,
            phi_shape=phi_shape.to(torch.float32),
            phi_x_gray=phi_x_gray.detach(),
            w_s=w_s, w_f=w_f,
            l_nocollapse=l_nocollapse,
            U_map=U_map,
            noise_s=noise_s,
            noise_f=noise_f,
            sigma_eff=sigma_eff.to(torch.float32),
        )
        return res, aux

    def forward(self, x):
        # ===== 主路（保持你原样）=====
        res, aux = self._forward_core(x)

        # ===== 可视化（保持你原样：eval 时落盘）=====
        if not self.training:
            with torch.no_grad():
                self._save_noise_debug(aux['noise_s'], aux['noise_f'], aux['U_map'], res)

        # ===== N2N 两视图 consistency（训练时可算；默认外部权重=0 不加到总 loss）=====
        # 只把 loss 放 aux 里，PSDDualRAWAdapter 决定是否写 hub
        l_n2n_sp = None
        l_n2n_psd = None
        l_n2n_total = None

        if self.training and self.use_n2n_views:
            # 用 neighbor masking 生成两视图（H/2,W/2）
            mode = "random"
            d1, d2 = make_neighboring_mask_views(x, mode=mode)

            # 空域：用“残差预测噪声 -> clean = noisy - noise”做 N2N（建议先 HP 版）
            # 这里为了最小改动：只用空域残差 head（不把 wiener 分支卷进 N2N，避免爆）
            _, aux1 = self._forward_core(d1)
            _, aux2 = self._forward_core(d2)

            clean1_s = d1 - aux1['noise_s']
            clean2_s = d2 - aux2['noise_s']

            # ===== debug：保存 HP(d1), HP(d2), HP(clean1_s) =====
            if self.debug_n2n_hp_vis:
                self._n2n_hp_vis_step += 1
                if (self._n2n_hp_vis_step % self.debug_n2n_hp_interval) == 0:
                    self._save_n2n_hp_debug(d1, d2, clean1_s, hp_blur_k=self.n2n_hp_blur_k)

            # ===== 1) HF noise-diff matching：HP(noise1 - noise2) ~ HP(d1 - d2) =====
            hp_t = _highpass((d1 - d2).detach(), blur_k=self.n2n_hp_blur_k)            # target HF diff
            hp_p = _highpass((aux1['noise_s'] - aux2['noise_s']), blur_k=self.n2n_hp_blur_k)  # pred HF diff
            l_diff = F.l1_loss(hp_p, hp_t)

            # ===== 2) low-frequency suppression：让 noise_s 不要承载颜色/亮度 =====
            lf1 = _avg_blur(aux1['noise_s'], k=self.n2n_hp_blur_k)
            lf2 = _avg_blur(aux2['noise_s'], k=self.n2n_hp_blur_k)
            l_lf = (lf1.abs().mean() + lf2.abs().mean())

            hp_c1 = _highpass(clean1_s, blur_k=self.n2n_hp_blur_k)
            hp_c2 = _highpass(clean2_s, blur_k=self.n2n_hp_blur_k)
            l_clean = F.l1_loss(hp_c1, hp_c2)

            # 组合：主力是 l_diff（直接学HF噪声），l_lf 防颜色漂移，l_clean 保边
            l_n2n_sp = 1.0 * l_diff + 0.2 * l_lf + 0.5 * l_clean



            # 频域：用观测谱 Φn_obs = 0.5*PSD(d1-d2) 监督 Φn_pred(view)
            eps = 1e-6
            phi_n_obs = 0.5 * _psd_gray_rfft((d1 - d2).to(torch.float32)) + eps
            # Φn_pred(view) = phi_shape * sigma_eff^2 * scale（复用你主路的定义）
            scale = F.softplus(self.psd_scale).to(torch.float32)

            phi_n1 = aux1['phi_shape'] * (aux1['sigma_eff'] ** 2) * scale + eps
            phi_n2 = aux2['phi_shape'] * (aux2['sigma_eff'] ** 2) * scale + eps

            l_n2n_psd = psd_log_l1_hf(phi_n1, phi_n_obs.detach(), r0=self.n2n_psd_r0, eps=eps) + \
                        psd_log_l1_hf(phi_n2, phi_n_obs.detach(), r0=self.n2n_psd_r0, eps=eps)

            l_n2n_total = l_n2n_sp + l_n2n_psd

        if self.return_aux:
            aux.update(dict(
                l_n2n_spatial=l_n2n_sp,
                l_n2n_psd=l_n2n_psd,
                l_n2n_total=l_n2n_total,
            ))
            return res, aux

        return res


# =========================
# 6) ISP相关模块（保持你原样）
# =========================
def conv_block(in_channels, out_channels, norm='BN'):
    if norm == 'IN':
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding="same"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU()
        )
    elif norm == 'BN':
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
    elif norm is None:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding="same"),
            nn.LeakyReLU()
        )

class RPEncoder(nn.Module):
    def __init__(self, img_size=256, in_channels=3, out_channels=128) -> None:
        super().__init__()
        self.img_size = img_size
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, 16, 7, padding="same"),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 5, padding="same"),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, out_channels, 3, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, x):
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear")
        return self.seq(x)

class RPDecoder(nn.Module):
    def __init__(self, out_channels, in_channels=128) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LeakyReLU(),
            nn.Linear(in_channels, out_channels),
        )

    def forward(self, x):
        return self.mlp(x)

def define_feature_fusion(in_channels, out_channels=3, ffm_params=None):
    if ffm_params is None:
        ffm_type = 'BN_HG'
        mid_channels = 32
    else:
        ffm_type = ffm_params.pop('ffm_type', 'BN_HG')
        mid_channels = ffm_params.pop('mid_channels', 32)

    if ffm_type == 'IN':
        ffm = nn.Sequential(
            conv_block(in_channels, 16, 'IN'),
            conv_block(16, mid_channels, 'IN'),
            nn.Conv2d(mid_channels, out_channels, 1, padding="same"),
        )
    elif ffm_type == 'HG':
        ffm = nn.Sequential(
            conv_block(in_channels, 16, None),
            conv_block(16, mid_channels, None),
            conv_block(mid_channels, 16, None),
            nn.Conv2d(16, out_channels, 1, padding="same"),
        )
    elif ffm_type == 'IN_HG':
        ffm = nn.Sequential(
            conv_block(in_channels, 16, 'IN'),
            conv_block(16, mid_channels, 'IN'),
            conv_block(mid_channels, 16, 'IN'),
            nn.Conv2d(16, out_channels, 1, padding="same"),
        )
    elif ffm_type == 'BN_HG':
        ffm = nn.Sequential(
            conv_block(in_channels, 8, 'BN'),
            conv_block(8, mid_channels, 'BN'),
            conv_block(mid_channels, 8, 'BN'),
            nn.Conv2d(8, out_channels, 1, padding="same"),
        )
    elif ffm_type == 'BN':
        ffm = nn.Sequential(
            conv_block(in_channels, 16, 'BN'),
            conv_block(16, mid_channels, 'BN'),
            nn.Conv2d(mid_channels, out_channels, 1, padding="same"),
        )
    return ffm


# =========================
# 7) PSDDualRAWAdapter：写 MessageHub
# =========================
@MODELS.register_module(name='PSDDualRAWAdapter')
class PSDDualRAWAdapter(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 img_size: int = 256,
                 out_channels: int = 64,
                 functions: list = None,
                 ffm_params: dict = None,
                 clamp_values: bool = False,

                 # ===== N2N 自监督开关（默认关）=====
                 use_n2n_views: bool = False,
                 consistency_weight: float = 0.0,
                 n2n_hp_blur_k: int = 5,
                 n2n_psd_r0: float = 0.4,

                 # ===== 二分支正则（默认关）=====
                 use_two_branch_reg: bool = False,

                 # ===== 高频谱上限（默认关）=====
                 use_psd_bound: bool = False,
                 psd_bound_rho: float = 1.02,
                 psd_hf_r0: float = 0.4,
                 psd_bound_use_log: bool = True,

                 **kwargs):
        super().__init__()
        self.message_hub = MessageHub.get_current_instance()
        self.in_channels = in_channels

        self.functions = functions if functions is not None else []
        assert len(self.functions) > 0, "functions list is empty"
        self.clamp_values = bool(clamp_values)

        # N2N
        self.use_n2n_views = bool(use_n2n_views)
        self.consistency_weight = float(consistency_weight)
        self.n2n_hp_blur_k = int(n2n_hp_blur_k)
        self.n2n_psd_r0 = float(n2n_psd_r0)

        # reg / bound
        self.use_two_branch_reg = bool(use_two_branch_reg)
        self.use_psd_bound = bool(use_psd_bound)
        self.psd_bound_rho = float(psd_bound_rho)
        self.psd_hf_r0 = float(psd_hf_r0)
        self.psd_bound_use_log = bool(psd_bound_use_log)

        # 去噪器：保留你原结构 + 可视化；N2N 开关从这里传入
        self.denoise = Denoise(
            return_aux=True,
            detach_sigma_in_psd_loss=True,
            use_n2n_views=self.use_n2n_views,
            n2n_hp_blur_k=self.n2n_hp_blur_k,
            n2n_psd_r0=self.n2n_psd_r0,
            # 你也可以在 config 里加 debug_n2n_hp_interval / debug_n2n_hp_max_b 覆盖
        )

        # ISP 分支
        self.encoder = RPEncoder(img_size=img_size,
                                 in_channels=in_channels,
                                 out_channels=out_channels)

        for function in self.functions:
            self.define_function(function,
                                 rpe_out_channels=out_channels,
                                 input_channels=in_channels)

        self.norm_layer = nn.BatchNorm2d(in_channels, affine=True)
        ffm_in_channels = in_channels * max(len(self.functions), 1)
        self.ffm = define_feature_fusion(in_channels=ffm_in_channels,
                                         ffm_params=ffm_params)

    def forward(self, x, training=True, **kwargs):
        training = bool(kwargs.get('training', training))
        outputs = []

        if self.clamp_values:
            x = torch.clamp(x, min=0)

        input_img = x

        # ===== 去噪 =====
        noise_pred, aux = self.denoise(x)
        x_deno = x - noise_pred

        # ===== 监控：w_s / w_f 均值（看空域分支是不是很弱）=====
        if (training and self.training) and (aux is not None):
            if aux.get('w_s', None) is not None:
                self.message_hub.update_scalar('train/w_s_mean', float(aux['w_s'].mean().detach()))
            if aux.get('w_f', None) is not None:
                self.message_hub.update_scalar('train/w_f_mean', float(aux['w_f'].mean().detach()))

        # ===== N2N consistency：默认 consistency_weight=0，不会影响总 loss =====
        if (training and self.training) and self.use_n2n_views and (aux is not None) and (aux.get('l_n2n_total', None) is not None):
            l_cons = aux['l_n2n_total']
            # 记录分量，方便你看是不是哪一项爆
            self.message_hub.update_info('ram_n2n_spatial', aux.get('l_n2n_spatial', None))
            self.message_hub.update_info('ram_n2n_psd', aux.get('l_n2n_psd', None))
            # 写入统一 key：detector 里用一个 factor 控制
            if self.consistency_weight > 0:
                self.message_hub.update_info('ram_consistency_loss', l_cons * self.consistency_weight)

        # ===== 高频谱上限（默认关；你后面要开再开）=====
        if (training and self.training) and self.use_psd_bound and (aux is not None):
            l_bound = psd_upper_bound_hf_loss(
                phi_n=aux['phi_n'],
                phi_x_gray=aux['phi_x_gray'],
                rho=self.psd_bound_rho,
                r0=self.psd_hf_r0,
                use_log=self.psd_bound_use_log
            )
            self.message_hub.update_info('ram_psd_bound_loss', l_bound)

        # ===== ISP 参数分支（保持你原样）=====
        x_rpe = self.encoder(x)
        for function in self.functions:
            out = self.apply_function(x_deno, function, x_rpe, training)
            outputs.append(out)

        x_fused = torch.cat(outputs, dim=1)
        x_fused = self.ffm(x_fused)
        x_fused = self.norm_layer(x_fused)

        # ===== 可视化（保持你原样）=====
        vis = torch.clamp(x_fused, 0.0, 1.0)
        if (not training) or (not self.training):
            self.message_hub.update_info('ram_vis_img', vis.detach())
            with torch.no_grad():
                B = input_img.shape[0]
                if not hasattr(self, 'vis_idx'):
                    self.vis_idx = 0
                self.vis_idx += 1
                out_dir = 'work_dirs/ram_noise'
                os.makedirs(out_dir, exist_ok=True)
                for b in range(B):
                    save_image(input_img[b].clamp(0., 1.),
                               os.path.join(out_dir, f'{self.vis_idx:06d}_b{b:02d}_input.png'))
                    save_image(vis[b],
                               os.path.join(out_dir, f'{self.vis_idx:06d}_b{b:02d}_enh.png'))

        return x_fused

    def learn_gamma(self, inputs, training):
        gamma = self.gamma(inputs).view(-1, 1, 1, 1)
        if gamma is not None and training:
            self.message_hub.update_scalar('train/gamma_val', float(gamma.mean()))
        return gamma

    def define_function(self, function, rpe_out_channels, input_channels):
        if function == 'gamma':
            self.gamma = nn.Sequential(
                RPDecoder(in_channels=rpe_out_channels, out_channels=1),
                nn.Sigmoid()
            )
        elif function == 'ccm':
            self.conv_cc = RPDecoder(in_channels=rpe_out_channels,
                                     out_channels=(input_channels ** 2))
        elif function == 'wb':
            self.conv_wb = RPDecoder(in_channels=rpe_out_channels,
                                     out_channels=input_channels)
        elif function == 'brightness':
            self.conv_bright = nn.Sequential(
                RPDecoder(in_channels=rpe_out_channels, out_channels=1),
                nn.Sigmoid()
            )

    def apply_function(self, input, function, x, training):
        bs = x.size(0)
        if function == 'gamma':
            gamma = self.learn_gamma(x, training)
            eps = 1e-3
            x_safe = torch.clamp(input, min=eps)
            out = torch.pow(x_safe, gamma)
            return out
            # gamma = self.learn_gamma(x, training)      # 原来是 (0,1)
            # gamma = 0.5 + 1.5 * gamma                  # -> (0.5, 2.0)

            # # 防止数值极端（再保险）
            # gamma = gamma.clamp(0.5, 2.0)

            # eps = 1e-3
            # x_safe = input.clamp(min=eps, max=1.0)     # 建议把 max 也 clamp
            # out = torch.pow(x_safe, gamma)
            # return out

        elif function == 'ccm':
            params_cc = self.conv_cc(x).reshape(bs, self.in_channels, self.in_channels)
            out = torch.einsum('bcij,bnc->bnij', input, params_cc)
            return out
        elif function == 'wb':
            params_wb = self.conv_wb(x).reshape(bs, self.in_channels, 1, 1)
            out = input * params_wb
            return out
        elif function == 'brightness':
            params_bright = self.conv_bright(x).view(-1, 1, 1, 1)
            out = input + params_bright
            return out
        else:
            return x
