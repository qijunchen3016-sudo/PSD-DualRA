from typing import Optional, Dict
import torch
from mmengine import MessageHub
from mmdet.registry import MODELS
from mmdet.models.detectors.faster_rcnn import FasterRCNN


@MODELS.register_module()
class PSDDualRAFasterRCNN(FasterRCNN):
    """Faster R-CNN with PSD-DualRA auxiliary objectives.

    Reads the following values produced by the RAW adapter:
      - ram_psd_bound_loss
      - ram_consistency_loss

    改动点（重要）：
      1) 无论 factor 是否为 0，都先 pull + detach（避免 hub 持图导致显存/图泄漏）
      2) raw 指标用不包含 'loss' 的 key 记录，确保不参与 total loss
      3) 可选 aux_loss_clip 防止爆炸
    """

    HUB_KEYS = ('ram_psd_bound_loss', 'ram_consistency_loss')

    def __init__(self,
                 *args,
                 psd_bound_loss_factor: float = 0.0,
                 consistency_loss_factor: float = 0.0,
                 clear_hub_each_iter: bool = True,
                 log_hub_raw: bool = True,
                 aux_loss_clip: Optional[float] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.psd_bound_loss_factor = float(psd_bound_loss_factor)
        self.consistency_loss_factor = float(consistency_loss_factor)
        self.clear_hub_each_iter = bool(clear_hub_each_iter)
        self.log_hub_raw = bool(log_hub_raw)
        self.aux_loss_clip = None if aux_loss_clip is None else float(aux_loss_clip)

    def _model_device(self) -> torch.device:
        return next(self.parameters()).device

    def _clear_hub_keys(self, hub: MessageHub):
        for k in self.HUB_KEYS:
            hub.update_info(k, None)

    def _pull_tensor_from_hub(self, hub: MessageHub, key: str) -> Optional[torch.Tensor]:
        v = hub.get_info(key, None)
        if v is None:
            return None

        if torch.is_tensor(v):
            v = v.to(self._model_device())
        else:
            v = torch.tensor(float(v), device=self._model_device())

        # Avoid retaining an autograd graph in the global message hub.
        hub.update_info(key, v.detach())
        return v

    def _maybe_clip(self, v: torch.Tensor) -> torch.Tensor:
        if self.aux_loss_clip is None:
            return v
        return v.clamp(max=self.aux_loss_clip)

    def loss(self, batch_inputs, batch_data_samples, **kwargs) -> Dict[str, torch.Tensor]:
        hub = MessageHub.get_current_instance()

        # 先清空，避免读到上个 iter 残留
        if self.clear_hub_each_iter:
            self._clear_hub_keys(hub)

        # 先走检测器主 loss（RAM 一般在 backbone/preproc 里 forward，会在这里写 hub）
        losses = super().loss(batch_inputs, batch_data_samples, **kwargs)

        # ===== 不管 factor 是否为 0，都 pull 一次：释放 hub 的计算图引用 =====
        v_bound = self._pull_tensor_from_hub(hub, 'ram_psd_bound_loss')
        v_cons  = self._pull_tensor_from_hub(hub, 'ram_consistency_loss')

        # ===== raw 记录（不含 'loss'，不会被计入 total loss，但会出现在 log_vars）=====
        if self.log_hub_raw:
            if v_bound is not None:
                losses['ram_psd_bound_raw'] = v_bound.detach()
            if v_cons is not None:
                losses['ram_consistency_raw'] = v_cons.detach()

        # ===== 真正加入 total loss（只有 factor>0 才加）=====
        if (v_bound is not None) and (self.psd_bound_loss_factor > 0):
            vb = self._maybe_clip(v_bound)
            losses['loss_psd_bound'] = vb * self.psd_bound_loss_factor

        if (v_cons is not None) and (self.consistency_loss_factor > 0):
            vc = self._maybe_clip(v_cons)
            losses['loss_ram_consistency'] = vc * self.consistency_loss_factor

        return losses

# # PSDDualRAFasterRCNN.predict(...):
#     def predict(self, batch_inputs, batch_data_samples, rescale=True):
#         rescale = False   # ★ 临时定位：强制在 resize/pad 坐标系
#         return super().predict(batch_inputs, batch_data_samples, rescale=rescale)

# ####
