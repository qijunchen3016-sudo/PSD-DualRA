<div align="center">

# PSD-DualRA

**PSD-Guided Dual-Domain RAW Adaptation for Low-Light Object Detection**

Xin Feng, Qijun Chen, Yunlong Wang, Ziteng Wang, Yuhan Zhang, Ming Di<sup>†</sup>

Chongqing University of Technology, Chongqing 400054, China

<sup>†</sup>Corresponding author — `{DiMing, xfeng, ylwang}@cqut.edu.cn`

**Accepted at PRCV 2026** — Chinese Conference on Pattern Recognition and Computer Vision

</div>

---

Official implementation of **PSD-DualRA**, a noise-aware RAW-domain framework for
low-light object detection that jointly integrates denoising and ISP adaptation.

Low-light images suffer from low brightness, reduced contrast, lost detail and severe
noise, which the ISP pipeline further amplifies — especially degrading object boundaries
and textures for small or low-texture targets. Existing spatial-only denoisers struggle to
model non-white noise with complex spectral characteristics. PSD-DualRA explicitly models
noise in the **frequency domain** with a power spectral density (PSD), learns the spectral
shape of the noise, suppresses it with a **Wiener filter**, and pairs this with a **spatial
residual branch**. A high-frequency spectral constraint prevents noise over-amplification,
and the denoised representation drives a learnable **ISP adapter** for detector-friendly
features. The adapter is trained end-to-end with either **YOLOX** or **Faster R-CNN**.

![overview](https://cdn.jsdelivr.net/gh/qijunchen3016-sudo/PSD-DualRA@main/docs/figures/overview.png)

## Method

The pipeline has three jointly-optimized parts:

1. **RAW preprocessing & self-supervised noisy-view construction.** Each RAW file is decoded
   and packed into a 4-channel Bayer tensor `(R, G1, G2, B)`; when sensor black/white levels are
   available the tensor is normalized in the sensor domain. Two half-resolution views are built
   by **Neighboring Pixel Masking** (complementary neighbor sampling over each `2×2` neighborhood),
   giving a self-supervised noise signal.
2. **PSD-guided dual-domain denoiser.** A structured PSD parameterization separates *spectral
   shape* from *noise magnitude*, `φ̂_N(ω) = φ̂_shape(ω)·σ̂_eff²·s`. A Wiener gain
   `H(ω) = φ_S / (φ_S + φ̂_N + ε)` filters the noise-dominant frequencies, while a spatial branch
   predicts pixel-wise residuals; the two are combined with learnable, softmax-normalized and
   weight-floored fusion weights `Ŝ = w_s Ŝ_spa + w_f Ŝ_freq`.
3. **Dual-domain noise-aware objective.** `L = L_det + L_con + L_bound + L_reg^dual`, where
   `L_con = α·L_N2N + β·L_PSD` (high-frequency spatial consistency + PSD consistency against the
   neighboring-view difference), `L_bound` is a high-frequency spectral upper bound, and
   `L_reg^dual` is a negative-log barrier that keeps both branches alive.

The denoised RAW feeds a learnable ISP adapter (`wb`, `ccm`, `gamma`, `brightness`), whose
parameters are supervised by the detection loss.

## Repository layout

```text
configs/
  yolox/          # YOLOX experiments on LOD, NOD, and PascalRAW
  faster_rcnn/    # Faster R-CNN experiments on LOD, NOD, and PascalRAW
mmdet/
  datasets/lod.py, nod.py, pascal_raw.py            # dataset registrations + class lists
  models/backbones/psd_dualra.py                    # PSDDualRAWAdapter (PSDDualRAWAdapter)
  models/detectors/psd_dualra_yolox.py              # PSDDualRAYOLOX
  models/detectors/psd_dualra_faster_rcnn.py        # PSDDualRAFasterRCNN
tools/train.py, tools/test.py
docs/figures/     # qualitative / ablation figures used in this README
```

The repository is based on **MMDetection 3.3.0**. Generated work directories, datasets,
checkpoints, caches and duplicated package data are intentionally excluded.

## Environment

| Item | Version / note |
| --- | --- |
| Python | 3.10 |
| Deep-learning framework | PyTorch (GPU) |
| Detection toolbox | MMDetection 3.3.0 · MMEngine `>=0.7.1,<1.0.0` · MMCV `>=2.0.0rc4,<2.2.0` |
| CUDA | 12.2 |
| GPU | single NVIDIA A10 (all reported experiments) |
| Optimizer | SGD, momentum 0.9, weight decay `5e-4` |
| LR schedule | initial `0.01` (Faster R-CNN) / `0.0025` (LOD-Dark YOLOX), warm-up + multi-step / cosine decay |

Installation:

```bash
conda create -n psd-dualra python=3.10 -y
conda activate psd-dualra

# Install the PyTorch build that matches your CUDA toolkit, e.g.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -U openmim
mim install "mmengine>=0.7.1,<1.0.0" "mmcv>=2.0.0rc4,<2.2.0"
pip install -v -e .
```

Pick a torch/torchvision/cuXXX build from the official PyTorch index first; MMDetection 3.3.0's
`OpenMIM` install resolves the matching `mmcv` automatically.

## Datasets

PSD-DualRA is evaluated on four RAW low-light detection benchmarks. All of them are public —
please download from the original sources and follow their licenses.

| Dataset | Config `data_root` | Classes | Source / download |
| --- | --- | --- | --- |
| **LOD** (LOD-Normal, LOD-Dark) | `data/LOD_RAW/` | 8: car, motorcycle, bicycle, chair, dining table, bottle, tv, bus | [ying-fu/LODDataset](https://github.com/ying-fu/LODDataset) — *Crafting Object Detection in Very Low Light*, BMVC 2021 |
| **NOD-Nikon** | `data/NOD_Nikon/` | 3: person, bicycle, car | [igor-morawski/RAW-NOD](https://github.com/igor-morawski/RAW-NOD) — *GenISP*, CVPRW 2022 |
| **NOD-Sony** | `data/NOD_Sony/` | 3: person, bicycle, car | same as above |
| **PascalRAW** | `data/PascalRAW/` | 3: car, person, bicycle | Stanford Digital Repository — [purl.stanford.edu/hq050zr7488](http://purl.stanford.edu/hq050zr7488) (Omid-Zohoor et al., 2015) |

Expected on-disk layout (paths must match each config):

```text
data/
├── LOD_RAW/
│   ├── annotations/  all_dark-train.json, all_dark-val.json
│   └── dark_raw/     *.npy
├── NOD_Nikon/
│   ├── annotations/  raw_new_Nikon750_train.json, raw_new_Nikon750_test.json
│   └── images/       *.npy
├── NOD_Sony/
│   ├── annotations/  ...
│   └── images/       *.npy
└── PascalRAW/
    ├── annotations/  dng_instances_train.json, dng_instances_test.json
    └── dng/          *.npy
```

> **Preprocessing.** RAW files are decoded, rearranged into a packed Bayer tensor
> `(R, G1, G2, B)`, normalized with sensor black/white levels when available, resized to a fixed
> resolution and stored as `.npy` (consumed by the `LoadNumpyFromFile` → `RGGBtoRGB` pipeline).
> This follows the RAM baseline preprocessing, so you can reuse its conversion scripts to turn the
> official RAW releases above into the `.npy` layout expected here.

## Pretrained weights

This repo releases **code only**; the detector initializations are the public ones declared in
the configs:

| Component | Initialization | Where |
| --- | --- | --- |
| YOLOX backbones | MMDetection **YOLOX-Tiny COCO** checkpoint | [`yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth`](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth) |
| Faster R-CNN backbone | torchvision **ResNet-50** (ImageNet) | `init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')` |

PSD-DualRA is built on top of the **RAM** baseline (Raw Adaptation Module) — same data splits,
detector initialization and training protocol, with the RAW adaptation front-end replaced.
Baseline code: [SonyResearch/RawAdaptationModule](https://github.com/SonyResearch/RawAdaptationModule)
(*Beyond RGB: Adaptive Parallel Processing for RAW Object Detection*, ICCV 2025).

## Training and evaluation

```bash
# YOLOX
python tools/train.py configs/yolox/psd_dualra_yolox_lod_dark.py
python tools/train.py configs/yolox/psd_dualra_yolox_nod_nikon.py

# Faster R-CNN
python tools/train.py configs/faster_rcnn/psd_dualra_faster_rcnn_lod_dark.py

# Test
python tools/test.py CONFIG CHECKPOINT
```

Available configs:

| Detector | Config |
| --- | --- |
| Faster R-CNN | `configs/faster_rcnn/psd_dualra_faster_rcnn_{lod_normal, lod_dark, nod_nikon, nod_sony, pascalraw}.py` |
| YOLOX | `configs/yolox/psd_dualra_yolox_{lod_dark, nod_nikon, nod_sony, pascalraw}.py` |

## Results

Detection performance with **Faster R-CNN** (mAP / mAP₅₀):

| Method | LOD-Normal | LOD-Dark | NOD-Nikon | NOD-Sony | PascalRAW |
| --- | --- | --- | --- | --- | --- |
| RAW | 33.0 / 54.0 | 28.7 / 51.0 | 27.0 / 51.2 | 28.9 / 54.8 | 62.3 / 91.0 |
| sRGB | 34.8 / 57.8 | 29.0 / 52.2 | 27.8 / 52.5 | 29.2 / 55.1 | 64.8 / 92.4 |
| LogRGB (CVPR'24) | 35.7 / 57.6 | 32.2 / 54.7 | 28.4 / 52.8 | 27.5 / 51.6 | 64.1 / 91.5 |
| YOLA (NeurIPS'24) | 38.1 / 60.9 | 32.3 / 55.4 | 28.6 / 52.9 | 29.8 / 55.6 | 66.2 / 92.6 |
| FeatEnHancer (ICCV'23) | 37.5 / 59.9 | 32.2 / 55.1 | 28.9 / 53.3 | 30.2 / 55.7 | 66.5 / 92.7 |
| GenISP (CVPRW'22) | 38.1 / 60.1 | 32.6 / 55.1 | 29.8 / 54.6 | 30.7 / 53.4 | 64.8 / 91.8 |
| RAOD (CVPR'23) | 36.5 / 58.0 | 32.3 / 54.8 | 28.9 / 53.8 | 29.1 / 55.7 | 65.3 / 90.9 |
| RawAdapter (ECCV'24) | 37.4 / 59.6 | 32.2 / 55.1 | 30.4 / 55.9 | 31.8 / 53.6 | 66.1 / 92.0 |
| RAM (ICCV'25) | 40.1 / 61.6 | 34.9 / 57.6 | 31.0 / 56.3 | 32.4 / 54.2 | 66.8 / 92.6 |
| **PSD-DualRA** | **43.7 / 64.2** | **36.8 / 58.8** | **32.3 / 57.6** | **34.0 / 59.2** | **67.1 / 93.4** |

With **YOLOX-Tiny** on LOD-Dark and NOD-Sony, PSD-DualRA reaches **41.4 / 59.9** and
**35.0 / 58.0** mAP with only **0.0757 M** parameters (vs. 1.05 M for RAM).

### Qualitative results

Qualitative comparison with GenISP, RAOD and RawAdapter on representative RAW low-light scenes.
PSD-DualRA better preserves scene structures, giving more accurate and stable detections.

![qualitative](https://cdn.jsdelivr.net/gh/qijunchen3016-sudo/PSD-DualRA@main/docs/figures/qualitative.png)

Detection results with YOLOX on LOD-Dark and NOD-Sony.

![yolox](https://cdn.jsdelivr.net/gh/qijunchen3016-sudo/PSD-DualRA@main/docs/figures/yolox.png)

### Ablation

Qualitative results on LOD-Dark under different loss configurations, and the effect of the
spatial / frequency branches.

![ablation](https://cdn.jsdelivr.net/gh/qijunchen3016-sudo/PSD-DualRA@main/docs/figures/ablation_vis.png)

Attention-based visualization on three RAW low-light benchmarks — responses concentrate on
structure-sensitive regions (boundaries, wheels/frames, pedestrian contours).

![attention](https://cdn.jsdelivr.net/gh/qijunchen3016-sudo/PSD-DualRA@main/docs/figures/attention.png)

A representative failure case on NOD-Sony: small, weakly-lit pedestrians mixed with complex
background structures can still be missed.

![failure](https://cdn.jsdelivr.net/gh/qijunchen3016-sudo/PSD-DualRA@main/docs/figures/failure.png)

## Citation

If you find this code useful, please cite:

```bibtex
@inproceedings{psddualra2026,
  title     = {PSD-DualRA: PSD-Guided Dual-Domain RAW Adaptation for Low-Light Object Detection},
  author    = {Feng, Xin and Chen, Qijun and Wang, Yunlong and Wang, Ziteng and Zhang, Yuhan and Di, Ming},
  booktitle = {Pattern Recognition and Computer Vision (PRCV)},
  year      = {2026}
}
```

Accepted at **PRCV 2026**. Please also cite the RAM baseline and the dataset papers listed above.

## Acknowledgement

This research was supported by the Graduate Innovation Project of Chongqing Municipality,
Grant No. **CYS25755**.

## License

This project follows the **Apache License 2.0** inherited from MMDetection. See [LICENSE](LICENSE).
