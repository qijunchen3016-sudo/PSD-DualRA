# PSD-DualRA

Official implementation of **PSD-DualRA: PSD-Guided Dual-Domain RAW
Adaptation for Low-Light Object Detection**.

PSD-DualRA combines spatial residual restoration with PSD-guided Wiener
filtering, high-frequency constraints, and a learnable ISP adapter. The
adapter is trained jointly with either YOLOX or Faster R-CNN.

## Repository layout

```text
configs/
  yolox/          # YOLOX experiments on LOD, NOD, and PascalRAW
  faster_rcnn/    # Faster R-CNN experiments on LOD, NOD, and PascalRAW
mmdet/
  models/backbones/psd_dualra.py
  models/detectors/psd_dualra_faster_rcnn.py
  models/detectors/psd_dualra_yolox.py
tools/train.py
tools/test.py
```

The repository is based on MMDetection 3.3.0. Generated work directories,
datasets, checkpoints, caches, legacy baselines, and duplicated package data
are intentionally excluded.

## Installation

```bash
conda create -n psd-dualra python=3.10 -y
conda activate psd-dualra
pip install -U openmim
mim install "mmengine>=0.7.1,<1.0.0" "mmcv>=2.0.0rc4,<2.2.0"
pip install -v -e .
```

## Data

Place datasets under `data/` or edit `data_root` in the selected config:

```text
data/
  LOD_RAW/
  NOD_Nikon/
  NOD_Sony/
  PascalRAW/
```

Dataset annotations and directory names must match the paths in each config.

## Training and evaluation

```bash
python tools/train.py configs/yolox/psd_dualra_yolox_lod_dark.py
python tools/train.py configs/faster_rcnn/psd_dualra_faster_rcnn_lod_dark.py

python tools/test.py CONFIG CHECKPOINT
```

The YOLOX configs initialize from the public MMDetection YOLOX-Tiny COCO
checkpoint. Faster R-CNN uses the torchvision ResNet-50 initialization declared
in its config.

## Citation

Please cite the paper if this code is useful in your research. The final
BibTeX entry will be added when publication metadata is available.

## License

This project follows the Apache License 2.0 inherited from MMDetection. See
[LICENSE](LICENSE).
