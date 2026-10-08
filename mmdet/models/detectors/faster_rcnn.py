# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .two_stage import TwoStageDetector


@MODELS.register_module()
class FasterRCNN(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 rpn_head: ConfigType,
                 roi_head: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 neck: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)


# # Copyright (c) OpenMMLab. All rights reserved.
# from typing import Dict
# from torch import Tensor
# import torch.nn as nn

# from mmdet.registry import MODELS
# from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
# from .two_stage import TwoStageDetector


# @MODELS.register_module()
# class FasterRCNN(TwoStageDetector):
#     """Faster R-CNN with extra reconstruction loss (r*l ~= img)."""

#     def __init__(self,
#                  backbone: ConfigType,
#                  rpn_head: ConfigType,
#                  roi_head: ConfigType,
#                  train_cfg: ConfigType,
#                  test_cfg: ConfigType,
#                  neck: OptConfigType = None,
#                  data_preprocessor: OptConfigType = None,
#                  init_cfg: OptMultiConfig = None,
#                  # ★ 新增：重构损失权重（默认开，按需调）
#                  recon_loss_weight: float = 10.0) -> None:
#         super().__init__(
#             backbone=backbone,
#             neck=neck,
#             rpn_head=rpn_head,
#             roi_head=roi_head,
#             train_cfg=train_cfg,
#             test_cfg=test_cfg,
#             init_cfg=init_cfg,
#             data_preprocessor=data_preprocessor)

#         self.recon_loss_weight = float(recon_loss_weight)
#         self._recon_crit = nn.MSELoss()

#     def loss(self, batch_inputs, batch_data_samples, **kwargs) -> Dict[str, Tensor]:
#         # 先跑标准两阶段流程（会调用 backbone.forward()）
#         losses = super().loss(batch_inputs, batch_data_samples, **kwargs)

#         # 再从 backbone 缓存里取 r/l/img 来计算重构损失
#         if self.recon_loss_weight > 0:
#             cache = getattr(self.backbone, "_ram_cache", None)
#             if cache is None:
#                 # 兼容某些实现把权重放在 backbone 上
#                 self.recon_loss_weight = float(
#                     getattr(self.backbone, "recon_loss_factor", self.recon_loss_weight)
#                 )
#             else:
#                 r, l, img = cache.get("r"), cache.get("l"), cache.get("img")
#                 if (r is not None) and (l is not None) and (img is not None):
#                     pred = r * l  # 预测重建

#                     # 通道对齐（1↔3）
#                     if pred.size(1) == 1 and img.size(1) == 3:
#                         pred = pred.repeat(1, 3, 1, 1)
#                     elif pred.size(1) == 3 and img.size(1) == 1:
#                         img = img.repeat(1, 3, 1, 1)

#                     # dtype/device 对齐（以 img 为准）
#                     if pred.dtype != img.dtype:
#                         pred = pred.to(img.dtype)
#                     if pred.device != img.device:
#                         pred = pred.to(img.device)

#                     losses["loss_recon"] = self._recon_crit(pred, img) * self.recon_loss_weight

#         return losses
