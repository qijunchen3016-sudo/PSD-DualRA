# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from mmdet.datasets import CocoDataset


@DATASETS.register_module()
class PASCAL_RAW(CocoDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        'classes':
        ('car', 'person', 'bicycle'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42)]
    }
