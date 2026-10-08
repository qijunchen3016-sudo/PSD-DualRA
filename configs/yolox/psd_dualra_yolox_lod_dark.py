_base_ = [
    '../_base_/default_runtime.py',
]

# 你当前本地下载的 yolox_m 权重
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth'

model = dict(
    type='PSDDualRAYOLOX',

    psd_bound_loss_factor=0.01,
    consistency_loss_factor=0.1,

    data_preprocessor=dict(
        type='DetDataPreprocessor',
        pad_size_divisor=32,
        mean=None,
        std=None,
    ),

    backbone=dict(
        type='CSPDarknet',
        deepen_factor=0.67,   # YOLOX-M
        widen_factor=0.75,    # YOLOX-M
        out_indices=(2, 3, 4),
        use_depthwise=False,
        spp_kernal_sizes=(5, 9, 13),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),

        preproc_transform=dict(
            type='PSDDualRAWAdapter',
            functions=['wb', 'ccm', 'gamma', 'brightness'],
            ffm_params=dict(ffm_type='BN_HG', mid_channels=32),
            clamp_values=True,

            use_n2n_views=True,
            consistency_weight=0.1,
            n2n_hp_blur_k=5,
            n2n_psd_r0=0.4,

            use_psd_bound=True,
            psd_bound_rho=1.10,
            psd_hf_r0=0.4,
            psd_bound_use_log=True,

            use_two_branch_reg=True,
        ),
    ),

    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[192, 384, 768],   # YOLOX-M
        out_channels=192,              # YOLOX-M
        num_csp_blocks=2,              # YOLOX-M
        use_depthwise=False,
        upsample_cfg=dict(scale_factor=2, mode='nearest'),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish')
    ),

    bbox_head=dict(
        type='YOLOXHead',
        num_classes=8,
        in_channels=192,       # YOLOX-M
        feat_channels=192,     # YOLOX-M
        stacked_convs=2,
        strides=(8, 16, 32),
        use_depthwise=False,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_bbox=dict(
            type='IoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=5.0),
        loss_obj=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_l1=dict(
            type='L1Loss',
            reduction='sum',
            loss_weight=1.0)
    ),

    train_cfg=dict(
        assigner=dict(type='SimOTAAssigner', center_radius=2.5)
    ),

    test_cfg=dict(
        score_thr=0.01,
        nms=dict(type='nms', iou_threshold=0.65)
    )
)

max_epochs = 300
num_last_epochs = 15
interval = 5

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=5
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

base_lr = 0.0025
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=base_lr,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True
    ),
    paramwise_cfg=dict(
        norm_decay_mult=0.,
        bias_decay_mult=0.
    )
)

param_scheduler = [
    dict(
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True
    ),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=5,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True
    ),
    dict(
        type='ConstantLR',
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    )
]

default_hooks = dict(
    checkpoint=dict(max_keep_ckpts=3, interval=5, save_best=['coco/bbox_mAP']),
    logger=dict(type='LoggerHook', interval=50),
)

# 这里只保留 SyncNormHook，先关闭 EMAHook
custom_hooks = [
    dict(type='SyncNormHook', priority=48),
]

train_pipeline = [
    dict(type='LoadNumpyFromFile'),
    dict(type='RGGBtoRGB'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=[(600, 320), (600, 400)],
        keep_ratio=True
    ),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id', 'img_path',
            'ori_shape', 'img_shape', 'pad_shape', 'batch_input_shape',
            'scale_factor', 'flip', 'flip_direction'
        )
    )
]

test_pipeline = [
    dict(type='LoadNumpyFromFile'),
    dict(type='RGGBtoRGB'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(600, 400), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id', 'img_path',
            'ori_shape', 'img_shape', 'pad_shape', 'batch_input_shape',
            'scale_factor', 'flip', 'flip_direction'
        )
    )
]

dataset_type = 'LODDataset'
data_root = 'data/LOD_RAW/'

train_dataloader = dict(
    batch_size=8,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/all_dark-train.json',
        data_prefix=dict(img='dark_raw/'),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/all_dark-val.json',
        data_prefix=dict(img='dark_raw/'),
        test_mode=True,
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/all_dark-val.json',
    metric='bbox',
    format_only=False,
    classwise=True
)
test_evaluator = val_evaluator