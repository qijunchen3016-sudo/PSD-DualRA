_base_ = [
    '../_base_/default_runtime.py',
]

# =========================
# MODIFIED: yolox_s -> yolox_m
# =========================
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth'
# 例如如果你有本地权重，就直接写绝对路径
model = dict(
    type='PSDDualRAYOLOX',
    psd_bound_loss_factor=0.01,
    consistency_loss_factor=1.0,

    data_preprocessor=dict(
        type='DetDataPreprocessor',
        pad_size_divisor=32,
        mean=(0, 0, 0),
        std=(16383.0, 16383.0, 16383.0)
    ),

    backbone=dict(
        type='CSPDarknet',
        # YOLOX-M
        deepen_factor=0.67,
        widen_factor=0.75,
        out_indices=(2, 3, 4),
        use_depthwise=False,
        spp_kernal_sizes=(5, 9, 13),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),

        preproc_transform=dict(
            type='PSDDualRAWAdapter',
            functions=['wb', 'ccm', 'gamma', 'brightness'],
            ffm_params=dict(ffm_type='BN_HG'),
            clamp_values=True,

            # ===== N2N 两视图一致性 =====
            use_n2n_views=True,
            consistency_weight=0.1,
            n2n_hp_blur_k=5,
            n2n_psd_r0=0.4,

            # ===== 高频谱上限 =====
            use_psd_bound=True,
            psd_bound_rho=1.10,
            psd_hf_r0=0.4,
            psd_bound_use_log=True,

            # ===== 双分支防坍缩正则 =====
            use_two_branch_reg=True,
        ),
    ),

    neck=dict(
        type='YOLOXPAFPN',
        # YOLOX-M: 128/256/512 -> 192/384/768
        in_channels=[192, 384, 768],
        out_channels=192,
        num_csp_blocks=2,
        use_depthwise=False,
        upsample_cfg=dict(scale_factor=2, mode='nearest'),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish')
    ),

    bbox_head=dict(
        type='YOLOXHead',
        num_classes=3,
        in_channels=192,
        feat_channels=192,
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

base_lr = 0.01

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=base_lr,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True),
    paramwise_cfg=dict(
        norm_decay_mult=0.,
        bias_decay_mult=0.)
)

param_scheduler = [
    dict(
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=5,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='ConstantLR',
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    )
]

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(max_keep_ckpts=3, interval=5, save_best=['coco/bbox_mAP']),
    logger=dict(type='LoggerHook', interval=50),
)

custom_hooks = [
    dict(type='SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49)
]

train_pipeline = [
    dict(type='LoadNumpyFromFile'),
    dict(type='RGGBtoRGB'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=[(600, 320), (600, 400)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

test_pipeline = [
    dict(type='LoadNumpyFromFile'),
    dict(type='RGGBtoRGB'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(600, 400), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

dataset_type = 'NODDataset'
data_root = 'data/NOD_Nikon/'

train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/raw_new_Nikon750_train.json',
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline)
)

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/raw_new_Nikon750_test.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=test_pipeline)
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/raw_new_Nikon750_test.json',
    metric='bbox',
    format_only=False,
    classwise=True
)

test_evaluator = val_evaluator
