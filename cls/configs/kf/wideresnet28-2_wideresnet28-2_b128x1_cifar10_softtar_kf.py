_base_ = [
    '../_base_/datasets/cifar10_bs128.py'
]

# 93.58
# checkpoint saving
checkpoint_config = dict(interval=10)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# optimizer
optimizer = dict(type='SGD',
                 lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[100, 150])
runner = dict(type='EpochBasedRunner', max_epochs=200)


# model settings
model = dict(
    type='KFImageClassifier',
    kd_loss=dict(type='SoftTarget',
                 temperature=10.0),
    train_cfg=dict(
        lambda_kd=0.1,
        lambda_feat=1.0,
        alpha=1.0,
        beta=1e-3,
        task_weight=1.0,
        teacher_checkpoint='/home/yangxingyi/NeuralFactor/NeuralFactor/work_dirs/wideresnet28-2-b128x1_cifar10/latest.pth',
        feat_channels=dict(student=[32, 64, 128],
                           teacher=[32, 64, 128]),
    ),
    backbone=dict(
        num_task=1,
        student=dict(
            CKN=dict(
                type='WideResNet_CIFAR',
                depth=28,
                stem_channels=16,
                base_channels=16 * 2,
                num_stages=3,
                strides=(1, 2, 2),
                dilations=(1, 1, 1),
                out_indices=(0, 1, 2),
                out_channel=128,
                style='pytorch'),
            TSN=dict(type='TSN_backbone',
                     backbone=dict(type='MobileNetV2_CIFAR',
                                   out_indices=(7, ),
                                   widen_factor=0.5),
                     in_channels=1280,
                     out_channels=128)
        ),
        teacher=dict(
            type='WideResNet_CIFAR',
            depth=28,
            stem_channels=16,
            base_channels=16 * 2,
            num_stages=3,
            strides=(1, 2, 2),
            dilations=(1, 1, 1),
            out_indices=(0, 1, 2),
            out_channel=128,
            style='pytorch')
    ),
    neck=dict(
        student=dict(type='GlobalAveragePooling'),
        teacher=dict(type='GlobalAveragePooling')
    ),
    head=dict(
        student=dict(
            type='LinearClsHead',
            num_classes=10,
            in_channels=128,
            loss=dict(
                type='LabelSmoothLoss',
                label_smooth_val=0.1,
                num_classes=10,
                reduction='mean',
                loss_weight=1.0),
        ),
        task=dict(
            type='LinearClsHead',
            num_classes=10,
            in_channels=128,
            loss=dict(
                type='LabelSmoothLoss',
                label_smooth_val=0.1,
                num_classes=10,
                reduction='mean',
                loss_weight=1.0),
        ),
        teacher=dict(
            type='LinearClsHead',
            num_classes=10,
            in_channels=128,
            loss=dict(type='CrossEntropyLoss',
                      loss_weight=1.0),
        )
    )
)

evaluation = dict(interval=5)
checkpoint_config = dict(max_keep_ckpts=1)
