_base_ = [
    '../_base_/datasets/imagenet_bs32_randaug.py',
    '../_base_/schedules/imagenet_bs256_coslr.py'
]


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

fp16 = dict(loss_scale=512.)
# model settings
model = dict(
    type='KFImageClassifier',
    kd_loss=dict(type='SoftTarget',
                 temperature=10.0),
    train_cfg=dict(
        augments=[
            dict(type='BatchMixup', alpha=0.1,
                 num_classes=1000, prob=0.5)
        ],
        lambda_kd=0.1,
        lambda_feat=1.0,
        alpha=1.0,
        beta=1e-3,
        task_weight=1.0,
        teacher_checkpoint=None, # Input your teacher checkpoint
        feat_channels=dict(student=[128, 256, 512],
                           teacher=[512, 1024, 2048]),
    ),
    backbone=dict(
        num_task=1,
        student=dict(
            CKN=dict(type='ResNet',
                     depth=18,
                     num_stages=4,
                     out_indices=(1, 2, 3),
                     style='pytorch'),
            TSN=dict(type='TSN_backbone',
                     backbone=dict(type='MobileNetV2',
                                   out_indices=(7, ),
                                   widen_factor=0.5),
                     in_channels=1280,
                     out_channels=512)
        ),
        teacher=dict(type='ResNet',
                     depth=50,
                     num_stages=4,
                     out_indices=(1, 2, 3),
                     style='pytorch'),
    ),
    neck=dict(
        student=dict(type='GlobalAveragePooling'),
        teacher=dict(type='GlobalAveragePooling')
    ),
    head=dict(
        student=dict(
            type='LinearClsHead',
            num_classes=1000,
            in_channels=512,
            loss=dict(
                type='LabelSmoothLoss',
                label_smooth_val=0.1,
                num_classes=1000,
                reduction='mean',
                loss_weight=1.0),
        ),
        task=dict(
            type='LinearClsHead',
            num_classes=1000,
            in_channels=512,
            loss=dict(
                type='LabelSmoothLoss',
                label_smooth_val=0.1,
                num_classes=1000,
                reduction='mean',
                loss_weight=1.0),
        ),
        teacher=dict(
            type='LinearClsHead',
            num_classes=1000,
            in_channels=2048,
            loss=dict(type='CrossEntropyLoss',
                      loss_weight=1.0),
        )
    )
)

checkpoint_config = dict(interval=10, max_keep_ckpts=2)
