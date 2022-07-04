_base_ = [
    '../_base_/datasets/imagenet_bs32_randaug.py',
    '../_base_/schedules/imagenet_bs256_coslr.py'
]

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

fp16 = dict(loss_scale=512.)
# model settings
model = dict(
    type='KDImageClassifier',
    kd_loss=dict(type='SoftTarget',
                 temperature=10.0),
    train_cfg=dict(
        augments=[
            dict(type='BatchMixup', alpha=0.1,
                 num_classes=1000, prob=0.5)
        ],
        lambda_kd=0.1,
        teacher_checkpoint=None, # Input your teacher checkpoint
    ),
    backbone=dict(
        student=dict(type='ResNet',
                     depth=18,
                     num_stages=4,
                     out_indices=(1, 2, 3),
                     style='pytorch'),
        teacher=dict(type='ResNet',
                     depth=18,
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
            in_channels=128,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5),
        ),
        teacher=dict(
            type='LinearClsHead',
            num_classes=1000,
            in_channels=128,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5),
        )
    ))

checkpoint_config = dict(max_keep_ckpts=1)
