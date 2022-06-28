_base_ = [
    '../_base_/datasets/dsprite.py',
    '../_base_/default_runtime.py'
]
# optimizer
optimizer = dict(type='Adam', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[10, 15])
runner = dict(type='EpochBasedRunner', max_epochs=20)

# model settings
model = dict(
    type='KFTaskParallelImageClassifier',
    kd_loss=dict(type='Logits'),
    train_cfg=dict(lambda_kd=0.1,
                   lambda_feat=1.0,
                   alpha=1.0,
                   beta=1e-3,
                   task_weight=1.0,
                   teacher_checkpoint='/home/yangxingyi/NeuralFactor/NeuralFactor/work_dirs/resnet18_b128x1_cifar10/latest.pth',
                   feat_channels=dict(student=[64, 128, 256, 512],
                                      teacher=[64, 128, 256, 512]),
                   infor_loss='l2'
                   ),
    backbone=dict(
        num_task=5,
        student=dict(
            CKN=dict(type='SimpleConv64',
                     latent_dim=10,
                     num_channels=1,
                     image_size=64),
            TSN=dict(type='SimpleGaussianConv64',
                     atent_dim=10,
                     num_channels=1,
                     image_size=64)
        ),
        teacher=dict(type='SimpleConv64',
                     latent_dim=10,
                     num_channels=1,
                     image_size=64)
    ),
    neck=dict(
        student=dict(type='GlobalAveragePooling'),
        teacher=dict(type='GlobalAveragePooling')
    ),
    head=dict(
        student=dict(
            type='MultiTaskLinearClsHead',
            num_classes=[3, 6, 40, 32, 32],
            in_channels=10,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        ),
        task=dict(
            type='MutiTaskClsIMBLinearClsHead',
            num_classes=[3, 6, 40, 32, 32],
            in_channels=10,
            loss=dict(type='CrossEntropyLoss',
                      use_sigmoid=True,
                      loss_weight=1.0),
        ),
        teacher=dict(
            type='MultiTaskLinearClsHead',
            num_classes=[3, 6, 40, 32, 32],
            in_channels=10,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        )
    )
)

checkpoint_config = dict(interval=5)
