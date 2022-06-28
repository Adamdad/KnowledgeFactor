_base_ = [
    '../_base_/datasets/shape3d.py',
    '../_base_/default_runtime.py'
]
# optimizer
optimizer = dict(type='Adam', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[3])
runner = dict(type='EpochBasedRunner', max_epochs=5)

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SimpleConv64',
        latent_dim=10,
        num_channels=3,
        image_size=64),
    head=dict(
        type='MultiTaskLinearClsHead',
        num_classes=[10, 10, 10, 8, 4, 15],
        in_channels=10,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)

checkpoint_config = dict(interval=5)
