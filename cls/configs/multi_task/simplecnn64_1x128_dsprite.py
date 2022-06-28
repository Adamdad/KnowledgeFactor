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

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SimpleConv64',
        latent_dim=10,
        num_channels=1,
        image_size=64),
    head=dict(
        type='MultiTaskLinearClsHead',
        num_classes=[3, 6, 40, 32, 32],
        in_channels=10,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))

checkpoint_config = dict(interval=5)
