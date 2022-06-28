# dataset settings
dataset_type = 'dSprites'
multi_task = True
img_norm_cfg = dict(
    mean=[0.5],
    std=[0.5],
    to_rgb=False)
train_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type, data_prefix='data/dsprite',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/dsprite',
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_prefix='data/dsprite',
        pipeline=test_pipeline,
        test_mode=True))
