# dataset settings
dataset_type = 'Shape3D'
multi_task = True
img_norm_cfg = dict(
    mean=[127.0, 127.0, 127.0],
    std=[127.0, 127.0, 127.0],
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
    samples_per_gpu=256,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type, data_prefix='data/shape3d',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/shape3d',
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_prefix='data/shape3d',
        pipeline=test_pipeline,
        test_mode=True))
