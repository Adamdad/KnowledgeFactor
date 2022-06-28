# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='WideResNet_CIFAR',
        depth=28,
        stem_channels=16,
        base_channels=16 * 10,
        num_stages=3,
        strides=(1, 2, 2),
        dilations=(1, 1, 1),
        out_indices=(2, ),
        out_channel=640,
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=640,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))