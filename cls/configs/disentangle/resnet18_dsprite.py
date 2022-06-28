_base_ = [
    '../_base_/datasets/cifar10_explain.py'
]
# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet_CIFAR',
        in_channels=1,
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiTaskLinearClsHead',
        num_classes=[3, 6, 40, 32, 32],
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0)
    )
)


load_from = '/home/yangxingyi/NeuralFactor/NeuralFactor/work_dirs/resnet18_1x128_dsprite/latest.pth'
# metric_names = ['dci', 'mig', 'sap_score', 'beta_vae', 'factor_vae', 'modularity_explicitness']
metric_names = ['beta_vae_sklearn', 'factor_vae_metric', 'modularity_explicitness']
data_name = 'dsprites_full'
data_path = '/home/yangxingyi/NeuralFactor/NeuralFactor/data'
model_attribute = dict(num_channels=1, image_size=64)
work_dir = 'disentangle_result/resnet18_1x128_dsprite'
