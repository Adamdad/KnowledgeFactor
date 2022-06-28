
# model settings
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
    )
    )


load_from = '/home/yangxingyi/NeuralFactor/NeuralFactor/work_dirs/simplecnn64_1x128_dsprite/latest.pth'
metric_names = ['dci', 'mig', 'sap_score', 'beta_vae_sklearn', 'factor_vae_metric', 'modularity_explicitness']
# metric_names = ['beta_vae_sklearn', 'factor_vae_metric', 'modularity_explicitness']
data_name = 'dsprites_full'
data_path = '/home/yangxingyi/NeuralFactor/NeuralFactor/data'
model_attribute = dict(num_channels=1, image_size=64)
work_dir = 'disentangle_result/simplecnn64_1x128_dsprite'
