# optimizer
optimizer = dict(type='AdamW', lr=0.003, weight_decay=0.3)
optimizer_config = dict(grad_clip=dict(max_norm=1.0))

# specific to vit pretrain
paramwise_cfg = dict(
    custom_keys={
        '.backbone.cls_token': dict(decay_mult=0.0),
        '.backbone.pos_embed': dict(decay_mult=0.0)
    })
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=10000,
    warmup_ratio=1e-4)
runner = dict(type='EpochBasedRunner', max_epochs=300)
