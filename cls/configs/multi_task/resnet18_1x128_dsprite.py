_base_ = [
    '../_base_/models/resnet18_dsprite.py',
    '../_base_/datasets/dsprite.py',
    '../_base_/schedules/dsprite_bs128.py', 
    '../_base_/default_runtime.py'
]
checkpoint_config = dict(interval=5)