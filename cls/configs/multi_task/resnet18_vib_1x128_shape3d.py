_base_ = [
    '../_base_/models/resnet18_vib_shape3d.py',
    '../_base_/datasets/shape3d.py',
    '../_base_/schedules/shape3d_bs128.py', 
    '../_base_/default_runtime.py'
]
checkpoint_config = dict(interval=5)