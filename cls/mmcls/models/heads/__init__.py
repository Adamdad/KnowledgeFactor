# Copyright (c) OpenMMLab. All rights reserved.
from .cls_head import ClsHead
from .factor_head import KFClsHead
from .linear_head import LinearBCEClsHead, LinearClsHead
from .multi_label_head import MultiLabelClsHead
from .multi_label_linear_head import MultiLabelLinearClsHead
from .multitask_linear_head import MultiTaskLinearClsHead
from .multitask_linear_imb_head import ClsIMBLinearClsHead, GlobalDiscriminator
from .multitask_linear_vib_head import (ClsVIBLinearClsHead,
                                        MultiTaskVIBLinearClsHead)
from .stacked_head import StackedLinearClsHead
from .vision_transformer_head import VisionTransformerClsHead

__all__ = [
    'ClsHead', 'LinearClsHead', 'StackedLinearClsHead', 'MultiLabelClsHead',
    'MultiLabelLinearClsHead', 'VisionTransformerClsHead', 'KFClsHead', 'MultiTaskLinearClsHead',
    'MultiTaskVIBLinearClsHead', 'ClsVIBLinearClsHead', 'GlobalDiscriminator', 'ClsIMBLinearClsHead', 
    'LinearBCEClsHead'
]
