import torch
from mmcv.parallel import is_module_wrapper
from mmcv.runner import (HOOKS, OPTIMIZER_BUILDERS, OPTIMIZERS,
                         DefaultOptimizerConstructor, Hook, OptimizerHook)
from mmcv.utils import build_from_cfg


@OPTIMIZER_BUILDERS.register_module()
class KDOptimizerBuilder(DefaultOptimizerConstructor):
    def __init__(self, optimizer_cfg, paramwise_cfg=None):
        super(KDOptimizerBuilder, self).__init__(optimizer_cfg,
                                                 paramwise_cfg)

    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module

        optimizer_cfg = self.optimizer_cfg.copy()

        # if no paramwise option is specified, just use the global setting
        if not self.paramwise_cfg:
            optimizer_cfg['params'] = model.student.parameters()
            student_optimizer = build_from_cfg(optimizer_cfg,
                                               OPTIMIZERS)
            return student_optimizer
