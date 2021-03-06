# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS
from .cls_head import ClsHead


@HEADS.register_module()
class MultiTaskLinearClsHead(ClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 *args,
                 **kwargs):
        super(MultiTaskLinearClsHead, self).__init__(
            init_cfg=init_cfg, *args, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_task = len(self.num_classes)

        self.fcs = nn.ModuleList(
            [nn.Linear(self.in_channels, self.num_classes[i])
             for i in range(self.num_task)]
        )

    def simple_test(self, x):
        """Test without augmentation."""
        if isinstance(x, tuple):
            x = x[-1]
        preds = []
        for i in range(self.num_task):
            cls_score = self.fcs[i](x)
            if isinstance(cls_score, list):
                cls_score = sum(cls_score) / float(len(cls_score))
            pred = F.softmax(
                cls_score, dim=1) if cls_score is not None else None
            preds.append(self.post_process(pred))
        return preds

    def get_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        logits = []
        for i in range(self.num_task):
            logit = self.fcs[i](x)
            logits.append(logit)
        return logits

    def forward_train(self, x, gt_label):
        if isinstance(x, tuple):
            x = x[-1]
        losses = dict()
        for i in range(self.num_task):
            cls_score = self.fcs[i](x)
            loss_task = self.loss(cls_score, gt_label[:, i])['loss']
            losses[f'task{i}_loss'] = loss_task
        return losses
