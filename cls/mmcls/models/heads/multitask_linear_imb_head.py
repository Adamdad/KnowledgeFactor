# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from numbers import Number
from torch._C import set_flush_denormal
from torch.nn.modules import loss

from ..builder import HEADS
from .cls_head import ClsHead
from .multitask_linear_vib_head import Common_encoder, Task_encoder, InfoMin_loss


@HEADS.register_module()
class GlobalDiscriminator(nn.Module):
    r"""
    input of GlobalDiscriminator is the `M` in Encoder.forward, so with
    channels : num_feature * 2, in_channels
    shape    : (input_shape[0]-3*2, input_shape[1]-3*2), M_shape
    """

    def __init__(self,
                 in_channels,
                 base_channels,
                 feat_channels,
                 alpha=0.1):
        super().__init__()

        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.base_channels = base_channels
        self.convs = nn.Sequential(
            nn.Conv2d(self.in_channels, self.base_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.base_channels, momentum=0.1),
            nn.ReLU(True),
            nn.Conv2d(self.base_channels, self.base_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.base_channels, momentum=0.1),
            nn.ReLU(True),
            nn.Conv2d(self.base_channels, self.base_channels,
                      kernel_size=3, stride=1, padding=1),
        )

        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf
        # input of self.l0 is the concatenate of E and flattened output of self.c1 (C)
        self.fcs = nn.Sequential(nn.Linear(self.base_channels, self.base_channels),
                                 nn.ReLU(True),
                                 nn.Linear(self.base_channels,
                                           self.base_channels),
                                 nn.ReLU(True),
                                 nn.Linear(self.base_channels, self.feat_channels))
        self.alpha = alpha

    def forward(self, img):
        feat_img = self.convs(img)
        feat_img = F.adaptive_avg_pool2d(feat_img, (1, 1))
        out = feat_img.view(feat_img.size(0), -1)
        out = self.fcs(out)

        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf
        # output of Table 5
        return out

    # def forward_train(self, img, feat):
    #     bs = img.size(0)
    #     if isinstance(feat, tuple):
    #         feat = feat[-1]

    #     img_feat = self(img)
    #     s = torch.matmul(feat, img_feat.permute(1, 0))
    #     mask_joint = torch.eye(bs).cuda()
    #     mask_marginal = 1 - mask_joint
    #     Ej =(s * mask_joint).mean()
    #     Em = torch.exp(s * mask_marginal).mean()
    #     # decoupled comtrastive learning?!!!!
    #     # infomax_loss = - (Ej - torch.log(Em)) * self.alpha
    #     infomax_loss = - (Ej - torch.log(Em)) * self.alpha
    #     return dict(infomax_loss=infomax_loss)
    def forward_train(self, img, feat):
        bs = img.size(0)
        if isinstance(feat, tuple):
            feat = feat[-1]
        img_feat = self(img)
        s = - torch.cdist(feat, img_feat, p=2)
        mask_joint = torch.eye(bs).cuda()
        mask_marginal = 1 - mask_joint

        Ej = (s * mask_joint).mean()
        Em = torch.exp(s * mask_marginal).mean()
        # decoupled comtrastive learning?!!!!
        # infomax_loss = - (Ej - torch.log(Em)) * self.alpha
        infomax_loss = - (Ej - torch.log(Em)) * self.alpha
        return dict(infomax_loss=infomax_loss)


@HEADS.register_module()
class ClsIMBLinearClsHead(ClsHead):
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
        super(ClsIMBLinearClsHead, self).__init__(
            init_cfg=init_cfg, *args, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fcs = nn.ModuleList(
            [nn.Linear(self.in_channels, 1)
             for i in range(self.num_classes)]
        )

    def simple_test(self, x):
        """Test without augmentation."""
        assert isinstance(x, list), 'input has to be a list of features'
        assert len(x) == self.num_classes, ('number of feature',
                                            'has to be matched with th number of classes')

        cls_scores = []
        for i in range(self.num_classes):
            if isinstance(x[i], tuple):
                input = x[i][-1]
            else:
                input = x[i]
            cls_score = self.fcs[i](input)
            cls_scores.append(cls_score)
        cls_score = torch.cat(cls_scores, dim=1)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(
            cls_score, dim=1) if cls_score is not None else None
        result = self.post_process(pred)
        return result

    def get_logits(self, x):
        assert isinstance(x, list), 'input has to be a list of features'
        assert len(x) == self.num_classes, ('number of feature',
                                            'has to be matched with th number of classes')
        logits = []
        for i in range(self.num_classes):
            if isinstance(x[i], tuple):
                input = x[i][-1]
            else:
                input = x[i]
            logit = self.fcs[i](input)
            logits.append(logit)
        logits = torch.cat(logits, dim=1)
        return logits

@HEADS.register_module()
class MutiTaskClsIMBLinearClsHead(ClsHead):
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
        super(MutiTaskClsIMBLinearClsHead, self).__init__(
            init_cfg=init_cfg, *args, **kwargs)
        assert isinstance(num_classes, list), 'num_classes must be positive for multitask classification'

        self.num_classes = num_classes
        self.num_task = len(self.num_classes)
        self.in_channels = in_channels
        self.fcs = nn.ModuleList(
            [nn.Linear(self.in_channels, self.num_classes[i])
             for i in range(self.num_task)]
        )

    def simple_test(self, x):
        """Test without augmentation."""
        assert isinstance(x, list), 'input has to be a list of features'
        assert len(x) == self.num_task, ('number of feature',
                                            'has to be matched with th number of classes')

        cls_scores = []
        for i in range(self.num_task):
            if isinstance(x[i], tuple):
                input = x[i][-1]
            else:
                input = x[i]
            cls_score = self.fcs[i](input)
            cls_scores.append(cls_score)
        cls_score = torch.cat(cls_scores, dim=1)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(
            cls_score, dim=1) if cls_score is not None else None
        result = self.post_process(pred)
        return result


    def get_logits(self, x):
        assert isinstance(x, list), 'input has to be a list of features'
        assert len(x) == self.num_task, ('number of feature',
                                            'has to be matched with th number of classes')

        logits = []
        for i in range(self.num_task):
            if isinstance(x[i], tuple):
                input = x[i][-1]
            else:
                input = x[i]
            logit = self.fcs[i](input)
            logits.append(logit)
        logits = torch.cat(logits, dim=1)
        return logits