import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (build_conv_layer, build_norm_layer)

from .resnet import ResNet, WideBasicBlock
from ..builder import BACKBONES



@BACKBONES.register_module()
class WideResNet_CIFAR(ResNet):
    """Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of
    channels which is twice larger in every block. The number of channels
    in outer 1x1 convolutions is the same, e.g. last block in ResNet-50
    has 2048-512-2048 channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    arch_settings = {
        28: (WideBasicBlock, (4, 4, 4)),
    }

    def __init__(self, depth, out_channel, deep_stem=False,
                 norm_cfg=dict(type='BN',
                               momentum=0.1,
                               requires_grad=True),
                 **kwargs):
        super(WideResNet_CIFAR, self).__init__(
            depth,
            deep_stem=deep_stem,
            norm_cfg=norm_cfg, **kwargs)
        assert not self.deep_stem, 'ResNet_CIFAR do not support deep_stem'
        self.norm_cfg = norm_cfg
        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, out_channel, postfix=1)
        self.add_module(self.norm1_name, norm1)

    def _make_stem_layer(self, in_channels, base_channels):
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            base_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)

        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                if i == self.out_indices[-1]:
                    x = self.relu(self.norm1(x))
                else:
                    x = self.relu(x)
                outs.append(x)
        else:
            return tuple(outs)