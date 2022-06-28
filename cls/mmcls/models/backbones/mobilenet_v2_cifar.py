from mmcls.models.backbones.mobilenet_v2 import MobileNetV2
from mmcls.models.builder import BACKBONES
from mmcv.cnn import ConvModule

from mmcls.models.utils import make_divisible

@BACKBONES.register_module()
class MobileNetV2_CIFAR(MobileNetV2):
    arch_settings = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 1],
                     [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2],
                     [6, 320, 1, 1]]
    def __init__(self,
                 widen_factor=1.,
                 out_indices=(7, ),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU6'),
                 norm_eval=False,
                 with_cp=False,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ]):
        super().__init__(widen_factor=widen_factor,
                         out_indices=out_indices,
                         frozen_stages=frozen_stages,
                         conv_cfg=conv_cfg,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         norm_eval=norm_eval,
                         with_cp=with_cp,
                         init_cfg=init_cfg)
        self.in_channels = make_divisible(32 * widen_factor, 8)
        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        
    def forward(self, x):
        x = self.conv1(x)
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)