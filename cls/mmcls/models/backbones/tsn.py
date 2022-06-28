from re import S
import torch.nn as nn
import torch

from ..builder import BACKBONES, build_backbone
from .base_backbone import BaseBackbone
import torch.nn.functional as F

@BACKBONES.register_module()
class TSN_backbone(BaseBackbone):
    def __init__(self, backbone, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder = build_backbone(backbone)
        self.fc = nn.Linear(self.in_channels, self.out_channels, bias=False)


    def forward(self, x):
        x = self.encoder(x)
        if isinstance(x, tuple):
            x = x[-1]
        
        x = F.adaptive_avg_pool2d(x, (1,1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        mu = torch.mean(x, 0)
        log_var = torch.log(torch.var(x, 0))
        return (mu, log_var), x
