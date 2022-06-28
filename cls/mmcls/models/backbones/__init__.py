# Copyright (c) OpenMMLab. All rights reserved.
from .mobilenet_v2 import MobileNetV2
from .mobilenet_v2_cifar import MobileNetV2_CIFAR
from .resnet import ResNet, ResNetV1d
from .resnet_cifar import ResNet_CIFAR
from .shufflenet_v2 import ShuffleNetV2
from .tsn import TSN_backbone
from .wideresnet import WideResNet_CIFAR
from .disentangle import SimpleConv64, SimpleGaussianConv64

__all__ = [
    'LeNet5', 'AlexNet', 'VGG', 'RegNet', 'ResNet', 'ResNeXt', 'ResNetV1d',
    'ResNeSt', 'ResNet_CIFAR', 'SEResNet', 'SEResNeXt', 'ShuffleNetV1',
    'ShuffleNetV2', 'MobileNetV2', 'MobileNetV3', 'VisionTransformer',
    'SwinTransformer', 'TNT', 'TIMMBackbone', 'MobileNetV2_CIFAR', 'TSN',
    'WideResNet_CIFAR', 'TSN_backbone', 'SimpleConv64', 'SimpleGaussianConv64'
]
