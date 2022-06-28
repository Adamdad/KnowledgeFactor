# Copyright (c) OpenMMLab. All rights reserved.
from .base_dataset import BaseDataset
from .builder import DATASETS, PIPELINES, build_dataloader, build_dataset
from .cifar import CIFAR10, CIFAR100, CIFAR10_MultiTask, CIFAR10_2Task, CIFAR10_Select
from .dataset_wrappers import (ClassBalancedDataset, ConcatDataset,
                               RepeatDataset)
from .imagenet import ImageNet, ImageNet_MultiTask
from .samplers import DistributedSampler
from .disentangle_data import dSprites, Shape3D


