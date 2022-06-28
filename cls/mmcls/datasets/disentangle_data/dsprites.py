# Copyright (c) OpenMMLab. All rights reserved.
import codecs
import numpy as np
import os
import os.path as osp
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info, master_only
from numpy import random

from mmcls.datasets.builder import DATASETS
from mmcls.datasets.utils import (download_and_extract_archive, download_url,
                                  rm_suffix)
from .multi_task import MultiTask


@DATASETS.register_module()
class dSprites(MultiTask):
    """Latent factor values
        Color: white
        Shape: square, ellipse, heart
        Scale: 6 values linearly spaced in [0.5, 1]
        Orientation: 40 values in [0, 2 pi]
        Position X: 32 values in [0, 1]
        Position Y: 32 values in [0, 1]
    """  # noqa: E501

    resources = 'https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
    split_ratio = 0.7
    num_tasks = 5
    CLASSES = [
    ]

    def load_annotations(self):
        filename = self.resources.rpartition('/')[2]
        data_file = osp.join(
            self.data_prefix, filename)

        if not osp.exists(data_file):
            self.download()

        _, world_size = get_dist_info()
        if world_size > 1:
            dist.barrier()
            assert osp.exists(data_file), \
                'Shared storage seems unavailable. Please download dataset ' \
                f'manually through {self.resource}.'

        data = np.load(data_file)
        num_data = len(data['imgs'])
        data_index = np.arange(num_data)
        np.random.seed(42)
        np.random.shuffle(data_index)
        train_index = data_index[:int(num_data*self.split_ratio)]
        test_index = data_index[int(num_data*self.split_ratio):]
        train_set = (data['imgs'][train_index],
                     data['latents_classes'][train_index][:, 1:])
        test_set = (data['imgs'][test_index],
                    data['latents_classes'][test_index][:, 1:])

        if not self.test_mode:
            imgs, gt_labels = train_set
        else:
            imgs, gt_labels = test_set

        data_infos = []
        for img, gt_label in zip(imgs, gt_labels):
            gt_label = np.array(gt_label, dtype=np.int64)
            info = {'img': img, 'gt_label': gt_label}
            data_infos.append(info)
        return data_infos

    @master_only
    def download(self):
        os.makedirs(self.data_prefix, exist_ok=True)

        # download files
        url = self.resources
        filename = url.rpartition('/')[2]
        download_url(url,
                     root=self.data_prefix,
                     filename=filename)
