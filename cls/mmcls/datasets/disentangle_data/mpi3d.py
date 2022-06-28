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
class MPI3d(MultiTask):
    """Factors	Possible Values
    object_color	white=0, green=1, red=2, blue=3, brown=4, olive=5
    object_shape	cone=0, cube=1, cylinder=2, hexagonal=3, pyramid=4, sphere=5
    object_size	    small=0, large=1
    camera_height	top=0, center=1, bottom=2
    background_color	purple=0, sea green=1, salmon=2
    horizontal_axis	0,...,39
    vertical_axis	0,...,39
    """  # noqa: E501

    resources = 'https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
    split_ratio = 0.7
    num_tasks = 7
    num_classes = [6,6,2,3,3,40,40]
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
        num_data = len(data['images'])
        label = self.create_label(num_data)
        data_index = np.arange(num_data)
        np.random.seed(42)
        np.random.shuffle(data_index)
        train_index = data_index[:int(num_data*self.split_ratio)]
        test_index = data_index[int(num_data*self.split_ratio):]
        train_set = (data['images'][train_index],
                     label[train_index])
        test_set = (data['images'][test_index],
                    label[test_index])

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

    def create_label(self, num_data):
        label = np.zeros((num_data, self.num_tasks))
        for i in range(self.num_tasks):
            num_per_cls = np.prod([num_factor for f, num_factor in enumerate(self.num_classes) if f != i])
            for j in range(self.num_classes[i]):
                label[:, j*num_per_cls:(j+1)*num_per_cls] = j
        return label
        
    @master_only
    def download(self):
        os.makedirs(self.data_prefix, exist_ok=True)

        # download files
        url = self.resources
        filename = url.rpartition('/')[2]
        download_url(url,
            root=self.data_prefix,
            filename=filename)


