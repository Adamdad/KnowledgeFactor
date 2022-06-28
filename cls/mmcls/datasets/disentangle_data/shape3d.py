# Copyright (c) OpenMMLab. All rights reserved.
import codecs
import os
import os.path as osp

import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info, master_only

from .multi_task import MultiTask
from mmcls.datasets.builder import DATASETS
from mmcls.datasets.utils import download_and_extract_archive, download_url, rm_suffix
import h5py


@DATASETS.register_module()
class Shape3D(MultiTask):
    """Latent factor values
        Color: white
        Shape: square, ellipse, heart
        Scale: 6 values linearly spaced in [0.5, 1]
        Orientation: 40 values in [0, 2 pi]
        Position X: 32 values in [0, 1]
        Position Y: 32 values in [0, 1]
    """  # noqa: E501

    resources = '3dshapes.h5'
    split_ratio = 0.7
    num_tasks = 6
    CLASSES = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
               'orientation']

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
        data = h5py.File(data_file, 'r')
        num_data = len(data['images'])
        labels = self.convert_value_to_label(data)
        train_set = (data['images'][:int(num_data*self.split_ratio)],
                     labels[:int(num_data*self.split_ratio)])
        test_set = (data['images'][int(num_data*self.split_ratio):],
                    labels[int(num_data*self.split_ratio):])

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

    def convert_value_to_label(self, data):
        labels = []
        for i in range(self.num_tasks):
            values = np.unique(data['labels'][:, i])
            values = np.sort(values)
            num_class = len(values)
            print(f'Task {i}, with {num_class} classes')
            
            value2cls = {values[c]:c for c in range(num_class)}

            label_converted = np.vectorize(value2cls.get)(data['labels'][:, i])
            labels.append(label_converted)
        labels = np.stack(labels, axis=-1)
        return labels
    @master_only
    def download(self):
        assert f'Shape3D dataset can only be downloaded manually'
