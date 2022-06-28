# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path
import pickle

import numpy as np
import torch.distributed as dist
from mmcv.runner import get_dist_info

from mmcls.datasets.disentangle_data.multi_task import MultiTask
from mmcls.datasets.pipelines.compose import Compose

from .base_dataset import BaseDataset
from .builder import DATASETS
from .utils import check_integrity, download_and_extract_archive


@DATASETS.register_module()
class CIFAR10(BaseDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This implementation is modified from
    https://github.com/pytorch/vision/blob/master/torchvision/datasets/cifar.py
    """  # noqa: E501

    base_folder = 'cifar-10-batches-py'
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    filename = 'cifar-10-python.tar.gz'
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def load_annotations(self):

        rank, world_size = get_dist_info()

        if rank == 0 and not self._check_integrity():
            download_and_extract_archive(
                self.url,
                self.data_prefix,
                filename=self.filename,
                md5=self.tgz_md5)

        if world_size > 1:
            dist.barrier()
            assert self._check_integrity(), \
                'Shared storage seems unavailable. ' \
                f'Please download the dataset manually through {self.url}.'

        if not self.test_mode:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.imgs = []
        self.gt_labels = []

        # load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.data_prefix, self.base_folder,
                                     file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.imgs.append(entry['data'])
                if 'labels' in entry:
                    self.gt_labels.extend(entry['labels'])
                else:
                    self.gt_labels.extend(entry['fine_labels'])

        self.imgs = np.vstack(self.imgs).reshape(-1, 3, 32, 32)
        self.imgs = self.imgs.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

        data_infos = []
        for img, gt_label in zip(self.imgs, self.gt_labels):
            gt_label = np.array(gt_label, dtype=np.int64)
            info = {'img': img, 'gt_label': gt_label}
            data_infos.append(info)
        return data_infos

    def _load_meta(self):
        path = os.path.join(self.data_prefix, self.base_folder,
                            self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError(
                'Dataset metadata file not found or corrupted.' +
                ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.CLASSES = data[self.meta['key']]

    def _check_integrity(self):
        root = self.data_prefix
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True


@DATASETS.register_module()
class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset."""

    base_folder = 'cifar-100-python'
    url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    filename = 'cifar-100-python.tar.gz'
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


@DATASETS.register_module()
class CIFAR10_2Task(CIFAR10):
    gt2newgt = {0: 0, 1: 1, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: 9, 8: 2, 9: 3}

    def load_annotations(self):

        rank, world_size = get_dist_info()

        if rank == 0 and not self._check_integrity():
            download_and_extract_archive(
                self.url,
                self.data_prefix,
                filename=self.filename,
                md5=self.tgz_md5)

        if world_size > 1:
            dist.barrier()
            assert self._check_integrity(), \
                'Shared storage seems unavailable. ' \
                f'Please download the dataset manually through {self.url}.'

        if not self.test_mode:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.imgs = []
        self.gt_labels = []

        # load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.data_prefix, self.base_folder,
                                     file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.imgs.append(entry['data'])
                if 'labels' in entry:
                    self.gt_labels.extend(entry['labels'])
                else:
                    self.gt_labels.extend(entry['fine_labels'])

        self.imgs = np.vstack(self.imgs).reshape(-1, 3, 32, 32)
        self.imgs = self.imgs.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

        data_infos = []
        for img, gt_label in zip(self.imgs, self.gt_labels):
            gt_label = np.array(self.gt2newgt[gt_label], dtype=np.int64)
            info = {'img': img, 'gt_label': gt_label}
            data_infos.append(info)
        return data_infos


@DATASETS.register_module()
class CIFAR10_Select(CIFAR10):
    def __init__(self,
                 data_prefix,
                 pipeline,
                 select_class=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                 classes=None,
                 ann_file=None,
                 test_mode=False):
        assert isinstance(select_class, list) or isinstance(
            select_class, tuple)
        self.select_class = select_class
        self.select2gt = {
            self.select_class[i]: i for i in range(len(self.select_class))}
        super(CIFAR10_Select, self).__init__(data_prefix,
                                             pipeline,
                                             classes=classes,
                                             ann_file=ann_file,
                                             test_mode=test_mode)

    def load_annotations(self):

        rank, world_size = get_dist_info()

        if rank == 0 and not self._check_integrity():
            download_and_extract_archive(
                self.url,
                self.data_prefix,
                filename=self.filename,
                md5=self.tgz_md5)

        if world_size > 1:
            dist.barrier()
            assert self._check_integrity(), \
                'Shared storage seems unavailable. ' \
                f'Please download the dataset manually through {self.url}.'

        if not self.test_mode:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.imgs = []
        self.gt_labels = []

        # load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.data_prefix, self.base_folder,
                                     file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.imgs.append(entry['data'])
                if 'labels' in entry:
                    self.gt_labels.extend(entry['labels'])
                else:
                    self.gt_labels.extend(entry['fine_labels'])

        self.imgs = np.vstack(self.imgs).reshape(-1, 3, 32, 32)
        self.imgs = self.imgs.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

        data_infos = []
        for img, gt_label in zip(self.imgs, self.gt_labels):
            if gt_label in self.select_class:
                gt_label = np.array(self.select2gt[gt_label], dtype=np.int64)
                info = {'img': img, 'gt_label': gt_label}
                data_infos.append(info)
        return data_infos


@DATASETS.register_module()
class CIFAR10_MultiTask(CIFAR10, MultiTask):
    num_tasks = 10

    def load_annotations(self):

        rank, world_size = get_dist_info()

        if rank == 0 and not self._check_integrity():
            download_and_extract_archive(
                self.url,
                self.data_prefix,
                filename=self.filename,
                md5=self.tgz_md5)

        if world_size > 1:
            dist.barrier()
            assert self._check_integrity(), \
                'Shared storage seems unavailable. ' \
                f'Please download the dataset manually through {self.url}.'

        if not self.test_mode:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.imgs = []
        self.gt_labels = []

        # load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.data_prefix, self.base_folder,
                                     file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.imgs.append(entry['data'])
                if 'labels' in entry:
                    self.gt_labels.extend(entry['labels'])
                else:
                    self.gt_labels.extend(entry['fine_labels'])

        self.gt_labels = np.eye(10)[self.gt_labels]
        self.imgs = np.vstack(self.imgs).reshape(-1, 3, 32, 32)
        self.imgs = self.imgs.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

        data_infos = []
        for img, gt_label in zip(self.imgs, self.gt_labels):
            gt_label = np.array(gt_label, dtype=np.int64)
            info = {'img': img, 'gt_label': gt_label}
            data_infos.append(info)
        return data_infos
