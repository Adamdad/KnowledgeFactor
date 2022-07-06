import copy
import numpy as np
import torch
import torch.nn.functional as F
import warnings
from shutil import ExecError
from torch import nn

from mmcls.models.losses.kd_loss import (InfoMax_loss, InfoMin_loss)
from ..builder import (CLASSIFIERS, build_backbone, build_head, build_loss,
                       build_neck)
from ..utils.augment import Augments
from .base import BaseClassifier


@CLASSIFIERS.register_module()
class KFImageClassifier(BaseClassifier):

    def __init__(self,
                 backbone,
                 kd_loss,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super(KFImageClassifier, self).__init__(init_cfg)

        if pretrained is not None:
            warnings.warn('DeprecationWarning: pretrained is a deprecated \
                key, please consider using init_cfg')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        assert 'student' in backbone.keys(), 'student network should be specified'
        assert 'teacher' in backbone.keys(), 'teacher network should be specified'
        return_tuple = backbone.pop('return_tuple', True)
        self.num_task = backbone['num_task']
        self.student = nn.ModuleDict(
            {
                'CKN': build_backbone(backbone['student']['CKN']),
                'TSN': nn.ModuleList([build_backbone(backbone['student']['TSN']) for i in range(self.num_task)]),
                'neck': build_neck(neck['student']),
                'head_task': build_head(head['task']),
                'head': build_head(head['student'])
            }
        )

        self.teacher = nn.ModuleDict(
            {
                'backbone': build_backbone(backbone['teacher']),
                'neck': build_neck(neck['teacher']),
                'head': build_head(head['teacher'])
            }
        )

        self.feat_channels_student = train_cfg['feat_channels']['student']
        self.feat_channels_teacher = train_cfg['feat_channels']['teacher']
        feat_fcs = []
        for i in range(len(self.feat_channels_student)):
            feat_fcs.append(nn.Sequential(
                nn.Linear(
                    self.feat_channels_teacher[i], self.feat_channels_student[i]),
                nn.BatchNorm1d(self.feat_channels_student[i]),
                nn.ReLU(True),
                nn.Linear(
                    self.feat_channels_student[i], self.feat_channels_student[i])
            )
            )
        self.feat_fcs = nn.ModuleList(feat_fcs)

        self.criterionCls = F.cross_entropy
        self.criterionTask = F.binary_cross_entropy_with_logits
        self.criterionKD = build_loss(kd_loss)

        self.lambda_kd = train_cfg['lambda_kd']
        self.alpha = train_cfg['alpha']
        self.beta = train_cfg['beta']
        self.lambda_feat = train_cfg['lambda_feat']
        self.teacher_ckpt = train_cfg['teacher_checkpoint']
        self.task_weight = train_cfg['task_weight']

        if return_tuple is False:
            warnings.warn(
                'The `return_tuple` is a temporary arg, we will force to '
                'return tuple in the future. Please handle tuple in your '
                'custom neck or head.', DeprecationWarning)
        self.return_tuple = return_tuple
        self.load_teacher()
        self.augments = None
        if train_cfg is not None:
            augments_cfg = train_cfg.get('augments', None)
            if augments_cfg is not None:
                self.augments = Augments(augments_cfg)
            else:
                # Considering BC-breaking
                mixup_cfg = train_cfg.get('mixup', None)
                cutmix_cfg = train_cfg.get('cutmix', None)
                assert mixup_cfg is None or cutmix_cfg is None, \
                    'If mixup and cutmix are set simultaneously,' \
                    'use augments instead.'
                if mixup_cfg is not None:
                    warnings.warn('The mixup attribute will be deprecated. '
                                  'Please use augments instead.')
                    cfg = copy.deepcopy(mixup_cfg)
                    cfg['type'] = 'BatchMixup'
                    # In the previous version, mixup_prob is always 1.0.
                    cfg['prob'] = 1.0
                    self.augments = Augments(cfg)
                if cutmix_cfg is not None:
                    warnings.warn('The cutmix attribute will be deprecated. '
                                  'Please use augments instead.')
                    cfg = copy.deepcopy(cutmix_cfg)
                    cutmix_prob = cfg.pop('cutmix_prob')
                    cfg['type'] = 'BatchCutMix'
                    cfg['prob'] = cutmix_prob
                    self.augments = Augments(cfg)

    def extract_feat(self, imgs):
        pass

    def load_teacher(self):
        split_lins = '*' * 20
        state_dict = torch.load(self.teacher_ckpt)
        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
        try:
            self.teacher.load_state_dict(state_dict)
            print(split_lins)
            print(
                f'Teacher pretrained model has been loaded {self.teacher_ckpt}')
            print(split_lins)
        except:
            print('Teacher model not loaded')
            print(state_dict.keys())
            print(self.teacher.state_dict().keys())
            AssertionError('Teacher model not loaded')
            exit()

        for param in self.teacher.parameters():
            param.requires_grad = False

    #####################################################
    # Functions for teacher network
    def extract_teacher_feat(self, img):
        """Directly extract features from the backbone + neck."""
        x = self.teacher['backbone'](img)
        if self.return_tuple:
            if not isinstance(x, tuple):
                x = (x, )
                warnings.simplefilter('once')
                warnings.warn(
                    'We will force all backbones to return a tuple in the '
                    'future. Please check your backbone and wrap the output '
                    'as a tuple.', DeprecationWarning)
        else:
            if isinstance(x, tuple):
                x = x[-1]
        # if self.with_neck:
        x = self.teacher['neck'](x)
        return x

    def get_teacher_logit(self, img):
        """Directly extract features from the backbone + neck."""
        x = self.extract_teacher_feat(img)
        if isinstance(x, tuple):
            last_x = x[-1]
        logit = self.teacher['head'].fc(last_x)  # head
        return logit, x
    #####################################################
    # Functions for student network

    def extract_common_feat(self, img):
        """Directly extract features from the backbone + neck."""
        x = self.student['CKN'](img)
        if self.return_tuple:
            if not isinstance(x, tuple):
                x = (x, )
                warnings.simplefilter('once')
                warnings.warn(
                    'We will force all backbones to return a tuple in the '
                    'future. Please check your backbone and wrap the output '
                    'as a tuple.', DeprecationWarning)
        else:
            if isinstance(x, tuple):
                x = x[-1]

        x = self.student['neck'](x)
        return x

    def extract_task_feat(self, img):
        """Directly extract features from the backbone + neck."""
        result = dict(feats=[],
                      mu_vars=[])

        for i in range(self.num_task):
            (mu, var), x = self.student['TSN'][i](img)
            if self.return_tuple:
                if not isinstance(x, tuple):
                    x = (x, )
            else:
                if isinstance(x, tuple):
                    x = x[-1]

            result['feats'].append(x)
            result['mu_vars'].append((mu, var))
        return result

    def extract_student_feat(self, img):
        common_xs = self.extract_common_feat(img)

        task_result = self.extract_task_feat(img)

        if self.num_task == 1:
            return common_xs, task_result['feats'][0], task_result
        else:
            return common_xs, task_result['feats'], task_result

    def get_student_logit(self, img):
        """Directly extract features from the backbone + neck."""
        common_xs, task_feat, task_result = self.extract_student_feat(
            img)
        if isinstance(common_xs, tuple):
            common_x = common_xs[-1]
        if isinstance(task_feat, tuple):
            task_feat = task_feat[-1]

        if isinstance(task_feat, list):
            feat = [common_x + task_f[-1] for task_f in task_feat]
        else:
            feat = common_x + task_feat
        logit = self.student['head'].get_logits(feat)  # head
        task_logit = self.student['head_task'].get_logits(task_feat)
        return logit, task_logit, common_xs, task_result
    
    def get_logit(self, img):
        logit, _, _, _ = self.get_student_logit(img)
        return logit

    def get_adv_logit(self, img):
        _, task_logit, _, _ = self.get_student_logit(
            img)
        return task_logit

    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            gt_label (Tensor): It should be of shape (N, 1) encoding the
                ground-truth label of input images for single label task. It
                shoulf be of shape (N, C) encoding the ground-truth label
                of input images for multi-labels task.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.augments is not None:
            img, gt_label = self.augments(img, gt_label)

        with torch.no_grad():
            teacher_logit, teacher_x = self.get_teacher_logit(img)

        student_logit, task_logit, student_common_x, task_result = self.get_student_logit(
            img)

        loss_infomax = 0.0
        # Deep feature simulation for KD
        assert len(teacher_x) == len(student_common_x)
        for layer_id, (teacher_x_layer, student_x_layer) in enumerate(zip(teacher_x, student_common_x)):
            loss_infomax += InfoMax_loss(self.feat_fcs[layer_id](teacher_x_layer),
                                         student_x_layer) * self.lambda_feat
        loss_infomax = loss_infomax/len(student_common_x)

        # Output simulation for KD
        loss_kd = self.criterionKD(
            student_logit, teacher_logit.detach()) * self.lambda_kd

        # Cls loss and infor loss
        loss_cls = self.student['head'].loss(student_logit, gt_label)['loss']
        # onehot_gt_label = F.one_hot(gt_label,
        #                             num_classes=student_logit.shape[1]).float()
        loss_task = self.student['head_task'].loss(task_logit, gt_label)['loss'] * self.task_weight
        

        # InfoMin Loss for task feature
        loss_infomin = 0.0
        for mu, log_var in task_result['mu_vars']:
            loss_infomin += InfoMin_loss(mu, log_var) * self.beta

        losses = dict(loss_infomax=loss_infomax,
                      loss_kd=loss_kd,
                      loss_cls=loss_cls,
                      loss_task=loss_task,
                      loss_infomin=loss_infomin)
        # print(losses)
        return losses

    def simple_test(self, img, img_metas):
        """Test without augmentation."""
        cls_score = self.get_logit(img)

        try:
            if isinstance(cls_score, list):
                cls_score = sum(cls_score) / float(len(cls_score))
            pred = F.softmax(
                cls_score, dim=1) if cls_score is not None else None
            res = self.student['head'].post_process(pred)
        except TypeError as e:
            if 'not tuple' in str(e) and self.return_tuple:
                return TypeError(
                    'Seems the head cannot handle tuple input. We have '
                    'changed all backbones\' output to a tuple. Please '
                    'update your custom head\'s forward function. '
                    'Temporarily, you can set "return_tuple=False" in '
                    'your backbone config to disable this feature.')
            raise e

        return res
