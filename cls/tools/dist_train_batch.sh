#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2 bash tools/dist_train.sh configs/IFC/wideresnet28-2_mobilenet_b128x1_cifar10_logit_deep_l2_paralell_10task_1e-3_taskloss.py 1
CUDA_VISIBLE_DEVICES=2 bash tools/dist_train.sh configs/IFC/resnet18_mobilenet_b128x1_cifar10_logit_deep_l2_paralell_singletask_1e-3_taskloss.py 1
CUDA_VISIBLE_DEVICES=2 bash tools/dist_train.sh configs/IFC/resnet18_mobilenet_b128x1_cifar10_logit_deep_l2_paralell_10task_1e-3_taskloss.py 1
