# KnowledgeFactor
Factorize the knowledge from a multi-talented teacher to students


## Installation
1. Create a conda environment and activate it. Install PyTorch following official instructions,

        conda create --name kf python=3.8 -y
        conda activate kf
        conda install pytorch torchvision -c pytorch

2. Install mmcv and mmcls via mim

        pip install -U openmim
        mim install mmcv-full
        mim install mmcls

3. Download the teacher checkpoint

| Name | Link|
|--- | --- |
| resnet-18-b128x1_cifar10 | [checkpoint(from mmcls)](https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_b16x8_cifar10_20210528-bd6371c8.pth) [log(from mmcls)](https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_b16x8_cifar10_20210528-bd6371c8.log.json)|
| wideresnet28-2-b128x1_cifar10 | [checkpoint](https://drive.google.com/file/d/1_MgpoL8F_2wwgC6UD_tdDcX_teGgg3Mh/view?usp=sharing)  [log](https://drive.google.com/file/d/179_3yTHX8xmxYMQWMdwiLSNeSPEUn1mY/view?usp=sharing)|
| wideresnet28-10-b128x1_cifar10 | [checkpoint](https://drive.google.com/file/d/1Gv4VRki5gToF5TNm84cdQJqU-HhU0Jy1/view?usp=sharing) [log](https://drive.google.com/file/d/1wQsAJ9gAKBJboghIO1K0eakRAsk5TwmP/view?usp=sharing)|