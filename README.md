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

| Model | Dataset | Link|
|--- |  --- | --- |
| WideResNet28-2 | CIFAR10 | [checkpoint](https://drive.google.com/file/d/1_MgpoL8F_2wwgC6UD_tdDcX_teGgg3Mh/view?usp=sharing)  [log](https://drive.google.com/file/d/179_3yTHX8xmxYMQWMdwiLSNeSPEUn1mY/view?usp=sharing)|
| WideResNet28-10 | CIFAR10 | [checkpoint](https://drive.google.com/file/d/1Gv4VRki5gToF5TNm84cdQJqU-HhU0Jy1/view?usp=sharing) [log](https://drive.google.com/file/d/1wQsAJ9gAKBJboghIO1K0eakRAsk5TwmP/view?usp=sharing)|
| ResNet-18 | CIFAR10 | [checkpoint(from mmcls)](https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_b16x8_cifar10_20210528-bd6371c8.pth) [log(from mmcls)](https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_b16x8_cifar10_20210528-bd6371c8.log.json)|

4. Prepare the data. 
- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) (Automatic downloaded)
- [ImageNet](https://www.image-net.org)
- [Shape3D](https://github.com/deepmind/3d-shapes) and [dSprites](https://github.com/deepmind/dsprites-dataset)

## Get Started
1. Given a $CONGIG file, we can simply run
        
        python tools/dist_train.py $CONGIG $NUM_GPUS