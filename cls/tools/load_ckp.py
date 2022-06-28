import torch

def main():
    # PATH = 'resnet18_R182R18_common_network_20211112v2.pth'
    # ckp_path = '/home/yangxingyi/NeuralFactor/Multi-task-Depth-Seg/result/NYUD/kd_resnet_50_to_resnet_18/multi_task_baseline/best_model.pth.tar'
    ckp_path = ''
    model_dict = torch.load(ckp_path)
    # save_dict= dict(common_network=model_dict)
    # torch.save(save_dict, PATH, _use_new_zipfile_serialization=False)
    print(model_dict.keys())

def ckp_to_load():
    ckp_path = '/home/yangxingyi/.cache/torch/checkpoints/resnet18-5c106cde.pth'
    model_dict = torch.load(ckp_path)
    new_dict = dict()
    for k in model_dict.keys():
        if k.startswith('fc'):
            k.replace('fc',)


if __name__ == '__main__':
    main()
    