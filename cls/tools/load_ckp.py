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
    ckp_path = '/home/yangxingyi/.cache/torch/checkpoints/resnet50_8xb32_in1k_20210831-ea4938fc.pth'
    save_path = '/home/yangxingyi/.cache/torch/checkpoints/resnet50_8xb32_in1k_20210831-ea4938fc_converted.pth'
    model_dict = torch.load(ckp_path)
    if 'state_dict' in model_dict.keys():
        model_dict = model_dict['state_dict']
    new_dict = dict()
    for k, v in model_dict.items():
        if k.startswith('fc'):
            new_k = 'head.{}'.format(k)
        else:
            new_k = 'backbone.{}'.format(k)
        print('Old Key:', k, '-> New Key:', new_k)
        new_dict[new_k] = v
    save_dict= dict(state_dict=new_dict)
    torch.save(save_dict, save_path, _use_new_zipfile_serialization=False)

if __name__ == '__main__':
    # main()
    ckp_to_load()
    