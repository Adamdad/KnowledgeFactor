import torch

def main():
    PATH = 'resnet18_R182R18_common_network_20211112v2.pth'
    ckp_path = '/Users/xingyiyang/Downloads/resnet18_a1_0-d63eafa0.pth'
    model_dict = torch.load(ckp_path)
    save_dict= dict(common_network=model_dict)
    torch.save(save_dict, PATH, _use_new_zipfile_serialization=False)
    print(model_dict.keys())
if __name__ == '__main__':
    main()
    