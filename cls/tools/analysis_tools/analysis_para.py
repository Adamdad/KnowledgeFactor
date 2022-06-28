import argparse
import torch
from mmcv import Config
from prettytable import PrettyTable

from mmcls.models.builder import build_classifier


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def parse_args():
    parser = argparse.ArgumentParser(description='Explain a model')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    print(cfg)
    model = build_classifier(cfg.model)

    count_parameters(model)


if __name__ == '__main__':
    main()
