import os

import torch
import argparse
from core.model import *
import torchvision.transforms.functional as f
from tools.train import train
from tools.test import test
from core.dataset import Fusion_Datasets
from core.dataset.Fusion_datasets import Datasets
import torchvision.transforms as transforms
from core.util import load_config, count_parameters

def get_args():
    parser = argparse.ArgumentParser(description='run')

    parser.add_argument('--config', type=str, default='./config/config.yaml')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)

    args = parser.parse_args()
    return args


def runner(args):
    configs = load_config(args.config)
    project_configs = configs['PROJECT']
    model_configs = configs['MODEL']
    train_configs = configs['TRAIN']
    test_configs = configs['TEST']
    train_dataset_configs = configs['TRAIN_DATASET']
    test_dataset_configs = configs['TEST_DATASET']
    input_size = train_dataset_configs['input_size']

    if train_dataset_configs['channels'] == 3:
        base_transforms = transforms.Compose(
            [transforms.ToTensor(),
             # transforms.RandomHorizontalFlip(),
             # transforms.RandomVerticalFlip(),
             ])
    elif train_dataset_configs['channels'] == 1:
        base_transforms = transforms.Compose(
            [transforms.Resize((input_size,input_size)),
             transforms.ToTensor(),
             ])

    train_datasets = MFAHIQ_Datasets(train_dataset_configs, base_transforms)
    test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_datasets = MFAHIQ_Datasets(test_dataset_configs, test_transforms)
    model = eval(model_configs['model_name'])(configs)

    print('Model Para:', count_parameters(model))

    if train_configs['resume'] != 'None':
        checkpoint = torch.load(train_configs['resume'])
        model.load_state_dict(checkpoint['model'].state_dict())

    if args.train:
        train(model, train_datasets, test_datasets, configs)
    if args.test:
        test1(model, test_datasets, configs, load_weight_path=True)


if __name__ == '__main__':
    args = get_args()
    runner(args)
