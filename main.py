# -*- coding: utf-8 -*-


import argparse
import os.path

from utils.datasets import get_dataset
from train import train_eval
from utils import tools
import time
from pprint import pprint
import random
from model.EnsNet import EnsNet


def get_parser():
    parser = argparse.ArgumentParser(description="Demo of argparse")
    parser.add_argument('--ds_name', default='NCI1',
                        choices=["NCI1", "NCI109", "Mutagenicity", "PROTEINS", "REDDIT-MULTI-12K", "IMDB-MULTI"])
    parser.add_argument('--gpu_id', default='1')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    ds_name = args.ds_name
    gpu_id = args.gpu_id
    if not os.path.exists(f"config/{ds_name}.json"):
        tools.save_data_to_json(f"config/{ds_name}.json", {})
    ds_config = tools.load_json(f"config/{ds_name}.json")
    config = tools.load_json("config/EnsNet.json")
    train_config = config["train_config"]
    net_config = config["net_params"]
    # net_config = config["net_params"]
    # train_config = config["train_config"]
    for key, value in ds_config.items():
        if key in net_config:
            net_config[key] = value
        else:
            train_config[key] = value
    device = "cuda:" + str(gpu_id)
    seed = 8971
    tools.set_seed(seed)
    dataset = get_dataset(ds_name, config["data_dir"])
    num_feature, num_classes = dataset.num_features, dataset.num_classes
    net_config["device"] = device
    net_config["in_channels"] = num_feature
    net_config["out_channels"] = num_classes
    config["dataset"] = ds_name
    net_config["ds_name"] = ds_name
    pprint(config)
    model = EnsNet
    time_str = time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y_') + str(random.randint(0, 100))
    acc, std, duration_mean = train_eval.cross_validation_with_acc_val_set(ds_name, "MCCD", dataset, seed, model,
                                                                           device, config, time_str)
    print(f"test acc is {acc}, test std is {std}, duration mean is {duration_mean}")
