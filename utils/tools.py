

import json
import os
import torch



def load_json(json_loc: str) -> dict:
    with open(json_loc) as f:
        config = json.load(f)
    return config


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("dir is created successful")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # torch.cuda.set_device(1)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:', torch.cuda.get_device_name(0))
        device = torch.device("cuda:" + str(gpu_id))
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


def save_data_to_json(outfile, data):
    with open(outfile, 'w') as f:
        json.dump(data, f)


def write_dirs(model_name, ex_out_dir, ds_name, out_dir, time_str):
    com_loc = os.path.join(model_name, ds_name, ex_out_dir)
    # com_loc =  model_name  + "/ex_out_dir/" + ds_name + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_log_dir = out_dir + 'logs/' + com_loc
    root_ckpt_dir = out_dir + 'checkpoints/' + com_loc
    write_file_name = out_dir + 'results/' + com_loc
    write_config_file = out_dir + 'configs/' + com_loc

    if not os.path.exists(write_file_name):
        os.makedirs(write_file_name)
    # 'out/graph_classfication/results/GCN/DD/first/21h42m50s_on_Mar_17_2021.txt'
    if not os.path.exists(write_config_file):
        os.makedirs(write_config_file)
    dirs = root_log_dir + time_str, root_ckpt_dir + time_str, write_file_name + time_str, write_config_file + time_str
    return dirs
