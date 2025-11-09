import yaml
from yaml import SafeLoader
import argparse
import torch
from pre_train import pretrain2
from model.GraphLoRA import transfer2
from util import get_parameter
import numpy as np
import random
import os


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)
    print(f"Random seed set to {seed}")


def main():

    set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_dataset', type=str, default='2009')
    parser.add_argument('--test_dataset', type=str, default='2010')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='./config.yaml')
    parser.add_argument('--para_config', type=str, default='./config2.yaml')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    args = get_parameter(args)

    pretrain_graph_file_path = f'H1N1_graph_{args.pretrain_dataset}.pt'
    pretrain_graph = torch.load(pretrain_graph_file_path, weights_only=False)
    downstream_graph_file_path = f'H1N1_graph_{args.test_dataset}.pt'
    downstream_graph = torch.load(downstream_graph_file_path, weights_only=False)

    # --- 预训练 ---
    print("\n" + "="*30 + "\nPRE-TRAINING\n" + "="*30)
    config_pretrain = yaml.load(open(args.config), Loader=SafeLoader)['Cora']   
    pretrained_gnn_state = pretrain2(pretrain_graph, "GRACE", config_pretrain, args.gpu_id, args.pretrain_dataset)

    # --- 微调 ---
    print("\n" + "="*30 + "\nFINE-TUNING\n" + "="*30)
    config_transfer = yaml.load(open(args.config), Loader=SafeLoader)['transfer']
    transfer2(pretrain_graph, downstream_graph, pretrained_gnn_state, args, config_transfer, args.gpu_id, args.seed, args.pretrain_dataset, args.test_dataset)


if __name__ == '__main__':
    main()
