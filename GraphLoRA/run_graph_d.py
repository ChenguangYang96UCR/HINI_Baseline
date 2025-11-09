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

    set_seed(0)

    pretrain_graph_file_path = 'H1N1_graph_2010.pt'
    pretrain_graph = torch.load(pretrain_graph_file_path)
    downstream_graph_file_path = 'H1N1_graph_2011.pt'
    downstream_graph = torch.load(downstream_graph_file_path)

    # --- 预训练 ---
    print("\n" + "="*30 + "\nPRE-TRAINING\n" + "="*30)
    config_pretrain = yaml.load(open('config.yaml'), Loader=SafeLoader)['Cora']
    
    gpu_id = 0
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model_path = 'pre_trained_gnn/2010.GRACE.GAT.False.pth'
    print(f"Loading pre-trained GNN state from: {model_path}")
    pretrained_gnn_state = torch.load(model_path, map_location=device)
    print("✓ Pre-trained GNN state loaded successfully!")
    
    # --- 微调 ---
    if pretrained_gnn_state:
        print("\n" + "="*30 + "\nLoRA FINE-TUNING\n" + "="*30)
        args_config = yaml.load(open('config2.yaml'), Loader=SafeLoader)['public']['Cora']
        
        class Args:
            r = 32; tau = 0.5; sup_weight = 0.2
            lr1, lr2, lr3, lr4= float(1e-3), float(1e-1), float(1e-4), float(1e-3)
            wd1, wd2, wd3, wd4 = float(args_config['wd1']), float(args_config['wd2']), float(args_config['wd3']), float(args_config['wd3'])
            l1, l2, l3, l4 = float(args_config['l1']), float(args_config['l2']), float(args_config['l3']), float(args_config['l4'])
            num_epochs = 200
        args = Args()
        
        config_finetune = config_pretrain
        
        transfer2(pretrain_graph, downstream_graph, pretrained_gnn_state, args, config_finetune, gpu_id)
if __name__ == '__main__':
    # 确保 H1N1_HA_aligned.fasta, config.yaml, config2.yaml 文件在目录中
    main()
