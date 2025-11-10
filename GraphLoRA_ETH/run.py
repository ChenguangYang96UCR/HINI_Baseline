import yaml
from yaml import SafeLoader
import argparse
import torch
from pre_train import pretrain2
from model.GraphLoRA import transfer2
from util import get_parameter
import random
import numpy as np

from network_month import (
    split_pretrain_downstream, 
    create_elementwise_similarity_matrix, 
    build_graph_from_data,
    SequenceData 
)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")

def main():

    set_seed(42)

    data = SequenceData('H1N1_HA_aligned.fasta')
    pretrain_year = 2013
    m = 1
    split_data = split_pretrain_downstream(data, pretrain_year, m)
    pretrain_set = split_data['pretrain_set']
    downstream_set = split_data['downstream_set']
    threshold=0.998
    _, pretrain_edges = create_elementwise_similarity_matrix(pretrain_set['sequences'], threshold)
    pretrain_graph = build_graph_from_data(pretrain_set['sequences'], pretrain_edges, pretrain_set['monthlabels'])
    _, downstream_edges = create_elementwise_similarity_matrix(downstream_set['sequences'], threshold)
    downstream_graph = build_graph_from_data(downstream_set['sequences'], downstream_edges, downstream_set['monthlabels'])
    
    # --- 预训练 ---
    print("\n" + "="*30 + "\nPRE-TRAINING\n" + "="*30)
    config_pretrain = yaml.load(open('config.yaml'), Loader=SafeLoader)['Cora']
    
    gpu_id = 0

    pretrained_gnn_state = pretrain2(pretrain_graph, "GRACE", config_pretrain, gpu_id, is_reduction=False)

    # --- 微调 ---
    if pretrained_gnn_state:
        print("\n" + "="*30 + "\nLoRA FINE-TUNING\n" + "="*30)
        args_config = yaml.load(open('config2.yaml'), Loader=SafeLoader)['public']['Cora']
        
        class Args:
            r = 32; tau = 0.5; sup_weight = 0.2
            lr1, lr2, lr3 = float(1e-3), float(1e-1), float(1e-4)
            wd1, wd2, wd3 = float(args_config['wd1']), float(args_config['wd2']), float(args_config['wd3'])
            l1, l2, l3, l4 = float(args_config['l1']), float(args_config['l2']), float(args_config['l3']), float(args_config['l4'])
            num_epochs = 1000
        args = Args()
        
        config_finetune = config_pretrain
        
        transfer2(pretrain_graph, downstream_graph, pretrained_gnn_state, args, config_finetune, gpu_id)
if __name__ == '__main__':
    # 确保 H1N1_HA_aligned.fasta, config.yaml, config2.yaml 文件在目录中
    main()
