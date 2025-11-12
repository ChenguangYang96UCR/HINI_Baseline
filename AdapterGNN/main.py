import argparse
import statistics
from torch_geometric.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd
import logging
import warnings
from tqdm import tqdm
import yaml
from yaml import SafeLoader
import os
from util import get_dataset, act
from GNN import GNN
from torch_geometric.transforms import SVDFeatureReduction
from adapterGNN import AdapterGNN
import time
from torch_geometric.utils import add_remaining_self_loops, to_undirected
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import recall_score, f1_score
import random

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


class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret


def train(model, logreg, train_dataloader, loss_fn, optimizer):
    model.train()
    all_train_preds = []
    all_train_labels = []
    for batch in train_dataloader:
        batch.edge_index = add_remaining_self_loops(batch.edge_index)[0]
        batch.edge_index = to_undirected(batch.edge_index)

        emb = model(batch)

        train_embs = emb[:batch.batch_size]
        train_labels = batch.y[:batch.batch_size]
        logits = logreg(train_embs)
        loss = loss_fn(logits, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_preds = torch.argmax(logits, dim=1)
        all_train_preds.append(train_preds.cpu())
        all_train_labels.append(train_labels.cpu())
    all_train_preds = torch.cat(all_train_preds, dim=0).numpy()
    all_train_labels = torch.cat(all_train_labels, dim=0).numpy()
    train_acc = (all_train_preds == all_train_labels).sum() / len(all_train_labels)
    train_recall = recall_score(all_train_labels, all_train_preds, average='macro', zero_division=0)
    train_f1 = f1_score(all_train_labels, all_train_preds, average='macro', zero_division=0)
    return train_acc, train_recall, train_f1


def eval(model, logreg, val_dataloader):
    model.eval()
    logreg.eval()
    all_val_preds = []
    all_val_labels = []
    with torch.no_grad():
        for batch in val_dataloader:
            batch.edge_index = add_remaining_self_loops(batch.edge_index)[0]
            batch.edge_index = to_undirected(batch.edge_index)
            emb = model(batch)[:batch.batch_size]
            val_labels = batch.y[:batch.batch_size]
            val_logits = logreg(emb)
            val_preds = torch.argmax(val_logits, dim=1)
            all_val_preds.append(val_preds.cpu())
            all_val_labels.append(val_labels.cpu())
    all_val_preds = torch.cat(all_val_preds, dim=0).numpy()
    all_val_labels = torch.cat(all_val_labels, dim=0).numpy()
    val_acc = (all_val_preds == all_val_labels).sum() / len(all_val_labels)
    val_recall = recall_score(all_val_labels, all_val_preds, average='macro', zero_division=0)
    val_f1 = f1_score(all_val_labels, all_val_preds, average='macro', zero_division=0)
    return val_acc, val_recall, val_f1


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0.0001,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default='H1N1')
    parser.add_argument('--input_model_file', type=str, default='')
    parser.add_argument('--filename', type=str, default='', help='output filename')
    parser.add_argument('--seed', type=int, default=0, help="Seed for splitting the dataset.")
    parser.add_argument('--split', type=str, default="scaffold", help="random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default=0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')
    parser.add_argument('--log', type=str)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--is_reduction', type=bool, default=True)
    parser.add_argument('--is_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_dataset', type=str, default='2009')
    parser.add_argument('--test_dataset', type=str, default='2010')


    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if args.dataset == 'H1N1':
        data_file_path = f'H1N1/H1N1_graph_{args.test_dataset}.pt'
    elif args.dataset == 'eth':
        data_file_path = f'graph_dat_eth/{args.test_dataset}.pt'

    data = torch.load(data_file_path, weights_only=False)
    data = data.to(device)
    model_path = F"./pre_trained_gnn/{args.dataset}/{args.pretrain_dataset}.pth"
    
    config_test = yaml.load(open(args.config), Loader=SafeLoader)['Cora']                
    input_dim = data.x.shape[1]
    data.edge_index = add_remaining_self_loops(data.edge_index)[0]

    index = np.arange(len(data.x))
    np.random.shuffle(index)
    train_mask = index[:int(len(index) * 0.7)]
    val_mask = index[int(len(index) * 0.7):int(len(index) * 1)]
    data.train_mask = train_mask
    data.val_mask = val_mask

    gnn = GNN(input_dim, config_test['output_dim'], act(config_test['activation']), config_test['gnn_type'], config_test['num_layers'])
    gnn.load_state_dict(torch.load(model_path, weights_only=False))
    gnn.to(device)
    for param in gnn.conv.parameters():
        param.requires_grad = False
    gnn.eval()
    model = AdapterGNN(gnn)
    model.to(device)

    logreg = LogReg(config_test['output_dim'], 13)
    logreg = logreg.to(device)
    loss_fn = nn.CrossEntropyLoss()

    # AdapterGNN
    model_param_group = []
    model_param_group.append({"params": model.prompts.parameters(), "lr": args.lr})
    model_param_group.append({"params": model.gating_parameter, "lr": args.lr})
    for name, p in model.named_parameters():
        if name.startswith('batch_norms'):
            model_param_group.append({"params": p})
        if 'mlp' in name and name.endswith('bias'):
            model_param_group.append({"params": p})
    model_param_group.append({"params": logreg.parameters(), "lr": args.lr})

    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

    train_dataloader = NeighborLoader(data, input_nodes=data.train_mask, num_neighbors=[15, 10], batch_size=128, shuffle=True)
    val_dataloader = NeighborLoader(data, input_nodes=data.val_mask, num_neighbors=[15, 10], batch_size=128, shuffle=True)

    best_epoch = 0
    best_train_acc = 0
    best_train_recall = 0
    best_train_f1 = 0
    best_val_acc = 0
    best_val_recall = 0
    best_val_f1 = 0

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_acc, train_recall, train_f1 = train(model, logreg, train_dataloader, loss_fn, optimizer)
        end = time.time()
        val_acc, val_recall, val_f1 = eval(model, logreg, val_dataloader)
        print('Epoch: {}, train_acc: {:.4f}, train_recall: {:.4f}, train_f1: {:.4f}, val_acc: {:4f}, val_recall: {:4f}, val_f1: {:4f}'.format(
            epoch, train_acc, train_recall, train_f1, val_acc, val_recall, val_f1))
        if best_val_acc <= val_acc:
            best_val_acc = val_acc
            best_val_recall = val_recall
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            best_train_acc = train_acc
            best_train_recall = train_recall
            best_train_f1 = train_f1
            print(f"✓ New best val_acc.")

    print('epoch: {}, train_acc: {:4f}, val_acc: {:4f}, val_recall: {:4f}, val_f1: {:4f}'.format(
        best_epoch, best_train_acc, best_val_acc, best_val_recall, best_val_f1))
    result_path = f'./result/{args.dataset}'
    os.makedirs(result_path, exist_ok=True)
    with open(result_path + '/AdapterGNN.txt', 'a') as f:
        f.write(f'{args.pretrain_dataset} to {args.test_dataset}: seed: %d, epoch: %d, train_acc: %f, train_recall: %f, train_f1: %f, val_acc: %f, val_recall: %f, val_f1: %f\n' % 
                (args.seed, best_epoch, best_train_acc, best_train_recall, best_train_f1, best_val_acc, best_val_recall, best_val_f1))


if __name__ == '__main__':
    main()
