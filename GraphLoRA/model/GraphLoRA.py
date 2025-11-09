from model.GNN_model import GNN, GNNLoRA
import torch
import torch.nn as nn
import os
from torch_geometric.transforms import SVDFeatureReduction
from util import get_dataset, act, SMMDLoss, mkdir, get_ppr_weight
from util import get_few_shot_mask, batched_smmd_loss, batched_gct_loss
from torch_geometric.utils import to_dense_adj, add_remaining_self_loops
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader
import random
from sklearn.metrics import recall_score, f1_score

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


class Projector(nn.Module):
    def __init__(self, input_size, output_size):
        super(Projector, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.initialize()

    def forward(self, x):
        return self.fc(x)

    def initialize(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)


class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)
        self.initialize()

    def forward(self, x):
        return self.fc(x)
    
    def initialize(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)


def transfer2(pretrain_data, downstream_data, pretrained_gnn_state, args, config, gpu_id, seed, pre_dataset, downstream_dataset,is_reduction = False):
    set_seed(seed)
    device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')
    pretrain_data = pretrain_data
    downstream_data = downstream_data

    if is_reduction:
        feature_reduce = SVDFeatureReduction(out_channels=100)
        pretrain_data = feature_reduce(pretrain_data)
        downstream_data = feature_reduce(downstream_data)
    

    pretrain_data.edge_index = add_remaining_self_loops(pretrain_data.edge_index, num_nodes=pretrain_data.num_nodes)[0]
    downstream_data.edge_index = add_remaining_self_loops(downstream_data.edge_index, num_nodes=downstream_data.num_nodes)[0]
    # pretrain_data = pretrain_data.to(device)
    # downstream_data = downstream_data.to(device)

    gnn = GNN(pretrain_data.x.shape[1], config['output_dim'], act(config['activation']), config['gnn_type'], config['num_layers'])
    gnn.load_state_dict(pretrained_gnn_state)
    # gnn.to(device)
    gnn.eval()
    for param in gnn.conv.parameters():
        param.requires_grad = False

    gnn2 = GNNLoRA(pretrain_data.x.shape[1], config['output_dim'], act(config['activation']), gnn, config['gnn_type'], config['num_layers'], r=args.r)
    # gnn2.to(device)
    gnn2.train()

    SMMD = SMMDLoss()
    projector = Projector(downstream_data.x.shape[1], pretrain_data.x.shape[1])
    # projector = projector.to(device)
    projector.train()

    # optimizer
    num_classes = 13
    logreg = LogReg(config['output_dim'], num_classes)
    # logreg = logreg.to(device)
    loss_fn = nn.CrossEntropyLoss()

    index = np.arange(downstream_data.x.shape[0])
    np.random.shuffle(index)
    train_mask = torch.zeros(downstream_data.x.shape[0]).bool()
    val_mask = torch.zeros(downstream_data.x.shape[0]).bool()
    test_mask = torch.zeros(downstream_data.x.shape[0]).bool()
    train_mask[index[:int(len(index) * 0.7)]] = True
    val_mask[index[int(len(index) * 0.7):int(len(index) * 1)]] = True
    test_mask[index[int(len(index) * 1):]] = True

    mask = torch.zeros((downstream_data.x.shape[0], downstream_data.x.shape[0]))
    idx_a = torch.tensor([])
    idx_b = torch.tensor([])
    for i in range(num_classes):
        train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
        train_label = downstream_data.y[train_idx]
        idx_a = torch.concat((idx_a, train_idx[train_label == i].repeat_interleave(len(train_idx[train_label == i]))))
        idx_b = torch.concat((idx_b, train_idx[train_label == i].repeat(len(train_idx[train_label == i]))))
    mask = torch.sparse_coo_tensor(indices=torch.stack((idx_a, idx_b)), values=torch.ones(len(idx_a)), size=[downstream_data.x.shape[0], downstream_data.x.shape[0]]).to_dense()
    mask = args.sup_weight * (mask - torch.diag_embed(torch.diag(mask))) + torch.eye(downstream_data.x.shape[0])
    
    optimizer = torch.optim.Adam([{"params": projector.parameters(), 'lr': args.lr1, 'weight_decay': args.wd1}, {"params": logreg.parameters(), 'lr': args.lr2, 'weight_decay': args.wd2}, {"params": gnn2.parameters(), 'lr': args.lr3, 'weight_decay': args.wd3}])

    downstream_data.train_mask = train_mask
    downstream_data.val_mask = val_mask
    downstream_data.test_mask = test_mask

    train_labels = downstream_data.y[train_mask]
    val_labels = downstream_data.y[val_mask]
    test_labels = downstream_data.y[test_mask]

    pretrain_graph_loader = DataLoader(pretrain_data.x, batch_size=128, shuffle=True)

    best_epoch = 0
    best_loss = 0
    best_train_acc = 0
    best_train_recall = 0
    best_train_f1 = 0
    best_val_acc = 0
    best_val_recall = 0
    best_val_f1 = 0

    num_nodes = downstream_data.x.shape[0]
    target_adj = to_dense_adj(downstream_data.edge_index, max_num_nodes=num_nodes)[0]
    ppr_weight = get_ppr_weight(downstream_data)

    for epoch in range(0, args.num_epochs):
        logreg.train()
        projector.train()
  
        pos_weight = float(target_adj.shape[0] * target_adj.shape[0] - target_adj.sum()) / target_adj.sum()
        weight_mask = target_adj.view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0))
        weight_tensor[weight_mask] = pos_weight

        feature_map = projector(downstream_data.x)
        emb, emb1, emb2 = gnn2(feature_map, downstream_data.edge_index)
        train_labels = downstream_data.y[train_mask]
        optimizer.zero_grad()

        smmd_loss_f = batched_smmd_loss(feature_map, pretrain_graph_loader, SMMD, ppr_weight, 128)      
        ct_loss = 0.5 * (batched_gct_loss(emb1, emb2, 1000, mask, args.tau) + batched_gct_loss(emb2, emb1, 1000, mask, args.tau)).mean()
        logits = logreg(emb)
        train_logits = logits[train_mask]

        rec_adj = torch.sigmoid(torch.matmul(torch.softmax(logits, dim=1), torch.softmax(logits, dim=1).T))
        loss_rec = F.binary_cross_entropy(rec_adj.view(-1), target_adj.view(-1), weight=weight_tensor)

        train_preds = torch.argmax(train_logits, dim=1)
        cls_loss = loss_fn(train_logits, train_labels)
        loss = args.l1 * cls_loss + args.l2 * smmd_loss_f +  args.l3 * ct_loss + args.l4 * loss_rec
        loss.backward()
        optimizer.step()

        train_acc = torch.sum(train_preds == train_labels).float() / train_labels.shape[0]
        train_preds_np = train_preds.cpu().numpy()
        train_labels_np = train_labels.cpu().numpy()
        train_recall = recall_score(train_labels_np, train_preds_np, average='macro', zero_division=0)
        train_f1 = f1_score(train_labels_np, train_preds_np, average='macro', zero_division=0)
    
        logreg.eval()
        projector.eval()
        with torch.no_grad():
            val_logits = logits[val_mask]
            val_preds = torch.argmax(val_logits, dim=1)
            val_acc = torch.sum(val_preds == val_labels).float() / val_labels.shape[0]
            val_labels_np = val_labels.cpu().numpy()
            val_preds_np = val_preds.cpu().numpy()
            val_recall = recall_score(val_labels_np, val_preds_np, average='macro', zero_division=0)
            val_f1 = f1_score(val_labels_np, val_preds_np, average='macro', zero_division=0)
            print('Epoch: {}, loss: {:.4f}, train_acc: {:.4f}, train_recall: {:.4f}, train_f1: {:.4f}, val_acc: {:4f}, val_recall: {:4f}, val_f1: {:4f}'.format(
                epoch, loss, train_acc, train_recall, train_f1, val_acc, val_recall, val_f1))

            if best_val_acc <= val_acc:
                best_val_acc = val_acc
                best_val_recall = val_recall
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                best_loss = loss
                best_train_acc = train_acc
                best_train_recall = train_recall
                best_train_f1 = train_f1
                print(f"✓ New best val_acc.")
                
                preds_to_save = val_preds.cpu().numpy()
                labels_to_save = val_labels.cpu().numpy()
                output_data = np.vstack((preds_to_save, labels_to_save)).T

                output_filename = f"predictions&true_seed{seed}.csv"
                np.savetxt(
                    output_filename, 
                    output_data, 
                    fmt='%d',
                    delimiter=',',
                    header='prediction,true_label',
                    comments=''
                )
                print(f"✓ Predictions and labels saved to {output_filename}")

    print('epoch: {}, train_acc: {:4f}, val_acc: {:4f}, val_recall: {:4f}, val_f1: {:4f}'.format(
        best_epoch, best_train_acc, best_val_acc, best_val_recall, best_val_f1))
    result_path = './result'
    mkdir(result_path)
    with open(result_path + '/result.txt', 'a') as f:
        f.write(f'{pre_dataset} to {downstream_dataset}: seed: %d, epoch: %d, train_loss: %f, train_acc: %f, train_recall: %f, train_f1: %f, val_acc: %f, val_recall: %f, val_f1: %f\n' % 
                (seed, best_epoch, best_loss, best_train_acc, best_train_recall, best_train_f1, best_val_acc, best_val_recall, best_val_f1))

