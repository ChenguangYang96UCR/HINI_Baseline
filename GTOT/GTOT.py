import yaml
from yaml import SafeLoader
from pre_train import pretrain2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.utils import to_dense_adj, add_self_loops
import os
import numpy as np
from torch_geometric.nn import GCNConv
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

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("create folder {}".format(path))

def act(act_type='leakyrelu'):
    if act_type == 'leakyrelu':
        return F.leaky_relu
    elif act_type == 'tanh':
        return torch.tanh
    elif act_type == 'relu':
        return F.relu
    elif act_type == 'prelu':
        return nn.PReLU()
    elif act_type == 'sigmiod':
        return F.sigmoid

class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)
        self.initialize()

    def forward(self, x):
        return self.fc(x)
    
    def initialize(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)

class GTOT(nn.Module):
    r"""
    GTOT implementation from the official repository.
    """
    def __init__(self, eps=0.1, thresh=0.1, max_iter=100):
        super(GTOT, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.thresh = thresh

    def forward(self, x, y, C=None, A=None):
        x = x.to(C.device)
        y = y.to(C.device)
        num_nodes = x.shape[0]
        mu = torch.ones(num_nodes, dtype=torch.float, device=C.device) / num_nodes
        nu = torch.ones(num_nodes, dtype=torch.float, device=C.device) / num_nodes
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        
        for i in range(self.max_iter):
            u1 = u
            u = self.eps * (torch.log(mu + 1e-8) - self.log_sum(self.exp_M(C, u, v, A=A), dim=-1)) + u
            v = self.eps * (torch.log(nu + 1e-8) - self.log_sum(self.exp_M(C, u, v, A=A).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).max()
            if err.item() < self.thresh:
                break
        
        U, V = u, v
        pi = self.exp_M(C, U, V, A=A)
        cost = torch.sum(pi * C)
        return cost, pi, C

    def M(self, C, u, v):
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    def exp_M(self, C, u, v, A=None):
        if A is not None:
            return torch.exp(self.M(C, u, v)) * (A > 0).float()
        else:
            return torch.exp(self.M(C, u, v))

    def log_sum(self, input_tensor, dim=-1):
        return torch.log(1e-8 + torch.sum(input_tensor, dim=dim))
    

class GNN(torch.nn.Module):
    def __init__(self, input_dim, out_dim, activation, gnn_type='GCN', gnn_layer_num=2):
        super().__init__()
        self.gnn_layer_num = gnn_layer_num
        self.activation = activation
        if gnn_type == 'GCN':
            GraphConv = GCNConv
        else:
            GraphConv = GCNConv

        self.gnn_type = gnn_type
        if gnn_layer_num < 1:
            raise ValueError('GNN layer_num should >=1 but you set {}'.format(gnn_layer_num))
        elif gnn_layer_num == 1:
            self.conv = nn.ModuleList([GraphConv(input_dim, out_dim)])
        elif gnn_layer_num == 2:
            self.conv = nn.ModuleList([GraphConv(input_dim, 2 * out_dim), GraphConv(2 * out_dim, out_dim)])
        else:
            layers = [GraphConv(input_dim, 2 * out_dim)]
            for i in range(gnn_layer_num - 2):
                layers.append(GraphConv(2 * out_dim, 2 * out_dim))
            layers.append(GraphConv(2 * out_dim, out_dim))
            self.conv = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        for conv in self.conv[0:-1]:
            x = conv(x, edge_index)
            x = self.activation(x)
        node_emb = self.conv[-1](x, edge_index)
        return node_emb
    
if __name__ == '__main__':
    set_seed(42)
    
    pretrain_graph_file_path = 'H1N1_graph_2010.pt'
    pretrain_graph = torch.load(pretrain_graph_file_path) 
    config_pretrain = yaml.load(open('config.yaml'), Loader=SafeLoader)['Cora']
    gpu_id = 0
    pretrained_gnn_state = pretrain2(pretrain_graph, "GRACE", config_pretrain, gpu_id)   

    seed = 0
    set_seed(seed)

    num_layers = 2
    lr = 0.02
    wd = 0.02
    epochs = 300
    trade_off_lambda = 0.01

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_file_path = 'H1N1_graph_2011.pt'
    data = torch.load(data_file_path)
    data.edge_index = add_self_loops(data.edge_index, num_nodes=data.x.shape[0])[0]
    data = data.to(device)
    adj_mask = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
    input_dim = data.x.shape[1]
    output_dim = 256
    activation = act('relu')
    gnn_type = 'GCN'

    model = GNN(input_dim, output_dim, activation, gnn_type, num_layers)
    source_model = GNN(input_dim, output_dim, activation, gnn_type, num_layers)
#    model_path = "./pre_trained_gnn/{}.pth".format(2010) 
#    model.load_state_dict(torch.load(model_path))
    model.load_state_dict(pretrained_gnn_state)
    model.to(device)    
#    source_model.load_state_dict(torch.load(model_path))
    source_model.load_state_dict(pretrained_gnn_state)
    source_model.to(device)
    source_model.eval()
    for param in source_model.parameters():
        param.requires_grad = False
    source_model.eval()
    with torch.no_grad():
        source_node_embs = source_model(data.x, data.edge_index)

    num_classes = 13
    logreg = LogReg(output_dim, num_classes)
    logreg = logreg.to(device)
    loss_fn = nn.CrossEntropyLoss()

    index = np.arange(data.x.shape[0])
    np.random.shuffle(index)
    train_mask = torch.zeros(data.x.shape[0]).bool().to(device)
    val_mask = torch.zeros(data.x.shape[0]).bool().to(device)
    test_mask = torch.zeros(data.x.shape[0]).bool().to(device)
    train_mask[index[:int(len(index) * 0.7)]] = True
    val_mask[index[int(len(index) * 0.7):int(len(index) * 1)]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask

    train_labels = data.y[train_mask]
    val_labels = data.y[val_mask]

    optimizer = torch.optim.Adam([{"params": logreg.parameters(), 'lr': lr, 'weight_decay': wd}, {"params": model.parameters(), 'lr': 0.00005, 'weight_decay': 0.1}])
    gtot_calculator = GTOT(eps=0.1, max_iter=20).to(device)

    best_epoch = 0
    best_loss = 0
    best_train_acc = 0
    best_train_recall = 0
    best_train_f1 = 0
    best_val_acc = 0
    best_val_recall = 0
    best_val_f1 = 0

    for epoch in range(1, epochs + 1):
        model.train()
        logreg.train()

        target_node_embs = model(data.x, data.edge_index)
        logits = logreg(target_node_embs)
        train_logits = logits[train_mask]
        train_preds = torch.argmax(train_logits, dim=1)
        loss_cls = loss_fn(train_logits, train_labels)
        C = 0.5 * (1 - F.cosine_similarity(source_node_embs.unsqueeze(1), target_node_embs.unsqueeze(0), dim=2))
        C = C / (C.max() + 1e-8)
        loss_gtot, _, _ = gtot_calculator(x=source_node_embs, y=target_node_embs, C=C, A=adj_mask)
        loss = loss_cls + trade_off_lambda * loss_gtot
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = torch.sum(train_preds == train_labels).float() / train_labels.shape[0]
        train_preds_np = train_preds.cpu().numpy()
        train_labels_np = train_labels.cpu().numpy()
        train_recall = recall_score(train_labels_np, train_preds_np, average='macro', zero_division=0)
        train_f1 = f1_score(train_labels_np, train_preds_np, average='macro', zero_division=0)

        logreg.eval()
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
    with open(result_path + '/GTOT.txt', 'a') as f:
        f.write('2010 to 2011: seed: %d, epoch: %d, train_loss: %f, train_acc: %f, train_recall: %f, train_f1: %f, val_acc: %f, val_recall: %f, val_f1: %f\n' % 
                (seed, best_epoch, best_loss, best_train_acc, best_train_recall, best_train_f1, best_val_acc, best_val_recall, best_val_f1))
