from model.GNN_model import GNN
import torch
import torch.nn as nn
import os
from util import act, mkdir
import torch.nn.functional as F
import numpy as np
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


class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)
        self.initialize()

    def forward(self, x):
        return self.fc(x)
    
    def initialize(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)


def transfer2(pretrain_data, downstream_data, pretrained_gnn_state, args, config, gpu_id, seed, train_dataset, test_dataset, dataset):
    set_seed(seed)
    device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')
    pretrain_data = pretrain_data
    downstream_data = downstream_data

    gnn = GNN(pretrain_data.x.shape[1], config['output_dim'], act(config['activation']), config['gnn_type'], config['num_layers'])
    gnn.load_state_dict(pretrained_gnn_state)
    # gnn.to(device)

    num_classes = 13
    logreg = LogReg(config['output_dim'], num_classes)
    # logreg = logreg.to(device)
    loss_fn = nn.CrossEntropyLoss()

    index = np.arange(downstream_data.x.shape[0])
    np.random.shuffle(index)
    train_mask = torch.zeros(downstream_data.x.shape[0]).bool()
    val_mask = torch.zeros(downstream_data.x.shape[0]).bool()
    test_mask = torch.zeros(downstream_data.x.shape[0]).bool()
    train_mask[index[:int(len(index) * 0.1)]] = True
    val_mask[index[int(len(index) * 0.2):int(len(index) * 1)]] = True
    test_mask[index[int(len(index) * 1):]] = True

    downstream_data.train_mask = train_mask
    downstream_data.val_mask = val_mask
    downstream_data.test_mask = test_mask
    train_labels = downstream_data.y[train_mask]
    val_labels = downstream_data.y[val_mask]
    test_labels = downstream_data.y[test_mask]

    optimizer = torch.optim.Adam([{"params": logreg.parameters(), 'lr': args.lr2, 'weight_decay': args.wd2}, {"params": gnn.parameters(), 'lr': args.lr3, 'weight_decay': args.wd3}])

    best_epoch = 0
    best_loss = 0
    best_train_acc = 0
    best_train_recall = 0
    best_train_f1 = 0
    best_val_acc = 0
    best_val_recall = 0
    best_val_f1 = 0

    for epoch in range(0, args.num_epochs):
        gnn.train()
        logreg.train()
    
        emb = gnn(downstream_data.x, downstream_data.edge_index)        
        train_labels = downstream_data.y[train_mask]
        optimizer.zero_grad()

        logits = logreg(emb)
        train_logits = logits[train_mask]
        train_preds = torch.argmax(train_logits, dim=1)
        loss = loss_fn(train_logits, train_labels)
        loss.backward()
        optimizer.step()

        train_acc = torch.sum(train_preds == train_labels).float() / train_labels.shape[0]
        train_preds_np = train_preds.cpu().numpy()
        train_labels_np = train_labels.cpu().numpy()
        train_recall = recall_score(train_labels_np, train_preds_np, average='macro', zero_division=0)
        train_f1 = f1_score(train_labels_np, train_preds_np, average='macro', zero_division=0)
    
        gnn.eval()
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
    result_path = f'./result/{dataset}'
    mkdir(result_path)
    with open(result_path + '/result.txt', 'a') as f:
        f.write(f'{train_dataset} to {test_dataset}: seed: %d, epoch: %d, train_loss: %f, train_acc: %f, train_recall: %f, train_f1: %f, val_acc: %f, val_recall: %f, val_f1: %f\n' % 
                (seed, best_epoch, best_loss, best_train_acc, best_train_recall, best_train_f1, best_val_acc, best_val_recall, best_val_f1))
