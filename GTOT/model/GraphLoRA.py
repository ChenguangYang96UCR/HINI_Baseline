from model.GNN_model import GNN
import torch
import torch.nn as nn
import os
from util import act, mkdir
import torch.nn.functional as F
import numpy as np


class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)
        self.initialize()

    def forward(self, x):
        return self.fc(x)
    
    def initialize(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)


def transfer2(pretrain_data, downstream_data, pretrained_gnn_state, args, config, gpu_id):
    device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')
    pretrain_data = pretrain_data.to(device)
    downstream_data = downstream_data.to(device)

    gnn = GNN(pretrain_data.x.shape[1], config['output_dim'], act(config['activation']), config['gnn_type'], config['num_layers'])
    gnn.load_state_dict(pretrained_gnn_state)
    gnn.to(device)

    num_classes = 13
    logreg = LogReg(config['output_dim'], num_classes)
    logreg = logreg.to(device)
    loss_fn = nn.CrossEntropyLoss()

    index = np.arange(downstream_data.x.shape[0])
    np.random.shuffle(index)
    train_mask = torch.zeros(downstream_data.x.shape[0]).bool().to(device)
    val_mask = torch.zeros(downstream_data.x.shape[0]).bool().to(device)
    test_mask = torch.zeros(downstream_data.x.shape[0]).bool().to(device)
    train_mask[index[:int(len(index) * 0.7)]] = True
    val_mask[index[int(len(index) * 0.7):int(len(index) * 1)]] = True
    test_mask[index[int(len(index) * 1):]] = True

    downstream_data.train_mask = train_mask
    downstream_data.val_mask = val_mask
    downstream_data.test_mask = test_mask
    train_labels = downstream_data.y[train_mask]
    val_labels = downstream_data.y[val_mask]
    test_labels = downstream_data.y[test_mask]

    optimizer = torch.optim.Adam([{"params": logreg.parameters(), 'lr': args.lr2, 'weight_decay': args.wd2}, {"params": gnn.parameters(), 'lr': args.lr3, 'weight_decay': args.wd3}])

    max_acc = 0
    max_test_acc = 0
    max_epoch = 0

    for epoch in range(0, args.num_epochs):
        gnn.train()
        logreg.train()
    
        emb = gnn(downstream_data.x, downstream_data.edge_index)        
        train_labels = downstream_data.y[train_mask]
        optimizer.zero_grad()

        logits = logreg(emb)
        train_logits = logits[train_mask]
        preds = torch.argmax(train_logits, dim=1)
        loss = loss_fn(train_logits, train_labels)
        loss.backward()
        optimizer.step()

        train_acc = torch.sum(preds == train_labels).float() / train_labels.shape[0]
        gnn.eval()
        logreg.eval()
        with torch.no_grad():
            val_logits = logits[val_mask]
            test_logits = logits[test_mask]
            val_preds = torch.argmax(val_logits, dim=1)
            test_preds = torch.argmax(test_logits, dim=1)
            val_acc = torch.sum(val_preds == val_labels).float() / val_labels.shape[0]
            test_acc = torch.sum(test_preds == test_labels).float() / test_labels.shape[0]
            print('Epoch: {}, train_loss: {:.4f}, train_acc: {:.4f}, val_acc: {:4f}, test_acc: {:4f}'.format(epoch, loss, train_acc, val_acc, test_acc))
            if max_acc < val_acc:
                max_acc = val_acc
                max_test_acc = test_acc
                max_epoch = epoch + 1
                print(f"✓ New best val_acc. Saving validation predictions...")
                
                preds_to_save = val_preds.cpu().numpy()
                labels_to_save = val_labels.cpu().numpy()
                output_data = np.vstack((preds_to_save, labels_to_save)).T

                output_filename = f"validation_predictions_r.csv"
                np.savetxt(
                    output_filename, 
                    output_data, 
                    fmt='%d',
                    delimiter=',',
                    header='prediction,true_label',
                    comments=''
                )
                print(f"✓ Predictions and labels saved to {output_filename}")

    print('epoch: {}, val_acc: {:4f}, test_acc: {:4f}'.format(max_epoch, max_acc, max_test_acc))
    result_path = './result'
    mkdir(result_path)
    with open(result_path + '/GRACEt.txt', 'a') as f:
        f.write('2010 to 2011: epoch: %d, train_locc: %f, val_acc: %f, test_acc: %f\n'%(max_epoch, loss, max_acc, max_test_acc))

