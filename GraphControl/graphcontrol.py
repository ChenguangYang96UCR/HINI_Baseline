import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import numpy as np
import random
from sklearn.metrics import recall_score, f1_score, accuracy_score


from utils.random import reset_random_seed
from utils.args import Arguments
from utils.sampling import collect_subgraphs
from utils.transforms import process_attributes, obtain_attributes
from models import load_model
from datasets import NodeDataset
from optimizers import create_optimizer

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
        
        torch.use_deterministic_algorithms(True)
    print(f"Random seed set to {seed}")

def preprocess(config, dataset_obj, device):
    kwargs = {'batch_size': config.batch_size, 'num_workers': 4, 'persistent_workers': True, 'pin_memory': True}
    
    print('generating subgraphs....')
    
    train_idx = dataset_obj.data.train_mask.nonzero().squeeze()
    test_idx = dataset_obj.data.test_mask.nonzero().squeeze()
    
    train_graphs = collect_subgraphs(train_idx, dataset_obj.data, walk_steps=config.walk_steps, restart_ratio=config.restart)
    test_graphs = collect_subgraphs(test_idx, dataset_obj.data, walk_steps=config.walk_steps, restart_ratio=config.restart)

    [process_attributes(g, use_adj=config.use_adj, threshold=config.threshold, num_dim=config.num_dim) for g in train_graphs]
    [process_attributes(g, use_adj=config.use_adj, threshold=config.threshold, num_dim=config.num_dim) for g in test_graphs]
    
        
    train_loader = DataLoader(train_graphs, shuffle=True, **kwargs)
    test_loader = DataLoader(test_graphs, **kwargs)

    return train_loader, test_loader


def finetune(config, model, train_loader, device, full_x_sim, test_loader):
    # freeze the pre-trained encoder (left branch)
    for k, v in model.named_parameters():
        if 'encoder' in k:
            v.requires_grad = False
            
    model.reset_classifier()
    eval_steps = 3
    patience = 15
    count = 0
    best_acc = 0
    best_recall = 0
    best_f1 = 0

    params  = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = create_optimizer(name=config.optimizer, parameters=params, lr=config.lr, weight_decay=config.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    process_bar = tqdm(range(config.epochs))

    for epoch in process_bar:
        for data in train_loader:
            optimizer.zero_grad()
            model.train()

            data = data.to(device)
            
            if not hasattr(data, 'root_n_id'):
                data.root_n_id = data.root_n_index

            sign_flip = torch.rand(data.x.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            x = data.x * sign_flip.unsqueeze(0)
            
            x_sim = full_x_sim[data.original_idx]
            preds = model.forward_subgraph(x, x_sim, data.edge_index, data.batch, data.root_n_id, frozen=True)
                
            loss = criterion(preds, data.y)
            loss.backward()
            optimizer.step()
    
        if epoch % eval_steps == 0:
            acc, recall, f1 = eval_subgraph(config, model, test_loader, device, full_x_sim)
            process_bar.set_postfix({"Epoch": epoch, "Acc": f"{acc:.4f}", "Recall": f"{recall:.4f}", "F1": f"{f1:.4f}"})
            if best_acc < acc:
                best_acc = acc
                best_recall = recall
                best_f1 = f1
                count = 0
            else:
                count += 1

        if count == patience:
            break

    return best_acc, best_recall, best_f1


def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    
    dataset_obj = NodeDataset(config.dataset, n_seeds=config.seeds)
    dataset_obj.print_statistics()
    
    # For large graph, we use cpu to preprocess it rather than gpu because of OOM problem.
    if dataset_obj.num_nodes < 30000:
        dataset_obj.to(device)
    x_sim = obtain_attributes(dataset_obj.data, use_adj=False, threshold=config.threshold).to(device)
    
    dataset_obj.to('cpu') # Otherwise the deepcopy will raise an error
    num_node_features = config.num_dim

    train_masks = dataset_obj.data.train_mask
    test_masks = dataset_obj.data.test_mask

    acc_list = []
    recall_list = []
    f1_list = []

    for i, seed in enumerate(config.seeds):
        set_seed(seed)
        if dataset_obj.random_split:
            dataset_obj.data.train_mask = train_masks[:, seed]
            dataset_obj.data.test_mask = test_masks[:, seed]
        
        train_loader, test_loader = preprocess(config, dataset_obj, device)
        
        model = load_model(num_node_features, dataset_obj.num_classes, config)
        model = model.to(device)

        # finetuning model
        best_acc, best_recall, best_f1 = finetune(config, model, train_loader, device, x_sim, test_loader)
        
        acc_list.append(best_acc)
        recall_list.append(best_recall)
        f1_list.append(best_f1)
        print(f'Seed: {seed}, Accuracy: {best_acc:.4f}, Recall (macro): {best_recall:.4f}, F1 (macro): {best_f1:.4f}')
        result_path = './result'
        with open(result_path + '/GraphControl.txt', 'a') as f:
            # 使用您提供的格式
            # 注意: 'best_loss' 和 'best_train_*' 指标在当前脚本中未计算，因此用 'nan' 填充
            f.write('2010: seed: %d, val_acc: %f, val_recall: %f, val_f1: %f\n' % 
                    (seed, best_acc, best_recall, best_f1))


    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")


def eval_subgraph(config, model, test_loader, device, full_x_sim):
    model.eval()
    all_preds = []
    all_labels = []
    for batch in test_loader:
        batch = batch.to(device)
        if not hasattr(batch, 'root_n_id'):
            batch.root_n_id = batch.root_n_index
        x_sim = full_x_sim[batch.original_idx]
        preds = model.forward_subgraph(batch.x, x_sim, batch.edge_index, batch.batch, batch.root_n_id, frozen=True).argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(batch.y.cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    acc = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return acc, recall, f1

if __name__ == '__main__':
    config = Arguments().parse_args()
    
    main(config)