import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, CitationFull, Amazon, Reddit
import functools
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder
import numpy as np

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("create folder {}".format(path))
    else:
        print("folder exists! {}".format(path))


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


def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'Computers', 'Reddit', 'Photo', 'Squirrel', 'Chameleon', 'ogbn_arxiv', 'ogbn_products']
    name = 'dblp' if name == 'DBLP' else name
    if name == 'dblp':
        return CitationFull(path, name, T.NormalizeFeatures())
    elif (name == 'Computers') | (name == 'Photo'):
        return Amazon(path, name, T.NormalizeFeatures())
    elif name == 'Reddit':
        return Reddit(path, transform=T.NormalizeFeatures())
    else:
        return Planetoid(path, name, transform=T.NormalizeFeatures())
    # if name == 'dblp':
    #     return CitationFull(path, name)
    # elif (name == 'Computers') | (name == 'Photo'):
    #     return Amazon(path, name)
    # elif name == 'Reddit':
    #     return Reddit(path)
    # else:
    #     return Planetoid(path, name)


def repeat(n_times):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            results = [f(*args, **kwargs) for _ in range(n_times)]
            statistics = {}
            for key in results[0].keys():
                values = [r[key] for r in results]
                statistics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)}
            print_statistics(statistics, f.__name__)
            return statistics
        return wrapper
    return decorator


def print_statistics(statistics, function_name):
    print(f'(E) | {function_name}:', end=' ')
    for i, key in enumerate(statistics.keys()):
        mean = statistics[key]['mean']
        std = statistics[key]['std']
        print(f'{key}={mean:.4f}+-{std:.4f}', end='')
        with open('./TransferGNN/result/GRACE.txt', 'a') as f:
            f.write(f'{key}={mean:.4f}+-{std:.4f}')
        if i != len(statistics.keys()) - 1:
            print(',', end=' ')
        else:
            print()


def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret


@repeat(3)
def label_classification(embeddings, y, ratio):
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(bool)

    X = normalize(X, norm='l2')

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1 - ratio)

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)

    acc = accuracy_score(y_test, y_pred)
    auc_micro = roc_auc_score(y_test, y_pred, average="micro")
    auc_macro = roc_auc_score(y_test, y_pred, average="macro")
    F1_micro = f1_score(y_test, y_pred, average="micro")
    F1_macro = f1_score(y_test, y_pred, average="macro")

    return {
        'acc': acc,
        'AucMi': auc_micro,
        'AucMa': auc_macro,
        'F1Mi': F1_micro,
        'F1Ma': F1_macro
    }


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # nn.init.xavier_uniform_(m.weight)
        # nn.init.xavier_normal_(m.weight)
        # nn.init.kaiming_normal_(m.weight)
        nn.init.kaiming_uniform_(m.weight)
        # nn.init.normal_(m.weight)
        # nn.init.uniform_(m.weight)
        nn.init.zeros_(m.bias)


class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target, ppr=None):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            if ppr is None:
                XX = torch.mean(kernels[:batch_size, :batch_size])
            else:
                XX = torch.mean(kernels[:batch_size, :batch_size] * ppr)
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss
