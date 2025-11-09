import torch
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
# from torch_geometric.nn import GCNConv, GATConv


num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3


class AdapterGNN(torch.nn.Module):
    def __init__(self, gnn, drop_ratio=0):
        super(AdapterGNN, self).__init__()
        self.gnn = gnn
        self.num_layer = len(gnn.conv)
        self.drop_ratio = drop_ratio

        bottleneck_dim = 15
        prompt_num = 2

        gating = 0.01
        self.gating_parameter = torch.nn.Parameter(torch.zeros(prompt_num, self.num_layer, 1))
        self.gating_parameter.data += gating
        self.register_parameter('gating_parameter', self.gating_parameter)
        self.gating = self.gating_parameter

        # ----------------------------------parameter-----------------------------------
        
        self.batch_norms = torch.nn.ModuleList()
        self.prompts = torch.nn.ModuleList()
        for i in range(prompt_num):
            self.prompts.append(torch.nn.ModuleList())

        for layer in range(self.num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(gnn.conv[layer].out_channels))
            for i in range(prompt_num):
                if bottleneck_dim>0:
                    self.prompts[i].append(torch.nn.Sequential(
                        torch.nn.Linear(gnn.conv[layer].in_channels, bottleneck_dim),
                        torch.nn.ReLU(),
                        torch.nn.Linear(bottleneck_dim, gnn.conv[layer].out_channels),
                        torch.nn.BatchNorm1d(gnn.conv[layer].out_channels)
                    ))
                    torch.nn.init.zeros_(self.prompts[i][-1][2].weight.data)
                    torch.nn.init.zeros_(self.prompts[i][-1][2].bias.data)
                else:
                    self.prompts[i].append(torch.nn.BatchNorm1d(gnn.conv[layer].out_channels))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h_list = [x]
        for layer in range(self.num_layer):
            h = h_list[layer]

            h_mlp = self.gnn.conv[layer](h, edge_index)
            x_aggr = self.gnn.conv[layer].mess(h, edge_index)

            h = self.batch_norms[layer](h_mlp)

            delta = self.prompts[0][layer](h_list[layer])
            h = h + delta * self.gating[0][layer]
            delta = self.prompts[1][layer](x_aggr)
            h = h + delta * self.gating[1][layer]

            if layer < self.num_layer - 1:
                h = F.relu(h)
            h = F.dropout(h, self.drop_ratio, training=self.training)

            h_list.append(h)

        node_representation = h_list[-1]

        return node_representation


if __name__ == "__main__":
    pass