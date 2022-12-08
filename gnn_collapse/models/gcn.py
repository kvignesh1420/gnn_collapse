import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from gnn_collapse.models.common import Normalize


class GCNSubModule(torch.nn.Module):
    def __init__(self, in_feature_dim, out_feature_dim, batch_norm):
        super().__init__()
        self.in_feature_dim = in_feature_dim
        self.out_feature_dim = out_feature_dim
        self.batch_norm=batch_norm
        self.conv = GCNConv(self.in_feature_dim, self.out_feature_dim)
        self.norm = Normalize(self.out_feature_dim, norm="batch")

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        if self.batch_norm:
            x = self.norm(x)
        x = F.relu(x)
        return x

class GCN(torch.nn.Module):
    def __init__(self, input_feature_dim, hidden_feature_dim, num_classes, L=3, batch_norm=True):
        super().__init__()
        self.name = "gcn"
        self.batch_norm = batch_norm
        self.fc1 = torch.nn.Linear(input_feature_dim, hidden_feature_dim)
        self.conv_layers = [
            GCNSubModule(in_feature_dim=hidden_feature_dim, out_feature_dim=hidden_feature_dim, batch_norm=batch_norm)
            for _ in range(L-1)
        ]
        self.conv_layers.append(GCNSubModule(in_feature_dim=hidden_feature_dim, out_feature_dim=num_classes, batch_norm=batch_norm))
        self.conv_layers = torch.nn.ModuleList(self.conv_layers)
        self.fc2 = torch.nn.Linear(num_classes, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.fc1(x)
        x = F.relu(x)
        for conv_layer in self.conv_layers:
            x = conv_layer(x, edge_index)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
