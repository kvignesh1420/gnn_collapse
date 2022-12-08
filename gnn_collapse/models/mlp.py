"""
Baseline and graph neural network models
"""

import torch
import torch.nn.functional as F
from gnn_collapse.models.common import Normalize

class MLPSubModule(torch.nn.Module):
    def __init__(self, in_feature_dim, out_feature_dim, batch_norm):
        super().__init__()
        self.in_feature_dim = in_feature_dim
        self.out_feature_dim = out_feature_dim
        self.batch_norm=batch_norm
        self.lin = torch.nn.Linear(self.in_feature_dim, self.out_feature_dim)
        self.norm = Normalize(self.out_feature_dim, norm="batch")

    def forward(self, x):
        x = self.lin(x)
        if self.batch_norm:
            x = self.norm(x)
        x = F.relu(x)
        return x

class MLP(torch.nn.Module):
    def __init__(self, input_feature_dim, hidden_feature_dim, num_classes, L=3, batch_norm=True):
        super().__init__()
        self.name = "mlp"
        self.batch_norm = batch_norm
        self.fc_init = torch.nn.Linear(input_feature_dim, hidden_feature_dim)
        self.fc_layers = [
            MLPSubModule(in_feature_dim=hidden_feature_dim, out_feature_dim=hidden_feature_dim, batch_norm=batch_norm)
            for _ in range(L-1)
        ]
        self.fc_layers.append(MLPSubModule(in_feature_dim=hidden_feature_dim, out_feature_dim=num_classes, batch_norm=batch_norm))
        self.fc_layers = torch.nn.ModuleList(self.fc_layers)
        self.fc2 = torch.nn.Linear(num_classes, num_classes)

    def forward(self, data):
        x = data.x
        x = self.fc_init(x)
        x = F.relu(x)
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        x = self.fc_final(x)
        return F.log_softmax(x, dim=1)
