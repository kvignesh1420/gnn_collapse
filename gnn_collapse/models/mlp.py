"""
Baseline and graph neural network models
"""

import torch
import torch.nn.functional as F
from gnn_collapse.models.common import Normalize

class MLPSubModule(torch.nn.Module):
    def __init__(self, num_features, batch_norm):
        super().__init__()
        self.num_features = num_features
        self.batch_norm=batch_norm
        self.lin = torch.nn.Linear(self.num_features, self.num_features)
        self.norm = Normalize(self.num_features, norm="batch")

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
        self.fc_layers = torch.nn.ModuleList([
            MLPSubModule(num_features=hidden_feature_dim, batch_norm=batch_norm)
            for _ in range(L)
        ])
        self.fc_final = torch.nn.Linear(hidden_feature_dim, num_classes)

    def forward(self, data):
        x = data.x
        x = self.fc_init(x)
        x = F.relu(x)
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        x = self.fc_final(x)
        return F.log_softmax(x, dim=1)
