"""
Baseline and graph neural network models
"""

import torch
import torch.nn.functional as F
from gnn_collapse.models.common import Normalize

class MLPSubModule(torch.nn.Module):
    def __init__(self, in_feature_dim, out_feature_dim, batch_norm, non_linearity, use_bias):
        super().__init__()
        self.in_feature_dim = in_feature_dim
        self.out_feature_dim = out_feature_dim
        self.non_linearity=non_linearity
        self.batch_norm=batch_norm
        self.lin = torch.nn.Linear(self.in_feature_dim, self.out_feature_dim, bias=use_bias)
        self.norm = Normalize(self.out_feature_dim, norm="batch")

    def forward(self, x):
        x = self.lin(x)
        if self.batch_norm:
            x = self.norm(x)
        if self.non_linearity == "relu":
            x = F.relu(x)
        return x

class MLP(torch.nn.Module):
    def __init__(self, input_feature_dim, hidden_feature_dim,
                 loss_type, num_classes, L=3, batch_norm=True,
                 non_linearity="relu", use_bias=True):
        super().__init__()
        self.name = "mlp"
        self.non_linearity = non_linearity
        self.batch_norm = batch_norm
        self.loss_type = loss_type
        self.fc_init = torch.nn.Linear(input_feature_dim, hidden_feature_dim)
        self.fc_layers = [
            MLPSubModule(
                in_feature_dim=hidden_feature_dim,
                out_feature_dim=hidden_feature_dim,
                batch_norm=batch_norm,
                non_linearity=non_linearity,
                bias=use_bias
            )
            for _ in range(L)
        ]
        self.fc_layers = torch.nn.ModuleList(self.fc_layers)
        self.final_layer = torch.nn.Linear(hidden_feature_dim, num_classes, bias=use_bias)

    def forward(self, data):
        x = data.x
        x = self.fc_init(x)
        if self.non_linearity == "relu":
            x = F.relu(x)
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        x = self.final_layer(x)
        if self.loss_type == "mse":
            return x
        return F.log_softmax(x, dim=1)
