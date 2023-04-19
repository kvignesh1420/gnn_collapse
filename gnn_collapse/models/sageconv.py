import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from gnn_collapse.models.common import Normalize

class GSageSubModule(torch.nn.Module):
    def __init__(self, in_feature_dim, out_feature_dim, batch_norm, non_linearity, use_bias):
        super().__init__()
        self.in_feature_dim = in_feature_dim
        self.out_feature_dim = out_feature_dim
        self.non_linearity = non_linearity
        self.batch_norm=batch_norm
        self.conv = SAGEConv(self.in_feature_dim, self.out_feature_dim, bias=use_bias)
        self.norm = Normalize(self.out_feature_dim, norm="batch")

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        if self.batch_norm:
            x = self.norm(x)
        if self.non_linearity == "relu":
            x = F.relu(x)
        return x

class GSage(torch.nn.Module):
    def __init__(self, input_feature_dim, hidden_feature_dim,
                 loss_type, num_classes, L=3, batch_norm=True,
                 non_linearity="relu", use_bias=True):
        super().__init__()
        self.name = "gsage"
        self.non_linearity = non_linearity
        self.batch_norm = batch_norm
        self.loss_type = loss_type
        self.proj_layer = SAGEConv(input_feature_dim, hidden_feature_dim, bias=use_bias)
        self.conv_layers = [
            GSageSubModule(
                in_feature_dim=hidden_feature_dim,
                out_feature_dim=hidden_feature_dim,
                batch_norm=batch_norm,
                non_linearity=non_linearity,
                use_bias=use_bias
            )
            for _ in range(L)
        ]
        self.conv_layers = torch.nn.ModuleList(self.conv_layers)
        self.final_layer = SAGEConv(hidden_feature_dim, num_classes, bias=use_bias)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.proj_layer(x, edge_index)
        if self.non_linearity == "relu":
            x = F.relu(x)
        for conv_layer in self.conv_layers:
            x = conv_layer(x, edge_index)
        x = self.final_layer(x, edge_index)
        if self.loss_type == "mse":
            return x
        return F.log_softmax(x, dim=1)


class GSageInc(torch.nn.Module):
    def __init__(self, input_feature_dim, hidden_feature_dim,
                 loss_type, num_classes, L=3, batch_norm=True,
                 non_linearity="relu", use_bias=True):
        super().__init__()
        self.name = "gsage_inc"
        self.non_linearity = non_linearity
        self.batch_norm = batch_norm
        self.loss_type = loss_type
        self.proj_layer = SAGEConv(input_feature_dim, hidden_feature_dim, bias=use_bias)
        self.conv_layers = [
            GSageSubModule(
                in_feature_dim=hidden_feature_dim,
                out_feature_dim=hidden_feature_dim,
                batch_norm=batch_norm,
                non_linearity=non_linearity,
                bias=use_bias
            )
            for _ in range(L)
        ]
        self.conv_layers = torch.nn.ModuleList(self.conv_layers)
        self.final_layer = SAGEConv(hidden_feature_dim, num_classes, bias=use_bias)

    def forward(self, data, layer_idx):
        x, edge_index = data.x, data.edge_index
        if layer_idx > 0:
            self.proj_layer.requires_grad = False
        x = self.proj_layer(x, edge_index)
        if self.non_linearity == "relu":
            x = F.relu(x)
        for idx, conv_layer in enumerate(self.conv_layers):
            if idx < layer_idx:
                conv_layer.requires_grad = False
                x = conv_layer(x, edge_index)
            elif idx == layer_idx:
                x = conv_layer(x, edge_index)
            else:
                continue
        x = self.final_layer(x, edge_index)
        if self.loss_type == "mse":
            return x
        return F.log_softmax(x, dim=1)
