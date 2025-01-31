# Make sure to add final layer

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import Linear
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.dense.linear import Linear
from gnn_collapse.models.common import Normalize
import torch.nn as nn

class GPSLayer(torch.nn.Module):
    """
    Graph Positional and Structural Layer that combines local GNN with transformer for global attention
    """
    def __init__(self, in_channels, out_channels, use_W1, bias=False, aggr='mean', heads=4, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_W1 = use_W1

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
            
        # Local MPNN
        self.local_model = GCNConv(in_channels[0], in_channels[1], add_self_loops=False)
        
        # Global attention - using TransformerConv for attention mechanism
        self.global_model = TransformerConv(
            in_channels[0], 
            in_channels[1] // heads, 
            heads=heads, 
            concat=True,
            beta=True  # Enable edge attention
        )
        
        # Layer norm for stability
        self.norm1 = nn.LayerNorm(in_channels[0])
        self.norm2 = nn.LayerNorm(in_channels[0])

        self.lin_rel = Linear(in_channels[0], 2 * in_channels[1], bias=bias)
        self.lin_root = Linear(2 * in_channels[1], in_channels[0], bias=bias)
        
        # FFN
        self.non_linear_layers = nn.ReLU()
        self.ffn = nn.Sequential(
            self.lin_rel,
            self.non_linear_layers,
            self.lin_root
        )
        
        # Parameters for combining local and global
        self.local_weight = nn.Parameter(torch.ones(1))
        self.global_weight = nn.Parameter(torch.ones(1))

        self.reset_parameters()

    def reset_parameters(self):
        # super().reset_parameters()
        self.lin_rel.reset_parameters()
        self.lin_root.reset_parameters()
    
    def forward(self, x, edge_index):
        # Ensure proper data types
        x = x.float()
        edge_index = edge_index.long()
        
        # Local MPNN
        local_out = self.local_model(x, edge_index)
        
        # Global attention
        global_out = self.global_model(x, edge_index)
        
        # Combine local and global (adaptive weights)
        x = self.local_weight * local_out + self.global_weight * global_out
        
        # First normalization and residual
        x = self.norm1(x + x)
        
        # FFN and second normalization
        out = self.ffn(x)
        out = self.norm2(out + x)
        
        return out

class GPSModel(torch.nn.Module):
    """GPS model for node classification"""
    def __init__(self, input_feature_dim, hidden_feature_dim,
                 loss_type, num_classes, L=3, batch_norm=True,
                 non_linearity="relu", use_W1=True, use_bias=False):
        super().__init__()
        self.name = "gps"
        self.L = L
        self.non_linearity = non_linearity
        self.loss_type = loss_type
        self.batch_norm = batch_norm
        self.norm = Normalize(hidden_feature_dim, norm="batch")
        
        # Initial projection
        self.proj_layer = Linear(input_feature_dim, hidden_feature_dim)
        if self.batch_norm:
            self.input_bn = nn.BatchNorm1d(hidden_feature_dim)
        
        # GPS layers
        self.conv_layers = nn.ModuleList()
        self.non_linear_layers = nn.ModuleList()
        self.normalize_layers = nn.ModuleList()
        for _ in range(L):
            gps_layer = GPSLayer(hidden_feature_dim, hidden_feature_dim, use_W1=use_W1, bias=use_bias)
            self.conv_layers.append(gps_layer)
            self.non_linear_layers.append(gps_layer.non_linear_layers)
            self.normalize_layers.append(gps_layer.norm2)
            if self.batch_norm:
                self.conv_layers.append(nn.BatchNorm1d(hidden_feature_dim))
        
        # Output projection
        self.output_proj = Linear(hidden_feature_dim, num_classes)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Initial projection
        x = self.proj_layer(x.float())
        if self.batch_norm:
            x = self.input_bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # GPS layers
        for l in len(self.conv_layers):
            x = self.conv_layers[l](x, edge_index)
            if self.batch_norm:
                self.normalize_layers[l][x]
                x = F.relu(x)
                x = self.dropout(x)
        
        # Output projection
        x = self.output_proj(x)
        
        return F.log_softmax(x, dim=1)