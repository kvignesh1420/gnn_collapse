from typing import Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)
from gnn_collapse.models.common import Normalize
from torch import spmm

DEFAULT_HEADS=3

class EasyGTModel(torch.nn.Module):
    def __init__(self, input_feature_dim, hidden_feature_dim,
                 loss_type, num_classes, L=3, batch_norm=True,
                 non_linearity="relu", use_bias=False, heads=DEFAULT_HEADS, use_W1=False):
        # use_w1 is a dummy parameter, not used at all by this model but included for easy compatibility with online.py
        super().__init__()
        self.name = "easy_gt" # String name assignment
        self.L = L # Number of Layers
        self.non_linearity = non_linearity # Type of nonlinearity
        self.loss_type = loss_type # Loss function
        self.batch_norm = batch_norm # Boolean - yes/no to using batch norm
        self.norm = Normalize(hidden_feature_dim*heads, norm="batch") # Batch norm itself
        self.proj_layer = TransformerConv(input_feature_dim, hidden_feature_dim, heads, bias=use_bias) # Projection from
                                                                                    # Input dimensionality to latent space dim.
        self.conv_layers = [
            TransformerConv(hidden_feature_dim*heads, hidden_feature_dim, heads, bias=use_bias)
            for _ in range(L)
        ] # Actual network layers

        if self.non_linearity == "relu":
            self.non_linear_layers = [torch.nn.ReLU()  for _ in range(L)] # All nonlinearities
        else:
            self.non_linear_layers = []
        if self.batch_norm:
            self.normalize_layers = [Normalize(hidden_feature_dim*heads, norm="batch")  for _ in range(L)] # All batch norms
        else:
            self.normalize_layers = []

        # Making graph "convolutions" (transformer blocks in our case), relus, batch norms, packaged and accessible
        self.conv_layers = torch.nn.ModuleList(self.conv_layers)
        self.non_linear_layers = torch.nn.ModuleList(self.non_linear_layers)
        self.normalize_layers = torch.nn.ModuleList(self.normalize_layers)
        self.final_layer = TransformerConv(hidden_feature_dim*heads, num_classes, heads=1, bias=use_bias) # Projection from latent space
                                                                                                # dim. to # of output classes

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.proj_layer(x, edge_index)
        if self.non_linearity == "relu":
            x = F.relu(x)
        if self.batch_norm:
            x = self.norm(x)
        for l in range(self.L):
            x = self.conv_layers[l](x, edge_index)
            if self.non_linearity == "relu":
                x = self.non_linear_layers[l](x)
            if self.batch_norm:
                x = self.normalize_layers[l](x)
        x = self.final_layer(x, edge_index)
        if self.loss_type == "mse":
            return x
        return F.log_softmax(x, dim=1)
