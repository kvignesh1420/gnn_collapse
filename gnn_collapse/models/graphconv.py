from typing import Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)
from gnn_collapse.models.common import Normalize
from torch import spmm


class GraphConv(MessagePassing):
    r"""The graph neural network operator from the `"Weisfeiler and Leman Go
    Neural: Higher-order Graph Neural Networks"
    <https://arxiv.org/abs/1810.02244>`_ paper

    The implementation is based on the `GraphConv` layer in pyg standard library
    """
    def __init__(self, in_channels, out_channels, use_W1, bias=False, aggr='mean', **kwargs):

        super().__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_W1 = use_W1

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        # W2 weights
        self.lin_rel = Linear(in_channels[0], out_channels, bias=bias)

        # W1 weights
        if self.use_W1:
            self.lin_root = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        # super().reset_parameters()
        self.lin_rel.reset_parameters()
        if self.use_W1:
            self.lin_root.reset_parameters()


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, size: Size = None) -> Tensor:

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=size)
        out = self.lin_rel(out)

        if self.use_W1:
            x_r = x[1]
            if x_r is not None:
                out = out + self.lin_root(x_r)

        return out


    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        return spmm(adj_t, x[0], reduce=self.aggr)


class GraphConvModel(torch.nn.Module):
    def __init__(self, input_feature_dim, hidden_feature_dim,
                 loss_type, num_classes, L=3, batch_norm=True,
                 non_linearity="relu", use_W1=True, use_bias=False):
        super().__init__()
        self.name = "graphconv"
        self.non_linearity = non_linearity
        self.loss_type = loss_type
        self.batch_norm = batch_norm
        self.norm = Normalize(hidden_feature_dim, norm="batch")
        self.proj_layer = GraphConv(input_feature_dim, hidden_feature_dim, use_W1=use_W1, bias=use_bias)
        self.conv_layers = [
            GraphConv(hidden_feature_dim, hidden_feature_dim, use_W1=use_W1, bias=use_bias)
            for _ in range(L)
        ]
        self.non_linear_layers = [
            torch.nn.ReLU() if self.non_linearity == "relu" else torch.nn.Identity() for _ in range(L)
        ]
        self.normalize_layers = [
            Normalize(hidden_feature_dim, norm="batch")
            if self.batch_norm else torch.nn.Identity() for _ in range(L)
        ]
        self.conv_layers = torch.nn.ModuleList(self.conv_layers)
        self.non_linear_layers = torch.nn.ModuleList(self.non_linear_layers)
        self.normalize_layers = torch.nn.ModuleList(self.normalize_layers)
        self.final_layer = GraphConv(hidden_feature_dim, num_classes, use_W1=use_W1, bias=use_bias)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.proj_layer(x, edge_index)
        if self.non_linearity == "relu":
            x = F.relu(x)
        if self.batch_norm:
            x = self.norm(x)
        for conv_layer, non_linear_layer, normalize_layer in zip(
            self.conv_layers, self.non_linear_layers, self.normalize_layers):
            x = conv_layer(x, edge_index)
            x = non_linear_layer(x)
            x = normalize_layer(x)
        x = self.final_layer(x, edge_index)
        if self.loss_type == "mse":
            return x
        return F.log_softmax(x, dim=1)
