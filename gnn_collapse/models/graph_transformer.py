from typing import Union, Optional
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
)
from torch_geometric.utils import softmax as geo_softmax


class GraphTransformerConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.0,
        use_W1: bool = True,
        bias: bool = False,
        **kwargs
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.use_W1 = use_W1
        # Modified scale factor with clipping
        self.scale = min((out_channels / heads) ** -0.5, 5.0)

        # W2 weights for Q, K, V transformations
        self.lin_rel_q = Linear(in_channels, out_channels, bias=bias)
        self.lin_rel_k = Linear(in_channels, out_channels, bias=bias)
        self.lin_rel_v = Linear(in_channels, out_channels, bias=bias)

        if self.use_W1:
            self.lin_root = Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize with smaller values to prevent explosion
        for module in [self.lin_rel_q, self.lin_rel_k, self.lin_rel_v]:
            torch.nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
        if self.use_W1:
            torch.nn.init.xavier_uniform_(self.lin_root.weight, gain=0.1)
            if self.lin_root.bias is not None:
                torch.nn.init.zeros_(self.lin_root.bias)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_weight: OptTensor = None,
        size: Size = None,
    ) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)

        # Compute Q, K, V transformations
        query = self.lin_rel_q(x[1]).view(-1, self.heads, self.out_channels // self.heads)
        key = self.lin_rel_k(x[0]).view(-1, self.heads, self.out_channels // self.heads)
        value = self.lin_rel_v(x[0]).view(-1, self.heads, self.out_channels // self.heads)

        # Clip values for stability
        query = torch.clamp(query, -50, 50)
        key = torch.clamp(key, -50, 50)
        value = torch.clamp(value, -50, 50)

        # Propagate attention scores and values
        out = self.propagate(
            edge_index,
            query=query,
            key=key,
            value=value,
            edge_weight=edge_weight,
            size=size,
        )

        # Reshape output and handle any NaN values
        out = out.view(-1, self.out_channels)
        out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)

        # Add self-connection if use_W1 is True
        if self.use_W1 and x[1] is not None:
            root = self.lin_root(x[1])
            root = torch.clamp(root, -50, 50)
            out = out + root

        return out

    def message(
        self,
        query_i: Tensor,
        key_j: Tensor,
        value_j: Tensor,
        edge_weight: OptTensor,
        index: Tensor,
        ptr: Optional[Tensor],
        size_i: Optional[int],
    ) -> Tensor:
        # Compute attention scores with numerical stability measures
        alpha = (query_i * key_j).sum(dim=-1) * self.scale
        alpha = torch.clamp(alpha, -50, 50)
        
        if edge_weight is not None:
            edge_weight = torch.clamp(edge_weight, -50, 50)
            alpha = alpha * edge_weight.view(-1, 1)
        
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j * alpha.unsqueeze(-1)
        return torch.clamp(out, -50, 50)


class GraphTransformer(torch.nn.Module):
    def __init__(
        self,
        input_feature_dim: int,
        hidden_feature_dim: int,
        loss_type: str,
        num_classes: int,
        L: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        batch_norm: bool = True,
        non_linearity: str = "relu",
        use_W1: bool = True,
        use_bias: bool = False,
    ):
        super().__init__()
        self.name = "graphtrans"
        self.L = L
        self.non_linearity = non_linearity
        self.loss_type = loss_type
        self.batch_norm = batch_norm
        
        # Layer normalization for improved stability
        self.layer_norm = torch.nn.LayerNorm(hidden_feature_dim)
        
        if self.batch_norm:
            self.norm = torch.nn.BatchNorm1d(hidden_feature_dim)
        
        # Initial projection layer
        self.proj_layer = GraphTransformerConv(
            input_feature_dim,
            hidden_feature_dim,
            heads=heads,
            dropout=dropout,
            use_W1=use_W1,
            bias=use_bias,
        )
        
        # Hidden transformer layers
        self.conv_layers = torch.nn.ModuleList([
            GraphTransformerConv(
                hidden_feature_dim,
                hidden_feature_dim,
                heads=heads,
                dropout=dropout,
                use_W1=use_W1,
                bias=use_bias,
            )
            for _ in range(L)
        ])
        
        # Layer normalization for each transformer layer
        self.layer_norms = torch.nn.ModuleList([
            torch.nn.LayerNorm(hidden_feature_dim) for _ in range(L)
        ])
        
        if self.non_linearity == "relu":
            self.non_linear_layers = torch.nn.ModuleList([
                torch.nn.ReLU() for _ in range(L)
            ])
        else:
            self.non_linear_layers = torch.nn.ModuleList([])
            
        if self.batch_norm:
            self.normalize_layers = torch.nn.ModuleList([
                torch.nn.BatchNorm1d(hidden_feature_dim) for _ in range(L)
            ])
        else:
            self.normalize_layers = torch.nn.ModuleList([])
            
        # Final classification layer
        self.final_layer = GraphTransformerConv(
            hidden_feature_dim,
            num_classes,
            heads=1,
            dropout=dropout,
            use_W1=use_W1,
            bias=use_bias,
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Initial projection with stability measures
        x = self.proj_layer(x, edge_index)
        x = self.layer_norm(x)
        
        if self.non_linearity == "relu":
            x = F.relu(x)
        if self.batch_norm:
            x = self.norm(x)
            
        # Hidden layers with residual connections and normalization
        for l in range(self.L):
            identity = x
            x = self.conv_layers[l](x, edge_index)
            x = self.layer_norms[l](x)
            
            if self.non_linearity == "relu":
                x = self.non_linear_layers[l](x)
            if self.batch_norm:
                x = self.normalize_layers[l](x)
            
            # Residual connection
            x = x + identity
            x = torch.clamp(x, -50, 50)
                
        # Final layer
        x = self.final_layer(x, edge_index)
        
        # Handle output based on loss type
        if self.loss_type == "mse":
            return x
        
        # Add numerical stability to log_softmax
        x = torch.clamp(x, -50, 50)
        return F.log_softmax(x, dim=1)


def softmax(src: Tensor, index: Tensor, ptr: Optional[Tensor] = None,
            num_nodes: Optional[int] = None) -> Tensor:
    # Add numerical stability to softmax
    src = torch.clamp(src, -50, 50)
    if ptr is not None:
        return F.softmax(src, dim=0)
    else:
        return geo_softmax(src, index, num_nodes=num_nodes)