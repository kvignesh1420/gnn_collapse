import torch
import torch.nn as nn
from torch_geometric.nn.conv import GINConv, GPSConv
from torch_geometric.nn.attention import PerformerAttention

class FinalLayer(nn.Module):
    def __init__(self, in_dim, out_dim, use_bias, use_W1):
        super().__init__()
        self.lin_root = nn.Linear(in_dim, out_dim, bias=use_bias)
        self.lin_rel = nn.Linear(in_dim, out_dim, bias=use_bias)
        self.use_W1 = use_W1

    def forward(self, x):
        if self.use_W1:
            return self.lin_root(x) + self.lin_rel(x)
        else:
            return self.lin_rel(x)

class RedrawProjection:
    def __init__(self, conv_layers, redraw_interval=None):
        self.conv_layers = conv_layers
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attns = [m for m in self.conv_layers.modules() if isinstance(m, PerformerAttention)]
            for fa in fast_attns:
                fa.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1

class GraphTransformer(nn.Module):
    def __init__(self, input_feature_dim, hidden_feature_dim, loss_type, num_classes,
                 L, batch_norm, non_linearity, use_bias, use_W1,
                 attn_type="performer", attn_kwargs=None):
        super().__init__()
        self.name = "graphtrans"
        self.conv_layers = nn.ModuleList()
        self.non_linear_layers = nn.ModuleList()
        self.normalize_layers = nn.ModuleList()

        self.input_linear = nn.Linear(input_feature_dim, hidden_feature_dim, bias=use_bias)

        for _ in range(L):
            nn_conv = nn.Sequential(nn.Linear(hidden_feature_dim, hidden_feature_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_feature_dim, hidden_feature_dim))
            conv = GPSConv(
                hidden_feature_dim,
                GINConv(nn_conv),
                heads=8,
                attn_type=attn_type,
                attn_kwargs=attn_kwargs
            )
            self.conv_layers.append(conv)
            if non_linearity == "relu":
                self.non_linear_layers.append(nn.ReLU())
            else:
                self.non_linear_layers.append(nn.Identity())
            if batch_norm:
                self.normalize_layers.append(nn.BatchNorm1d(hidden_feature_dim))
            else:
                self.normalize_layers.append(nn.Identity())

        self.final_layer = FinalLayer(hidden_feature_dim, num_classes, use_bias, use_W1)
        self.redraw_projection = RedrawProjection(
            self.conv_layers,
            1000 if attn_type == "performer" else None
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.input_linear(x)
        for conv, nl, bn in zip(self.conv_layers, self.non_linear_layers, self.normalize_layers):
            x = conv(x, edge_index)
            x = nl(x)
            x = bn(x)
        return self.final_layer(x)