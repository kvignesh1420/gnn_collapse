import argparse
import os.path as osp
from typing import Any, Dict, Optional

import torch
from torch.nn import (
    BatchNorm1d,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
    CrossEntropyLoss
)
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GINConv, GPSConv
from torch_geometric.nn.attention import PerformerAttention

# Load Cora dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0].to(device)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--attn_type', default='multihead',
    help="Global attention type such as 'multihead' or 'performer'.")
args = parser.parse_args()

class GPS(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int,
                 num_layers: int, attn_type: str, attn_kwargs: Dict[str, Any]):
        super().__init__()
        
        self.lin1 = Linear(in_channels, hidden_channels)
        
        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
            )
            conv = GPSConv(hidden_channels, GINConv(nn), heads=8,
                          attn_type=attn_type, attn_kwargs=attn_kwargs)
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            ReLU(),
            Linear(hidden_channels // 2, num_classes),
        )
        
        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None)

    def forward(self, x, edge_index):
        x = self.lin1(x)
        
        for conv in self.convs:
            x = conv(x, edge_index)
        
        return self.mlp(x)


class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                 redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1


# Model initialization
attn_kwargs = {'dropout': 0.3}
model = GPS(
    in_channels=dataset.num_features,
    hidden_channels=64,
    num_classes=dataset.num_classes,
    num_layers=4,
    attn_type=args.attn_type,
    attn_kwargs=attn_kwargs
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    
    loss.backward()
    optimizer.step()
    model.redraw_projection.redraw_projections()
    
    return loss.item()

@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    
    pred = out.argmax(dim=-1)
    
    train_correct = pred[data.train_mask] == data.y[data.train_mask]
    val_correct = pred[data.val_mask] == data.y[data.val_mask]
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    
    train_acc = train_correct.sum().item() / data.train_mask.sum().item()
    val_acc = val_correct.sum().item() / data.val_mask.sum().item()
    test_acc = test_correct.sum().item() / data.test_mask.sum().item()
    
    return train_acc, val_acc, test_acc

best_val_acc = 0
for epoch in range(1, 201):
    loss = train()
    train_acc, val_acc, test_acc = test()
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc
    
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
          f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')