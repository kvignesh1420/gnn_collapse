"""
Model factory
"""

from gnn_collapse.models.baselines import BetheHessian
from gnn_collapse.models.mlp import MLP
from gnn_collapse.models.gcn import GCN
from gnn_collapse.models.sageconv import GSage

factory = {
    "bethe_hessian": BetheHessian,
    "mlp": MLP,
    "gcn": GCN,
    "gsage": GSage
}