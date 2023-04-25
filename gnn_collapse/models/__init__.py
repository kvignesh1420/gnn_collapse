"""
Model factory
"""

from gnn_collapse.models.baselines import BetheHessian
from gnn_collapse.models.baselines import NormalizedLaplacian
from gnn_collapse.models.mlp import MLP
from gnn_collapse.models.gcn import GCN
from gnn_collapse.models.sageconv import GSage
from gnn_collapse.models.sageconv import GSageV2
from gnn_collapse.models.sageconv import GSageInc
from gnn_collapse.models.graphconv import GraphConvModel

factory = {
    "bethe_hessian": BetheHessian,
    "normalized_laplacian": NormalizedLaplacian,
    # "mlp": MLP,
    # "gcn": GCN,
    # "gsage": GSageV2,
    # "gsage_inc": GSageInc,
    "graphconv": GraphConvModel
}