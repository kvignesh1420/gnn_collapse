"""
Model factory
"""

from gnn_collapse.models.baselines import BetheHessian
from gnn_collapse.models.baselines import NormalizedLaplacian
from gnn_collapse.models.graphconv import GraphConvModel

Spectral_factory = {
    "bethe_hessian": BetheHessian,
    "normalized_laplacian": NormalizedLaplacian,
}

GNN_factory = {
    "graphconv": GraphConvModel
}