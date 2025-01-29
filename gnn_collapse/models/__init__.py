"""
Model factory
"""

from gnn_collapse.models.baselines import BetheHessian
from gnn_collapse.models.baselines import NormalizedLaplacian
from gnn_collapse.models.gps import GPSModel
from gnn_collapse.models.graph_transformer import GraphTransformer
from gnn_collapse.models.graphconv import GraphConvModel

Spectral_factory = {
    "bethe_hessian": BetheHessian,
    "normalized_laplacian": NormalizedLaplacian,
}

GNN_factory = {
    # A factory to support additional model designs in the future!
    "graphtrans": GraphTransformer,
    "graphconv": GraphConvModel,
    "gps": GPSModel,
}