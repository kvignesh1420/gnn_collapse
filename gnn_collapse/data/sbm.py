"""
Stochastic block model graphs
"""

from enum import Enum
import math
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset


class FeatureStrategy(Enum):
    EMPTY = "empty"
    DEGREE = "degree"
    RANDOM = "random"
    DEGREE_RANDOM = "degree_random"

    @classmethod
    def present(cls, val):
        return val in [mem.value for mem in cls.__members__.values()]


class SBM(Dataset):
    """A pytorch dataset of SBM graphs

    Args:
        root: root directory for saving graphs
        n: Number of nodes in the graph.
        k: Number of clusters in the graph.
        p: Probability vector on [k] = {1,2,..,k}, i.e the prior on k-communities.
        W: k x k symmetric matrix of connection probabilities among clusters.
        num_graphs: Number of graphs to generate
        feature_strategy: The strategy to generate features for nodes.
            Can be one of 'empty', 'degree', 'random', 'degree_random'
        feature_dim: the desired dimension of the features. This is applicable
            only for 'random' and 'degree_random' strategies.
    """
    def __init__(self, n, k, p, W, num_graphs, feature_strategy="empty", feature_dim=0):
        self.n = n
        self.k = k
        self.p = np.array(p)
        self.W = np.array(W)
        self.num_graphs = num_graphs
        self.feature_strategy = feature_strategy
        self.feature_dim = feature_dim
        self.validate()

    def validate(self):
        """Validate the parameters of the model"""
        if len(self.p) != self.k:
            raise ValueError("length of {} should be equal to {}".format(self.p, self.k))
        if np.sum(self.p) != 1.0:
            raise ValueError("Values of {} should sum to 1".format(self.p))
        if self.W.shape[0] != self.W.shape[1]:
            raise ValueError("{} should be symmetric".format(self.W))
        if not np.all(self.W == self.W.transpose()):
            raise ValueError("{} should be symmetric".format(self.W))
        if self.W.shape[0] != self.k:
            raise ValueError("Shape of {} should be ({}, {})".format(self.W, self.k, self.k))
        if not FeatureStrategy.present(self.feature_strategy):
            raise ValueError("Invalid feature_strategy={}. \
                Should be one of 'empty', 'degree', 'random', 'degree_random'".format(self.feature_strategy))
        if self.feature_strategy in ["random", "degree_random"] and self.feature_dim == 0:
            raise ValueError("feature_dim = 0 when random features were desired.")

    def generate_single_graph(self):
        """Generate a single SBM graph"""
        M = torch.ones((self.n, self.n))
        comm_num_nodes = [math.floor(self.n * p_c) for p_c in self.p[:-1]]
        comm_num_nodes.append(self.n - np.sum(comm_num_nodes))
        row_offset = 0
        labels = []
        for comm_idx in range(self.k):
            curr_comm_num_nodes = comm_num_nodes[comm_idx]
            labels.extend([comm_idx for _ in range(curr_comm_num_nodes)])
            col_offset = 0
            for iter_idx, _num_nodes in enumerate(comm_num_nodes):
                M[row_offset:row_offset+curr_comm_num_nodes, col_offset:col_offset+_num_nodes] = self.W[comm_idx, iter_idx]
                col_offset += _num_nodes
            row_offset += curr_comm_num_nodes

        if not torch.allclose(M, M.t()):
            raise ValueError("Error in preparing X matrix", M)

        labels = torch.Tensor(labels).type(torch.int)
        Adj = torch.rand((self.n, self.n)) < M
        Adj = Adj.type(torch.int)
        Adj = Adj * (torch.ones(self.n) - torch.eye(self.n))
        Adj = torch.maximum(Adj, Adj.t())
        X = self.get_features(Adj=Adj)

        # permute nodes and corresponding features, labels
        perm = torch.randperm(self.n)
        labels = labels[perm]
        Adj = Adj[perm]
        Adj = Adj[:, perm]
        if self.feature_strategy != "empty":
            X = X[perm]

        indices = torch.nonzero(Adj)
        edge_index = indices.to(torch.long)
        data = Data(x=X, y=labels, edge_index=edge_index.t().contiguous())
        return data

    def get_features(self, Adj):
        """Prepare the features for the nodes based on
        feature_strategy and adjacency matrix

        Args:
            Adj: Adjacency matrix of the graph

        Returns:
            Feature tensor X of shape (n, d) if d > 0 else
            returns an empty tensor
        """
        if self.feature_strategy == "empty":
            return torch.Tensor(())
        elif self.feature_strategy == "degree":
            X = torch.sum(Adj, 1).unsqueeze(1)
            return X
        elif self.feature_strategy == "random":
            X = torch.rand((self.n, self.feature_dim))
            return X
        elif self.feature_strategy == "degree_random":
            X = torch.zeros((self.n, self.feature_dim))
            X[:, 0] = torch.sum(Adj, 1)
            X[:, 1:] = torch.rand((self.n, self.feature_dim-1))
            return X

    def __len__(self):
        """Return the number of graphs to be sampled"""
        return self.num_graphs

    def __getitem__(self, index):
        """Return a single sbm graph"""
        return self.generate_single_graph()
