"""
Stochastic block model graphs
"""

import os
from enum import Enum
import math
import numpy as np
import torch
# torch.set_printoptions(profile="full")
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import networkx as nx

class FeatureStrategy(Enum):
    EMPTY = "empty"
    DEGREE = "degree"
    RANDOM = "random"
    RANDOM_NORMAL = "random_normal"
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
        permute_nodes: Permute the nodes to avoid an already clustered adjacency matrix
    """
    def __init__(self, N, C, Pr, p, q, num_graphs, feature_strategy="empty", feature_dim=0, permute_nodes=True, dataset_dir="", is_training=True):
        self.N = N
        self.C = C
        self.Pr = np.array(Pr)
        self.W = self.prepare_W(p=p, q=q)
        self.num_graphs = num_graphs
        self.feature_strategy = feature_strategy
        self.feature_dim = feature_dim
        self.permute_nodes = permute_nodes
        self.dataset_dir = dataset_dir
        self.is_training = is_training
        self.graphs_list = []
        self.validate()
        self.prepare_paths()
        self.load_data()

    def prepare_W(self, p, q):
        W = []
        for i in range(self.C):
            row = []
            for j in range(self.C):
                val = p if i==j else q
                row.append(val)
            W.append(row)
        return np.array(W)

    def validate(self):
        """Validate the parameters of the model"""
        if len(self.Pr) != self.C:
            raise ValueError("length of {} should be equal to {}".format(self.Pr, self.C))
        if np.sum(self.Pr) != 1.0:
            raise ValueError("Values of {} should sum to 1".format(self.Pr))
        if self.W.shape[0] != self.W.shape[1]:
            raise ValueError("{} should be symmetric".format(self.W))
        if not np.all(self.W == self.W.transpose()):
            raise ValueError("{} should be symmetric".format(self.W))
        if self.W.shape[0] != self.C:
            raise ValueError("Shape of {} should be ({}, {})".format(self.W, self.C, self.C))
        if not FeatureStrategy.present(self.feature_strategy):
            raise ValueError("Invalid feature_strategy={}. \
                Should be one of 'empty', 'degree', 'random', 'degree_random'".format(self.feature_strategy))
        if self.feature_strategy in ["random", "degree_random"] and self.feature_dim == 0:
            raise ValueError("feature_dim = 0 when random features were desired.")

    def prepare_paths(self):
        data_dir = "N_{}_C_{}_Pr_{}_p_{}_q_{}_num_graphs_{}_feat_strat_{}_feat_dim_{}_permute_{}".format(
            self.N, self.C, self.Pr, self.W[0,0], self.W[0,1], self.num_graphs, self.feature_strategy, self.feature_dim, self.permute_nodes
        )
        if self.is_training:
            dataset_dir = os.path.join(self.dataset_dir, "data/sbm/train")
        else:
            dataset_dir = os.path.join(self.dataset_dir, "data/sbm/test")
        self.dataset_path = os.path.join(dataset_dir, data_dir)

    def save_data(self):
        print("Saving data")
        os.makedirs(self.dataset_path)
        torch.save(self.graphs_list, self.dataset_path+"/data.pt")

    def load_data(self):
        if os.path.exists(self.dataset_path):
            print("Loading data from filesystem")
            self.graphs_list = torch.load(self.dataset_path+"/data.pt")
        else:
            print("Generating data")
            self.generate_data()
            self.save_data()

    def generate_single_graph(self):
        """Generate a single SBM graph"""
        M = torch.ones((self.N, self.N))
        comm_num_nodes = [math.floor(self.N * p_c) for p_c in self.Pr[:-1]]
        comm_num_nodes.append(self.N - np.sum(comm_num_nodes))
        row_offset = 0
        labels = []
        for comm_idx in range(self.C):
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
        Adj = torch.rand((self.N, self.N)) < M
        Adj = Adj.type(torch.int)
        # comment the following line to experiment with self-loop graphs.
        Adj = Adj * (torch.ones(self.N) - torch.eye(self.N))
        Adj = torch.maximum(Adj, Adj.t())
        X = self.get_features(Adj=Adj)

        # permute nodes and corresponding features, labels
        if self.permute_nodes:
            perm = torch.randperm(self.N)
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
            X = torch.rand((self.N, self.feature_dim))
            return X
        elif self.feature_strategy == "random_normal":
            X = torch.randn((self.N, self.feature_dim))
            return X
        elif self.feature_strategy == "degree_random":
            X = torch.zeros((self.N, self.feature_dim))
            X[:, 0] = torch.sum(Adj, 1)
            X[:, 1:] = torch.rand((self.N, self.feature_dim-1))
            return X

    def generate_data(self):
        print("Generating {} graphs".format(self.num_graphs))
        for _ in range(self.num_graphs):
            res = self.generate_single_graph()
            self.graphs_list.append(res)

    def __len__(self):
        """Return the number of graphs to be sampled"""
        return self.num_graphs

    def __getitem__(self, index):
        """Return a single sbm graph"""
        return self.graphs_list[index]

class SBMRegular(SBM):
    def __init__(self, N, C, Pr, p, q, num_graphs, feature_strategy="empty", feature_dim=0, permute_nodes=True, dataset_dir="", is_training=True):
        super().__init__(N, C, Pr, p, q, num_graphs, feature_strategy, feature_dim, permute_nodes, dataset_dir, is_training)

    def prepare_paths(self):
        data_dir = "N_{}_C_{}_Pr_{}_p_{}_q_{}_num_graphs_{}_feat_strat_{}_feat_dim_{}_permute_{}".format(
            self.N, self.C, self.Pr, self.W[0,0], self.W[0,1], self.num_graphs, self.feature_strategy, self.feature_dim, self.permute_nodes
        )
        if self.is_training:
            dataset_dir = os.path.join(self.dataset_dir, "data/sbm_reg/train")
        else:
            dataset_dir = os.path.join(self.dataset_dir, "data/sbm_reg/test")
        self.dataset_path = os.path.join(dataset_dir, data_dir)

    
    def generate_k_regular_bipartite(self, k, num_nodes):
        valid = False
        max_tries = 1000
        tries = 0
        while(not valid and tries < max_tries):
            A = np.zeros(shape=(num_nodes, num_nodes))
            second_half = np.arange(num_nodes // 2, num_nodes)
            for _edge in range(k):
                perm_half = np.random.permutation(second_half)
                for i in range(num_nodes//2):
                    A[i, perm_half[i]] = 1.0
                    A[perm_half[i], i] = 1.0
            degrees = np.sum(A, axis=1)
            if np.allclose(A, A.transpose()) and np.all(degrees == degrees[0]):
                valid = True
            else:
                tries += 1
        return A

    def generate_single_graph(self):
        """Generate a single SBM graph"""
        Adj = torch.zeros((self.N, self.N))

        comm_num_nodes = [math.floor(self.N * p_c) for p_c in self.Pr[:-1]]
        comm_num_nodes.append(self.N - np.sum(comm_num_nodes))
        row_offset = 0
        labels = []
        for comm_idx in range(self.C):
            curr_comm_num_nodes = comm_num_nodes[comm_idx]
            labels.extend([comm_idx for _ in range(curr_comm_num_nodes)])
            col_offset = 0
            for iter_idx, _num_nodes in enumerate(comm_num_nodes):
                num_selections = math.ceil(self.W[comm_idx, iter_idx]*self.N/self.C)
                print(comm_idx, iter_idx, row_offset, col_offset, curr_comm_num_nodes, _num_nodes, num_selections)

                if comm_idx == iter_idx:
                    G = nx.random_regular_graph(num_selections, _num_nodes)
                    nx_adj = nx.adjacency_matrix(G).todense()
                    Adj[row_offset:row_offset+curr_comm_num_nodes, col_offset:col_offset+_num_nodes] = torch.Tensor(nx_adj)
                elif comm_idx < iter_idx:
                    A_bp = self.generate_k_regular_bipartite(k=num_selections, num_nodes=2*_num_nodes)
                    A_ = torch.Tensor(A_bp[0:_num_nodes, _num_nodes:2*_num_nodes])
                    Adj[row_offset:row_offset+curr_comm_num_nodes, col_offset:col_offset+_num_nodes] = A_
                    Adj[col_offset:col_offset+_num_nodes, row_offset:row_offset+curr_comm_num_nodes] = \
                        Adj[row_offset:row_offset+curr_comm_num_nodes, col_offset:col_offset+_num_nodes].t().clone()
                degrees = torch.sum(Adj[row_offset:row_offset+curr_comm_num_nodes, col_offset:col_offset+_num_nodes], dim=1)
                col_offset += _num_nodes
            row_offset += curr_comm_num_nodes

        labels = torch.Tensor(labels).type(torch.int)
        Adj = Adj.type(torch.int)
        if not torch.allclose(Adj, Adj.transpose(0, 1)):
            raise ValueError("Error in preparing Adj matrix", Adj)
        degrees = torch.sum(Adj, dim=1)
        if not torch.all(degrees == degrees[0]):
            raise ValueError("The sub-graph is not regular", degrees)
        X = self.get_features(Adj=Adj)
        # permute nodes and corresponding features, labels
        if self.permute_nodes:
            perm = torch.randperm(self.N)
            labels = labels[perm]
            Adj = Adj[perm]
            Adj = Adj[:, perm]
            if self.feature_strategy != "empty":
                X = X[perm]

        indices = torch.nonzero(Adj)
        edge_index = indices.to(torch.long)
        data = Data(x=X, y=labels, edge_index=edge_index.t().contiguous())
        return data
