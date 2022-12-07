"""
Spectral method baselines
"""

import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

class BetheHessian:
    def __init__(self, Adj):
        self.A = Adj

    def compute(self, r=None):
        degrees = torch.sum(self.A, dim=1)
        if r is None:
            r = torch.sqrt(torch.mean(degrees))
        self.BH = (r**2 - 1)*torch.eye(degrees.shape[0]) - r*self.A + torch.diag(degrees)
        return self.BH

    def cluster(self, k):
        evals, evecs = torch.linalg.eig(self.BH)
        evals = evals.type(torch.float)
        evecs = evecs.type(torch.float)
        sorted_eval_indices = torch.argsort(evals)
        # consider the k smallest evals as BH is a variant of the graph laplacian
        k_eval_indices = sorted_eval_indices[:k]
        X = evecs[:, k_eval_indices]
        # Cluster the features to get the labels
        cluster_model = KMeans(n_clusters=k, random_state=0)
        cluster_model.fit(X=X)
        pred = cluster_model.labels_
        cluster_means = cluster_model.cluster_centers_
        return {
            "pred": F.one_hot(torch.from_numpy(pred).type(torch.int64), num_classes=k),
            "centroids": torch.from_numpy(cluster_means)
        }

    def pi_fiedler_pred(self, num_iters):
        # Use this only for 2 community classification

        evals, evecs = torch.linalg.eig(self.BH)
        evals = evals.type(torch.float)
        evecs = evecs.type(torch.float)
        sorted_eval_indices = torch.argsort(evals)
        # consider the smallest eigen vector of self.BH
        k_eval_indices = sorted_eval_indices[0]
        v = evecs[:, k_eval_indices]

        BH_hat = torch.norm(self.BH, p=2)*torch.eye(self.BH.shape[0]) - self.BH
        w = torch.ones_like(v)
        for i in range(num_iters):
            y = BH_hat @ w
            w = y - torch.dot(y, v)*v
            w = w/torch.norm(w, p=2)

        # print(y, evecs[:, sorted_eval_indices[1]])
        pred = (y < 0).type(torch.int64)
        # print(pred)
        return F.one_hot(pred, num_classes=2)