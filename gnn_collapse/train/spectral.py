"""
Spectral clustering
"""

import numpy as np
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm
from gnn_collapse.models.baselines import BetheHessian
from gnn_collapse.utils.losses import compute_accuracy_multiclass
import matplotlib.pyplot as plt


def spectral_clustering(dataloader, args):
    """clustering based on spectral methods for sbm node classification

    Args:
        dataloader: The dataloader of SBM graphs
        args: settings for training
    """

    accuracies = []
    for step_idx, data in tqdm(enumerate(dataloader)):
        device = args["device"]
        data = data.to(device)
        Adj = to_dense_adj(data.edge_index)[0]
        model = BetheHessian(Adj=Adj)
        model.compute()
        pred = model.pi_fiedler_pred(num_iters=args["num_layers"])
        acc = compute_accuracy_multiclass(pred=pred, labels=data.y, k=args["k"])
        accuracies.append(acc)

    print('Avg test acc', np.mean(accuracies))
    print('Std test acc', np.std(accuracies))
    with open(args["results_file"], 'a') as f:
        f.write("""Avg test acc: {}\n Std test acc: {}\n""".format(
            np.mean(accuracies), np.std(accuracies)))
    plt.plot(accuracies)
    plt.savefig("{}test_acc.png".format(args["vis_dir"]))
    plt.clf()