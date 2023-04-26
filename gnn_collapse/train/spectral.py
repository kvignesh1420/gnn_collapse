"""
Spectral clustering
"""

import numpy as np
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm
from gnn_collapse.utils.losses import compute_accuracy_multiclass
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 40,
    'lines.linewidth': 5,
    'axes.titlepad': 20,
    'axes.linewidth': 2,
    'figure.figsize': (15, 15)
})

from gnn_collapse.utils.node_properties import compute_nc1
from gnn_collapse.utils.node_properties import plot_test_graphs_nc1


def spectral_clustering(model_class, dataloader, args):
    """clustering based on spectral methods for sbm node classification

    Args:
        model_class: one of BetheHessian or NormalizedLaplacian classes
        dataloader: The dataloader of SBM graphs
        args: settings for training
    """

    model = model_class(args=args)
    accuracies = []
    nc1_snapshots = []
    for step_idx, data in tqdm(enumerate(dataloader)):
        device = args["device"]
        data = data.to(device)
        Adj = to_dense_adj(data.edge_index)[0]

        pred, features = model.pi_fiedler_pred(A=Adj, labels=data.y)
        acc = compute_accuracy_multiclass(pred=pred, labels=data.y, C=args["C"])
        accuracies.append(acc)
        if args["track_nc"]:
            # print("index: {} acc: {}".format(step_idx, acc))
            nc1_snapshots.append(compute_nc1(features=features, labels=data.y))
    
    plot_test_graphs_nc1(features_nc1_snapshots=nc1_snapshots, non_linear_features_nc1_snapshots=[],
                          normalized_features_nc1_snapshots=[], weight_sv_info=[], args=args, epoch=1)

    print('Avg test acc', np.mean(accuracies))
    print('Std test acc', np.std(accuracies))
    with open(args["results_file"], 'a') as f:
        f.write("""Avg test acc: {}\n Std test acc: {}\n""".format(
            np.mean(accuracies), np.std(accuracies)))
    plt.grid()
    plt.plot(accuracies)
    plt.savefig("{}test_acc.png".format(args["vis_dir"]))
    plt.clf()