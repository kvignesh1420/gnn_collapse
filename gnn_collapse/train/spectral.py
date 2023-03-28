"""
Spectral clustering
"""

import numpy as np
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm
from gnn_collapse.utils.losses import compute_accuracy_multiclass
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 25, 'lines.linewidth': 5, 'axes.titlepad': 20, "figure.figsize": (15, 15)})


def spectral_clustering(model_class, dataloader, args):
    """clustering based on spectral methods for sbm node classification

    Args:
        model_class: one of BetheHessian or NormalizedLaplacian classes
        dataloader: The dataloader of SBM graphs
        args: settings for training
    """

    accuracies = []
    for step_idx, data in tqdm(enumerate(dataloader)):
        device = args["device"]
        data = data.to(device)
        Adj = to_dense_adj(data.edge_index)[0]
        model = model_class(Adj=Adj)
        model.compute()
        enable_tracking = args["track_nc"] and step_idx==0
        pred = model.pi_fiedler_pred(labels=data.y, args=args, enable_tracking=enable_tracking)
        acc = compute_accuracy_multiclass(pred=pred, labels=data.y, C=args["C"])
        accuracies.append(acc)
        if enable_tracking:
            print("index: {} acc: {}".format(step_idx, acc))

    print('Avg test acc', np.mean(accuracies))
    print('Std test acc', np.std(accuracies))
    with open(args["results_file"], 'a') as f:
        f.write("""Avg test acc: {}\n Std test acc: {}\n""".format(
            np.mean(accuracies), np.std(accuracies)))
    plt.plot(accuracies)
    plt.savefig("{}test_acc.png".format(args["vis_dir"]))
    plt.clf()