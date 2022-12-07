"""
Analyse the node properties via collapse metrics
"""
from collections import defaultdict
import numpy as np
import torch
from torch_scatter import scatter
import matplotlib.pyplot as plt
import seaborn as sns

def plot_penultimate_layer_features(features, labels, model_name):
    """
    Plot the penultimate layer features of the model by
    fixind the output dim to 2 in 2 class classification setting
    """
    num_layers = len(features)
    last_hidden_layer = num_layers-1
    feat = features[last_hidden_layer].cpu().numpy()
    # print(feat)
    colors = ["blue" if i == 0 else "orange" for i in labels]
    ax = plt.scatter(x=feat[:,0], y = feat[:,1], c=colors)
    fig = ax.get_figure()
    fig.savefig("plots/featplot_{}.png".format(model_name))
    plt.clf()


def spectral_matching(Adj, features, labels):
    """
    compute cosine of angle between the penultimate layer features
    and the fiedler vector of Adj.
    """
    num_layers = len(features)
    last_hidden_layer = num_layers-1
    feat = features[last_hidden_layer]

    evals, evecs = torch.linalg.eig(Adj)
    evals = evals.type(torch.float)
    evecs = evecs.type(torch.float)
    sorted_eval_indices = torch.argsort(evals)
    # consider the eigen vector of second largest eigen value
    k_eval_indices = sorted_eval_indices[-2]
    fiedler = evecs[:, k_eval_indices]
    angles = []
    for i in range(feat.shape[1]):
        fiedler_ = fiedler/torch.norm(fiedler, p=2)
        feat_ = feat[:, i]/torch.norm(feat[:, i], p=2)
        angles.append(torch.dot(fiedler_, feat_))
    print(angles)


def compute_nc1(features, labels):
    """Compute the variability collapse metric from
    the list of node features across layers and time.
    """

    collapse_metrics = {}
    for layer_name, feat in features.items():
        class_means = scatter(feat, labels.type(torch.int64), dim=0, reduce="mean")
        expanded_class_means = torch.index_select(class_means, dim=0, index=labels)
        z = feat - expanded_class_means
        num_nodes = z.shape[0]
        S_W = 0
        for i in range(num_nodes):
            S_W += z[i, :].unsqueeze(1) @ z[i, :].unsqueeze(0)
        S_W /= num_nodes

        global_mean = torch.mean(class_means, dim=0)
        z = class_means - global_mean
        num_classes = class_means.shape[0]
        S_B = 0
        for i in range(num_classes):
            S_B += z[i, :].unsqueeze(1) @ z[i, :].unsqueeze(0)
        S_B /= num_classes

        collapse_metric = torch.trace(S_W @ torch.linalg.pinv(S_B)) / num_classes
        collapse_metrics[layer_name] = collapse_metric.detach().cpu().numpy()
    return collapse_metrics

def plot_nc1(nc1_snapshots, model_name):
    layers_nc1 = defaultdict(list)
    for snapshot in nc1_snapshots:
        for layer_name, collapse_metric in snapshot.items():
            layers_nc1[layer_name].append(collapse_metric)

    heatmap_data = []
    heatmap_labels = []
    for layer_name, collapse_metric_trend in layers_nc1.items():
        heatmap_data.append(collapse_metric_trend)
        heatmap_labels.append(layer_name)

    # clip the NC1 metric value at 10 for better color ranges in visualization
    heatmap_data = np.log10(np.array(heatmap_data)[::-1])
    # print(heatmap_data)
    fig, ax = plt.subplots(figsize=(20, 20))
    ax = sns.heatmap(heatmap_data, cmap="crest")
    _ = ax.set(xlabel="epoch/20", ylabel="depth")
    ax.set_xticklabels(ax.get_xticks(), rotation=90)
    ax.set_yticklabels(labels=heatmap_labels[::-1], rotation=0)
    fig = ax.get_figure()
    fig.savefig("plots/nc1_{}.png".format(model_name))
    plt.clf()
