"""
Analyse the node properties via collapse metrics
"""
from collections import defaultdict
import numpy as np
import torch
from torch_scatter import scatter
import matplotlib.pyplot as plt
import seaborn as sns


def compute_nc1(features, labels):
    """Compute the variability collapse metric from
    the list of node features across layers and time.
    """

    collapse_metrics = {}
    for layer_name, features in features.items():
        class_means = scatter(features, labels.type(torch.int64), dim=0, reduce="mean")
        expanded_class_means = torch.index_select(class_means, dim=0, index=labels)
        z = features - expanded_class_means
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
    heatmap_data = np.array(heatmap_data)[::-1]
    # print(heatmap_data)
    fig, ax = plt.subplots(figsize=(20, 20))
    ax = sns.heatmap(heatmap_data, cmap="crest", vmin=0.1, vmax=1)
    _ = ax.set(xlabel="epoch/20", ylabel="depth")
    ax.set_xticklabels(ax.get_xticks(), rotation=90)
    ax.set_yticklabels(labels=heatmap_labels[::-1], rotation=0)
    fig = ax.get_figure()
    fig.savefig("plots/nc1_{}.png".format(model_name)) 
    plt.clf()
