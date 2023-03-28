"""
Analyse the node properties via collapse metrics
"""
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from torch_scatter import scatter
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 25, 'lines.linewidth': 5, 'axes.titlepad': 20, "figure.figsize": (15, 15)})
import seaborn as sns

def plot_penultimate_layer_features(features, labels, args):
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
    fig.savefig("{}featplot.png".format(args["vis_dir"]))
    plt.clf()


def spectral_matching(Adj, features, labels, args):
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
        collapse_metrics[layer_name] = {}
        collapse_metrics[layer_name]["trace_S_W_pinv_S_B"] = collapse_metric.detach().cpu().numpy()
        collapse_metrics[layer_name]["trace_S_W_div_S_B"] = (torch.trace(S_W)/torch.trace(S_B)).detach().cpu().numpy()
        collapse_metrics[layer_name]["trace_S_W"] = torch.trace(S_W).detach().cpu().numpy()
        collapse_metrics[layer_name]["trace_S_B"] = torch.trace(S_B).detach().cpu().numpy()
    return collapse_metrics

def plot_nc1(nc1_snapshots, args, layer_idx=None):
    layers_nc1 = defaultdict(list)
    for snapshot in nc1_snapshots:
        for layer_name, collapse_metrics in snapshot.items():
            layers_nc1[layer_name].append(collapse_metrics["trace_S_W_pinv_S_B"])

    heatmap_data = []
    heatmap_labels = []
    for layer_name, collapse_metric_trend in layers_nc1.items():
        heatmap_data.append(collapse_metric_trend)
        heatmap_labels.append(layer_name)

    heatmap_data = np.log10(np.array(heatmap_data)[::-1])
    # print(heatmap_data)
    fig, ax = plt.subplots(figsize=(80, 80))
    ax = sns.heatmap(heatmap_data, cmap="crest")
    _ = ax.set(xlabel="epoch/{}".format(args["nc_interval"]), ylabel="depth")
    ax.set_xticklabels(ax.get_xticks(), rotation=90)
    ax.set_yticklabels(labels=heatmap_labels[::-1], rotation=0)
    fig = ax.get_figure()
    if layer_idx is None:
        filename = "{}nc1.png".format(args["vis_dir"])
    else:
        filename = "{}nc1_{}.png".format(args["vis_dir"], layer_idx)
    fig.savefig(filename)
    plt.clf()
    plt.close()

def plot_single_graph_nc1(nc1_snapshots, args):
    """
    Plot the nc1 metric across depth for a single graph passed through
    the gnn
    """
    x = []
    y_nc1_type1 = []
    y_nc1_type2 = []
    y_S_W = []
    y_S_B = []
    assert len(nc1_snapshots) == 1
    for layer_name, collapse_metrics in nc1_snapshots[0].items():
        y_nc1_type1.append(np.log10(collapse_metrics["trace_S_W_pinv_S_B"]))
        y_nc1_type2.append(np.log10(collapse_metrics["trace_S_W_div_S_B"]))
        y_S_W.append(np.log10(collapse_metrics["trace_S_W"]))
        y_S_B.append(np.log10(collapse_metrics["trace_S_B"]))
        x.append(layer_name)

    plt.plot(x, y_nc1_type1, label="$Tr(S_WS_B^{-1})$")
    plt.plot(x, y_nc1_type2, label="$Tr(S_W)/Tr(S_B)$")
    plt.plot(x, y_S_W, linestyle="dashed", label="$Tr(S_W)$")
    plt.plot(x, y_S_B, linestyle="dashed", label="$Tr(S_B)$")
    plt.legend()
    plt.title("nc1 (test) across layers")
    plt.xlabel("layer idx")
    plt.ylabel("$NC_1$ (log scale)")
    plt.savefig("{}nc1_test.png".format(args["vis_dir"]))
    plt.clf()


def plot_feature_mean_distances(features, labels, args):
    """Compute the distance between node features and the respective
    class means across nodes.
    """
    pdist = torch.nn.PairwiseDistance(p=2)
    dist_metrics = defaultdict(list)
    for layer_name, feat in features.items():
        class_means = scatter(feat, labels.type(torch.int64), dim=0, reduce="mean")
        expanded_class_means = torch.index_select(class_means, dim=0, index=labels)
        dists = []
        num_nodes = feat.shape[0]
        for i in range(num_nodes):
            node_feat = feat[i, :]
            class_mean = expanded_class_means[i, :]
            dist = pdist(node_feat/(torch.norm(node_feat, p=2) + 1e-6), class_mean/(torch.norm(class_mean, p=2) + 1e-6) )
            dists.append(dist.cpu())
        dist_metrics[layer_name] = dists

    heatmap_data = []
    heatmap_labels = []
    for layer_name, dist_trend in dist_metrics.items():
        heatmap_data.append(dist_trend)
        heatmap_labels.append(layer_name)

    heatmap_data = np.array(heatmap_data)[::-1]
    sorted_indices = np.argsort(labels.cpu())
    heatmap_data = heatmap_data[:, sorted_indices]
    # print(heatmap_data)
    fig, ax = plt.subplots(figsize=(20, 20))
    ax = sns.heatmap(heatmap_data, cmap="crest")
    _ = ax.set(xlabel="node idx", ylabel="depth")
    ax.set_xticklabels(ax.get_xticks(), rotation=90)
    ax.set_yticklabels(labels=heatmap_labels[::-1], rotation=0)
    fig = ax.get_figure()
    fig.savefig("{}dist.png".format(args["vis_dir"]))
    plt.clf()
    plt.close()


def plot_feature_mean_angles(features, labels, args):
    """Compute the angle between node features and the respective
    class means across nodes.
    """
    angle_metrics = defaultdict(list)
    for layer_name, feat in features.items():
        class_means = scatter(feat, labels.type(torch.int64), dim=0, reduce="mean")
        expanded_class_means = torch.index_select(class_means, dim=0, index=labels)
        angles = []
        num_nodes = feat.shape[0]
        for i in range(num_nodes):
            node_feat = feat[i, :]
            class_mean = expanded_class_means[i, :]
            angle = torch.dot(node_feat, class_mean)/(torch.norm(node_feat, p=2)*torch.norm(class_mean, p =2) + 1e-6)
            angles.append(angle)
        angle_metrics[layer_name] = angles

    heatmap_data = []
    heatmap_labels = []
    for layer_name, angle_trend in angle_metrics.items():
        heatmap_data.append(angle_trend)
        heatmap_labels.append(layer_name)

    heatmap_data = np.array(heatmap_data)[::-1]
    sorted_indices = np.argsort(labels)
    heatmap_data = heatmap_data[:, sorted_indices]
    # print(heatmap_data)
    fig, ax = plt.subplots(figsize=(20, 20))
    ax = sns.heatmap(heatmap_data, cmap="crest")
    _ = ax.set(xlabel="node idx", ylabel="depth")
    ax.set_xticklabels(ax.get_xticks(), rotation=90)
    ax.set_yticklabels(labels=heatmap_labels[::-1], rotation=0)
    fig = ax.get_figure()
    fig.savefig("{}angles.png".format(args["vis_dir"]))
    plt.clf()
    plt.close()
