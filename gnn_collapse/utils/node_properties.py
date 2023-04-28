"""
Analyse the node properties via collapse metrics
"""
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from torch_scatter import scatter
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 40,
    'lines.linewidth': 5,
    'axes.titlepad': 20,
    'axes.linewidth': 2,
    'figure.figsize': (30, 30)
})
from gnn_collapse.utils.tracker import Metric
from gnn_collapse.models import Spectral_factory
# import seaborn as sns

def compute_nc1(features, labels):
    """Compute the variability collapse metric from
    the list of node features across layers and time.
    NOTE: for feat matrices of shape: N x d
    """

    collapse_metrics = {}
    for layer_name, feat in features.items():
        class_means = scatter(feat, labels.type(torch.int64), dim=0, reduce="mean")
        expanded_class_means = torch.index_select(class_means, dim=0, index=labels)
        z = feat - expanded_class_means
        num_nodes = z.shape[0]
        # S_W = 0
        # for i in range(num_nodes):
        #     S_W += z[i, :].unsqueeze(1) @ z[i, :].unsqueeze(0)

        # S_W : d x d
        S_W = z.t() @ z  
        S_W /= num_nodes

        global_mean = torch.mean(class_means, dim=0)
        z = class_means - global_mean
        num_classes = class_means.shape[0]
        # S_B = 0
        # for i in range(num_classes):
        #     S_B += z[i, :].unsqueeze(1) @ z[i, :].unsqueeze(0)
        # S_W : d x d
        S_B = z.t() @ z
        S_B /= num_classes

        collapse_metric = torch.trace(S_W @ torch.linalg.pinv(S_B)) / num_classes
        collapse_metrics[layer_name] = {}
        collapse_metrics[layer_name]["trace_S_W_pinv_S_B"] = collapse_metric.detach().cpu().numpy()
        collapse_metrics[layer_name]["trace_S_W_div_S_B"] = (torch.trace(S_W)/torch.trace(S_B)).detach().cpu().numpy()
        collapse_metrics[layer_name]["trace_S_W"] = torch.trace(S_W).detach().cpu().numpy()
        collapse_metrics[layer_name]["trace_S_B"] = torch.trace(S_B).detach().cpu().numpy()
        collapse_metrics[layer_name]["S_W"] = S_W.detach().cpu().numpy()
        collapse_metrics[layer_name]["S_B"] = S_B.detach().cpu().numpy()
    return collapse_metrics


def _prepare_nc1_metrics(x, snapshots, suffix):
    # metric objects
    y_nc1_type1 = Metric(label=r"$Tr(\Sigma_W \Sigma_B^{-1})$ : " + suffix)
    y_nc1_type2 = Metric(label=r"$Tr(\Sigma_W)/Tr(\Sigma_B)$ : " + suffix)
    y_S_W = Metric(label=r"$Tr(\Sigma_W)$ : " + suffix)
    y_S_B = Metric(label=r"$Tr(\Sigma_B)$ : " + suffix)

    for layer_name in x:
        # temporary arrays
        y_nc1_type1_arr = []
        y_nc1_type2_arr = []
        y_S_W_arr = []
        y_S_B_arr = []

        for snapshot_idx in range(len(snapshots)):
            collapse_metrics = snapshots[snapshot_idx][layer_name]

            y_nc1_type1_arr.append(np.log10(collapse_metrics["trace_S_W_pinv_S_B"]))
            y_nc1_type2_arr.append(np.log10(collapse_metrics["trace_S_W_div_S_B"]))
            y_S_W_arr.append(np.log10(collapse_metrics["trace_S_W"]))
            y_S_B_arr.append(np.log10(collapse_metrics["trace_S_B"]))

        y_nc1_type1.update_mean_std(y_nc1_type1_arr)
        y_nc1_type2.update_mean_std(y_nc1_type2_arr)
        y_S_W.update_mean_std(y_S_W_arr)
        y_S_B.update_mean_std(y_S_B_arr)

    return {
        "nc1_type1": y_nc1_type1,
        "nc1_type2": y_nc1_type2,
        "S_W" : y_S_W,
        "S_B" : y_S_B,
    }


def plot_test_graphs_nc1(features_nc1_snapshots, non_linear_features_nc1_snapshots,
                          normalized_features_nc1_snapshots, args, epoch):
    """
    Plot the nc1 metric across depth for multiple test graphs passed through
    a well trained gnn
    """

    assert len(features_nc1_snapshots) > 0
    x = []
    for layer_name in features_nc1_snapshots[0]:
        x.append(layer_name)

    if args["model_name"] in Spectral_factory:
        suffix = "PI"
    else:
        suffix = "Op"

    features_metrics = _prepare_nc1_metrics(x=x, snapshots=features_nc1_snapshots, suffix=suffix)
    metrics_array = [features_metrics]

    if len(non_linear_features_nc1_snapshots) > 0:
        non_linear_features_metrics = _prepare_nc1_metrics(
            x=x, snapshots=non_linear_features_nc1_snapshots, suffix="non-lin")
        metrics_array.append(non_linear_features_metrics)

    if len(normalized_features_nc1_snapshots) > 0:
        normalized_features_metrics = _prepare_nc1_metrics(
            x=x, snapshots=normalized_features_nc1_snapshots, suffix="normalize")
        metrics_array.append(normalized_features_metrics)

    # plot nc1_type1 and nc1_type2 metrics
    plt.grid(True)
    for metric in metrics_array:
        plt.plot(x, metric["nc1_type1"].get_means(), label=metric["nc1_type1"].label)
        plt.fill_between(
            x,
            metric["nc1_type1"].get_means() - metric["nc1_type1"].get_stds(),
            metric["nc1_type1"].get_means() + metric["nc1_type1"].get_stds(),
            alpha=0.2,
            interpolate=True,
        )

    for metric in metrics_array:
        plt.plot(x, metric["nc1_type2"].get_means(), linestyle="dashed", label=metric["nc1_type2"].label)
        plt.fill_between(
            x,
            metric["nc1_type2"].get_means() - metric["nc1_type2"].get_stds(),
            metric["nc1_type2"].get_means() + metric["nc1_type2"].get_stds(),
            alpha=0.2,
            interpolate=True,
        )
    plt.legend(fontsize=30)
    if args["model_name"] in Spectral_factory:
        title="$NC_1$ of $H$ across PI"
        xlabel="PI idx"
    else:
        title="$NC_1$ of $H$ across layers"
        xlabel="layer idx"
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("$NC_1$ (log10 scale)")
    plt.savefig("{}nc1_test_epoch_{}.png".format(args["vis_dir"], epoch))
    plt.clf()

    # plot S_W and S_H

    plt.grid(True)
    for metric in metrics_array:
        plt.plot(x, metric["S_W"].get_means(), label=metric["S_W"].label)
        plt.fill_between(
            x,
            metric["S_W"].get_means() - metric["S_W"].get_stds(),
            metric["S_W"].get_means() + metric["S_W"].get_stds(),
            alpha=0.2,
            interpolate=True,
        )

    for metric in metrics_array:
        plt.plot(x, metric["S_B"].get_means(), linestyle="dashed", label=metric["S_B"].label)
        plt.fill_between(
            x,
            metric["S_B"].get_means() - metric["S_B"].get_stds(),
            metric["S_B"].get_means() + metric["S_B"].get_stds(),
            alpha=0.2,
            interpolate=True,
        )

    plt.legend(fontsize=30)
    if args["model_name"] in Spectral_factory:
        title=r"$Tr(\Sigma_W), Tr(\Sigma_B)$ of $H$ across PI"
        xlabel="PI idx"
    else:
        title=r"$Tr(\Sigma_W), Tr(\Sigma_B)$ of $H$ across layers"
        xlabel="layer idx"
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Trace (log10 scale)")
    plt.savefig("{}cov_trace_test_epoch_{}.png".format(args["vis_dir"], epoch))
    plt.clf()


# def plot_nc1_heatmap(nc1_snapshots, args, layer_type, layer_idx=None):
#     layers_nc1 = defaultdict(list)
#     for snapshot in nc1_snapshots:
#         for layer_name, collapse_metrics in snapshot.items():
#             layers_nc1[layer_name].append(collapse_metrics["trace_S_W_pinv_S_B"])

#     heatmap_data = []
#     heatmap_labels = []
#     for layer_name, collapse_metric_trend in layers_nc1.items():
#         heatmap_data.append(collapse_metric_trend)
#         heatmap_labels.append(layer_name)

#     heatmap_data = np.log10(np.array(heatmap_data)[::-1])
#     # print(heatmap_data)
#     fig, ax = plt.subplots(figsize=(80, 80))
#     ax = sns.heatmap(heatmap_data, cmap="crest")
#     _ = ax.set(xlabel="epoch/{}".format(args["nc_interval"]), ylabel="depth")
#     ax.set_xticklabels(ax.get_xticks(), rotation=90)
#     ax.set_yticklabels(labels=heatmap_labels[::-1], rotation=0)
#     fig = ax.get_figure()
#     if layer_idx is None:
#         filename = "{}{}_nc1_heatmap.png".format(args["vis_dir"], layer_type)
#     else:
#         filename = "{}{}_nc1_heatmap_{}.png".format(args["vis_dir"], layer_type, layer_idx)
#     fig.savefig(filename)
#     plt.clf()
#     plt.close()


# def plot_single_graph_nc1(features_nc1_snapshots, non_linear_features_nc1_snapshots,
#                           normalized_features_nc1_snapshots, weight_sv_data, args, epoch):
#     """
#     Plot the nc1 metric across depth for a single graph passed through
#     the gnn
#     """
#     x = []
#     # features
#     y_features_nc1_type1 = []
#     y_features_nc1_type2 = []
#     y_features_S_W = []
#     y_features_S_B = []

#     # non_linear features
#     y_non_linear_features_nc1_type1 = []
#     y_non_linear_features_nc1_type2 = []
#     y_non_linear_features_S_W = []
#     y_non_linear_features_S_B = []

#     # normalized features
#     y_normalized_features_nc1_type1 = []
#     y_normalized_features_nc1_type2 = []
#     y_normalized_features_S_W = []
#     y_normalized_features_S_B = []

#     # # dummy variable to skip matplotlib plot
#     # y_S_B_weight_cov = [-np.inf]
#     # # dummy variable to skip matplotlib plot
#     # y_S_B_weight_cov_sv_prod_sum = [-np.inf]

#     assert len(features_nc1_snapshots) == 1
#     assert len(non_linear_features_nc1_snapshots) == 1
#     assert len(normalized_features_nc1_snapshots) == 1

#     for layer_name in features_nc1_snapshots[0]:

#         x.append(layer_name)

#         features_collapse_metrics = features_nc1_snapshots[0][layer_name]
#         non_linear_features_collapse_metrics = non_linear_features_nc1_snapshots[0][layer_name]
#         normalized_features_collapse_metrics = normalized_features_nc1_snapshots[0][layer_name]

#         y_features_nc1_type1.append(np.log10(features_collapse_metrics["trace_S_W_pinv_S_B"]))
#         y_features_nc1_type2.append(np.log10(features_collapse_metrics["trace_S_W_div_S_B"]))
#         y_features_S_W.append(np.log10(features_collapse_metrics["trace_S_W"]))
#         y_features_S_B.append(np.log10(features_collapse_metrics["trace_S_B"]))

#         y_non_linear_features_nc1_type1.append(np.log10(non_linear_features_collapse_metrics["trace_S_W_pinv_S_B"]))
#         y_non_linear_features_nc1_type2.append(np.log10(non_linear_features_collapse_metrics["trace_S_W_div_S_B"]))
#         y_non_linear_features_S_W.append(np.log10(non_linear_features_collapse_metrics["trace_S_W"]))
#         y_non_linear_features_S_B.append(np.log10(non_linear_features_collapse_metrics["trace_S_B"]))

#         y_normalized_features_nc1_type1.append(np.log10(normalized_features_collapse_metrics["trace_S_W_pinv_S_B"]))
#         y_normalized_features_nc1_type2.append(np.log10(normalized_features_collapse_metrics["trace_S_W_div_S_B"]))
#         y_normalized_features_S_W.append(np.log10(normalized_features_collapse_metrics["trace_S_W"]))
#         y_normalized_features_S_B.append(np.log10(normalized_features_collapse_metrics["trace_S_B"]))

#         # next_layer_name = str(int(layer_name)+1)
#         # if next_layer_name in weight_sv_data:
#         #     W1 = weight_sv_data[next_layer_name]["W1"]["val"]
#         #     W2 = weight_sv_data[next_layer_name]["W2"]["val"]
#         #     S_B = collapse_metrics["S_B"]
#         #     p = args["p"]
#         #     q = args["q"]
#         #     scaled_W_cov = ( W1 + W2*(p-q)/(p+q) ).transpose() @ ( W1 + W2*(p-q)/(p+q) )
#         #     X_S_B = S_B @ scaled_W_cov
#         #     y_S_B_weight_cov.append(np.log10(np.trace(X_S_B)))
#         #     sv_S_B = np.linalg.svd(S_B, compute_uv=False)
#         #     sv_scaled_W_cov = np.linalg.svd(scaled_W_cov, compute_uv=False)
#         #     sv_prod_sum = np.sum(sv_S_B * sv_scaled_W_cov)
#         #     y_S_B_weight_cov_sv_prod_sum.append(np.log10(sv_prod_sum))

#     # plot nc1 metrics
#     plt.grid(True)
#     plt.plot(x, y_features_nc1_type1, label="$Tr(S_WS_B^{-1})$ : conv")
#     if args["non_linearity"] != "":
#         plt.plot(x, y_non_linear_features_nc1_type1, label="$Tr(S_WS_B^{-1})$ : non-lin")
#     if args["batch_norm"]:
#         plt.plot(x, y_normalized_features_nc1_type1, label="$Tr(S_WS_B^{-1})$ : normal")

#     plt.plot(x, y_features_nc1_type2, linestyle="dashed", label="$Tr(S_W)/Tr(S_B)$ : conv")
#     if args["non_linearity"] != "":
#         plt.plot(x, y_non_linear_features_nc1_type2, linestyle="dashed", label="$Tr(S_W)/Tr(S_B)$ : non-lin")
#     if args["batch_norm"]:
#         plt.plot(x, y_normalized_features_nc1_type2, linestyle="dashed", label="$Tr(S_W)/Tr(S_B)$ : normal")
#     plt.legend(fontsize=30)
#     plt.title("nc1 (test) across layers")
#     plt.xlabel("layer idx")
#     plt.ylabel("$NC_1$ (log scale)")
#     plt.savefig("{}nc1_test_epoch_{}.png".format(args["vis_dir"], epoch))
#     plt.clf()

#     plt.grid(True)
#     plt.plot(x, y_features_S_W, label="$Tr(S_W)$ : conv")
#     if args["non_linearity"] != "":
#         plt.plot(x, y_non_linear_features_S_W, label="$Tr(S_W)$ : non-lin")
#     if args["batch_norm"]:
#         plt.plot(x, y_normalized_features_S_W, label="$Tr(S_W)$ : normal")

#     plt.plot(x, y_features_S_B, linestyle="dashed", label="$Tr(S_B)$ : conv")
#     if args["non_linearity"] != "":
#         plt.plot(x, y_non_linear_features_S_B, linestyle="dashed", label="$Tr(S_B)$ : non-lin")
#     if args["batch_norm"]:
#         plt.plot(x, y_normalized_features_S_B, linestyle="dashed", label="$Tr(S_B)$ : normal")
#     # plt.plot(x, y_S_B_weight_cov, linestyle="dashed", label="$Tr(S_B@(W_1+rW_2)^{T}@(W_1+rW_2)$")
#     # plt.plot(x, y_S_B_weight_cov_sv_prod_sum, linestyle="dashed", label="$\sum \lambda(S_B)\lambda((W_1+rW_2)^{T}@(W_1+rW_2))$")
#     plt.legend(fontsize=30)
#     plt.title("$Tr(S_W), Tr(S_B)$ across layers")
#     plt.xlabel("layer idx")
#     plt.ylabel("Trace (log scale)")
#     plt.savefig("{}cov_trace_test_epoch_{}.png".format(args["vis_dir"], epoch))
#     plt.clf()



# def plot_feature_mean_distances(features, labels, args):
#     """Compute the distance between node features and the respective
#     class means across nodes.
#     """
#     pdist = torch.nn.PairwiseDistance(p=2)
#     dist_metrics = defaultdict(list)
#     for layer_name, feat in features.items():
#         class_means = scatter(feat, labels.type(torch.int64), dim=0, reduce="mean")
#         expanded_class_means = torch.index_select(class_means, dim=0, index=labels)
#         dists = []
#         num_nodes = feat.shape[0]
#         for i in range(num_nodes):
#             node_feat = feat[i, :]
#             class_mean = expanded_class_means[i, :]
#             dist = pdist(node_feat/(torch.norm(node_feat, p=2) + 1e-6), class_mean/(torch.norm(class_mean, p=2) + 1e-6) )
#             dists.append(dist.cpu())
#         dist_metrics[layer_name] = dists

#     heatmap_data = []
#     heatmap_labels = []
#     for layer_name, dist_trend in dist_metrics.items():
#         heatmap_data.append(dist_trend)
#         heatmap_labels.append(layer_name)

#     heatmap_data = np.array(heatmap_data)[::-1]
#     sorted_indices = np.argsort(labels.cpu())
#     heatmap_data = heatmap_data[:, sorted_indices]
#     # print(heatmap_data)
#     fig, ax = plt.subplots(figsize=(20, 20))
#     ax = sns.heatmap(heatmap_data, cmap="crest")
#     _ = ax.set(xlabel="node idx", ylabel="depth")
#     ax.set_xticklabels(ax.get_xticks(), rotation=90)
#     ax.set_yticklabels(labels=heatmap_labels[::-1], rotation=0)
#     fig = ax.get_figure()
#     fig.savefig("{}dist.png".format(args["vis_dir"]))
#     plt.clf()
#     plt.close()


# def plot_feature_mean_angles(features, labels, args):
#     """Compute the angle between node features and the respective
#     class means across nodes.
#     """
#     angle_metrics = defaultdict(list)
#     for layer_name, feat in features.items():
#         class_means = scatter(feat, labels.type(torch.int64), dim=0, reduce="mean")
#         expanded_class_means = torch.index_select(class_means, dim=0, index=labels)
#         angles = []
#         num_nodes = feat.shape[0]
#         for i in range(num_nodes):
#             node_feat = feat[i, :]
#             class_mean = expanded_class_means[i, :]
#             angle = torch.dot(node_feat, class_mean)/(torch.norm(node_feat, p=2)*torch.norm(class_mean, p =2) + 1e-6)
#             angles.append(angle)
#         angle_metrics[layer_name] = angles

#     heatmap_data = []
#     heatmap_labels = []
#     for layer_name, angle_trend in angle_metrics.items():
#         heatmap_data.append(angle_trend)
#         heatmap_labels.append(layer_name)

#     heatmap_data = np.array(heatmap_data)[::-1]
#     sorted_indices = np.argsort(labels)
#     heatmap_data = heatmap_data[:, sorted_indices]
#     # print(heatmap_data)
#     fig, ax = plt.subplots(figsize=(20, 20))
#     ax = sns.heatmap(heatmap_data, cmap="crest")
#     _ = ax.set(xlabel="node idx", ylabel="depth")
#     ax.set_xticklabels(ax.get_xticks(), rotation=90)
#     ax.set_yticklabels(labels=heatmap_labels[::-1], rotation=0)
#     fig = ax.get_figure()
#     fig.savefig("{}angles.png".format(args["vis_dir"]))
#     plt.clf()
#     plt.close()
