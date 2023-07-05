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
})
from gnn_collapse.utils.tracker import Metric
from gnn_collapse.models import Spectral_factory


def compute_nc1(features, labels, A_hat=None):
    """Compute the variability collapse metric from
    the list of node features across layers and time.
    NOTE: for feat matrices of shape: N x d
    """

    collapse_metrics = {}
    for layer_name, feat in features.items():
        if A_hat is not None:
            H = feat.t()
            HA_hat = H @ A_hat
            feat = HA_hat.t()
        class_means = scatter(feat, labels.type(torch.int64), dim=0, reduce="mean")
        expanded_class_means = torch.index_select(class_means, dim=0, index=labels)
        z = feat - expanded_class_means
        num_nodes = z.shape[0]

        # S_W : d x d
        S_W = z.t() @ z
        S_W /= num_nodes

        global_mean = torch.mean(class_means, dim=0)
        z = class_means - global_mean
        num_classes = class_means.shape[0]

        # S_B : d x d
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


def _prepare_trace_ratio_metrics(features_nc1_snapshots, normalized_features_nc1_snapshots):
    x = []
    for layer_name in features_nc1_snapshots[0]:
        x.append(layer_name)
    # metric objects
    y_S_W_ratio = Metric(label=r"$Tr(\Sigma^{Op(l)}_W)/Tr(\Sigma^{IN(l-1)}_W)$")
    y_S_B_ratio = Metric(label=r"$Tr(\Sigma^{Op(l)}_B)/Tr(\Sigma^{IN(l-1)}_B)$")

    for idx in range(len(x)-1):
        # temporary arrays
        y_S_W_ratio_arr = []
        y_S_B_ratio_arr = []

        for snapshot_idx in range(len(features_nc1_snapshots)):
            # capture features from next layer
            features_collapse_metrics = features_nc1_snapshots[snapshot_idx][x[idx+1]]
            # capture normalized features from current layer
            normalized_features_collapse_metrics = normalized_features_nc1_snapshots[snapshot_idx][x[idx]]

            # compute ratios
            S_W_trace_ratio = features_collapse_metrics["trace_S_W"]/normalized_features_collapse_metrics["trace_S_W"]
            S_B_trace_ratio = features_collapse_metrics["trace_S_B"]/normalized_features_collapse_metrics["trace_S_B"]
            y_S_W_ratio_arr.append(np.log10(S_W_trace_ratio))
            y_S_B_ratio_arr.append(np.log10(S_B_trace_ratio))

        y_S_W_ratio.update_mean_std(y_S_W_ratio_arr)
        y_S_B_ratio.update_mean_std(y_S_B_ratio_arr)

    return {
        "S_W_ratio" : y_S_W_ratio,
        "S_B_ratio" : y_S_B_ratio,
    }


def _prepare_nc1_metrics(x, snapshots, suffix):
    # metric objects
    y_nc1_type1 = Metric(label=r"$Tr(\Sigma_W \Sigma_B^{-1})/C$ : " + suffix)
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
                          normalized_features_nc1_snapshots, args, epoch, use_A_hat=False):
    """
    Plot the nc1 metric across depth for multiple test graphs passed through
    a well trained gnn
    """

    assert len(features_nc1_snapshots) > 0
    x = []
    for layer_name in features_nc1_snapshots[0]:
        x.append(layer_name)

    features_metrics = _prepare_nc1_metrics(x=x, snapshots=features_nc1_snapshots, suffix="Op")
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
    plt.legend()

    plot_feat_name_fig = "HA_hat" if use_A_hat else "H"
    plot_feat_name_title = "H\hat{A}" if use_A_hat else "H"

    if args["model_name"] in Spectral_factory:
        title=r"$NC_1$ of ${}$ across PI".format(plot_feat_name_title)
        xlabel="PI idx"
    else:
        title=r"$NC_1$ of ${}$ across layers".format(plot_feat_name_title)
        xlabel="layer idx"
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("$NC_1({})$ (log10 scale)".format(plot_feat_name_title))
    plt.tight_layout()
    plt.savefig("{}{}_nc1_test_epoch_{}.png".format(args["vis_dir"], plot_feat_name_fig, epoch))
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

    plt.legend()
    if args["model_name"] in Spectral_factory:
        title=r"$Tr(\Sigma_W), Tr(\Sigma_B)$ of ${}$ across PI".format(plot_feat_name_title)
        xlabel="PI idx"
    else:
        title=r"$Tr(\Sigma_W), Tr(\Sigma_B)$ of ${}$ across layers".format(plot_feat_name_title)
        xlabel="layer idx"
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Trace (log10 scale)")
    plt.tight_layout()
    plt.savefig("{}{}_cov_trace_test_epoch_{}.png".format(args["vis_dir"], plot_feat_name_fig, epoch))
    plt.clf()

    # plot ratio of S_W and S_B from norm feat to feat
    plt.grid(True)
    trace_ratio_metrics = _prepare_trace_ratio_metrics(
        features_nc1_snapshots=features_nc1_snapshots,
        normalized_features_nc1_snapshots=normalized_features_nc1_snapshots
    )
    plt.plot(x[1:],trace_ratio_metrics["S_W_ratio"].get_means(), label=trace_ratio_metrics["S_W_ratio"].label)
    plt.fill_between(
        x[1:],
        trace_ratio_metrics["S_W_ratio"].get_means() - trace_ratio_metrics["S_W_ratio"].get_stds(),
        trace_ratio_metrics["S_W_ratio"].get_means() + trace_ratio_metrics["S_W_ratio"].get_stds(),
        alpha=0.2,
        interpolate=True,
    )

    plt.plot(x[1:],trace_ratio_metrics["S_B_ratio"].get_means(), label=trace_ratio_metrics["S_B_ratio"].label)
    plt.fill_between(
        x[1:],
        trace_ratio_metrics["S_B_ratio"].get_means() - trace_ratio_metrics["S_B_ratio"].get_stds(),
        trace_ratio_metrics["S_B_ratio"].get_means() + trace_ratio_metrics["S_B_ratio"].get_stds(),
        alpha=0.2,
        interpolate=True,
    )

    plt.legend()
    plt.title("ratio of feat cov and normal feat cov")
    plt.xlabel("layer idx")
    plt.ylabel("Trace ratio (log10 scale)")
    plt.tight_layout()
    plt.savefig("{}{}_cov_trace_ratio_test_epoch_{}.png".format(args["vis_dir"], plot_feat_name_fig, epoch))
    plt.clf()
