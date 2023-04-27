"""
Analyse the weight properties
"""

import numpy as np
from collections import OrderedDict
import torch
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 40,
    'lines.linewidth': 5,
    'axes.titlepad': 20,
    'axes.linewidth': 2,
    'figure.figsize': (15, 15)
})
from gnn_collapse.utils.tracker import Metric


class WeightTracker:
    def __init__(self, state_dict, features_nc1_snapshots, non_linear_features_nc1_snapshots,
                normalized_features_nc1_snapshots, args):
        self.state_dict = state_dict
        self.features_nc1_snapshots=features_nc1_snapshots
        self.non_linear_features_nc1_snapshots=non_linear_features_nc1_snapshots
        self.normalized_features_nc1_snapshots=normalized_features_nc1_snapshots
        self.args = args

    def compute_and_plot(self):
        self.prepare_weights_data()
        self.prepare_scaled_cov_data()
        self.plot_weights_sv()

    def prepare_weights_data(self):
        """
        Create an ordered dict containing values and
        singular values of W1, W2 across conv_layers. We rely on
        the ordered `state_dict` for ordering the layers.
        """
        self.sv_data = OrderedDict()
        for param_name, param_value in self.state_dict.items():
            if "conv_layers" in param_name and "lin_" in param_name:
                sv_tensor = torch.linalg.svdvals(param_value).detach().cpu().numpy()
                sv_sum = np.sum(sv_tensor)
                layer_idx = param_name.split(".")[1]
                weights_data = self.sv_data.get(layer_idx, {})
                if "lin_root" in param_name:
                    weights_data["W1"] = {
                        "val": param_value.detach().cpu().numpy(),
                        "sv_sum": sv_sum
                    }
                elif "lin_rel" in param_name:
                    weights_data["W2"] = {
                        "val": param_value.detach().cpu().numpy(),
                        "sv_sum": sv_sum
                    }
                self.sv_data[layer_idx] = weights_data

    def prepare_scaled_cov_data(self):
        for layer_idx, weights_data in self.sv_data.items():
            if self.args["use_W1"]:
                W1 = weights_data["W1"]["val"]
            W2 = weights_data["W2"]["val"]
            p = self.args["p_train"]
            q = self.args["q_train"]
            if self.args["use_W1"]:
                scaled_cov = (W1 + W2*(p-q)/(p+q))@((W1 + W2*(p-q)/(p+q)).transpose())
            else:
                scaled_cov = (W2*(p-q)/(p+q))@((W2*(p-q)/(p+q)).transpose())
            sv_array = np.linalg.svd(scaled_cov, compute_uv=False)
            sv_sum = np.sum(sv_array)
            weights_data["scaled_cov"] = {
                "val": scaled_cov,
                "sv_sum": sv_sum
            }
            self.sv_data[layer_idx] = weights_data

    def prepare_trace_ratio_metrics(self):
        x = []
        for layer_name in self.features_nc1_snapshots[0]:
            x.append(layer_name)
        # metric objects
        y_S_B_ratio = Metric(label=r"$Tr(S^{Op(l+1)}_B)/Tr(S^{IN(l)}_B)$")

        for idx in range(len(x)-1):
            # temporary arrays
            y_S_B_ratio_arr = []
            for snapshot_idx in range(len(self.features_nc1_snapshots)):
                # capture features from next layer
                features_collapse_metrics = self.features_nc1_snapshots[snapshot_idx][x[idx+1]]
                # capture normalized features from current layer
                normalized_features_collapse_metrics = self.normalized_features_nc1_snapshots[snapshot_idx][x[idx]]

                trace_ratio = features_collapse_metrics["trace_S_B"]/normalized_features_collapse_metrics["trace_S_B"]
                y_S_B_ratio_arr.append(np.log10(trace_ratio))

            y_S_B_ratio.update_mean_std(y_S_B_ratio_arr)

        return {
            "S_B_ratio" : y_S_B_ratio,
        }

    def plot_weights_sv(self):
        """
        Plot the singular value metrics across depth for a single graph
        passed through the "trained" gnn
        """
        x = []
        y_sv_sum_W1 = []
        y_sv_sum_W2 = []
        y_sv_sum_scaled_cov = []
        trace_ratio_metrics =  self.prepare_trace_ratio_metrics()

        for layer_name, weights_data in self.sv_data.items():
            if self.args["use_W1"]:
                y_sv_sum_W1.append(weights_data["W1"]["sv_sum"])
            y_sv_sum_W2.append(weights_data["W2"]["sv_sum"])
            y_sv_sum_scaled_cov.append(weights_data["scaled_cov"]["sv_sum"])
            x.append(layer_name)

        plt.grid(True)
        if self.args["use_W1"]:
            plt.plot(np.log10(y_sv_sum_W1), label="$\sum \lambda_i(W_1)$")
        plt.plot(np.log10(y_sv_sum_W2), label="$\sum \lambda_i(W_2)$")
        if self.args["use_W1"]:
            scaled_cov_label = r"$\sum \lambda_i(( W_1 + W_2 \frac{p-q}{p+q})( W_1 + W_2 \frac{p-q}{p+q})^T)$"
        else:
            scaled_cov_label = r"$\sum \lambda_i(( W_2 \frac{p-q}{p+q})( W_2 \frac{p-q}{p+q})^T)$"
        plt.plot(np.log10(y_sv_sum_scaled_cov), label=scaled_cov_label)

        plt.plot(x, trace_ratio_metrics["S_B_ratio"].get_means(), linestyle="dashed", label=trace_ratio_metrics["S_B_ratio"].label)
        plt.fill_between(
            x,
            trace_ratio_metrics["S_B_ratio"].get_means() - trace_ratio_metrics["S_B_ratio"].get_stds(),
            trace_ratio_metrics["S_B_ratio"].get_means() + trace_ratio_metrics["S_B_ratio"].get_stds(),
            alpha=0.2,
            interpolate=True,
        )

        plt.legend(fontsize=30)
        plt.title("sum of singular values across layers")
        plt.xlabel("layer idx")
        plt.ylabel("$\sum \lambda_i$ (log10 scale)")
        plt.savefig("{}sv_sum.png".format(self.args["vis_dir"]))
        plt.clf()
