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
})
from gnn_collapse.utils.tracker import Metric


class WeightTracker:
    def __init__(self, state_dict, features_nc1_snapshots, non_linear_features_nc1_snapshots,
                normalized_features_nc1_snapshots, epoch, args):
        self.state_dict = state_dict
        self.features_nc1_snapshots=features_nc1_snapshots
        self.non_linear_features_nc1_snapshots=non_linear_features_nc1_snapshots
        self.normalized_features_nc1_snapshots=normalized_features_nc1_snapshots
        self.epoch=epoch
        self.args=args

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
                layer_idx = int(param_name.split(".")[1])
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
            # take transpose of weights as we want a shape of
            # d_{l} x d_{l-1} for notational consistency
            W2 = weights_data["W2"]["val"].transpose()
            if self.args["use_W1"]:
                W1 = weights_data["W1"]["val"].transpose()
            else:
                W1 = np.zeros_like(W2)

            p = self.args["p_train"]
            q = self.args["q_train"]
            n = self.args["N_train"]/self.args["C"]

            beta_1 = (p-q)/(p+q)
            beta_2 = p/(n*(p+q))
            beta_3 = (p*p + q*q)/(n*(p+q)*(p+q))

            T_B = (W1 + beta_1 * W2).transpose() @ (W1 + beta_1 * W2)
            T_W = W1.transpose()@W1 + beta_2*( W2.transpose()@W1 + W1.transpose()@W2 ) + beta_3*W2.transpose()@W2

            weights_data["T_B"] = T_B
            weights_data["T_W"] = T_W
            self.sv_data[layer_idx] = weights_data
        # print(list(self.sv_data.keys()))
    
    def _compute_T_B_bound(self, S_B, T_B):
        T_B_sv_array = np.sort(np.linalg.svd(T_B, compute_uv=False))
        S_B_sv_array = np.sort(np.linalg.svd(S_B, compute_uv=False))
        lower_bound = np.dot(T_B_sv_array[::-1], S_B_sv_array)/np.sum(S_B_sv_array)
        upper_bound = np.dot(T_B_sv_array, S_B_sv_array)/np.sum(S_B_sv_array)
        return lower_bound, upper_bound

    def _compute_T_W_bound(self, S_W, T_W):
        T_W_sv_array = np.sort(np.linalg.svd(T_W, compute_uv=False))
        S_W_sv_array = np.sort(np.linalg.svd(S_W, compute_uv=False))
        lower_bound = np.dot(T_W_sv_array[::-1], S_W_sv_array)/np.sum(S_W_sv_array)
        upper_bound = np.dot(T_W_sv_array, S_W_sv_array)/np.sum(S_W_sv_array)
        return lower_bound, upper_bound


    def prepare_trace_ratio_metrics(self):
        x = []
        for layer_name in self.features_nc1_snapshots[0]:
            x.append(layer_name)
        # metric objects
        y_S_W_ratio = Metric(label=r"$Tr(\Sigma^{Op(l)}_W)/Tr(\Sigma^{IN(l-1)}_W)$")
        y_T_W_upper = Metric(label=r"UB: $Tr(\Sigma^{Op(l)}_W)/Tr(\Sigma^{IN(l-1)}_W)$")
        y_T_W_lower = Metric(label=r"LB: $Tr(\Sigma^{Op(l)}_W)/Tr(\Sigma^{IN(l-1)}_W)$")
        y_S_B_ratio = Metric(label=r"$Tr(\Sigma^{Op(l)}_B)/Tr(\Sigma^{IN(l-1)}_B)$")
        y_T_B_upper = Metric(label=r"UB: $Tr(\Sigma^{Op(l)}_B)/Tr(\Sigma^{IN(l-1)}_B)$")
        y_T_B_lower = Metric(label=r"LB: $Tr(\Sigma^{Op(l)}_B)/Tr(\Sigma^{IN(l-1)}_B)$")
        # # hack to plot this L-1 length array with L length weight based arrays
        # y_S_W_ratio.means.append(-np.inf)
        # y_S_W_ratio.stds.append(-np.inf)
        # y_S_B_ratio.means.append(-np.inf)
        # y_S_B_ratio.stds.append(-np.inf)

        for idx in range(len(x)-1):
            # temporary arrays
            y_S_W_ratio_arr = []
            y_T_W_upper_arr = []
            y_T_W_lower_arr = []

            y_S_B_ratio_arr = []
            y_T_B_upper_arr = []
            y_T_B_lower_arr = []

            for snapshot_idx in range(len(self.features_nc1_snapshots)):
                # capture features from next layer
                features_collapse_metrics = self.features_nc1_snapshots[snapshot_idx][x[idx+1]]
                # capture normalized features from current layer
                normalized_features_collapse_metrics = self.normalized_features_nc1_snapshots[snapshot_idx][x[idx]]

                # compute ratios
                S_W_trace_ratio = features_collapse_metrics["trace_S_W"]/normalized_features_collapse_metrics["trace_S_W"]
                S_B_trace_ratio = features_collapse_metrics["trace_S_B"]/normalized_features_collapse_metrics["trace_S_B"]
                y_S_W_ratio_arr.append(np.log10(S_W_trace_ratio))
                y_S_B_ratio_arr.append(np.log10(S_B_trace_ratio))

                # compute bound values
                T_W_lower_bound, T_W_upper_bound = self._compute_T_W_bound(
                    S_W=normalized_features_collapse_metrics["S_W"],
                    T_W=self.sv_data[x[idx]]["T_W"]
                )
                y_T_W_lower_arr.append(np.log10(T_W_lower_bound))
                y_T_W_upper_arr.append(np.log10(T_W_upper_bound))

                T_B_lower_bound, T_B_upper_bound = self._compute_T_B_bound(
                    S_B=normalized_features_collapse_metrics["S_B"],
                    T_B=self.sv_data[x[idx]]["T_B"]
                )
                y_T_B_lower_arr.append(np.log10(T_B_lower_bound))
                y_T_B_upper_arr.append(np.log10(T_B_upper_bound))

            y_S_W_ratio.update_mean_std(y_S_W_ratio_arr)
            y_S_B_ratio.update_mean_std(y_S_B_ratio_arr)
            y_T_W_lower.update_mean_std(y_T_W_lower_arr)
            y_T_W_upper.update_mean_std(y_T_W_upper_arr)
            y_T_B_lower.update_mean_std(y_T_B_lower_arr)
            y_T_B_upper.update_mean_std(y_T_B_upper_arr)

        return {
            "S_W_ratio" : y_S_W_ratio,
            "S_B_ratio" : y_S_B_ratio,
            "T_W_lower" : y_T_W_lower,
            "T_W_upper" : y_T_W_upper,
            "T_B_lower" : y_T_B_lower,
            "T_B_upper" : y_T_B_upper
        }

    def plot_weights_sv(self):
        """
        Plot the singular value metrics across depth for a single graph
        passed through the "trained" gnn
        """
        x = []
        trace_ratio_metrics =  self.prepare_trace_ratio_metrics()

        for layer_name, weights_data in self.sv_data.items():
            x.append(layer_name)
        
        # S_W
        plt.grid(True)
        plt.plot(x[1:], trace_ratio_metrics["S_W_ratio"].get_means(), label=trace_ratio_metrics["S_W_ratio"].label)
        plt.fill_between(
            x[1:],
            trace_ratio_metrics["S_W_ratio"].get_means() - trace_ratio_metrics["S_W_ratio"].get_stds(),
            trace_ratio_metrics["S_W_ratio"].get_means() + trace_ratio_metrics["S_W_ratio"].get_stds(),
            alpha=0.2,
            interpolate=True,
        )
        plt.plot(x[1:], trace_ratio_metrics["T_W_upper"].get_means(), linestyle="dashed",
                    label=trace_ratio_metrics["T_W_upper"].label)
        plt.fill_between(
            x[1:],
            trace_ratio_metrics["T_W_upper"].get_means() - trace_ratio_metrics["T_W_upper"].get_stds(),
            trace_ratio_metrics["T_W_upper"].get_means() + trace_ratio_metrics["T_W_upper"].get_stds(),
            alpha=0.2,
            interpolate=True,
        )
        plt.plot(x[1:], trace_ratio_metrics["T_W_lower"].get_means(), linestyle="dashed",
                    label=trace_ratio_metrics["T_W_lower"].label)
        plt.fill_between(
            x[1:],
            trace_ratio_metrics["T_W_lower"].get_means() - trace_ratio_metrics["T_W_lower"].get_stds(),
            trace_ratio_metrics["T_W_lower"].get_means() + trace_ratio_metrics["T_W_lower"].get_stds(),
            alpha=0.2,
            interpolate=True,
        )
        plt.legend()
        plt.title("bounding ratio of traces")
        plt.xlabel("layer idx")
        plt.ylabel("ratio (log10 scale)")
        plt.tight_layout()
        plt.savefig("{}S_W_trace_bounds_epoch_{}.png".format(self.args["vis_dir"], self.epoch))
        plt.clf()

        # S_B

        plt.grid(True)
        plt.plot(x[1:], trace_ratio_metrics["S_B_ratio"].get_means(), label=trace_ratio_metrics["S_B_ratio"].label)
        plt.fill_between(
            x[1:],
            trace_ratio_metrics["S_B_ratio"].get_means() - trace_ratio_metrics["S_B_ratio"].get_stds(),
            trace_ratio_metrics["S_B_ratio"].get_means() + trace_ratio_metrics["S_B_ratio"].get_stds(),
            alpha=0.2,
            interpolate=True,
        )
        plt.plot(x[1:], trace_ratio_metrics["T_B_upper"].get_means(), linestyle="dashed",
                    label=trace_ratio_metrics["T_B_upper"].label)
        plt.fill_between(
            x[1:],
            trace_ratio_metrics["T_B_upper"].get_means() - trace_ratio_metrics["T_B_upper"].get_stds(),
            trace_ratio_metrics["T_B_upper"].get_means() + trace_ratio_metrics["T_B_upper"].get_stds(),
            alpha=0.2,
            interpolate=True,
        )
        plt.plot(x[1:], trace_ratio_metrics["T_B_lower"].get_means(), linestyle="dashed",
                    label=trace_ratio_metrics["T_B_lower"].label)
        plt.fill_between(
            x[1:],
            trace_ratio_metrics["T_B_lower"].get_means() - trace_ratio_metrics["T_B_lower"].get_stds(),
            trace_ratio_metrics["T_B_lower"].get_means() + trace_ratio_metrics["T_B_lower"].get_stds(),
            alpha=0.2,
            interpolate=True,
        )
        plt.legend()
        plt.title("bounding ratio of traces")
        plt.xlabel("layer idx")
        plt.ylabel("ratio (log10 scale)")
        plt.tight_layout()
        plt.savefig("{}S_B_trace_bounds_epoch_{}.png".format(self.args["vis_dir"], self.epoch))
        plt.clf()


    # def plot_weights_sv(self):
    #     """
    #     Plot the singular value metrics across depth for a single graph
    #     passed through the "trained" gnn
    #     """
    #     x = []
    #     y_sv_sum_W1 = []
    #     y_sv_sum_W2 = []
    #     y_sv_sum_scaled_cov = []
    #     trace_ratio_metrics =  self.prepare_trace_ratio_metrics()

    #     for layer_name, weights_data in self.sv_data.items():
    #         if self.args["use_W1"]:
    #             y_sv_sum_W1.append(weights_data["W1"]["sv_sum"])
    #         y_sv_sum_W2.append(weights_data["W2"]["sv_sum"])
    #         y_sv_sum_scaled_cov.append(weights_data["scaled_cov"]["sv_sum"])
    #         x.append(layer_name)

    #     plt.grid(True)
    #     if self.args["use_W1"]:
    #         plt.plot(np.log10(y_sv_sum_W1), label="$\sum \lambda_i(W_1)$")
    #     plt.plot(np.log10(y_sv_sum_W2), label="$\sum \lambda_i(W_2)$")
    #     if self.args["use_W1"]:
    #         scaled_cov_label = r"$\sum \lambda_i(( W_1 + W_2 \frac{p-q}{p+q})( W_1 + W_2 \frac{p-q}{p+q})^T)$"
    #     else:
    #         scaled_cov_label = r"$\sum \lambda_i(( W_2 \frac{p-q}{p+q})( W_2 \frac{p-q}{p+q})^T)$"
    #     plt.plot(np.log10(y_sv_sum_scaled_cov), label=scaled_cov_label)

    #     plt.plot(trace_ratio_metrics["S_B_ratio"].get_means(), linestyle="dashed", label=trace_ratio_metrics["S_B_ratio"].label)

    #     plt.legend()
    #     plt.title("sum of singular values across layers")
    #     plt.xlabel("layer idx")
    #     plt.ylabel("$\sum \lambda_i$ (log10 scale)")
    #     plt.tight_layout()
    #     plt.savefig("{}sv_sum_epoch_{}.png".format(self.args["vis_dir"], self.epoch))
    #     plt.clf()
