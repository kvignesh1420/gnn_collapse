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


class WeightTracker:
    def __init__(self, state_dict, args):
        self.state_dict = state_dict
        self.args = args

    def compute_and_plot(self):
        self.prepare_weights_data()
        self.prepare_scaled_cov_data()
        self.plot_single_graph_weights_sv()

    def prepare_weights_data(self):
        """
        Create an ordered dict containing values and
        singular values of W1, W2 across conv_layers. We rely on
        the ordered `state_dict` for ordering the layers.
        """
        self.sv_data = OrderedDict()
        for param_name, param_value in self.state_dict.items():
            if "conv_layers" in param_name and "lin_" in param_name:
                sv_tensor = torch.linalg.svdvals(param_value).detach().numpy()
                sv_sum = np.sum(sv_tensor)
                layer_idx = param_name.split(".")[1]
                weights_data = self.sv_data.get(layer_idx, {})
                if "lin_l" in param_name:
                    weights_data["W1"] = {
                        "val": param_value.detach().numpy(),
                        "sv_sum": sv_sum
                    }
                elif "lin_r" in param_name:
                    weights_data["W2"] = {
                        "val": param_value.detach().numpy(),
                        "sv_sum": sv_sum
                    }
                self.sv_data[layer_idx] = weights_data

    def prepare_scaled_cov_data(self):
        for layer_idx, weights_data in self.sv_data.items():
            W1 = weights_data["W1"]["val"]
            W2 = weights_data["W2"]["val"]
            p = self.args["p"]
            q = self.args["q"]
            scaled_cov = (W1 + W2*(p-q)/(p+q))@((W1 + W2*(p-q)/(p+q)).transpose())
            sv_array = np.linalg.svd(scaled_cov, compute_uv=False)
            sv_sum = np.sum(sv_array)
            weights_data["scaled_cov"] = {
                "val": scaled_cov,
                "sv_sum": sv_sum
            }
            self.sv_data[layer_idx] = weights_data

    def plot_single_graph_weights_sv(self):
        """
        Plot the singular value metrics across depth for a single graph
        passed through the "trained" gnn
        """
        x = []
        y_sv_sum_W1 = []
        y_sv_sum_W2 = []
        y_sv_sum_scaled_cov = []

        for layer_name, weights_data in self.sv_data.items():
            y_sv_sum_W1.append(weights_data["W1"]["sv_sum"])
            y_sv_sum_W2.append(weights_data["W2"]["sv_sum"])
            y_sv_sum_scaled_cov.append(weights_data["scaled_cov"]["sv_sum"])
            x.append(layer_name)

        plt.grid(True)
        plt.plot(x, y_sv_sum_W1, label="sv_sum_W1")
        plt.plot(x, y_sv_sum_W2, label="sv_sum_W2")
        plt.plot(x, y_sv_sum_scaled_cov, label="sv_sum_scaled_cov")
        plt.legend(fontsize=30)
        plt.title("sum of sv of weights across layers")
        plt.xlabel("layer idx")
        plt.ylabel("sum of sv")
        plt.savefig("{}sv_sum_test.png".format(self.args["vis_dir"]))
        plt.clf()
