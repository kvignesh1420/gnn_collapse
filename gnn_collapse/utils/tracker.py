import os
import imageio
import numpy as np
import torch
from torch_scatter import scatter
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 40,
    'lines.linewidth': 5,
    'axes.titlepad': 20,
    'axes.linewidth': 2,
    'figure.figsize': (15, 15)
})

class Metric:
    def __init__(self, label):
        self.means = []
        self.stds = []
        self.label = label

    def update_mean_std(self, arr):
        self.means.append(np.mean(arr))
        self.stds.append(np.std(arr))

    def size(self):
        return len(self.means)

    def get_means(self):
        return np.array(self.means)

    def get_stds(self):
        return np.array(self.stds)


class GUFMMetricTracker:
    def __init__(self, args) -> None:
        self.args = args
        # training stats
        self.train_loss = []
        self.train_accuracy = []

        # NC1 traces

        self.H_S_W_traces = Metric(label="$Tr(S_W)$")
        self.H_S_B_traces = Metric(label="$Tr(S_B)$")
        self.H_nc1_type1s = Metric(label="$Tr(S_WS_B^{-1})$")
        self.H_nc1_type2s = Metric(label="$Tr(S_W)/Tr(S_B)$")

        self.HA_hat_S_W_traces = Metric(label="$Tr(S_W)$")
        self.HA_hat_S_B_traces = Metric(label="$Tr(S_B)$")
        self.HA_hat_nc1_type1s = Metric(label="$Tr(S_WS_B^{-1})$")
        self.HA_hat_nc1_type2s = Metric(label="$Tr(S_W)/Tr(S_B)$")

        # NC1 SNR
        self.W1H_NC1_SNR = Metric(label="$W_1H$")
        self.W2HA_hat_NC1_SNR = Metric(label="$W_2H\hat{A}$")

        # Norms
        self.W1_frobenius_norms = Metric(label="$||W_1||_F$")
        self.W2_frobenius_norms = Metric(label="$||W_2||_F$")
        self.H_frobenius_norms = Metric(label="$||H||_F$")
        self.HA_hat_frobenius_norms = Metric(label="$||H\hat{A}||_F$")

        # NC2 ETF
        self.H_NC2_ETF = Metric(label="$H$ (ETF)")
        self.HA_hat_NC2_ETF = Metric(label="$H\hat{A}$ (ETF)")
        self.W1_NC2_ETF = Metric(label="$W_1$ (ETF)")
        self.W2_NC2_ETF = Metric(label="$W_2$ (ETF)")
        # NC2 OF
        self.H_NC2_OF = Metric(label="$H$ (OF)")
        self.HA_hat_NC2_OF =  Metric(label="$H\hat{A}$ (OF)")
        self.W1_NC2_OF =  Metric(label="$W_1$ (OF)")
        self.W2_NC2_OF = Metric(label="$W_2$ (OF)")

        # NC3 ETF
        self.W1_H_NC3_ETF = Metric("$(W_1H, ETF)$")
        self.W2_HA_hat_NC3_ETF = Metric(label="$(W_2H\hat{A}, ETF)$")
        self.W1H_W2HA_hat_NC3_ETF = Metric(label="$(\hat{Y}, ETF)$")
        # NC3 OF
        self.W1_H_NC3_OF = Metric("$(W_1H, OF)$")
        self.W2_HA_hat_NC3_OF = Metric(label="$(W_2H\hat{A}, OF)$")
        self.W1H_W2HA_hat_NC3_OF = Metric(label="$(\hat{Y}, OF)$")
        # plain alignment
        self.W1_H_NC3 = Metric(label="$(W_1, H)$")
        self.W2_HA_hat_NC3 = Metric(label="$(W_2, H\hat{A})$")

        self.x = []

    def get_W_feat_NC1_SNR(self, feat, labels, W):
        with torch.no_grad():
            class_means = scatter(feat, labels.type(torch.int64), dim=1, reduce="mean")
            expanded_class_means = torch.index_select(class_means, dim=1, index=labels)
            z = feat - expanded_class_means
            signal = W @ expanded_class_means
            noise = W @ z
            signal_res = torch.norm(signal)
            noise_res = torch.norm(noise)
            res = signal_res/noise_res
            return res

    def get_nc1(self, feat, labels):
        with torch.no_grad():
            # print("FEAT shape ", feat.shape)
            class_means = scatter(feat, labels.type(torch.int64), dim=1, reduce="mean")
            expanded_class_means = torch.index_select(class_means, dim=1, index=labels)
            z = feat - expanded_class_means
            num_nodes = z.shape[1]
            S_W = 0
            for i in range(num_nodes):
                S_W += z[:, i].unsqueeze(1) @ z[:, i].unsqueeze(0)
            S_W /= num_nodes
            # print(S_W)
            # print("class means shape: ",class_means.shape)
            global_mean = torch.mean(class_means, dim=1).unsqueeze(-1)
            # print("global mean shape: ", global_mean.shape)
            z = class_means - global_mean
            num_classes = class_means.shape[1]
            S_B = 0
            for i in range(num_classes):
                # print(z[:, i])
                S_B += z[:, i].unsqueeze(1) @ z[:, i].unsqueeze(0)
                # print(S_B)
            S_B /= num_classes
            # print(S_W, S_B)
            collapse_metric_type1 = torch.trace(S_W @ torch.linalg.pinv(S_B)) / num_classes
            collapse_metric_type2 = torch.trace(S_W)/torch.trace(S_B)
        return torch.trace(S_W), torch.trace(S_B), collapse_metric_type1, collapse_metric_type2

    def get_weights_or_feat_ETF_relation(self, M):
        """Adapted from: https://github.com/tding1/Neural-Collapse/blob/main/validate_NC.py
        Args:
            M: Can be weights W_1, W_2 or means of class features w.r.t H, HA_hat
        """
        with torch.no_grad():
            K = M.shape[0]
            assert K == self.args["C"]
            MMT = torch.mm(M, M.T)
            MMT /= torch.norm(MMT, p='fro')
            # print(MMT)
            sub = (torch.eye(K) - 1 / K * torch.ones((K, K))) / pow(K - 1, 0.5)
            sub = sub.to(self.args["device"])
            ETF_metric = torch.norm(MMT - sub, p='fro')
        return ETF_metric

    def compute_W_H_ETF_relation(self, W, feat, labels):
        """Adapted from: https://github.com/tding1/Neural-Collapse/blob/main/validate_NC.py"""
        with torch.no_grad():
            class_means = scatter(feat, labels.type(torch.int64), dim=1, reduce="mean")
            # global_mean = torch.mean(class_means, dim=1).unsqueeze(-1)
            z = class_means
            Wz = torch.mm(W, z)
            Wz /= torch.norm(Wz, p='fro')
            K = W.shape[0]
            assert K == self.args["C"]
            sub = 1 / pow(K - 1, 0.5) * (torch.eye(K) - 1 / K * torch.ones((K, K)))
            sub = sub.to(self.args["device"])
            res = torch.norm(Wz - sub, p='fro')
        return res

    def get_weights_or_feat_OF_relation(self, M):
        with torch.no_grad():
            K = M.shape[0]
            assert K == self.args["C"]
            MMT = torch.mm(M, M.T)
            MMT /= torch.norm(MMT, p='fro')
            sub = torch.eye(K)/np.sqrt(K)
            sub = sub.to(self.args["device"])
            ETF_metric = torch.norm(MMT - sub, p='fro')
        return ETF_metric

    def compute_W_H_OF_relation(self, W, feat, labels):
        with torch.no_grad():
            class_means = scatter(feat, labels.type(torch.int64), dim=1, reduce="mean")
            z = class_means
            Wz = torch.mm(W, z)
            Wz /= torch.norm(Wz, p='fro')
            K = W.shape[0]
            assert K == self.args["C"]
            sub = torch.eye(K)/np.sqrt(K)
            sub = sub.to(self.args["device"])
            res = torch.norm(Wz - sub, p='fro')
        return res

    def compute_W_H_alignment(self, W, feat, labels):
        with torch.no_grad():
            class_means = scatter(feat, labels.type(torch.int64), dim=1, reduce="mean")
            # global_mean = torch.mean(class_means, dim=1).unsqueeze(-1)
            z = class_means
            res = torch.norm(W/torch.norm(W) - z.t()/torch.norm(z), p='fro')
        return res

    def plot_train_loss(self, ax, train_loss_array, nc_interval):
        self.train_loss.append(np.mean(train_loss_array))
        self.x = range(len(self.train_loss))
        ax[0, 0].plot(self.x, np.array(self.train_loss))
        ax[0, 0].grid(True)
        _ = ax[0, 0].set(xlabel=r"$iter\%{}$".format(nc_interval), ylabel="loss", title="Train loss")
        return ax

    def plot_train_accuracy(self, ax, train_accuracy_array, nc_interval):
        self.train_accuracy.append(np.mean(train_accuracy_array))
        ax[0, 1].plot(self.x, np.array(self.train_accuracy))
        ax[0, 1].grid(True)
        _ = ax[0, 1].set(xlabel=r"$iter\%{}$".format(nc_interval), ylabel="overlap", title="Train overlap")
        return ax

    def plot_NC1_H(self, ax, H_array, labels_array, nc_interval):
        S_W_traces = []
        S_B_traces = []
        nc1_type1s = []
        nc1_type2s = []
        for H, labels in zip(H_array, labels_array):
            feat = H
            S_W_trace, S_B_trace, nc1_type1, nc1_type2 = self.get_nc1(feat=feat, labels=labels)
            S_W_traces.append(S_W_trace.detach().cpu().numpy())
            S_B_traces.append(S_B_trace.detach().cpu().numpy())
            nc1_type1s.append(nc1_type1.detach().cpu().numpy())
            nc1_type2s.append(nc1_type2.detach().cpu().numpy())

        self.H_S_W_traces.update_mean_std(np.log10(np.array(S_W_traces)))
        self.H_S_B_traces.update_mean_std(np.log10(np.array(S_B_traces)))
        self.H_nc1_type1s.update_mean_std(np.log10(np.array(nc1_type1s)))
        self.H_nc1_type2s.update_mean_std(np.log10(np.array(nc1_type2s)))

        for metric in [self.H_S_W_traces, self.H_S_B_traces, self.H_nc1_type1s, self.H_nc1_type2s]:
            ax[0, 2].plot(self.x, metric.get_means(), label=metric.label)
            ax[0, 2].fill_between(
                self.x,
                metric.get_means() - metric.get_stds(),
                metric.get_means() + metric.get_stds(),
                alpha=0.2,
                interpolate=True,
            )

        ax[0, 2].grid(True)
        _ = ax[0, 2].set(
            xlabel=r"$iter\%{}$".format(nc_interval),
            ylabel="$NC_1$ (log10 scale)",
            title="$NC_1$ of H"
        )
        ax[0, 2].legend(fontsize=30)
        return ax

    def plot_NC1_HA_hat(self, ax, H_array, A_hat_array, labels_array, nc_interval):

        S_W_traces = []
        S_B_traces = []
        nc1_type1s = []
        nc1_type2s = []
        for H, A_hat, labels in zip(H_array, A_hat_array, labels_array):
            feat = H @ A_hat
            S_W_trace, S_B_trace, nc1_type1, nc1_type2 = self.get_nc1(feat=feat, labels=labels)
            S_W_traces.append(S_W_trace.detach().cpu().numpy())
            S_B_traces.append(S_B_trace.detach().cpu().numpy())
            nc1_type1s.append(nc1_type1.detach().cpu().numpy())
            nc1_type2s.append(nc1_type2.detach().cpu().numpy())

        self.HA_hat_S_W_traces.update_mean_std(np.log10(np.array(S_W_traces)))
        self.HA_hat_S_B_traces.update_mean_std(np.log10(np.array(S_B_traces)))
        self.HA_hat_nc1_type1s.update_mean_std(np.log10(np.array(nc1_type1s)))
        self.HA_hat_nc1_type2s.update_mean_std(np.log10(np.array(nc1_type2s)))

        for metric in [self.HA_hat_S_W_traces, self.HA_hat_S_B_traces, self.HA_hat_nc1_type1s, self.HA_hat_nc1_type2s]:
            ax[0, 3].plot(self.x, metric.get_means(), label=metric.label)
            ax[0, 3].fill_between(
                self.x,
                metric.get_means() - metric.get_stds(),
                metric.get_means() + metric.get_stds(),
                alpha=0.2,
                interpolate=True,
            )

        ax[0, 3].grid(True)
        _ = ax[0, 3].set(
            xlabel=r"$iter\%{}$".format(nc_interval),
            ylabel="$NC_1$ (log10 scale)",
            title="$NC_1$ of $H\hat{{A}}$"
        )
        ax[0, 3].legend(fontsize=30)
        return ax

    def plot_NC1_SNR(self, ax, W_1, W_2, H_array, A_hat_array, labels_array, nc_interval):
        W1H_NC1_SNR_arr = []
        W2HA_hat_NC1_SNR_arr = []

        for H, A_hat, labels in zip(H_array, A_hat_array, labels_array):
            res = self.get_W_feat_NC1_SNR(feat=H, labels=labels, W=W_1)
            W1H_NC1_SNR_arr.append(res.detach().cpu().numpy())
            res = self.get_W_feat_NC1_SNR(feat=H@A_hat, labels=labels, W=W_2)
            W2HA_hat_NC1_SNR_arr.append(res.detach().cpu().numpy())

        self.W1H_NC1_SNR.update_mean_std(np.log10(np.array(W1H_NC1_SNR_arr)))
        self.W2HA_hat_NC1_SNR.update_mean_std(np.log10(np.array(W2HA_hat_NC1_SNR_arr)))
        
        metrics = [self.W1H_NC1_SNR] if self.args["use_W1"] else []
        metrics.append(self.W2HA_hat_NC1_SNR)
        for metric in metrics:
            ax[1, 0].plot(self.x, metric.get_means(), label=metric.label)
            ax[1, 0].fill_between(
                self.x,
                metric.get_means() - metric.get_stds(),
                metric.get_means() + metric.get_stds(),
                alpha=0.2,
                interpolate=True,
            )

        ax[1, 0].grid(True)
        _ = ax[1, 0].set(xlabel=r"$iter\%{}$".format(nc_interval), ylabel="SNR (log10 scale)", title="$NC_1$ SNR")
        ax[1, 0].legend(fontsize=30)
        return ax

    def plot_fro_norms(self, ax, W_1, W_2, H_array, A_hat_array, nc_interval):
        W1_fro_norm = torch.norm(W_1, p="fro").detach().cpu().numpy()
        self.W1_frobenius_norms.update_mean_std([W1_fro_norm])

        W2_fro_norm = torch.norm(W_2, p="fro").detach().cpu().numpy()
        self.W2_frobenius_norms.update_mean_std([W2_fro_norm])

        H_fro_norms = [torch.norm(H, p="fro").detach().cpu().numpy() for H in H_array]
        self.H_frobenius_norms.update_mean_std(H_fro_norms)

        HA_hat_fro_norms = [torch.norm(H@A_hat, p="fro").detach().cpu().numpy()
                            for H, A_hat in zip(H_array, A_hat_array)]
        self.HA_hat_frobenius_norms.update_mean_std(HA_hat_fro_norms)

        metrics = [self.W1_frobenius_norms] if self.args["use_W1"] else []
        metrics.extend([self.W2_frobenius_norms, self.H_frobenius_norms, self.HA_hat_frobenius_norms])
        for metric in metrics:
            ax[1, 1].plot(self.x, metric.get_means(), label=metric.label)
            ax[1, 1].fill_between(
                self.x,
                metric.get_means() - metric.get_stds(),
                metric.get_means() + metric.get_stds(),
                alpha=0.2,
                interpolate=True,
            )

        ax[1, 1].grid(True)
        _ = ax[1, 1].set(xlabel=r"$iter\%{}$".format(nc_interval), ylabel="$||.||_F$", title="Frobenius norms")
        ax[1, 1].legend(fontsize=30)
        return ax

    def plot_NC2(self, ax, W_1, W_2, H_array, A_hat_array, labels_array, nc_interval):

        # temporary arrays
        W1_NC2_ETF_arr = []
        W2_NC2_ETF_arr = []
        H_NC2_ETF_arr = []
        HA_hat_NC2_ETF_arr = []

        W1_NC2_OF_arr = []
        W2_NC2_OF_arr = []
        H_NC2_OF_arr = []
        HA_hat_NC2_OF_arr = []

        # NC2 ETF Alignment
        W1_ETF_alignment = self.get_weights_or_feat_ETF_relation(M=W_1)
        W1_NC2_ETF_arr.append(W1_ETF_alignment.detach().cpu().numpy())

        W2_ETF_alignment = self.get_weights_or_feat_ETF_relation(M=W_2)
        W2_NC2_ETF_arr.append(W2_ETF_alignment.detach().cpu().numpy())

        for H, A_hat, labels in zip(H_array, A_hat_array, labels_array):
            H_class_means = scatter(H, labels.type(torch.int64), dim=1, reduce="mean")
            # transpose is needed to have shape[0] = C
            H_class_means = H_class_means.t()
            H_class_means_ETF_alignment = self.get_weights_or_feat_ETF_relation(M=H_class_means)
            H_NC2_ETF_arr.append(H_class_means_ETF_alignment.detach().cpu().numpy())

            HA_hat_class_means = scatter(H@A_hat, labels.type(torch.int64), dim=1, reduce="mean")
                # transpose is needed to have feat.shape[0] = C
            HA_hat_class_means = HA_hat_class_means.t()
            HA_hat_class_means_ETF_alignment = self.get_weights_or_feat_ETF_relation(M=HA_hat_class_means)
            HA_hat_NC2_ETF_arr.append(HA_hat_class_means_ETF_alignment.detach().cpu().numpy())

        # NC2 OF Alignment
        W1_OF_alignment = self.get_weights_or_feat_OF_relation(M=W_1)
        W1_NC2_OF_arr.append(W1_OF_alignment.detach().cpu().numpy())

        W2_OF_alignment = self.get_weights_or_feat_OF_relation(M=W_2)
        W2_NC2_OF_arr.append(W2_OF_alignment.detach().cpu().numpy())

        for H, A_hat, labels in zip(H_array, A_hat_array, labels_array):
            # no need to subtract global mean to compute alignment with OF
            H_class_means = scatter(H, labels.type(torch.int64), dim=1, reduce="mean")
            # transpose is needed to have feat.shape[0] = C
            H_class_means = H_class_means.t()
            H_class_means_OF_alignment = self.get_weights_or_feat_OF_relation(M=H_class_means)
            H_NC2_OF_arr.append(H_class_means_OF_alignment.detach().cpu().numpy())

            # no need to subtract global mean to compute alignment with OF
            HA_hat_class_means = scatter(H@A_hat, labels.type(torch.int64), dim=1, reduce="mean")
            # transpose is needed to have feat.shape[0] = C
            HA_hat_class_means = HA_hat_class_means.t()
            HA_hat_class_means_OF_alignment = self.get_weights_or_feat_OF_relation(M=HA_hat_class_means)
            HA_hat_NC2_OF_arr.append(HA_hat_class_means_OF_alignment.detach().cpu().numpy())

        # NC2 ETF
        self.H_NC2_ETF.update_mean_std(np.log10(np.array(H_NC2_ETF_arr)))
        self.HA_hat_NC2_ETF.update_mean_std(np.log10(np.array(HA_hat_NC2_ETF_arr)))
        self.W1_NC2_ETF.update_mean_std(np.log10(np.array(W1_NC2_ETF_arr)))
        self.W2_NC2_ETF.update_mean_std(np.log10(np.array(W2_NC2_ETF_arr)))
        # NC2 OF
        self.H_NC2_OF.update_mean_std(np.log10(np.array(H_NC2_OF_arr)))
        self.HA_hat_NC2_OF.update_mean_std(np.log10(np.array(HA_hat_NC2_OF_arr)))
        self.W1_NC2_OF.update_mean_std(np.log10(np.array(W1_NC2_OF_arr)))
        self.W2_NC2_OF.update_mean_std(np.log10(np.array(W2_NC2_OF_arr)))

        metrics = [self.W1_NC2_ETF] if self.args["use_W1"] else []
        metrics.extend([self.W2_NC2_ETF, self.H_NC2_ETF, self.HA_hat_NC2_ETF])
        for metric in metrics:
            ax[1, 2].plot(self.x, metric.get_means(), label=metric.label)
            ax[1, 2].fill_between(
                self.x,
                metric.get_means() - metric.get_stds(),
                metric.get_means() + metric.get_stds(),
                alpha=0.2,
                interpolate=True,
            )

        metrics = [self.W1_NC2_OF] if self.args["use_W1"] else []
        metrics.extend([self.W2_NC2_OF, self.H_NC2_OF, self.HA_hat_NC2_OF])
        for metric in metrics:
            ax[1, 2].plot(self.x, metric.get_means(), linestyle="dashed", label=metric.label)
            ax[1, 2].fill_between(
                self.x,
                metric.get_means() - metric.get_stds(),
                metric.get_means() + metric.get_stds(),
                alpha=0.2,
                interpolate=True,
            )

        ax[1, 2].grid(True)
        _ = ax[1, 2].set(xlabel=r"$iter\%{}$".format(nc_interval), ylabel="$NC_2$ (log10 scale)", title="$NC_2$")
        ax[1, 2].legend(fontsize=30)
        return ax

    def plot_NC3(self, ax, W_1, W_2, H_array, A_hat_array, labels_array, nc_interval):

        # temporary arrays
        # NC3 ETF
        W1_H_NC3_ETF_arr = []
        W2_HA_hat_NC3_ETF_arr = []
        W1H_W2HA_hat_NC3_ETF_arr = []
        # NC3 OF
        W1_H_NC3_OF_arr = []
        W2_HA_hat_NC3_OF_arr = []
        W1H_W2HA_hat_NC3_OF_arr = []
        # plain alignment
        W1_H_NC3_arr = []
        W2_HA_hat_NC3_arr = []


        # NC3 ETF Alignment
        for H, A_hat, labels in zip(H_array, A_hat_array, labels_array):
            W1_H_ETF_alignment = self.compute_W_H_ETF_relation(W=W_1, feat=H, labels=labels)
            W1_H_NC3_ETF_arr.append(W1_H_ETF_alignment.detach().cpu().numpy())

            W2_HA_hat_ETF_alignment = self.compute_W_H_ETF_relation(W=W_2, feat=H@A_hat, labels=labels)
            W2_HA_hat_NC3_ETF_arr.append(W2_HA_hat_ETF_alignment.detach().cpu().numpy())

            # Z = W_1H + W_2HA_hat
            Z = W_1 @ H + W_2 @ H @ A_hat
            dummy_W = torch.eye(W_1.shape[0]).type(torch.double).to(self.args["device"])
            Z_ETF_alignment = self.compute_W_H_ETF_relation(W=dummy_W, feat=Z, labels=labels)
            W1H_W2HA_hat_NC3_ETF_arr.append(Z_ETF_alignment.detach().cpu().numpy())

            # NC3 OF Alignment
            W1_H_OF_alignment = self.compute_W_H_OF_relation(W=W_1, feat=H, labels=labels)
            W1_H_NC3_OF_arr.append(W1_H_OF_alignment.detach().cpu().numpy())
            W2_HA_hat_OF_alignment = self.compute_W_H_OF_relation(W=W_2, feat=H@A_hat, labels=labels)
            W2_HA_hat_NC3_OF_arr.append(W2_HA_hat_OF_alignment.detach().cpu().numpy())

            # Z = W_1H + W_2HA_hat
            Z = W_1 @ H + W_2 @ H @ A_hat
            dummy_W = torch.eye(W_1.shape[0]).type(torch.double).to(self.args["device"])
            Z_OF_alignment = self.compute_W_H_OF_relation(W=dummy_W, feat=Z, labels=labels)
            W1H_W2HA_hat_NC3_OF_arr.append(Z_OF_alignment.detach().cpu().numpy())

            # Weights and features alignment
            W1_H_alignment = self.compute_W_H_alignment(W=W_1, feat=H, labels=labels)
            W1_H_NC3_arr.append(W1_H_alignment.detach().cpu().numpy())
            W2_HA_hat_alignment = self.compute_W_H_alignment(W=W_2, feat=H@A_hat, labels=labels)
            W2_HA_hat_NC3_arr.append(W2_HA_hat_alignment.detach().cpu().numpy())


        # NC3 ETF
        self.W1_H_NC3_ETF.update_mean_std(np.log10(np.array(W1_H_NC3_ETF_arr)))
        self.W2_HA_hat_NC3_ETF.update_mean_std(np.log10(np.array(W2_HA_hat_NC3_ETF_arr)))
        self.W1H_W2HA_hat_NC3_ETF.update_mean_std(np.log10(np.array(W1H_W2HA_hat_NC3_ETF_arr)))
        # NC3 OF
        self.W1_H_NC3_OF.update_mean_std(np.log10(np.array(W1_H_NC3_OF_arr)))
        self.W2_HA_hat_NC3_OF.update_mean_std(np.log10(np.array(W2_HA_hat_NC3_OF_arr)))
        self.W1H_W2HA_hat_NC3_OF.update_mean_std(np.log10(np.array(W1H_W2HA_hat_NC3_OF_arr)))
        # plain alignment
        self.W1_H_NC3.update_mean_std(np.log10(np.array(W1_H_NC3_arr)))
        self.W2_HA_hat_NC3.update_mean_std(np.log10(np.array(W2_HA_hat_NC3_arr)))

        metrics = [self.W1_H_NC3] if self.args["use_W1"] else []
        metrics.append(self.W2_HA_hat_NC3)
        if self.args["use_W1"]: metrics.append(self.W1_H_NC3_ETF)
        metrics.extend([self.W2_HA_hat_NC3_ETF, self.W1H_W2HA_hat_NC3_ETF])

        for metric in metrics:
            ax[1, 3].plot(self.x, metric.get_means(), label=metric.label)
            ax[1, 3].fill_between(
                self.x,
                metric.get_means() - metric.get_stds(),
                metric.get_means() + metric.get_stds(),
                alpha=0.2,
                interpolate=True,
            )

        metrics = [self.W1_H_NC3_OF] if self.args["use_W1"] else []
        metrics.extend([self.W2_HA_hat_NC3_OF, self.W1H_W2HA_hat_NC3_OF])
        for metric in metrics:
            ax[1, 3].plot(self.x, metric.get_means(), linestyle="dashed", label=metric.label)
            ax[1, 3].fill_between(
                self.x,
                metric.get_means() - metric.get_stds(),
                metric.get_means() + metric.get_stds(),
                alpha=0.2,
                interpolate=True,
            )

        ax[1, 3].grid(True)
        _ = ax[1, 3].set(xlabel=r"$iter\%{}$".format(nc_interval), ylabel="$NC_3$ (log10 scale)", title="$NC_3$")
        ax[1, 3].legend(fontsize=30)
        return ax


    @torch.no_grad()
    def compute_metrics(self, H_array, A_hat_array, W_1, W_2, labels_array, iter,
                        train_loss_array, train_accuracy_array, filename, nc_interval):

        fig, ax = plt.subplots(2, 4, figsize=(50, 25))

        print("plotting train loss")
        ax = self.plot_train_loss(ax=ax, train_loss_array=train_loss_array, nc_interval=nc_interval)
        print("plotting train accuracy")
        ax = self.plot_train_accuracy(ax=ax, train_accuracy_array=train_accuracy_array, nc_interval=nc_interval)
        print("plotting NC1 metrics for H")
        ax = self.plot_NC1_H(ax=ax, H_array=H_array, labels_array=labels_array, nc_interval=nc_interval)
        print("plotting NC1 metrics for HA_hat")
        ax = self.plot_NC1_HA_hat(ax=ax, H_array=H_array, A_hat_array=A_hat_array, labels_array=labels_array, nc_interval=nc_interval)


        print("plotting NC1 SNR")
        ax = self.plot_NC1_SNR(ax=ax, W_1=W_1, W_2=W_2, H_array=H_array,
                            A_hat_array=A_hat_array, labels_array=labels_array,
                            nc_interval=nc_interval)
        print("plotting fro norms")
        ax = self.plot_fro_norms(ax=ax, W_1=W_1, W_2=W_2, H_array=H_array, A_hat_array=A_hat_array,
                                nc_interval=nc_interval)

        print("plotting NC2 metrics")
        ax = self.plot_NC2(ax=ax, W_1=W_1, W_2=W_2, H_array=H_array, A_hat_array=A_hat_array,
                            labels_array=labels_array, nc_interval=nc_interval)


        print("plotting NC3 metrics")
        ax = self.plot_NC3(ax=ax, W_1=W_1, W_2=W_2, H_array=H_array, A_hat_array=A_hat_array,
                            labels_array=labels_array, nc_interval=nc_interval)

        fig.tight_layout()
        plt.savefig(filename)
        plt.clf()
        plt.close()


    def _compute_metrics(self, H, A_hat, W_1, W_2, labels, iter, train_loss, train_accuracy, filename, nc_interval):

        fig, ax = plt.subplots(2, 4, figsize=(50, 25))

        with torch.no_grad():

            self.train_loss.append(train_loss)
            ax[0, 0].plot(np.array(self.train_loss))
            ax[0, 0].grid(True)
            _ = ax[0, 0].set(xlabel=r"$iter\%{}$".format(nc_interval), ylabel="loss", title="Train loss")

            self.train_accuracy.append(train_accuracy)
            ax[0, 1].plot(np.array(self.train_accuracy))
            ax[0, 1].grid(True)
            _ = ax[0, 1].set(xlabel=r"$iter\%{}$".format(nc_interval), ylabel="accuracy", title="Train accuracy")

            feat = H
            S_W_trace, S_B_trace, nc1_type1, nc1_type2 = self.get_nc1(feat=feat, labels=labels)

            if nc1_type1 <= 0:
                print("nc1_type1 is less than 0 : ", nc1_type1)
            self.H_covariance_traces.append(
                (   np.log10(S_W_trace.detach().cpu().numpy()),
                    np.log10(S_B_trace.detach().cpu().numpy()),
                    np.log10(nc1_type1.detach().cpu().numpy()),
                    np.log10(nc1_type2.detach().cpu().numpy())
                )
            )

            ax[0, 2].plot(np.array(self.H_covariance_traces))
            ax[0, 2].grid(True)
            _ = ax[0, 2].set(
                xlabel=r"$iter\%{}$".format(nc_interval),
                ylabel="$NC_1$ (log10 scale)",
                title="$NC_1$ of H"
            )
            ax[0, 2].legend(labels=["$Tr(S_W)$", "$Tr(S_B)$", "$Tr(S_WS_B^{-1})$", "$Tr(S_W)/Tr(S_B)$"], fontsize=30)

            feat = H @ A_hat
            S_W_trace, S_B_trace, collapse_metric_type1, collapse_metric_type2 = self.get_nc1(feat=feat, labels=labels)

            self.HA_hat_covariance_traces.append(
                (   np.log10(S_W_trace.detach().cpu().numpy()),
                    np.log10(S_B_trace.detach().cpu().numpy()),
                    np.log10(collapse_metric_type1.detach().cpu().numpy()),
                    np.log10(collapse_metric_type2.detach().cpu().numpy())
                )
            )

            ax[0, 3].plot(np.array(self.HA_hat_covariance_traces))
            ax[0, 3].grid(True)
            _ = ax[0, 3].set(
                xlabel=r"$iter\%{}$".format(nc_interval),
                ylabel="$NC_1$ (log10 scale)",
                title="$NC_1$ of $H\hat{{A}}$",
            )
            ax[0, 3].legend(labels=["$Tr(S_W)$", "$Tr(S_B)$", "$Tr(S_WS_B^{-1})$", "$Tr(S_W)/Tr(S_B)$"], fontsize=30)

            res = self.get_W_feat_NC1_SNR(feat=H, labels=labels, W=W_1)
            self.W1H_NC1_SNR.append(res.detach().cpu().numpy())
            res = self.get_W_feat_NC1_SNR(feat=H@A_hat, labels=labels, W=W_2)
            self.W2HA_hat_NC1_SNR.append(res.detach().cpu().numpy())

            if self.args["use_W1"]:
                ax[1, 0].plot(np.log10(np.array(self.W1H_NC1_SNR)), label="$W_1H$")
            ax[1, 0].plot(np.log10(np.array(self.W2HA_hat_NC1_SNR)), label="$W_2H\hat{A}$")
            ax[1, 0].grid(True)
            _ = ax[1, 0].set(xlabel=r"$iter\%{}$".format(nc_interval), ylabel="SNR (log10 scale)", title="$NC_1$ SNR")
            ax[1, 0].legend(fontsize=30)

            W1_fro_norm = torch.norm(W_1, p="fro").detach().cpu().numpy()
            self.W1_frobenius_norms.append(W1_fro_norm)
            W2_fro_norm = torch.norm(W_2, p="fro").detach().cpu().numpy()
            self.W2_frobenius_norms.append(W2_fro_norm)
            H_fro_norm = torch.norm(H, p="fro").detach().cpu().numpy()
            self.H_frobenius_norms.append(H_fro_norm)
            HA_hat_fro_norm = torch.norm(H@A_hat, p="fro").detach().cpu().numpy()
            self.HA_hat_frobenius_norms.append(HA_hat_fro_norm)

            if self.args["use_W1"]:
                ax[1, 1].plot(np.array(self.W1_frobenius_norms), label="$||W_1||_F$")
            ax[1, 1].plot(np.array(self.W2_frobenius_norms), label="$||W_2||_F$")
            ax[1, 1].plot(np.array(self.H_frobenius_norms), label="$||H||_F$")
            ax[1, 1].plot(np.array(self.HA_hat_frobenius_norms), label="$||H\hat{A}||_F$")
            ax[1, 1].grid(True)
            _ = ax[1, 1].set(xlabel=r"$iter\%{}$".format(nc_interval), ylabel="$||.||_F$", title="Frobenius norms")
            ax[1, 1].legend(fontsize=30)

            # NC2 ETF Alignment
            W1_ETF_alignment = self.get_weights_or_feat_ETF_relation(M=W_1).detach().cpu().numpy()
            self.W1_NC2_ETF.append(W1_ETF_alignment)

            W2_ETF_alignment = self.get_weights_or_feat_ETF_relation(M=W_2).detach().cpu().numpy()
            self.W2_NC2_ETF.append(W2_ETF_alignment)

            H_class_means = scatter(H, labels.type(torch.int64), dim=1, reduce="mean")
            # transpose is needed to have shape[0] = C
            H_class_means = H_class_means.t()
            H_class_means_ETF_alignment = self.get_weights_or_feat_ETF_relation(M=H_class_means).detach().cpu().numpy()
            self.H_NC2_ETF.append(H_class_means_ETF_alignment)

            HA_hat_class_means = scatter(H@A_hat, labels.type(torch.int64), dim=1, reduce="mean")
             # transpose is needed to have feat.shape[0] = C
            HA_hat_class_means = HA_hat_class_means.t()
            HA_hat_class_means_ETF_alignment = self.get_weights_or_feat_ETF_relation(M=HA_hat_class_means).detach().cpu().numpy()
            self.HA_hat_NC2_ETF.append(HA_hat_class_means_ETF_alignment)

            # NC2 OF Alignment
            W1_OF_alignment = self.get_weights_or_feat_OF_relation(M=W_1).detach().cpu().numpy()
            self.W1_NC2_OF.append(W1_OF_alignment)

            W2_OF_alignment = self.get_weights_or_feat_OF_relation(M=W_2).detach().cpu().numpy()
            self.W2_NC2_OF.append(W2_OF_alignment)

            # no need to subtract global mean to compute alignment with OF
            H_class_means = scatter(H, labels.type(torch.int64), dim=1, reduce="mean")
            # transpose is needed to have feat.shape[0] = C
            H_class_means = H_class_means.t()
            H_class_means_OF_alignment = self.get_weights_or_feat_OF_relation(M=H_class_means).detach().cpu().numpy()
            self.H_NC2_OF.append(H_class_means_OF_alignment)

            # no need to subtract global mean to compute alignment with OF
            HA_hat_class_means = scatter(H@A_hat, labels.type(torch.int64), dim=1, reduce="mean")
            # transpose is needed to have feat.shape[0] = C
            HA_hat_class_means = HA_hat_class_means.t()
            HA_hat_class_means_OF_alignment = self.get_weights_or_feat_OF_relation(M=HA_hat_class_means).detach().cpu().numpy()
            self.HA_hat_NC2_OF.append(HA_hat_class_means_OF_alignment)

            if self.args["use_W1"]:
                ax[1, 2].plot(np.log10(np.array(self.W1_NC2_ETF)), label="$W_1$ (ETF)")
            ax[1, 2].plot(np.log10(np.array(self.W2_NC2_ETF)), label="$W_2$ (ETF)")
            ax[1, 2].plot(np.log10(np.array(self.H_NC2_ETF)), label="$H$ (ETF)")
            ax[1, 2].plot(np.log10(np.array(self.HA_hat_NC2_ETF)), label="$H\hat{A}$ (ETF)")

            if self.args["use_W1"]:
                ax[1, 2].plot(np.log10(np.array(self.W1_NC2_OF)), linestyle="dashed", label="$W_1$ (OF)")
            ax[1, 2].plot(np.log10(np.array(self.W2_NC2_OF)), linestyle="dashed", label="$W_2$ (OF)")
            ax[1, 2].plot(np.log10(np.array(self.H_NC2_OF)), linestyle="dashed", label="$H$ (OF)")
            ax[1, 2].plot(np.log10(np.array(self.HA_hat_NC2_OF)), linestyle="dashed", label="$H\hat{A}$ (OF)")

            ax[1, 2].grid(True)
            _ = ax[1, 2].set(xlabel=r"$iter\%{}$".format(nc_interval), ylabel="$NC_2$ (log10 scale)", title="$NC_2$")
            ax[1, 2].legend(fontsize=30)

            # NC3 ETF Alignment
            W1_H_ETF_alignment = self.compute_W_H_ETF_relation(W=W_1, feat=H, labels=labels).detach().cpu().numpy()
            self.W1_H_NC3_ETF.append(W1_H_ETF_alignment)
            W2_HA_hat_ETF_alignment = self.compute_W_H_ETF_relation(W=W_2, feat=H@A_hat, labels=labels).detach().cpu().numpy()
            self.W2_HA_hat_NC3_ETF.append(W2_HA_hat_ETF_alignment)
            # Z = W_1H + W_2HA_hat
            Z = W_1 @ H + W_2 @ H @ A_hat
            dummy_W = torch.eye(W_1.shape[0]).type(torch.double).to(self.args["device"])
            Z_ETF_alignment = self.compute_W_H_ETF_relation(W=dummy_W, feat=Z, labels=labels).detach().cpu().numpy()
            self.W1H_W2HA_hat_NC3_ETF.append(Z_ETF_alignment)

            # NC3 OF Alignment
            W1_H_OF_alignment = self.compute_W_H_OF_relation(W=W_1, feat=H, labels=labels).detach().cpu().numpy()
            self.W1_H_NC3_OF.append(W1_H_OF_alignment)
            W2_HA_hat_OF_alignment = self.compute_W_H_OF_relation(W=W_2, feat=H@A_hat, labels=labels).detach().cpu().numpy()
            self.W2_HA_hat_NC3_OF.append(W2_HA_hat_OF_alignment)
            # Z = W_1H + W_2HA_hat
            Z = W_1 @ H + W_2 @ H @ A_hat
            dummy_W = torch.eye(W_1.shape[0]).type(torch.double).to(self.args["device"])
            Z_OF_alignment = self.compute_W_H_OF_relation(W=dummy_W, feat=Z, labels=labels).detach().cpu().numpy()
            self.W1H_W2HA_hat_NC3_OF.append(Z_OF_alignment)

            # Weights and features alignment
            W1_H_alignment = self.compute_W_H_alignment(W=W_1, feat=H, labels=labels).detach().cpu().numpy()
            self.W1_H_NC3.append(W1_H_alignment)
            W2_HA_hat_alignment = self.compute_W_H_alignment(W=W_2, feat=H@A_hat, labels=labels).detach().cpu().numpy()
            self.W2_HA_hat_NC3.append(W2_HA_hat_alignment)

            if self.args["use_W1"]:
                ax[1, 3].plot(np.log10(np.array(self.W1_H_NC3)), label="$(W_1, H)$")
            ax[1, 3].plot(np.log10(np.array(self.W2_HA_hat_NC3)), label="$(W_2, H\hat{A})$")

            if self.args["use_W1"]:
                ax[1, 3].plot(np.log10(np.array(self.W1_H_NC3_ETF)), label="$(W_1H, ETF)$")
            ax[1, 3].plot(np.log10(np.array(self.W2_HA_hat_NC3_ETF)), label="$(W_2H\hat{A}, ETF)$")
            ax[1, 3].plot(np.log10(np.array(self.W1H_W2HA_hat_NC3_ETF)), label="$(Z, ETF)$")

            if self.args["use_W1"]:
                ax[1, 3].plot(np.log10(np.array(self.W1_H_NC3_OF)), linestyle="dashed", label="$(W_1H, OF)$")
            ax[1, 3].plot(np.log10(np.array(self.W2_HA_hat_NC3_OF)), linestyle="dashed", label="$(W_2H\hat{A}, OF)$")
            ax[1, 3].plot(np.log10(np.array(self.W1H_W2HA_hat_NC3_OF)), linestyle="dashed", label="$(Z, OF)$")

            ax[1, 3].grid(True)
            _ = ax[1, 3].set(xlabel=r"$iter\%{}$".format(nc_interval), ylabel="$NC_3$ (log10 scale)", title="$NC_3$")
            ax[1, 3].legend(fontsize=30)

        fig.tight_layout()
        plt.savefig(filename)
        plt.clf()
        plt.close()

    @staticmethod
    def prepare_animation(image_filenames, animation_filename):
        images = []
        for idx, image_filename in enumerate(image_filenames):
            images.append(imageio.imread(image_filename))
            if idx != len(image_filenames)-1:
                os.remove(image_filename)
        imageio.mimsave(animation_filename, images, fps=5)

