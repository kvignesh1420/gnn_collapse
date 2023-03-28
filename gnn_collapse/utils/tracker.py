import os
import imageio
import numpy as np
import torch
from torch_scatter import scatter
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 25, 'lines.linewidth': 5, 'axes.titlepad': 20, "figure.figsize": (15, 15)})

class GUFMMetricTracker:
    def __init__(self) -> None:
        self.train_loss = []
        self.train_accuracy = []
        self.H_covariance_traces = []
        self.HA_hat_covariance_traces = []
        self.W1H_NC1_SNR = []
        self.W2HA_hat_NC1_SNR = []
        self.frobenius_norms = []
        self.H_NC2 = []
        self.HA_hat_NC2 = []
        self.W1_NC2 = []
        self.W2_NC2 = []
        self.W1_H_NC3 = []
        self.W2_HA_hat_NC3 = []
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
            class_means = scatter(feat, labels.type(torch.int64), dim=1, reduce="mean")
            expanded_class_means = torch.index_select(class_means, dim=1, index=labels)
            z = feat - expanded_class_means
            # signal = W_2 @ expanded_class_means
            # noise = W_2 @ z
            # signal_res = torch.norm(signal)
            # noise_res = torch.norm(noise)
            # self.y.append((signal_res/noise_res).detach().numpy())
            num_nodes = z.shape[1]
            S_W = 0
            for i in range(num_nodes):
                S_W += z[:, i].unsqueeze(1) @ z[:, i].unsqueeze(0)
            S_W /= num_nodes
            # print(class_means.shape)
            global_mean = torch.mean(class_means, dim=1).unsqueeze(-1)
            # print(global_mean.shape)
            z = class_means - global_mean
            num_classes = class_means.shape[1]
            S_B = 0
            for i in range(num_classes):
                S_B += z[:, i].unsqueeze(1) @ z[:, i].unsqueeze(0)
            S_B /= num_classes
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
            MMT = torch.mm(M, M.T)
            MMT /= torch.norm(MMT, p='fro')

            sub = (torch.eye(K) - 1 / K * torch.ones((K, K))) / pow(K - 1, 0.5)
            ETF_metric = torch.norm(MMT - sub, p='fro')
        return ETF_metric

    def compute_W_H_ETF_relation(self, W, feat, labels):
        """Adapted from: https://github.com/tding1/Neural-Collapse/blob/main/validate_NC.py"""
        with torch.no_grad():
            class_means = scatter(feat, labels.type(torch.int64), dim=1, reduce="mean")
            global_mean = torch.mean(class_means, dim=1).unsqueeze(-1)
            z = class_means - global_mean
            Wz = torch.mm(W, z)
            Wz /= torch.norm(Wz, p='fro')
            K = W.shape[0]
            sub = 1 / pow(K - 1, 0.5) * (torch.eye(K) - 1 / K * torch.ones((K, K)))

            res = torch.norm(Wz - sub, p='fro')
        return res


    def compute_metrics(self, H, A_hat, W_1, W_2, labels, epoch, train_loss, train_accuracy, filename, nc_interval):

        fig, ax = plt.subplots(2, 4, figsize=(50, 20))

        self.train_loss.append(train_loss)
        ax[0, 0].plot(np.array(self.train_loss))
        _ = ax[0, 0].set(xlabel="epoch%{}".format(nc_interval), ylabel="loss", title="train loss until epoch: {}".format(epoch))

        self.train_accuracy.append(train_accuracy)
        ax[0, 1].plot(np.array(self.train_accuracy))
        _ = ax[0, 1].set(xlabel="epoch%{}".format(nc_interval), ylabel="accuracy", title="train accuracy until epoch: {}".format(epoch))

        with torch.no_grad():
            feat = H
            S_W_trace, S_B_trace, nc1_type1, nc1_type2 = self.get_nc1(feat=feat, labels=labels)

            self.H_covariance_traces.append(
                (   np.log10(S_W_trace.detach().numpy()),
                    np.log10(S_B_trace.detach().numpy()),
                    np.log10(nc1_type1.detach().numpy()),
                    np.log10(nc1_type2.detach().numpy())
                )
            )

        ax[0, 2].plot(np.array(self.H_covariance_traces))
        _ = ax[0, 2].set(
            xlabel="epoch%{}".format(nc_interval),
            ylabel="$NC_1 (\log10 scale)$",
            title="$NC_1$ of H until epoch: {}".format(epoch)
        )
        ax[0, 2].legend(labels=["$Tr(S_W)$", "$Tr(S_B)$", "$Tr(S_WS_B^{-1})$", "$Tr(S_W)/Tr(S_B)$"])

        with torch.no_grad():
            feat = H @ A_hat
            S_W_trace, S_B_trace, collapse_metric_type1, collapse_metric_type2 = self.get_nc1(feat=feat, labels=labels)

            self.HA_hat_covariance_traces.append(
                (   np.log10(S_W_trace.detach().numpy()),
                    np.log10(S_B_trace.detach().numpy()),
                    np.log10(collapse_metric_type1.detach().numpy()),
                    np.log10(collapse_metric_type2.detach().numpy())
                )
            )

        ax[0, 3].plot(np.array(self.HA_hat_covariance_traces))
        _ = ax[0, 3].set(
            xlabel="epoch%{}".format(nc_interval),
            ylabel="$NC_1$ (log10 scale)",
            title="$NC_1$ of $H\hat{{A}}$ until epoch: {}".format(epoch),
        )
        ax[0, 3].legend(labels=["$Tr(S_W)$", "$Tr(S_B)$", "$Tr(S_WS_B^{-1})$", "$Tr(S_W)/Tr(S_B)$"])

        with torch.no_grad():
            res = self.get_W_feat_NC1_SNR(feat=H, labels=labels, W=W_1)
            self.W1H_NC1_SNR.append(res.detach().numpy())
            res = self.get_W_feat_NC1_SNR(feat=H@A_hat, labels=labels, W=W_2)
            self.W2HA_hat_NC1_SNR.append(res.detach().numpy())

        ax[1, 0].plot(np.array(list(zip(self.W1H_NC1_SNR, self.W2HA_hat_NC1_SNR))))
        _ = ax[1, 0].set(xlabel="epoch%{}".format(nc_interval), ylabel="SNR", title="$NC_1$ SNR until epoch: {}".format(epoch))
        ax[1, 0].legend(labels=["$W_1H$", "$W_2H\hat{A}$"])

        with torch.no_grad():
            W1_fro_norm = torch.norm(W_1, p="fro").detach().numpy()
            W2_fro_norm = torch.norm(W_2, p="fro").detach().numpy()
            H_fro_norm = torch.norm(H, p="fro").detach().numpy()
            HA_hat_fro_norm = torch.norm(H@A_hat, p="fro").detach().numpy()
            norms_data = (W1_fro_norm, W2_fro_norm, H_fro_norm, HA_hat_fro_norm)
            self.frobenius_norms.append(norms_data)

        ax[1, 1].plot(np.array(self.frobenius_norms))
        _ = ax[1, 1].set(xlabel="epoch%{}".format(nc_interval), ylabel="frobenius norm", title="$||.||_F$ until epoch: {}".format(epoch))
        ax[1, 1].legend(labels=["$||W_1||_F$", "$||W_2||_F$", "$||H||_F$", "$||H\hat{A}||_F$"])

        with torch.no_grad():
            W1_ETF_alignment = self.get_weights_or_feat_ETF_relation(M=W_1).detach().numpy()
            self.W1_NC2.append(W1_ETF_alignment)

            W2_ETF_alignment = self.get_weights_or_feat_ETF_relation(M=W_2).detach().numpy()
            self.W2_NC2.append(W2_ETF_alignment)

            H_class_means = scatter(H, labels.type(torch.int64), dim=1, reduce="mean")
            H_class_means_ETF_alignment = self.get_weights_or_feat_ETF_relation(M=H_class_means).detach().numpy()
            self.H_NC2.append(H_class_means_ETF_alignment)

            HA_hat_class_means = scatter(H@A_hat, labels.type(torch.int64), dim=1, reduce="mean")
            HA_hat_class_means_ETF_alignment = self.get_weights_or_feat_ETF_relation(M=HA_hat_class_means).detach().numpy()
            self.HA_hat_NC2.append(HA_hat_class_means_ETF_alignment)

        ax[1, 2].plot(np.array(list(zip(self.W1_NC2, self.W2_NC2, self.H_NC2, self.HA_hat_NC2))))
        _ = ax[1, 2].set(xlabel="epoch%{}".format(nc_interval), ylabel="$NC_2$", title="$NC_2$ until epoch: {}".format(epoch))
        ax[1, 2].legend(labels=["$W_1$", "$W_2$", "$H$", "$H\hat{A}$"])

        with torch.no_grad():
            W1_H_alignment = self.compute_W_H_ETF_relation(W=W_1, feat=H, labels=labels).detach().numpy()
            self.W1_H_NC3.append(W1_H_alignment)
            W2_HA_hat_alignment = self.compute_W_H_ETF_relation(W=W_2, feat=H@A_hat, labels=labels)
            self.W2_HA_hat_NC3.append(W2_HA_hat_alignment)

        ax[1, 3].plot(np.array(list(zip(self.W1_H_NC3, self.W2_HA_hat_NC3))))
        _ = ax[1, 3].set(xlabel="epoch%{}".format(nc_interval), ylabel="$NC_3$", title="$NC_3$ until epoch: {}".format(epoch))
        ax[1, 3].legend(labels=["$(W_1, H)$", "$(W_2, H\hat{A})$"])

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

