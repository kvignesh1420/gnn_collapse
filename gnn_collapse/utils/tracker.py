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

class GUFMMetricTracker:
    def __init__(self, args) -> None:
        self.args = args
        # training stats
        self.train_loss = []
        self.train_accuracy = []
        # NC1 traces
        self.H_covariance_traces = []
        self.HA_hat_covariance_traces = []
        # NC1 SNR
        self.W1H_NC1_SNR = []
        self.W2HA_hat_NC1_SNR = []
        # norms
        self.frobenius_norms = []
        # NC2 ETF
        self.H_NC2_ETF = []
        self.HA_hat_NC2_ETF = []
        self.W1_NC2_ETF = []
        self.W2_NC2_ETF = []
        # NC2 OF
        self.H_NC2_OF = []
        self.HA_hat_NC2_OF = []
        self.W1_NC2_OF = []
        self.W2_NC2_OF = []
        # NC3 ETF
        self.W1_H_NC3_ETF = []
        self.W2_HA_hat_NC3_ETF = []
        self.W1H_W2HA_hat_NC3_ETF = []
        # NC3 OF
        self.W1_H_NC3_OF = []
        self.W2_HA_hat_NC3_OF = []
        self.W1H_W2HA_hat_NC3_OF = []
        # plain alignment
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
            print("FEAT shape ", feat.shape)
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

            res = torch.norm(Wz - sub, p='fro')
        return res

    def get_weights_or_feat_OF_relation(self, M):
        with torch.no_grad():
            K = M.shape[0]
            assert K == self.args["C"]
            MMT = torch.mm(M, M.T)
            MMT /= torch.norm(MMT, p='fro')
            sub = torch.eye(K)/np.sqrt(K)
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
            res = torch.norm(Wz - sub, p='fro')
        return res

    def compute_W_H_alignment(self, W, feat, labels):
        with torch.no_grad():
            class_means = scatter(feat, labels.type(torch.int64), dim=1, reduce="mean")
            # global_mean = torch.mean(class_means, dim=1).unsqueeze(-1)
            z = class_means
            res = torch.norm(W/torch.norm(W) - z.t()/torch.norm(z), p='fro')
        return res


    def compute_metrics(self, H, A_hat, W_1, W_2, labels, epoch, train_loss, train_accuracy, filename, nc_interval):

        fig, ax = plt.subplots(2, 4, figsize=(50, 25))

        with torch.no_grad():

            self.train_loss.append(train_loss)
            ax[0, 0].plot(np.array(self.train_loss))
            ax[0, 0].grid(True)
            _ = ax[0, 0].set(xlabel=r"$epoch\%{}$".format(nc_interval), ylabel="loss", title="Train loss")

            self.train_accuracy.append(train_accuracy)
            ax[0, 1].plot(np.array(self.train_accuracy))
            ax[0, 1].grid(True)
            _ = ax[0, 1].set(xlabel=r"$epoch\%{}$".format(nc_interval), ylabel="accuracy", title="Train accuracy")

            feat = H
            S_W_trace, S_B_trace, nc1_type1, nc1_type2 = self.get_nc1(feat=feat, labels=labels)

            if nc1_type1 <= 0:
                print("nc1_type1 is less than 0 : ", nc1_type1)
            self.H_covariance_traces.append(
                (   np.log10(S_W_trace.detach().numpy()),
                    np.log10(S_B_trace.detach().numpy()),
                    np.log10(nc1_type1.detach().numpy()),
                    np.log10(nc1_type2.detach().numpy())
                )
            )

            ax[0, 2].plot(np.array(self.H_covariance_traces))
            ax[0, 2].grid(True)
            _ = ax[0, 2].set(
                xlabel=r"$epoch\%{}$".format(nc_interval),
                ylabel="$NC_1$ (log10 scale)",
                title="$NC_1$ of H"
            )
            ax[0, 2].legend(labels=["$Tr(S_W)$", "$Tr(S_B)$", "$Tr(S_WS_B^{-1})$", "$Tr(S_W)/Tr(S_B)$"], fontsize=30)

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
            ax[0, 3].grid(True)
            _ = ax[0, 3].set(
                xlabel=r"$epoch\%{}$".format(nc_interval),
                ylabel="$NC_1$ (log10 scale)",
                title="$NC_1$ of $H\hat{{A}}$",
            )
            ax[0, 3].legend(labels=["$Tr(S_W)$", "$Tr(S_B)$", "$Tr(S_WS_B^{-1})$", "$Tr(S_W)/Tr(S_B)$"], fontsize=30)

            res = self.get_W_feat_NC1_SNR(feat=H, labels=labels, W=W_1)
            self.W1H_NC1_SNR.append(res.detach().numpy())
            res = self.get_W_feat_NC1_SNR(feat=H@A_hat, labels=labels, W=W_2)
            self.W2HA_hat_NC1_SNR.append(res.detach().numpy())

            ax[1, 0].plot(np.log10(np.array(list(zip(self.W1H_NC1_SNR, self.W2HA_hat_NC1_SNR)))))
            ax[1, 0].grid(True)
            _ = ax[1, 0].set(xlabel=r"$epoch\%{}$".format(nc_interval), ylabel="SNR (log10 scale)", title="$NC_1$ SNR")
            ax[1, 0].legend(labels=["$W_1H$", "$W_2H\hat{A}$"], fontsize=30)

            W1_fro_norm = torch.norm(W_1, p="fro").detach().numpy()
            W2_fro_norm = torch.norm(W_2, p="fro").detach().numpy()
            H_fro_norm = torch.norm(H, p="fro").detach().numpy()
            HA_hat_fro_norm = torch.norm(H@A_hat, p="fro").detach().numpy()
            norms_data = (W1_fro_norm, W2_fro_norm, H_fro_norm, HA_hat_fro_norm)
            self.frobenius_norms.append(norms_data)

            ax[1, 1].plot(np.array(self.frobenius_norms))
            ax[1, 1].grid(True)
            _ = ax[1, 1].set(xlabel=r"$epoch\%{}$".format(nc_interval), ylabel="$||.||_F$", title="Frobenius norms")
            ax[1, 1].legend(labels=["$||W_1||_F$", "$||W_2||_F$", "$||H||_F$", "$||H\hat{A}||_F$"], fontsize=30)

            # NC2 ETF Alignment
            W1_ETF_alignment = self.get_weights_or_feat_ETF_relation(M=W_1).detach().numpy()
            self.W1_NC2_ETF.append(W1_ETF_alignment)

            W2_ETF_alignment = self.get_weights_or_feat_ETF_relation(M=W_2).detach().numpy()
            self.W2_NC2_ETF.append(W2_ETF_alignment)

            H_class_means = scatter(H, labels.type(torch.int64), dim=1, reduce="mean")
            # transpose is needed to have shape[0] = C
            H_class_means = H_class_means.t()
            H_class_means_ETF_alignment = self.get_weights_or_feat_ETF_relation(M=H_class_means).detach().numpy()
            self.H_NC2_ETF.append(H_class_means_ETF_alignment)

            HA_hat_class_means = scatter(H@A_hat, labels.type(torch.int64), dim=1, reduce="mean")
             # transpose is needed to have feat.shape[0] = C
            HA_hat_class_means = HA_hat_class_means.t()
            HA_hat_class_means_ETF_alignment = self.get_weights_or_feat_ETF_relation(M=HA_hat_class_means).detach().numpy()
            self.HA_hat_NC2_ETF.append(HA_hat_class_means_ETF_alignment)

            # NC2 OF Alignment
            W1_OF_alignment = self.get_weights_or_feat_OF_relation(M=W_1).detach().numpy()
            self.W1_NC2_OF.append(W1_OF_alignment)

            W2_OF_alignment = self.get_weights_or_feat_OF_relation(M=W_2).detach().numpy()
            self.W2_NC2_OF.append(W2_OF_alignment)

            # no need to subtract global mean to compute alignment with OF
            H_class_means = scatter(H, labels.type(torch.int64), dim=1, reduce="mean")
            # transpose is needed to have feat.shape[0] = C
            H_class_means = H_class_means.t()
            H_class_means_OF_alignment = self.get_weights_or_feat_OF_relation(M=H_class_means).detach().numpy()
            self.H_NC2_OF.append(H_class_means_OF_alignment)

            # no need to subtract global mean to compute alignment with OF
            HA_hat_class_means = scatter(H@A_hat, labels.type(torch.int64), dim=1, reduce="mean")
            # transpose is needed to have feat.shape[0] = C
            HA_hat_class_means = HA_hat_class_means.t()
            HA_hat_class_means_OF_alignment = self.get_weights_or_feat_OF_relation(M=HA_hat_class_means).detach().numpy()
            self.HA_hat_NC2_OF.append(HA_hat_class_means_OF_alignment)

            ax[1, 2].plot(np.log10(np.array(list(zip(
                self.W1_NC2_ETF, self.W2_NC2_ETF, self.H_NC2_ETF, self.HA_hat_NC2_ETF,
                self.W1_NC2_OF, self.W2_NC2_OF, self.H_NC2_OF, self.HA_hat_NC2_OF
            )))))
            ax[1, 2].grid(True)
            _ = ax[1, 2].set(xlabel=r"$epoch\%{}$".format(nc_interval), ylabel="$NC_2$ (log10 scale)", title="$NC_2$")
            ax[1, 2].legend(labels=[
                "$W_1$ (ETF)", "$W_2$ (ETF)", "$H$ (ETF)",  "$H\hat{A}$ (ETF)",
                "$W_1$ (OF)", "$W_2$ (OF)", "$H$ (OF)",  "$H\hat{A}$ (OF)",
            ], fontsize=30)

            # NC3 ETF Alignment
            W1_H_ETF_alignment = self.compute_W_H_ETF_relation(W=W_1, feat=H, labels=labels).detach().numpy()
            self.W1_H_NC3_ETF.append(W1_H_ETF_alignment)
            W2_HA_hat_ETF_alignment = self.compute_W_H_ETF_relation(W=W_2, feat=H@A_hat, labels=labels).detach().numpy()
            self.W2_HA_hat_NC3_ETF.append(W2_HA_hat_ETF_alignment)
            # Z = W_1H + W_2HA_hat
            Z = W_1 @ H + W_2 @ H @ A_hat
            dummy_W = torch.eye(W_1.shape[0]).type(torch.double)
            Z_ETF_alignment = self.compute_W_H_ETF_relation(W=dummy_W, feat=Z, labels=labels).detach().numpy()
            self.W1H_W2HA_hat_NC3_ETF.append(Z_ETF_alignment)

            # NC3 OF Alignment
            W1_H_OF_alignment = self.compute_W_H_OF_relation(W=W_1, feat=H, labels=labels).detach().numpy()
            self.W1_H_NC3_OF.append(W1_H_OF_alignment)
            W2_HA_hat_OF_alignment = self.compute_W_H_OF_relation(W=W_2, feat=H@A_hat, labels=labels).detach().numpy()
            self.W2_HA_hat_NC3_OF.append(W2_HA_hat_OF_alignment)
            # Z = W_1H + W_2HA_hat
            Z = W_1 @ H + W_2 @ H @ A_hat
            dummy_W = torch.eye(W_1.shape[0]).type(torch.double)
            Z_OF_alignment = self.compute_W_H_OF_relation(W=dummy_W, feat=Z, labels=labels).detach().numpy()
            self.W1H_W2HA_hat_NC3_OF.append(Z_OF_alignment)

            # Weights and features alignment
            W1_H_alignment = self.compute_W_H_alignment(W=W_1, feat=H, labels=labels).detach().numpy()
            self.W1_H_NC3.append(W1_H_alignment)
            W2_HA_hat_alignment = self.compute_W_H_alignment(W=W_2, feat=H@A_hat, labels=labels).detach().numpy()
            self.W2_HA_hat_NC3.append(W2_HA_hat_alignment)

            ax[1, 3].plot(np.log10(np.array(list(zip(
                self.W1_H_NC3, self.W2_HA_hat_NC3,
                self.W1_H_NC3_ETF, self.W2_HA_hat_NC3_ETF, self.W1H_W2HA_hat_NC3_ETF,
                self.W1_H_NC3_OF, self.W2_HA_hat_NC3_OF, self.W1H_W2HA_hat_NC3_OF
            )))))
            ax[1, 3].grid(True)
            _ = ax[1, 3].set(xlabel=r"$epoch\%{}$".format(nc_interval), ylabel="$NC_3$ (log10 scale)", title="$NC_3$")
            ax[1, 3].legend(labels=[
                "$(W_1, H)$", "$(W_2, H\hat{A})$",
                "$(W_1H, ETF)$", "$(W_2H\hat{A}, ETF)$", "$(Z, ETF)$",
                "$(W_1H, OF)$", "$(W_2H\hat{A}, OF)$", "$(Z, OF)$"
            ], fontsize=30)

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

