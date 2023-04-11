"""
Spectral method baselines
"""
import os
import imageio
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 25, 'lines.linewidth': 5, 'axes.titlepad': 20, "figure.figsize": (15, 15)})
import seaborn as sns

from gnn_collapse.utils.node_properties import compute_nc1
from gnn_collapse.utils.node_properties import plot_single_graph_nc1


class BetheHessian:
    def __init__(self, Adj):
        self.A = Adj

    def compute(self, r=None):
        degrees = torch.sum(self.A, dim=1)
        if r is None:
            r = torch.sqrt(torch.mean(degrees))
        self.BH = (r**2 - 1)*torch.eye(degrees.shape[0]) - r*self.A + torch.diag(degrees)
        return self.BH

    def _cluster(self, C):
        """
        Cluster using the exact eigen vectors computed using svd
        and not via power iteration.
        """
        evals, evecs = torch.linalg.eig(self.BH)
        evals = evals.type(torch.float)
        evecs = evecs.type(torch.float)
        sorted_eval_indices = torch.argsort(evals)
        # consider the C smallest evals as BH is a variant of the graph laplacian
        k_eval_indices = sorted_eval_indices[:C]
        X = evecs[:, k_eval_indices]
        # Cluster the features to get the labels
        cluster_model = KMeans(n_clusters=C, random_state=0)
        cluster_model.fit(X=X)
        pred = cluster_model.labels_
        cluster_means = cluster_model.cluster_centers_
        return {
            "pred": F.one_hot(torch.from_numpy(pred).type(torch.int64), num_classes=C),
            "centroids": torch.from_numpy(cluster_means)
        }

    def pi_fiedler_pred(self, labels, args, enable_tracking=False):
        """
        Fiedler vector computation using power iteration
        NOTE: Use this only for 2 community classification.
        """
        num_iters = args["num_layers"]
        evals, evecs = torch.linalg.eig(self.BH)
        evals = evals.type(torch.float)
        evecs = evecs.type(torch.float)
        sorted_eval_indices = torch.argsort(evals)
        # consider the smallest eigen vector of self.BH
        k_eval_indices = sorted_eval_indices[0]
        v = evecs[:, k_eval_indices]

        BH_hat = torch.norm(self.BH, p=2)*torch.eye(self.BH.shape[0]) - self.BH
        w = torch.randn_like(v)
        filenames = []
        self.features = {}
        for i in range(num_iters):
            y = BH_hat @ w
            w = y - torch.dot(y, v)*v
            w = w/torch.norm(w, p=2)
            if enable_tracking:
                # pred = torch.sign(y)
                pred = y
                self.features[i] = torch.Tensor(pred).unsqueeze(-1)
                # print(self.features[i].shape)
                plt.hist(pred[labels==0], alpha=0.8, histtype='bar', edgecolor='black', label="c=0")
                plt.hist(pred[labels==1], alpha=0.8, histtype='bar', edgecolor='black', label="c=1")
                plt.title("feat of approx fiedler vector. Iter: {}".format(i))
                plt.legend()
                filename = "{}belief_hist_pi_{}.png".format(args["vis_dir"], i)
                filenames.append(filename)
                plt.savefig(filename)
                plt.close()
                plt.clf()

        if enable_tracking:
            gt_fiedler = evecs[:, sorted_eval_indices[1]]
            self.plot_evals_and_fiedler(args=args, evals=evals, sorted_eval_indices=sorted_eval_indices, gt_fiedler=gt_fiedler)
            animation_filename = "{}belief_hist.mp4".format(args["vis_dir"])
            self.prepare_animation(image_filenames=filenames, animation_filename=animation_filename)
            nc1_snapshots = []
            nc1_snapshots.append(compute_nc1(features=self.features, labels=labels))
            plot_single_graph_nc1(nc1_snapshots=nc1_snapshots, args=args)

        pred = (y < 0).type(torch.int64)
        pred = F.one_hot(pred, num_classes=2)
        return pred

    def prepare_animation(self, image_filenames, animation_filename):
        images = []
        for image_filename in image_filenames:
            images.append(imageio.imread(image_filename))
            os.remove(image_filename)
        imageio.mimsave(animation_filename, images)

    def plot_evals_and_fiedler(self, args, evals, sorted_eval_indices, gt_fiedler):
        with open(args["results_file"], 'a') as f:
            f.write("""smallest K+1 eigenvals: {}\n""".format(evals[sorted_eval_indices[:args["C"]+1]]))
        hist_plot = sns.histplot(evals, bins="auto")
        hist_plot.set(title="eigen values")
        fig = hist_plot.get_figure()
        fig.savefig("{}ev_hist.png".format(args["vis_dir"]))
        plt.clf()
        fiedler_pred = gt_fiedler
        hist_plot = sns.histplot(fiedler_pred, bins="auto")
        hist_plot.set(title="features of fiedler vector")
        fig = hist_plot.get_figure()
        fig.savefig("{}belief_hist_gt.png".format(args["vis_dir"]))
        plt.clf()


class NormalizedLaplacian:
    def __init__(self, Adj):
        self.A = Adj

    def compute(self):
        degrees = torch.sum(self.A, dim=1)
        D = torch.diag(degrees)
        D_inv = torch.nan_to_num(torch.diag(1/degrees), nan=0.0, posinf=0.0, neginf=0.0)
        D_inv_sqrt = torch.sqrt(D_inv)
        self.L = torch.eye(D.shape[0]) - D_inv_sqrt @ self.A @ D_inv_sqrt
        return self.L

    def _cluster(self, C):
        """
        Cluster using the exact eigen vectors computed using svd
        and not via power iteration.
        """
        evals, evecs = torch.linalg.eig(self.L)
        evals = evals.type(torch.float)
        evecs = evecs.type(torch.float)
        sorted_eval_indices = torch.argsort(evals)
        k_eval_indices = sorted_eval_indices[:C]
        X = evecs[:, k_eval_indices]
        cluster_model = KMeans(n_clusters=C, random_state=0)
        cluster_model.fit(X=X)
        pred = cluster_model.labels_
        cluster_means = cluster_model.cluster_centers_
        return {
            "pred": F.one_hot(torch.from_numpy(pred).type(torch.int64), num_classes=C),
            "centroids": torch.from_numpy(cluster_means)
        }

    def pi_fiedler_pred(self, labels, args, enable_tracking=False):
        """
        Fiedler vector computation using power iteration
        NOTE: Use this only for 2 community classification.
        """
        num_iters = args["num_layers"]
        evals, evecs = torch.linalg.eig(self.L)
        evals = evals.type(torch.float)
        evecs = evecs.type(torch.float)
        sorted_eval_indices = torch.argsort(evals)
        k_eval_indices = sorted_eval_indices[0]
        v = evecs[:, k_eval_indices]

        L_hat = torch.norm(self.L, p=2)*torch.eye(self.L.shape[0]) - self.L
        w = torch.randn_like(v)
        filenames = []
        self.features = {}
        for i in range(num_iters):
            y = L_hat @ w
            w = y - torch.dot(y, v)*v
            w = w/torch.norm(w, p=2)
            if enable_tracking:
                # pred = torch.sign(y)
                pred = y
                self.features[i] = torch.Tensor(pred).unsqueeze(-1)
                plt.hist(pred[labels==0], alpha=0.8, histtype='bar', edgecolor='black', label="c=0")
                plt.hist(pred[labels==1], alpha=0.8, histtype='bar', edgecolor='black', label="c=1")
                plt.title("feat of approx fiedler vector. Iter: {}".format(i))
                plt.legend()
                filename = "{}belief_hist_pi_{}.png".format(args["vis_dir"], i)
                filenames.append(filename)
                plt.savefig(filename)
                plt.close()
                plt.clf()

        if enable_tracking:
            gt_fiedler = evecs[:, sorted_eval_indices[1]]
            self.plot_evals_and_fiedler(args=args, evals=evals, sorted_eval_indices=sorted_eval_indices, gt_fiedler=gt_fiedler)
            animation_filename = "{}belief_hist.mp4".format(args["vis_dir"])
            self.prepare_animation(image_filenames=filenames, animation_filename=animation_filename)
            nc1_snapshots = []
            nc1_snapshots.append(compute_nc1(features=self.features, labels=labels))
            plot_single_graph_nc1(nc1_snapshots=nc1_snapshots, args=args)

        pred = (y < 0).type(torch.int64)
        pred = F.one_hot(pred, num_classes=2)
        return pred

    def prepare_animation(self, image_filenames, animation_filename):
        images = []
        for image_filename in image_filenames:
            images.append(imageio.imread(image_filename))
            os.remove(image_filename)
        imageio.mimsave(animation_filename, images)

    def plot_evals_and_fiedler(self, args, evals, sorted_eval_indices, gt_fiedler):
        with open(args["results_file"], 'a') as f:
            f.write("""smallest K+1 eigenvals: {}\n""".format(evals[sorted_eval_indices[:args["C"]+1]]))
        hist_plot = sns.histplot(evals, bins="auto")
        hist_plot.set(title="eigen values")
        fig = hist_plot.get_figure()
        fig.savefig("{}ev_hist.png".format(args["vis_dir"]))
        plt.clf()
        fiedler_pred = gt_fiedler
        hist_plot = sns.histplot(fiedler_pred, bins="auto")
        hist_plot.set(title="features of fiedler vector")
        fig = hist_plot.get_figure()
        fig.savefig("{}belief_hist_gt.png".format(args["vis_dir"]))
        plt.clf()