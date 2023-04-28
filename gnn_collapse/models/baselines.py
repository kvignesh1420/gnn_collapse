"""
Spectral method baselines
"""
import os
import imageio
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 40,
    'lines.linewidth': 5,
    'axes.titlepad': 20,
    'axes.linewidth': 2,
    'figure.figsize': (15, 15)
})


class SpectralOperator:
    """A base class for spectral operator based community detection."""
    def __init__(self, args):
        self.args = args
        self.plot_belief_hist = True
        # placeholder for adjacency matrix

    def compute(self, A):
        raise NotImplementedError

    def pi_fiedler_pred(self, A, labels):
        """
        Fiedler vector computation using power iteration
        NOTE: Use this only for C=2 community classification.
        """
        M = self.compute(A)
        M.requires_grad = False
        # subtract scaled identity and perform projected power iterations
        # to obtain the second largest eigenvector of M_hat (i.e, corresponds to the
        # second smallest eigenvector of M).
        M_hat = torch.norm(M, p=2)*torch.eye(M.shape[0]).to(self.args["device"]) - M
        M_hat = M_hat.to(self.args["device"])

        evals, evecs = torch.linalg.eig(M_hat)
        evals = evals.type(torch.float)
        evecs = evecs.type(torch.float)
        # print("Shape of evals: {} evecs: {}".format(evals.shape, evecs.shape))
        sorted_eval_indices = torch.argsort(evals)
        # consider the eigenvector corresponding to largest eigenvalue of M_hat
        largest_ev_index = sorted_eval_indices[-1]
        v = evecs[:, largest_ev_index]

        # start with random features for approximating the second largest eigenvector
        if self.args["feature_strategy"] == "random_normal":
            w = torch.randn_like(v)
        elif self.args["feature_strategy"] == "degree":
            w = torch.sum(A, 1)
        else:
            sys.exit("invalid strategy")

        filenames = []
        self.features = {}
        num_iters = self.args["num_layers"]
        for i in range(num_iters):
            y = M_hat @ w
            w = y - torch.dot(y, v)*v
            w = w/torch.norm(w, p=2)
            if self.args["track_nc"]:
                # pred = torch.sign(y)
                pred = y
                self.features[i] = torch.Tensor(pred).unsqueeze(-1) # set feature shape to N x 1
                # print(self.features[i].shape)
                if self.plot_belief_hist:
                    plt.grid(True)
                    plt.hist(pred[labels==0].detach().cpu().numpy(), alpha=0.8,
                            histtype='bar', edgecolor='black', label="c=0")
                    plt.hist(pred[labels==1].detach().cpu().numpy(), alpha=0.8,
                            histtype='bar', edgecolor='black', label="c=1")
                    plt.title("feat of approx fiedler vector. Iter: {}".format(i))
                    plt.legend()
                    filename = "{}belief_hist_pi_{}.png".format(self.args["vis_dir"], i)
                    filenames.append(filename)
                    plt.savefig(filename)
                    plt.close()
                    plt.clf()

        if self.args["track_nc"] and self.plot_belief_hist:
            gt_fiedler = evecs[:, sorted_eval_indices[-2]]
            self.plot_evals_and_fiedler(evals=evals, sorted_eval_indices=sorted_eval_indices, gt_fiedler=gt_fiedler)
            animation_filename = "{}belief_hist.mp4".format(self.args["vis_dir"])
            self.prepare_animation(image_filenames=filenames, animation_filename=animation_filename)

        # plot histogram based plots for one test graph
        self.plot_belief_hist = False
        pred = (y < 0).type(torch.int64)
        pred = F.one_hot(pred, num_classes=2)
        return pred, self.features

    def prepare_animation(self, image_filenames, animation_filename):
        images = []
        for image_filename in image_filenames:
            images.append(imageio.imread(image_filename))
            os.remove(image_filename)
        imageio.mimsave(animation_filename, images)

    def plot_evals_and_fiedler(self, evals, sorted_eval_indices, gt_fiedler):

        with open(self.args["results_file"], 'a') as f:
            f.write("""smallest C+1 eigenvals: {}\n""".format(
                evals[sorted_eval_indices[:self.args["C"]+1]])
            )

        plt.grid(True)
        plt.hist(evals.detach().cpu().numpy(), bins="auto")
        plt.title("eigen values")
        plt.savefig("{}ev_hist.png".format(self.args["vis_dir"]))
        plt.clf()

        plt.grid(True)
        plt.hist(gt_fiedler.detach().cpu().numpy(), bins="auto")
        plt.title("features of fiedler vector")
        plt.savefig("{}belief_hist_gt.png".format(self.args["vis_dir"]))
        plt.clf()


class BetheHessian(SpectralOperator):
    def __init__(self, args):
        super().__init__(args)

    def compute(self, A):
        r = self.args["BH_r"]
        degrees = torch.sum(A, dim=1).to(self.args["device"])
        if r is None or r == "":
            r = torch.sqrt(torch.mean(degrees)).to(self.args["device"])
        BH = (r**2 - 1)*torch.eye(degrees.shape[0]).to(self.args["device"]) - \
                r*A + torch.diag(degrees).to(self.args["device"])
        BH = BH.to(self.args["device"])
        return BH


class NormalizedLaplacian(SpectralOperator):
    def __init__(self, args):
        super().__init__(args)

    def compute(self, A):
        degrees = torch.sum(A, dim=1).to(self.args["device"])
        D = torch.diag(degrees).to(self.args["device"])
        D_inv = torch.nan_to_num(torch.diag(1/degrees), nan=0.0, posinf=0.0, neginf=0.0)
        D_inv_sqrt = torch.sqrt(D_inv).to(self.args["device"])
        L = torch.eye(D.shape[0]).to(self.args["device"]) - D_inv_sqrt @ A @ D_inv_sqrt
        return L
