"""
Online training of SBM graphs
"""
import os
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.utils import to_dense_adj

from gnn_collapse.utils.losses import compute_loss_multiclass
from gnn_collapse.utils.losses import compute_accuracy_multiclass
from gnn_collapse.utils.node_properties import plot_penultimate_layer_features
from gnn_collapse.utils.node_properties import plot_feature_mean_distances
from gnn_collapse.utils.node_properties import plot_feature_mean_angles
from gnn_collapse.utils.node_properties import compute_nc1
from gnn_collapse.utils.node_properties import plot_nc1_heatmap
from gnn_collapse.utils.node_properties import plot_test_graphs_nc1
from gnn_collapse.utils.weight_properties import WeightTracker
from gnn_collapse.utils.tracker import GUFMMetricTracker
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 40,
    'lines.linewidth': 5,
    'axes.titlepad': 20,
    'axes.linewidth': 2,
    'figure.figsize': (15, 15)
})
import imageio
from sklearn.linear_model import LogisticRegression


class OnlineRunner:
    def __init__(self, args):
        self.args = args
        self.features = {}
        self.non_linear_features = {}
        self.normalized_features = {}
        self.metric_tracker = GUFMMetricTracker(args=self.args)

    def probe_features(self, name):
        def hook(model, inp, out):
            self.features[name] = out.detach()
        return hook

    def probe_non_linear_features(self, name):
        def hook(model, inp, out):
            self.non_linear_features[name] = out.detach()
        return hook

    def probe_normalized_features(self, name):
        def hook(model, inp, out):
            self.normalized_features[name] = out.detach()
        return hook

    def assign_hooks(self, model):
        if model.name == "mlp":
            for i in range(len(model.fc_layers)):
                layer_name = i
                model.fc_layers[i].register_forward_hook(self.probe_features(name=layer_name))
        else:
            for i in range(len(model.conv_layers)):
                layer_name = i
                model.conv_layers[i].register_forward_hook(self.probe_features(name=layer_name))
        for i in range(len(model.non_linear_layers)):
            layer_name = i
            model.non_linear_layers[i].register_forward_hook(self.probe_non_linear_features(name=layer_name))
        for i in range(len(model.normalize_layers)):
            layer_name = i
            model.normalize_layers[i].register_forward_hook(self.probe_normalized_features(name=layer_name))
        return model

    def run(self, train_dataloader, test_dataloader, model):

        if self.args["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.args["lr"],
                momentum=self.args["sgd_momentum"],
                weight_decay=self.args["weight_decay"]
            )
        elif self.args["optimizer"] == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.args["lr"],
                weight_decay=self.args["weight_decay"]
            )

        # assign hooks
        if self.args["track_nc"]:
            model = self.assign_hooks(model=model)

        # get stats before training

        # self.track_belief_histograms(dataloader=test_dataloader, model=model, epoch=0)
        model = self.train_loop(
            dataloader=train_dataloader,
            model=model,
            optimizer=optimizer,
        )
        self.test_loop(
            dataloader=test_dataloader,
            model=model,
        )
        # get stats after training
        # self.track_belief_histograms(dataloader=test_dataloader, model=model, epoch=self.args["num_epochs"])
        self.track_test_graphs_intermediate_nc(dataloader=test_dataloader, model=model, epoch=self.args["num_epochs"])

    def train_loop(self, dataloader, model, optimizer):
        """Training loop for sbm node classification

        Args:
            dataloader: The dataloader of SBM graphs
            model: baseline or gnn models to train
            optimizer: The torch optimizer to update weights, ex: Adam, SGD.
        """

        if self.args["track_nc"]:
            # This list stores the snapshots of nc1 metrics over time
            self.features_nc1_snapshots = []
            self.non_linear_features_nc1_snapshots = []
            self.normalized_features_nc1_snapshots = []

        model.train()
        losses = []
        accuracies = []
        filenames = []
        max_iters = self.args["num_epochs"]*len(dataloader)
        for epoch in range(self.args["num_epochs"]):
            for step_idx, data in tqdm(enumerate(dataloader)):
                device = self.args["device"]
                data = data.to(device)
                pred = model(data)
                loss = compute_loss_multiclass(type=self.args["loss_type"], pred=pred, labels=data.y, C=self.args["C"])
                model.zero_grad()
                loss.backward()
                optimizer.step()
                acc = compute_accuracy_multiclass(pred=pred, labels=data.y, C=self.args["C"])
                losses.append(loss.detach().cpu().numpy())
                accuracies.append(acc)

                iter_count =  epoch*len(dataloader) + step_idx
                if self.args["track_nc"] and (iter_count%self.args["nc_interval"] == 0 or iter_count + 1 == max_iters):
                    filename = "{}/nc_tracker_{}.png".format(self.args["vis_dir"], iter_count)
                    filenames.append(filename)
                    last_layer_idx = max(list(self.normalized_features.keys()))
                    H = self.normalized_features[last_layer_idx]
                    H = H.t().type(torch.double)
                    W2 = torch.clone(model.final_layer.lin_rel.weight).type(torch.double)
                    if self.args["use_W1"]:
                        W1 = torch.clone(model.final_layer.lin_root.weight).type(torch.double)
                    else:
                        W1 = torch.zeros_like(W2).type(torch.double)

                    A = to_dense_adj(data.edge_index)[0].to(self.args["device"])
                    D_inv = torch.diag(1/torch.sum(A, 1)).to(self.args["device"])
                    A_hat = (D_inv @ A).type(torch.double).to(self.args["device"])

                    print("Shape of H : {}  W1 : {}  W2: {}".format(H.shape, W1.shape, W2.shape))

                    self.metric_tracker.compute_metrics(
                        H=H,
                        A_hat=A_hat,
                        W_1=W1,
                        W_2=W2,
                        labels=data.y,
                        iter=iter_count,
                        train_loss=loss.detach().cpu().numpy(),
                        train_accuracy=acc,
                        filename=filename,
                        nc_interval=self.args["nc_interval"])

        print('Avg train loss', np.mean(losses))
        print('Avg train acc', np.mean(accuracies))
        print('Std train acc', np.std(accuracies))

        # save model
        print("Saving the model")
        torch.save(model.state_dict(), "{}model.pt".format(self.args["results_dir"]))

        with open(self.args["results_file"], 'a') as f:
            f.write("""Avg train loss: {}\n Avg train acc: {}\n Std train acc: {}\n""".format(
                np.mean(losses), np.mean(accuracies), np.std(accuracies)))

        if self.args["track_nc"]:
            print("track_nc enabled!")
            animation_filename = "{}/nc_tracker.mp4".format(self.args["vis_dir"])
            self.metric_tracker.prepare_animation(image_filenames=filenames, animation_filename=animation_filename)

        #     print("Length of feature_snapshots list: {}".format(len(self.features_nc1_snapshots)))
        #     print("Number of layers tracked: {}".format(len(self.features_nc1_snapshots[0])))
        #     plot_nc1_heatmap(nc1_snapshots=self.features_nc1_snapshots, args=self.args, layer_type="conv")
        #     plot_nc1_heatmap(nc1_snapshots=self.non_linear_features_nc1_snapshots, args=self.args, layer_type="non_linear")
        #     plot_nc1_heatmap(nc1_snapshots=self.normalized_features_nc1_snapshots, args=self.args, layer_type="normalize")
        return model

    def test_loop(self, dataloader, model):
        """Testing loop for sbm node classification

        Args:
            dataloader: The dataloader of SBM graphs
            model: baseline or gnn models to train
        """
        # model.train()
        losses = []
        accuracies = []
        for step_idx, data in tqdm(enumerate(dataloader)):
            device = self.args["device"]
            data = data.to(device)
            pred = model(data)
            loss = compute_loss_multiclass(type=self.args["loss_type"], pred=pred, labels=data.y, C=self.args["C"])
            acc = compute_accuracy_multiclass(pred=pred, labels=data.y, C=self.args["C"])
            losses.append(loss.detach().cpu().numpy())
            accuracies.append(acc)

        print ('Avg test loss', np.mean(losses))
        print ('Avg test acc', np.mean(accuracies))
        print ('Std test acc', np.std(accuracies))

        with open(self.args["results_file"], 'a') as f:
            f.write("""Avg test loss: {}\n Avg test acc: {}\n Std test acc: {}\n""".format(
                np.mean(losses), np.mean(accuracies), np.std(accuracies)))

        plt.grid(True)
        plt.plot(losses)
        plt.xlabel("iter")
        plt.ylabel("loss")
        plt.savefig("{}test_losses.png".format(self.args["vis_dir"]))
        plt.clf()

        plt.grid(True)
        plt.plot(accuracies)
        plt.xlabel("iter")
        plt.ylabel("overlap")
        plt.savefig("{}test_acc.png".format(self.args["vis_dir"]))
        plt.clf()

    def prepare_animation(self, image_filenames, animation_filename):
        images = []
        for image_filename in image_filenames:
            images.append(imageio.imread(image_filename))
            os.remove(image_filename)
        imageio.mimsave(animation_filename, images, fps=10)

    def track_belief_histograms(self, dataloader, model, epoch):
        """
        Track the beliefs of the classes using logistic regression on
        features of every layer on a test graph. Currently, relevant for k=2.
        """
        for data in dataloader:
            # capture the features
            _ = model(data)
            filenames = []
            for layer_name, feat in self.features.items():
                print("layer name: {} feat shape: {}".format(layer_name, feat.shape))
                X = feat.cpu().numpy()
                Y = data.y.cpu().numpy()
                clf = LogisticRegression(random_state=0).fit(X=X, y=Y)
                probs = clf.predict_proba(X)
                classes = clf.classes_
                colors = ["blue", "orange"]
                fig, ax = plt.subplots(1,2)
                for i in classes:
                    indices = np.argwhere(Y==i).squeeze(-1)
                    probs_ = probs[indices, :]
                    prob_i = probs_[:, i]
                    ax[i].hist(prob_i, bins=10, label="prob_{}".format(i), color=colors[i])
                    ax[i].set_xlabel("probability")
                    ax[i].set_ylabel("number of nodes")
                    ax[i].set_title("gt: {} num: {}".format(i, probs_.shape[0]))
                    ax[i].legend()
                fig.suptitle("Layer: {}".format(layer_name))
                fig.tight_layout()
                filename = "{}belief_hist_epoch_{}_layer_{}.png".format(self.args["vis_dir"], epoch, layer_name)
                filenames.append(filename)
                plt.savefig(filename)
                plt.clf()
                plt.close()

            animation_filename = "{}belief_hist_epoch_{}.mp4".format(self.args["vis_dir"], epoch)
            self.prepare_animation(image_filenames=filenames, animation_filename=animation_filename)
            break

    def track_test_graphs_intermediate_nc(self, dataloader, model, epoch):
        """
        Track the NC metrics on test graphs after model reaches TPT
        """
        features_nc1_snapshots = []
        non_linear_features_nc1_snapshots = []
        normalized_features_nc1_snapshots = []
        weight_sv_info = []
        print("Tracking NC metrics on test graphs")
        for data in tqdm(dataloader):
            # capture the features
            _ = model(data)
            features_nc1_snapshots.append(
                compute_nc1(features=self.features, labels=data.y)
            )
            non_linear_features_nc1_snapshots.append(
                compute_nc1(features=self.non_linear_features, labels=data.y)
            )
            normalized_features_nc1_snapshots.append(
                compute_nc1(features=self.normalized_features, labels=data.y)
            )
            # weight_tracker = WeightTracker(state_dict=model.state_dict(), args=self.args)
            # weight_tracker.compute_and_plot()
            # weight_sv_info.append(weight_tracker.sv_data)

        plot_test_graphs_nc1(
            features_nc1_snapshots=features_nc1_snapshots,
            non_linear_features_nc1_snapshots=non_linear_features_nc1_snapshots,
            normalized_features_nc1_snapshots=normalized_features_nc1_snapshots,
            weight_sv_info=weight_sv_info,
            args=self.args,
            epoch=epoch
        )


class OnlineIncRunner:
    def __init__(self, track_nc=False):
        self.track_nc = track_nc
        self.features = {}

    def probe_features(self, name):
        def hook(model, inp, out):
            self.features[name] = out.detach()
        return hook

    def assign_hooks(self, model, layer_idx):
        model.conv_layers[layer_idx].register_forward_hook(self.probe_features(name=layer_idx))
        return model

    def run(self, train_dataloader, test_dataloader, model, args):
        for layer_idx in range(len(model.conv_layers)):
            print("Training layer {} with fixed previous layers".format(layer_idx))
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
            model = self.train_loop(
                dataloader=train_dataloader,
                model=model,
                optimizer=optimizer,
                args=args,
                layer_idx=layer_idx
            )
            self.test_loop(
                dataloader=test_dataloader,
                model=model,
                args=args,
                layer_idx=layer_idx
            )


    def train_loop(self, dataloader, model, optimizer, args, layer_idx):
        """Training loop for sbm node classification

        Args:
            dataloader: The dataloader of SBM graphs
            model: baseline or gnn models to train
            optimizer: The torch optimizer to update weights, ex: Adam, SGD.
            args: settings for training
        """
        model.train()

        if self.track_nc:
            model = self.assign_hooks(model=model, layer_idx=layer_idx)
            # This list stores the snapshots of self.features over time
            self.nc1_snapshots = []

        losses = []
        accuracies = []

        for step_idx, data in tqdm(enumerate(dataloader)):
            device = args["device"]
            data = data.to(device)
            pred = model(data, layer_idx)
            loss = compute_loss_multiclass(type=args["loss_type"], pred=pred, labels=data.y, C=args["C"])
            model.zero_grad()
            loss.backward()
            optimizer.step()
            acc = compute_accuracy_multiclass(pred=pred, labels=data.y, C=args["C"])
            losses.append(loss.detach().cpu().numpy())
            accuracies.append(acc)

            if self.track_nc and step_idx%args["nc_interval"] == 0:
                # self.feature_snapshots.append(self.features)
                self.nc1_snapshots.append(
                    compute_nc1(features=self.features, labels=data.y)
                )

        print('Avg train loss', np.mean(losses))
        print('Avg train acc', np.mean(accuracies))
        print('Std train acc', np.std(accuracies))

        with open(args["results_file"], 'a') as f:
            f.write("""Training layer {} Avg train loss: {}\n Avg train acc: {}\n Std train acc: {}\n""".format(
                layer_idx, np.mean(losses), np.mean(accuracies), np.std(accuracies)))

        plt.plot(losses)
        plt.xlabel("iter")
        plt.ylabel("loss")
        plt.savefig("{}train_losses_{}.png".format(args["vis_dir"], layer_idx))
        plt.clf()
        plt.plot(accuracies)
        plt.xlabel("iter")
        plt.ylabel("acc(overlap)")
        plt.savefig("{}train_acc_{}.png".format(args["vis_dir"], layer_idx))
        plt.clf()

        if self.track_nc:
            print("track_nc enabled!")
            print("Length of feature_snapshots list: {}".format(len(self.nc1_snapshots)))
            print("Number of layers tracked: {}".format(len(self.nc1_snapshots[0])))
            plot_nc1_heatmap(nc1_snapshots=self.nc1_snapshots, args=args, layer_idx=layer_idx)

        return model

    def test_loop(self, dataloader, model, args, layer_idx):
        """Testing loop for sbm node classification

        Args:
            dataloader: The dataloader of SBM graphs
            model: baseline or gnn models to train
            args: settings for training
        """
        # model.train()
        losses = []
        accuracies = []
        for step_idx, data in tqdm(enumerate(dataloader)):
            device = args["device"]
            data = data.to(device)
            pred = model(data, layer_idx)
            loss = compute_loss_multiclass(type=args["loss_type"], pred=pred, labels=data.y, C=args["C"])
            acc = compute_accuracy_multiclass(pred=pred, labels=data.y, C=args["C"])
            losses.append(loss.detach().cpu().numpy())
            accuracies.append(acc)

        print ('Avg test loss', np.mean(losses))
        print ('Avg test acc', np.mean(accuracies))
        print ('Std test acc', np.std(accuracies))

        with open(args["results_file"], 'a') as f:
            f.write("""Testing layer {} Avg test loss: {}\n Avg test acc: {}\n Std test acc: {}\n""".format(
                layer_idx, np.mean(losses), np.mean(accuracies), np.std(accuracies)))

        plt.plot(losses)
        plt.xlabel("iter")
        plt.ylabel("loss")
        plt.savefig("{}test_losses_{}.png".format(args["vis_dir"], layer_idx))
        plt.clf()
        plt.plot(accuracies)
        plt.xlabel("iter")
        plt.ylabel("acc(overlap)")
        plt.savefig("{}test_acc_{}.png".format(args["vis_dir"], layer_idx))
        plt.clf()