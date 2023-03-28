"""
Online training of SBM graphs
"""
import os
import numpy as np
from tqdm import tqdm
import torch
from gnn_collapse.utils.losses import compute_loss_multiclass
from gnn_collapse.utils.losses import compute_accuracy_multiclass
from gnn_collapse.utils.node_properties import plot_penultimate_layer_features
from gnn_collapse.utils.node_properties import plot_feature_mean_distances
from gnn_collapse.utils.node_properties import plot_feature_mean_angles
from gnn_collapse.utils.node_properties import compute_nc1
from gnn_collapse.utils.node_properties import plot_nc1
from gnn_collapse.utils.node_properties import plot_single_graph_nc1
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 25, 'lines.linewidth': 5, 'axes.titlepad': 20, "figure.figsize": (15, 15)})
import imageio
from sklearn.linear_model import LogisticRegression


class OnlineRunner:
    def __init__(self, track_nc=False):
        self.track_nc = track_nc
        self.features = {}

    def probe_features(self, name):
        def hook(model, inp, out):
            self.features[name] = out.detach()
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
        return model

    def run(self, train_dataloader, test_dataloader, model, args):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        model = self.train_loop(
            dataloader=train_dataloader,
            model=model,
            optimizer=optimizer,
            args=args
        )
        self.test_loop(
            dataloader=test_dataloader,
            model=model,
            args=args
        )
        self.track_belief_histograms(dataloader=test_dataloader, model=model, args=args)

    def train_loop(self, dataloader, model, optimizer, args):
        """Training loop for sbm node classification

        Args:
            dataloader: The dataloader of SBM graphs
            model: baseline or gnn models to train
            optimizer: The torch optimizer to update weights, ex: Adam, SGD.
            args: settings for training
        """

        if self.track_nc:
            model = self.assign_hooks(model=model)
            # This list stores the snapshots of self.features over time
            self.nc1_snapshots = []

        model.train()
        losses = []
        accuracies = []
        num_epochs = 1
        if args["num_train_graphs"] == 1:
            num_epochs = 1000
        for epoch in range(num_epochs):
            for step_idx, data in tqdm(enumerate(dataloader)):
                device = args["device"]
                data = data.to(device)
                pred = model(data)
                loss = compute_loss_multiclass(type=args["loss_type"], pred=pred, labels=data.y, C=args["C"])
                model.zero_grad()
                loss.backward()
                optimizer.step()
                acc = compute_accuracy_multiclass(pred=pred, labels=data.y, C=args["C"])
                losses.append(loss.detach().cpu().numpy())
                accuracies.append(acc)

                if self.track_nc and (epoch*len(dataloader) + step_idx)%args["nc_interval"] == 0:
                    # self.feature_snapshots.append(self.features)
                    self.nc1_snapshots.append(
                        compute_nc1(features=self.features, labels=data.y)
                    )
                    # if args["C"] == 2:
                    #     plot_penultimate_layer_features(features=self.features, labels=data.y, args=args)
                        # plot_feature_mean_distances(features=self.features, labels=data.y, args=args)
                    # Adj = to_dense_adj(data.edge_index)[0]
                    # spectral_matching(Adj=Adj, features=self.features, labels=data.y)

        print('Avg train loss', np.mean(losses))
        print('Avg train acc', np.mean(accuracies))
        print('Std train acc', np.std(accuracies))

        with open(args["results_file"], 'a') as f:
            f.write("""Avg train loss: {}\n Avg train acc: {}\n Std train acc: {}\n""".format(
                np.mean(losses), np.mean(accuracies), np.std(accuracies)))

        plt.plot(losses)
        plt.xlabel("iter")
        plt.ylabel("loss")
        plt.savefig("{}train_losses.png".format(args["vis_dir"]))
        plt.clf()
        plt.plot(accuracies)
        plt.xlabel("iter")
        plt.ylabel("acc(overlap)")
        plt.savefig("{}train_acc.png".format(args["vis_dir"]))
        plt.clf()

        if self.track_nc:
            print("track_nc enabled!")
            print("Length of feature_snapshots list: {}".format(len(self.nc1_snapshots)))
            print("Number of layers tracked: {}".format(len(self.nc1_snapshots[0])))
            plot_nc1(nc1_snapshots=self.nc1_snapshots, args=args)
        return model

    def test_loop(self, dataloader, model, args):
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
            pred = model(data)
            loss = compute_loss_multiclass(type=args["loss_type"], pred=pred, labels=data.y, C=args["C"])
            acc = compute_accuracy_multiclass(pred=pred, labels=data.y, C=args["C"])
            losses.append(loss.detach().cpu().numpy())
            accuracies.append(acc)

        print ('Avg test loss', np.mean(losses))
        print ('Avg test acc', np.mean(accuracies))
        print ('Std test acc', np.std(accuracies))

        with open(args["results_file"], 'a') as f:
            f.write("""Avg test loss: {}\n Avg test acc: {}\n Std test acc: {}\n""".format(
                np.mean(losses), np.mean(accuracies), np.std(accuracies)))

        plt.plot(losses)
        plt.xlabel("iter")
        plt.ylabel("loss")
        plt.savefig("{}test_losses.png".format(args["vis_dir"]))
        plt.clf()
        plt.plot(accuracies)
        plt.xlabel("iter")
        plt.ylabel("acc(overlap)")
        plt.savefig("{}test_acc.png".format(args["vis_dir"]))
        plt.clf()

    def prepare_animation(self, image_filenames, animation_filename):
        images = []
        for image_filename in image_filenames:
            images.append(imageio.imread(image_filename))
            os.remove(image_filename)
        imageio.mimsave(animation_filename, images, fps=10)

    def track_belief_histograms(self, dataloader, model, args):
        """
        Track the beliefs of the classes using logistic regression on
        features of every layer. Currently, relevant for k=2.
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
                filename = "{}belief_hist_layer_{}.png".format(args["vis_dir"], layer_name)
                filenames.append(filename)
                plt.savefig(filename)
                plt.clf()
                plt.close()

            animation_filename = "{}belief_hist.mp4".format(args["vis_dir"])
            self.prepare_animation(image_filenames=filenames, animation_filename=animation_filename)
            nc1_snapshots = []
            nc1_snapshots.append(compute_nc1(features=self.features, labels=data.y))
            plot_single_graph_nc1(nc1_snapshots=nc1_snapshots, args=args)
            break

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
            plot_nc1(nc1_snapshots=self.nc1_snapshots, args=args, layer_idx=layer_idx)

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