"""
Online training of SBM graphs
"""

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
import matplotlib.pyplot as plt


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

    def train_loop(self, dataloader, model, args):
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
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        for step_idx, data in tqdm(enumerate(dataloader)):
            device = args["device"]
            data = data.to(device)
            pred = model(data)
            loss = compute_loss_multiclass(pred=pred, labels=data.y, k=args["k"])
            model.zero_grad()
            loss.backward()
            optimizer.step()
            acc = compute_accuracy_multiclass(pred=pred, labels=data.y, k=args["k"])
            losses.append(loss.detach().cpu().numpy())
            accuracies.append(acc)

            if self.track_nc and step_idx%args["nc_interval"] == 0:
                # self.feature_snapshots.append(self.features)
                self.nc1_snapshots.append(
                    compute_nc1(features=self.features, labels=data.y)
                )
                if args["k"] == 2:
                    plot_penultimate_layer_features(features=self.features, labels=data.y, args=args)
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
            loss = compute_loss_multiclass(pred=pred, labels=data.y, k=args["k"])
            acc = compute_accuracy_multiclass(pred=pred, labels=data.y, k=args["k"])
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
            loss = compute_loss_multiclass(pred=pred, labels=data.y, k=args["k"])
            model.zero_grad()
            loss.backward()
            optimizer.step()
            acc = compute_accuracy_multiclass(pred=pred, labels=data.y, k=args["k"])
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
            loss = compute_loss_multiclass(pred=pred, labels=data.y, k=args["k"])
            acc = compute_accuracy_multiclass(pred=pred, labels=data.y, k=args["k"])
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