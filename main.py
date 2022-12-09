import os
import sys
import time
import json
import argparse
import pprint
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm
from gnn_collapse.data.sbm import SBM
from gnn_collapse.models import factory
from gnn_collapse.models.baselines import BetheHessian
from gnn_collapse.utils.losses import compute_loss_multiclass
from gnn_collapse.utils.losses import compute_accuracy_multiclass
from gnn_collapse.utils.node_properties import plot_penultimate_layer_features
from gnn_collapse.utils.node_properties import plot_feature_mean_distances
from gnn_collapse.utils.node_properties import plot_feature_mean_angles
from gnn_collapse.utils.node_properties import compute_nc1
from gnn_collapse.utils.node_properties import plot_nc1
import matplotlib.pyplot as plt


def spectral_clustering(dataloader, args):
    """clustering based on spectral methods for sbm node classification

    Args:
        dataloader: The dataloader of SBM graphs
        args: settings for training
    """

    accuracies = []
    for step_idx, data in tqdm(enumerate(dataloader)):
        device = args["device"]
        data = data.to(device)
        Adj = to_dense_adj(data.edge_index)[0]
        model = BetheHessian(Adj=Adj)
        model.compute()
        pred = model.pi_fiedler_pred(num_iters=args["num_layers"])
        acc = compute_accuracy_multiclass(pred=pred, labels=data.y, k=args["k"])
        accuracies.append(acc)

    print('Avg test acc', np.mean(accuracies))
    print('Std test acc', np.std(accuracies))
    with open(args["results_file"], 'a') as f:
        f.write("""Avg test acc: {}\n Std test acc: {}\n""".format(
            np.mean(accuracies), np.std(accuracies)))
    plt.plot(accuracies)
    plt.savefig("{}test_acc.png".format(args["vis_dir"]))
    plt.clf()


class Runner:
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

    def online_train_loop(self, dataloader, model, optimizer, args):
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
                    plot_feature_mean_distances(features=self.features, labels=data.y, args=args)
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

    def online_test_loop(self, dataloader, model, args):
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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_run_args():
    parser = argparse.ArgumentParser(description='Arguments for running the experiments')
    parser.add_argument('config_file',  type=str, help='config file for the run')
    parsed_args = parser.parse_args()

    with open(parsed_args.config_file) as f:
        args = json.load(fp=f)
    args["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args)

    if args["model_name"] not in factory:
        sys.exit("Invalid model type. Should be one of: {}".format(list(factory.keys())))

    vis_dir = args["out_dir"] + args["model_name"] + "/" + time.strftime('%Hh_%Mm_%Ss_on_%b_%d_%Y') + "/plots/"
    results_dir = args["out_dir"] + args["model_name"] + "/" + time.strftime('%Hh_%Mm_%Ss_on_%b_%d_%Y') + "/results/"
    results_file = results_dir + "run.txt"
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    args["vis_dir"] = vis_dir
    args["results_file"] = results_file

    with open(results_file, 'a') as f:
        f.write("""CONFIG: \n{}\n""".format(pprint.pformat(args, sort_dicts=False)))

    return args


if __name__ == "__main__":

    args = get_run_args()
    train_sbm_dataset = SBM(
        n=args["n"],
        k=args["k"],
        p=args["p"],
        W=args["W"],
        num_graphs=args["num_train_graphs"],
        feature_strategy=args["feature_strategy"],
        feature_dim=args["input_feature_dim"]
    )
    # keep batch size = 1 for consistent measurement of loss and accuracies under
    # permutation of classes.
    train_dataloader = DataLoader(dataset=train_sbm_dataset, batch_size=1)
    test_sbm_dataset = SBM(
        n=args["n"],
        k=args["k"],
        p=args["p"],
        W=args["W"],
        num_graphs=args["num_test_graphs"],
        feature_strategy=args["feature_strategy"],
        feature_dim=args["input_feature_dim"]
    )
    test_dataloader = DataLoader(dataset=test_sbm_dataset, batch_size=1)

    model_class = factory[args["model_name"]]
    if args["model_name"] not in ["bethe_hessian"]:
        model = model_class(
            input_feature_dim=args["input_feature_dim"],
            hidden_feature_dim=args["hidden_feature_dim"],
            num_classes=args["k"],
            L=args["num_layers"],
            batch_norm=args["batch_norm"]
        ).to(args["device"])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        print("# parameters: ", count_parameters(model=model))
        # NOTE: Batch norm is key for performance, since we are sampling new graphs
        # it is better to unfreeze the batch norm values during testing.
        runner = Runner(track_nc=args["track_nc"])
        runner.online_train_loop(dataloader=train_dataloader, model=model, optimizer=optimizer, args=args)
        runner.online_test_loop(dataloader=test_dataloader, model=model, args=args)
    else:
        spectral_clustering(dataloader=test_dataloader, args=args)
