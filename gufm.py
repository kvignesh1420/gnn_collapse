"""
A graph based unconstrained features model
"""

import os
import json
import argparse
import sys
import time
import pprint
import numpy as np
import torch
from torch_scatter import scatter
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from gnn_collapse.data.sbm import SBM, SBMRegular
from gnn_collapse.utils.losses import compute_loss_multiclass
from gnn_collapse.utils.losses import compute_accuracy_multiclass
from tqdm import tqdm
from gnn_collapse.models.ufm import GUFM
from gnn_collapse.utils.tracker import GUFMMetricTracker


SBM_FACTORY = {
    "sbm": SBM,
    "sbm_reg": SBMRegular
}

def get_run_args():
    parser = argparse.ArgumentParser(description='Arguments for running the experiments')
    parser.add_argument('config_file',  type=str, help='config file for the run')
    parsed_args = parser.parse_args()

    with open(parsed_args.config_file) as f:
        args = json.load(fp=f)
    args["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args)

    if args["model_name"] != "gufm":
        sys.exit("Invalid model_name. Should be 'gufm'")

    if args["train_sbm_type"] not in SBM_FACTORY:
        sys.exit("Invalid train_sbm_type. Should be one of: {}".format(list(SBM_FACTORY.keys())))

    if args["test_sbm_type"] not in SBM_FACTORY:
        sys.exit("Invalid test_sbm_type. Should be one of: {}".format(list(SBM_FACTORY.keys())))

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


def ufm_validate(dataloader, model, C):
    model.eval()
    accuracies = []
    for _, data in tqdm(enumerate(dataloader)):
        A = to_dense_adj(data.edge_index)[0]
        D_inv = torch.diag(1/torch.sum(A, 1))
        A_hat = D_inv@A
        pred = model(A_hat)
        acc = compute_accuracy_multiclass(pred=pred.t(), labels=data.y, C=C)
        accuracies.append(acc)
    print ('Avg test acc', np.mean(accuracies))
    print ('Std test acc', np.std(accuracies))


def ufm_train(train_dataloader, val_dataloader, model, args):
    optimizer = torch.optim.SGD([
        {"params": model.params["W_1"], "weight_decay": 5e-3},
        # {"params": model.params["W_2"], "weight_decay": 5e-1},
        {"params": model.params["H"], "weight_decay": 5e-3},
        ], lr=0.1
    )
    optimizer.zero_grad()
    filenames = []
    tracker = GUFMMetricTracker()
    for epoch in range(num_epochs+1):
        for data in train_dataloader:
            model.train()
            # A = to_dense_adj(data.edge_index)[0]
            # D_inv = torch.diag(1/torch.sum(A, 1))
            # A_hat = D_inv@A
            A_hat = torch.eye(args["N"])
            pred = model(A_hat)
            loss = compute_loss_multiclass(
                type=args["loss_type"],
                pred=pred.t(),
                labels=data.y,
                C=args["C"], 
                permute=False
            )
            loss.backward()
            optimizer.step()
            acc = compute_accuracy_multiclass(torch.clone(pred.t()), data.y, C=args["C"], permute=False)
        if epoch%args["nc_interval"] == 0:
            print("epoch: {} loss: {} acc: {}".format(epoch, loss.detach().cpu().numpy(), acc))
            filename = args["vis_dir"] + "/gufm_tracker_{}.png".format(epoch)
            filenames.append(filename)
            ufm_validate(val_dataloader, model, args["C"])
            tracker.compute_metrics(
                H=torch.clone(model.params['H']),
                A_hat=A_hat,
                W_1=torch.clone(model.params['W_1']),
                W_2=torch.zeros_like(model.params['W_1']),
                labels=data.y,
                epoch=epoch,
                train_loss=loss.detach().cpu().numpy(),
                train_accuracy=acc,
                filename=filename,
                nc_interval=args["nc_interval"])
    animation_filename = args["vis_dir"] + "/gufm_tracker.mp4"
    tracker.prepare_animation(image_filenames=filenames, animation_filename=animation_filename)


if __name__ == "__main__":

    args = get_run_args()
    num_epochs=50000
    train_sbm_dataset = SBM_FACTORY[args["train_sbm_type"]](
        N=args["N"],
        C=args["C"],
        Pr=args["Pr"],
        p=args["p"],
        q=args["q"],
        num_graphs=args["num_train_graphs"],
        feature_strategy=args["feature_strategy"],
        feature_dim=args["input_feature_dim"],
        permute_nodes=False,
        is_training=True
    )
    train_dataloader = DataLoader(dataset=train_sbm_dataset, batch_size=1)

    val_sbm_dataset = SBM_FACTORY[args["test_sbm_type"]](
        N=args["N"],
        C=args["C"],
        Pr=args["Pr"],
        p=args["p"],
        q=args["q"],
        num_graphs=args["num_test_graphs"],
        feature_strategy=args["feature_strategy"],
        feature_dim=args["input_feature_dim"],
        permute_nodes=False,
        is_training=False
    )
    # keep batch size = 1 for consistent measurement of loss and accuracies under
    # permutation of classes.
    val_dataloader = DataLoader(dataset=val_sbm_dataset, batch_size=1)
    model = GUFM(
        in_feature_dim=args["input_feature_dim"],
        out_feature_dim=args["C"],
        num_nodes=args["N"],
        H_xn_gain=args["H_xn_gain"]
    )
    ufm_train(train_dataloader=train_dataloader, val_dataloader=val_dataloader, model=model, args=args)
