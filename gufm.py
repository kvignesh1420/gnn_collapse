import os
import argparse
import pprint
import hashlib
import json
import sys
from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from gnn_collapse.utils.tracker import GUFMMetricTracker
import time
from gnn_collapse.data.sbm import SBM, SBMRegular


SBM_FACTORY = {
    "sbm": SBM,
    "sbm_reg": SBMRegular
}

def prepare_config_hash(args):
    _string_args = json.dumps(args, sort_keys=True).encode("utf-8")
    parsed_args_hash = hashlib.md5(_string_args).hexdigest()
    return parsed_args_hash

def get_run_args():
    parser = argparse.ArgumentParser(description='Arguments for running the experiments')
    parser.add_argument('config_file',  type=str, help='config file for the run')
    parsed_args = parser.parse_args()

    with open(parsed_args.config_file) as f:
        args = json.load(fp=f)

    # create a unique hash for the model
    config_uuid = prepare_config_hash(args=args)
    args["config_uuid"] = config_uuid
    args["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args)

    if args["model_name"] != "gufm":
        sys.exit("Invalid model_name. Should be 'gufm'")

    if args["train_sbm_type"] not in SBM_FACTORY:
        sys.exit("Invalid train_sbm_type. Should be one of: {}".format(list(SBM_FACTORY.keys())))

    vis_dir = args["out_dir"] + args["model_name"] + "/" + args["config_uuid"] + "/plots/"
    results_dir = args["out_dir"] + args["model_name"] + "/" + args["config_uuid"] + "/results/"
    results_file = results_dir + "run.txt"
    if not os.path.exists(vis_dir):
        print("Vis folder does not exist. Creating one!")
        os.makedirs(vis_dir)
    if not os.path.exists(results_dir):
        print("Resuls folder does not exist. Creating one!")
        os.makedirs(results_dir)
    args["vis_dir"] = vis_dir
    args["results_file"] = results_file

    with open(results_file, 'a') as f:
        f.write("""CONFIG: \n{}\n""".format(pprint.pformat(args, sort_dicts=False)))

    return args


def loss_func(args, W1, W2, H, A_hat, Y, N):
    Z = 0
    if args["use_W1"]:
        Z = W1 @ H
    if args["use_W2"]:
        Z = Z + W2 @ H @ A_hat

    loss_value = 0.5*(1/N)*(  torch.norm(Z - Y) )**2
    if args["use_W1"]: loss_value += 0.5*lambda_W1*torch.norm(W1)**2
    if args["use_W2"]: loss_value += 0.5*lambda_W2*torch.norm(W2)**2
    loss_value += 0.5*lambda_H*torch.norm(H)**2
    return loss_value


@torch.no_grad()
def nc_helper(args, W1, W2, H_array, A_hat_array, labels_array):
    loss_array = []
    acc_array = []

    for step_idx in range(len(A_hat_array)):

        H = H_array[step_idx]
        A_hat = A_hat_array[step_idx]
        labels_gt = labels_array[step_idx]

        Z = 0
        if args["use_W1"]:
            Z += W1 @ H
        if args["use_W2"]:
            Z += W2 @ H @ A_hat

        labels_pred = torch.argmax(Z, axis=0).type(torch.int32)

        loss = loss_func(args=args, W1=W1, W2=W2, H=H, A_hat=A_hat, Y=Y, N=N).type(torch.double)
        acc = torch.mean((labels_pred == labels_gt).type(torch.float))
        # measure accuracy in terms of overlap since we are dealing with community detection
        # however, observe that during TPT, overlap=train_acc=1.
        C = args["C"]
        acc = (acc - 1/C) / (1 - 1/C)
        # print(loss, acc)
        loss_array.append(loss.detach().cpu().numpy())
        acc_array.append(acc.detach().cpu().numpy())

    return loss_array, acc_array


def train_loop(args, W1, W2, H_array, A_hat_array, labels_array):

    tracker = GUFMMetricTracker(args=args)
    filenames = []
    max_iters = args["num_epochs"]*len(A_hat_array)
    for epoch in tqdm(range(args["num_epochs"])):
        for step_idx in range(len(A_hat_array)):

            iter_count = epoch*len(A_hat_array) + step_idx

            H = H_array[step_idx]
            A_hat = A_hat_array[step_idx]
            labels_gt = labels_array[step_idx]

            Z = 0
            if args["use_W1"]:
                Z += W1 @ H
            if args["use_W2"]:
                Z += W2 @ H @ A_hat
            dZ = (Z - Y)/N

            if args["use_W1"]:
                dW1 = dZ @ H.t() + lambda_W1 * W1
            if args["use_W2"]:
                dW2 = dZ @ (H @ A_hat).t() + lambda_W2 * W2

            dH = lambda_H * H
            if args["use_W1"]:
                dH += W1.t() @ dZ
            if args["use_W2"]:
                dH += W2.t() @ dZ @ A_hat.t()

            if args["use_W1"]:
                W1 -= args["lr"] * dW1
            if args["use_W2"]:
                W2 -= args["lr"] * dW2
            H -= args["lr"] * dH
            H_array[step_idx] = H

            if (iter_count % args["nc_interval"] == 0 or iter_count + 1 == max_iters):
                loss_array, acc_array = nc_helper(args=args, W1=W1, W2=W2, H_array=H_array,
                                        A_hat_array=A_hat_array, labels_array=labels_array)
                filename = "{}/gufm_tracker_{}.png".format(args["vis_dir"], iter_count)
                filenames.append(filename)
                tracker.compute_metrics(
                    H_array=H_array,
                    A_hat_array=A_hat_array,
                    W1=W1,
                    W2=W2,
                    labels_array=labels_array,
                    iter=iter_count,
                    train_loss_array=loss_array,
                    train_accuracy_array=acc_array,
                    filename=filename,
                    nc_interval=args["nc_interval"])

    animation_filename = "{}/gufm_tracker.mp4".format(args["vis_dir"])
    tracker.prepare_animation(image_filenames=filenames, animation_filename=animation_filename)


def init_params(args, C, d, N, H_stddev_factor):

    if args["use_W1"]:
        W1 = torch.randn(C, d, requires_grad=False).type(torch.double).to(args["device"])
    else:
        W1 = torch.zeros(C, d, requires_grad=False).type(torch.double).to(args["device"])

    if args["use_W2"]:
        W2 = torch.randn(C, d, requires_grad=False).type(torch.double).to(args["device"])
    else:
        W2 = torch.zeros(C, d, requires_grad=False).type(torch.double).to(args["device"])

    # unconstrained features
    H_array = []
    for i in range(args["num_train_graphs"]):
        H = torch.randn(d, N, requires_grad=False).type(torch.double) * H_stddev_factor
        H = H.to(args["device"])
        H_array.append(H)

    return W1, W2, H_array

if __name__ == "__main__":

    args = get_run_args()
    C = args["C"]
    d = args["hidden_feature_dim"]
    N = args["N"]
    n = N//C
    Y = torch.kron(torch.eye(C), torch.ones(1, n)).to(args["device"])
    print("shape of Y", Y.shape)

    lambda_W1 = args["lambda_W1"]
    lambda_W2 = args["lambda_W2"]
    lambda_H = args["lambda_H"]

    train_sbm_dataset = SBM_FACTORY[args["train_sbm_type"]](
        args=args,
        N=N,
        C=C,
        Pr=args["Pr"],
        p=args["p"],
        q=args["q"],
        num_graphs=args["num_train_graphs"],
        feature_strategy=args["feature_strategy"],
        feature_dim=args["hidden_feature_dim"],
        permute_nodes=False,
        is_training=True
    )
    train_dataloader = DataLoader(dataset=train_sbm_dataset, batch_size=1)
    A_hat_array = []
    labels_array = []
    for data in train_dataloader:
        A = to_dense_adj(data.edge_index)[0].to(args["device"])
        D_inv = torch.diag(1/torch.sum(A, 1)).to(args["device"])
        A_hat = (A @ D_inv).type(torch.double).to(args["device"])
        A_hat.requires_grad = False
        A_hat_array.append(A_hat)
        labels = torch.argmax(Y, axis=0).type(torch.int32).to(args["device"])
        labels.requires_grad = False
        labels_array.append(labels)

    W1, W2, H_array = init_params(args=args, C=C, d=d, N=N, H_stddev_factor=args["H_stddev_factor"])
    train_loop(args=args, W1=W1, W2=W2, H_array=H_array, A_hat_array=A_hat_array, labels_array=labels_array)
