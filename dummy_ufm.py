import os
import argparse
import pprint
import json
import sys
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

def get_run_args():
    parser = argparse.ArgumentParser(description='Arguments for running the experiments')
    parser.add_argument('config_file',  type=str, help='config file for the run')
    parsed_args = parser.parse_args()

    with open(parsed_args.config_file) as f:
        args = json.load(fp=f)
    args["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args)

    if args["model_name"] != "dummy_ufm":
        sys.exit("Invalid model_name. Should be 'dummy_ufm'")

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


def loss_func(W1, W2, H, A_hat, Y, N):
    Z = W1 @ H + W2 @ H @ A_hat
    return 0.5*(1/N)*(  torch.norm(Z - Y) )**2 \
        + 0.5*lambda_W1*torch.norm(W1)**2 \
        + 0.5*lambda_W2*torch.norm(W2)**2 \
        + 0.5*lambda_H*torch.norm(H)**2


def train_loop(args, W1, W2, H, A_hat):

    tracker = GUFMMetricTracker(args=args)
    filenames = []

    for i in range(args["num_epochs"]+1):

        Z = W1 @ H  + W2 @ H @ A_hat
        dZ = (Z - Y)/N

        dW1 = dZ @ H.t() + lambda_W1 * W1
        dW2 = dZ @ (H @ A_hat).t() + lambda_W2 * W2
        dH = W1.t() @ dZ + W2.t() @ dZ @ A_hat.t() + lambda_H * H

        loss = loss_func(W1=W1, W2=W2, H=H, A_hat=A_hat, Y=Y, N=N).type(torch.double)
        if i%args["nc_interval"] == 0:
            labels_pred = torch.argmax(Z, axis=0).type(torch.int32)
            acc = torch.mean((labels_pred == labels_gt).type(torch.float))
            print(loss, acc)
            filename = "{}/gufm_tracker_{}.png".format(args["vis_dir"], i)
            filenames.append(filename)
            tracker.compute_metrics(
                H=H,
                A_hat=A_hat,
                W_1=W1,
                W_2=W2,
                labels=labels_gt,
                epoch=i,
                train_loss=loss.detach().cpu().numpy(),
                train_accuracy=acc,
                filename=filename,
                nc_interval=args["nc_interval"])
        if args["use_W1"]:
            W1 -= args["lr"] * dW1
        if args["use_W2"]:
            W2 -= args["lr"] * dW2
        H -= args["lr"] * dH

    animation_filename = "{}/gufm_tracker.mp4".format(args["vis_dir"])
    tracker.prepare_animation(image_filenames=filenames, animation_filename=animation_filename)


def init_params(C, d, N, H_stddev_factor):

    if args["use_W1"]:
        W1 = torch.randn(C, d).type(torch.double)
    else:
        W1 = torch.zeros(C, d).type(torch.double)

    if args["use_W2"]:
        W2 = torch.randn(C, d).type(torch.double)
    else:
        W2 = torch.zeros(C, d).type(torch.double)

    # unconstrained features
    H = torch.randn(d, N).type(torch.double) * H_stddev_factor

    return W1, W2, H

if __name__ == "__main__":
    
    args = get_run_args()
    C = args["C"]
    d = args["hidden_feature_dim"]
    N = args["N"]
    n = N//C
    Y = torch.kron(torch.eye(C), torch.ones(1, n))
    print("shape of Y", Y.shape)
    labels_gt = torch.argmax(Y, axis=0).type(torch.int32)

    lambda_W1 = args["lambda_W1"]
    lambda_W2 = args["lambda_W2"]
    lambda_H = args["lambda_H"]

    train_sbm_dataset = SBM_FACTORY[args["train_sbm_type"]](
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
    for data in train_dataloader:
        A = to_dense_adj(data.edge_index)[0]
        D_inv = torch.diag(1/torch.sum(A, 1))
        A_hat = (D_inv @ A).type(torch.double)

    W1, W2, H = init_params(C=C, d=d, N=N, H_stddev_factor=args["H_stddev_factor"])
    train_loop(args=args, W1=W1, W2=W2, H=H, A_hat=A_hat)
