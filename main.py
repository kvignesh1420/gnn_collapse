import os
import sys
import time
import json
import argparse
import pprint
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from gnn_collapse.data.sbm import SBM
from gnn_collapse.models import factory
from gnn_collapse.train.online import OnlineRunner
from gnn_collapse.train.online import OnlineIncRunner
from gnn_collapse.train.spectral import spectral_clustering

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
    test_fac = 1
    test_sbm_dataset = SBM(
        n=args["n"]*test_fac,
        k=args["k"],
        p=args["p"],
        W=np.array(args["W"])*np.log(test_fac*args["n"])/(test_fac*np.log(args["n"])),
        num_graphs=args["num_test_graphs"],
        feature_strategy=args["feature_strategy"],
        feature_dim=args["input_feature_dim"]
    )
    test_dataloader = DataLoader(dataset=test_sbm_dataset, batch_size=1)

    model_class = factory[args["model_name"]]
    if args["model_name"] not in ["bethe_hessian", "normalized_laplacian"]:
        model = model_class(
            input_feature_dim=args["input_feature_dim"],
            hidden_feature_dim=args["hidden_feature_dim"],
            loss_type=args["loss_type"],
            num_classes=args["k"],
            L=args["num_layers"],
            batch_norm=args["batch_norm"],
        ).to(args["device"])
        print("# parameters: ", count_parameters(model=model))
        # NOTE: Batch norm is key for performance, since we are sampling new graphs
        # it is better to unfreeze the batch norm values during testing.
        if "_inc" not in args["model_name"]:
            runner = OnlineRunner(track_nc=args["track_nc"])
        else:
            runner = OnlineIncRunner(track_nc=args["track_nc"])
        runner.run(train_dataloader=train_dataloader, test_dataloader=test_dataloader, model=model, args=args)
    else:
        spectral_clustering(model_class=model_class, dataloader=test_dataloader, args=args)
