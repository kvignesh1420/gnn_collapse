import os
import sys
import time
import hashlib
import json
import argparse
import pprint
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from gnn_collapse.data.sbm import SBM
from gnn_collapse.models import GNN_factory
from gnn_collapse.models import Spectral_factory
from gnn_collapse.train.online import OnlineRunner
from gnn_collapse.train.online import OnlineIncRunner
from gnn_collapse.train.spectral import spectral_clustering

def prepare_gnn_model_hash(args):
    model_args = {}
    keys = [
        "model_name", "N_train", "C", "Pr", "p_train", "q_train", "num_train_graphs", "feature_strategy",
        "input_feature_dim", "hidden_feature_dim", "num_layers", "use_W1", "use_bias",
        "batch_norm", "non_linearity", "loss_type", "optimizer", "lr", "weight_decay",
        "sgd_momentum"
    ]
    for key in keys:
        model_args[key] = args[key]

    _string_args = json.dumps(model_args, sort_keys=True).encode("utf-8")
    parsed_args_hash = hashlib.md5(_string_args).hexdigest()
    return parsed_args_hash

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
    model_uuid = ""
    if args["model_name"] in GNN_factory:
        model_uuid = prepare_gnn_model_hash(args=args)
     # create a unique hash for the model
    config_uuid = prepare_config_hash(args=args)

    args["model_uuid"] = model_uuid
    args["config_uuid"] = config_uuid
    args["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args)

    if args["model_name"] not in GNN_factory and args["model_name"] not in Spectral_factory:
        valid_options = list(GNN_factory.keys()) + list(Spectral_factory.keys())
        sys.exit("Invalid model type. Should be one of: {}".format(valid_options))

    if args["model_name"] in GNN_factory and args["non_linearity"] not in ["", "relu"]:
        sys.exit("Invalid non_linearity. Should be one of: '', 'relu' ")

    if args["model_name"] in GNN_factory and args["optimizer"] not in ["sgd", "adam"]:
        sys.exit("Invalid non_linearity. Should be one of: 'sgd', 'adam' ")

    vis_dir = args["out_dir"] + args["model_name"] + "/" + args["config_uuid"] + "/plots/"
    results_dir = args["out_dir"] + args["model_name"] + "/" + args["config_uuid"] + "/results/"
    results_file = results_dir + "run.txt"
    if not os.path.exists(vis_dir):
        print("Vis folder does not exist. Creating one!")
        os.makedirs(vis_dir)
    if not os.path.exists(results_dir):
        print("Resuls folder does not exist. Creating one!")
        os.makedirs(results_dir)
    else:
        print("Folder already exists!")
        sys.exit()
    args["vis_dir"] = vis_dir
    args["results_dir"] = results_dir
    args["results_file"] = results_file

    with open(results_file, 'a') as f:
        f.write("""CONFIG: \n{}\n""".format(pprint.pformat(args, sort_dicts=False)))

    return args


if __name__ == "__main__":

    args = get_run_args()
    if args["model_name"] not in ["bethe_hessian", "normalized_laplacian"]:
        train_sbm_dataset = SBM(
            args=args,
            N=args["N_train"],
            C=args["C"],
            Pr=args["Pr"],
            p=args["p_train"],
            q=args["q_train"],
            num_graphs=args["num_train_graphs"],
            feature_strategy=args["feature_strategy"],
            feature_dim=args["input_feature_dim"],
            is_training=True
        )
        nc_sbm_dataset = SBM(
            args=args,
            N=args["N_train"],
            C=args["C"],
            Pr=args["Pr"],
            p=args["p_train"],
            q=args["q_train"],
            num_graphs=args["num_train_graphs"],
            feature_strategy=args["feature_strategy"],
            feature_dim=args["input_feature_dim"],
            is_training=True
        )
        # keep batch size = 1 for consistent measurement of loss and accuracies under
        # permutation of classes.
        train_dataloader = DataLoader(dataset=train_sbm_dataset, batch_size=1)
        nc_dataloader = DataLoader(dataset=nc_sbm_dataset, batch_size=1)
    test_sbm_dataset = SBM(
        args=args,
        N=args.get("N_test", args.get("N")),
        C=args["C"],
        Pr=args["Pr"],
        p=args.get("p_test", args.get("p")),
        q=args.get("q_test", args.get("q")),
        num_graphs=args["num_test_graphs"],
        feature_strategy=args["feature_strategy"],
        feature_dim=args["input_feature_dim"],
        is_training=False
    )
    test_dataloader = DataLoader(dataset=test_sbm_dataset, batch_size=1)

    if args["model_name"] in GNN_factory:
        model_class = GNN_factory[args["model_name"]]
        runner = OnlineRunner(args=args, model_class=model_class)
        runner.run(train_dataloader=train_dataloader, nc_dataloader=nc_dataloader,
                    test_dataloader=test_dataloader)
    else:
        model_class = Spectral_factory[args["model_name"]]
        spectral_clustering(model_class=model_class, dataloader=test_dataloader, args=args)
