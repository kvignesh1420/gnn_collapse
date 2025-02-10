# Exploring neural collapse in graph neural networks

This repository contains the code for the DSC 180B Senior Capstone at UC San Diego. In this study, we seek to observe the presence of Neural Collapse in Graph Transformers. This repository is built upon the existing work by Vignesh Kothapalli, Tom Tirer, and Joan Bruna in their [paper](https://arxiv.org/abs/2307.01951) "A Neural Collapse Perspective on Feature Evolution in Graph Neural Networks". The original repository for this work can be found [here](https://github.com/kvignesh1420/gnn_collapse/tree/main)

## Setup

```bash
$ python3.9 -m virtualenv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

## Data

As written in the original repository:

We randomly sample graphs from the stochastic block model to control the properties of planted communities. The `SBM` class in `gnn_collapse.data.sbm` is an instance of `torch_geometric.data.Dataset` and facilitates direct encapsulation with torch `DataLoader`. Currently, we support the following feature strategies:

```python
class FeatureStrategy(Enum):
    EMPTY = "empty"
    DEGREE = "degree"
    RANDOM = "random"
    RANDOM_NORMAL = "random_normal"
    DEGREE_RANDOM = "degree_random"
    DEGREE_RANDOM_NORMAL = "degree_random_normal"
```

## Models

The models used in the original paper, as well as their instructions to add new models, can be found at the end of this section. Following these guidelines, we implement NC tracking across several additional types of layers. The primary layer we are working with at this time is `GPSConv`. This Model is based on the [GraphGPS architecture](https://proceedings.neurips.cc/paper_files/paper/2022/file/5d4834a159f1547b267a05a4e2b7cf5e-Paper-Conference.pdf), however we ablate the message-passing steps in this model to reduce it to nearly a vanilla transformer. The layer is used for ease of implementation.

As written in the original repository:

We primarily focus on the `GraphConv` model due to it's simplicity and similarity with a wide variety of message passing approaches. We customize the source code of `class GraphConv(MessagePassing)` (available [here](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/graph_conv.html#GraphConv)) to control whether the `lin_root` weight matrix ($W_1$ in the paper) is applied or not.

To add new models, one key point to consider is the naming convention of the weight matrices in various layers. For instance, the [GCNConv](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gcn_conv.html#GCNConv) layer has a single `lin` property that corresponds to the weight matrix. To handle such scenarios, it is best to modify the weight variable allocation in the `track_train_graphs_final_nc(...)` method (in the `gnn_collapse.train.online.OnlineRunner()` class).

Finally, to register a new model, please add an entry in the `gnn_collapse.models.GNN_factory` dictionary. This will facilitate model name validation and custom behaviours (such as the weight matrix selection, mentioned above) during training/inference. 

_NOTE: The code for `gnn_collapse.models.graphconv.GraphConvModel()` can be used as a reference to add new models._

## Experiments

As written in the original repository:

We employ a config based design to run and hash the experiments. The `configs` folder contains the `final` folder to maintain the set of experiments that have been presented in the paper. The `experimental` folder is a placeholder for new contributions. A config file is a JSON formatted file which is passed to the python script for parsing. The config determines the runtime parameters of the experiment and is hashed for uniqueness.

To run GNN experiments:
```bash
$ bash run_gnn.sh
```

To run gUFM experiments
```bash
$ bash run_ufm.sh
```

To run GNN experiments with larger depth
```bash
$ bash run_gnn_deeper.sh
```

To run spectral methods experiments
```bash
$ bash run_spectral.sh
```

A new folder called `out` will be created and the results are stored in a folder named after the hash of the config.

## Citation

```bibtex
@inproceedings{kothapalli2023neural,
  title={A Neural Collapse Perspective on Feature Evolution in Graph Neural Networks},
  author={Kothapalli, Vignesh and Tirer, Tom and Bruna, Joan},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```

## Contributing

As written in the original repository:

Please feel free to open [issues](https://github.com/kvignesh1420/gnn_collapse/issues) and create [pull requests](https://github.com/kvignesh1420/gnn_collapse/pulls) to fix bugs and improve performance.

## License

[MIT](LICENSE)