# Exploring neural collapse in graph neural networks

This repo contains the code for our [paper](https://arxiv.org/abs/2307.01951) titled "A Neural Collapse Perspective on Feature Evolution in Graph Neural Networks"

## Setup

```bash
$ python3.9 -m virtualenv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

## Experiments

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

## Citation

```bibtex
@misc{kothapalli2023neural,
      title={A Neural Collapse Perspective on Feature Evolution in Graph Neural Networks}, 
      author={Vignesh Kothapalli and Tom Tirer and Joan Bruna},
      year={2023},
      eprint={2307.01951},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```