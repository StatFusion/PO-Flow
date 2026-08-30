# PO-Flow

Official code for **[Flow-based Generative Modeling of Potential Outcomes and Counterfactuals](https://arxiv.org/abs/2505.16051)**.

PO-Flow uses conditional flow matching to model potential-outcome distributions and to
generate factual-conditioned counterfactual outcomes. This repository contains the scalar
experiments on ACIC 2016, ACIC 2018, IHDP, and the IBM causal benchmark.

## Setup

Python 3.9+ is recommended. Create an environment and install the dependencies:

```bash
conda create -n poflow python=3.10 -y
conda activate poflow
pip install torch torchdiffeq numpy pandas matplotlib scikit-learn pyyaml POT
```

For GPU acceleration, install the PyTorch build matching your CUDA version. The code
automatically uses CUDA when available and otherwise runs on CPU.

## Run an experiment

From the repository root, select one of the provided configurations:

```bash
python main.py --PO_Flow_config ACIC2016.yaml
python main.py --PO_Flow_config ACIC2018.yaml
python main.py --PO_Flow_config IHDP.yaml
python main.py --PO_Flow_config IBM.yaml
```

The configurations specify the dataset, number of epochs, batch size, and evaluation
frequency. The corresponding processed datasets are already included under `datasets/`.

To run all four experiments sequentially:

```bash
for config in ACIC2016.yaml ACIC2018.yaml IHDP.yaml IBM.yaml; do
  python main.py --PO_Flow_config "$config"
done
```

## Outputs

Each run writes:

- evaluation metrics to `results/<DATASET>_metrics.csv`;
- the trained model to `NN_params/<DATASET>.pth`.

The default random seed is 42 and each dataset uses a reproducible 80/20 train/test split.
Reported metrics include counterfactual RMSE, PEHE, potential-outcome RMSE, and
Wasserstein distance.

## Repository structure

```text
main.py        PO-Flow model, training, ODE sampling, and evaluation
utils.py       dataset construction and Wasserstein distance
*.yaml         experiment configurations
datasets/      processed scalar benchmark datasets
```

## Citation

```bibtex
@article{wu2025flow,
  title   = {Flow-based Generative Modeling of Potential Outcomes and Counterfactuals},
  author  = {Wu, Dongze and Inouye, David I. and Xie, Yao},
  journal = {arXiv preprint arXiv:2505.16051},
  year    = {2025}
}
```
