# PO-Flow

Official implementation of **Flow-based Generative Modeling of Potential
Outcomes and Counterfactuals**. PO-Flow uses conditional flow matching to model
potential-outcome distributions and to generate factual-conditioned
counterfactual outcomes.

This release contains both the original scalar implementation and the frozen
overlap-aware scalar, vector, and image experiments used in the journal
extension. The release intentionally excludes checkpoints, logs, search grids,
and rejected intermediate configurations.

## Final release contents

- `configs/scalar/`: four final IPW training configurations for ACIC 2016,
  ACIC 2018, IHDP, and IBM.
- `configs/vector/`: seven final configurations for IHDP-Vector,
  ACIC16-Vector, and MorphoMNIST-Intensity.
- `configs/baselines/`: the fixed TARNet, DragonNet, CFR, GANITE, and CEVAE
  adaptation settings used in the matched comparisons.
- `reproducibility/final_results.csv`: the exact five-seed aggregate results
  attached to those configurations.
- `verify_release.py`: verifies frozen code, config, and processed-data hashes.

The four root YAML files are the original, unweighted scalar configurations
published with the conference paper. They are preserved for backward
compatibility; all final overlap-aware configurations live under `configs/`.

## Environment

The frozen runs used Python 3.8.20, PyTorch 2.4.1+cu121, CUDA 12.1,
NumPy 1.24.4, pandas 1.5.3, scikit-learn 1.3.2, PyYAML 6.0.2, and
torchdiffeq 0.2.5. A compatible environment can be installed with:

```bash
conda create -n poflow python=3.8 -y
conda activate poflow
pip install -r requirements.txt
```

Install the PyTorch build appropriate for the local CUDA driver when the
default wheel is not suitable.

## Verify the release

The processed archives required by the seven vector/image configurations are
included. Before launching a long run, verify that the code, configs, data, and
result manifest are mutually consistent:

```bash
python verify_release.py
python -m pytest -q tests
```

The verifier checks the exact SHA-256 fingerprints frozen with the formal runs.
Dataset construction and source information are documented in
`multidim_benchmarks/README.md`.

## Reproduce the final PO-Flow runs

Each YAML begins with a direct single-seed command. To run all 11 final
configurations for seeds 42--46:

```bash
python run_tuned_ipw.py run \
  --datasets ACIC2016,ACIC2018,IHDP,IBM,IHDP_Vector_d2,IHDP_Vector_d5,IHDP_Vector_d10,ACIC2016_Vector_d2,ACIC2016_Vector_d5,ACIC2016_Vector_d10,MorphoMNIST_Intensity_d784 \
  --seeds 42,43,44,45,46 --jobs 1
python run_tuned_ipw.py summarize
```

`--jobs 1` is the safe default for one GPU. Training creates checkpoints in
`NN_params/` and metrics/logs in `results/`; both directories are ignored by
Git.

Examples of direct calls are:

```bash
# Scalar outcome
python main_IPW.py \
  --PO_Flow_config configs/scalar/IHDP.yaml \
  --run-name FinalIPW_IHDP_seed42 --seed 42 --device cuda \
  --test-epochs 90 --eval-samples 1 --log-every 90

# Vector outcome
python main_multidim.py \
  --config configs/vector/IHDP_Vector_d5.yaml \
  --run-name FinalIPW_IHDP_Vector_d5_seed42 --seed 42 --device cuda
```

The scalar runner performs a separate 100-draw evaluation after training. That
two-stage call is encoded in `run_tuned_ipw.py` and should be used when
reproducing the reported scalar numbers.

## Baselines

Run the fixed matched baseline suites with:

```bash
bash run_additional_baselines.sh   # TARNet and DragonNet
bash run_multidim_baselines.sh     # CFR, GANITE, and CEVAE
```

The scripts are restart-safe and skip runs whose metric files already exist.

## Original conference experiments

The original scalar implementation remains available:

```bash
python main.py --PO_Flow_config ACIC2016.yaml
python main.py --PO_Flow_config ACIC2018.yaml
python main.py --PO_Flow_config IHDP.yaml
python main.py --PO_Flow_config IBM.yaml
```

## Repository structure

```text
main.py                         flow architectures and ODE maps
main_IPW.py                     scalar IPW training/evaluation
main_multidim.py                vector/image training/evaluation
ipw.py                          cross-fitted propensity estimation and weights
evaluate_IPW.py                 high-draw scalar checkpoint evaluation
multidim_benchmarks/            frozen archives and deterministic builder
configs/                        canonical final configurations only
reproducibility/                compact final-result provenance
tests/                          unit and numerical tests
```

## Citation

```bibtex
@inproceedings{wu2026flow,
  title        = {Flow-based generative modeling of potential outcomes and counterfactuals},
  author       = {Wu, Dongze and Xie, Yao and Inouye, David I.},
  booktitle    = {2026 IEEE International Symposium on Information Theory (ISIT)},
  pages        = {1--6},
  year         = {2026},
  organization = {IEEE}
}
```
