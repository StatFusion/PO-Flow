"""Fail-fast integrity audit for the formal multidimensional baseline sweep."""

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "multidim_baselines"
CHECKPOINTS = ROOT / "NN_params" / "multidim_baselines"
DATASETS = (
    "IHDP_Vector_d2",
    "ACIC2016_Vector_d2",
    "IHDP_Vector_d5",
    "ACIC2016_Vector_d5",
    "IHDP_Vector_d10",
    "ACIC2016_Vector_d10",
    "MorphoMNIST_Intensity_d784",
)
METHODS = ("CFR", "GANITE", "CEVAE")
SEEDS = (42, 43, 44, 45, 46)
FORMAL = re.compile(
    r"(?P<dataset>.+)_(?P<method>CFR|GANITE|CEVAE)_seed"
    r"(?P<seed>42|43|44|45|46)_metrics\.csv$"
)
POINT_METRICS = (
    "counterfactual_NRMSE",
    "vector_PEHE",
    "PO_NRMSE",
    "sliced_W1_mean_distance",
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    expected = {
        (dataset, method, seed)
        for dataset in DATASETS
        for method in METHODS
        for seed in SEEDS
    }
    observed = {}
    for path in RESULTS.glob("*_metrics.csv"):
        match = FORMAL.fullmatch(path.name)
        if match:
            key = (
                match.group("dataset"),
                match.group("method"),
                int(match.group("seed")),
            )
            require(key not in observed, f"duplicate run key: {key}")
            observed[key] = path
    require(set(observed) == expected, f"missing={expected-set(observed)}, extra={set(observed)-expected}")

    total_wall_seconds = 0.0
    for (dataset, method, seed), metric_path in sorted(observed.items()):
        frame = pd.read_csv(metric_path)
        require(len(frame) == 1, f"{metric_path}: expected exactly one row")
        row = frame.iloc[0]
        require(row["dataset"] == dataset, f"{metric_path}: dataset mismatch")
        require(row["method"] == method, f"{metric_path}: method mismatch")
        require(int(row["seed"]) == seed, f"{metric_path}: seed mismatch")
        values = row[list(POINT_METRICS)].to_numpy(dtype=float)
        require(np.isfinite(values).all() and (values >= 0).all(), f"{metric_path}: invalid point metric")
        total_wall_seconds += float(row["wall_seconds"])

        distributional = float(row["sliced_W1_distance"])
        distributional_se = float(row["sliced_W1_distance_MC_SE"])
        applicable = bool(row["distributional_metric_applicable"])
        if method == "CEVAE":
            require(applicable, f"{metric_path}: CEVAE distribution flag is false")
            require(np.isfinite(distributional) and distributional >= 0, f"{metric_path}: invalid SW1")
            require(np.isfinite(distributional_se) and distributional_se >= 0, f"{metric_path}: invalid SW1 SE")
        else:
            require(not applicable, f"{metric_path}: point method distribution flag is true")
            require(np.isnan(distributional) and np.isnan(distributional_se), f"{metric_path}: expected N/A SW1")

        stem = metric_path.name[: -len("_metrics.csv")]
        metadata_path = RESULTS / f"{stem}_metadata.json"
        log_path = RESULTS / "logs" / f"{stem}.log"
        checkpoint_path = CHECKPOINTS / f"{stem}.pth"
        require(metadata_path.is_file() and metadata_path.stat().st_size > 0, f"missing {metadata_path}")
        require(log_path.is_file() and log_path.stat().st_size > 0, f"missing {log_path}")
        require(checkpoint_path.is_file() and checkpoint_path.stat().st_size > 0, f"missing {checkpoint_path}")
        metadata = json.loads(metadata_path.read_text())
        require(metadata["run_name"] == stem, f"{metadata_path}: run name mismatch")
        log_text = log_path.read_text(errors="replace")
        require("Traceback" not in log_text, f"{log_path}: traceback")
        require("out of memory" not in log_text.lower(), f"{log_path}: OOM")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        require(isinstance(state, dict) and state, f"{checkpoint_path}: empty state dict")
        require(all(torch.isfinite(value).all() for value in state.values()), f"{checkpoint_path}: non-finite tensor")

    print(f"formal_runs={len(observed)}")
    print(f"metrics={len(observed)} metadata={len(observed)} logs={len(observed)} checkpoints={len(observed)}")
    print("point_metrics=105/105 finite")
    print("distributional_SW1=35/35 CEVAE finite; CFR/GANITE=70/70 correctly N/A")
    print("checkpoint_tensors=105/105 loadable_and_finite")
    print(f"summed_per_run_wall_seconds={total_wall_seconds:.3f}")


if __name__ == "__main__":
    main()
