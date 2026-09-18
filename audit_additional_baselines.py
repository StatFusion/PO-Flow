"""Integrity audit for the 110 formal TARNet and DragonNet runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "additional_baselines"
MODELS = ROOT / "NN_params" / "additional_baselines"
DATASETS = [
    "ACIC2016", "ACIC2018", "IHDP", "IBM",
    "IHDP_Vector_d2", "ACIC2016_Vector_d2",
    "IHDP_Vector_d5", "ACIC2016_Vector_d5",
    "IHDP_Vector_d10", "ACIC2016_Vector_d10",
    "MorphoMNIST_Intensity_d784",
]
METHODS = ["TARNet", "DragonNet"]
SEEDS = [42, 43, 44, 45, 46]
UPSTREAM_COMMITS = {
    "TARNet": "8fc242edbbd49ba6ff5dbca0201ed9b5bc79c5bb",
    "DragonNet": "2e51b2478854df389893bac85430bca5b70f5a26",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_state(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    missing = []
    invalid = []
    checked = []
    for dataset in DATASETS:
        for method in METHODS:
            for seed in SEEDS:
                run = f"{dataset}_{method}_seed{seed}"
                paths = {
                    "metrics": RESULTS / f"{run}_metrics.csv",
                    "metadata": RESULTS / f"{run}_metadata.json",
                    "log": RESULTS / "logs" / f"{run}.log",
                    "checkpoint": MODELS / f"{run}.pth",
                }
                for kind, path in paths.items():
                    if not path.is_file() or path.stat().st_size == 0:
                        missing.append({"run": run, "kind": kind, "path": str(path)})
                if any(not path.is_file() or path.stat().st_size == 0 for path in paths.values()):
                    continue
                row = pd.read_csv(paths["metrics"]).iloc[0]
                point = row[["counterfactual", "pehe", "po", "mean_w1"]].to_numpy(float)
                if not np.isfinite(point).all() or (point < 0).any():
                    invalid.append({"run": run, "reason": "invalid point metric"})
                if not np.isfinite(float(row["wall_seconds"])) or float(row["wall_seconds"]) <= 0:
                    invalid.append({"run": run, "reason": "invalid wall time"})
                if int(row["parameter_count"]) <= 0:
                    invalid.append({"run": run, "reason": "invalid parameter count"})
                if pd.notna(row["distribution_w1"]) or bool(row["distributional_metric_applicable"]):
                    invalid.append({"run": run, "reason": "point estimator reported distributional metric"})
                metadata = json.loads(paths["metadata"].read_text())
                if metadata.get("dataset") != dataset or metadata.get("method") != method or int(metadata.get("seed", -1)) != seed:
                    invalid.append({"run": run, "reason": "metadata key mismatch"})
                if metadata.get("device") != "cuda" or not metadata.get("device_name"):
                    invalid.append({"run": run, "reason": "formal run did not record a CUDA device"})
                if not metadata.get("hostname") or not metadata.get("slurm_job_id"):
                    invalid.append({"run": run, "reason": "missing compute-node provenance"})
                if len(metadata.get("data_sha256", "")) != 64:
                    invalid.append({"run": run, "reason": "missing data SHA-256"})
                if len(metadata.get("config_sha256", "")) != 64 or len(metadata.get("baseline_config_sha256", "")) != 64:
                    invalid.append({"run": run, "reason": "missing configuration SHA-256"})
                if metadata.get("upstream_commit") != UPSTREAM_COMMITS[method]:
                    invalid.append({"run": run, "reason": "upstream revision mismatch"})
                log_text = paths["log"].read_text(errors="replace")
                for marker in ("Traceback", "CUDA out of memory", "RuntimeError"):
                    if marker in log_text:
                        invalid.append({"run": run, "reason": f"log contains {marker}"})
                state = load_state(paths["checkpoint"])
                if not state or any(not torch.isfinite(tensor).all() for tensor in state.values()):
                    invalid.append({"run": run, "reason": "nonfinite or empty checkpoint"})
                checked.append(run)

    aggregate_paths = [
        RESULTS / "additional_baseline_all_seed_metrics.csv",
        RESULTS / "additional_baseline_summary.csv",
        RESULTS / "scalar_journal_comparison.csv",
        RESULTS / "vector_journal_comparison.csv",
        RESULTS / "vector_journal_winners.csv",
        RESULTS / "vector_journal_winner_counts.csv",
    ]
    for path in aggregate_paths:
        if not path.is_file() or path.stat().st_size == 0:
            missing.append({"run": "aggregate", "kind": "aggregate", "path": str(path)})
    artifact_paths = [
        ROOT / "additional_baselines.py",
        ROOT / "summarize_additional_baselines.py",
        ROOT / "audit_additional_baselines.py",
        ROOT / "run_additional_baselines.sh",
        ROOT / "run_additional_baselines.sbatch",
        ROOT / "configs" / "baselines" / "Neural_Point_Baselines.yaml",
        *aggregate_paths,
    ]
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in artifact_paths if path.is_file()}
    report = {
        "expected_runs": 110,
        "checked_runs": len(checked),
        "missing": missing,
        "invalid": invalid,
        "status": "pass" if len(checked) == 110 and not missing and not invalid else "fail",
        "sha256": hashes,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "additional_baseline_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
