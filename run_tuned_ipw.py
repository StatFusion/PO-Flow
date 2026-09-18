"""Run and summarize the 11 canonical final PO-Flow+IPW configurations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List

import yaml


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "results" / "final_ipw" / "logs"
SUMMARY_DIR = ROOT / "results" / "final_ipw"
SCALAR = {
    "ACIC2016": ROOT / "configs" / "scalar" / "ACIC2016.yaml",
    "ACIC2018": ROOT / "configs" / "scalar" / "ACIC2018.yaml",
    "IHDP": ROOT / "configs" / "scalar" / "IHDP.yaml",
    "IBM": ROOT / "configs" / "scalar" / "IBM.yaml",
}
VECTOR = {
    path.stem: path
    for path in sorted((ROOT / "configs" / "vector").glob("*.yaml"))
}
METRICS = ("counterfactual", "pehe", "po", "distribution_w1", "mean_w1")


def _run(command: List[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    with log_path.open("w") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode:
        raise RuntimeError(f"exit {completed.returncode}: {' '.join(command)}; see {log_path}")


def _scalar_job(dataset: str, seed: int, force: bool) -> str:
    run_name = f"FinalIPW_{dataset}_seed{seed}"
    output = ROOT / "results" / f"{run_name}_stable_metrics.json"
    if output.exists() and not force:
        return f"SKIP {run_name}"
    config = SCALAR[dataset]
    payload = yaml.safe_load(config.read_text())
    epochs = int(payload["training"]["epochs"])
    _run(
        [
            sys.executable,
            "-u",
            "main_IPW.py",
            "--PO_Flow_config",
            str(config),
            "--run-name",
            run_name,
            "--seed",
            str(seed),
            "--device",
            "cuda",
            "--test-epochs",
            str(epochs),
            "--eval-samples",
            "1",
            "--log-every",
            str(epochs),
        ],
        LOG_DIR / f"{run_name}_train.log",
    )
    _run(
        [
            sys.executable,
            "-u",
            "evaluate_IPW.py",
            "--PO_Flow_config",
            str(config),
            "--checkpoint",
            str(ROOT / "NN_params" / f"{run_name}_swa_final.pth"),
            "--run-name",
            run_name,
            "--seed",
            str(seed),
            "--samples",
            "100",
            "--counterfactual-ode-steps",
            "100",
            "--potential-ode-steps",
            "50",
            "--device",
            "cuda",
        ],
        LOG_DIR / f"{run_name}_evaluate.log",
    )
    return f"DONE {run_name}"


def _vector_job(dataset: str, seed: int, force: bool) -> str:
    run_name = f"FinalIPW_{dataset}_seed{seed}"
    output = ROOT / "results" / "multidim" / f"{run_name}_metrics.csv"
    if output.exists() and not force:
        return f"SKIP {run_name}"
    _run(
        [
            sys.executable,
            "-u",
            "main_multidim.py",
            "--config",
            str(VECTOR[dataset]),
            "--run-name",
            run_name,
            "--seed",
            str(seed),
            "--device",
            "cuda",
        ],
        LOG_DIR / f"{run_name}.log",
    )
    return f"DONE {run_name}"


def run(datasets: Iterable[str], seeds: Iterable[int], jobs: int, force: bool) -> None:
    datasets = tuple(datasets)
    seeds = tuple(seeds)
    work = []
    for dataset in datasets:
        if dataset in SCALAR:
            function = _scalar_job
        elif dataset in VECTOR:
            function = _vector_job
        else:
            raise ValueError(f"unknown dataset: {dataset}")
        for seed in seeds:
            work.append((function, dataset, seed, force))
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(*item) for item in work]
        for future in as_completed(futures):
            try:
                print(future.result(), flush=True)
            except Exception as error:
                print(f"FAIL {error}", flush=True)


def _mean(values):
    return sum(values) / len(values)


def _std(values):
    if len(values) < 2:
        return 0.0
    center = _mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def summarize() -> None:
    all_rows = []
    for dataset in [*SCALAR, *VECTOR]:
        for seed in range(42, 47):
            scalar_path = ROOT / "results" / f"FinalIPW_{dataset}_seed{seed}_stable_metrics.json"
            vector_path = ROOT / "results" / "multidim" / f"FinalIPW_{dataset}_seed{seed}_metrics.csv"
            if scalar_path.exists():
                payload = json.loads(scalar_path.read_text())
                metrics = {
                    "counterfactual": float(payload["counterfactual_RMSE"]),
                    "pehe": float(payload["PEHE"]),
                    "po": float(payload["PO_RMSE"]),
                    "distribution_w1": float(payload["W1_distance"]),
                    "mean_w1": float(payload["W1_mean_distance"]),
                }
            elif vector_path.exists():
                with vector_path.open(newline="") as handle:
                    payload = list(csv.DictReader(handle))[-1]
                metrics = {
                    "counterfactual": float(payload["counterfactual_NRMSE"]),
                    "pehe": float(payload["vector_PEHE"]),
                    "po": float(payload["PO_NRMSE"]),
                    "distribution_w1": float(payload["sliced_W1_distance"]),
                    "mean_w1": float(payload["sliced_W1_mean_distance"]),
                }
            else:
                continue
            all_rows.append({"dataset": dataset, "seed": seed, **metrics})
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    with (SUMMARY_DIR / "tuned_ipw_all_seed_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("dataset", "seed", *METRICS))
        writer.writeheader()
        writer.writerows(all_rows)
    summary = []
    for dataset in sorted({row["dataset"] for row in all_rows}):
        rows = [row for row in all_rows if row["dataset"] == dataset]
        output = {"dataset": dataset, "seed_count": len(rows)}
        for metric in METRICS:
            values = [row[metric] for row in rows]
            output[f"{metric}_mean"] = _mean(values)
            output[f"{metric}_std"] = _std(values)
        summary.append(output)
    fields = ("dataset", "seed_count", *(item for metric in METRICS for item in (f"{metric}_mean", f"{metric}_std")))
    with (SUMMARY_DIR / "tuned_ipw_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(row)


def _items(value: str):
    return [item for item in value.split(",") if item]


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--datasets", required=True)
    run_parser.add_argument("--seeds", default="42,43,44,45,46")
    run_parser.add_argument("--jobs", type=int, default=1)
    run_parser.add_argument("--force", action="store_true")
    subparsers.add_parser("summarize")
    args = parser.parse_args()
    if args.command == "run":
        run(_items(args.datasets), (int(seed) for seed in _items(args.seeds)), args.jobs, args.force)
    else:
        summarize()


if __name__ == "__main__":
    main()
