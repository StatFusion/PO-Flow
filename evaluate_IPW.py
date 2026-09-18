"""Evaluate a saved PO-Flow checkpoint with stable Monte Carlo PO means.

The legacy training entry point uses one draw by default to remain comparable
with ``main.py``.  Journal results should instead approximate
E[Y(a) | X] with enough draws that PEHE and PO RMSE are not dominated by one
Monte Carlo sample.  W1 distributional fit is averaged over independent
generated test-set distributions, and ``W1_mean_distance`` separately
diagnoses mean-distribution fit.  This script rebuilds the deterministic split for a
run and evaluates its final checkpoint without retraining.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from main import FlowMatchingNet
from main_IPW import PROJECT_DIR, evaluate, seed_everything
from utils import CounterfactualDatasetBuilder


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stable evaluation of a PO-Flow checkpoint")
    parser.add_argument("--PO_Flow_config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--counterfactual-ode-steps", type=int, default=100)
    parser.add_argument("--potential-ode-steps", type=int, default=50)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_DIR / candidate


def main() -> None:
    args = _parser().parse_args()
    if min(
        args.samples,
        args.batch_size,
        args.counterfactual_ode_steps,
        args.potential_ode_steps,
    ) <= 0:
        raise ValueError("sample, batch, and ODE-step counts must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")

    config_path = _resolve(args.PO_Flow_config)
    checkpoint_path = _resolve(args.checkpoint)
    with config_path.open("r") as file:
        config = yaml.safe_load(file)

    seed_everything(args.seed)
    data_name = config["data"]["name"]
    frame = pd.read_csv(PROJECT_DIR / "datasets" / f"{data_name}_sample.csv")
    _, test_dataset = CounterfactualDatasetBuilder(frame).build()
    test_loader = DataLoader(
        test_dataset,
        batch_size=min(args.batch_size, len(test_dataset)),
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    model = FlowMatchingNet(
        y_dim=config["data"]["Ydim"],
        x_dim=frame.shape[1] - 5,
        hidden_dim=config["training"].get("hidden_dim", 128),
    ).to(device)
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)

    cuda_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(args.seed + 100_000)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + 100_000)
        metrics = evaluate(
            model,
            test_loader,
            device,
            potential_outcome_samples=args.samples,
            ode_steps=args.counterfactual_ode_steps,
            potential_ode_steps=args.potential_ode_steps,
        )

    payload = {
        "dataset": data_name,
        "seed": args.seed,
        "checkpoint": str(checkpoint_path),
        "potential_outcome_samples": args.samples,
        "counterfactual_ode_steps": args.counterfactual_ode_steps,
        "potential_ode_steps": args.potential_ode_steps,
        **metrics,
    }
    results_dir = PROJECT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / f"{args.run_name}_stable_metrics.json"
    with output_path.open("w") as file:
        json.dump(payload, file, indent=2)
    print(json.dumps(payload, indent=2))
    print(f"Saved stable metrics to {output_path}")


if __name__ == "__main__":
    main()
