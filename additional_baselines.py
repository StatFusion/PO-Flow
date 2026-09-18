"""Matched TARNet and DragonNet baselines for scalar and vector PO-Flow data.

The runner uses the exact PO-Flow train/test split for every seed, fits all
preprocessing statistics on training rows only, and evaluates the same point
metrics as the journal experiments.  TARNet and DragonNet are point estimators;
their distributional Wasserstein metric is intentionally recorded as NaN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset

from main_IPW import seed_everything
from main_multidim import load_archive, sliced_wasserstein, standardize_archive
from multidim_baselines import CFRVector, factual
from utils import CounterfactualDatasetBuilder, wasserstein_dist


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results" / "additional_baselines"
MODEL_DIR = PROJECT_DIR / "NN_params" / "additional_baselines"


@dataclass
class BenchmarkData:
    name: str
    x_train: torch.Tensor
    a_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    a_test: torch.Tensor
    y_test: torch.Tensor
    y0_test: torch.Tensor
    y1_test: torch.Tensor
    mu0_test: torch.Tensor
    mu1_test: torch.Tensor
    outcome_center: torch.Tensor
    outcome_scale: torch.Tensor
    x_center: torch.Tensor
    x_scale: torch.Tensor
    metric_kind: str
    image: bool
    source_path: Path


class DragonNetVector(nn.Module):
    """Vector-outcome DragonNet with a treatment head and targeted correction."""

    def __init__(
        self, x_dim: int, y_dim: int, hidden: int,
        representation_layers: int = 3, outcome_head_layers: int = 2,
    ):
        super().__init__()
        body = []
        current = x_dim
        for _ in range(representation_layers):
            body.extend((nn.Linear(current, hidden), nn.ELU()))
            current = hidden
        self.representation = nn.Sequential(*body)

        def outcome_head():
            modules = []
            for _ in range(outcome_head_layers):
                modules.extend((nn.Linear(hidden, hidden), nn.ELU()))
            modules.append(nn.Linear(hidden, y_dim))
            return nn.Sequential(*modules)

        self.head0 = outcome_head()
        self.head1 = outcome_head()
        self.propensity = nn.Linear(hidden, 1)
        self.epsilon = nn.Parameter(torch.zeros(y_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        representation = self.representation(x)
        return self.head0(representation), self.head1(representation), self.propensity(representation).squeeze(1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _standardize_x(x_train: torch.Tensor, x_test: torch.Tensor):
    center = x_train.mean(0, keepdim=True)
    scale = x_train.std(0, keepdim=True).clamp_min(1e-6)
    return (x_train - center) / scale, (x_test - center) / scale, center, scale


def _load_scalar(config: Dict, config_path: Path, seed: int) -> BenchmarkData:
    name = config["data"]["name"]
    source = PROJECT_DIR / "datasets" / f"{name}_sample.csv"
    frame = pd.read_csv(source)
    # This is the same seeded torch.randperm split used by main_IPW.py.
    builder = CounterfactualDatasetBuilder(frame)
    train, test = builder.build()
    x_train = train.x_tensor.float()
    x_test = test.x_tensor.float()
    x_train, x_test, x_center, x_scale = _standardize_x(x_train, x_test)
    a_train = train.base_dataset.a.long()
    a_test = test.base_dataset.a.long()
    y_train_raw = train.base_dataset.y.float().view(-1, 1)
    y_test = test.base_dataset.y.float().view(-1, 1)
    counterfactual = test.counter_y_tensor.float().view(-1, 1)
    y0_test = torch.where(a_test[:, None] == 0, y_test, counterfactual)
    y1_test = torch.where(a_test[:, None] == 1, y_test, counterfactual)
    outcome_center = y_train_raw.mean(0, keepdim=True)
    outcome_scale = y_train_raw.std(0, keepdim=True).clamp_min(1e-6)
    y_train = (y_train_raw - outcome_center) / outcome_scale
    return BenchmarkData(
        name=name,
        x_train=x_train,
        a_train=a_train,
        y_train=y_train,
        x_test=x_test,
        a_test=a_test,
        y_test=y_test,
        y0_test=y0_test,
        y1_test=y1_test,
        mu0_test=test.mu0_tensor.float().view(-1, 1),
        mu1_test=test.mu1_tensor.float().view(-1, 1),
        outcome_center=outcome_center,
        outcome_scale=outcome_scale,
        x_center=x_center,
        x_scale=x_scale,
        metric_kind="scalar",
        image=False,
        source_path=source,
    )


def _load_vector(config: Dict, config_path: Path) -> BenchmarkData:
    source = PROJECT_DIR / config["data"]["archive"]
    arrays = load_archive(source)
    arrays, _, _ = standardize_archive(arrays, config["data"].get("image_scale", False))
    train_idx, test_idx = arrays["train_idx"], arrays["test_idx"]
    x_train = torch.as_tensor(arrays["X"][train_idx], dtype=torch.float32)
    x_test = torch.as_tensor(arrays["X"][test_idx], dtype=torch.float32)
    x_train, x_test, x_center, x_scale = _standardize_x(x_train, x_test)
    a_train = torch.as_tensor(arrays["A"][train_idx], dtype=torch.long)
    a_test = torch.as_tensor(arrays["A"][test_idx], dtype=torch.long)
    y0_train = torch.as_tensor(arrays["Y0"][train_idx], dtype=torch.float32)
    y1_train = torch.as_tensor(arrays["Y1"][train_idx], dtype=torch.float32)
    y0_test = torch.as_tensor(arrays["Y0"][test_idx], dtype=torch.float32)
    y1_test = torch.as_tensor(arrays["Y1"][test_idx], dtype=torch.float32)
    return BenchmarkData(
        name=config["data"]["name"],
        x_train=x_train,
        a_train=a_train,
        y_train=factual(y0_train, y1_train, a_train),
        x_test=x_test,
        a_test=a_test,
        y_test=factual(y0_test, y1_test, a_test),
        y0_test=y0_test,
        y1_test=y1_test,
        mu0_test=torch.as_tensor(arrays["MU0"][test_idx], dtype=torch.float32),
        mu1_test=torch.as_tensor(arrays["MU1"][test_idx], dtype=torch.float32),
        outcome_center=torch.zeros((1, y0_test.shape[1])),
        outcome_scale=torch.ones((1, y0_test.shape[1])),
        x_center=x_center,
        x_scale=x_scale,
        metric_kind="vector",
        image=bool(config["data"].get("image_scale", False)),
        source_path=source,
    )


def load_benchmark(config_path: Path, seed: int) -> BenchmarkData:
    config = yaml.safe_load(config_path.read_text())
    return _load_vector(config, config_path) if "archive" in config["data"] else _load_scalar(config, config_path, seed)


def make_loader(data: BenchmarkData, batch_size: int, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(data.x_train, data.a_train, data.y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
    )


def train_tarnet(model, loader, device, epochs, learning_rate, log_every, weight_decay=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    final = {}
    for epoch in range(1, epochs + 1):
        total = 0.0
        batches = 0
        model.train()
        for x, a, y in loader:
            x, a, y = x.to(device), a.to(device), y.to(device)
            y0, y1, _ = model(x)
            loss = F.mse_loss(factual(y0, y1, a), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        final = {"loss": total / batches, "factual_loss": total / batches}
        if epoch == 1 or epoch == epochs or epoch % log_every == 0:
            print(f"epoch={epoch} factual={final['factual_loss']:.6f}", flush=True)
    return final


def train_dragonnet(
    model, loader, device, epochs, learning_rate, propensity_strength,
    targeted_strength, propensity_clip, log_every, weight_decay=1e-4,
    max_grad_norm=10.0,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    final = {}
    for epoch in range(1, epochs + 1):
        totals = np.zeros(4, dtype=np.float64)
        batches = 0
        model.train()
        for x, a, y in loader:
            x, a, y = x.to(device), a.to(device), y.to(device)
            y0, y1, propensity_logit = model(x)
            prediction = factual(y0, y1, a)
            factual_loss = F.mse_loss(prediction, y)
            propensity_loss = F.binary_cross_entropy_with_logits(propensity_logit, a.float())
            propensity = torch.sigmoid(propensity_logit).clamp(propensity_clip, 1.0 - propensity_clip)
            clever_covariate = a.float() / propensity - (1.0 - a.float()) / (1.0 - propensity)
            targeted_prediction = prediction + clever_covariate[:, None] * model.epsilon[None, :]
            targeted_loss = F.mse_loss(targeted_prediction, y)
            loss = factual_loss + propensity_strength * propensity_loss + targeted_strength * targeted_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            totals += (loss.item(), factual_loss.item(), propensity_loss.item(), targeted_loss.item())
            batches += 1
        values = totals / batches
        final = {
            "loss": values[0],
            "factual_loss": values[1],
            "propensity_loss": values[2],
            "targeted_loss": values[3],
        }
        if epoch == 1 or epoch == epochs or epoch % log_every == 0:
            print(
                f"epoch={epoch} loss={values[0]:.6f} factual={values[1]:.6f} "
                f"propensity={values[2]:.6f} targeted={values[3]:.6f}",
                flush=True,
            )
    return final


@torch.no_grad()
def predict(model, method: str, x: torch.Tensor):
    if method == "tarnet":
        y0, y1, _ = model(x)
    elif method == "dragonnet":
        y0, y1, _ = model(x)
    else:
        raise ValueError(method)
    return y0, y1


@torch.no_grad()
def evaluate(model, method: str, data: BenchmarkData, device, batch_size: int, projections: int, seed: int):
    model.eval()
    y0_parts, y1_parts = [], []
    for start in range(0, len(data.x_test), batch_size):
        x = data.x_test[start : start + batch_size].to(device)
        y0, y1 = predict(model, method, x)
        y0_parts.append(y0.cpu())
        y1_parts.append(y1.cpu())
    y0_hat = torch.cat(y0_parts) * data.outcome_scale + data.outcome_center
    y1_hat = torch.cat(y1_parts) * data.outcome_scale + data.outcome_center
    cf_hat = torch.where(data.a_test[:, None] == 0, y1_hat, y0_hat)
    cf_true = torch.where(data.a_test[:, None] == 0, data.y1_test, data.y0_test)
    po_hat = factual(y0_hat, y1_hat, data.a_test)
    if data.metric_kind == "scalar":
        mean_w1 = 0.5 * (
            wasserstein_dist(y0_hat, data.mu0_test) + wasserstein_dist(y1_hat, data.mu1_test)
        )
    else:
        mean_w1 = 0.5 * (
            sliced_wasserstein(y0_hat.to(device), data.mu0_test.to(device), projections, seed + 20)
            + sliced_wasserstein(y1_hat.to(device), data.mu1_test.to(device), projections, seed + 21)
        )
    return {
        "counterfactual": torch.sqrt(F.mse_loss(cf_hat, cf_true)).item(),
        "pehe": torch.sqrt(F.mse_loss(y1_hat - y0_hat, data.mu1_test - data.mu0_test)).item(),
        "po": torch.sqrt(F.mse_loss(po_hat, data.y_test)).item(),
        "distribution_w1": float("nan"),
        "mean_w1": float(mean_w1),
        "distributional_metric_applicable": False,
        "metric_kind": data.metric_kind,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline-config", default="configs/baselines/Neural_Point_Baselines.yaml")
    parser.add_argument("--method", choices=("tarnet", "dragonnet"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--no-output", action="store_true", help="run without writing metrics, metadata, or checkpoints")
    return parser.parse_args()


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_DIR / candidate


def main():
    started = time.perf_counter()
    args = parse_args()
    seed_everything(args.seed)
    config_path = _resolve(args.config)
    baseline_config_path = _resolve(args.baseline_config)
    baseline_config = yaml.safe_load(baseline_config_path.read_text())
    data = load_benchmark(config_path, args.seed)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    family = "image" if data.image else ("scalar" if data.metric_kind == "scalar" else "tabular")
    method_config = baseline_config["baselines"][args.method]
    family_config = method_config[family]
    epochs = args.epochs or family_config["epochs"]
    batch_size = args.batch_size or family_config["batch_size"]
    hidden = family_config["hidden_dim"]
    learning_rate = family_config["learning_rate"]
    architecture = method_config["architecture"]
    if architecture["activation"].lower() != "elu":
        raise ValueError("only the audited ELU architecture is supported")
    if args.method == "tarnet":
        if architecture["representation_hidden_layers"] != 2 or architecture["outcome_head_hidden_layers"] != 2:
            raise ValueError("TARNet must match the audited two-layer CFR adaptation")
        model = CFRVector(data.x_train.shape[1], data.y_train.shape[1], hidden).to(device)
    else:
        if architecture["propensity_head"] != "linear" or architecture["targeted_epsilon"] != "per_outcome_coordinate":
            raise ValueError("unsupported DragonNet head configuration")
        model = DragonNetVector(
            data.x_train.shape[1], data.y_train.shape[1], hidden,
            representation_layers=architecture["representation_hidden_layers"],
            outcome_head_layers=architecture["outcome_head_hidden_layers"],
        ).to(device)
    loader = make_loader(data, batch_size, args.seed)
    print(json.dumps({
        "dataset": data.name,
        "method": args.method,
        "seed": args.seed,
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "hidden_dim": hidden,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }), flush=True)
    if args.method == "tarnet":
        train_metrics = train_tarnet(
            model, loader, device, epochs, learning_rate, method_config.get("log_every", 20),
            weight_decay=method_config["weight_decay"],
        )
    else:
        train_metrics = train_dragonnet(
            model,
            loader,
            device,
            epochs,
            learning_rate,
            method_config["propensity_strength"],
            method_config["targeted_strength"],
            method_config["propensity_clip"],
            method_config.get("log_every", 20),
            weight_decay=method_config["weight_decay"],
            max_grad_norm=method_config["max_grad_norm"],
        )
    evaluation = baseline_config["evaluation"][family]
    cuda_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        seed_everything(args.seed + 100_000)
        metrics = evaluate(
            model,
            args.method,
            data,
            device,
            evaluation["batch_size"],
            evaluation.get("sliced_wasserstein_projections", 1),
            args.seed + 200_000,
        )
    method_label = "TARNet" if args.method == "tarnet" else "DragonNet"
    if args.no_output:
        print(json.dumps(metrics, indent=2, allow_nan=True), flush=True)
        return
    run_name = args.run_name or f"{data.name}_{method_label}_seed{args.seed}"
    if Path(run_name).name != run_name:
        raise ValueError("run-name must not contain a path")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not args.no_checkpoint:
        torch.save(model.state_dict(), MODEL_DIR / f"{run_name}.pth")
    record = {
        "run_name": run_name,
        "dataset": data.name,
        "method": method_label,
        "seed": args.seed,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "outcome_dim": data.y_train.shape[1],
        "epochs": epochs,
        "batch_size": batch_size,
        "hidden_dim": hidden,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "wall_seconds": time.perf_counter() - started,
        **train_metrics,
        **metrics,
    }
    pd.DataFrame([record]).to_csv(RESULTS_DIR / f"{run_name}_metrics.csv", index=False)
    metadata = {
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "baseline_config": str(baseline_config_path),
        "baseline_config_sha256": _sha256(baseline_config_path),
        "data_source": str(data.source_path),
        "data_sha256": _sha256(data.source_path),
        "split_protocol": "exact seeded PO-Flow split",
        "preprocessing": "training-only X standardization; scalar Y standardized for fitting and inverted for evaluation",
        "port_note": "PyTorch scalar/vector adaptation of the published objective; not a bit-identical upstream execution",
        "upstream_repository": (
            "https://github.com/clinicalml/cfrnet"
            if args.method == "tarnet"
            else "https://github.com/claudiashi57/dragonnet"
        ),
        "upstream_commit": (
            "8fc242edbbd49ba6ff5dbca0201ed9b5bc79c5bb"
            if args.method == "tarnet"
            else "2e51b2478854df389893bac85430bca5b70f5a26"
        ),
        "x_center": data.x_center.squeeze(0).tolist(),
        "x_scale": data.x_scale.squeeze(0).tolist(),
        "outcome_center": data.outcome_center.squeeze(0).tolist(),
        "outcome_scale": data.outcome_scale.squeeze(0).tolist(),
        **record,
    }
    (RESULTS_DIR / f"{run_name}_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=True) + "\n")
    print(json.dumps(metrics, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
