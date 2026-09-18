"""Train overlap-aware PO-Flow on vector-valued potential outcomes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, Dataset

from ipw import IPWConfig, WeightedDataset, build_ipw, format_diagnostics
from main import FlowMatchingNet, TimeConditionedFlowMatchingNet, flow_to_noise, flow_to_sample
from main_IPW import seed_everything, train_flow_matching_ipw


PROJECT_DIR = Path(__file__).resolve().parent


class VectorPotentialOutcomeDataset(Dataset):
    def __init__(self, x, a, y0, y1, mu0, mu1, indices, test: bool):
        self.x_tensor = torch.as_tensor(x[indices], dtype=torch.float32)
        self.a_tensor = torch.as_tensor(a[indices], dtype=torch.int64)
        self.y0 = torch.as_tensor(y0[indices], dtype=torch.float32)
        self.y1 = torch.as_tensor(y1[indices], dtype=torch.float32)
        self.mu0 = torch.as_tensor(mu0[indices], dtype=torch.float32)
        self.mu1 = torch.as_tensor(mu1[indices], dtype=torch.float32)
        self.test = test

    def __len__(self):
        return len(self.a_tensor)

    def __getitem__(self, index):
        a = self.a_tensor[index]
        factual = torch.where(a == 0, self.y0[index], self.y1[index])
        if not self.test:
            return factual, a, self.x_tensor[index]
        counterfactual = torch.where(a == 0, self.y1[index], self.y0[index])
        return (
            factual,
            a,
            self.x_tensor[index],
            counterfactual,
            self.mu0[index],
            self.mu1[index],
            self.y0[index],
            self.y1[index],
        )


def load_archive(path: Path):
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}
    required = {"A", "X", "Y0", "Y1", "MU0", "MU1", "train_idx", "test_idx"}
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
    n = len(arrays["A"])
    if any(len(arrays[name]) != n for name in ("X", "Y0", "Y1", "MU0", "MU1")):
        raise ValueError(f"row mismatch in {path}")
    return arrays


def standardize_archive(arrays, image_scale: bool):
    train = arrays["train_idx"]
    a = arrays["A"]
    factual = np.where(a[:, None] == 0, arrays["Y0"], arrays["Y1"]).astype(np.float32)
    if image_scale:
        center = np.full((1, factual.shape[1]), 127.5, dtype=np.float32)
        scale = np.full((1, factual.shape[1]), 127.5, dtype=np.float32)
    else:
        center = factual[train].mean(axis=0, keepdims=True)
        scale = factual[train].std(axis=0, keepdims=True)
        scale = np.maximum(scale, 1e-6)
    transformed = dict(arrays)
    for name in ("Y0", "Y1", "MU0", "MU1"):
        transformed[name] = (arrays[name].astype(np.float32) - center) / scale
    return transformed, center.reshape(-1), scale.reshape(-1)


def standardize_covariates(arrays):
    """Standardize X using only training rows, matching the baseline protocol."""
    train = arrays["train_idx"]
    x = arrays["X"].astype(np.float32)
    center = x[train].mean(axis=0, keepdims=True)
    # torch.std used by the vector baselines applies Bessel's correction.
    scale = x[train].std(axis=0, ddof=1, keepdims=True)
    scale = np.maximum(scale, 1e-6)
    transformed = dict(arrays)
    transformed["X"] = (x - center) / scale
    return transformed, center.reshape(-1), scale.reshape(-1)


def potential_outcome_means(
    model, x, device, samples, ode_steps, return_draws=False, use_auxiliary_means=False
):
    """Estimate both potential-outcome means using paired latent draws.

    When requested, the returned draws have shape ``[units, samples, d_y]``.
    Keeping the Monte Carlo index explicit lets the vector evaluation mirror
    the scalar W1 metric: each index defines one independently generated
    test-set distribution.
    """
    batch = len(x)
    x_repeated = x.repeat_interleave(samples, dim=0)
    z = torch.randn(batch * samples, model.y_dim, device=device)
    a0 = torch.zeros(batch * samples, device=device)
    a1 = torch.ones(batch * samples, device=device)
    y0 = flow_to_sample(model, z, a0, x_repeated, device, train=False, num_steps=ode_steps)
    y1 = flow_to_sample(model, z, a1, x_repeated, device, train=False, num_steps=ode_steps)
    y0 = y0.view(batch, samples, -1)
    y1 = y1.view(batch, samples, -1)
    mean0, mean1 = y0.mean(1), y1.mean(1)
    if use_auxiliary_means:
        if not hasattr(model, "auxiliary_mean"):
            raise TypeError("auxiliary mean evaluation requires a model with auxiliary_mean")
        mean0 = model.auxiliary_mean(x, torch.zeros(batch, device=device))
        mean1 = model.auxiliary_mean(x, torch.ones(batch, device=device))
    if return_draws:
        return mean0, mean1, y0, y1
    return mean0, mean1


def sliced_wasserstein(x: torch.Tensor, y: torch.Tensor, projections: int, seed: int) -> float:
    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
        raise ValueError("sliced Wasserstein expects equally shaped [n,d] tensors")
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    directions = torch.randn(x.shape[1], projections, generator=generator, device=x.device)
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)
    x_projected = torch.sort(x @ directions, dim=0).values
    y_projected = torch.sort(y @ directions, dim=0).values
    return torch.mean(torch.abs(x_projected - y_projected)).item()


def evaluate(
    model, loader, device, samples, ode_steps, projections, seed, use_auxiliary_means=False
):
    model.eval()
    rows = {name: [] for name in ("cf_hat", "cf", "y0_hat", "y1_hat", "mu0", "mu1", "y0", "y1", "a")}
    y0_draw_rows, y1_draw_rows = [], []
    with torch.no_grad():
        for batch in loader:
            y, a, x, cf, mu0, mu1, y0, y1 = [value.to(device) for value in batch]
            z = flow_to_noise(model, y, a, x, device, train=False, num_steps=ode_steps)
            cf_hat = flow_to_sample(model, z, 1 - a, x, device, train=False, num_steps=ode_steps)
            y0_hat, y1_hat, y0_draw, y1_draw = potential_outcome_means(
                model,
                x,
                device,
                samples,
                ode_steps,
                return_draws=True,
                use_auxiliary_means=use_auxiliary_means,
            )
            for name, value in (
                ("cf_hat", cf_hat), ("cf", cf), ("y0_hat", y0_hat), ("y1_hat", y1_hat),
                ("mu0", mu0), ("mu1", mu1), ("y0", y0), ("y1", y1), ("a", a),
            ):
                rows[name].append(value.cpu())
            y0_draw_rows.append(y0_draw.cpu())
            y1_draw_rows.append(y1_draw.cpu())
    values = {name: torch.cat(parts).to(device) for name, parts in rows.items()}
    # Retain large Monte Carlo tensors in host memory and transfer only one
    # complete generated distribution at a time. This is essential for the
    # 784-dimensional image benchmark on a 16 GB V100.
    y0_draws = torch.cat(y0_draw_rows, dim=0)
    y1_draws = torch.cat(y1_draw_rows, dim=0)
    treatment = values["a"].view(-1, 1)
    po_hat = torch.where(treatment == 0, values["y0_hat"], values["y1_hat"])
    po_true = torch.where(treatment == 0, values["y0"], values["y1"])
    sw1_mean = 0.5 * (
        sliced_wasserstein(values["y0_hat"], values["mu0"], projections, seed)
        + sliced_wasserstein(values["y1_hat"], values["mu1"], projections, seed + 1)
    )
    sw1_by_draw = []
    for draw_index in range(samples):
        y0_draw = y0_draws[:, draw_index].to(device)
        y1_draw = y1_draws[:, draw_index].to(device)
        sw1_by_draw.append(
            0.5
            * (
                sliced_wasserstein(y0_draw, values["mu0"], projections, seed)
                + sliced_wasserstein(y1_draw, values["mu1"], projections, seed + 1)
            )
        )
    sw1_by_draw = np.asarray(sw1_by_draw, dtype=np.float64)
    return {
        "counterfactual_NRMSE": torch.sqrt(F.mse_loss(values["cf_hat"], values["cf"])).item(),
        "vector_PEHE": torch.sqrt(F.mse_loss(values["y1_hat"] - values["y0_hat"], values["mu1"] - values["mu0"])).item(),
        "PO_NRMSE": torch.sqrt(F.mse_loss(po_hat, po_true)).item(),
        "sliced_W1_distance": float(sw1_by_draw.mean()),
        "sliced_W1_distance_MC_SE": float(
            sw1_by_draw.std(ddof=1) / np.sqrt(len(sw1_by_draw)) if len(sw1_by_draw) > 1 else 0.0
        ),
        "sliced_W1_mean_distance": sw1_mean,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--config", required=True)
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--seed", type=int)
    result.add_argument("--epochs", type=int)
    result.add_argument("--batch-size", type=int)
    result.add_argument("--eval-samples", type=int)
    result.add_argument("--ode-steps", type=int)
    result.add_argument("--run-name")
    result.add_argument(
        "--ipw-strength",
        type=float,
        help="override YAML shrinkage strength; use 0 for the paired unweighted baseline",
    )
    result.add_argument("--propensity-only", action="store_true")
    return result


def main() -> None:
    run_started = time.perf_counter()
    cli = parser().parse_args()
    config_path = Path(cli.config)
    if not config_path.is_absolute():
        config_path = PROJECT_DIR / config_path
    config = yaml.safe_load(config_path.read_text())
    seed = cli.seed if cli.seed is not None else config.get("seed", 42)
    seed_everything(seed)
    if cli.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cli.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    archive_path = PROJECT_DIR / config["data"]["archive"]
    arrays = load_archive(archive_path)
    arrays, center, scale = standardize_archive(arrays, config["data"].get("image_scale", False))
    standardize_x = bool(config["data"].get("standardize_x", False))
    if standardize_x:
        arrays, x_center, x_scale = standardize_covariates(arrays)
    else:
        x_center = np.zeros(arrays["X"].shape[1], dtype=np.float32)
        x_scale = np.ones(arrays["X"].shape[1], dtype=np.float32)
    train = VectorPotentialOutcomeDataset(
        arrays["X"], arrays["A"], arrays["Y0"], arrays["Y1"], arrays["MU0"], arrays["MU1"], arrays["train_idx"], False
    )
    test = VectorPotentialOutcomeDataset(
        arrays["X"], arrays["A"], arrays["Y0"], arrays["Y1"], arrays["MU0"], arrays["MU1"], arrays["test_idx"], True
    )
    ipw_values = dict(config.get("ipw", {}))
    ipw_values["seed"] = seed
    if cli.ipw_strength is not None:
        ipw_values["ipw_strength"] = cli.ipw_strength
    ipw_config = IPWConfig.from_mapping(ipw_values)
    ipw_result = build_ipw(train.x_tensor.numpy(), train.a_tensor.numpy(), ipw_config)
    print(format_diagnostics(ipw_result.diagnostics))

    run_name = cli.run_name or config["data"]["name"]
    if Path(run_name).name != run_name:
        raise ValueError("run-name must not contain a path")
    results_dir = PROJECT_DIR / "results" / "multidim"
    model_dir = PROJECT_DIR / "NN_params" / "multidim"
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = {
        **ipw_result.diagnostics,
        "archive": str(archive_path),
        "config": str(config_path),
        "outcome_dim": int(arrays["Y0"].shape[1]),
        "x_dim": int(arrays["X"].shape[1]),
        "center": center.tolist(),
        "scale": scale.tolist(),
        "standardize_x": standardize_x,
        "x_center": x_center.tolist(),
        "x_scale": x_scale.tolist(),
    }
    (results_dir / f"{run_name}_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    if cli.propensity_only:
        return

    training = config["training"]
    evaluation = config["evaluation"]
    epochs = cli.epochs if cli.epochs is not None else training["epochs"]
    batch_size = cli.batch_size if cli.batch_size is not None else training["batch_size"]
    eval_samples = cli.eval_samples if cli.eval_samples is not None else evaluation["samples"]
    ode_steps = cli.ode_steps if cli.ode_steps is not None else evaluation["ode_steps"]
    weighted = WeightedDataset(train, ipw_result.weight)
    train_loader = DataLoader(weighted, batch_size=batch_size, shuffle=True, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test, batch_size=evaluation.get("batch_size", batch_size), shuffle=False)
    architecture = training.get("architecture", "legacy")
    if architecture == "legacy":
        model = FlowMatchingNet(
            y_dim=arrays["Y0"].shape[1],
            x_dim=arrays["X"].shape[1],
            hidden_dim=training.get("hidden_dim", 128),
        ).to(device)
    elif architecture == "time_residual":
        model = TimeConditionedFlowMatchingNet(
            y_dim=arrays["Y0"].shape[1],
            x_dim=arrays["X"].shape[1],
            core_width=training.get("core_width", 192),
            depth=training.get("depth", 3),
            time_dim=training.get("time_embedding_dim", 32),
            expansion=training.get("residual_expansion", 2),
            dropout=training.get("dropout", 0.0),
            head_mode=training.get("head_mode", "separate"),
        ).to(device)
    else:
        raise ValueError(f"unknown training architecture: {architecture}")
    learning_rate = training.get("learning_rate", 1e-3)
    weight_decay = training.get("weight_decay", 0.0)
    mean_aux_strength = float(training.get("mean_aux_strength", 0.0))
    if mean_aux_strength < 0.0:
        raise ValueError("mean_aux_strength must be nonnegative")
    use_auxiliary_means = bool(training.get("use_auxiliary_means", mean_aux_strength > 0.0))
    if use_auxiliary_means and mean_aux_strength <= 0.0:
        raise ValueError("use_auxiliary_means requires a positive mean_aux_strength")
    if training.get("optimizer", "adam").lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = None
    if training.get("scheduler", "none").lower() == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=training.get("minimum_learning_rate", learning_rate * 0.05)
        )
    swa_fraction = training.get("swa_start_fraction", 0.8)
    swa_start = max(1, int(np.ceil(epochs * swa_fraction)))
    averaged = AveragedModel(model)
    history = []
    for epoch in range(1, epochs + 1):
        train_metrics = train_flow_matching_ipw(
            model,
            train_loader,
            optimizer,
            device,
            training.get("max_grad_norm"),
            mean_aux_strength=mean_aux_strength,
        )
        if scheduler is not None:
            scheduler.step()
        if epoch >= swa_start:
            averaged.update_parameters(model)
        if epoch == 1 or epoch == epochs or epoch % training.get("log_every", 10) == 0:
            print(f"epoch={epoch} weighted_loss={train_metrics['weighted_flow_matching_loss']:.6f}")
    evaluation_model = averaged.module if epochs >= swa_start else model
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.type == "cuda" else []):
        seed_everything(seed + 100_000)
        metrics = evaluate(
            evaluation_model, test_loader, device, eval_samples, ode_steps,
            evaluation.get("sliced_wasserstein_projections", 100), seed + 200_000,
            use_auxiliary_means=use_auxiliary_means,
        )
    history.append(
        {
            "run_name": run_name,
            "dataset": config["data"]["name"],
            "seed": seed,
            "architecture": architecture,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "ipw_strength": ipw_config.ipw_strength,
            "standardize_x": standardize_x,
            "mean_aux_strength": mean_aux_strength,
            "use_auxiliary_means": use_auxiliary_means,
            "epoch": epochs,
            "eval_samples": eval_samples,
            "ode_steps": ode_steps,
            "sliced_wasserstein_projections": evaluation.get("sliced_wasserstein_projections", 100),
            "wall_seconds": time.perf_counter() - run_started,
            **train_metrics,
            **metrics,
        }
    )
    pd.DataFrame(history).to_csv(results_dir / f"{run_name}_metrics.csv", index=False)
    torch.save(evaluation_model.state_dict(), model_dir / f"{run_name}_swa.pth")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
