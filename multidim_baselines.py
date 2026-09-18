"""Vector-outcome ports of the CFRNet, GANITE, and CEVAE baselines.

The upstream repositories in ../PO_comparisons are TensorFlow-1/Edward code
with scalar outcome heads.  This module leaves those dirty working trees
untouched and ports their defining objectives to the project's PyTorch
environment.  All methods consume the exact processed archives and splits used
by main_multidim.py.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset

from main_IPW import seed_everything
from main_multidim import load_archive, sliced_wasserstein, standardize_archive


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results" / "multidim_baselines"
MODEL_DIR = PROJECT_DIR / "NN_params" / "multidim_baselines"


def mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    modules = []
    current = in_dim
    for _ in range(layers):
        modules.extend((nn.Linear(current, hidden), nn.ELU()))
        current = hidden
    modules.append(nn.Linear(current, out_dim))
    return nn.Sequential(*modules)


class CFRVector(nn.Module):
    """CFRNet with shared representation and vector-valued treatment heads."""

    def __init__(self, x_dim: int, y_dim: int, hidden: int):
        super().__init__()
        self.representation = mlp(x_dim, hidden, hidden, layers=2)
        self.head0 = mlp(hidden, hidden, y_dim, layers=2)
        self.head1 = mlp(hidden, hidden, y_dim, layers=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        representation = self.representation(x)
        return self.head0(representation), self.head1(representation), representation


def linear_mmd(treated: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    if len(treated) < 2 or len(control) < 2:
        return treated.new_zeros(())
    # CFRNet exposes linear MMD as one of its native IPM choices.  It is O(bh)
    # instead of the O(b^2 h) RBF variant and is therefore the defensible
    # choice for the 70,000-unit image benchmark under a fixed GPU budget.
    difference = treated.mean(dim=0) - control.mean(dim=0)
    return difference.square().mean()


class GANITEVector(nn.Module):
    """Continuous vector-outcome adaptation of GANITE's three networks."""

    def __init__(self, x_dim: int, y_dim: int, hidden: int, output_bound: float = 5.0):
        super().__init__()
        self.y_dim = y_dim
        self.output_bound = output_bound
        self.generator_body = mlp(x_dim + y_dim + 1, hidden, hidden, layers=2)
        self.generator0 = mlp(hidden, hidden, y_dim, layers=1)
        self.generator1 = mlp(hidden, hidden, y_dim, layers=1)
        self.discriminator = mlp(x_dim + 2 * y_dim, hidden, 1, layers=2)
        self.inference_body = mlp(x_dim, hidden, hidden, layers=2)
        self.inference0 = mlp(hidden, hidden, y_dim, layers=1)
        self.inference1 = mlp(hidden, hidden, y_dim, layers=1)

    def generate(self, x: torch.Tensor, a: torch.Tensor, y: torch.Tensor):
        h = self.generator_body(torch.cat((x, a[:, None], y), dim=1))
        return self._bounded(self.generator0(h)), self._bounded(self.generator1(h))

    def discriminate(self, x: torch.Tensor, y0: torch.Tensor, y1: torch.Tensor):
        return self.discriminator(torch.cat((x, y0, y1), dim=1)).squeeze(1)

    def infer(self, x: torch.Tensor):
        h = self.inference_body(x)
        return self._bounded(self.inference0(h)), self._bounded(self.inference1(h))

    def _bounded(self, value: torch.Tensor):
        # The upstream binary-outcome implementation uses sigmoid-bounded
        # heads.  A smooth +/-5 bound is its continuous standardized-outcome
        # analogue and prevents an adversary from winning by sending an
        # unidentified counterfactual head to infinity.
        return self.output_bound * torch.tanh(value / self.output_bound)


class CEVAEVector(nn.Module):
    """Factorized-Gaussian vector port of CEVAE's generative/recognition model."""

    def __init__(self, x_dim: int, y_dim: int, hidden: int, z_dim: int):
        super().__init__()
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.encoder = mlp(x_dim + y_dim + 1, hidden, 2 * z_dim, layers=2)
        self.decoder_x = mlp(z_dim, hidden, x_dim, layers=2)
        self.decoder_t = mlp(z_dim, hidden, 1, layers=1)
        self.decoder_y0 = mlp(z_dim, hidden, y_dim, layers=2)
        self.decoder_y1 = mlp(z_dim, hidden, y_dim, layers=2)
        self.aux_t = mlp(x_dim, hidden, 1, layers=1)
        self.aux_y0 = mlp(x_dim, hidden, y_dim, layers=2)
        self.aux_y1 = mlp(x_dim, hidden, y_dim, layers=2)
        self.log_sigma_y = nn.Parameter(torch.zeros(y_dim))

    def encode(self, x: torch.Tensor, a: torch.Tensor, y: torch.Tensor):
        stats = self.encoder(torch.cat((x, a[:, None], y), dim=1))
        mean, logvar = stats.chunk(2, dim=1)
        return mean, logvar.clamp(-8.0, 8.0)

    @staticmethod
    def sample_z(mean: torch.Tensor, logvar: torch.Tensor):
        return mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)

    def decode_y(self, z: torch.Tensor, a: torch.Tensor):
        y0, y1 = self.decoder_y0(z), self.decoder_y1(z)
        return torch.where(a[:, None] == 0, y0, y1)

    def auxiliary_y(self, x: torch.Tensor, a: torch.Tensor):
        y0, y1 = self.aux_y0(x), self.aux_y1(x)
        return torch.where(a[:, None] == 0, y0, y1)

    def arm_latent(self, x: torch.Tensor, arm: int):
        a = torch.full((len(x),), float(arm), device=x.device)
        y_aux = self.auxiliary_y(x, a)
        mean, logvar = self.encode(x, a, y_aux)
        return a, mean, logvar


def factual(y0: torch.Tensor, y1: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return torch.where(a[:, None] == 0, y0, y1)


def make_loader(x, a, y, batch_size: int, seed: int, shuffle: bool = True):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(x, a, y), batch_size=batch_size, shuffle=shuffle,
        generator=generator if shuffle else None, drop_last=False,
    )


def train_cfr(model, loader, device, epochs, lr, balance_strength, log_every):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    final = {}
    for epoch in range(1, epochs + 1):
        model.train()
        totals = np.zeros(3, dtype=np.float64)
        batches = 0
        for x, a, y in loader:
            x, a, y = x.to(device), a.to(device), y.to(device)
            y0, y1, representation = model(x)
            prediction = factual(y0, y1, a)
            prediction_loss = F.mse_loss(prediction, y)
            balance_loss = linear_mmd(representation[a == 1], representation[a == 0])
            loss = prediction_loss + balance_strength * balance_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals += (loss.item(), prediction_loss.item(), balance_loss.item())
            batches += 1
        final = dict(loss=totals[0] / batches, factual_loss=totals[1] / batches, balance_loss=totals[2] / batches)
        if epoch == 1 or epoch == epochs or epoch % log_every == 0:
            print(f"epoch={epoch} loss={final['loss']:.6f} factual={final['factual_loss']:.6f} mmd={final['balance_loss']:.6f}", flush=True)
    return final


def set_requires_grad(module: nn.Module, enabled: bool):
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def train_ganite(model, loader, device, epochs, lr, adversarial_strength, log_every):
    generator_parameters = list(model.generator_body.parameters()) + list(model.generator0.parameters()) + list(model.generator1.parameters())
    inference_parameters = list(model.inference_body.parameters()) + list(model.inference0.parameters()) + list(model.inference1.parameters())
    optimizer_g = torch.optim.Adam(generator_parameters, lr=lr, weight_decay=1e-5)
    optimizer_d = torch.optim.Adam(model.discriminator.parameters(), lr=lr, weight_decay=1e-5)
    final = {}
    for epoch in range(1, epochs + 1):
        totals = np.zeros(3, dtype=np.float64)
        batches = 0
        for x, a, y in loader:
            x, a, y = x.to(device), a.to(device), y.to(device)
            with torch.no_grad():
                generated0, generated1 = model.generate(x, a, y)
                completed0 = torch.where(a[:, None] == 0, y, generated0)
                completed1 = torch.where(a[:, None] == 1, y, generated1)
            discriminator_loss = F.binary_cross_entropy_with_logits(
                model.discriminate(x, completed0, completed1), a.float()
            )
            optimizer_d.zero_grad(set_to_none=True)
            discriminator_loss.backward()
            optimizer_d.step()

            set_requires_grad(model.discriminator, False)
            generated0, generated1 = model.generate(x, a, y)
            completed0 = torch.where(a[:, None] == 0, y, generated0)
            completed1 = torch.where(a[:, None] == 1, y, generated1)
            generated_factual = factual(generated0, generated1, a)
            factual_loss = F.mse_loss(generated_factual, y)
            adversarial_loss = F.binary_cross_entropy_with_logits(
                model.discriminate(x, completed0, completed1), 1.0 - a.float()
            )
            generator_loss = factual_loss + adversarial_strength * adversarial_loss
            optimizer_g.zero_grad(set_to_none=True)
            generator_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator_parameters, 5.0)
            optimizer_g.step()
            set_requires_grad(model.discriminator, True)
            totals += (generator_loss.item(), factual_loss.item(), discriminator_loss.item())
            batches += 1
        final = dict(loss=totals[0] / batches, factual_loss=totals[1] / batches, discriminator_loss=totals[2] / batches)
        if epoch == 1 or epoch == epochs or epoch % log_every == 0:
            print(f"adv_epoch={epoch} generator={final['loss']:.6f} factual={final['factual_loss']:.6f} discriminator={final['discriminator_loss']:.6f}", flush=True)

    optimizer_i = torch.optim.Adam(inference_parameters, lr=lr, weight_decay=1e-5)
    for epoch in range(1, epochs + 1):
        total = 0.0
        batches = 0
        for x, a, y in loader:
            x, a, y = x.to(device), a.to(device), y.to(device)
            with torch.no_grad():
                generated0, generated1 = model.generate(x, a, y)
                target0 = torch.where(a[:, None] == 0, y, generated0)
                target1 = torch.where(a[:, None] == 1, y, generated1)
            inferred0, inferred1 = model.infer(x)
            loss = F.mse_loss(inferred0, target0) + F.mse_loss(inferred1, target1)
            optimizer_i.zero_grad(set_to_none=True)
            loss.backward()
            optimizer_i.step()
            total += loss.item()
            batches += 1
        final["inference_loss"] = total / batches
        if epoch == 1 or epoch == epochs or epoch % log_every == 0:
            print(f"inference_epoch={epoch} loss={final['inference_loss']:.6f}", flush=True)
    return final


def train_cevae(model, loader, device, epochs, lr, beta_kl, x_weight, log_every):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    final = {}
    for epoch in range(1, epochs + 1):
        totals = np.zeros(6, dtype=np.float64)
        batches = 0
        for x, a, y in loader:
            x, a, y = x.to(device), a.to(device), y.to(device)
            mean, logvar = model.encode(x, a.float(), y)
            z = model.sample_z(mean, logvar)
            y_mean = model.decode_y(z, a.float())
            sigma = torch.exp(model.log_sigma_y.clamp(-4.0, 2.0))[None, :]
            y_nll = torch.mean(0.5 * ((y - y_mean) / sigma).square() + torch.log(sigma))
            x_loss = F.mse_loss(model.decoder_x(z), x)
            t_loss = F.binary_cross_entropy_with_logits(model.decoder_t(z).squeeze(1), a.float())
            kl = 0.5 * torch.mean(torch.sum(mean.square() + logvar.exp() - 1.0 - logvar, dim=1))
            aux_t = F.binary_cross_entropy_with_logits(model.aux_t(x).squeeze(1), a.float())
            aux_y = F.mse_loss(model.auxiliary_y(x, a.float()), y)
            loss = y_nll + x_weight * x_loss + t_loss + beta_kl * kl + aux_t + aux_y
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            totals += (loss.item(), y_nll.item(), x_loss.item(), t_loss.item(), kl.item(), aux_y.item())
            batches += 1
        averages = totals / batches
        final = dict(loss=averages[0], outcome_nll=averages[1], x_loss=averages[2], treatment_loss=averages[3], kl=averages[4], auxiliary_outcome_loss=averages[5])
        if epoch == 1 or epoch == epochs or epoch % log_every == 0:
            print(f"epoch={epoch} loss={final['loss']:.6f} y_nll={final['outcome_nll']:.6f} kl={final['kl']:.6f} aux_y={final['auxiliary_outcome_loss']:.6f}", flush=True)
    return final


@torch.no_grad()
def predict_point(model, method: str, x: torch.Tensor, a: torch.Tensor, y: torch.Tensor):
    if method == "cfr":
        y0, y1, _ = model(x)
        cf = torch.where(a[:, None] == 0, y1, y0)
    elif method == "ganite":
        y0, y1 = model.infer(x)
        generated0, generated1 = model.generate(x, a.float(), y)
        cf = torch.where(a[:, None] == 0, generated1, generated0)
    elif method == "cevae":
        a0, mean0, _ = model.arm_latent(x, 0)
        a1, mean1, _ = model.arm_latent(x, 1)
        y0 = model.decode_y(mean0, a0)
        y1 = model.decode_y(mean1, a1)
        factual_mean, _ = model.encode(x, a.float(), y)
        cf = model.decode_y(factual_mean, 1.0 - a.float())
    else:
        raise ValueError(method)
    return y0, y1, cf


@torch.no_grad()
def cevae_distributional_sw1(model, x, mu0, mu1, device, batch_size, samples, projections, seed):
    values = []
    generator = torch.Generator(device=device).manual_seed(seed)
    for draw in range(samples):
        arms = []
        for arm in (0, 1):
            predictions = []
            for start in range(0, len(x), batch_size):
                xb = x[start : start + batch_size].to(device)
                ab, mean, logvar = model.arm_latent(xb, arm)
                noise = torch.randn(mean.shape, generator=generator, device=device)
                z = mean + noise * torch.exp(0.5 * logvar)
                prediction = model.decode_y(z, ab)
                observation_noise = torch.randn(prediction.shape, generator=generator, device=device)
                prediction = prediction + observation_noise * torch.exp(model.log_sigma_y.clamp(-4.0, 2.0))[None, :]
                predictions.append(prediction.cpu())
            arms.append(torch.cat(predictions).to(device))
        sw1 = 0.5 * (
            sliced_wasserstein(arms[0], mu0.to(device), projections, seed + 10)
            + sliced_wasserstein(arms[1], mu1.to(device), projections, seed + 11)
        )
        values.append(sw1)
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(len(values)) if len(values) > 1 else 0.0)


@torch.no_grad()
def evaluate(model, method, x, a, y, y0_true, y1_true, mu0, mu1, device, batch_size, projections, eval_samples, seed):
    model.eval()
    y0_parts, y1_parts, cf_parts = [], [], []
    for start in range(0, len(x), batch_size):
        xb = x[start : start + batch_size].to(device)
        ab = a[start : start + batch_size].to(device)
        yb = y[start : start + batch_size].to(device)
        p0, p1, pcf = predict_point(model, method, xb, ab, yb)
        y0_parts.append(p0.cpu())
        y1_parts.append(p1.cpu())
        cf_parts.append(pcf.cpu())
    y0_hat, y1_hat, cf_hat = map(torch.cat, (y0_parts, y1_parts, cf_parts))
    cf_true = torch.where(a[:, None] == 0, y1_true, y0_true)
    po_hat = factual(y0_hat, y1_hat, a)
    po_true = factual(y0_true, y1_true, a)
    sw1_mean = 0.5 * (
        sliced_wasserstein(y0_hat.to(device), mu0.to(device), projections, seed + 20)
        + sliced_wasserstein(y1_hat.to(device), mu1.to(device), projections, seed + 21)
    )
    distributional, distributional_se = float("nan"), float("nan")
    if method == "cevae":
        distributional, distributional_se = cevae_distributional_sw1(
            model, x, mu0, mu1, device, batch_size, eval_samples, projections, seed + 30
        )
    return {
        "counterfactual_NRMSE": torch.sqrt(F.mse_loss(cf_hat, cf_true)).item(),
        "vector_PEHE": torch.sqrt(F.mse_loss(y1_hat - y0_hat, mu1 - mu0)).item(),
        "PO_NRMSE": torch.sqrt(F.mse_loss(po_hat, po_true)).item(),
        "sliced_W1_distance": distributional,
        "sliced_W1_distance_MC_SE": distributional_se,
        "sliced_W1_mean_distance": sw1_mean,
        "distributional_metric_applicable": method == "cevae",
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--baseline-config",
        default="configs/baselines/Multidim_Baselines.yaml",
    )
    parser.add_argument("--method", choices=("cfr", "ganite", "cevae"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--no-checkpoint", action="store_true")
    return parser.parse_args()


def main():
    started = time.perf_counter()
    args = parse_args()
    seed_everything(args.seed)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_DIR / config_path
    config = yaml.safe_load(config_path.read_text())
    baseline_config_path = Path(args.baseline_config)
    if not baseline_config_path.is_absolute():
        baseline_config_path = PROJECT_DIR / baseline_config_path
    baseline_config = yaml.safe_load(baseline_config_path.read_text())
    archive_path = PROJECT_DIR / config["data"]["archive"]
    arrays = load_archive(archive_path)
    arrays, _, _ = standardize_archive(arrays, config["data"].get("image_scale", False))
    train_idx, test_idx = arrays["train_idx"], arrays["test_idx"]
    x_train = torch.as_tensor(arrays["X"][train_idx], dtype=torch.float32)
    x_test = torch.as_tensor(arrays["X"][test_idx], dtype=torch.float32)
    x_center = x_train.mean(0, keepdim=True)
    x_scale = x_train.std(0, keepdim=True).clamp_min(1e-6)
    x_train = (x_train - x_center) / x_scale
    x_test = (x_test - x_center) / x_scale
    a_train = torch.as_tensor(arrays["A"][train_idx], dtype=torch.long)
    a_test = torch.as_tensor(arrays["A"][test_idx], dtype=torch.long)
    y0_train = torch.as_tensor(arrays["Y0"][train_idx], dtype=torch.float32)
    y1_train = torch.as_tensor(arrays["Y1"][train_idx], dtype=torch.float32)
    y_train = factual(y0_train, y1_train, a_train)
    y0_test = torch.as_tensor(arrays["Y0"][test_idx], dtype=torch.float32)
    y1_test = torch.as_tensor(arrays["Y1"][test_idx], dtype=torch.float32)
    mu0_test = torch.as_tensor(arrays["MU0"][test_idx], dtype=torch.float32)
    mu1_test = torch.as_tensor(arrays["MU1"][test_idx], dtype=torch.float32)
    y_test = factual(y0_test, y1_test, a_test)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    baseline = baseline_config["baselines"]
    method_config = baseline[args.method]
    dimension_key = "image" if config["data"].get("image_scale", False) else "tabular"
    epochs = args.epochs or method_config[dimension_key]["epochs"]
    batch_size = args.batch_size or method_config[dimension_key]["batch_size"]
    hidden = method_config[dimension_key]["hidden_dim"]
    learning_rate = method_config[dimension_key]["learning_rate"]
    y_dim = y_train.shape[1]
    if args.method == "cfr":
        model = CFRVector(x_train.shape[1], y_dim, hidden).to(device)
    elif args.method == "ganite":
        model = GANITEVector(x_train.shape[1], y_dim, hidden).to(device)
    else:
        model = CEVAEVector(x_train.shape[1], y_dim, hidden, method_config[dimension_key]["z_dim"]).to(device)
    loader = make_loader(x_train, a_train, y_train, batch_size, args.seed)
    print(json.dumps({"dataset": config["data"]["name"], "method": args.method, "seed": args.seed, "device": str(device), "epochs": epochs, "batch_size": batch_size, "hidden_dim": hidden}), flush=True)
    if args.method == "cfr":
        train_metrics = train_cfr(model, loader, device, epochs, learning_rate, method_config["balance_strength"], method_config.get("log_every", 20))
    elif args.method == "ganite":
        train_metrics = train_ganite(model, loader, device, epochs, learning_rate, method_config["adversarial_strength"], method_config.get("log_every", 20))
    else:
        train_metrics = train_cevae(model, loader, device, epochs, learning_rate, method_config["beta_kl"], method_config["x_reconstruction_weight"], method_config.get("log_every", 20))
    evaluation = baseline["evaluation"][dimension_key]
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.type == "cuda" else []):
        seed_everything(args.seed + 100_000)
        metrics = evaluate(
            model, args.method, x_test, a_test, y_test, y0_test, y1_test, mu0_test, mu1_test,
            device, evaluation["batch_size"], evaluation["sliced_wasserstein_projections"],
            evaluation["cevae_samples"], args.seed + 200_000,
        )
    run_name = args.run_name or f"{config['data']['name']}_{args.method}_seed{args.seed}"
    if Path(run_name).name != run_name:
        raise ValueError("run-name must not contain a path")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "run_name": run_name,
        "dataset": config["data"]["name"],
        "method": args.method.upper() if args.method != "ganite" else "GANITE",
        "seed": args.seed,
        "outcome_dim": y_dim,
        "epochs": epochs,
        "batch_size": batch_size,
        "hidden_dim": hidden,
        "wall_seconds": time.perf_counter() - started,
        **train_metrics,
        **metrics,
    }
    pd.DataFrame([record]).to_csv(RESULTS_DIR / f"{run_name}_metrics.csv", index=False)
    metadata = {
        "archive": str(archive_path),
        "config": str(config_path),
        "baseline_config": str(baseline_config_path),
        "upstream_repository": str(PROJECT_DIR.parent / "PO_comparisons" / {"cfr": "cfrnet", "ganite": "GANITE", "cevae": "CEVAE"}[args.method]),
        "port_note": "PyTorch vector-outcome port; upstream TensorFlow-1/Edward tree left untouched",
        "x_center": x_center.squeeze(0).tolist(),
        "x_scale": x_scale.squeeze(0).tolist(),
        **record,
    }
    (RESULTS_DIR / f"{run_name}_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=True) + "\n")
    if not args.no_checkpoint:
        torch.save(model.state_dict(), MODEL_DIR / f"{run_name}.pth")
    print(json.dumps(metrics, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
