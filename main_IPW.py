"""Train PO-Flow with cross-fitted inverse-probability weighting.

This is an ablation-compatible entry point: it reuses the PO-Flow architecture
and ODE routines from ``main.py`` while changing only the empirical
flow-matching objective and adding propensity/overlap diagnostics.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader

from ipw import IPWConfig, WeightedDataset, build_ipw, format_diagnostics
from main import FlowMatchingNet, flow_to_noise, flow_to_sample
from utils import CounterfactualDatasetBuilder, wasserstein_dist


PROJECT_DIR = Path(__file__).resolve().parent


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_flow_matching_ipw(
    model,
    dataloader,
    optimizer,
    device,
    max_grad_norm: Optional[float] = None,
    mean_aux_strength: float = 0.0,
) -> Dict[str, float]:
    """Run one epoch of IPW conditional flow matching."""

    model.train()
    weighted_loss_sum = 0.0
    unweighted_loss_sum = 0.0
    auxiliary_loss_sum = 0.0
    sample_count = 0

    for y_data, treatment, x, weight in dataloader:
        y_data = y_data.to(device)
        treatment = treatment.to(device).float().view(-1, 1)
        x = x.to(device)
        weight = weight.to(device).float().view(-1)
        batch_size = y_data.size(0)

        z = torch.randn(batch_size, model.y_dim, device=device)
        t = torch.rand(batch_size, 1, device=device)
        y_t = (1.0 - t) * y_data + t * z
        target_velocity = z - y_data

        velocity_0, velocity_1 = model(y_t, t, treatment, x)
        predicted_velocity = torch.where(treatment == 0, velocity_0, velocity_1)
        per_sample_loss = torch.square(predicted_velocity - target_velocity).flatten(1).mean(1)
        per_sample_objective = per_sample_loss
        per_sample_auxiliary = torch.zeros_like(per_sample_loss)
        if mean_aux_strength > 0.0:
            if not hasattr(model, "auxiliary_mean"):
                raise TypeError("mean auxiliary loss requires a model with auxiliary_mean")
            auxiliary_prediction = model.auxiliary_mean(x, treatment)
            per_sample_auxiliary = torch.square(auxiliary_prediction - y_data).flatten(1).mean(1)
            per_sample_objective = per_sample_objective + mean_aux_strength * per_sample_auxiliary
        loss = torch.mean(weight * per_sample_objective)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        weighted_loss_sum += torch.sum(weight * per_sample_loss).item()
        unweighted_loss_sum += torch.sum(per_sample_loss).item()
        auxiliary_loss_sum += torch.sum(weight * per_sample_auxiliary).item()
        sample_count += batch_size

    return {
        "weighted_flow_matching_loss": weighted_loss_sum / sample_count,
        "unweighted_flow_matching_loss": unweighted_loss_sum / sample_count,
        "weighted_auxiliary_mean_loss": auxiliary_loss_sum / sample_count,
    }


def potential_outcome_means(
    model,
    x: torch.Tensor,
    device: torch.device,
    samples: int,
    ode_steps: int,
    return_distribution_draw: bool = False,
):
    """Estimate PO means and optionally retain all paired distribution draws."""

    batch_size = x.size(0)
    x_repeated = x.repeat_interleave(samples, dim=0)
    treatment_0 = torch.zeros(batch_size * samples, device=device)
    treatment_1 = torch.ones(batch_size * samples, device=device)
    z = torch.randn(batch_size * samples, model.y_dim, device=device)
    y_0 = flow_to_sample(
        model,
        z,
        treatment_0,
        x_repeated,
        device,
        train=False,
        num_steps=ode_steps,
    )
    y_1 = flow_to_sample(
        model,
        z,
        treatment_1,
        x_repeated,
        device,
        train=False,
        num_steps=ode_steps,
    )
    y_0 = y_0.view(batch_size, samples, -1)
    y_1 = y_1.view(batch_size, samples, -1)
    distribution_y_0 = y_0
    distribution_y_1 = y_1
    y_0 = distribution_y_0.mean(dim=1)
    y_1 = distribution_y_1.mean(dim=1)
    if return_distribution_draw:
        return y_0, y_1, distribution_y_0, distribution_y_1
    return y_0, y_1


def evaluate_kl(
    model,
    x: torch.Tensor,
    mu_0: torch.Tensor,
    mu_1: torch.Tensor,
    device: torch.device,
    samples: int,
    ode_steps: int,
) -> Dict[str, float]:
    """Mirror the Gaussian semi-synthetic KL diagnostic from the baseline."""

    batch_size = x.size(0)
    x_repeated = x.repeat_interleave(samples, dim=0)
    a_0 = torch.zeros(batch_size * samples, 1, device=device)
    a_1 = torch.ones(batch_size * samples, 1, device=device)
    z = torch.randn(batch_size * samples, model.y_dim, device=device)
    mu_0_repeated = mu_0.view(-1, 1).repeat_interleave(samples, dim=0)
    mu_1_repeated = mu_1.view(-1, 1).repeat_interleave(samples, dim=0)

    y_0, log_q_0 = flow_to_sample(
        model,
        z,
        a_0,
        x_repeated,
        device,
        train=False,
        nll=True,
        num_steps=ode_steps,
    )
    y_1, log_q_1 = flow_to_sample(
        model,
        z,
        a_1,
        x_repeated,
        device,
        train=False,
        nll=True,
        num_steps=ode_steps,
    )
    log_p_0 = -0.5 * (torch.square(y_0 - mu_0_repeated) + np.log(2.0 * np.pi))
    log_p_1 = -0.5 * (torch.square(y_1 - mu_1_repeated) + np.log(2.0 * np.pi))
    kl_0 = (log_q_0.view(-1, 1) - log_p_0).view(batch_size, samples).mean(dim=1)
    kl_1 = (log_q_1.view(-1, 1) - log_p_1).view(batch_size, samples).mean(dim=1)
    return {"KL_y0": kl_0.mean().item(), "KL_y1": kl_1.mean().item()}


def evaluate(
    model,
    dataloader,
    device: torch.device,
    potential_outcome_samples: int,
    ode_steps: int,
    potential_ode_steps: Optional[int] = None,
    kl_samples: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate PO, CATE, distributional, and factual-conditioned CF metrics."""

    model.eval()
    estimated_counterfactuals = []
    true_counterfactuals = []
    estimated_y_0 = []
    estimated_y_1 = []
    distribution_y_0 = []
    distribution_y_1 = []
    true_y_0 = []
    true_y_1 = []
    true_mu_0 = []
    true_mu_1 = []
    treatments = []
    kl_rows = []

    with torch.no_grad():
        for batch in dataloader:
            y, treatment, x, counterfactual_y, mu_0, mu_1 = [value.to(device) for value in batch]
            y = y.view(-1, 1)
            treatment_column = treatment.float().view(-1, 1)
            counterfactual_y = counterfactual_y.view(-1, 1)

            factual_z = flow_to_noise(
                model,
                y,
                treatment,
                x,
                device,
                train=False,
                num_steps=ode_steps,
            )
            mapped_counterfactual = flow_to_sample(
                model,
                factual_z,
                1 - treatment,
                x,
                device,
                train=False,
                num_steps=ode_steps,
            )
            y_0_hat, y_1_hat, y_0_draw, y_1_draw = potential_outcome_means(
                model,
                x,
                device,
                samples=potential_outcome_samples,
                ode_steps=potential_ode_steps or ode_steps,
                return_distribution_draw=True,
            )

            y_0 = y * (1.0 - treatment_column) + counterfactual_y * treatment_column
            y_1 = counterfactual_y * (1.0 - treatment_column) + y * treatment_column

            estimated_counterfactuals.append(mapped_counterfactual.cpu())
            true_counterfactuals.append(counterfactual_y.cpu())
            estimated_y_0.append(y_0_hat.view(-1, 1).cpu())
            estimated_y_1.append(y_1_hat.view(-1, 1).cpu())
            distribution_y_0.append(y_0_draw.cpu())
            distribution_y_1.append(y_1_draw.cpu())
            true_y_0.append(y_0.cpu())
            true_y_1.append(y_1.cpu())
            true_mu_0.append(mu_0.view(-1, 1).cpu())
            true_mu_1.append(mu_1.view(-1, 1).cpu())
            treatments.append(treatment_column.cpu())

            if kl_samples is not None:
                kl_rows.append(
                    evaluate_kl(model, x, mu_0, mu_1, device, kl_samples, ode_steps)
                )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    concatenate = lambda values: torch.cat(values, dim=0)
    counterfactual_hat = concatenate(estimated_counterfactuals)
    counterfactual_true = concatenate(true_counterfactuals)
    y_0_hat = concatenate(estimated_y_0)
    y_1_hat = concatenate(estimated_y_1)
    # [test units, Monte Carlo draws, outcome dimension].  W1 is computed on
    # the full test-set distribution for every draw index and then averaged.
    # With one draw this exactly reproduces the legacy metric, while journal
    # evaluation can reduce Monte Carlo error by using many independent draws.
    y_0_draw = concatenate(distribution_y_0)
    y_1_draw = concatenate(distribution_y_1)
    y_0 = concatenate(true_y_0)
    y_1 = concatenate(true_y_1)
    mu_0 = concatenate(true_mu_0)
    mu_1 = concatenate(true_mu_1)
    treatment = concatenate(treatments)

    po_hat = torch.where(treatment == 0, y_0_hat, y_1_hat)
    po_true = torch.where(treatment == 0, y_0, y_1)
    w1_by_draw = [
        0.5
        * (
            wasserstein_dist(y_0_draw[:, draw_index], mu_0)
            + wasserstein_dist(y_1_draw[:, draw_index], mu_1)
        )
        for draw_index in range(y_0_draw.shape[1])
    ]
    metrics = {
        "counterfactual_RMSE": torch.sqrt(
            F.mse_loss(counterfactual_hat, counterfactual_true)
        ).item(),
        "PEHE": torch.sqrt(F.mse_loss(y_1_hat - y_0_hat, mu_1 - mu_0)).item(),
        "PO_RMSE": torch.sqrt(F.mse_loss(po_hat, po_true)).item(),
        "W1_distance": float(np.mean(w1_by_draw)),
        "W1_distance_MC_SE": float(
            np.std(w1_by_draw, ddof=1) / np.sqrt(len(w1_by_draw))
            if len(w1_by_draw) > 1
            else 0.0
        ),
        "W1_mean_distance": float(
            0.5 * (wasserstein_dist(y_0_hat, mu_0) + wasserstein_dist(y_1_hat, mu_1))
        ),
    }
    if kl_rows:
        metrics["KL_y0"] = float(np.mean([row["KL_y0"] for row in kl_rows]))
        metrics["KL_y1"] = float(np.mean([row["KL_y1"] for row in kl_rows]))
    return metrics


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_DIR / candidate


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PO-Flow with cross-fitted IPW")
    parser.add_argument("--PO_Flow_config", default="ACIC2018.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--test-epochs", type=int, default=None)
    parser.add_argument("--eval-samples", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="optional model state used to initialize IPW fine-tuning",
    )
    parser.add_argument(
        "--swa-start-fraction",
        type=float,
        default=None,
        help="average epoch-end parameters from this fraction of training onward",
    )
    parser.add_argument(
        "--propensity-only",
        action="store_true",
        help="estimate weights and save diagnostics without training PO-Flow",
    )
    parser.add_argument("--propensity-model", choices=("logistic", "mlp", "hist_gbm"))
    parser.add_argument("--crossfit-folds", type=int)
    parser.add_argument("--calibration", choices=("none", "sigmoid", "isotonic"))
    parser.add_argument("--propensity-clip", type=float)
    parser.add_argument("--weight-clip-quantile", type=float)
    parser.add_argument("--weight-clip-max", type=float)
    parser.add_argument("--logistic-c", type=float)
    parser.add_argument("--ipw-strength", type=float)
    stabilization = parser.add_mutually_exclusive_group()
    stabilization.add_argument("--stabilized", action="store_true", dest="stabilized")
    stabilization.add_argument("--unstabilized", action="store_false", dest="stabilized")
    parser.set_defaults(stabilized=None)
    normalization = parser.add_mutually_exclusive_group()
    normalization.add_argument("--normalize-weights", action="store_true", dest="normalize")
    normalization.add_argument("--no-normalize-weights", action="store_false", dest="normalize")
    parser.set_defaults(normalize=None)
    arm_normalization = parser.add_mutually_exclusive_group()
    arm_normalization.add_argument(
        "--arm-normalize", action="store_true", dest="arm_normalize"
    )
    arm_normalization.add_argument(
        "--global-normalize", action="store_false", dest="arm_normalize"
    )
    parser.set_defaults(arm_normalize=None)
    return parser


def _ipw_config(yaml_config, cli_args) -> IPWConfig:
    values = dict(yaml_config.get("ipw", {}))
    overrides = {
        "model": cli_args.propensity_model,
        "crossfit_folds": cli_args.crossfit_folds,
        "calibration": cli_args.calibration,
        "propensity_clip": cli_args.propensity_clip,
        "weight_clip_quantile": cli_args.weight_clip_quantile,
        "weight_clip_max": cli_args.weight_clip_max,
        "logistic_c": cli_args.logistic_c,
        "ipw_strength": cli_args.ipw_strength,
        "stabilized": cli_args.stabilized,
        "normalize": cli_args.normalize,
        "arm_normalize": cli_args.arm_normalize,
        "seed": cli_args.seed,
    }
    values.update({key: value for key, value in overrides.items() if value is not None})
    return IPWConfig.from_mapping(values)


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    return torch.device(requested)


def main() -> None:
    cli_args = _build_parser().parse_args()
    config_path = _resolve_path(cli_args.PO_Flow_config)
    with config_path.open("r") as file:
        yaml_config = yaml.safe_load(file)

    ipw_config = _ipw_config(yaml_config, cli_args)
    seed = ipw_config.seed
    seed_everything(seed)
    device = _device(cli_args.device)

    data_name = yaml_config["data"]["name"]
    data_path = PROJECT_DIR / "datasets" / f"{data_name}_sample.csv"
    data_frame = pd.read_csv(data_path)
    x_dim = data_frame.shape[1] - 5
    y_dim = yaml_config["data"]["Ydim"]
    builder = CounterfactualDatasetBuilder(data_frame)
    training_dataset, test_dataset = builder.build()

    training_x = training_dataset.x_tensor.cpu().numpy()
    training_treatment = training_dataset.base_dataset.a.cpu().numpy()
    ipw_result = build_ipw(training_x, training_treatment, ipw_config)
    weighted_training_dataset = WeightedDataset(training_dataset, ipw_result.weight)
    print(format_diagnostics(ipw_result.diagnostics))

    run_name = cli_args.run_name or f"{data_name}_IPW"
    if Path(run_name).name != run_name:
        raise ValueError("run-name must be a filename-safe name, not a path")
    results_dir = PROJECT_DIR / "results"
    parameter_dir = PROJECT_DIR / "NN_params"
    results_dir.mkdir(exist_ok=True)
    parameter_dir.mkdir(exist_ok=True)

    diagnostics = dict(ipw_result.diagnostics)
    diagnostics.update(
        {
            "dataset": data_name,
            "data_path": str(data_path),
            "config_path": str(config_path),
            "device": str(device),
        }
    )
    with (results_dir / f"{run_name}_diagnostics.json").open("w") as file:
        json.dump(diagnostics, file, indent=2)
    pd.DataFrame(
        {
            "training_row": np.arange(len(training_dataset)),
            "treatment": training_treatment,
            "propensity_oof": ipw_result.propensity,
            "propensity_clipped": ipw_result.clipped_propensity,
            "raw_weight": ipw_result.raw_weight,
            "full_ipw_weight": ipw_result.ipw_weight,
            "final_weight": ipw_result.weight,
        }
    ).to_csv(results_dir / f"{run_name}_weights.csv", index=False)
    pd.DataFrame(
        {
            "covariate": data_frame.columns[5:],
            "abs_SMD_before": ipw_result.smd_before,
            "abs_SMD_full_ipw": ipw_result.smd_ipw,
            "abs_SMD_after": ipw_result.smd_after,
        }
    ).to_csv(results_dir / f"{run_name}_balance.csv", index=False)
    if cli_args.propensity_only:
        print(f"Saved propensity diagnostics to {results_dir}; skipping PO-Flow training.")
        return

    batch_size = (
        cli_args.batch_size
        if cli_args.batch_size is not None
        else yaml_config["training"]["batch_size"]
    )
    epochs = cli_args.epochs if cli_args.epochs is not None else yaml_config["training"]["epochs"]
    test_epochs = (
        cli_args.test_epochs
        if cli_args.test_epochs is not None
        else yaml_config["test"]["epochs"]
    )
    eval_samples = (
        cli_args.eval_samples
        if cli_args.eval_samples is not None
        else yaml_config["test"].get("PO_samples", 1)
    )
    ode_steps = yaml_config["test"].get("ODE_steps", 100)
    if min(
        batch_size,
        epochs,
        test_epochs,
        eval_samples,
        ode_steps,
        cli_args.log_every,
    ) <= 0:
        raise ValueError("training and evaluation counts must all be positive")

    training_loader = DataLoader(
        weighted_training_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    hidden_dim = (
        cli_args.hidden_dim
        if cli_args.hidden_dim is not None
        else yaml_config["training"].get("hidden_dim", 128)
    )
    if hidden_dim <= 0:
        raise ValueError("hidden dimension must be positive")
    model = FlowMatchingNet(y_dim=y_dim, x_dim=x_dim, hidden_dim=hidden_dim).to(device)
    if cli_args.init_checkpoint is not None:
        initialization_path = _resolve_path(cli_args.init_checkpoint)
        try:
            initialization_state = torch.load(
                initialization_path, map_location=device, weights_only=True
            )
        except TypeError:
            initialization_state = torch.load(initialization_path, map_location=device)
        model.load_state_dict(initialization_state)
        print(f"Initialized model from {initialization_path}")
    learning_rate = (
        cli_args.learning_rate
        if cli_args.learning_rate is not None
        else yaml_config["training"].get("learning_rate", 1e-3)
    )
    if learning_rate <= 0.0:
        raise ValueError("learning rate must be positive")
    swa_start_fraction = (
        cli_args.swa_start_fraction
        if cli_args.swa_start_fraction is not None
        else yaml_config["training"].get("swa_start_fraction")
    )
    if swa_start_fraction is not None and not (0.0 <= swa_start_fraction < 1.0):
        raise ValueError("swa-start-fraction must lie in [0, 1)")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    averaged_model = AveragedModel(model) if swa_start_fraction is not None else None
    swa_start_epoch = (
        max(1, int(np.ceil(epochs * swa_start_fraction)))
        if swa_start_fraction is not None
        else None
    )

    print(
        f"Training {run_name} on {device}: n_train={len(training_dataset)}, "
        f"n_test={len(test_dataset)}, epochs={epochs}, batch_size={batch_size}"
    )
    history = []
    metrics_path = results_dir / f"{run_name}_metrics.csv"
    for epoch in range(1, epochs + 1):
        training_metrics = train_flow_matching_ipw(model, training_loader, optimizer, device)
        if averaged_model is not None and epoch >= swa_start_epoch:
            averaged_model.update_parameters(model)
        if epoch == 1 or epoch == epochs or epoch % cli_args.log_every == 0:
            print(
                f"Epoch {epoch:04d}: weighted loss="
                f"{training_metrics['weighted_flow_matching_loss']:.6f}, "
                f"unweighted loss={training_metrics['unweighted_flow_matching_loss']:.6f}"
            )

        if epoch % test_epochs != 0 and epoch != epochs:
            continue
        test_config = yaml_config["test"]
        compute_kl = bool(test_config.get("if_test_KL_metric", False)) and (
            epoch == test_config.get("KL_metric_epoch")
        )
        # Evaluation uses a fixed Monte Carlo stream and cannot perturb the
        # subsequent training RNG.  This makes weight ablations paired and
        # keeps the learned model independent of the test frequency.
        cuda_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed + 100_000)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed + 100_000)
            metrics = evaluate(
                model,
                test_loader,
                device,
                potential_outcome_samples=eval_samples,
                ode_steps=ode_steps,
                kl_samples=test_config.get("KL_samples", 100) if compute_kl else None,
            )
        row = {
            "epoch": epoch,
            "iterations": epoch * len(training_dataset) // batch_size,
            **training_metrics,
            **metrics,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(metrics_path, index=False)
        print("  " + ", ".join(f"{name}={value:.6f}" for name, value in metrics.items()))

    torch.save(model.state_dict(), parameter_dir / f"{run_name}_final.pth")
    if averaged_model is not None:
        torch.save(
            averaged_model.module.state_dict(),
            parameter_dir / f"{run_name}_swa_final.pth",
        )
        print(
            f"Saved SWA checkpoint averaged over epochs {swa_start_epoch}--{epochs}"
        )
    print(f"Saved metrics to {metrics_path}")
    print(
        f"Final counterfactual RMSE: {history[-1]['counterfactual_RMSE']:.6f}; "
        f"final PO RMSE: {history[-1]['PO_RMSE']:.6f}"
    )


if __name__ == "__main__":
    main()
