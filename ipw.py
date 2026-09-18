"""Inverse-probability weighting utilities for PO-Flow.

The propensity score is estimated out of fold.  This keeps each training
observation's flow-matching loss from being weighted by a propensity model
that was fitted on that observation.  The resulting weights are attached to
the dataset before minibatch shuffling, so they always remain aligned with the
corresponding factual sample.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class IPWConfig:
    """Configuration for cross-fitted propensity estimation and weighting."""

    model: str = "logistic"
    crossfit_folds: int = 5
    calibration: str = "none"
    calibration_folds: int = 3
    propensity_clip: float = 0.01
    stabilized: bool = False
    normalize: bool = True
    # Global normalization only rescales the loss.  Arm normalization is the
    # finite-sample Hajek variant: each treatment arm receives total mass n/2.
    arm_normalize: bool = False
    # Shrink noisy IPW weights toward the unweighted objective.  One gives
    # full IPW and zero is an exactly comparable unweighted ablation.
    ipw_strength: float = 1.0
    # A 1% upper-tail truncation is a variance-control default.  Set this to
    # 1.0 (or null in YAML) for the exact, untruncated IPW objective.
    weight_clip_quantile: Optional[float] = 0.99
    weight_clip_max: Optional[float] = None
    logistic_c: float = 1.0
    mlp_hidden_dims: Tuple[int, ...] = (128, 64)
    mlp_alpha: float = 1e-4
    mlp_max_iter: int = 500
    seed: int = 42

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]]) -> "IPWConfig":
        values = dict(values or {})
        if "mlp_hidden_dims" in values:
            values["mlp_hidden_dims"] = tuple(values["mlp_hidden_dims"])
        known = cls.__dataclass_fields__
        unexpected = sorted(set(values) - set(known))
        if unexpected:
            raise ValueError(f"Unknown IPW configuration fields: {unexpected}")
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        if self.model not in {"logistic", "mlp", "hist_gbm"}:
            raise ValueError("model must be one of: logistic, mlp, hist_gbm")
        if self.crossfit_folds < 2:
            raise ValueError("crossfit_folds must be at least 2")
        if self.calibration not in {"none", "sigmoid", "isotonic"}:
            raise ValueError("calibration must be one of: none, sigmoid, isotonic")
        if self.calibration != "none" and self.calibration_folds < 2:
            raise ValueError("calibration_folds must be at least 2")
        if not 0.0 < self.propensity_clip < 0.5:
            raise ValueError("propensity_clip must lie strictly between 0 and 0.5")
        if self.weight_clip_quantile is not None:
            if not 0.0 < self.weight_clip_quantile <= 1.0:
                raise ValueError("weight_clip_quantile must lie in (0, 1]")
        if self.weight_clip_max is not None and self.weight_clip_max <= 0.0:
            raise ValueError("weight_clip_max must be positive")
        if self.logistic_c <= 0.0:
            raise ValueError("logistic_c must be positive")
        if self.arm_normalize and not self.normalize:
            raise ValueError("arm_normalize requires normalize=True")
        if not 0.0 <= self.ipw_strength <= 1.0:
            raise ValueError("ipw_strength must lie in [0, 1]")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IPWResult:
    propensity: np.ndarray
    clipped_propensity: np.ndarray
    raw_weight: np.ndarray
    ipw_weight: np.ndarray
    weight: np.ndarray
    smd_ipw: np.ndarray
    smd_before: np.ndarray
    smd_after: np.ndarray
    diagnostics: Dict[str, Any]


class WeightedDataset(Dataset):
    """Append a fixed scalar weight to every item of an existing dataset."""

    def __init__(self, dataset: Dataset, weights: Sequence[float]):
        if len(dataset) != len(weights):
            raise ValueError("dataset and weights must have the same length")
        self.dataset = dataset
        self.weights = torch.as_tensor(weights, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        if not isinstance(item, tuple):
            raise TypeError("WeightedDataset expects the wrapped dataset to return a tuple")
        return (*item, self.weights[index])


def _make_propensity_model(config: IPWConfig, seed: int):
    """Construct a probabilistic binary classifier without fitting it."""

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if config.model == "logistic":
        classifier = LogisticRegression(
            C=config.logistic_c,
            max_iter=2000,
            random_state=seed,
            solver="lbfgs",
        )
        model = make_pipeline(StandardScaler(), classifier)
    elif config.model == "mlp":
        classifier = MLPClassifier(
            hidden_layer_sizes=config.mlp_hidden_dims,
            alpha=config.mlp_alpha,
            batch_size="auto",
            early_stopping=True,
            max_iter=config.mlp_max_iter,
            random_state=seed,
        )
        model = make_pipeline(StandardScaler(), classifier)
    else:
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=100,
            l2_regularization=1e-4,
            random_state=seed,
        )

    if config.calibration == "none":
        return model

    from sklearn.calibration import CalibratedClassifierCV

    # sklearn renamed base_estimator to estimator.  Supporting both keeps this
    # experiment entry point usable in the older environment of the project.
    try:
        return CalibratedClassifierCV(
            estimator=model,
            method=config.calibration,
            cv=config.calibration_folds,
        )
    except TypeError:
        return CalibratedClassifierCV(
            base_estimator=model,
            method=config.calibration,
            cv=config.calibration_folds,
        )


def cross_fitted_propensity(
    x: np.ndarray,
    treatment: np.ndarray,
    config: IPWConfig,
) -> np.ndarray:
    """Return out-of-fold estimates of P(A=1 | X) for every training row."""

    from sklearn.model_selection import StratifiedKFold

    x = np.asarray(x, dtype=np.float64)
    treatment = np.asarray(treatment, dtype=np.int64).reshape(-1)
    if x.ndim != 2 or len(x) != len(treatment):
        raise ValueError("x must be a 2-D array aligned with treatment")
    unique = np.unique(treatment)
    if not np.array_equal(unique, np.array([0, 1])):
        raise ValueError(f"binary treatments 0 and 1 are required; observed {unique.tolist()}")
    if not np.all(np.isfinite(x)):
        raise ValueError("covariates contain NaN or infinite values")

    smallest_arm = int(np.bincount(treatment, minlength=2).min())
    folds = min(config.crossfit_folds, smallest_arm)
    if folds < 2:
        raise ValueError("each treatment arm needs at least two observations for cross-fitting")

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.seed)
    propensity = np.full(len(treatment), np.nan, dtype=np.float64)
    for fold, (fit_index, holdout_index) in enumerate(splitter.split(x, treatment)):
        model = _make_propensity_model(config, seed=config.seed + fold)
        model.fit(x[fit_index], treatment[fit_index])
        propensity[holdout_index] = model.predict_proba(x[holdout_index])[:, 1]

    if not np.all(np.isfinite(propensity)):
        raise RuntimeError("cross-fitting failed to produce a finite propensity for every row")
    return propensity


def _effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    denominator = np.square(weights).sum()
    return float(np.square(weights.sum()) / denominator) if denominator > 0.0 else 0.0


def _weighted_mean_and_variance(
    x: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=np.float64)
    weight_sum = weights.sum()
    if weight_sum <= 0.0:
        raise ValueError("weights must have positive mass")
    normalized = weights / weight_sum
    mean = np.sum(x * normalized[:, None], axis=0)
    variance = np.sum(np.square(x - mean) * normalized[:, None], axis=0)
    return mean, variance


def standardized_mean_differences(
    x: np.ndarray,
    treatment: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute absolute treatment-control SMDs for all supplied covariates."""

    x = np.asarray(x, dtype=np.float64)
    treatment = np.asarray(treatment, dtype=np.int64).reshape(-1)
    weights = np.ones(len(treatment), dtype=np.float64) if weights is None else np.asarray(weights)
    treated = treatment == 1
    control = ~treated
    mean_1, var_1 = _weighted_mean_and_variance(x[treated], weights[treated])
    mean_0, var_0 = _weighted_mean_and_variance(x[control], weights[control])
    scale = np.sqrt((var_1 + var_0) / 2.0)
    difference = np.abs(mean_1 - mean_0)
    return np.divide(difference, scale, out=np.zeros_like(difference), where=scale > 1e-12)


def binned_mutual_information(
    propensity: np.ndarray,
    treatment: np.ndarray,
    weights: Optional[np.ndarray] = None,
    bins: int = 10,
) -> float:
    """Estimate I(A; binned propensity(X)) in nats as a balance diagnostic."""

    propensity = np.asarray(propensity, dtype=np.float64)
    treatment = np.asarray(treatment, dtype=np.int64).reshape(-1)
    weights = np.ones(len(treatment), dtype=np.float64) if weights is None else np.asarray(weights)
    weights = weights / weights.sum()

    edges = np.unique(np.quantile(propensity, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) <= 2:
        return 0.0
    bin_index = np.clip(np.digitize(propensity, edges[1:-1]), 0, len(edges) - 2)
    joint = np.zeros((2, len(edges) - 1), dtype=np.float64)
    for arm in (0, 1):
        for index in range(joint.shape[1]):
            joint[arm, index] = weights[(treatment == arm) & (bin_index == index)].sum()
    marginal_a = joint.sum(axis=1, keepdims=True)
    marginal_bin = joint.sum(axis=0, keepdims=True)
    independent = marginal_a @ marginal_bin
    mask = (joint > 0.0) & (independent > 0.0)
    return float(np.sum(joint[mask] * np.log(joint[mask] / independent[mask])))


def _propensity_metrics(propensity: np.ndarray, treatment: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score

    epsilon = np.finfo(np.float64).eps
    probability = np.clip(propensity, epsilon, 1.0 - epsilon)
    log_loss = -np.mean(treatment * np.log(probability) + (1 - treatment) * np.log(1 - probability))
    return {
        "roc_auc": float(roc_auc_score(treatment, propensity)),
        "brier_score": float(np.mean(np.square(propensity - treatment))),
        "log_loss": float(log_loss),
    }


def build_ipw(
    x: np.ndarray,
    treatment: np.ndarray,
    config: IPWConfig,
) -> IPWResult:
    """Estimate propensities, form robust IPW weights, and compute diagnostics."""

    x = np.asarray(x, dtype=np.float64)
    treatment = np.asarray(treatment, dtype=np.int64).reshape(-1)
    propensity = cross_fitted_propensity(x, treatment, config)
    clipped_propensity = np.clip(
        propensity,
        config.propensity_clip,
        1.0 - config.propensity_clip,
    )
    observed_probability = np.where(treatment == 1, clipped_propensity, 1.0 - clipped_propensity)
    numerator = np.ones_like(observed_probability)
    if config.stabilized:
        treated_probability = treatment.mean()
        numerator = np.where(treatment == 1, treated_probability, 1.0 - treated_probability)
    raw_weight = numerator / observed_probability
    weight = raw_weight.copy()

    clipping_threshold = None
    if config.weight_clip_quantile is not None and config.weight_clip_quantile < 1.0:
        clipping_threshold = float(np.quantile(weight, config.weight_clip_quantile))
    if config.weight_clip_max is not None:
        clipping_threshold = (
            float(config.weight_clip_max)
            if clipping_threshold is None
            else min(clipping_threshold, float(config.weight_clip_max))
        )
    if clipping_threshold is not None:
        weight = np.minimum(weight, clipping_threshold)

    arm_0 = treatment == 0
    arm_1 = treatment == 1
    normalization_factor: Any = float(weight.mean())
    if config.normalize:
        if config.arm_normalize:
            target_mass = len(weight) / 2.0
            arm_factors = {
                "control": float(weight[arm_0].sum() / target_mass),
                "treated": float(weight[arm_1].sum() / target_mass),
            }
            weight[arm_0] = weight[arm_0] / arm_factors["control"]
            weight[arm_1] = weight[arm_1] / arm_factors["treated"]
            normalization_factor = arm_factors
        else:
            weight = weight / normalization_factor

    # Convex shrinkage controls the finite-sample variance of IPW while
    # retaining a nested, exactly unweighted ablation at strength zero.
    ipw_weight = weight.copy()
    weight = 1.0 + config.ipw_strength * (ipw_weight - 1.0)

    smd_before = standardized_mean_differences(x, treatment)
    smd_ipw = standardized_mean_differences(x, treatment, ipw_weight)
    smd_after = standardized_mean_differences(x, treatment, weight)
    diagnostics: Dict[str, Any] = {
        "config": config.to_dict(),
        "n": int(len(treatment)),
        "n_control": int(arm_0.sum()),
        "n_treated": int(arm_1.sum()),
        "treatment_prevalence": float(treatment.mean()),
        "propensity_metrics_oof": _propensity_metrics(propensity, treatment),
        "propensity_min": float(propensity.min()),
        "propensity_max": float(propensity.max()),
        "propensity_quantiles": {
            str(q): float(np.quantile(propensity, q))
            for q in (0.01, 0.05, 0.5, 0.95, 0.99)
        },
        "fraction_propensity_clipped": float(np.mean(propensity != clipped_propensity)),
        "raw_weight_min": float(raw_weight.min()),
        "raw_weight_max": float(raw_weight.max()),
        "final_weight_min": float(weight.min()),
        "final_weight_max": float(weight.max()),
        "full_ipw_weight_min": float(ipw_weight.min()),
        "full_ipw_weight_max": float(ipw_weight.max()),
        "weight_clipping_threshold": clipping_threshold,
        "fraction_weight_clipped": (
            float(np.mean(raw_weight > clipping_threshold))
            if clipping_threshold is not None
            else 0.0
        ),
        "weight_normalization_factor": normalization_factor,
        "effective_sample_size": _effective_sample_size(weight),
        "effective_sample_size_control": _effective_sample_size(weight[arm_0]),
        "effective_sample_size_treated": _effective_sample_size(weight[arm_1]),
        "full_ipw_effective_sample_size": _effective_sample_size(ipw_weight),
        "weighted_treatment_mass": float(weight[arm_1].sum() / weight.sum()),
        "full_ipw_treatment_mass": float(ipw_weight[arm_1].sum() / ipw_weight.sum()),
        "max_abs_smd_before": float(smd_before.max()),
        "max_abs_smd_full_ipw": float(smd_ipw.max()),
        "max_abs_smd_after": float(smd_after.max()),
        "mean_abs_smd_before": float(smd_before.mean()),
        "mean_abs_smd_full_ipw": float(smd_ipw.mean()),
        "mean_abs_smd_after": float(smd_after.mean()),
        "fraction_abs_smd_above_0.1_before": float(np.mean(smd_before > 0.1)),
        "fraction_abs_smd_above_0.1_after": float(np.mean(smd_after > 0.1)),
        "binned_mutual_information_before_nats": binned_mutual_information(
            propensity, treatment
        ),
        "binned_mutual_information_after_nats": binned_mutual_information(
            propensity, treatment, weight
        ),
    }
    return IPWResult(
        propensity=propensity,
        clipped_propensity=clipped_propensity,
        raw_weight=raw_weight,
        ipw_weight=ipw_weight,
        weight=weight,
        smd_ipw=smd_ipw,
        smd_before=smd_before,
        smd_after=smd_after,
        diagnostics=diagnostics,
    )


def format_diagnostics(diagnostics: Mapping[str, Any]) -> str:
    """Create a compact, human-readable propensity/weight summary."""

    propensity_metrics = diagnostics["propensity_metrics_oof"]
    return (
        "IPW diagnostics (out-of-fold):\n"
        f"  propensity AUC={propensity_metrics['roc_auc']:.4f}, "
        f"Brier={propensity_metrics['brier_score']:.4f}, "
        f"range=[{diagnostics['propensity_min']:.4f}, {diagnostics['propensity_max']:.4f}]\n"
        f"  training weights range=[{diagnostics['final_weight_min']:.4f}, "
        f"{diagnostics['final_weight_max']:.4f}], "
        f"ESS={diagnostics['effective_sample_size']:.1f}/{diagnostics['n']}\n"
        f"  max |SMD| {diagnostics['max_abs_smd_before']:.4f} -> "
        f"{diagnostics['max_abs_smd_after']:.4f}; "
        f"I(A; binned e(X)) {diagnostics['binned_mutual_information_before_nats']:.5f} -> "
        f"{diagnostics['binned_mutual_information_after_nats']:.5f} nats"
    )
