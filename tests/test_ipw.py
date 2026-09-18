import numpy as np
import torch
from torch.utils.data import TensorDataset

from ipw import IPWConfig, WeightedDataset, build_ipw


def _toy_observational_data(seed=7, n=400):
    generator = np.random.default_rng(seed)
    x = generator.normal(size=(n, 4))
    propensity = 1.0 / (1.0 + np.exp(-(0.8 * x[:, 0] - 0.5 * x[:, 1])))
    treatment = generator.binomial(1, propensity)
    return x, treatment


def test_oof_ipw_formula_normalization_and_determinism():
    x, treatment = _toy_observational_data()
    config = IPWConfig(seed=11, weight_clip_quantile=None)
    first = build_ipw(x, treatment, config)
    second = build_ipw(x, treatment, config)

    expected_raw = 1.0 / np.where(
        treatment == 1,
        first.clipped_propensity,
        1.0 - first.clipped_propensity,
    )
    np.testing.assert_allclose(first.raw_weight, expected_raw)
    np.testing.assert_allclose(first.propensity, second.propensity)
    np.testing.assert_allclose(first.weight, second.weight)
    np.testing.assert_allclose(first.weight.mean(), 1.0)
    assert np.all(np.isfinite(first.weight))
    assert np.all(first.weight > 0.0)


def test_shrinkage_zero_is_exact_unweighted_ablation():
    x, treatment = _toy_observational_data()
    result = build_ipw(x, treatment, IPWConfig(seed=3, ipw_strength=0.0))
    np.testing.assert_array_equal(result.weight, np.ones(len(treatment)))


def test_arm_hajek_normalization_has_equal_arm_mass():
    x, treatment = _toy_observational_data()
    result = build_ipw(x, treatment, IPWConfig(seed=5, arm_normalize=True))
    target_mass = len(treatment) / 2.0
    np.testing.assert_allclose(result.weight[treatment == 0].sum(), target_mass)
    np.testing.assert_allclose(result.weight[treatment == 1].sum(), target_mass)


def test_weighted_dataset_keeps_row_weight_alignment():
    base = TensorDataset(torch.arange(6), torch.arange(6) + 10)
    weights = np.linspace(0.5, 1.5, num=6)
    weighted = WeightedDataset(base, weights)
    for index in range(len(weighted)):
        first, second, weight = weighted[index]
        assert first.item() == index
        assert second.item() == index + 10
        assert np.isclose(weight.item(), weights[index])
