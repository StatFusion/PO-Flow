import numpy as np
import torch

from main import FlowMatchingNet, TimeConditionedFlowMatchingNet, flow_to_noise, flow_to_sample
from main_multidim import (
    load_archive,
    sliced_wasserstein,
    standardize_archive,
    standardize_covariates,
)


def test_vector_ode_round_trip(tmp_path):
    torch.manual_seed(7)
    model = FlowMatchingNet(y_dim=3, x_dim=2, hidden_dim=16)
    y = torch.randn(4, 3)
    x = torch.randn(4, 2)
    a = torch.tensor([0, 1, 0, 1])
    z = flow_to_noise(model, y, a, x, torch.device("cpu"), train=False, num_steps=20)
    recovered = flow_to_sample(model, z, a, x, torch.device("cpu"), train=False, num_steps=20)
    assert z.shape == y.shape
    assert recovered.shape == y.shape
    assert torch.allclose(recovered, y, atol=2e-5, rtol=2e-5)


def test_time_conditioned_network_uses_time_and_round_trips():
    torch.manual_seed(17)
    model = TimeConditionedFlowMatchingNet(
        y_dim=3, x_dim=2, core_width=8, depth=1, time_dim=4, expansion=1
    )
    y = torch.randn(4, 3)
    x = torch.randn(4, 2)
    a = torch.tensor([0, 1, 0, 1])
    velocity0 = model(y, torch.zeros(4, 1), a, x)[0]
    velocity1 = model(y, torch.ones(4, 1), a, x)[0]
    assert not torch.allclose(velocity0, velocity1)
    z = flow_to_noise(model, y, a, x, torch.device("cpu"), train=False, num_steps=8)
    recovered = flow_to_sample(model, z, a, x, torch.device("cpu"), train=False, num_steps=8)
    assert torch.allclose(recovered, y, atol=2e-3, rtol=2e-3)


def test_time_conditioned_network_expands_the_backbone():
    legacy = FlowMatchingNet(y_dim=2, x_dim=25, hidden_dim=128)
    expanded = TimeConditionedFlowMatchingNet(
        y_dim=2, x_dim=25, core_width=192, depth=3, time_dim=32, expansion=2
    )
    legacy_parameters = sum(parameter.numel() for parameter in legacy.parameters())
    expanded_parameters = sum(parameter.numel() for parameter in expanded.parameters())
    assert expanded_parameters > 20 * legacy_parameters


def test_shared_head_is_still_treatment_conditional():
    torch.manual_seed(23)
    model = TimeConditionedFlowMatchingNet(
        y_dim=2, x_dim=3, core_width=8, depth=1, time_dim=4, expansion=1, head_mode="shared"
    )
    y = torch.randn(5, 2)
    x = torch.randn(5, 3)
    time = torch.full((5, 1), 0.4)
    control = model(y, time, torch.zeros(5), x)
    treated = model(y, time, torch.ones(5), x)
    assert torch.allclose(control[0], control[1])
    assert torch.allclose(treated[0], treated[1])
    assert not torch.allclose(control[0], treated[0])


def test_auxiliary_mean_is_parameter_free_and_treatment_conditional():
    torch.manual_seed(29)
    model = TimeConditionedFlowMatchingNet(
        y_dim=2, x_dim=3, core_width=8, depth=1, time_dim=4, expansion=1
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    x = torch.randn(5, 3)
    mean0 = model.auxiliary_mean(x, torch.zeros(5))
    mean1 = model.auxiliary_mean(x, torch.ones(5))
    assert mean0.shape == mean1.shape == (5, 2)
    assert not torch.allclose(mean0, mean1)
    assert sum(parameter.numel() for parameter in model.parameters()) == parameter_count


def test_archive_standardization_uses_train_factual(tmp_path):
    path = tmp_path / "toy.npz"
    np.savez(
        path,
        A=np.array([0, 1, 0, 1]),
        X=np.ones((4, 2), dtype=np.float32),
        Y0=np.array([[0], [1], [2], [3]], dtype=np.float32),
        Y1=np.array([[4], [5], [6], [7]], dtype=np.float32),
        MU0=np.array([[0], [1], [2], [3]], dtype=np.float32),
        MU1=np.array([[4], [5], [6], [7]], dtype=np.float32),
        train_idx=np.array([0, 1, 2]),
        test_idx=np.array([3]),
    )
    arrays = load_archive(path)
    transformed, center, scale = standardize_archive(arrays, image_scale=False)
    factual = np.array([0.0, 5.0, 2.0])
    assert np.allclose(center, factual.mean())
    assert np.allclose(scale, factual.std())
    assert transformed["Y0"].shape == (4, 1)


def test_covariate_standardization_uses_training_rows_only():
    arrays = {
        "X": np.array([[1.0, 10.0], [3.0, 14.0], [100.0, -50.0]], dtype=np.float32),
        "train_idx": np.array([0, 1]),
    }
    transformed, center, scale = standardize_covariates(arrays)
    np.testing.assert_allclose(center, np.array([2.0, 12.0]))
    np.testing.assert_allclose(scale, np.array([np.sqrt(2.0), np.sqrt(8.0)]))
    np.testing.assert_allclose(transformed["X"][:2].mean(0), np.zeros(2), atol=1e-7)
    assert transformed["X"][2, 0] > 10.0


def test_sliced_wasserstein_is_zero_for_identical_samples():
    samples = torch.tensor([[0.0, 1.0], [2.0, -1.0], [3.0, 4.0]])
    assert sliced_wasserstein(samples, samples, projections=16, seed=9) == 0.0


def test_sliced_wasserstein_detects_translation():
    samples = torch.tensor([[0.0, 1.0], [2.0, -1.0], [3.0, 4.0]])
    shifted = samples + torch.tensor([1.0, -2.0])
    assert sliced_wasserstein(samples, shifted, projections=128, seed=9) > 0.5
