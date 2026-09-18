import torch

from additional_baselines import DragonNetVector, train_dragonnet
from multidim_baselines import CFRVector


def test_additional_baseline_output_shapes():
    x = torch.randn(7, 5)
    tarnet = CFRVector(5, 3, 8)
    y0, y1, representation = tarnet(x)
    assert y0.shape == y1.shape == (7, 3)
    assert representation.shape == (7, 8)

    dragonnet = DragonNetVector(5, 3, 8)
    y0, y1, propensity = dragonnet(x)
    assert y0.shape == y1.shape == (7, 3)
    assert propensity.shape == (7,)
    assert dragonnet.epsilon.shape == (3,)


def test_dragonnet_targeted_training_is_finite():
    torch.manual_seed(4)
    model = DragonNetVector(4, 2, 8)
    x = torch.randn(20, 4)
    a = torch.tensor([0, 1] * 10)
    y = torch.randn(20, 2)
    loader = [(x, a, y)]
    result = train_dragonnet(
        model, loader, torch.device("cpu"), epochs=2, learning_rate=1e-3,
        propensity_strength=1.0, targeted_strength=1.0, propensity_clip=0.01,
        log_every=2,
    )
    assert all(torch.isfinite(torch.tensor(value)) for value in result.values())
