import math

import torch

import main


class _LinearVelocity(torch.nn.Module):
    y_dim = 1

    def __init__(self, coefficient):
        super().__init__()
        self.coefficient = coefficient

    def forward(self, y, time, treatment, covariates):
        velocity = self.coefficient * y
        return velocity, velocity


def test_reverse_cnf_log_density_sign(monkeypatch):
    coefficient = 0.3

    def exact_divergence(velocity, y, noises, t, net):
        divergence = torch.full(
            (y.shape[0],), coefficient, dtype=y.dtype, device=y.device
        )
        return divergence, torch.zeros_like(divergence)

    monkeypatch.setattr(main, "divergence_approx", exact_divergence)
    latent = torch.tensor([[0.2], [-0.7], [1.1]], dtype=torch.float32)
    sample, log_density = main.flow_to_sample(
        _LinearVelocity(coefficient),
        latent,
        torch.zeros(3),
        torch.zeros(3, 1),
        torch.device("cpu"),
        train=False,
        nll=True,
        num_steps=101,
    )

    expected_sample = latent * math.exp(-coefficient)
    latent_log_density = -0.5 * (
        latent.square() + math.log(2.0 * math.pi)
    ).view(-1)
    expected_log_density = latent_log_density + coefficient
    torch.testing.assert_close(sample, expected_sample, atol=2e-6, rtol=0.0)
    torch.testing.assert_close(log_density, expected_log_density, atol=3e-6, rtol=0.0)
