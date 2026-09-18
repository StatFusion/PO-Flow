import torch

from multidim_baselines import CEVAEVector, CFRVector, GANITEVector, factual, linear_mmd


def test_vector_baseline_shapes():
    x = torch.randn(7, 4)
    a = torch.tensor([0, 1, 0, 1, 0, 1, 0])
    y = torch.randn(7, 3)
    cfr = CFRVector(4, 3, 8)
    y0, y1, representation = cfr(x)
    assert y0.shape == y1.shape == (7, 3)
    assert representation.shape == (7, 8)
    ganite = GANITEVector(4, 3, 8)
    g0, g1 = ganite.generate(x, a.float(), y)
    i0, i1 = ganite.infer(x)
    assert g0.shape == g1.shape == i0.shape == i1.shape == (7, 3)
    cevae = CEVAEVector(4, 3, 8, 5)
    mean, logvar = cevae.encode(x, a.float(), y)
    assert mean.shape == logvar.shape == (7, 5)
    assert cevae.decode_y(mean, a.float()).shape == (7, 3)


def test_factual_selection_and_linear_mmd_identity():
    y0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    a = torch.tensor([0, 1])
    assert torch.equal(factual(y0, y1, a), torch.tensor([[1.0, 2.0], [7.0, 8.0]]))
    representation = torch.randn(5, 6)
    assert torch.allclose(linear_mmd(representation, representation), torch.tensor(0.0))
