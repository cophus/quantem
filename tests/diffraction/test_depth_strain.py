import numpy as np
import torch
from ase.build import bulk

from quantem.diffraction import Crystal, DepthStrain


def _small_model(eps=0.0):
    data = np.ones((8, 61))
    ds = DepthStrain.from_qx(data, x_sampling=20.0, q_sampling=0.02, energy=300e3)
    ds.set_crystal(Crystal.from_ase(bulk("GaAs", "zincblende", a=5.6533, cubic=True)), (2, 2, 0))
    ds.set_model(thickness=300.0, semiconvergence=0.2, strain_shape=(3, 3), dz=1.0, tilt_step=0.05)
    ds.params["strain"].fill_(eps)
    return ds


def test_split_step_matches_exact_propagator():
    """Uniform strain only shifts the excitation errors: compare with one matrix exponential."""
    eps = 2e-3
    ds = _small_model(eps)
    m, lam = ds.model, ds.wavelength
    phi = ds._propagate(ds.params)[:, :, 0]
    K, gn = m["K"], m["gn"]
    diag = -gn * (gn + 2 * K[:, None]) + 2 * (K[:, None] + gn) * eps / lam + 1j * m["u0_imag"]
    A = m["U"][None] + torch.diag_embed(diag)
    exact = torch.linalg.matrix_exp(1j * np.pi * lam * 300.0 * A)[:, :, m["i0"]]
    assert torch.allclose(phi, exact, atol=2e-4)


def test_strain_recovered_from_model_qx_plot():
    ds = _small_model()
    truth = torch.tensor([[0.0, 3e-3, 0.0], [0.0, -3e-3, 0.0], [0.0, 0.0, 0.0]])
    ds.params["strain"] = truth.to(torch.float64)
    ds.model["data"] = torch.as_tensor(ds.simulate())
    ds.params["strain"] = torch.zeros(3, 3, dtype=torch.float64)
    ds.reconstruct(max_iter=60, verbose=False)
    assert ds.result["loss"] < 1e-3 * ds.result["initial_loss"]
    assert torch.allclose(ds.params["strain"], truth.to(torch.float64), atol=5e-4)
