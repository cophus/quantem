"""CBED simulation against the single-beam Bloch solver and physics checks."""

import numpy as np
import torch
from ase.build import bulk

from quantem.diffraction import bloch
from quantem.diffraction.crystal import Crystal
from quantem.diffraction.rotations import quat_from_zone_axis


def _si(absorptive: bool) -> Crystal:
    si = Crystal.from_ase(
        bulk("Si", "diamond", a=5.431, cubic=True), name="Si", verbose=False
    )
    si.calculate_structure_factors(k_max=3.0)
    if absorptive:
        si.calculate_dynamical_structure_factors(energy_ev=200e3, k_max=3.0)
    return si


def _zone_110() -> torch.Tensor:
    return quat_from_zone_axis(torch.tensor([[1.0, 1.0, 0.0]]) / np.sqrt(2))[0]


def test_zero_tilt_matches_dynamical_pattern():
    si = _si(absorptive=True)
    q = _zone_110()
    t = torch.tensor([800.0])
    tilts = torch.zeros((1, 2), dtype=torch.float64)
    I, g_xy, hkl = bloch._cbed_amplitudes(
        si, q, tilts, t, 200e3, sg_max=0.08, k_max=1.3
    )
    ref = bloch.dynamical_pattern(
        si, q, t, energy_ev=200e3, sg_max=0.08, k_max=1.3
    )
    # same beams (000 first in CBED) and identical intensities
    assert hkl.shape[0] == ref["hkl"].shape[0] + 1
    assert torch.allclose(
        I[0, 0, 1:], ref["intensity"][0], rtol=1e-10, atol=1e-12
    )


def test_unitarity_without_absorption():
    si = _si(absorptive=False)
    q = _zone_110()
    tilts = bloch.tilt_grid(2.0, 200e3, n_rings=2)
    I, _, _ = bloch._cbed_amplitudes(
        si, q, tilts, torch.tensor([500.0, 1500.0]), 200e3, sg_max=0.08, k_max=1.3
    )
    # Hermitian structure matrix: evolution is unitary in the beam space
    total = I.sum(dim=-1)
    assert torch.allclose(total, torch.ones_like(total), atol=1e-8)


def test_lacbed_centrosymmetric_disk():
    si = _si(absorptive=True)
    q = _zone_110()
    res = bloch.calculate_lacbed(
        si, q, 800.0, hkl=(0, 0, 0),
        energy_ev=200e3, semiconv_mrad=6.0, n_pixels=24, sg_max=0.08, k_max=1.3,
    )
    disk = res["disk"]
    # Si is centrosymmetric: the bright field rocking surface at a zone axis
    # is inversion symmetric, I(t) = I(-t). Small residuals come from beam
    # truncation at the s_g cutoff (|s_g| differs slightly for +g and -g),
    # so the tolerance is physical rather than numerical.
    flipped = disk[::-1, ::-1]
    m = np.isfinite(disk) & np.isfinite(flipped)
    assert m.sum() > 100
    assert np.allclose(disk[m], flipped[m], rtol=2e-3, atol=1e-5)


def test_cbed_library_common_grid():
    si = _si(absorptive=True)
    q0 = _zone_110()
    quats = torch.stack([q0, q0])
    lib = bloch.calculate_cbed_library(
        si, quats, thickness_A=600.0,
        energy_ev=200e3, semiconv_mrad=3.0, k_max=1.0, progress_bar=False,
    )
    assert lib["patterns"].shape[0] == 2
    assert lib["patterns"].shape[1] == lib["patterns"].shape[2]
    assert np.allclose(lib["patterns"][0], lib["patterns"][1])
    assert lib["patterns"][0].max() > 0


def test_kossel_bright_field_matches_lacbed():
    si = _si(absorptive=True)
    q = _zone_110()
    kw = dict(energy_ev=200e3, semiconv_mrad=15.0, sg_max=0.06, k_max=1.2)
    kos = bloch.calculate_kossel(
        si, q, 900.0, n_pixels=32, progress_bar=False, **kw
    )
    lac = bloch.calculate_lacbed(si, q, 900.0, hkl=(0, 0, 0), n_pixels=32, **kw)
    a, b = kos["bright_field"], lac["disk"]
    m = np.isfinite(a) & np.isfinite(b)
    assert m.sum() > 300
    assert np.allclose(a[m], b[m], rtol=1e-10, atol=1e-12)

    # the summed pattern includes the direct beam plus every diffracted
    # cone, so inside the aperture it can only exceed the bright field
    pat = kos["pattern"]
    assert (pat[m] >= a[m] - 1e-9).mean() > 0.99
    assert np.all(pat[~np.isfinite(a)] == 0)


def test_master_pattern_lookup():
    from scipy.ndimage import gaussian_filter

    si = _si(absorptive=True)
    q = _zone_110()
    # the master stores the pattern at its own angular resolution
    # (angle_step_mrad); compare against the direct calculation blurred to
    # the same resolution
    master = bloch.calculate_kossel_master(
        si, [800.0], energy_ev=200e3,
        angle_step_mrad=2.0, sg_max=0.06, k_max=1.0, progress_bar=False,
    )
    fast = bloch.kossel_from_master(master, q, semiconv_mrad=25.0, n_pixels=48)
    direct = bloch.calculate_kossel(
        si, q, 800.0, energy_ev=200e3, semiconv_mrad=25.0,
        n_pixels=48, sg_max=0.06, k_max=1.0, progress_bar=False,
    )
    a, b = fast["bright_field"], direct["bright_field"]
    m = np.isfinite(a) & np.isfinite(b)
    assert m.sum() > 1000
    # blur both to the master's angular resolution before comparing
    px_mrad = 2 * 25.0 / 48
    sigma = 2.0 / px_mrad / 2.355
    af = gaussian_filter(np.nan_to_num(a), sigma)
    bf = gaussian_filter(np.nan_to_num(b), sigma)
    cc = np.corrcoef(af[m], bf[m])[0, 1]
    assert cc > 0.9
