import numpy as np
import pytest
import torch

from quantem.diffraction import Crystal, ReverseMonteCarlo
from quantem.diffraction.reverse_monte_carlo import cubic_rotations

CIF = """data_VNb
_cell_length_a 3.2
_cell_length_b 3.2
_cell_length_c 3.2
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'I m -3 m'
_symmetry_Int_Tables_number 229
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Nb1 Nb 0 0 0 0.7
V1 V 0 0 0 0.3
"""


@pytest.fixture
def rmc(tmp_path):
    path = tmp_path / "vnb.cif"
    path.write_text(CIF)
    rng = np.random.default_rng(1)
    images = [rng.random((64, 64)) + 1.0 for _ in range(2)]
    out = ReverseMonteCarlo.from_images(
        images, zone_axes=[(0, 0, 1), (0, 1, 1)], sampling=0.02, bin_factor=2
    )
    out.set_crystal(Crystal.from_cif(path, verbose=False))
    out.geometry = dict(
        centers=[np.array([32.0, 32.0])] * 2,
        matrices=[np.eye(2) / 0.02, np.array([[0.0, -1.0], [1.0, 0.0]]) / 0.02],
        tilts=[np.zeros(2), np.array([0.01, 0.0])],
    )
    out.set_mask(bragg_radius=0.06, q_max=0.6, center_radius=0.1, edge_px=2)
    out.build_supercell(cells=4, seed=0, device="cpu")
    out.fit_background()
    return out


def test_cubic_rotations():
    ops = cubic_rotations()
    assert ops.shape == (24, 3, 3)
    assert np.allclose([np.linalg.det(o) for o in ops], 1)
    assert len({o.tobytes() for o in ops}) == 24


def test_binary_site_and_composition(rmc):
    assert rmc.species == ["Nb", "V"]
    assert rmc.concentration == pytest.approx(0.3)
    assert len(rmc.sigma) == 2 * 4**3
    assert rmc.sigma.sum() == round(0.3 * len(rmc.sigma))


def test_swap_score_matches_recompute(rmc):
    """The incremental loss change of one swap equals the loss after recomputing G from scratch."""
    loss0 = rmc._update_residual()
    j_on = int(np.nonzero(~rmc.sigma)[0][0])
    j_off = int(np.nonzero(rmc.sigma)[0][0])
    n_sites = len(rmc.sigma)
    c1, s1 = rmc._phases(torch.tensor([j_on]))
    c2, s2 = rmc._phases(torch.tensor([j_off]))
    dGr, dGi = c1 - c2, -s1 + s2
    d_int = (2 * (rmc._Gr * dGr + rmc._Gi * dGi) + dGr**2 + dGi**2) / n_sites
    d_used = d_int[:, rmc._sym_index].mean(dim=1)
    dm = rmc._scale * rmc._u * rmc._read(d_used)
    dL = float((rmc._w * (dm**2 - 2 * rmc._r * dm)).sum())

    rmc.sigma[j_on], rmc.sigma[j_off] = True, False
    rmc._recompute_G()
    loss1 = rmc._update_residual()
    assert loss1 - loss0 == pytest.approx(dL, rel=1e-3, abs=1e-4 * loss0)


def test_run_lowers_loss_and_keeps_composition(rmc):
    n_b = rmc.sigma.sum()
    rmc.run(n_sweeps=3, batch=8, progress=False)
    assert rmc.sigma.sum() == n_b
    assert rmc.loss_history[-1] <= rmc.loss_history[0] + 1e-6


def test_warren_cowley_random_is_near_zero(rmc):
    sro = rmc.warren_cowley(n_shells=2)
    assert np.allclose(sro["radius"], [3.2 * np.sqrt(3) / 2, 3.2], atol=1e-3)
    assert np.all(np.abs(sro["alpha"]) < 0.15)


def test_mask_is_zero_on_bragg_peaks(rmc):
    w = rmc.mask["w"][0]
    b = rmc.bin_factor
    for r, c in rmc.bragg_positions(0):
        r, c = int(r // b), int(c // b)
        if 0 <= r < w.shape[0] and 0 <= c < w.shape[1]:
            assert w[r, c] < 0.3


def test_sro_section_random_is_near_laue(rmc):
    img, _, node_dist = rmc.diffuse_section((0, 0, 1), extent=1.0, smooth=True)
    between = img[node_dist > 0.2]
    assert 0.5 < between.mean() < 1.5
