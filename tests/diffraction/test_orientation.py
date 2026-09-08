"""Round-trip tests for quantem.diffraction.orientation."""

import numpy as np
import pytest
import torch
from ase.build import bulk

from quantem.core.datastructures.vector import Vector
from quantem.diffraction.crystal import Crystal
from quantem.diffraction.orientation import OrientationMap
from quantem.diffraction.rotations import misorientation_angle_deg, qnormalize


def _make_peaks(xtl, q_true, sigma=0.02):
    N = q_true.shape[0]
    peaks = Vector.from_shape(
        (1, N), fields=["qx", "qy", "intensity"], units=["A^-1"] * 3, name="t"
    )
    for i in range(N):
        p = xtl.generate_pattern(q_true[i], energy_ev=200e3, sigma_excitation=sigma)
        peaks[0, i] = np.stack([p["qx"].numpy(), p["qy"].numpy(), p["intensity"].numpy()], axis=1)
    return peaks


@pytest.mark.parametrize(
    "builder,kwargs",
    [
        (bulk, dict(name="Ti", crystalstructure="bcc", a=3.31, cubic=True)),
        (bulk, dict(name="Ti", crystalstructure="hcp", a=2.95, c=4.686)),
    ],
)
def test_roundtrip_matching(builder, kwargs):
    torch.manual_seed(3)
    xtl = Crystal.from_ase(builder(**kwargs))
    xtl.calculate_structure_factors(k_max=1.5)
    N = 15
    q_true = qnormalize(torch.randn(N, 4, dtype=torch.float64))
    peaks = _make_peaks(xtl, q_true)

    om = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3)
    om.build_plan(angle_step_zone_axis_deg=2.0, angle_step_in_plane_deg=2.0, power_intensity=0.0)
    om.match_orientations(progress_bar=False)
    # noiseless synthetic data: the envelope tilt is exact, so allow the
    # full grid-scale correction (the default trust region is sized for
    # noisy measured intensities)
    om.refine_orientations(zone_max_total_deg=1.5, progress_bar=False)

    err = misorientation_angle_deg(q_true, om.quats[0, :, 0], xtl.sym_quats).numpy()
    # majority recovered to well below the grid step; a small number of
    # kinematically (near-)degenerate orientations may land elsewhere
    assert np.median(err) < 0.1
    assert (err < 1.0).mean() >= 0.7


def test_normalized_scores_and_reliability():
    torch.manual_seed(0)
    xtl = Crystal.from_ase(bulk("Ti", "bcc", a=3.31, cubic=True))
    xtl.calculate_structure_factors(k_max=1.5)
    q_true = qnormalize(torch.randn(6, 4, dtype=torch.float64))
    peaks = _make_peaks(xtl, q_true)
    om = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3)
    om.build_plan(angle_step_zone_axis_deg=3.0, angle_step_in_plane_deg=3.0)
    om.match_orientations(progress_bar=False)

    assert float(om.corr.max()) <= 1.0 + 1e-9
    assert float(om.corr.min()) >= 0.0
    assert om.reliability is not None
    assert (om.reliability[0] > 0).all()


def test_mirror_channel():
    """Orientations in the opposite hemisphere are matched via the mirror."""
    torch.manual_seed(5)
    xtl = Crystal.from_ase(bulk("Ti", "bcc", a=3.31, cubic=True))
    xtl.calculate_structure_factors(k_max=1.5)
    q_true = qnormalize(torch.randn(10, 4, dtype=torch.float64))
    peaks = _make_peaks(xtl, q_true)
    om = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3)
    om.build_plan(angle_step_zone_axis_deg=2.0, angle_step_in_plane_deg=2.0)
    om.match_orientations(progress_bar=False)
    om.refine_orientations(progress_bar=False)
    err = misorientation_angle_deg(q_true, om.quats[0, :, 0], xtl.sym_quats).numpy()
    used_mirror = om.mirror[0, :, 0].numpy()
    # both channels appear and mirror matches are as accurate as direct ones
    assert used_mirror.any()
    assert (~used_mirror).any()
    ok = err < 5
    assert ok.mean() >= 0.7
    assert np.median(err[ok & used_mirror]) < 0.5


def test_square_detector_correction():
    """Peaks clipped by a square detector: the aperture-normalized match
    recovers the orientation as well as the unclipped case."""
    torch.manual_seed(7)
    xtl = Crystal.from_ase(bulk("Ti", "hcp", a=2.95, c=4.686))
    xtl.calculate_structure_factors(k_max=1.5)
    q_true = qnormalize(torch.randn(10, 4, dtype=torch.float64))
    q_det = 0.9  # detector half-width < k_max: corners clipped

    peaks = Vector.from_shape(
        (1, 10), fields=["qx", "qy", "intensity"], units=["A^-1"] * 3, name="t"
    )
    for i in range(10):
        p = xtl.generate_pattern(q_true[i], energy_ev=200e3, sigma_excitation=0.02)
        keep = (p["qx"].abs() < q_det) & (p["qy"].abs() < q_det)
        peaks[0, i] = np.stack(
            [p["qx"][keep].numpy(), p["qy"][keep].numpy(), p["intensity"][keep].numpy()],
            axis=1,
        )

    om = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3)
    om.build_plan(
        angle_step_zone_axis_deg=2.0,
        angle_step_in_plane_deg=2.0,
        detector_q_max=q_det,
    )
    om.match_orientations(progress_bar=False)
    err = misorientation_angle_deg(q_true, om.quats[0, :, 0], xtl.sym_quats).numpy()
    assert (err < 5).mean() >= 0.7
    # with the aperture correction, kernel leakage at the hard detector edge
    # can push the normalized score a few percent above 1
    assert float(om.corr.max()) <= 1.05


def _ase(spacegroup, symbols, basis, cellpar):
    from ase.spacegroup import crystal as ase_crystal

    return ase_crystal(symbols, basis=basis, spacegroup=spacegroup, cellpar=cellpar)


@pytest.mark.parametrize(
    "label,atoms,step",
    [
        (
            "Bi -3m",
            lambda: _ase(166, ["Bi"], [(0, 0, 0.234)], [4.55, 4.55, 11.86, 90, 90, 120]),
            2.0,
        ),
        # ilmenite's projections are nearly mirror symmetric, so the flipped
        # orientation is a close rival and needs the finer zone grid
        (
            "ilmenite -3",
            lambda: _ase(
                148,
                ["Fe", "Ti", "O"],
                [(0, 0, 0.355), (0, 0, 0.146), (0.317, 0.023, 0.245)],
                [5.09, 5.09, 14.09, 90, 90, 120],
            ),
            1.0,
        ),
    ],
)
def test_roundtrip_low_symmetry(label, atoms, step):
    # low-symmetry crystals see errors that cubic and hexagonal symmetry
    # hides: a wrong wedge (trigonal) or a redundant library
    torch.manual_seed(5)
    xtl = Crystal.from_ase(atoms(), name=label, verbose=False)
    xtl.calculate_structure_factors(k_max=1.3)
    N = 20
    q_true = qnormalize(torch.randn(N, 4, dtype=torch.float64))
    peaks = _make_peaks(xtl, q_true)
    om = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3)
    om.build_plan(angle_step_zone_axis_deg=step, angle_step_in_plane_deg=2.0, verbose=False)
    om.match_orientations(progress_bar=False)
    err = misorientation_angle_deg(q_true, om.quats[0, :, 0], xtl.sym_quats).numpy()
    assert (err < 2.5).mean() >= 0.8
    # symmetry copies of the best zone must not count as the second best
    assert float(np.median(om.reliability[0].numpy())) > 0.02


def test_reliability_with_hemisphere_library():
    # Laue 2/m has no wedge: the hemisphere library holds every zone twice,
    # and reliability must still see past the symmetry copy
    torch.manual_seed(2)
    atoms = _ase(
        14,
        ["Zr", "O", "O"],
        [(0.275, 0.040, 0.208), (0.070, 0.332, 0.345), (0.450, 0.758, 0.479)],
        [5.15, 5.21, 5.32, 90, 99.2, 90],
    )
    xtl = Crystal.from_ase(atoms, name="ZrO2", pseudo_symmetry_tol=None, verbose=False)
    xtl.calculate_structure_factors(k_max=1.2)
    assert xtl.zone_axis_wedge() is None
    q_true = qnormalize(torch.randn(8, 4, dtype=torch.float64))
    peaks = _make_peaks(xtl, q_true)
    om = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3)
    om.build_plan(angle_step_zone_axis_deg=3.0, angle_step_in_plane_deg=3.0, verbose=False)
    om.match_orientations(progress_bar=False)
    assert float(np.median(om.reliability[0].numpy())) > 0.02


def test_pseudo_symmetry_warning_on_plan():
    import warnings

    from ase import Atoms

    ortho = Atoms("Au", positions=[[0, 0, 0]], cell=[4.000, 4.001, 4.002], pbc=True)
    xtl = Crystal.from_ase(ortho, verbose=False)
    xtl.calculate_structure_factors(k_max=1.2)
    q_true = qnormalize(torch.randn(2, 4, dtype=torch.float64))
    peaks = _make_peaks(xtl, q_true)
    om = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        om.build_plan(angle_step_zone_axis_deg=5.0, angle_step_in_plane_deg=5.0, verbose=False)
    assert any("pseudo-symmetry" in str(x.message) for x in w)


def test_metadata_inheritance():
    # each stage records its hyperparameters; later stages inherit what is
    # left as None, so one tuned value propagates through the whole chain
    from quantem.diffraction import bloch
    from quantem.diffraction.phase import PhaseMap

    torch.manual_seed(1)
    xtl = Crystal.from_ase(bulk("Ti", "bcc", a=3.31, cubic=True), verbose=False)
    xtl.calculate_structure_factors(k_max=1.5)
    q_true = qnormalize(torch.randn(4, 4, dtype=torch.float64))
    peaks = _make_peaks(xtl, q_true)
    om = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3, precession_deg=0.7)
    om.build_plan(
        angle_step_zone_axis_deg=3.0, corr_kernel_size=0.04, power_intensity=0.3, verbose=False
    )
    om.match_orientations(progress_bar=False)
    om.refine_orientations(progress_bar=False)
    assert om.metadata["plan"]["pair_distance"] == 0.04
    assert om.metadata["refine"]["pair_distance"] == 0.04
    assert om.metadata["match"]["min_number_peaks"] == 5
    pm = PhaseMap.from_orientation_maps([om])
    pm.fit(progress_bar=False)
    assert pm.metadata["fit"]["pair_distance"] == 0.04
    assert pm.metadata["fit"]["power_intensity"] == 0.3
    assert pm.metadata["fit"]["min_number_peaks"] == 5
    xtl.calculate_dynamical_structure_factors(energy_ev=200e3, k_max=2.0)
    res = bloch.refine_dynamical(
        pm,
        thicknesses_A=[300.0, 400.0],
        tilt_stages=((0.1, 0.1),),
        n_precession=4,
        k_max=1.0,
        mask=np.array([[True, False, False, False]]),
        progress_bar=False,
    )
    md = res["metadata"]
    assert md["precession_deg"] == 0.7 and md["pair_distance"] == 0.04
    assert md["power_intensity"] == 0.3 and md["min_number_peaks"] == 5
    assert pm.metadata["dynamical"] is md
    # explicit values still win
    res2 = bloch.refine_dynamical(
        pm,
        thicknesses_A=[300.0],
        tilt_stages=((0.1, 0.1),),
        n_precession=4,
        precession_deg=0.0,
        pair_distance=0.06,
        k_max=1.0,
        mask=np.array([[True, False, False, False]]),
        progress_bar=False,
    )
    assert res2["metadata"]["precession_deg"] == 0.0 and res2["metadata"]["pair_distance"] == 0.06


def test_precession_envelope_matches_quadrature():
    # the analytic ring-averaged envelope equals the positive quadrature of
    # the static envelope over the exact excitation errors on the ring
    from quantem.diffraction.illumination import ring_disk_quadrature
    from quantem.diffraction.rotations import qrotate

    xtl = Crystal.from_ase(bulk("Ti", "bcc", a=3.31, cubic=True), verbose=False)
    xtl.calculate_structure_factors(k_max=1.5)
    torch.manual_seed(4)
    q = qnormalize(torch.randn(4, dtype=torch.float64))
    energy_ev, sigma, prec = 200e3, 0.04, 0.6
    from quantem.core.utils.utils import electron_wavelength_angstrom

    lam = electron_wavelength_angstrom(energy_ev)
    k0 = 1.0 / lam
    pat = xtl.generate_pattern(q, energy_ev, sigma_excitation=sigma, precession_deg=prec)
    g = qrotate(q, xtl.g_vec)
    hkl_map = {tuple(h): i for i, h in enumerate(xtl.hkl.tolist())}
    idx = torch.tensor([hkl_map[tuple(h)] for h in pat["hkl"].tolist()])
    gs = g[idx].numpy()
    r = k0 * np.sin(np.deg2rad(prec))
    t, w = ring_disk_quadrature(r, 0.0, n_phi=256)
    kz = np.sqrt(k0**2 - (t**2).sum(1))[:, None]
    s = (2 * kz * gs[:, 2] - 2 * (t @ gs[:, :2].T) - (gs**2).sum(1)) / (2 * (kz - gs[:, 2]))
    ref = (w[:, None] * np.exp(-0.5 * (s / sigma) ** 2)).sum(0) * xtl.struct_factors_int[
        idx
    ].numpy()
    assert np.allclose(pat["intensity"].numpy(), ref, rtol=1e-6, atol=1e-9)
    # without precession the static envelope is recovered exactly
    pat0 = xtl.generate_pattern(q, energy_ev, sigma_excitation=sigma)
    s0 = pat0["s_g"].numpy()
    assert np.allclose(
        pat0["intensity"].numpy(),
        xtl.struct_factors_int[[hkl_map[tuple(h)] for h in pat0["hkl"].tolist()]].numpy()
        * np.exp(-0.5 * (s0 / sigma) ** 2),
    )


def test_roundtrip_with_precession():
    # library, matching and refinement with the precession-averaged
    # envelope: patterns simulated with precession are recovered
    torch.manual_seed(6)
    xtl = Crystal.from_ase(bulk("Ti", "hcp", a=2.95, c=4.686), verbose=False)
    xtl.calculate_structure_factors(k_max=1.5)
    N = 12
    q_true = qnormalize(torch.randn(N, 4, dtype=torch.float64))
    peaks = Vector.from_shape(
        (1, N), fields=["qx", "qy", "intensity"], units=["A^-1"] * 3, name="t"
    )
    for i in range(N):
        p = xtl.generate_pattern(
            q_true[i], energy_ev=200e3, sigma_excitation=0.02, precession_deg=0.7
        )
        peaks[0, i] = np.stack([p["qx"].numpy(), p["qy"].numpy(), p["intensity"].numpy()], axis=1)
    om = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3, precession_deg=0.7)
    om.build_plan(angle_step_zone_axis_deg=2.0, verbose=False)
    om.match_orientations(progress_bar=False)
    om.refine_orientations(zone_max_total_deg=1.5, progress_bar=False)
    err = misorientation_angle_deg(q_true, om.quats[0, :, 0], xtl.sym_quats).numpy()
    assert np.median(err) < 0.3
    assert (err < 1.5).mean() >= 0.75
    assert om.metadata["precession_deg"] == 0.7
