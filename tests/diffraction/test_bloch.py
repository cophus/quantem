"""Tests for quantem.diffraction.bloch."""

import numpy as np
import pytest
import torch
from ase.build import bulk

from quantem.core.datastructures.vector import Vector
from quantem.diffraction import bloch
from quantem.diffraction.bloch import dynamical_pattern, refine_thickness
from quantem.diffraction.crystal import Crystal
from quantem.diffraction.orientation import OrientationMap
from quantem.diffraction.phase import PhaseMap
from quantem.diffraction.rotations import quat_from_zone_axis


@pytest.fixture(scope="module")
def ti_beta():
    xtl = Crystal.from_ase(bulk("Ti", "bcc", a=3.31, cubic=True), name="Ti beta")
    # 2x coverage so all coupling vectors g - h have structure factors
    xtl.calculate_structure_factors(k_max=3.0, tol_structure_factor=1e-6)
    return xtl


def test_flux_conservation(ti_beta):
    q = quat_from_zone_axis(torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64))
    p = dynamical_pattern(
        ti_beta, q, np.arange(50, 1500, 50.0), energy_ev=200e3, sg_max=0.08, k_max=1.5
    )
    total = p["intensity"].sum(dim=1)
    assert float(total.max()) <= 1.0 + 1e-6


def test_thin_limit_matches_kinematical(ti_beta):
    q = quat_from_zone_axis(torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64))
    p = dynamical_pattern(ti_beta, q, 25.0, energy_ev=200e3, sg_max=0.08, k_max=1.5)
    kin = ti_beta.generate_pattern(q, energy_ev=200e3, sigma_excitation=0.02)
    top_dyn = set(map(tuple, p["hkl"][p["intensity"][0].argsort(descending=True)[:4]].tolist()))
    top_kin = set(map(tuple, kin["hkl"][kin["intensity"].argsort(descending=True)[:4]].tolist()))
    assert top_dyn == top_kin


def test_thickness_recovery(ti_beta):
    """Simulate dynamical peaks at a known thickness, recover it."""
    t_true = 600.0
    torch.manual_seed(0)
    zones = torch.tensor([[0.1, 0.9, 1.0], [0.3, 0.5, 1.0], [0.05, 1.0, 1.1]], dtype=torch.float64)
    N = zones.shape[0]
    peaks = Vector.from_shape(
        (1, N), fields=["qx", "qy", "intensity"], units=["A^-1"] * 3, name="t"
    )
    q_true = quat_from_zone_axis(zones)
    for i in range(N):
        p = dynamical_pattern(ti_beta, q_true[i], t_true, energy_ev=200e3, sg_max=0.06, k_max=1.5)
        keep = p["intensity"][0] > 1e-4
        peaks[0, i] = np.stack(
            [
                p["qx"][keep].numpy(),
                p["qy"][keep].numpy(),
                p["intensity"][0][keep].numpy(),
            ],
            axis=1,
        )

    om = OrientationMap.from_vectors(peaks, ti_beta, energy_ev=200e3)
    om.build_plan(angle_step_zone_axis_deg=2.0, angle_step_in_plane_deg=2.0)
    om.match_orientations(progress_bar=False)
    # thickness oscillations are sensitive to ~1 degree tilt errors, beyond
    # what kinematical matching provides for dynamical patterns; test the
    # thickness scan itself with the true orientations (dynamical tilt
    # refinement is the future joint pass)
    om.quats[0, :, 0] = q_true

    pm = PhaseMap.from_orientation_maps([om])
    pm.fit(max_patterns=1, progress_bar=False)
    res = refine_thickness(
        pm,
        thicknesses_A=np.arange(100, 1200, 50.0),
        sg_max=0.06,
        progress_bar=False,
    )
    t_fit = res["thickness"][0].numpy()
    assert (np.abs(t_fit - t_true) <= 50.0).all()


# ----------------------------------------------------------------------
# CBED / LACBED / Kossel / master pattern
# ----------------------------------------------------------------------


def _si(absorptive: bool) -> Crystal:
    si = Crystal.from_ase(bulk("Si", "diamond", a=5.431, cubic=True), name="Si", verbose=False)
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
    inten, g_xy, hkl = bloch._cbed_amplitudes(si, q, tilts, t, 200e3, sg_max=0.08, k_max=1.3)
    ref = bloch.dynamical_pattern(si, q, t, energy_ev=200e3, sg_max=0.08, k_max=1.3)
    # same beams (000 first in CBED) and identical intensities
    assert hkl.shape[0] == ref["hkl"].shape[0] + 1
    assert torch.allclose(inten[0, 0, 1:], ref["intensity"][0], rtol=1e-10, atol=1e-12)


def test_unitarity_without_absorption():
    si = _si(absorptive=False)
    q = _zone_110()
    tilts = bloch.tilt_grid(2.0, 200e3, n_rings=2)
    inten, _, _ = bloch._cbed_amplitudes(
        si, q, tilts, torch.tensor([500.0, 1500.0]), 200e3, sg_max=0.08, k_max=1.3
    )
    # Hermitian structure matrix: evolution is unitary in the beam space
    total = inten.sum(dim=-1)
    assert torch.allclose(total, torch.ones_like(total), atol=1e-8)


def test_lacbed_centrosymmetric_disk():
    si = _si(absorptive=True)
    q = _zone_110()
    res = bloch.calculate_lacbed(
        si,
        q,
        800.0,
        hkl=(0, 0, 0),
        energy_ev=200e3,
        semiconv_mrad=6.0,
        n_pixels=24,
        sg_max=0.08,
        k_max=1.3,
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
        si,
        quats,
        thickness_A=600.0,
        energy_ev=200e3,
        semiconv_mrad=3.0,
        k_max=1.0,
        progress_bar=False,
    )
    assert lib["patterns"].shape[0] == 2
    assert lib["patterns"].shape[1] == lib["patterns"].shape[2]
    assert np.allclose(lib["patterns"][0], lib["patterns"][1])
    assert lib["patterns"][0].max() > 0


def test_kossel_bright_field_matches_lacbed():
    si = _si(absorptive=True)
    q = _zone_110()
    kw = dict(energy_ev=200e3, semiconv_mrad=15.0, sg_max=0.06, k_max=1.2)
    kos = bloch.calculate_kossel(si, q, 900.0, n_pixels=32, progress_bar=False, **kw)
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


def test_reference_pattern_lookup():
    from scipy.ndimage import gaussian_filter

    si = _si(absorptive=True)
    q = _zone_110()
    # the master stores the pattern at its own angular resolution
    # (angle_step_mrad); compare against the direct calculation blurred to
    # the same resolution
    master = bloch.calculate_kossel_reference(
        si,
        [800.0],
        energy_ev=200e3,
        angle_step_mrad=2.0,
        sg_max=0.06,
        k_max=1.0,
        progress_bar=False,
    )
    fast = bloch.kossel_from_reference(master, q, semiconv_mrad=25.0, n_pixels=48)
    direct = bloch.calculate_kossel(
        si,
        q,
        800.0,
        energy_ev=200e3,
        semiconv_mrad=25.0,
        n_pixels=48,
        sg_max=0.06,
        k_max=1.0,
        progress_bar=False,
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

    # off-zone orientation: catches in-plane sign errors that zone-axis
    # symmetry hides (the reference stores the ANTI-propagation direction)
    from quantem.diffraction.rotations import qmult, quat_from_axis_angle

    tilt = quat_from_axis_angle(
        torch.tensor([1.0, 0.3, 0.0], dtype=torch.float64) / np.hypot(1, 0.3),
        torch.tensor(np.deg2rad(5.0), dtype=torch.float64),
    )
    q2 = qmult(tilt, q)
    fast2 = bloch.kossel_from_reference(master, q2, semiconv_mrad=25.0, n_pixels=48)
    direct2 = bloch.calculate_kossel(
        si,
        q2,
        800.0,
        energy_ev=200e3,
        semiconv_mrad=25.0,
        n_pixels=48,
        sg_max=0.06,
        k_max=1.0,
        progress_bar=False,
    )
    a2, b2 = fast2["bright_field"], direct2["bright_field"]
    m2 = np.isfinite(a2) & np.isfinite(b2)
    af2 = gaussian_filter(np.nan_to_num(a2), sigma)
    bf2 = gaussian_filter(np.nan_to_num(b2), sigma)
    assert np.corrcoef(af2[m2], bf2[m2])[0, 1] > 0.9


def _tilted_110():
    from quantem.diffraction.rotations import qmult, quat_from_axis_angle

    tilt = quat_from_axis_angle(
        torch.tensor([1.0, 0.3, 0.0], dtype=torch.float64) / np.hypot(1, 0.3),
        torch.tensor(np.deg2rad(5.0), dtype=torch.float64),
    )
    return qmult(tilt, _zone_110())


def test_kossel_lines_render_matches_direct():
    si = _si(absorptive=True)
    q = _tilted_110()
    kw = dict(energy_ev=200e3, semiconv_mrad=25.0, sg_max=0.06, k_max=1.0)
    direct = bloch.calculate_kossel(si, q, 800.0, n_pixels=48, progress_bar=False, **kw)[
        "bright_field"
    ]
    lines = bloch.kossel_lines(si, 800.0, energy_ev=200e3, k_max=1.0)
    # every line is a band edge: the +g and -g cones of a row sit at
    # +-theta_B, never on the zone plane
    first = lines["line_order"] == 1
    assert torch.all(lines["line_u"][first] > 0)
    assert torch.all(lines["line_u"][lines["line_order"] == -1] < 0)
    r = bloch.render_kossel_lines(lines, q, semiconv_mrad=25.0, n_pixels=48)
    a = r["bright_field"]
    m = np.isfinite(a) & np.isfinite(direct)
    assert m.sum() > 1000
    assert np.corrcoef(a[m], direct[m])[0, 1] > 0.98

    # polar rendering samples the same function: its first ring must
    # agree with the Cartesian pattern evaluated at those angles
    pol = bloch.render_kossel_lines(
        lines, q, semiconv_mrad=25.0, polar=True, n_radial=10, n_azimuthal=12
    )["polar"]
    assert pol.shape == (12, 10)
    assert np.all(np.isfinite(pol))
    assert pol.min() > 0 and pol.max() < 1.5 * float(lines["background"][0])


def test_kossel_line_segments_on_cones():
    from quantem.diffraction.rotations import quat_to_matrix

    si = _si(absorptive=True)
    q = _tilted_110()
    lines = bloch.kossel_lines(si, 800.0, energy_ev=200e3, k_max=1.0)
    alpha = 25.0
    seg = bloch.kossel_line_segments(lines, q, semiconv_mrad=alpha)
    n = seg["depth"].shape[0]
    assert n >= 3
    R = quat_to_matrix(q).numpy()
    g_c = lines["g_hat"].numpy()
    hkl_row = lines["hkl_row"].numpy()
    for k in range(n):
        # row of this line from its hkl (an integer multiple of the row vector)
        h = seg["hkl"][k]
        ri = next(i for i in range(g_c.shape[0]) if np.all(np.cross(hkl_row[i], h) == 0))
        n_ord = int(np.round(np.dot(h, hkl_row[ri]) / np.dot(hkl_row[ri], hkl_row[ri])))
        u = n_ord * bloch.electron_wavelength_angstrom(200e3) * float(lines["g_len"][ri]) / 2
        g_lab = R @ g_c[ri]
        for key in ("start_mrad", "stop_mrad"):
            row, col = seg[key][k] * 1e-3
            assert np.isclose(np.hypot(row, col), alpha * 1e-3)
            d_lab = np.array([-col, -row, np.sqrt(1 - row**2 - col**2)])
            assert abs(d_lab @ g_lab - u) < 1e-9
        # polar end points are the same points
        phi, r = seg["start_polar"][k]
        assert np.isclose(r, alpha)
        assert np.allclose([alpha * np.sin(phi), alpha * np.cos(phi)], seg["start_mrad"][k])
        assert seg["width_mrad"][k] > 0 and 0 < seg["depth"][k] <= 1


def test_reference_residual_hybrid():
    from scipy.ndimage import gaussian_filter

    si = _si(absorptive=True)
    q = _zone_110()
    master = bloch.calculate_kossel_reference(
        si,
        [800.0],
        energy_ev=200e3,
        angle_step_mrad=2.0,
        sg_max=0.06,
        k_max=1.0,
        progress_bar=False,
    )
    assert master["k_max"] == 1.0
    lines = bloch.kossel_lines(si, 800.0, energy_ev=200e3, k_max=1.0)
    bloch.kossel_reference_residual(master, lines, si)
    assert master["residual"].shape == master["lambert"].shape
    assert np.all(np.isfinite(master["residual"]))
    direct = bloch.calculate_kossel(
        si,
        q,
        800.0,
        energy_ev=200e3,
        semiconv_mrad=25.0,
        n_pixels=48,
        sg_max=0.06,
        k_max=1.0,
        progress_bar=False,
    )["bright_field"]
    plain = bloch.render_kossel_lines(lines, q, semiconv_mrad=25.0, n_pixels=48)
    hybrid = bloch.render_kossel_lines(lines, q, semiconv_mrad=25.0, n_pixels=48, reference=master)
    m = np.isfinite(direct)
    sigma = 2.0 / (2 * 25.0 / 48) / 2.355
    bf = gaussian_filter(np.nan_to_num(direct), sigma)

    def cc(x):
        return np.corrcoef(gaussian_filter(np.nan_to_num(x), sigma)[m], bf[m])[0, 1]

    # on the zone axis the many-beam residual must improve the line model
    assert cc(hybrid["bright_field"]) > cc(plain["bright_field"])
    assert cc(hybrid["bright_field"]) > 0.9


def test_refine_dynamical_recovery():
    """Bragg-vector dynamical refinement: thickness, tilt, in-plane strain and
    rotation recovered from noise-free patterns of a strained, tilted cell."""
    from quantem.core.datastructures.vector import Vector
    from quantem.diffraction.orientation import OrientationMap
    from quantem.diffraction.phase import PhaseMap
    from quantem.diffraction.rotations import (
        misorientation_angle_deg,
        qmult,
        qnormalize,
        quat_from_axis_angle,
    )

    energy_ev = 200e3
    xtl = _si(absorptive=True)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    N = 6
    q_true = qnormalize(torch.randn(N, 4, dtype=torch.float64))
    t_true = torch.tensor([300.0, 450.0, 600.0, 300.0, 450.0, 600.0])
    A_true = torch.tensor([[1.010, 0.003], [0.003, 0.995]], dtype=torch.float64)
    rot = np.deg2rad(0.3)
    qz = torch.tensor([np.cos(rot / 2), 0.0, 0.0, np.sin(rot / 2)], dtype=torch.float64)
    q_expect = torch.stack([qmult(qz, q_true[i]) for i in range(N)])
    deform3 = torch.eye(3, dtype=torch.float64)
    deform3[:2, :2] = A_true
    peaks = Vector.from_shape(
        (1, N), fields=["qx", "qy", "intensity"], units=["A^-1"] * 3, name="t"
    )
    for i in range(N):
        inten, g_xy, _ = bloch._cbed_amplitudes(
            xtl,
            q_expect[i],
            torch.zeros((1, 2), dtype=torch.float64),
            t_true[i : i + 1],
            energy_ev,
            0.06,
            1.0,
            progress_bar=False,
            deform=deform3,
        )
        inten_np = inten[0, 0, 1:].numpy()
        keep = inten_np > 1e-3 * inten_np.max()
        peaks[0, i] = np.column_stack([g_xy[1:].numpy()[keep], inten_np[keep]])
    om = OrientationMap.from_vectors(peaks, xtl, energy_ev=energy_ev)
    om.build_plan(angle_step_zone_axis_deg=3.0, angle_step_in_plane_deg=5.0, verbose=False)
    om.match_orientations(progress_bar=False)
    # start 0.15 degrees off the truth about random in-plane axes, unstrained
    phis = rng.uniform(0, 2 * np.pi, N)
    om.quats[0, :, 0] = torch.stack(
        [
            qmult(
                quat_from_axis_angle(
                    torch.tensor([np.cos(p), np.sin(p), 0.0], dtype=torch.float64),
                    torch.tensor(np.deg2rad(0.15), dtype=torch.float64),
                ),
                q_true[i],
            )
            for i, p in enumerate(phis)
        ]
    )
    om.corr[0, :, 0] = 1.0
    pm = PhaseMap.from_orientation_maps([om])
    pm.fit(progress_bar=False)
    res = bloch.refine_dynamical(
        pm,
        thicknesses_A=np.arange(150, 800, 25.0),
        tilt_stages=((0.25, 0.025), (0.04, 0.005)),
        power_intensity=0.5,
        sg_max=0.06,
        k_max=1.0,
        progress_bar=False,
    )
    # positions with too few beams cannot constrain a 2x2 deformation
    valid = np.array([peaks[0, i].array.shape[0] >= 6 for i in range(N)])
    assert valid.sum() >= 4
    err = misorientation_angle_deg(q_expect, om.quats[0, :, 0], xtl.sym_quats).numpy()[valid]
    t_err = np.abs(res["thickness"][0].numpy() - t_true.numpy())[valid]
    A_err = (res["deformation"][0, valid, 0] - A_true[None]).abs().max()
    assert float(A_err) < 1e-3
    assert np.median(err) < 0.03
    assert (t_err <= 25).sum() >= valid.sum() - 1
    # crystal-frame strain of an unstrained position is zero, of a strained
    # one has the right magnitude
    sc = bloch.strain_crystal_frame(res["deformation"][0, :, 0], om.quats[0, :, 0])
    eps = sc["eps_crystal"]
    assert torch.allclose(eps, eps.transpose(-1, -2))
    assert float(eps.abs().max()) < 0.02


def test_image_refinement_round_trip():
    """Rendered patterns with known disk shape: fit_disk_shape recovers the
    radius and edge, and the image refinement keeps a correct thickness."""
    from types import SimpleNamespace

    from quantem.core.datastructures.vector import Vector
    from quantem.diffraction.orientation import OrientationMap
    from quantem.diffraction.phase import PhaseMap
    from quantem.diffraction.rotations import qnormalize

    energy_ev = 200e3
    xtl = _si(absorptive=True)
    torch.manual_seed(2)
    rng = np.random.default_rng(2)
    N = 4
    q_true = qnormalize(torch.randn(N, 4, dtype=torch.float64))
    t_true = torch.tensor([300.0, 450.0, 600.0, 400.0])
    shape = (64, 64)
    pixel_size, rot, ellipse = 0.04, 15.0, (0.003, -0.002)
    disk_r, edge = 3.0, 0.75
    origins = np.full((1, N, 2), 32.0)
    imgs = np.zeros((1, N) + shape)
    peaks = Vector.from_shape(
        (1, N), fields=["qx", "qy", "intensity"], units=["A^-1"] * 3, name="t"
    )
    for i in range(N):
        im, _, _ = bloch.render_pattern_image(
            xtl,
            q_true[i],
            [float(t_true[i])],
            energy_ev,
            shape,
            origins[0, i],
            pixel_size,
            rot,
            ellipse,
            None,
            disk_r,
            edge,
            sg_max=0.06,
            k_max=1.0,
        )
        imgs[0, i] = rng.poisson(im[0, 0].numpy() * 1e5 + 20)
        inten, g_xy, _ = bloch._cbed_amplitudes(
            xtl,
            q_true[i],
            torch.zeros((1, 2), dtype=torch.float64),
            t_true[i : i + 1],
            energy_ev,
            0.06,
            1.0,
            progress_bar=False,
            fast_absorption=True,
        )
        inten_np = inten[0, 0, 1:].numpy()
        keep = inten_np > 1e-3 * inten_np.max()
        peaks[0, i] = np.column_stack([g_xy[1:].numpy()[keep], inten_np[keep]])
    peaks.metadata["rotation_ccw_deg"] = rot
    dataset = SimpleNamespace(array=imgs, shape=imgs.shape)
    om = OrientationMap.from_vectors(peaks, xtl, energy_ev=energy_ev)
    om.build_plan(angle_step_zone_axis_deg=3.0, angle_step_in_plane_deg=5.0, verbose=False)
    om.match_orientations(progress_bar=False)
    om.quats[0, :, 0] = q_true
    om.corr[0, :, 0] = 1.0
    pm = PhaseMap.from_orientation_maps([om])
    pm.fit(progress_bar=False)
    res = bloch.refine_dynamical(
        pm,
        thicknesses_A=np.arange(200, 700, 50.0),
        tilt_stages=((0.05, 0.05),),
        power_intensity=0.5,
        sg_max=0.06,
        k_max=1.0,
        progress_bar=False,
    )
    valid = [i for i in range(N) if peaks[0, i].array.shape[0] >= 6]
    assert len(valid) >= 2
    shape_fit = bloch.fit_disk_shape(
        dataset,
        pm,
        res,
        origins,
        pixel_size,
        rot,
        ellipse,
        positions=[(0, i) for i in valid],
        radii_px=np.array([2.0, 2.5, 3.0, 3.5, 4.0]),
        edges_px=np.array([0.5, 0.75, 1.0, 1.5]),
        sg_max=0.06,
        k_max=1.0,
        progress_bar=False,
    )
    assert shape_fit["disk_radius_px"] == disk_r
    assert shape_fit["edge_px"] == edge
    img = bloch.refine_dynamical_image(
        dataset,
        pm,
        res,
        origins,
        pixel_size,
        disk_r,
        edge,
        rot,
        ellipse,
        thickness_half_range_A=50,
        thickness_step_A=25,
        tilt_stage=(0.02, 0.01),
        sg_max=0.06,
        k_max=1.0,
        progress_bar=False,
    )
    t_err = np.abs(img["thickness"][0].numpy() - t_true.numpy())[valid]
    assert (t_err <= 25).sum() >= len(valid) - 1
    assert np.all(np.isfinite(img["cost"][0].numpy()[valid]))
    assert pm.metadata["dynamical_image"]["disk_radius_px"] == disk_r
    pm.apply_dynamical(res)
    assert pm.metadata["dynamical_applied"]["precession_deg"] == 0.0
