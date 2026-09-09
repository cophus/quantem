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


def test_coupling_lookup_cannot_alias():
    # a difference vector outside the stored factor box must come back as a
    # missing factor (zero), never as another reflection's factor
    from types import SimpleNamespace

    crystal = SimpleNamespace(
        hkl_dyn=torch.tensor([[1, 0, 0], [-1, 0, 0], [-1, 1, 0]]),
        U_dyn=torch.tensor([1, 1, 7], dtype=torch.complex128),
    )
    U, _, _ = bloch._coupling_matrix(crystal, torch.tensor([[1, 0, 0], [-1, 0, 0]]), 1.0)
    assert U[0, 1] == 0 and U[1, 0] == 0


def test_illumination_nodes_moments():
    # the convergence disk is integrated with the uniform-area measure: the
    # second moment of a disk of radius R is R^2 / 4 per axis; the ring is
    # normalized and its mean vanishes
    lam = bloch.electron_wavelength_angstrom(200e3)
    k0 = 1.0 / lam
    t, w = bloch.illumination_nodes(200e3, semiconv_mrad=5.0, n_disk_radial=3, n_disk_azimuthal=16)
    R = k0 * np.sin(5e-3)
    assert np.isclose(float(w.sum()), 1.0)
    assert np.isclose(float((w * t[:, 0] ** 2).sum()), R**2 / 4, rtol=1e-10)
    t, w = bloch.illumination_nodes(200e3, precession_deg=0.5, n_precession=16)
    assert np.isclose(float(w.sum()), 1.0) and float(t.mean(0).abs().max()) < 1e-12
    assert np.allclose(torch.linalg.norm(t, dim=1).numpy(), k0 * np.sin(np.deg2rad(0.5)))
    t, w = bloch.illumination_nodes(200e3)
    assert t.shape == (1, 2) and float(w[0]) == 1.0


def test_mean_absorption_and_forbidden_beam():
    """Pure mean absorption damps the total intensity as exp(-2 pi u0 z/k0);
    a glide-forbidden reflection (Si 200) acquires intensity through double
    diffraction, which requires it to be in the beam list."""
    si = _si(absorptive=True)
    q = _zone_110()
    z = torch.tensor([400.0, 800.0], dtype=torch.float64)
    inten, g_xy, hkl = bloch._cbed_amplitudes(
        si, q, torch.zeros((1, 2), dtype=torch.float64), z, 200e3, sg_max=0.06, k_max=1.0
    )
    keys = [tuple(h) for h in hkl.tolist()]
    assert (0, 0, 2) in keys or (2, 0, 0) in keys or (0, 2, 0) in keys
    i200 = next(i for i, h in enumerate(keys) if sorted(abs(v) for v in h) == [0, 0, 2])
    assert float(inten[0, 1, i200]) > 1e-4  # populated by multiple scattering
    # mean absorption alone: strip the off-diagonal absorptive part
    U, u0, absorptive = bloch._coupling_matrix(si, hkl, bloch.relativistic_gamma(200e3))
    Uel = 0.5 * (U + U.conj().T)
    lam = bloch.electron_wavelength_angstrom(200e3)
    k0 = 1.0 / lam
    s_t = torch.zeros((1, hkl.shape[0]), dtype=torch.float64)
    gl = bloch.qrotate(q, hkl[1:].to(torch.float64) @ si.lat_recip)
    s_t[0, 1:] = (2 * gl[:, 2] - lam * (gl**2).sum(1)) / (2 - 2 * lam * gl[:, 2])
    inten_np = bloch._bloch_solve(Uel, u0, True, s_t, k0, z, fast_absorption=False)
    total = inten_np[0].sum(dim=1).numpy()
    assert np.allclose(total, np.exp(-2 * np.pi * u0 * z.numpy() / k0), rtol=1e-8)


def test_fourier_ring_matches_quadrature():
    """The harmonic propagation of the centered precession ring reproduces a
    converged azimuthal quadrature, with the full complex coupling."""
    si = _si(absorptive=True)
    q = _zone_110()
    z = torch.tensor([300.0, 600.0])
    trial = torch.zeros((1, 2), dtype=torch.float64)
    beams = bloch.select_dynamical_beams(si, q, 200e3, np.deg2rad(0.4), 0.06, 1.0)
    ring, w = bloch.illumination_nodes(200e3, precession_deg=0.4, n_precession=96)
    inten, g_xy, _ = bloch._cbed_amplitudes(
        si, q, ring, z, 200e3, 0.06, 1.0, tilt_batch=128, beams=beams
    )
    ref = (inten * w[:, None, None]).sum(0)
    got, g2 = bloch.average_bloch_fourier(si, q, trial, z, 200e3, 0.4, 0.06, 1.0, beams=beams)
    assert np.allclose(g2.numpy(), g_xy.numpy())
    assert np.allclose(got[0].numpy(), ref.numpy(), atol=1e-11, rtol=1e-9)
    # a displaced ring through the sampled coefficients
    trial = torch.tensor([[0.15, -0.1]], dtype=torch.float64)
    inten, _, _ = bloch._cbed_amplitudes(
        si, q, ring + trial, z, 200e3, 0.06, 1.0, tilt_batch=128, beams=beams
    )
    ref = (inten * w[:, None, None]).sum(0)
    got, _ = bloch.average_bloch_fourier(
        si, q, trial, z, 200e3, 0.4, 0.06, 1.0, beams=beams, n_geometry=128
    )
    assert np.allclose(got[0].numpy(), ref.numpy(), atol=1e-9, rtol=1e-7)


def test_refine_dynamical_reported_cost_reproducible():
    """The stored cost and thickness belong to the stored orientation."""
    from quantem.core.datastructures.vector import Vector
    from quantem.diffraction.orientation import OrientationMap
    from quantem.diffraction.phase import PhaseMap
    from quantem.diffraction.rotations import qnormalize

    energy_ev = 200e3
    xtl = _si(absorptive=True)
    torch.manual_seed(3)
    q_true = qnormalize(torch.randn(3, 4, dtype=torch.float64))
    peaks = Vector.from_shape(
        (1, 3), fields=["qx", "qy", "intensity"], units=["A^-1"] * 3, name="t"
    )
    for i in range(3):
        inten, g_xy, _ = bloch._cbed_amplitudes(
            xtl,
            q_true[i],
            torch.zeros((1, 2), dtype=torch.float64),
            torch.tensor([450.0]),
            energy_ev,
            0.06,
            1.0,
            progress_bar=False,
        )
        inten_np = inten[0, 0, 1:].numpy()
        keep = inten_np > 1e-3 * inten_np.max()
        peaks[0, i] = np.column_stack([g_xy[1:].numpy()[keep], inten_np[keep]])
    om = OrientationMap.from_vectors(peaks, xtl, energy_ev=energy_ev)
    om.build_plan(angle_step_zone_axis_deg=3.0, verbose=False)
    om.match_orientations(progress_bar=False)
    om.quats[0, :, 0] = q_true
    om.corr[0, :, 0] = 1.0
    pm = PhaseMap.from_orientation_maps([om])
    pm.fit(progress_bar=False)
    res = bloch.refine_dynamical(
        pm,
        thicknesses_A=np.arange(300, 600, 50.0),
        tilt_stages=((0.1, 0.05),),
        sg_max=0.06,
        k_max=1.0,
        progress_bar=False,
    )
    for i in range(3):
        if not torch.isfinite(res["cost"][0, i, 0]) or peaks[0, i].array.shape[0] < 5:
            continue
        q = res["quats"][0, i, 0]
        d3 = torch.eye(3, dtype=torch.float64)
        d3[:2, :2] = res["deformation"][0, i, 0]
        beams = bloch.select_dynamical_beams(
            xtl, res["quats_base"][0, i, 0], energy_ev, np.deg2rad(0.1) * np.sqrt(2), 0.06, 1.0, d3
        )
        inten, g_xy, _ = bloch._cbed_amplitudes(
            xtl,
            q,
            torch.zeros((1, 2), dtype=torch.float64),
            np.arange(300, 600, 50.0),
            energy_ev,
            0.06,
            1.0,
            fast_absorption=False,
            deform=d3,
            beams=beams,
        )
        data = peaks[0, i].array
        qxy = torch.as_tensor(data[:, :2])
        im = torch.as_tensor(data[:, 2]).clamp_min(0) ** 0.25
        cost, _, _, _ = bloch._dynamical_cost(inten[:, :, 1:], g_xy[1:], qxy, im, 0.05, 0.25, 0.02)
        t_idx = int(np.argmin(np.abs(np.arange(300, 600, 50.0) - float(res["thickness"][0, i]))))
        assert np.isclose(float(cost[0, t_idx]), float(res["cost"][0, i, 0]), rtol=1e-6, atol=1e-9)


def test_image_cost_radial_background():
    """A quadratic radial floor is removed by the radial background model
    and biases the constant one."""
    torch.manual_seed(0)
    shape = (48, 48)
    yy, xx = np.mgrid[0:48, 0:48]
    radius = torch.as_tensor(np.hypot(yy - 24.0, xx - 24.0))
    centers = torch.tensor([[24.0, 24.0], [30.0, 35.0], [15.0, 20.0], [36.0, 12.0]])
    inten = torch.tensor([0.8, 0.05, 0.02, 0.01], dtype=torch.float64)
    sim = bloch.render_disks(centers, inten, shape, 3.0, 0.7)
    sim_wrong = bloch.render_disks(
        centers, inten * torch.tensor([1.0, 0.5, 2.0, 1.0]), shape, 3.0, 0.7
    )
    floor = 5.0 + 0.2 * radius - 0.004 * radius**2
    meas = 1000 * sim + floor
    mask = bloch._image_mask(shape, (24.0, 24.0), None, 4.5)
    c_const = bloch._image_cost(meas, torch.stack([sim, sim_wrong]), mask, 0.5, "constant")
    c_rad = bloch._image_cost(meas, torch.stack([sim, sim_wrong]), mask, 0.5, "radial", radius)
    assert float(c_rad[0]) < 1e-12  # exact model with the right background
    assert float(c_const[0]) > 1e-4  # the constant background cannot absorb it
    assert float(c_rad[1]) > float(c_rad[0])


def test_refine_dynamical_with_precession_and_convergence():
    """End-to-end recovery with a precession ring and a convergence disk:
    the ground truth is integrated with denser illumination nodes than
    the model uses."""
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
    torch.manual_seed(5)
    rng = np.random.default_rng(5)
    N = 3
    q_true = qnormalize(torch.randn(N, 4, dtype=torch.float64))
    t_true = torch.tensor([300.0, 450.0, 600.0])
    ring, w = bloch.illumination_nodes(
        energy_ev,
        precession_deg=0.4,
        n_precession=32,
        semiconv_mrad=1.5,
        n_disk_radial=3,
        n_disk_azimuthal=12,
    )
    peaks = Vector.from_shape(
        (1, N), fields=["qx", "qy", "intensity"], units=["A^-1"] * 3, name="t"
    )
    for i in range(N):
        inten, g_xy, _ = bloch._cbed_amplitudes(
            xtl,
            q_true[i],
            ring,
            t_true[i : i + 1],
            energy_ev,
            0.06,
            1.0,
            tilt_batch=256,
            progress_bar=False,
        )
        I_avg = (inten[:, 0, 1:] * w[:, None]).sum(0).numpy()
        keep = I_avg > 1e-3 * I_avg.max()
        peaks[0, i] = np.column_stack([g_xy[1:].numpy()[keep], I_avg[keep]])
    om = OrientationMap.from_vectors(
        peaks, xtl, energy_ev=energy_ev, precession_deg=0.4, semiconv_mrad=1.5
    )
    om.build_plan(angle_step_zone_axis_deg=3.0, verbose=False)
    om.match_orientations(progress_bar=False)
    phis = rng.uniform(0, 2 * np.pi, N)
    om.quats[0, :, 0] = torch.stack(
        [
            qmult(
                quat_from_axis_angle(
                    torch.tensor([np.cos(p), np.sin(p), 0.0], dtype=torch.float64),
                    torch.tensor(np.deg2rad(0.12), dtype=torch.float64),
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
        thicknesses_A=np.arange(200, 700, 25.0),
        tilt_stages=((0.15, 0.05), (0.03, 0.01)),
        n_precession=12,
        n_precession_search=12,
        n_disk_radial=2,
        n_disk_azimuthal=6,
        power_intensity=0.5,
        sg_max=0.06,
        k_max=1.0,
        progress_bar=False,
    )
    assert res["metadata"]["precession_deg"] == 0.4 and res["metadata"]["semiconv_mrad"] == 1.5
    valid = np.array([peaks[0, i].array.shape[0] >= 6 for i in range(N)])
    assert valid.sum() >= 2
    err = misorientation_angle_deg(q_true, om.quats[0, :, 0], xtl.sym_quats).numpy()[valid]
    t_err = np.abs(res["thickness"][0].numpy() - t_true.numpy())[valid]
    assert np.median(err) < 0.03
    assert (t_err <= 25).sum() >= valid.sum() - 1


def _smooth_map_setup(n, tilt_start_deg, corrupt=None):
    """A 1 x n 'map' of one grain: orientations a few hundredths of a degree
    apart, thickness varying slowly, strained cell; starts tilted by
    tilt_start_deg (and one position by `corrupt` degrees)."""
    from quantem.core.datastructures.vector import Vector
    from quantem.diffraction.orientation import OrientationMap
    from quantem.diffraction.phase import PhaseMap
    from quantem.diffraction.rotations import qmult, quat_from_axis_angle

    energy_ev = 200e3
    xtl = _si(absorptive=True)
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    # a well-populated pattern: 1.5 degrees off the [110] zone axis
    base = qmult(
        quat_from_axis_angle(
            torch.tensor([0.6, 0.8, 0.0], dtype=torch.float64),
            torch.tensor(np.deg2rad(1.5), dtype=torch.float64),
        ),
        _zone_110(),
    )
    q_true = torch.stack(
        [
            qmult(
                quat_from_axis_angle(
                    torch.tensor([1.0, 0.3, 0.0], dtype=torch.float64) / np.hypot(1, 0.3),
                    torch.tensor(np.deg2rad(0.03 * i), dtype=torch.float64),
                ),
                base,
            )
            for i in range(n)
        ]
    )
    t_true = torch.tensor([400.0 + 25.0 * i for i in range(n)])
    A_true = torch.tensor([[1.008, 0.002], [0.002, 0.996]], dtype=torch.float64)
    deform3 = torch.eye(3, dtype=torch.float64)
    deform3[:2, :2] = A_true
    peaks = Vector.from_shape(
        (1, n), fields=["qx", "qy", "intensity"], units=["A^-1"] * 3, name="t"
    )
    for i in range(n):
        inten, g_xy, _ = bloch._cbed_amplitudes(
            xtl,
            q_true[i],
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
    om.build_plan(angle_step_zone_axis_deg=3.0, verbose=False)
    om.match_orientations(progress_bar=False)
    phis = rng.uniform(0, 2 * np.pi, n)
    starts = []
    for i, p in enumerate(phis):
        ang = tilt_start_deg if (corrupt is None or i != corrupt[0]) else corrupt[1]
        starts.append(
            qmult(
                quat_from_axis_angle(
                    torch.tensor([np.cos(p), np.sin(p), 0.0], dtype=torch.float64),
                    torch.tensor(np.deg2rad(ang), dtype=torch.float64),
                ),
                q_true[i],
            )
        )
    om.quats[0, :, 0] = torch.stack(starts)
    om.corr[0, :, 0] = 1.0
    pm = PhaseMap.from_orientation_maps([om])
    pm.fit(progress_bar=False)
    return xtl, om, pm, q_true, t_true, A_true


def test_refine_dynamical_warm_start_matches_cold():
    from quantem.diffraction.rotations import misorientation_angle_deg

    n = 5
    kw = dict(
        thicknesses_A=np.arange(300, 700, 25.0),
        tilt_stages=((0.25, 0.05), (0.04, 0.01)),
        power_intensity=0.5,
        sg_max=0.06,
        k_max=1.0,
        neighbor_rescue=False,
        progress_bar=False,
    )
    out = {}
    for warm in (False, True):
        xtl, om, pm, q_true, t_true, A_true = _smooth_map_setup(n, 0.1)
        res = bloch.refine_dynamical(pm, warm_start=warm, **kw)
        out[warm] = (res, om.quats[0, :, 0].clone(), q_true, t_true, xtl)
    res_c, q_c, q_true, t_true, xtl = out[False]
    res_w, q_w, _, _, _ = out[True]
    assert not res_c["warm_started"].any()
    assert res_w["warm_started"][0, 1:, 0].all() and not res_w["warm_started"][0, 0, 0]
    # same solution from both routes, both correct
    d = misorientation_angle_deg(q_c, q_w, xtl.sym_quats).numpy()
    assert d.max() < 0.03
    assert np.allclose(res_c["thickness"][0].numpy(), res_w["thickness"][0].numpy())
    err = misorientation_angle_deg(q_true, q_w, xtl.sym_quats).numpy()
    assert err.max() < 0.03
    assert np.abs(res_w["thickness"][0].numpy() - t_true.numpy()).max() <= 25


def test_refine_dynamical_neighbor_rescue():
    from quantem.diffraction.rotations import misorientation_angle_deg

    n = 5
    # position 2 starts 0.45 degrees off: outside the coarse stage's reach,
    # so its cold search settles in a wrong basin; its neighbors are right
    xtl, om, pm, q_true, t_true, A_true = _smooth_map_setup(n, 0.1, corrupt=(2, 0.45))
    kw = dict(
        thicknesses_A=np.arange(300, 700, 25.0),
        tilt_stages=((0.25, 0.05), (0.04, 0.01)),
        power_intensity=0.5,
        sg_max=0.06,
        k_max=1.0,
        warm_start=False,
        progress_bar=False,
    )
    res = bloch.refine_dynamical(pm, neighbor_rescue=False, **kw)
    err0 = misorientation_angle_deg(q_true, om.quats[0, :, 0], xtl.sym_quats).numpy()
    assert err0[2] > 0.1  # the cold start fails there
    xtl, om, pm, q_true, t_true, A_true = _smooth_map_setup(n, 0.1, corrupt=(2, 0.45))
    res = bloch.refine_dynamical(pm, neighbor_rescue=True, **kw)
    err1 = misorientation_angle_deg(q_true, om.quats[0, :, 0], xtl.sym_quats).numpy()
    assert bool(res["rescued"][0, 2])
    assert err1[2] < 0.03
    assert abs(float(res["thickness"][0, 2]) - float(t_true[2])) <= 25
    # map-level outputs
    maps = bloch.dynamical_maps(res, pm, crystal_index=0)
    assert maps["mask"][0].all()
    assert set(maps["strain"]) == {"aa", "bb", "cc", "ab", "ac", "bc"}
    assert torch.isfinite(maps["gain"][0]).all()
