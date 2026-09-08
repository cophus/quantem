"""Dynamical (Bloch wave) diffraction for orientation and phase refinement.

Second-pass refinement: kinematical matching fixes orientations from peak
positions (which dynamical scattering does not move), then this module
recomputes peak intensities with multiple scattering to refine specimen
thickness and phase assignment for the top candidates.

Follows the Bloch wave formulation of De Graef (2003), ch. 5. The structure
matrix uses U_g = gamma_rel * F_g / pi with F_g the kinematical structure
factors (scattering amplitude per volume, 1/Angstrom^2), off-diagonals
U_(g-h) and diagonal 2 k0 s_g. Without absorption the matrix is Hermitian,
so one eigendecomposition per orientation gives the diffracted intensities
at every thickness essentially for free:

    psi(t) = C exp(2 pi i gamma t) C^-1 psi_0,   A C = 2 k0 gamma C
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from quantem.core.utils.utils import electron_wavelength_angstrom
from quantem.diffraction.crystal import Crystal
from quantem.diffraction.rotations import qrotate, sample_zone_axes


def relativistic_gamma(energy_ev: float) -> float:
    """Relativistic mass factor 1 + eV / (m0 c^2)."""
    return 1.0 + float(energy_ev) / 510998.95


def _coupling_matrix(
    crystal: Crystal, hkl_beams: torch.Tensor, gamma_rel: float
) -> tuple[torch.Tensor, float, bool]:
    """Off-diagonal Bloch coupling matrix U_(g-h) for a beam list.

    Prefers the absorptive Weickenmeier-Kohl factors when the crystal has
    them (calculate_dynamical_structure_factors); they carry the
    relativistic and 1/pi factors already. Falls back to the kinematical
    (Lobato) factors, purely elastic.

    Returns
    -------
    U : torch.Tensor
        (nb, nb) complex coupling matrix with zero diagonal.
    u0_imag : float
        Imaginary part of U_000 (mean absorption), 0 without absorption.
    absorptive : bool
        Whether absorptive factors were used.
    """
    absorptive = getattr(crystal, "U_dyn", None) is not None
    if absorptive:
        hkl_all = crystal.hkl_dyn
        U_all = crystal.U_dyn
    else:
        hkl_all = crystal.hkl
        U_all = crystal.struct_factors * (gamma_rel / np.pi)
    span = 2 * int(hkl_all.abs().max()) + 1
    key_mult = torch.tensor([1, span, span**2], dtype=torch.long)

    def keys(h):
        return (h * key_mult[None, :]).sum(dim=1)

    lut = {int(k): i for i, k in enumerate(keys(hkl_all))}

    nb = hkl_beams.shape[0]
    diff = hkl_beams[:, None, :] - hkl_beams[None, :, :]  # (nb, nb, 3)
    diff_keys = (diff * key_mult[None, None, :]).sum(dim=-1)
    U = torch.zeros((nb, nb), dtype=torch.complex128)
    idx = torch.tensor(
        [lut.get(int(k), -1) for k in diff_keys.reshape(-1)], dtype=torch.long
    ).reshape(nb, nb)
    has = idx >= 0
    U[has] = U_all[idx[has]]
    U.fill_diagonal_(0)

    u0_imag = 0.0
    if absorptive:
        i0 = lut.get(0, -1)
        if i0 >= 0:
            u0_imag = float(U_all[i0].imag)
    return U, u0_imag, absorptive


def dynamical_pattern(
    crystal: Crystal,
    orientation: torch.Tensor,
    thicknesses_A: torch.Tensor | np.ndarray | float,
    energy_ev: float = 300e3,
    sg_max: float = 0.1,
    k_max: float | None = None,
) -> dict[str, torch.Tensor]:
    """Bloch-wave diffraction intensities for one orientation, all thicknesses.

    Parameters
    ----------
    crystal : Crystal
        With structure factors calculated. For accurate couplings,
        calculate_structure_factors should cover 2x the k_max used here so
        every difference vector g - h has a structure factor.
    orientation : torch.Tensor
        Unit quaternion (4,) rotating crystal vectors into the lab frame.
    thicknesses_A : array-like or float
        Specimen thicknesses in Angstroms.
    sg_max : float, default=0.1
        Excitation error cutoff (1/Angstroms) for including a beam.
    k_max : float | None
        In-plane scattering vector cutoff for included beams.

    Returns
    -------
    dict
        'qx', 'qy' (N,), 'hkl' (N, 3), 'intensity' (T, N) diffracted
        intensities per thickness, 's_g' (N,).
    """
    if crystal.g_vec is None:
        raise RuntimeError("Run crystal.calculate_structure_factors() first.")
    lam = electron_wavelength_angstrom(energy_ev)
    k0 = 1.0 / lam
    gamma_rel = relativistic_gamma(energy_ev)

    t = torch.atleast_1d(torch.as_tensor(thicknesses_A, dtype=torch.float64))

    # beam selection in the lab frame
    g_lab = qrotate(orientation, crystal.g_vec)
    gz, g2 = g_lab[:, 2], (g_lab**2).sum(dim=1)
    s_g = (2 * gz - lam * g2) / (2 - 2 * lam * gz)
    sel = torch.abs(s_g) < sg_max
    if k_max is not None:
        sel &= crystal.g_len <= k_max
    hkl_sel = crystal.hkl[sel]
    g_sel = g_lab[sel]
    s_sel = s_g[sel]
    n = int(sel.sum())

    # beams list includes the (000) beam at index 0
    hkl_beams = torch.cat([torch.zeros((1, 3), dtype=torch.long), hkl_sel])
    s_beams = torch.cat([torch.zeros(1, dtype=torch.float64), s_sel])

    U, u0_imag, absorptive = _coupling_matrix(crystal, hkl_beams, gamma_rel)
    A = U.clone()
    diag = (2 * k0 * s_beams).to(torch.complex128)
    if absorptive:
        # mean absorption: imaginary part of U_000 damps every beam
        diag = diag + 1j * u0_imag
    A += torch.diag(diag)

    if absorptive:
        # non-Hermitian: general eigendecomposition, complex gamma damps
        evals, C = torch.linalg.eig(A)
        gam = evals / (2 * k0)
    else:
        evals, C = torch.linalg.eigh(A)
        gam = (evals.real / (2 * k0)).to(torch.complex128)
    psi0 = torch.linalg.inv(C)[:, 0]  # C^-1 @ e_0
    phase = torch.exp(2j * np.pi * gam[None, :] * t.to(torch.complex128)[:, None])
    psi = torch.einsum("ij,tj,j->ti", C, phase, psi0)  # (T, nb)
    intensity = torch.abs(psi[:, 1:]) ** 2  # drop the (000) beam

    return {
        "qx": g_sel[:, 0],
        "qy": g_sel[:, 1],
        "hkl": hkl_sel,
        "s_g": s_sel,
        "intensity": intensity,
        "thicknesses": t,
    }


def refine_thickness(
    phase_map,
    thicknesses_A: np.ndarray | None = None,
    pair_distance: float = 0.05,
    power_intensity: float = 0.25,
    sg_max: float = 0.1,
    k_max: float | None = None,
    min_number_peaks: int = 3,
    progress_bar: bool = True,
):
    """Second-pass thickness and phase refinement with dynamical intensities.

    For every probe position, the winning candidates of a fitted PhaseMap are
    re-simulated with Bloch waves over a thickness grid. The peak pairing is
    fixed (positions are kinematic); the intensity cost is evaluated for all
    thicknesses from a single eigendecomposition per candidate, and the best
    (thickness, candidate) combination updates the phase decision.

    Parameters
    ----------
    phase_map : PhaseMap
        A fitted PhaseMap (fit() has been run).
    thicknesses_A : np.ndarray | None
        Thickness grid in Angstroms; default 50 ... 1000 in 25 A steps.

    Returns
    -------
    dict
        'thickness' (R, C) best-fit thickness map, 'cost' (R, C, F) dynamical
        costs per candidate at its best thickness, 'phase_index' (R, C)
        updated phase assignment.
    """
    if thicknesses_A is None:
        thicknesses_A = np.arange(50.0, 1000.0, 25.0)
    t_grid = torch.as_tensor(thicknesses_A, dtype=torch.float64)

    oms = phase_map.orientation_maps
    cands = phase_map.candidates
    peaks = oms[0].peaks
    R, C = peaks.shape[0], peaks.shape[1]
    F = len(cands)
    delta = pair_distance

    fields = peaks.fields
    ix = [fields.index(f) for f in ("qx", "qy", "intensity")]

    cost_out = torch.full((R, C, F), torch.nan, dtype=torch.float64)
    thick_out = torch.full((R, C, F), torch.nan, dtype=torch.float64)

    iterator = list(np.ndindex(R, C))
    if progress_bar:
        iterator = tqdm(iterator, desc="dynamical refinement")
    for rx, ry in iterator:
        data = peaks[rx, ry].array
        if data.shape[0] < min_number_peaks:
            continue
        qxy = torch.as_tensor(data[:, ix[:2]], dtype=torch.float64)
        im = torch.as_tensor(data[:, ix[2]], dtype=torch.float64).clamp_min(0)
        im = im**power_intensity
        int_total = float(im.sum())

        for f, (i_om, m) in enumerate(cands):
            om = oms[i_om]
            if om.corr[rx, ry, m] <= 0:
                continue
            # only refine candidates that won weight in the first pass
            if phase_map.phase_weights is not None and float(
                phase_map.phase_weights[rx, ry, f]
            ) <= 0:
                continue
            sim = dynamical_pattern(
                om.crystal,
                om.quats[rx, ry, m],
                t_grid,
                energy_ev=om.energy_ev,
                sg_max=sg_max,
                k_max=k_max,
            )
            sq = torch.stack((sim["qx"], sim["qy"]), dim=1)
            if sq.shape[0] == 0:
                continue
            si = sim["intensity"] ** power_intensity  # (T, N)
            d = torch.cdist(sq, qxy)
            d_min, j_min = d.min(dim=1)
            pair = d_min < delta
            frac = (d_min[pair] / delta).clamp(0, 1)

            a = si[:, pair] * (1 - frac)[None, :]  # (T, P)
            b = im[j_min[pair]][None, :]
            w = (a * b).sum(dim=1) / (a * a).sum(dim=1).clamp_min(1e-12)  # (T,)
            w = w.clamp_min(0)

            c_paired = (
                (b - w[:, None] * a).abs() * (1 - frac)[None, :]
                + w[:, None] * a * frac[None, :]
            ).sum(dim=1)
            c_unpaired_sim = 0.5 * w * si[:, ~pair].sum(dim=1)
            matched = torch.zeros(im.shape[0], dtype=torch.bool)
            matched[j_min[pair]] = True
            c_unpaired_exp = 0.5 * float(im[~matched].sum())
            cost_t = (c_paired + c_unpaired_sim + c_unpaired_exp) / (int_total + 1e-12)

            t_best = int(cost_t.argmin())
            cost_out[rx, ry, f] = cost_t[t_best]
            thick_out[rx, ry, f] = t_grid[t_best]

    # updated per-crystal phase decision from the dynamical costs
    n_maps = len(oms)
    cost_phase = torch.full((R, C, n_maps), torch.inf, dtype=torch.float64)
    for f, (i_om, _) in enumerate(cands):
        c = torch.nan_to_num(cost_out[..., f], nan=torch.inf)
        cost_phase[..., i_om] = torch.minimum(cost_phase[..., i_om], c)
    phase_index = cost_phase.argmin(dim=-1)

    f_best = torch.nan_to_num(cost_out, nan=torch.inf).argmin(dim=-1)
    thickness = torch.gather(thick_out, 2, f_best[..., None]).squeeze(-1)

    return {
        "thickness": thickness,
        "cost": cost_out,
        "phase_index": phase_index,
        "thickness_per_candidate": thick_out,
    }


def _cbed_amplitudes(
    crystal: Crystal,
    orientation: torch.Tensor,
    tilts: torch.Tensor,
    thicknesses_A: torch.Tensor,
    energy_ev: float,
    sg_max: float,
    k_max: float | None,
    tilt_batch: int = 64,
    progress_bar: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bloch intensities of every beam at every incident tilt.

    The coupling matrix is built once; only the diagonal (excitation
    errors) changes with tilt, and the eigendecompositions are batched
    over tilt chunks.

    Parameters
    ----------
    tilts : torch.Tensor
        (M, 2) in-plane incident wavevector components (1/Angstroms).

    Returns
    -------
    intensity : torch.Tensor
        (M, T, nb) beam intensities per tilt and thickness; beam 0 is the
        direct (000) beam.
    g_xy : torch.Tensor
        (nb, 2) in-plane reciprocal vectors of the beams (000 first).
    hkl_beams : torch.Tensor
        (nb, 3) Miller indices of the beams.
    """
    lam = electron_wavelength_angstrom(energy_ev)
    k0 = 1.0 / lam
    gamma_rel = relativistic_gamma(energy_ev)
    t_thick = torch.atleast_1d(
        torch.as_tensor(thicknesses_A, dtype=torch.float64)
    )

    # beam selection: near the Ewald sphere for ANY tilt in the aperture --
    # a tilt t shifts s_g by at most |t| * |g| / k0 to leading order
    g_lab = qrotate(orientation, crystal.g_vec)
    gz, g2 = g_lab[:, 2], (g_lab**2).sum(dim=1)
    s0 = (2 * gz - lam * g2) / (2 - 2 * lam * gz)
    alpha_max = float(torch.linalg.norm(tilts, dim=1).max()) / k0
    sel = torch.abs(s0) < sg_max + alpha_max * crystal.g_len
    if k_max is not None:
        sel &= crystal.g_len <= k_max
    hkl_beams = torch.cat(
        [torch.zeros((1, 3), dtype=torch.long), crystal.hkl[sel]]
    )
    g_beams = torch.cat(
        [torch.zeros((1, 3), dtype=torch.float64), g_lab[sel]]
    )
    nb = hkl_beams.shape[0]

    U, u0_imag, absorptive = _coupling_matrix(crystal, hkl_beams, gamma_rel)

    gx, gy, gzb = g_beams[:, 0], g_beams[:, 1], g_beams[:, 2]
    g2b = (g_beams**2).sum(dim=1)
    M = tilts.shape[0]
    out = torch.zeros((M, t_thick.shape[0], nb), dtype=torch.float64)
    chunks = range(0, M, tilt_batch)
    if progress_bar:
        chunks = tqdm(chunks, desc="Bloch tilts")
    for m0 in chunks:
        m1 = min(m0 + tilt_batch, M)
        tt = tilts[m0:m1]  # (B, 2)
        kz = torch.sqrt(k0**2 - (tt**2).sum(dim=1))  # (B,)
        # s_g for incident k = (tx, ty, -kz), surface normal along z
        num = (
            2 * kz[:, None] * gzb[None, :]
            - 2 * (tt[:, 0, None] * gx[None, :] + tt[:, 1, None] * gy[None, :])
            - g2b[None, :]
        )
        den = 2 * (kz[:, None] - gzb[None, :])
        s_t = num / den  # (B, nb)

        out[m0:m1] = _bloch_solve(U, u0_imag, absorptive, s_t, k0, t_thick)
    return out, g_beams[:, :2], hkl_beams


def _bloch_solve(
    U: torch.Tensor,
    u0_imag: float,
    absorptive: bool,
    s_t: torch.Tensor,
    k0: float,
    t_thick: torch.Tensor,
    perturbative: bool = False,
) -> torch.Tensor:
    """Batched Bloch solve: intensities (B, T, nb) for excitation errors s_t
    (B, nb) with a shared coupling matrix U (nb, nb).

    With perturbative=True the Hermitian part is diagonalized (eigh, much
    faster and better batched than the general complex eig) and the weak
    absorption enters first order: gamma_imag = diag(C^dagger U'' C)/(2 k0).
    Standard for master-pattern computations; the absorptive parts of U are
    a few percent of the elastic parts, so the first-order error is small.
    """
    nb = U.shape[0]
    if absorptive and perturbative:
        H = 0.5 * (U + U.conj().T)
        W = (U - H) / 1j  # Hermitian absorptive part (off-diagonal)
        A = H[None].expand(s_t.shape[0], nb, nb).clone()
        A += torch.diag_embed((2 * k0 * s_t).to(torch.complex128))
        evals, C = torch.linalg.eigh(A)
        gam_r = evals / (2 * k0)  # (B, nb) real
        CW = torch.einsum("bji,jk,bki->bi", C.conj(), W, C).real
        gam_i = (CW + u0_imag) / (2 * k0)  # (B, nb)
        gam = gam_r.to(torch.complex128) + 1j * gam_i
        psi0 = C.conj().transpose(1, 2)[:, :, 0]  # unitary: C^-1 = C^dagger
    else:
        diag = (2 * k0 * s_t).to(torch.complex128)
        if absorptive:
            diag = diag + 1j * u0_imag
        A = U[None].expand(s_t.shape[0], nb, nb).clone()
        A += torch.diag_embed(diag)
        if absorptive:
            evals, C = torch.linalg.eig(A)
            gam = evals / (2 * k0)
        else:
            evals, C = torch.linalg.eigh(A)
            gam = (evals.real / (2 * k0)).to(torch.complex128)
        psi0 = torch.linalg.inv(C)[:, :, 0]  # (B, nb)
    phase = torch.exp(
        2j * np.pi * gam[:, None, :] * t_thick.to(torch.complex128)[None, :, None]
    )  # (B, T, nb)
    psi = torch.einsum("bij,btj,bj->bti", C, phase, psi0)
    return torch.abs(psi) ** 2


def tilt_grid(semiconv_mrad: float, energy_ev: float, n_rings: int = 8):
    """Concentric-ring sampling of the illumination aperture.

    Returns (M, 2) in-plane incident wavevectors (1/Angstroms) covering the
    disk of semiangle `semiconv_mrad`, with approximately uniform density.
    """
    lam = electron_wavelength_angstrom(energy_ev)
    alpha_k = semiconv_mrad * 1e-3 / lam
    pts = [(0.0, 0.0)]
    for r in range(1, n_rings + 1):
        rad = alpha_k * r / n_rings
        n_az = int(np.ceil(2 * np.pi * r))
        th = 2 * np.pi * (np.arange(n_az) + 0.5 * (r % 2)) / n_az
        pts += [(rad * np.cos(a), rad * np.sin(a)) for a in th]
    return torch.tensor(pts, dtype=torch.float64)


def calculate_cbed(
    crystal: Crystal,
    orientation: torch.Tensor,
    thicknesses_A,
    energy_ev: float = 300e3,
    semiconv_mrad: float = 3.0,
    n_rings: int = 8,
    sg_max: float = 0.1,
    k_max: float | None = None,
    pixel_size: float | None = None,
    q_max_plot: float | None = None,
    tilt_batch: int = 64,
) -> dict:
    """Simulate a CBED pattern with Bloch waves.

    Every incident direction inside the aperture is an independent plane
    wave (incoherent illumination): its Bloch intensities are placed at
    g + t in the detector plane, filling each diffraction disk with the
    rocking-curve intensity variation. Uses the absorptive
    Weickenmeier-Kohl structure factors when the crystal carries them.

    Parameters
    ----------
    crystal : Crystal
        With structure factors calculated (cover 2x k_max so every
        difference vector g - h has a coupling).
    orientation : torch.Tensor
        Unit quaternion (4,).
    thicknesses_A : float | array-like
        One or more specimen thicknesses in Angstroms.
    semiconv_mrad : float, default=3.0
        Convergence semiangle. Disks overlap when it exceeds half the
        smallest g spacing times the wavelength.
    n_rings : int, default=8
        Radial sampling rings across the aperture (~200 tilts at 8).
    sg_max : float, default=0.1
        Excitation error cutoff for beam selection (widened automatically
        by the aperture tilt range).
    k_max : float | None
        In-plane cutoff for included beams.
    pixel_size : float | None
        Detector sampling (1/Angstroms per pixel); default disk radius / 12.
    q_max_plot : float | None
        Half-width of the detector; default covers all beams plus a disk.

    Returns
    -------
    dict with 'pattern' ((T, H, W), squeezed to (H, W) for one thickness),
    'sampling' (1/Angstroms per pixel), 'disk_radius' (1/Angstroms),
    'thicknesses', 'hkl', 'g_xy'.
    """
    lam = electron_wavelength_angstrom(energy_ev)
    alpha_k = semiconv_mrad * 1e-3 / lam
    tilts = tilt_grid(semiconv_mrad, energy_ev, n_rings=n_rings)
    t_thick = torch.atleast_1d(torch.as_tensor(thicknesses_A, dtype=torch.float64))

    inten, g_xy, hkl_beams = _cbed_amplitudes(
        crystal, orientation, tilts, t_thick, energy_ev, sg_max, k_max, tilt_batch
    )

    if pixel_size is None:
        pixel_size = alpha_k / 12
    if q_max_plot is None:
        q_max_plot = float(torch.linalg.norm(g_xy, dim=1).max()) + 2 * alpha_k
    half = int(np.ceil(q_max_plot / pixel_size))
    H = 2 * half + 1

    # deposit every (beam, tilt) sample with bilinear weights
    qx = (g_xy[:, 0][None, :] + tilts[:, 0][:, None]).numpy()  # (M, nb)
    qy = (g_xy[:, 1][None, :] + tilts[:, 1][:, None]).numpy()
    fx = qx / pixel_size + half
    fy = qy / pixel_size + half
    ix0 = np.floor(fx).astype(int)
    iy0 = np.floor(fy).astype(int)
    wx = fx - ix0
    wy = fy - iy0

    T = t_thick.shape[0]
    pattern = np.zeros((T, H, H))
    I = inten.numpy()  # (M, T, nb)
    for dx in (0, 1):
        for dy in (0, 1):
            w = (wx if dx else 1 - wx) * (wy if dy else 1 - wy)
            jx = np.clip(ix0 + dx, 0, H - 1)
            jy = np.clip(iy0 + dy, 0, H - 1)
            for ti in range(T):
                np.add.at(pattern[ti], (jx, jy), w * I[:, ti, :])
    pattern /= tilts.shape[0]

    return {
        "pattern": pattern[0] if T == 1 else pattern,
        "sampling": float(pixel_size),
        "disk_radius": float(alpha_k),
        "thicknesses": t_thick.numpy(),
        "hkl": hkl_beams.numpy(),
        "g_xy": g_xy.numpy(),
    }


def calculate_lacbed(
    crystal: Crystal,
    orientation: torch.Tensor,
    thicknesses_A,
    hkl,
    energy_ev: float = 300e3,
    semiconv_mrad: float = 10.0,
    n_pixels: int = 48,
    sg_max: float = 0.1,
    k_max: float | None = None,
    tilt_batch: int = 64,
) -> dict:
    """Large-angle CBED: one reflection's rocking surface over the aperture.

    The intensity of the chosen reflection is mapped over the incident-tilt
    disk on a square grid (parallax / LACBED view of a single disk, without
    the geometric overlap of neighboring disks).

    Returns
    -------
    dict with 'disk' ((T, n, n) squeezed), 'tilt_max' (1/Angstroms),
    'thicknesses'. Pixels outside the aperture are NaN.
    """
    lam = electron_wavelength_angstrom(energy_ev)
    alpha_k = semiconv_mrad * 1e-3 / lam
    ax = torch.linspace(-alpha_k, alpha_k, n_pixels, dtype=torch.float64)
    ty, tx = torch.meshgrid(ax, ax, indexing="ij")
    inside = (tx**2 + ty**2) <= alpha_k**2
    tilts = torch.stack([tx[inside], ty[inside]], dim=1)
    t_thick = torch.atleast_1d(torch.as_tensor(thicknesses_A, dtype=torch.float64))

    inten, _, hkl_beams = _cbed_amplitudes(
        crystal, orientation, tilts, t_thick, energy_ev, sg_max, k_max, tilt_batch
    )
    match = (hkl_beams == torch.as_tensor(hkl, dtype=torch.long)[None, :]).all(dim=1)
    if not bool(match.any()):
        raise ValueError(f"reflection {tuple(hkl)} is not among the excited beams")
    b = int(match.nonzero()[0])

    T = t_thick.shape[0]
    disk = np.full((T, n_pixels, n_pixels), np.nan)
    m = inside.numpy()
    for ti in range(T):
        plane = np.full((n_pixels, n_pixels), np.nan)
        plane[m] = inten[:, ti, b].numpy()
        disk[ti] = plane
    return {
        "disk": disk[0] if T == 1 else disk,
        "tilt_max": float(alpha_k),
        "thicknesses": t_thick.numpy(),
    }


def calculate_cbed_library(
    crystal: Crystal,
    orientations: torch.Tensor,
    thickness_A: float,
    energy_ev: float = 300e3,
    semiconv_mrad: float = 3.0,
    k_max: float | None = None,
    q_max_plot: float | None = None,
    pixel_size: float | None = None,
    progress_bar: bool = True,
    **kwargs,
) -> dict:
    """A stack of simulated CBED patterns on one common detector grid.

    The starting point for CBED orientation matching: all patterns share
    the same sampling and extent, ready for polar transformation and
    correlation. One entry per orientation.

    Returns
    -------
    dict with 'patterns' (N, H, W), 'quats' (N, 4), 'sampling',
    'disk_radius', 'thickness_A'.
    """
    lam = electron_wavelength_angstrom(energy_ev)
    alpha_k = semiconv_mrad * 1e-3 / lam
    if pixel_size is None:
        pixel_size = alpha_k / 12
    if q_max_plot is None:
        base = k_max if k_max is not None else float(crystal.k_max) / 2
        q_max_plot = base + 2 * alpha_k

    quats = torch.atleast_2d(torch.as_tensor(orientations, dtype=torch.float64))
    pats = []
    it = range(quats.shape[0])
    if progress_bar:
        it = tqdm(it, desc="CBED library")
    for i in it:
        res = calculate_cbed(
            crystal,
            quats[i],
            thickness_A,
            energy_ev=energy_ev,
            semiconv_mrad=semiconv_mrad,
            k_max=k_max,
            pixel_size=pixel_size,
            q_max_plot=q_max_plot,
            **kwargs,
        )
        pats.append(res["pattern"])
    return {
        "patterns": np.stack(pats),
        "quats": quats.numpy(),
        "sampling": float(pixel_size),
        "disk_radius": float(alpha_k),
        "thickness_A": float(thickness_A),
    }


def calculate_kossel(
    crystal: Crystal,
    orientation: torch.Tensor,
    thicknesses_A,
    energy_ev: float = 300e3,
    semiconv_mrad: float = 40.0,
    n_pixels: int = 192,
    sg_max: float = 0.05,
    k_max: float | None = None,
    tilt_batch: int = 64,
    progress_bar: bool = True,
) -> dict:
    """Wide-angle convergent beam (Kossel) pattern with Bloch waves.

    At convergence angles far beyond the Bragg angles the diffraction disks
    overlap completely and the pattern becomes a continuous map of
    deficiency and excess lines (the Kossel regime of CBED; the bright
    field disk alone is the LACBED view). One Bloch computation over the
    incident-tilt grid yields both:

    - 'bright_field': the (000) beam intensity at each incident tilt --
      the deficiency (dark) line system, every line at a Bragg condition.
    - 'pattern': the full detector intensity, the incoherent sum of every
      diffracted cone shifted by its g -- deficiency lines from the direct
      beam plus the excess (bright) lines of the diffracted beams.

    Line positions are exact; line profiles carry the many-beam dynamical
    structure, with the deficiency/excess asymmetry from the absorptive
    structure factors when the crystal has them.

    Parameters
    ----------
    crystal : Crystal
        With structure factors calculated (cover 2x k_max for couplings),
        and ideally calculate_dynamical_structure_factors for absorption.
    orientation : torch.Tensor
        Unit quaternion (4,).
    thicknesses_A : float | array-like
        One or more thicknesses in Angstroms.
    semiconv_mrad : float, default=40.0
        Convergence semiangle; the pattern covers this angular radius.
    n_pixels : int, default=192
        Detector pixels across the pattern (also the tilt sampling; the
        1-2 mrad dynamical line widths need ~0.5 mrad per pixel).
    sg_max : float, default=0.05
        Excitation error cutoff; the beam list is widened by the aperture
        automatically.
    k_max : float | None
        In-plane cutoff for included reflections.

    Returns
    -------
    dict with 'bright_field' and 'pattern' ((T, n, n), squeezed for one
    thickness; NaN / 0 outside the aperture), 'sampling' (1/Angstroms per
    pixel), 'mrad_per_pixel', 'thicknesses', 'hkl'.
    """
    lam = electron_wavelength_angstrom(energy_ev)
    alpha_k = semiconv_mrad * 1e-3 / lam
    ax = torch.linspace(-alpha_k, alpha_k, n_pixels, dtype=torch.float64)
    px = float(ax[1] - ax[0])
    ty, tx = torch.meshgrid(ax, ax, indexing="ij")
    inside = (tx**2 + ty**2) <= alpha_k**2
    tilts = torch.stack([tx[inside], ty[inside]], dim=1)
    t_thick = torch.atleast_1d(torch.as_tensor(thicknesses_A, dtype=torch.float64))
    T = t_thick.shape[0]

    inten, g_xy, hkl_beams = _cbed_amplitudes(
        crystal, orientation, tilts, t_thick, energy_ev, sg_max, k_max,
        tilt_batch, progress_bar=progress_bar,
    )
    I = inten.numpy()  # (M, T, nb)
    m = inside.numpy()

    # bright field: beam 0 on the tilt grid directly
    bright = np.full((T, n_pixels, n_pixels), np.nan)
    for ti in range(T):
        plane = np.full((n_pixels, n_pixels), np.nan)
        plane[m] = I[:, ti, 0]
        bright[ti] = plane

    # full pattern: every diffracted cone shifted by its g, bilinear deposit
    pattern = np.zeros((T, n_pixels, n_pixels))
    tx_in = tilts[:, 0].numpy()
    ty_in = tilts[:, 1].numpy()
    g_np = g_xy.numpy()
    for b in range(g_np.shape[0]):
        fx = (tx_in + g_np[b, 0] + alpha_k) / px
        fy = (ty_in + g_np[b, 1] + alpha_k) / px
        ix0 = np.floor(fx).astype(int)
        iy0 = np.floor(fy).astype(int)
        wx = fx - ix0
        wy = fy - iy0
        for dx in (0, 1):
            for dy in (0, 1):
                jx = ix0 + dx
                jy = iy0 + dy
                ok = (jx >= 0) & (jx < n_pixels) & (jy >= 0) & (jy < n_pixels)
                w = (wx if dx else 1 - wx) * (wy if dy else 1 - wy)
                for ti in range(T):
                    np.add.at(
                        pattern[ti], (jx[ok], jy[ok]), (w * I[:, ti, b])[ok]
                    )
    pattern[:, ~m] = 0.0

    return {
        "bright_field": bright[0] if T == 1 else bright,
        "pattern": pattern[0] if T == 1 else pattern,
        "sampling": px,
        "mrad_per_pixel": px * lam * 1e3,
        "thicknesses": t_thick.numpy(),
        "hkl": hkl_beams.numpy(),
    }


def calculate_kossel_master(
    crystal: Crystal,
    thicknesses_A,
    energy_ev: float = 300e3,
    angle_step_mrad: float = 1.0,
    sg_max: float = 0.05,
    k_max: float | None = None,
    theta_max_deg: float = 90.0,
    chunk: int = 256,
    perturbative: bool = True,
    progress_bar: bool = True,
) -> dict:
    """Kossel master pattern: the dynamical bright field over all directions.

    The bright field intensity depends only on the incident beam direction
    in the CRYSTAL frame (each incident plane wave is independent), so one
    Bloch computation over the symmetry-reduced direction wedge gives the
    pattern for every specimen orientation at once -- the EMsoft master
    pattern strategy. Patterns for arbitrary orientations, convergence
    angles, and all precomputed thicknesses are then interpolation lookups
    via kossel_from_master(), microseconds instead of a fresh dynamical
    calculation.

    The wedge samples are expanded by the crystal's proper rotations plus
    inversion and rasterized onto a Lambert azimuthal equal-area grid of
    the upper hemisphere. (The inversion expansion assumes Friedel symmetry
    of the bright field; for non-centrosymmetric crystals with absorption
    this neglects a small polarity contrast.)

    Parameters
    ----------
    crystal : Crystal
        With structure factors calculated (cover 2x k_max), and ideally
        calculate_dynamical_structure_factors for absorption.
    thicknesses_A : float | array-like
        Thickness grid; all thicknesses share the eigendecompositions, so a
        thickness AXIS is nearly free -- precompute the matching range here.
    angle_step_mrad : float, default=1.0
        Angular sampling of the wedge. The dynamical line widths are
        1-2 mrad; 0.5 for production masters, 1-2 for quick looks.
    theta_max_deg : float, default=90.0
        Polar cutoff of the WEDGE samples. Keep at 90 unless the wedge's
        far corners are never observed: cutting the wedge leaves coverage
        holes at all their symmetry equivalents.

    Returns
    -------
    dict with 'lambert' (T, n, n) master on the equal-area grid (NaN where
    unsampled), 'rho_max', 'thicknesses', 'energy_ev', and the raw wedge
    'directions' / 'intensity'.
    """
    lam = electron_wavelength_angstrom(energy_ev)
    k0 = 1.0 / lam
    gamma_rel = relativistic_gamma(energy_ev)
    t_thick = torch.atleast_1d(torch.as_tensor(thicknesses_A, dtype=torch.float64))
    T = t_thick.shape[0]

    wedge = crystal.zone_axis_wedge()
    step_deg = np.rad2deg(angle_step_mrad * 1e-3)
    if wedge is None:
        n_dirs = int(np.ceil(2 * np.pi / np.deg2rad(step_deg) ** 2))
        dirs = fibonacci_hemisphere(n_dirs)
    else:
        dirs, _ = sample_zone_axes(wedge, step_deg)
    keep = dirs[:, 2] >= np.cos(np.deg2rad(theta_max_deg))
    dirs = dirs[keep]
    N = dirs.shape[0]

    g = crystal.g_vec  # crystal frame, orientation is identity
    gz, g2 = g[:, 2], (g**2).sum(dim=1)
    g_len = crystal.g_len

    out = torch.zeros((N, T), dtype=torch.float64)
    chunks = range(0, N, chunk)
    if progress_bar:
        chunks = tqdm(chunks, desc="Kossel master")
    for c0 in chunks:
        c1 = min(c0 + chunk, N)
        d = dirs[c0:c1]  # (B, 3) beam directions in the crystal frame
        d_c = d.mean(dim=0)
        d_c = d_c / torch.linalg.norm(d_c)
        radius = float(torch.arccos((d @ d_c).clamp(-1, 1)).max())

        # normal-tracking geometry: each sample is computed with the foil
        # normal along the sampled direction (the crystal is conceptually
        # re-tilted per sample). The slab problem is then a function of the
        # crystal-frame beam direction ALONE, which is what makes the
        # symmetry expansion below exact; a pattern lookup only ever probes
        # directions within the convergence semiangle of the true normal,
        # so the approximation error is O(alpha^2).
        u_c = g @ d_c
        s_c = (2 * k0 * u_c - g2) / (2 * (k0 - u_c))
        sel = torch.abs(s_c) < sg_max + (radius + 1e-4) * g_len
        if k_max is not None:
            sel &= g_len <= k_max
        hkl_beams = torch.cat(
            [torch.zeros((1, 3), dtype=torch.long), crystal.hkl[sel]]
        )
        g_b = torch.cat([torch.zeros((1, 3), dtype=torch.float64), g[sel]])
        U, u0_imag, absorptive = _coupling_matrix(crystal, hkl_beams, gamma_rel)

        g2b = (g_b**2).sum(dim=1)
        u = torch.einsum("bk,nk->bn", d, g_b)  # g . d_hat per sample
        s_t = (2 * k0 * u - g2b[None, :]) / (2 * (k0 - u))
        I = _bloch_solve(
            U, u0_imag, absorptive, s_t, k0, t_thick, perturbative=perturbative
        )
        out[c0:c1] = I[:, :, 0]

    # symmetry expansion and Lambert raster: with normal-tracking geometry
    # the intensity is a function of the crystal-frame beam direction only,
    # so proper rotations apply directly; the reversed beam (with reversed
    # normal) gives the same bright field by reciprocity.
    from quantem.diffraction.rotations import quat_to_matrix

    Rs = quat_to_matrix(crystal.sym_quats_matching)
    d_all = torch.einsum("sij,nj->sni", Rs, dirs).reshape(-1, 3)
    I_all = out[None, :, :].expand(Rs.shape[0], -1, -1).reshape(-1, T)
    d_all = torch.cat([d_all, -d_all])
    I_all = torch.cat([I_all, I_all])
    up = d_all[:, 2] >= 0
    d_all, I_all = d_all[up], I_all[up]

    # grid always spans the full hemisphere: symmetry expansion moves wedge
    # samples to any polar angle, and clipping them onto a smaller rim
    # corrupts the equatorial region
    rho_max = float(np.sqrt(2.0))
    step = angle_step_mrad * 1e-3
    half = int(np.ceil(rho_max / step))
    n = 2 * half + 1
    rho = torch.sqrt((2 * (1 - d_all[:, 2])).clamp_min(0))
    dxy = torch.linalg.norm(d_all[:, :2], dim=1).clamp_min(1e-12)
    px_x = (d_all[:, 0] / dxy * rho / step + half).numpy()
    px_y = (d_all[:, 1] / dxy * rho / step + half).numpy()

    acc = np.zeros((T, n, n))
    wgt = np.zeros((n, n))
    ix0 = np.floor(px_x).astype(int)
    iy0 = np.floor(px_y).astype(int)
    wx = px_x - ix0
    wy = px_y - iy0
    I_np = I_all.numpy()
    for dx in (0, 1):
        for dy in (0, 1):
            jx = np.clip(ix0 + dx, 0, n - 1)
            jy = np.clip(iy0 + dy, 0, n - 1)
            w = (wx if dx else 1 - wx) * (wy if dy else 1 - wy)
            np.add.at(wgt, (jx, jy), w)
            for ti in range(T):
                np.add.at(acc[ti], (jx, jy), w * I_np[:, ti])
    lambert = np.where(wgt[None] > 1e-6, acc / np.maximum(wgt[None], 1e-6), np.nan)

    # fill raster holes (unhit pixels between splatted samples) from their
    # neighbors so bilinear lookups never touch NaN inside the disk
    yy, xx = np.mgrid[0:n, 0:n]
    in_disk = ((xx - half) ** 2 + (yy - half) ** 2) <= (rho_max / step) ** 2
    for ti in range(lambert.shape[0]):
        L = lambert[ti]
        for _ in range(4):
            holes = np.isnan(L) & in_disk
            if not holes.any():
                break
            Lp = np.pad(L, 1, constant_values=np.nan)
            stack = np.stack(
                [Lp[1 + dy : n + 1 + dy, 1 + dx : n + 1 + dx]
                 for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
            )
            with np.errstate(all="ignore"):
                fill = np.nanmean(stack, axis=0)
            L[holes] = fill[holes]
        lambert[ti] = L

    return {
        "lambert": lambert,
        "rho_max": rho_max,
        "step": step,
        "thicknesses": t_thick.numpy(),
        "energy_ev": float(energy_ev),
        "directions": dirs.numpy(),
        "intensity": out.numpy(),
    }


def kossel_from_master(
    master: dict,
    orientation: torch.Tensor,
    semiconv_mrad: float = 40.0,
    n_pixels: int = 192,
) -> dict:
    """Extract a bright field Kossel pattern from a master pattern.

    Interpolation only -- microseconds per pattern per thickness. The
    detector tilt grid is mapped into the crystal frame by the orientation
    and looked up on the master's Lambert grid.

    Returns
    -------
    dict with 'bright_field' ((T, n, n), squeezed), 'mrad_per_pixel',
    'thicknesses'.
    """
    lam = electron_wavelength_angstrom(master["energy_ev"])
    k0 = 1.0 / lam
    alpha_k = semiconv_mrad * 1e-3 / lam
    ax = torch.linspace(-alpha_k, alpha_k, n_pixels, dtype=torch.float64)
    ty, tx = torch.meshgrid(ax, ax, indexing="ij")
    inside = (tx**2 + ty**2) <= alpha_k**2
    tz = torch.sqrt((k0**2 - tx**2 - ty**2).clamp_min(0))
    d_lab = torch.stack([tx, ty, tz], dim=-1) / k0  # incident directions

    from quantem.diffraction.rotations import quat_to_matrix

    R = quat_to_matrix(
        torch.atleast_2d(torch.as_tensor(orientation, dtype=torch.float64))[0]
    ).to(torch.float64)
    d_c = torch.einsum("ji,rcj->rci", R, d_lab)  # crystal frame, R^T d
    d_c = torch.where(d_c[..., 2:3] < 0, -d_c, d_c)  # reciprocity fold

    step = master["step"]
    lambert = master["lambert"]
    half = (lambert.shape[-1] - 1) // 2
    rho = torch.sqrt((2 * (1 - d_c[..., 2])).clamp_min(0))
    dxy = torch.linalg.norm(d_c[..., :2], dim=-1).clamp_min(1e-12)
    fx = (d_c[..., 0] / dxy * rho / step + half).numpy()
    fy = (d_c[..., 1] / dxy * rho / step + half).numpy()

    T = lambert.shape[0]
    n_l = lambert.shape[-1]
    ix0 = np.clip(np.floor(fx).astype(int), 0, n_l - 2)
    iy0 = np.clip(np.floor(fy).astype(int), 0, n_l - 2)
    wx = np.clip(fx - ix0, 0, 1)
    wy = np.clip(fy - iy0, 0, 1)
    bf = np.full((T, n_pixels, n_pixels), np.nan)
    m = inside.numpy()
    for ti in range(T):
        L = lambert[ti]
        val = (
            L[ix0, iy0] * (1 - wx) * (1 - wy)
            + L[ix0 + 1, iy0] * wx * (1 - wy)
            + L[ix0, iy0 + 1] * (1 - wx) * wy
            + L[ix0 + 1, iy0 + 1] * wx * wy
        )
        val[~m] = np.nan
        bf[ti] = val

    return {
        "bright_field": bf[0] if T == 1 else bf,
        "mrad_per_pixel": 2 * semiconv_mrad / n_pixels,
        "thicknesses": master["thicknesses"],
    }
