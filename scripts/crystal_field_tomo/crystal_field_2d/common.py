"""Shared conventions for the 2D crystal-field tomography toy.

Geometry
--------
The sample lives in the (x, z) plane; the tilt axis is y (out of plane) and the
beam travels along lab +z. At tilt theta the sample is rotated rigidly:
r_lab = R(theta) r_sample with R = [[cos, sin], [-sin, cos]] acting on (x, z).
The rotation centre is the origin of both frames. Reciprocal vectors rotate the
same way, so g_lab . r_lab = g . r.

Units: Angstrom, radians. A crystal is described by its phase density
rho(r) = sum_j P_j delta(r - r_j) (blurred), in rad / Angstrom: the phase a ray
picks up is the line integral of rho. Its Fourier coefficients rho_g are the
two-beam couplings, extinction distance xi_g = pi / |rho_g|.

Cells
-----
A Friedel pair (g, -g) is filed under one cell (theta_c, u): theta_c in [-90, 90)
is the tilt at which g_lab,z = 0, and u > 0 is g_lab,x of the member pointing
to +x at that tilt. At tilt theta_c + delta the members have
s_plus = u sin(delta) - lam u^2 / 2 and s_minus = -u sin(delta) - lam u^2 / 2.
"""

import numpy as np
import torch


def wavelength(energy_ev: float) -> float:
    """Relativistic electron wavelength in Angstrom."""
    m0c2 = 510998.95
    hc = 12398.419843320026
    return hc / np.sqrt(energy_ev * (2 * m0c2 + energy_ev))


def rot(theta_rad):
    """R(theta) for (x, z) vectors, shape (..., 2, 2)."""
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.stack([np.stack([c, s], -1), np.stack([-s, c], -1)], -2)


def chi(q, lam, defocus, cs):
    """Aberration phase chi(q) = pi lam df q^2 + pi/2 Cs lam^3 q^4."""
    return np.pi * lam * defocus * q**2 + 0.5 * np.pi * cs * lam**3 * q**4


def displacement(u, lam, defocus, cs):
    """Image-plane shift of a beam at spatial frequency u: chi'(u) / 2 pi."""
    return lam * defocus * u + cs * lam**3 * u**3


def crop_intensity(I, n_out):
    """Fourier-crop 1D intensities (..., n_in) to n_out samples on the same field
    of view. The Nyquist bin is dropped so nothing aliases; the mean is kept."""
    n_in = I.shape[-1]
    F = torch.fft.fft(I, dim=-1)
    half = n_out // 2
    out = torch.zeros(I.shape[:-1] + (n_out,), dtype=F.dtype, device=F.device)
    out[..., :half] = F[..., :half]
    out[..., -half + 1 :] = F[..., -half + 1 :]
    return torch.fft.ifft(out, dim=-1).real * (n_out / n_in)


def cell_of(g, lam=None):
    """Canonical cell (theta_c [deg], u, sign) for sample-frame reciprocal
    vectors g (..., 2). sign = +1 if g itself is the +u member, -1 if -g is."""
    g = np.asarray(g, dtype=np.float64)
    th = np.arctan2(g[..., 1], g[..., 0])  # g_lab,z = -sin th gx + cos th gz = 0
    th = (th + np.pi / 2) % np.pi - np.pi / 2
    gx_lab = np.cos(th) * g[..., 0] + np.sin(th) * g[..., 1]
    sign = np.where(gx_lab >= 0, 1, -1)
    return np.degrees(th), np.abs(gx_lab), sign


def phase_reflections(phase, orient_deg, mirror_axis_deg=None, g_min=0.25, g_max=1.4):
    """Reciprocal vectors and couplings rho_g of one grain (sample frame).

    phase: dict(a1, a2, basis=[(frac_xy, P), ...], sigma)
    orient_deg: in-plane rotation applied to the phase's lattice.
    mirror_axis_deg: optional mirror of the rotated lattice across a line
        through the origin at this angle (used to build twins).
    Returns g (n, 2), rho (n,) complex (relative to the lattice origin).
    """
    A = lattice_matrix(phase, orient_deg, mirror_axis_deg)  # columns a1, a2
    B = np.linalg.inv(A).T  # columns b1, b2
    hmax = int(np.ceil(g_max / np.min(np.linalg.norm(B, axis=0)))) + 1
    hk = np.array(
        [(h, k) for h in range(-hmax, hmax + 1) for k in range(-hmax, hmax + 1) if (h, k) != (0, 0)]
    )
    g = hk @ B.T
    gl = np.linalg.norm(g, axis=1)
    keep = (gl > g_min) & (gl < g_max)
    g, gl = g[keep], gl[keep]
    area = abs(np.linalg.det(A))
    rho = np.zeros(len(g), dtype=np.complex128)
    for frac, P in phase["basis"]:
        r = A @ np.asarray(frac, dtype=np.float64)
        rho += P * np.exp(-2j * np.pi * g @ r)
    rho *= np.exp(-2 * np.pi**2 * phase["sigma"] ** 2 * gl**2) / area
    return g, rho


def lattice_matrix(phase, orient_deg, mirror_axis_deg=None):
    a = np.stack([np.asarray(phase["a1"], float), np.asarray(phase["a2"], float)], axis=1)
    t = np.radians(orient_deg)
    Rm = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    A = Rm @ a
    if mirror_axis_deg is not None:
        m = np.radians(mirror_axis_deg)
        d = np.array([np.cos(m), np.sin(m)])
        M = 2 * np.outer(d, d) - np.eye(2)
        A = M @ A
    return A
