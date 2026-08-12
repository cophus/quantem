"""
2D pair-angle distribution function for in-plane (non-isotropic) samples.

The standard PADF assumes 3D orientation averaging and expands the
correlation in Legendre polynomials / spherical Bessel functions. For
samples that only rotate in the plane - 2D materials at zone axis, surface
layers, the FFT-of-image test geometry - the correct decomposition is
circular harmonics and Hankel (J_n Bessel) transforms:

    C_n(q, q') = < conj(I_hat_n(q)) I_hat_n(q') >_patterns        (exact,
        read directly from the angular FFT - no basis inversion needed)

    C_n(r, r') = (2 pi)^2 int int  q q' J_n(2 pi q r) J_n(2 pi q' r')
                                   C_n(q, q') dq dq'

    Theta(r, r', dtheta) = sum_n C_n(r, r') e^{i n dtheta}

Compared with pushing 2D data through the 3D machinery this removes the
~2% systematic peak-position bias of the mismatched j_l kernel, needs no
SVD regularization, and has no sin(theta) sampling corrections (the 2D
angular measure is uniform).

Friedel symmetry of single kinematic patterns makes odd n vanish, and the
relative angle folds to [0, 180]. When Friedel is broken (dynamical
scattering from a non-centrosymmetric projection), the odd harmonics are
retained naturally and the full 0-360 degree relative angle is recovered -
the 3D pipeline discards odd terms by construction.

Class: PairAngleDistributionFunction2D
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import jv


class PairAngleDistributionFunction2D:
    """
    2D (in-plane) pair-angle distribution function.

    Parameters
    ----------
    polar : Polar4dstem or torch.Tensor or ndarray
        Polar diffraction data, angular axis second-to-last, radial axis
        last: (..., N_phi, N_q). Leading dims are flattened into patterns.
    n_max : int
        Maximum circular harmonic order retained (angular resolution
        ~180 / n_max degrees).
    dq : float or None
        Radial calibration (q units per bin). Read from the polar dataset's
        sampling when available; required otherwise.
    q_min : float
        q of the first radial bin (from the polar dataset when available).
    r_min, r_max, r_step : float
        Real-space grid, in the reciprocal units of q.
    fq : scalar or (N_q,) array or None
        Atomic form factor f(q) per radial bin; the intensity is divided by
        f(q)^2 (same eq-7 rescaling as the 3D pipeline). Leaving the atom-
        shape damping in shrinks the effective q range and makes the
        real-space maps broad and oscillatory. Floor small values before
        passing to avoid amplifying the high-q noise floor.
    q_taper : float
        Fraction of the radial range cosine-tapered to zero at q_max
        (default 0.25). Softens the hard-cutoff ringing of the Hankel
        transform; 0 disables.
    normalize_mean : bool
        Normalize the intensity to unit mean (magnitude convention shared
        with the 3D PADF pipeline).
    batch_size : int
        Patterns per chunk when accumulating C_n(q, q').
    """

    def __init__(self, polar, n_max: int = 60, dq: float | None = None,
                 q_min: float | None = None, r_min: float = 0.0,
                 r_max: float = 20.0, r_step: float = 0.02,
                 fq=None, q_taper: float = 0.25,
                 normalize_mean: bool = True, batch_size: int = 4096):
        if dq is None and hasattr(polar, "sampling"):
            dq = float(np.asarray(polar.sampling)[3])
        if q_min is None:
            q_min = float(np.asarray(polar.origin)[3]) if hasattr(polar, "origin") else 0.0
        if dq is None:
            raise ValueError("dq must be given when polar carries no calibration.")

        tensor = polar.tensor if hasattr(polar, "tensor") else torch.as_tensor(np.asarray(polar))
        tensor = tensor.to(torch.float64)
        n_phi, n_q = tensor.shape[-2], tensor.shape[-1]
        if n_max >= n_phi // 2:
            n_max = n_phi // 2 - 1
        self.n_max = int(n_max)
        self.dq = float(dq)
        self.q = q_min + np.arange(n_q) * self.dq

        data = tensor.reshape(-1, n_phi, n_q)
        if fq is not None:
            fq_t = torch.as_tensor(np.broadcast_to(np.asarray(fq, dtype=np.float64), (n_q,)).copy())
            data = data / fq_t[None, None, :] ** 2
        if q_taper > 0:
            taper = torch.ones(n_q, dtype=torch.float64)
            i0 = int((1.0 - q_taper) * n_q)
            ramp = np.arange(n_q - i0) / max(n_q - i0, 1)
            taper[i0:] = torch.from_numpy(0.5 * (1 + np.cos(np.pi * ramp)))
            data = data * taper[None, None, :]
        if normalize_mean:
            data = data / data.mean()
        n_patterns = data.shape[0]

        # circular harmonic correlation matrices C_n(q, q'), accumulated
        # directly from the angular FFT (per-radian normalization)
        Cn = torch.zeros((self.n_max + 1, n_q, n_q), dtype=torch.complex128)
        for start in range(0, n_patterns, batch_size):
            F = torch.fft.rfft(data[start:start + batch_size], dim=1) / n_phi
            F = F[:, : self.n_max + 1, :]  # (b, n, q)
            Cn += torch.einsum("bnq,bnp->nqp", torch.conj(F), F)
        self.Cn_q = (Cn / n_patterns).numpy()

        # Hankel transform of order n on both q axes, nondimensionalized by
        # q_max^2 per axis (2D quadrature measure q dq / q_max^2)
        r = np.arange(r_min, r_max, r_step)
        self.r = r
        q_max = self.q[-1]
        w = 2.0 * np.pi * self.q * self.dq / q_max**2  # (Nq,) quadrature weight
        arg = 2.0 * np.pi * np.outer(r, self.q)        # (Nr, Nq)
        self.Cn_r = np.zeros((self.n_max + 1, r.size, r.size), dtype=np.complex128)
        for n in range(self.n_max + 1):
            K = jv(n, arg) * w[None, :]
            self.Cn_r[n] = K @ self.Cn_q[n] @ K.T

    def reconstruct(self, n_theta: int = 360, include_n0: bool = False):
        """
        Reconstruct Theta(r, r', dtheta) on n_theta angles over [0, 360).

        include_n0 : keep the isotropic n = 0 term (a dtheta-independent
        offset containing the uncorrelated background); excluded by
        default, matching the 3D pipeline's l = 0 handling.

        Returns
        -------
        theta_deg : (n_theta,) angles in degrees.
        padf : (Nr, Nr, n_theta) real array.
        """
        theta = 2.0 * np.pi * np.arange(n_theta) / n_theta
        n0 = 0 if include_n0 else 1
        n_vals = np.arange(n0, self.n_max + 1)
        # Hermitian sum: C_{-n} = conj(C_n) -> 2 Re for n > 0
        E = np.exp(1j * np.outer(n_vals, theta))  # (Nn, n_theta)
        padf = 2.0 * np.real(np.einsum("nij,nt->ijt", self.Cn_r[n0:], E))
        if include_n0:
            padf -= np.real(self.Cn_r[0])[:, :, None]  # counted twice above
        self.theta_deg = np.degrees(theta)
        return self.theta_deg, padf

    def plot_diagonal(self, r_max_display: float = 8.0, n_theta: int = 360,
                      theta_max: float = 360.0, r_display_power: int = 1,
                      title: str | None = None,
                      figsize: tuple[float, float] = (5.4, 3.6),
                      returnfig: bool = False):
        """
        Map of the r = r' diagonal vs relative angle, weighted by
        r^r_display_power (r^1 by default), zero-centered RdBu_r.
        theta_max = 180 shows the folded view (sufficient for
        Friedel-symmetric data); 360 shows the full relative angle
        available when odd harmonics are present. Color limits are set
        from interior angles and r <= 0.92 * r_max_display, so edge
        artifacts do not compress the scale.
        """
        theta_deg, padf = self.reconstruct(n_theta=n_theta)
        r = self.r
        rs = r <= r_max_display
        ts = theta_deg <= theta_max + 1e-9
        diag = np.einsum("iik->ik", padf)[rs][:, ts]
        disp = (diag * (r[rs] ** r_display_power)[:, None]).T
        interior = (theta_deg[ts] % 180 > 15) & (theta_deg[ts] % 180 < 165)
        r_inner = r[rs] <= 0.92 * r_max_display
        vmax = np.quantile(np.abs(disp[np.ix_(interior, r_inner)]), 0.999)

        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        im = ax.pcolormesh(r[rs], theta_deg[ts], disp, cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, shading="auto")
        ax.set_xlabel("r = r' (Å)")
        ax.set_ylabel("$\\Delta\\theta$ (deg)")
        ax.set_yticks(np.arange(0, theta_max + 1, 45))
        fig.colorbar(im, ax=ax, label=f"$\\Theta_{{2D}} \\cdot r^{{{r_display_power}}}$")
        if title is not None:
            ax.set_title(title, fontsize=10)
        if returnfig:
            return fig, ax
        plt.show()
