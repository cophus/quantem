import matplotlib.pyplot as plt
import torch
import numpy as np
from numpy.typing import NDArray
from scipy.special import eval_legendre, spherical_jn
from numpy.polynomial.legendre import legval

from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.datastructures.polar4dstem import Polar4dstem
from quantem.core.io.serialize import AutoSerialize
from quantem.core.io.serialize import load
from quantem.diffraction.polar_transform import (
    as_dataset4dstem,
    find_origin_angular_grid,
    polar_transform,
)

class PairAngleDistributionFunction(AutoSerialize):
    """
    Compute the pair-angle distribution function from diffraction data:
    a single 2D pattern (Dataset2d), a 3D stack of patterns, or a 4D-STEM
    dataset. Non-4D inputs are wrapped as (1, 1, H, W) / (N, 1, H, W).

    Run the following pipeline:
    padf = PairAngleDistributionFunction(ds)
    1. find_origin_angular_grid(ds)
    2. polar_transform(ds, origin_array=self.origin)
    3. rescale_intensity(self.polar, rho=1, fq=1)
    4. compute_avg_angular_correlation(self.polar_rescaled)
    5. correct_angular_correlation(self.ang_corr)  # theta-mean subtraction + |sin(theta)| factor
    6. extract_Bl_matrices(self.ang_corr_corrected)
    7. transform_to_real_space(self.Bl_mats[0], self.Bl_mats[1])
    8. reconstruct_PADF(self.real_Bl_mats, self.Bl_mats[1], self.ds.shape[0] * self.ds.shape[1])

    """
    def __init__(self,
                 ds=None,
                 origin: NDArray | None = None,
                 polar_ds: Polar4dstem | None = None,
                 ang_corr: torch.Tensor | None = None,
                 Bl_mats = None,
                 rho: float = 1.0,
                 fq: NDArray | torch.Tensor | float | None = None,
                 ):
        super().__init__()

        # accept a single 2D pattern, a 3D stack, or a 4D-STEM dataset
        self.ds = as_dataset4dstem(ds) if ds is not None else None
        self.origin = origin if origin is not None else find_origin_angular_grid(ds)
        self.polar = polar_ds if polar_ds is not None else polar_transform(ds, origin_array=self.origin)

        # TODO: look up rho and fq from sample identity when not provided
        self.polar_rescaled = self.rescale_intensity(self.polar, rho=rho, fq=fq)

        self.ang_corr = ang_corr if ang_corr is not None else self.compute_avg_angular_correlation(self.polar_rescaled)
        self.ang_corr_corrected = self.correct_angular_correlation(self.ang_corr)
        self.Bl_mats = Bl_mats if Bl_mats is not None else self.extract_Bl_matrices(self.ang_corr_corrected)
        # radial q calibration of the polar dataset: bin i sits at q_min + i * dq
        dq = float(np.asarray(self.polar.sampling)[3])
        q_min = float(np.asarray(self.polar.origin)[3])
        self.real_Bl_mats = self.transform_to_real_space(self.Bl_mats[0], self.Bl_mats[1], dq=dq, q_min=q_min)
        self.padf = self.reconstruct_PADF(self.real_Bl_mats, self.Bl_mats[1], 20) # TODO: determine atoms in beam
    
    def rescale_intensity(self,
                          data: Polar4dstem | Dataset4dstem | None = None,
                          rho: float | int = 1.0,
                          fq: NDArray | torch.Tensor | float | int | None = None):
        """
        Equation 7: Take the raw intensity and divide by the following factors
        - Mean number density (rho)
        - Mean atomic scattering factor squared (f(q)^2)
        - Number of atoms in the beam (N_a)
        - phi_0 (dependent on experimental parameters)

        fq may be a scalar or a per-radial-bin array f(q) of length Nq
        (broadcast along the last axis of the polar data). Dividing by
        f(q)^2 removes the atom-shape damping of the diffraction rings;
        leaving it in biases the real-space peak positions of the PADF.
        Floor small f(q) values before passing to avoid amplifying the
        high-q noise floor.

        Finally the intensity is normalized to unit mean. This is the
        analogue of pypadf's scalar 1/beamnorm^2 correlation scaling: it
        absorbs the unmeasurable overall factors (phi_0, N_a, detector
        gain) so the PADF magnitude is independent of the raw intensity
        scale.

        Handles dtype conversion to float64
        """

        intensity = data.tensor.to(dtype=torch.float64)
        if fq is None:
            fq2 = 1.0
        elif isinstance(fq, (int, float)):
            fq2 = float(fq) ** 2
        else:
            fq_t = torch.as_tensor(np.asarray(fq), dtype=torch.float64)
            if fq_t.shape[-1] != intensity.shape[-1]:
                raise ValueError(
                    f"fq has length {fq_t.shape[-1]} but the polar data has "
                    f"{intensity.shape[-1]} radial bins."
                )
            fq2 = fq_t ** 2  # broadcasts along the radial (last) axis
        rescaled_intensity = intensity / (rho * fq2)
        rescaled_intensity /= rescaled_intensity.mean()
        return rescaled_intensity
    
    # phi_0 is not measureable, make a function to fit. (edit: Maybe not necessary)
    def fit_phi_0(self):
        pass
        

    def compute_avg_angular_correlation(self, data):
        """
        Equation 8: angular cross correlation implemented via Fourier correlation theorem
        """

        scan_row, scan_col, phi, r = data.shape
        g_all = data.reshape(scan_row * scan_col, phi, r) # Flatten grid of dps to one dimension, makes for loop simpler
        N_alpha = g_all.shape[0] # Number of dps we have = number of configurations (N_alpha)

        dphi = 2.0 * torch.pi / phi # A small change in angle

        G = torch.fft.rfft(g_all, dim=1) # Fourier transform along phi axis
        P = torch.einsum('afi,afj->fij', G.conj(), G) # Outer product of r between G and its conjugate, summed over all N_alpha
        C_sum = torch.fft.irfft(P, n=phi, dim=0)

        C_avg = (C_sum / N_alpha) * dphi
        return C_avg

    def correct_angular_correlation(self,
                                    C_avg: torch.Tensor,
                                    subtract_mean: bool = True,
                                    sintheta: bool = True):
        """
        Corrections applied to the angular correlation before B_l extraction,
        matching the pypadf maskcorr.py defaults:

        - subtract_mean: subtract the theta-average from each (q, q') ring,
          removing the dominant isotropic (uncorrelated) background.
        - sintheta: multiply by |sin(theta)| (evaluated at bin centers).
          This is the conversion factor between the correlation function and
          the PADF (eq 11), applied in q-space as in the pypadf workflow.
        """
        C = C_avg.clone()
        Nphi = C.shape[0]
        if subtract_mean:
            C -= C.mean(dim=0, keepdim=True)
        if sintheta:
            theta = 2.0 * torch.pi * (torch.arange(Nphi, dtype=C.dtype) + 0.5) / Nphi
            C *= torch.abs(torch.sin(theta))[:, None, None]
        return C

    def extract_Bl_matrices(self, C_avg, l_max=40, sv_cutoff=0.05):
        # Equation 9, 10, 11
        # for each l:
        #     build the linear system relating C to B_l(q, q')
        #     solve it by SVD
        # Ignore odd l-terms, Friedel symmetry?
        # 5% SVD cutoff
        Nphi, Nq, Nqp = C_avg.shape
        dphi = torch.linspace(0, 2 * torch.pi, Nphi + 1, dtype=torch.float64)[:-1] # Ranges from [0, 2pi)
        l_values = torch.arange(0, l_max, 2) # Even only
        Nl = len(l_values)
        cos_dphi = torch.cos(dphi)

        C_flat = C_avg.reshape(Nphi, Nq * Nqp)

        # Build matrix
        L = l_values.reshape(1, Nl)
        X = cos_dphi.reshape(Nphi, 1)
        leg_matrix = eval_legendre(L, X) # Shape due to broadcasting (Nphi, Nl)

        U, S, Vh = torch.linalg.svd(leg_matrix, full_matrices=False)
        S_max = S.max()
        S_inv = torch.where(S > sv_cutoff * S_max, 1.0 / S, torch.zeros_like(S))

        tmp = U.T @ C_flat
        tmp = S_inv.unsqueeze(1) * tmp
        B_flat = Vh.T @ tmp

        return B_flat.reshape(Nl, Nq, Nqp), l_values

        # TODO: Compare each function to the martin code to see the difference/similarity or understand the repo better

    def transform_to_real_space(self, Bl_mats, l_values, dq=0.01, q_min=0.0,
                                r_min=0.0, r_max=20.0, r_step=0.02):
        """
        Transform B_l(q, q') to B_l(r, r') by applying the spherical Bessel
        transform (eq 8) along each q axis via direct quadrature.

        dq : radial q step of the polar dataset (e.g. angstrom^-1 / bin).
        q_min : q value of the first radial bin (nonzero if the polar
            transform used radial_min > 0).
        r_min, r_max, r_step : real-space grid, in the reciprocal units of q.

        The transform kernel is nondimensionalized by q_max (the quadrature
        uses (q / q_max)^2 d(q / q_max) per axis), so the output B_l(r, r')
        is dimensionless rather than carrying units of q^6. Combined with
        the unit-mean intensity normalization in rescale_intensity, this
        keeps PADF values of order 1.
        """
        # Equation 12, 13
        # apply bessel transform twice for each l
        # → shape = (l, r, r')

        # Step one is to define q
        q = q_min + torch.arange(0, Bl_mats.shape[1], dtype=torch.float64) * dq
        q_max = float(q[-1])
        r = torch.arange(r_min, r_max, r_step, dtype=torch.float64)
        self.r = r.numpy()
        real_Bl = torch.zeros((Bl_mats.shape[0], r.shape[0], r.shape[0]), dtype=torch.float64)

        for l in range(len(l_values)):
            Bl = Bl_mats[l].to(dtype=torch.float64) # The corresponding q x q' matrix
            arg = 2 * torch.pi * torch.outer(r, q) # Shape len(r) x len(q)
            jl = torch.from_numpy(spherical_jn(l_values[l], arg)).to(dtype=torch.float64)

            # Representing DSBT (eq 12) as a transformation matrix "sbessel",
            # with q scaled by q_max so the quadrature is dimensionless
            sbessel = 4 * torch.pi * jl * (q**2) * dq / q_max**3
            # Applying it twice (once along each axis)
            real_Bl[l] = sbessel @ Bl @ sbessel.T * (-1)**l_values[l]
        
        return real_Bl

    def reconstruct_PADF(self, real_Bl, l_values, Na):
        """
        Reconstruct the pair-angle distribution function from B_l(r, r')
        For each theta:
        - Sum Pl(cos theta) times B_l() over all l
        - Multiply by n_alpha * 2 pi

        NOTE: Theta will go from 0 to pi
        EDIT: Na should be number of atoms not number of dps
        """
        padf = torch.zeros((real_Bl.shape[1], real_Bl.shape[2], 180), dtype=torch.float64)
        theta = np.linspace(0, np.pi, 180)
        self.theta = theta
        self.theta_deg = np.degrees(theta)
        cos_theta = np.cos(theta)
        chosen_l_values = l_values[1:]

        # Sum over all l
        for l in range(len(chosen_l_values)): # skipping l = 0
            coeffs = np.zeros(chosen_l_values[-1] + 1)
            coeffs[chosen_l_values[l]] = 1
            Pl = torch.tensor(legval(cos_theta, coeffs))
            Bl = real_Bl[l + 1] # skipping l = 0
            padf += Bl[:, :, np.newaxis] * Pl[np.newaxis, np.newaxis, :]

        padf *= 2 * torch.pi * Na
        return padf

    def plot_g2_g3(
        self,
        r_min: float | None = None,
        r_max: float | None = None,
        r_display_power: int = 1,
        r_max_display: float | None = None,
        r_search_min: float = 0.5,
        r_marks=None,
        markers=None,
        title: str | None = None,
        figsize: tuple[float, float] = (4.8, 4.4),
        returnfig: bool = False,
    ):
        """
        Stacked 2-body / 3-body correlation summary of the PADF, after
        atomode's G3 explorer. The two panels share the radial axis.

        Top panel: 3-body correlation map - Theta(r, r', theta) integrated
        over r in [r_min, r_max] (symmetrized over both radial axes),
        plotted as angle theta vs r'.

        Bottom panel: 2-body correlation profile - the |sin(theta)|-weighted
        RMS of the r = r' diagonal over the interior angular range
        (15-165 deg),
            sqrt( sum_theta Theta(r, r, theta)^2 sin(theta)
                  / sum_theta sin(theta) ).
        This stands in for the isotropic pair correlation (whose true l = 0
        component is removed by the correlation mean subtraction; a plain
        sin-weighted integral of Theta cancels to ~zero by Legendre
        orthogonality). It peaks at the radial shells where the PADF has
        angular structure and is used to choose the near-neighbor shell;
        the integration shell [r_min, r_max] is shaded.

        Both panels are weighted by r^(2 * r_display_power) for display, and
        the map's color limits are taken from the interior angular range
        (15-165 deg) so the collinear theta = 0 / 180 bands saturate rather
        than compressing the color scale.

        Parameters
        ----------
        r_min, r_max : float or None
            Integration shell bounds in the r units of the PADF (Angstroms
            for calibrated data). If None, the first peak of g2 is selected
            automatically: from the last point below 5% of the peak height
            up to the first local minimum past the peak (atomode's default
            shell heuristic).
        r_display_power : int
            Display weighting exponent; each panel is multiplied by
            r^(2 * r_display_power). 0 plots the raw values.
        r_max_display : float or None
            Upper limit of the radial axes; defaults to min(8 A, r.max()).
        r_search_min : float
            Ignore r below this value when auto-locating the g2 peak
            (excludes the r ~ 0 reconstruction artifacts).
        r_marks : sequence of float or None
            Radii (e.g. the 1st/2nd/3rd neighbor shell distances) marked
            with dotted vertical lines on both panels.
        markers : sequence of (r, theta_deg) or None
            Expected 3-body maxima overlaid on the map as open circles -
            e.g. the (shell distance, arm-arm angle) targets from a known
            structure. A third element per tuple, if present, scales the
            marker size (relative multiplicity).
        title : str or None
            Figure title; a shell summary is used if None.
        returnfig : bool
            Return (fig, (ax_g2, ax_g3)) instead of calling plt.show().
        """
        r = np.asarray(self.r)
        theta = np.asarray(self.theta)  # radians
        theta_deg = np.asarray(self.theta_deg)
        arr = self.padf.numpy() if hasattr(self.padf, "numpy") else np.asarray(self.padf)

        # pair profile: sin(theta)-weighted RMS of the diagonal over the
        # interior angles (a plain integral cancels by Legendre orthogonality
        # since the l = 0 term is removed by the mean subtraction)
        diag = np.einsum("iik->ik", arr)
        interior_t = (theta_deg > 15) & (theta_deg < 165)
        w_t = np.abs(np.sin(theta[interior_t]))
        g2 = np.sqrt((diag[:, interior_t] ** 2 * w_t[None, :]).sum(axis=1) / w_t.sum())

        # auto-select the first g2 peak if no shell was given
        if r_min is None or r_max is None:
            kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
            kernel /= kernel.sum()
            smooth = np.convolve(np.where(r < r_search_min, 0.0, g2), kernel, mode="same")
            i_pk = None
            for idx in range(1, smooth.size - 1):
                if smooth[idx] > 0 and smooth[idx] >= smooth[idx - 1] and smooth[idx] > smooth[idx + 1]:
                    i_pk = idx
                    break
            if i_pk is None:
                i_pk = int(np.argmax(smooth))
            below = np.flatnonzero(smooth[:i_pk] < 0.05 * smooth[i_pk])
            i_lo = int(below[-1]) if below.size else int(np.searchsorted(r, r_search_min))
            i_hi = min(smooth.size - 1, 2 * i_pk - i_lo)
            for idx in range(i_pk + 1, smooth.size - 1):
                if smooth[idx] <= smooth[idx - 1] and smooth[idx] <= smooth[idx + 1]:
                    i_hi = idx
                    break
            if r_min is None:
                r_min = float(r[i_lo])
            if r_max is None:
                r_max = float(r[i_hi])

        shell = (r >= r_min) & (r <= r_max)
        if not np.any(shell):
            shell[int(np.argmin(np.abs(r - 0.5 * (r_min + r_max))))] = True

        # g3 map: integrate over the shell, symmetrized over both r axes
        g3_map = 0.5 * (arr[shell].sum(axis=0) + arr[:, shell].sum(axis=1))  # (Nr', Ntheta)

        # display weighting and color limits
        if r_max_display is None:
            r_max_display = float(min(8.0, r[-1]))
        weight = r ** (2 * r_display_power)
        g2_disp = g2 * weight
        g3_disp = (g3_map * weight[:, None]).T  # (Ntheta, Nr')
        rs = r <= r_max_display
        interior = (theta_deg > 15) & (theta_deg < 165)
        vmax = np.quantile(np.abs(g3_disp[np.ix_(interior, rs)]), 0.999)

        fig, (ax_g3, ax_g2) = plt.subplots(
            2, 1, sharex=True, figsize=figsize,
            gridspec_kw={"height_ratios": [2, 1]},
            constrained_layout=True,
        )
        im = ax_g3.pcolormesh(
            r[rs], theta_deg, g3_disp[:, rs],
            cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto",
        )
        ax_g3.set_ylabel("$\\theta$ (deg)")
        ax_g3.set_yticks(np.arange(0, 181, 45))
        fig.colorbar(im, ax=[ax_g3, ax_g2], label="3-body correlation", pad=0.02)

        if markers is not None:
            for mk in markers:
                r_mk, t_mk = mk[0], mk[1]
                size = 12.0 * (mk[2] ** 0.5) if len(mk) > 2 else 10.0
                if r_mk <= r_max_display:
                    ax_g3.plot(r_mk, t_mk, "o", ms=max(size, 5.0), mfc="none",
                               mec="k", mew=1.0)

        ax_g2.plot(r[rs], g2_disp[rs], "k-", lw=1)
        ax_g2.axvspan(r_min, r_max, color="C1", alpha=0.25)
        ax_g2.axhline(0, color="0.85", lw=0.8, zorder=0)

        if r_marks is not None:
            for r_mk in r_marks:
                for ax in (ax_g3, ax_g2):
                    ax.axvline(r_mk, color="k", ls=":", lw=0.8)
        ax_g2.set_ylabel("2-body corr.")
        ax_g2.set_xlabel("r, r' (Å)")

        ax_g3.set_title(
            title if title is not None
            else f"shell r = {r_min:.2f} - {r_max:.2f}",
            fontsize=10,
        )

        if returnfig:
            return fig, (ax_g3, ax_g2)
        plt.show()

    def plot_diagonal(
        self,
        rmin: float | None = None,
        rmax: float | None = None,
        r_display_power: int = 1,
        r_max_display: float | None = None,
        interior_theta_deg: tuple[float, float] = (15.0, 165.0),
        r_marks=None,
        markers=None,
        title: str | None = None,
        figsize: tuple[float, float] = (4.8, 4.4),
        returnfig: bool = False,
    ):
        """
        Map of the PADF's r = r' diagonal, Theta(r, r, theta), plotted as angle
        theta (y-axis) vs radius r (x-axis).

        The full PADF Theta(r, r', theta) lives on a 3D (r, r', theta) grid. The
        r = r' diagonal is the self-correlation slice: for every radius r, how
        the angular correlation Theta(r, r, theta) varies with theta. This is
        the same diagonal collapsed to a single r^(2*r_display_power)-weighted
        RMS number by the bottom (g2) panel of plot_g2_g3; here the theta
        dependence is kept instead of being reduced away, so structure that g2
        averages out (e.g. which angles dominate at a given shell) is visible.

        Unlike plot_g2_g3's g3 map (which integrates the PADF over a chosen
        radial shell on one axis, symmetrized over both r axes), this map takes
        no integration and needs no symmetrization: r = r' is already a single,
        well-defined line through the (r, r', theta) volume, so the diagonal is
        read off directly as Theta(r, r, theta).

        As in plot_g2_g3, color limits are taken from the interior angular
        range (theta in interior_theta_deg) rather than the full 0-180 range,
        so the collinear theta ~ 0 / 180 bands - where the diagonal tends to be
        large simply because forward/backward-scattered triplets are geometrically
        degenerate - don't saturate the color scale and hide the interior-angle
        structure. The display is weighted by r^(2 * r_display_power), matching
        plot_g2_g3's convention (equivalent to weighting each of the two
        coincident r, r' axes by r^r_display_power, as plot_theta does for its
        off-diagonal map).

        Parameters
        ----------
        padf : PairAngleDistributionFunction
            Reconstructed PADF object exposing `.r` (radii), `.theta_deg`
            (theta axis in degrees), and `.padf` (the (Nr, Nr, Ntheta) tensor
            or array of Theta(r, r', theta)).
        rmin, rmax : float or None
            Display range of the radial axis (Angstroms). Defaults to
            [0, r_max_display].
        r_display_power : int
            Display weighting exponent; the map is multiplied by
            r^(2 * r_display_power). 0 plots the raw diagonal values.
        r_max_display : float or None
            Upper limit of the radial axis if rmax is not given; defaults to
            min(8 A, r.max()).
        interior_theta_deg : (float, float)
            Theta range (degrees) used both to select the color-scale quantile
            and to mark the excluded collinear bands are outside of; defaults
            to (15, 165), matching plot_g2_g3.
        r_marks : sequence of float or None
            Radii (e.g. known neighbor shell distances) marked with dotted
            vertical lines.
        markers : sequence of (r, theta_deg) or None
            Expected diagonal maxima overlaid as open circles - e.g. the
            (shell distance, arm-arm angle) targets from a known structure. A
            third element per tuple, if present, scales the marker size
            (relative multiplicity).
        title : str or None
            Figure title; a generic default is used if None.
        figsize : (float, float)
            Figure size in inches.
        returnfig : bool
            Return (fig, ax) instead of calling plt.show().

        Returns
        -------
        (fig, ax) if returnfig is True, otherwise None (the figure is shown).
        """
        r = np.asarray(self.r)
        theta_deg = np.asarray(self.theta_deg)
        arr = self.padf.numpy() if hasattr(self.padf, "numpy") else np.asarray(self.padf)

        # r = r' diagonal: Theta(r, r, theta), shape (Nr, Ntheta)
        diag = np.einsum("iik->ik", arr)

        if r_max_display is None:
            r_max_display = float(min(8.0, r[-1]))
        if rmax is None:
            rmax = r_max_display
        rmin_eff = 0.0 if rmin is None else rmin
        if rmax <= rmin_eff:
            raise ValueError(f"rmax must be > rmin (got rmin={rmin_eff}, rmax={rmax}).")

        r_mask = (r >= rmin_eff) & (r <= rmax)
        if not np.any(r_mask):
            raise ValueError("Requested plot range contains no data.")

        # display weighting and orientation: theta on rows (y-axis), r on columns (x-axis)
        weight = r ** (2 * r_display_power)
        diag_disp = (diag * weight[:, None]).T  # (Ntheta, Nr)

        # color limits from the interior angular range only, so the collinear
        # theta ~ 0 / 180 bands don't saturate the scale
        t_lo, t_hi = interior_theta_deg
        interior_t = (theta_deg > t_lo) & (theta_deg < t_hi)
        interior_vals = diag_disp[np.ix_(interior_t, r_mask)]
        vmax = np.quantile(np.abs(interior_vals), 0.999) if interior_vals.size else \
            np.quantile(np.abs(diag_disp[:, r_mask]), 0.999)

        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        im = ax.pcolormesh(
            r[r_mask], theta_deg, diag_disp[:, r_mask],
            cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto",
        )
        fig.colorbar(im, ax=ax, label=r"$\Theta(r, r, \theta)$", pad=0.02)

        if markers is not None:
            for mk in markers:
                r_mk, t_mk = mk[0], mk[1]
                size = 12.0 * (mk[2] ** 0.5) if len(mk) > 2 else 10.0
                if rmin_eff <= r_mk <= rmax:
                    ax.plot(r_mk, t_mk, "o", ms=max(size, 5.0), mfc="none",
                            mec="k", mew=1.0)

        if r_marks is not None:
            for r_mk in r_marks:
                ax.axvline(r_mk, color="k", ls=":", lw=0.8)

        ax.set_xlabel("r = r' (Å)")
        ax.set_ylabel(r"$\theta$ (deg)")
        ax.set_yticks(np.arange(0, 181, 45))
        ax.set_title(
            title if title is not None
            else f"r = r' diagonal, θ ∈ [0°, 180°]",
            fontsize=10,
        )

        if returnfig:
            return fig, ax
        plt.show()

    # Theta helper
    def _range_mask(self, x, xmin, xmax):
        """Return a boolean mask selecting `x` within [xmin, xmax], defaulting to all True."""
        if xmin is None and xmax is None:
            return np.ones_like(x, dtype=bool)
        xmin_eff = x.min() if xmin is None else xmin
        xmax_eff = x.max() if xmax is None else xmax
        if xmax_eff <= xmin_eff:
            raise ValueError(f"xmax must be > xmin (got xmin={xmin_eff}, xmax={xmax_eff}).")
        mask = (x >= xmin_eff) & (x <= xmax_eff)
        if not np.any(mask):
            raise ValueError("Requested plot range contains no data.")
        return mask

    # Theta helper
    def _nearest_index(self, x, value):
        return int(np.argmin(np.abs(x - value)))

    def plot_theta(
        self,
        theta_deg: float,
        rmin: float | None = None,
        rmax: float | None = None,
        r_display_power: int = 0,
        r_max_display: float | None = None,
        r_search_min: float = 0.5,
        r_marks=None,
        markers=None,
        title: str | None = None,
        figsize: tuple[float, float] = (6, 5),
        labels: bool = True,
        cbar: bool = True,
        returnfig: bool = False,
    ):
        """
        Plot Theta(r, r', theta) as a 2D map over the (r, r') grid at a fixed
        angle theta (degrees), generalizing plot_theta0_map to any theta.

        The raw slice mixes two very different signals: a strong ridge along
        r = r' (essentially the g2-like self-correlation already shown by
        plot_g2_g3's bottom panel) and the much weaker off-diagonal r != r'
        structure that encodes genuine 3-body angular correlations. Plotted
        naively with a linear color scale, the diagonal ridge saturates the
        colorbar and the off-diagonal structure disappears. This version:

        - Symmetrizes the slice over r <-> r', since swapping which atom is
          "first" is just a relabeling and any asymmetry is numerical noise
          from the reconstruction.
        - Sets color limits from the off-diagonal region only (excluding a
          small artifact zone near r = r' < r_search_min), analogous to how
          plot_g2_g3 excludes the collinear theta bands when choosing vmax.
        - Uses a diverging RdBu_r colormap centered at zero, since Theta can
          be positive or negative.
        - Optionally weights the display by (r * r')^r_display_power to
          compensate for the PADF's natural radial amplitude decay.
        - Draws the r = r' diagonal as a dashed reference line, and supports
          the same r_marks / markers overlays as plot_g2_g3 for annotating
          known neighbor distances or expected (r, r') pair maxima.
        - Uses a square aspect ratio, since r and r' share the same units
          and axis.

        Parameters
        ----------
        theta_deg : float
            Angle (degrees) at which to take the slice; snapped to the
            nearest available theta bin.
        rmin, rmax : float or None
            Display range for both r and r' axes (Angstroms). Defaults to
            [0, r_max_display].
        r_display_power : int
            Display weighting exponent; the map is multiplied by
            (r * r')^r_display_power. 0 plots the raw values.
        r_max_display : float or None
            Upper limit of both axes if rmax is not given; defaults to
            min(8 A, r.max()).
        r_search_min : float
            Radii below this are excluded when computing the off-diagonal
            color scale, avoiding r ~ 0 reconstruction artifacts.
        r_marks : sequence of float or None
            Radii (e.g. known neighbor shell distances) marked with dotted
            lines on both axes.
        markers : sequence of (r, r', [size]) or None
            Expected (r, r') pair maxima overlaid as open circles; a third
            element per tuple scales the marker size.
        title : str or None
            Figure title; a theta-value summary is used if None.
        returnfig : bool
            Return (fig, ax) instead of calling plt.show().
        """
        r = np.asarray(self.r)
        theta_axis = np.asarray(self.theta_deg)

        padf_arr = self.padf.numpy() if hasattr(self.padf, "numpy") else np.asarray(self.padf)
        theta_idx = self._nearest_index(theta_axis, theta_deg)
        slice_2d = padf_arr[:, :, theta_idx]  # (Nr, Nr')

        # symmetrize over r <-> r'
        slice_2d = 0.5 * (slice_2d + slice_2d.T)

        if r_max_display is None:
            r_max_display = float(min(8.0, r[-1]))
        if rmax is None:
            rmax = r_max_display

        r_mask = self._range_mask(r, rmin, rmax)
        r_sel = r[r_mask]
        slice_disp = slice_2d[np.ix_(r_mask, r_mask)]

        # display weighting
        weight = r_sel ** r_display_power
        slice_disp = slice_disp * weight[:, None] * weight[None, :]

        # color limits from off-diagonal region only, excluding a small
        # zone near the origin where reconstruction artifacts dominate
        keep = r_sel >= r_search_min
        off_diag = ~np.eye(len(r_sel), dtype=bool)
        keep_mask = np.outer(keep, keep)
        sub_off = slice_disp[off_diag & keep_mask]
        vmax = np.quantile(np.abs(sub_off), 0.999) if sub_off.size else np.quantile(np.abs(slice_disp), 0.999)

        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        im = ax.pcolormesh(
            r_sel, r_sel, slice_disp.T,
            cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto",
        )
        if cbar:
            fig.colorbar(im, ax=ax, label=r"$\Theta(r, r', \theta)$", pad=0.02)

        # r = r' reference line
        ax.plot([r_sel[0], r_sel[-1]], [r_sel[0], r_sel[-1]],
                color="k", lw=0.6, ls="--", alpha=0.5)

        if r_marks is not None:
            for r_mk in r_marks:
                ax.axvline(r_mk, color="k", ls=":", lw=0.8)
                ax.axhline(r_mk, color="k", ls=":", lw=0.8)

        if markers is not None:
            for mk in markers:
                r_mk, rp_mk = mk[0], mk[1]
                size = 12.0 * (mk[2] ** 0.5) if len(mk) > 2 else 10.0
                if r_mk <= r_sel[-1] and rp_mk <= r_sel[-1]:
                    ax.plot(r_mk, rp_mk, "o", ms=max(size, 5.0),
                            mfc="none", mec="k", mew=1.0)

        ax.set_aspect("equal")
        if labels:
            ax.set_xlabel("r (Å)")
            ax.set_ylabel("r' (Å)")
        ax.set_title(
            title if title is not None
            else rf"$\Theta(r, r', \theta={theta_axis[theta_idx]:.1f}\degree)$",
            fontsize=10,
        )

        if returnfig:
            return fig, ax
        plt.show()

    ########################## OBSOLETE ##########################

    def plot_theta_simple(
        self,
        theta_deg: float,
        rmin: float | None = None,
        rmax: float | None = None,
        figsize: tuple[float, float] = (6, 5),
        returnfig: bool = False,
    ):
        """
        Plot Theta(r, r', theta) as a 2D map over the full (r, r') grid at a
        fixed angle theta (degrees), generalizing plot_theta0_map to any theta.
        """
        r = self.r
        theta_axis = self.theta_deg

        padf_arr = self.padf.numpy() if hasattr(self.padf, "numpy") else np.asarray(self.padf)
        theta_idx = self._nearest_index(theta_axis, theta_deg)
        slice_2d = padf_arr[:, :, theta_idx]  # (Nr, Nr')

        r_mask = self._range_mask(r, rmin, rmax)
        r_sel = r[r_mask]
        slice_2d = slice_2d[np.ix_(r_mask, r_mask)]

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.pcolormesh(r_sel, r_sel, slice_2d.T, shading="auto")
        fig.colorbar(im, ax=ax, label=r"$\Theta(r, r', \theta)$")

        ax.set_xlabel("r (Å)")
        ax.set_ylabel("r' (Å)")
        ax.set_title(rf"$\Theta(r, r', \theta={theta_axis[theta_idx]:.1f}\degree)$")
        fig.tight_layout()

        if returnfig:
            return fig
        plt.show()



