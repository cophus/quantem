from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.datastructures.polar4dstem import Polar4dstem
from quantem.diffraction.polar import PairDistributionFunction
from quantem.diffraction.polar_transform import find_origin_angular_grid, polar_transform


def _arpls_baseline(
    y: torch.Tensor, lam: float = 0.1, ratio: float = 1e-2, niter: int = 100
) -> torch.Tensor:
    """Asymmetrically reweighted penalized least squares baseline.

    Smoothness-regularised lower-envelope estimator that handles both
    polycrystalline (many sharp peaks on a flat baseline) and amorphous
    (one broad diffuse halo) profile topologies. Iteratively reweights
    points that sit above the current estimate so peaks get pushed away
    instead of absorbed.

    Reference: Baek, Park, Ahn & Choo, Analyst 140 (2015) 250-257.

    Parameters
    ----------
    y : torch.Tensor
        1D signal to fit a baseline under.
    lam : float
        Smoothness penalty. Larger -> stiffer baseline.
    ratio : float
        Convergence tolerance on the relative change in weights.
    niter : int
        Maximum iterations.

    Returns
    -------
    z : torch.Tensor
        Smooth baseline, same length, device, and dtype-promoted-to-float64
        as ``y``.
    """
    y = y.to(dtype=torch.float64)
    L = y.numel()
    device = y.device
    # 2nd-difference operator D is L x (L-2); column j has [1, -2, 1] at
    # rows j, j+1, j+2. H = lam * D @ D.T is the smoothness penalty (band-5).
    col = torch.arange(L - 2, device=device)
    D = torch.zeros(L, L - 2, device=device, dtype=torch.float64)
    D[col, col] = 1.0
    D[col + 1, col] = -2.0
    D[col + 2, col] = 1.0
    H = lam * (D @ D.T)
    diag_rows = torch.arange(L, device=device)

    w = torch.ones(L, device=device, dtype=torch.float64)
    z = y.clone()
    for _ in range(niter):
        A = H.clone()
        A[diag_rows, diag_rows] += w  # A = diag(w) + H, symmetric PD
        z = torch.linalg.solve(A, w * y)
        d = y - z
        dn = d[d < 0]
        if dn.numel() == 0:
            break
        m = dn.mean()
        s = dn.std(correction=0)  # ddof=0 to match numpy default
        # Asymmetric reweighting; clip the logistic exponent so peaks don't overflow
        expo = torch.clamp(2.0 * (d - (2.0 * s - m)) / (s + 1e-12), -50.0, 50.0)
        wt = 1.0 / (1.0 + torch.exp(expo))
        if torch.linalg.norm(w - wt) / (torch.linalg.norm(w) + 1e-12) < ratio:
            break
        w = wt
    return z


def _find_peaks_torch(
    y: torch.Tensor,
    distance: int | None = None,
    prominence: float | None = None,
    height: float | None = None,
) -> torch.Tensor:
    """Torch-native 1D peak detection.

    Strict interior local maxima (``y[i] > y[i-1] AND y[i] > y[i+1]``),
    optionally filtered by absolute ``height``, minimum sample
    ``distance`` between peaks (greedy: smaller peaks dropped first to
    enforce the separation), and topographic ``prominence`` (vertical
    distance from the peak to the higher of the two surrounding valley
    minima, where each valley min is the running min from the peak to
    the nearest higher peak or array edge).

    Matches the semantics of ``scipy.signal.find_peaks`` for the
    filter set used by :meth:`ReciprocalCalibration.find_peaks`.
    Plateau peaks (equal-valued runs surrounded by lower values) are
    NOT detected — strict ``>`` only.

    Parameters
    ----------
    y : 1D torch.Tensor
        Signal to search.
    distance : int or None
        Minimum samples between peaks.
    prominence : float or None
        Minimum topographic prominence.
    height : float or None
        Minimum absolute peak value.

    Returns
    -------
    peaks : 1D torch.Tensor (long), peak indices sorted ascending.
    """
    n = y.numel()
    if n < 3:
        return torch.empty(0, dtype=torch.long, device=y.device)

    # Pull to Python floats. The data is small (typical radial profile
    # length is ~100), so a tight Python loop over a few candidates is
    # cheaper than any vectorized prominence scheme. The scipy version
    # we're replacing is also CPU-bound for the same reason.
    vals = y.detach().cpu().tolist()

    # Strict interior local maxima
    candidates = [
        i for i in range(1, n - 1) if vals[i] > vals[i - 1] and vals[i] > vals[i + 1]
    ]

    # Absolute height
    if height is not None:
        candidates = [i for i in candidates if vals[i] >= height]

    # Minimum separation — greedy by descending height, drop smaller
    # candidates within `distance` of any already-kept candidate
    if distance is not None and candidates:
        by_height = sorted(candidates, key=lambda i: vals[i], reverse=True)
        kept: list[int] = []
        for c in by_height:
            if all(abs(c - k) >= distance for k in kept):
                kept.append(c)
        candidates = sorted(kept)

    # Prominence — walk left/right from each candidate tracking running min
    # until a strictly higher value (or edge) is hit
    if prominence is not None and candidates:
        kept_prom: list[int] = []
        for c in candidates:
            h = vals[c]
            left_min = h
            j = c - 1
            while j >= 0:
                v = vals[j]
                if v > h:
                    break
                if v < left_min:
                    left_min = v
                j -= 1
            right_min = h
            j = c + 1
            while j < n:
                v = vals[j]
                if v > h:
                    break
                if v < right_min:
                    right_min = v
                j += 1
            if h - max(left_min, right_min) >= prominence:
                kept_prom.append(c)
        candidates = kept_prom

    return torch.tensor(candidates, dtype=torch.long, device=y.device)

# ---------------------------------------------------------------------------
# Reference material data
# {material: (lattice_param_Å, {hkl_label: h²+k²+l²})}
#   Au -- FCC, allowed when (h,k,l) all same parity.
#   Si -- diamond cubic, allowed when (h,k,l) all odd, OR all even with
#         h+k+l = 4n. The 200, 222, 420 FCC reflections are forbidden.
# ---------------------------------------------------------------------------
MATERIALS: dict[str, tuple[float, dict[str, int]]] = {
    "Au": (
        4.0782,
        {
            "111": 3, "200": 4, "220": 8, "311": 11,
            "222": 12, "400": 16, "331": 19, "420": 20,
        },
    ),
    "Si": (
        5.4307,
        {
            "111": 3, "220": 8, "311": 11, "400": 16,
            "331": 19, "422": 24, "511": 27, "440": 32,
            "531": 35, "620": 40,
        },
    ),
}


def _q_values_for(material: str) -> dict[str, float]:
    """Return q = sqrt(h²+k²+l²) / a for each allowed reflection of `material`."""
    if material not in MATERIALS:
        raise ValueError(
            f"Unknown material '{material}'. Available: {sorted(MATERIALS)}"
        )
    a, reflections = MATERIALS[material]
    return {hkl: np.sqrt(n) / a for hkl, n in reflections.items()}


class ReciprocalCalibration:
    """Calibrate reciprocal-space pixel size from polycrystalline diffraction rings.

    This class determines the pixel-to-inverse-Angstrom conversion factor by
    matching detected ring positions in a radial profile to known d-spacings
    of a reference material (currently Au FCC).

    Construction is enforced through :meth:`from_data`.
    """

    _token = object()

    def __init__(
        self,
        pdf: PairDistributionFunction,
        radial_profile: NDArray,
        radial_pixel_positions: NDArray,
        mean_dp: NDArray,
        input_data: Any | None = None,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError(
                "Direct instantiation of ReciprocalCalibration is not allowed. "
                "Use ReciprocalCalibration.from_data() to instantiate this class."
            )

        self._pdf = pdf  # used for fit_bg
        self.radial_profile = radial_profile  # 1-D, length = n_radial_bins
        self.radial_pixel_positions = radial_pixel_positions  # 1-D pixel indices
        self.mean_dp = mean_dp  # 2-D mean diffraction pattern
        self.input_data = input_data

        # Set after fit_bg() / find_peaks()
        self.background: NDArray | None = None
        self.peak_indices: NDArray | None = None
        self.peak_pixel_positions: NDArray | None = None

        # Set after calibrate()
        self.pixel_size: float | None = None  # 1/Å per pixel
        self.material: str | None = None
        self.matched_hkl: list[str] | None = None
        self.matched_q: NDArray | None = None
        self.matched_pixel_positions: NDArray | None = None

    @classmethod
    def from_data(
        cls,
        data: Dataset4dstem | Dataset2d | Polar4dstem,
        *,
        mask_realspace: NDArray | None = None,
        voltage_kV: float = 200,
        find_origin: bool = True,
        origin_row: float | None = None,
        origin_col: float | None = None,
        radial_min: float = 0.0,
        radial_max: float | None = None,
        radial_step: float = 1.0,
        num_annular_bins: int = 180,
        ellipse_params: tuple[float, float, float] | None = None,
    ) -> "ReciprocalCalibration":
        """Create a ReciprocalCalibration from diffraction data.

        Parameters
        ----------
        data
            A Dataset4dstem, Dataset2d, or Polar4dstem.
        mask_realspace
            Boolean array (scan_y, scan_x) selecting which probe positions
            to include when averaging diffraction patterns.  Only used with
            Dataset4dstem input.
        voltage_kV
            Accelerating voltage in kV (stored for metadata / future use).
        find_origin, origin_row, origin_col
            Origin-finding parameters forwarded to the polar transform.
        radial_min, radial_max, radial_step, num_annular_bins
            Polar-transform parameters.
        ellipse_params
            Elliptical-distortion correction ``(a, b, theta_deg)`` applied
            during origin finding and the polar transform. Must match the
            value passed to the PDF's polar transform so the calibration is
            measured on the same (ellipse-corrected) geometry.
        """
        # Polar4dstem input: use directly 
        if isinstance(data, Polar4dstem):
            polar = data
            mean_dp = None
        else:
            # Dataset2d: wrap as 1×1 4DSTEM
            if isinstance(data, Dataset2d):
                arr2d = data.array
                arr4 = arr2d[None, None, ...]
                data = Dataset4dstem.from_array(
                    array=arr4,
                    name=f"{data.name}_as4dstem"
                    if getattr(data, "name", None)
                    else "cal_4dstem_from_2d",
                    origin=np.concatenate(
                        [np.zeros(2, dtype=float), np.asarray(data.origin, dtype=float)]
                    ),
                    sampling=np.concatenate(
                        [np.ones(2, dtype=float), np.asarray(data.sampling, dtype=float)]
                    ),
                    units=["pixels", "pixels"] + list(data.units),
                    signal_units=data.signal_units,
                )
                mask_realspace = None  # irrelevant for 1×1 scan
            # Dataset4dstem: compute mean DP then polar transform
            if not isinstance(data, Dataset4dstem):
                raise TypeError(
                    f"Unsupported data type {type(data).__name__}. "
                    "Expected Dataset4dstem, Dataset2d, or Polar4dstem."
                )
            arr4d_t = (
                torch.from_numpy(np.ascontiguousarray(data.array))
                if data.array is not None
                else data.tensor
            )
            if mask_realspace is not None:
                n_pos = int(mask_realspace.sum())
                print(f"Averaging {n_pos} masked diffraction patterns ...")
                mask_t = torch.from_numpy(np.asarray(mask_realspace, dtype=bool)).to(
                    arr4d_t.device
                )
                mean_dp = arr4d_t[mask_t].mean(dim=0, dtype=torch.float64).cpu().numpy()
            else:
                n_pos = data.shape[0] * data.shape[1]
                print(f"Averaging all {n_pos} diffraction patterns ...")
                mean_dp = arr4d_t.mean(dim=(0, 1), dtype=torch.float64).cpu().numpy()
            mean_ds = Dataset4dstem.from_array(
                array=mean_dp[None, None, ...],
                name="calibration_mean_dp",
                origin=np.array([0.0, 0.0, data.origin[2], data.origin[3]]),
                sampling=np.array([1.0, 1.0, data.sampling[2], data.sampling[3]]),
                units=["pixels", "pixels"] + list(data.units[2:]),
                signal_units=data.signal_units,
            )

            if find_origin:
                origin_array = find_origin_angular_grid(
                    mean_ds,
                    ellipse_params=ellipse_params,
                    num_annular_bins=num_annular_bins,
                    radial_min=radial_min,
                    radial_max=radial_max,
                    radial_step=radial_step,
                )
            else:
                ny, nx = mean_dp.shape
                if origin_row is None:
                    origin_row = (ny - 1) / 2.0
                if origin_col is None:
                    origin_col = (nx - 1) / 2.0
                origin_array = np.zeros((1, 1, 2), dtype=float)
                origin_array[0, 0, 0] = origin_row
                origin_array[0, 0, 1] = origin_col

            polar = polar_transform(
                mean_ds,
                origin_array=origin_array,
                ellipse_params=ellipse_params,
                num_annular_bins=num_annular_bins,
                radial_min=radial_min,
                radial_max=radial_max,
                radial_step=radial_step,
            )

        # --- Build a PairDistributionFunction for background fitting --------
        # PairDistributionFunction.from_data does not accept a Polar4dstem
        # directly, so construct via the class-level token.
        pdf_input = data if isinstance(data, Dataset4dstem) else None
        pdf = PairDistributionFunction(
            polar=polar,
            input_data=pdf_input,
            _token=PairDistributionFunction._token,
        )
        pdf.calculate_radial_mean()

        # --- Compute radial profile (azimuthal average) ---------------------
        radial_profile = pdf.Ik.detach().cpu().numpy()
        n_r = radial_profile.shape[0]
        radial_pixel_positions = np.arange(n_r, dtype=float)

        if mean_dp is None:
            mean_dp = np.zeros((1, 1), dtype=float)

        return cls(
            pdf=pdf,
            radial_profile=radial_profile,
            radial_pixel_positions=radial_pixel_positions,
            mean_dp=mean_dp,
            input_data=data,
            _token=cls._token,
        )

    def fit_bg(
        self,
        radial_min: int = 0,
        lam: float = 0.001,
    ) -> NDArray:
        """Estimate the smooth background under the diffraction rings.

        Runs arPLS (asymmetrically reweighted penalized least squares;
        see :func:`_arpls_baseline`) in log-space on
        ``radial_profile[radial_min:]``. Below ``radial_min`` the
        background is set equal to the profile, so the residual is ~0
        there and the central-beam region can't dominate peak finding.
        Tunable via one smoothness parameter.

        Parameters
        ----------
        radial_min : int
            Below this pixel index, ``bg = radial_profile`` (no fit).
        lam : float
            arPLS smoothness penalty. Larger -> stiffer baseline.

        Returns
        -------
        bg : NDArray
            Background curve, same length as :attr:`radial_profile`.
        """
        profile = np.asarray(self.radial_profile, dtype=float)
        bg = profile.copy()
        # arPLS in log space so the steep central-beam decay is tractable
        log_seg = np.log(np.clip(profile[radial_min:], 0.0, None) + 1.0)
        log_seg_t = torch.from_numpy(np.ascontiguousarray(log_seg))
        bg_seg = (torch.exp(_arpls_baseline(log_seg_t, lam=lam)) - 1.0).cpu().numpy()
        bg[radial_min:] = bg_seg
        bg = np.clip(bg, 0.0, None)
        self.background = bg
        return self.background

    def find_peaks(
        self,
        radial_min: int = 0,
        n_peaks: int | None = None,
        distance: int | None = None,
        prominence: float | None = None,
        height: float | None = None,
    ) -> NDArray:
        """Detect peaks in the background-subtracted radial profile.

        Runs :func:`_find_peaks_torch` on ``radial_profile - background``,
        then refines each detected peak to sub-pixel precision via a
        parabolic vertex fit to its three nearest samples. The integer
        bins remain in :attr:`peak_indices` (for residual-value lookups);
        the refined positions go in :attr:`peak_pixel_positions` and feed
        :meth:`calibrate`'s least-squares fit.

        If :attr:`background` has not been fit yet, calls :meth:`fit_bg`
        with defaults first.

        Parameters
        ----------
        radial_min
            Ignore the profile below this pixel index (skip the central beam).
        n_peaks
            If provided, keep only the first ``n_peaks`` detected peaks
            (the innermost rings). Useful when the detector finds more
            candidates than you want to use for the calibration fit.
        distance, prominence, height
            Passed through to :func:`_find_peaks_torch` (scipy-equivalent
            semantics).

        Returns
        -------
        peak_positions : NDArray
            Sub-pixel-refined peak positions (same units as
            :attr:`radial_pixel_positions`).
        """
        if self.background is None:
            self.fit_bg(radial_min=radial_min)
        residual = self.radial_profile - self.background

        # zero out central beam region
        residual[:radial_min] = 0.0
        self._residual = residual

        residual_t = torch.from_numpy(np.ascontiguousarray(residual))
        indices_t = _find_peaks_torch(
            residual_t,
            distance=distance,
            prominence=prominence,
            height=height,
        )
        indices = indices_t.cpu().numpy()
        indices = indices[indices >= radial_min]
        if n_peaks is not None:
            indices = indices[:n_peaks]

        # Sub-pixel parabolic refinement: vertex offset of the parabola
        # through residual[idx-1], residual[idx], residual[idx+1]. Clipped
        # to [-1, 1] bin steps to guard against flat / inverted curvature.
        positions = self.radial_pixel_positions
        step = float(positions[1] - positions[0]) if len(positions) > 1 else 1.0
        refined = np.empty_like(indices, dtype=float)
        for k, idx in enumerate(indices):
            base = float(positions[idx])
            if 0 < idx < len(residual) - 1:
                y0 = float(residual[idx - 1])
                y1 = float(residual[idx])
                y2 = float(residual[idx + 1])
                denom = y0 - 2.0 * y1 + y2
                if denom != 0.0:
                    delta = 0.5 * (y0 - y2) / denom
                    delta = max(-1.0, min(1.0, delta))
                else:
                    delta = 0.0
                refined[k] = base + delta * step
            else:
                refined[k] = base

        self.peak_indices = indices
        self.peak_pixel_positions = refined
        return refined

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def calibrate(
        self,
        material: str = "Au",
        peak_indices: NDArray | list[int] | None = None,
    ) -> float:
        """Match detected peaks to known ring positions and fit pixel size.

        Performs a zero-intercept least-squares fit: ``q = pixel_size * r_pixels``
        to determine the reciprocal-space pixel size in 1/Å per pixel.

        Parameters
        ----------
        material
            Reference material; must be a key in :data:`MATERIALS`
            (currently ``'Au'`` or ``'Si'``).
        peak_indices
            Indices of peaks (into :attr:`radial_profile`) to use for the
            fit.  If ``None``, uses all peaks found by :meth:`find_peaks`.

        Returns
        -------
        pixel_size : float
            Calibrated pixel size in 1/Å per pixel.
        """
        if peak_indices is not None:
            peak_pix = self.radial_pixel_positions[np.asarray(peak_indices)]
        elif self.peak_pixel_positions is not None:
            peak_pix = self.peak_pixel_positions
        else:
            raise RuntimeError(
                "No peaks available. Call find_peaks() first or provide peak_indices."
            )

        q_ref = _q_values_for(material)
        self.material = material
        hkl_labels = list(q_ref.keys())
        q_vals = np.array(list(q_ref.values()))
        sorted_order = np.argsort(q_vals)
        q_sorted = q_vals[sorted_order]
        hkl_sorted = [hkl_labels[i] for i in sorted_order]
        peak_pix = np.sort(peak_pix.astype(float))
        n_peaks = len(peak_pix)
        n_ref = len(q_sorted)

        # Try all ordered subsets of n_peaks reflections from the n_ref
        # available and find the assignment with the lowest relative fit
        # residual.  Since both peaks and q-values are sorted, we only
        # need to consider combinations (order-preserving).
        from itertools import combinations

        best_rss = np.inf
        best_combo = None
        for combo in combinations(range(n_ref), n_peaks):
            q_sel = q_sorted[list(combo)]
            # Zero-intercept LS: pixel_size = (r · q) / (r · r)
            ps = float(np.dot(peak_pix, q_sel) / np.dot(peak_pix, peak_pix))
            if ps <= 0:
                continue
            # Relative residual sum of squares
            predicted_q = ps * peak_pix
            rss = float(np.sum(((predicted_q - q_sel) / q_sel) ** 2))
            if rss < best_rss:
                best_rss = rss
                best_combo = combo
        if best_combo is None:
            raise RuntimeError("Could not find a valid peak-to-reflection matching.")

        matched_q = q_sorted[list(best_combo)]
        matched_hkl = [hkl_sorted[i] for i in best_combo]
        pixel_size = float(np.dot(peak_pix, matched_q) / np.dot(peak_pix, peak_pix))
        self.pixel_size = pixel_size
        self.matched_hkl = matched_hkl
        self.matched_q = matched_q
        self.matched_pixel_positions = peak_pix
        # Print matching for user verification
        print(f"Calibrated pixel size: {pixel_size:.6f} 1/Å/px")
        print("Peak matching:")
        for r_px, hkl, q in zip(peak_pix, matched_hkl, matched_q):
            q_pred = pixel_size * r_px
            print(f"  r={r_px:.1f} px → {hkl} (q={q:.4f}, predicted={q_pred:.4f})")
        return pixel_size

    def apply(self, dataset: Dataset4dstem) -> None:
        """Update a Dataset4dstem's diffraction-axis sampling and units."""
        if self.pixel_size is None:
            raise RuntimeError("No calibration available. Call calibrate() first.")
        new_sampling = list(dataset.sampling)
        new_sampling[2] = self.pixel_size
        new_sampling[3] = self.pixel_size
        dataset.sampling = new_sampling
        new_units = list(dataset.units)
        new_units[2] = "1/Angstrom"
        new_units[3] = "1/Angstrom"
        dataset.units = new_units

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------
    def plot_radial_profile(self) -> plt.Figure:
        """Plot the radial profile, background fit, and residual with peaks.

        Peak markers sit at the sub-pixel-refined positions stored in
        :attr:`peak_pixel_positions` (their y-values are linearly
        interpolated onto the underlying curve). If :meth:`calibrate` has
        been run, each marker is annotated with the matched ``{hkl}``.
        """
        has_bg = self.background is not None
        n_axes = 2 if has_bg else 1

        fig, axes = plt.subplots(1, n_axes, figsize=(5 * n_axes, 4))
        if n_axes == 1:
            axes = [axes]

        marker_kw = dict(s=40, color="red", alpha=0.5, edgecolors="none",
                         zorder=5, label="Peaks")
        ann_kw = dict(textcoords="offset points", xytext=(5, 6),
                      fontsize=7, color="red", alpha=0.9)

        def _annotate_peaks(ax, y_curve):
            if self.peak_pixel_positions is None:
                return
            peak_y = np.interp(self.peak_pixel_positions,
                               self.radial_pixel_positions, y_curve)
            ax.scatter(self.peak_pixel_positions, peak_y, **marker_kw)
            if self.matched_hkl is not None:
                for r_sub, hkl, y in zip(self.peak_pixel_positions,
                                          self.matched_hkl, peak_y):
                    ax.annotate(f"{{{hkl}}}", (r_sub, y), **ann_kw)

        # --- Left panel: raw profile + background ---
        ax = axes[0]
        ax.plot(self.radial_pixel_positions, self.radial_profile, label="I(r)")
        if has_bg:
            ax.plot(self.radial_pixel_positions, self.background, "--", label="Background")
        _annotate_peaks(ax, self.radial_profile)
        ax.set_yscale("log")
        ax.set_xlabel("Radial position (pixels)")
        ax.set_ylabel("Mean intensity")
        ax.set_title("Radial profile + background")
        ax.legend()

        # --- Right panel: residual with peaks ---
        if has_bg:
            ax2 = axes[1]
            residual = getattr(self, "_residual", self.radial_profile - self.background)
            ax2.plot(self.radial_pixel_positions, residual, label="I(r) - B(r)")
            _annotate_peaks(ax2, residual)
            ax2.axhline(0, color="gray", linewidth=0.5)
            ax2.set_xlabel("Radial position (pixels)")
            ax2.set_ylabel("Residual intensity")
            ax2.set_title("Background-subtracted profile")
            ax2.legend()

        fig.tight_layout()
        return fig

    def plot_trial_pixel_sizes(
        self,
        trial_sizes: list[float] | None = None,
        material: str | None = None,
    ) -> plt.Figure:
        """Overlay expected ring positions on the residual for several
        trial pixel sizes so the user can visually identify the correct
        matching.

        Parameters
        ----------
        trial_sizes
            List of pixel sizes (1/Å per pixel) to try.  Defaults to
            three values spanning a plausible range.
        material
            Reference material whose rings are overlaid. Defaults to
            ``self.material`` (set by :meth:`calibrate`) or ``'Au'``.
        """
        if trial_sizes is None:
            trial_sizes = [0.018, 0.026, 0.040]

        if material is None:
            material = self.material or "Au"
        q_ref = _q_values_for(material)
        residual = getattr(self, "_residual", self.radial_profile)
        r_max = float(self.radial_pixel_positions[-1])

        n = len(trial_sizes)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]

        for ax, ps in zip(axes, trial_sizes):
            ax.plot(self.radial_pixel_positions, residual, "b-", linewidth=1)
            if self.peak_pixel_positions is not None:
                peak_y = np.interp(self.peak_pixel_positions,
                                   self.radial_pixel_positions, residual)
                ax.scatter(
                    self.peak_pixel_positions,
                    peak_y,
                    s=40,
                    color="red",
                    alpha=0.5,
                    edgecolors="none",
                    zorder=5,
                )
            y_top = max(residual.max() * 1.05, 1.0)
            for hkl, qv in q_ref.items():
                r_exp = qv / ps
                if r_exp < r_max:
                    ax.axvline(r_exp, color="green", alpha=0.5, linestyle="--", linewidth=1)
                    ax.text(
                        r_exp + 0.5,
                        y_top * 0.92,
                        hkl,
                        rotation=90,
                        va="top",
                        fontsize=7,
                        color="green",
                    )
            ax.set_title(f"pixel_size = {ps} 1/\u00c5/px")
            ax.set_xlabel("Radial position (pixels)")
            ax.set_ylabel("Residual intensity")
            ax.set_xlim(0, r_max)
            ax.set_ylim(None, y_top)

        fig.tight_layout()
        return fig

    def plot_calibration_fit(self) -> plt.Figure:
        """Plot the linear calibration fit and residuals.

        Left panel shows matched peak positions vs. known q-values with
        the zero-intercept linear fit.  Right panel shows the residuals
        (measured q - predicted q) for each matched reflection.
        """
        if self.pixel_size is None or self.matched_q is None:
            raise RuntimeError("No calibration available. Call calibrate() first.")

        r = self.matched_pixel_positions
        q_pred = self.pixel_size * r
        residuals = self.matched_q - q_pred

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # --- Left: linear fit ---
        ax1.scatter(r, self.matched_q, zorder=3, label="Matched peaks")
        r_fit = np.linspace(0, r.max() * 1.1, 100)
        ax1.plot(
            r_fit, self.pixel_size * r_fit, "--", label=f"Fit: {self.pixel_size:.6f} 1/\u00c5/px"
        )
        for i, hkl in enumerate(self.matched_hkl):
            ax1.annotate(hkl, (r[i], self.matched_q[i]), textcoords="offset points", xytext=(5, 5))
        ax1.set_xlabel("Radial position (pixels)")
        ax1.set_ylabel("q (1/\u00c5)")
        ax1.set_title("Calibration fit")
        ax1.legend()

        # --- Right: residuals ---
        ax2.stem(r, residuals, markerfmt="o", basefmt="k-")
        for i, hkl in enumerate(self.matched_hkl):
            ax2.annotate(
                hkl, (r[i], residuals[i]), textcoords="offset points", xytext=(5, 5), fontsize=8
            )
        ax2.axhline(0, color="gray", linewidth=0.5)
        ax2.set_xlabel("Radial position (pixels)")
        ax2.set_ylabel("\u0394q (1/\u00c5)")
        ax2.set_title("Residuals (measured - predicted)")

        fig.tight_layout()
        return fig
