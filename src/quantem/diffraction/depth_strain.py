"""Depth-resolved strain from 4DSTEM qx plots by inverting dynamical diffraction.

A systematic-row CBED profile recorded at every probe position along a line (a
"qx plot", intensity versus diffraction coordinate q and position x) is fitted
with a dynamical forward model whose unknown is the depth derivative of the
displacement along g, eps(x, z) = d u_g / d z, on a coarse x-z grid. Method of
L. Niermann, T. Niermann, C. Song and C. Ophus, Appl. Phys. Lett. 127, 111902
(2025), https://doi.org/10.1063/5.0274964; the qx plot follows T. Niermann,
L. Niermann and M. Lehmann, Nat. Commun. 15, 1356 (2024).

Forward model: the Darwin-Howie-Whelan equations in the column approximation
for the beams n g of one systematic row,

    d phi_n / dz = i pi lambda sum_m A_nm phi_m + 2 pi i (K + n g) eps(x, z) phi_n
    A = U_(n-m) - diag(n g (n g + 2 K)) + i U0''

with one column per probe position x and incident tilt K along g. The K eps
part of the strain term is the same for every beam, so it only matters where
disks overlap and are summed coherently. The strain term is diagonal, so the z
integration is a Strang split step: one fixed propagator exp(i pi lambda h A(K))
per tilt and a phase per beam and slice. The exit amplitudes are placed in
their CBED disks with soft aperture edges, summed coherently or incoherently,
and compared with the measured qx plot by weighted least squares with a
closed-form intensity scale (global, or per probe position) and a penalty on
the strain nodes: L2 (the paper), L1 or total variation. Gradients come from
torch autograd; scipy BFGS does the fit, or a torch optimizer with a
soft-threshold (shrinkage) step after each update. ``l_curve`` picks the
penalty weight from the data: the L-curve corner, or the best prediction of an
independent half set of the scan.

Workflow from 4DSTEM data: ``from_dataset4dstem`` -> ``set_crystal`` ->
``fit_geometry`` -> ``extract_qx`` -> ``set_model`` -> ``reconstruct`` (or
``l_curve``) -> ``plot_*``. From an extracted qx plot, ``from_qx`` replaces the
first and the extraction steps. Optional: ``fit_calibration``,
``set_strain_grid`` (coarse to fine) and ``align_twins``.

Units are Angstroms and 1/Angstroms (no 2 pi); the beam energy is in eV.

Limitations: one systematic row and one strain component, column
approximation, no probe blur along x. Mirroring the field about the midplane,
eps(x, z) -> eps(x, t - z), gives the same intensities, so every fit has a
twin (``plot_strain(mirror=True)``); in the column approximation this holds
column by column, which can fold an inclined feature at the midplane
(``align_twins``).
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage, optimize
from scipy.signal import fftconvolve
from tqdm.auto import tqdm

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.io.serialize import AutoSerialize
from quantem.core.ml.optimizer_mixin import OptimizerMixin
from quantem.diffraction.bloch import _coupling_matrix, relativistic_gamma
from quantem.diffraction.crystal import Crystal
from quantem.diffraction.disk_detection import estimate_central_beam
from quantem.diffraction.reverse_monte_carlo import ReverseMonteCarlo, electron_wavelength

# optimizer variable = parameter * scale, so that a unit step is a typical change
_SCALES = dict(
    strain=1e3,
    thickness=0.1,
    tilt=100.0,
    shear=1e4,
    q_offset=10.0,
    q_scale=1e3,
    x_offset=0.1,
    semiconvergence=100.0,
)


def _length_to_angstrom(value: float, units: str) -> float:
    u = str(units).lower()
    if u in ("nm", "nanometer", "nanometers"):
        return float(value) * 10.0
    if u in ("1/nm", "nm^-1", "nm-1"):
        return float(value) * 0.1
    return float(value)


def _hat(coord: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
    """Linear-interpolation weights (len(coord), len(nodes)), held constant outside the nodes."""
    if nodes.numel() == 1:
        return torch.ones((coord.numel(), 1), dtype=coord.dtype, device=coord.device)
    step = nodes[1] - nodes[0]
    c = coord.clamp(nodes[0], nodes[-1])
    return torch.relu(1.0 - (c[:, None] - nodes[None, :]).abs() / step)


class DepthStrain(AutoSerialize, OptimizerMixin):
    """Depth-resolved strain eps(x, z) from a qx plot by inverting dynamical diffraction.

    Notes
    -----
    Workflow:

    1. ``from_dataset4dstem``: average the 4DSTEM patterns along the scan
       direction parallel to the feature, giving one CBED pattern per x; or
       ``from_qx`` with an extracted qx plot.
    2. ``set_crystal``: crystal and systematic-row reflection g.
    3. ``fit_geometry``: direct beam, disk radius, row direction and q sampling
       (from |g|), and the descan of the direct beam along x.
    4. ``extract_qx``: the profile along the row through every pattern.
    5. ``set_model``: thickness, tilt, convergence, strain grid and sampling.
    6. ``fit_calibration`` (optional): thickness, tilt, ... with the strain
       fixed, on a strain-free window.
    7. ``reconstruct``: the strain nodes, or ``l_curve`` to choose the penalty
       weight as well. ``set_strain_grid`` refines the nodes between fits, and
       ``align_twins`` unfolds a feature folded at the midplane.
    8. ``plot_geometry``, ``plot_qx``, ``plot_fit``, ``plot_strain``,
       ``plot_loss``, ``plot_l_curve``.

    Parameters live in ``params`` (torch tensors): ``strain`` (nz, nx) node
    values, ``thickness`` (A), ``tilt`` (incident tilt along g at the disk
    center, 1/A), ``shear`` (disk shift along q per unit x, 1/A^2: descan and
    pivot error), ``q_offset`` and ``q_scale`` (q of data column 0 and per
    column, 1/A), ``x_offset`` (x of data row 0, A) and ``semiconvergence``
    (aperture radius, 1/A).
    """

    _token = object()

    def __init__(
        self,
        data,
        x_sampling,
        q_sampling,
        x_origin,
        q_origin,
        energy,
        stack=None,
        _token=None,
    ):
        if _token is not self._token:
            raise RuntimeError("Use DepthStrain.from_dataset4dstem() or DepthStrain.from_qx().")
        self.stack = stack
        self.data = None if data is None else np.asarray(data, dtype=np.float64)
        self.x_sampling = float(x_sampling)
        self.x_origin = float(x_origin)
        self.q_sampling = None if q_sampling is None else float(q_sampling)
        self.q_origin = None if q_origin is None else float(q_origin)
        self.energy = float(energy)
        self.wavelength = electron_wavelength(self.energy)
        self.crystal: Crystal | None = None
        self.g_hkl: tuple[int, int, int] | None = None
        self.g: float | None = None
        self.geometry: dict | None = None
        self.model: dict | None = None
        self.params: dict | None = None
        self.result: dict | None = None
        self.loss_history: list[float] = []
        self.l_curve_result: dict | None = None
        OptimizerMixin.__init__(self)
        self._opt_vars: torch.Tensor | None = None

    # optimizer state of OptimizerMixin, rebuilt by every torch fit: not saved
    _OPTIMIZER_STATE = ("_optimizer", "_scheduler", "_optimizer_params", "_scheduler_params")

    def save(self, path, mode="w", store="auto", skip=(), compression_level=4) -> None:
        """Save with ``AutoSerialize`` (zarr), without the optimizer state.

        Pass ``skip=("stack",)`` to leave out the CBED patterns.
        """
        skip = (skip,) if isinstance(skip, str) else tuple(skip)
        super().save(path, mode, store, skip + self._OPTIMIZER_STATE, compression_level)

    def _post_load(self) -> None:
        """AutoSerialize hook: restore the empty optimizer state."""
        if not hasattr(self, "_optimizer_params"):
            OptimizerMixin.__init__(self)

    # -------------------------------------------------------------------- input

    @classmethod
    def from_dataset4dstem(
        cls,
        dataset,
        energy: float,
        average_axis: int = 1,
        x_sampling: float | None = None,
        average_index=None,
    ) -> "DepthStrain":
        """CBED patterns along x from a 4DSTEM scan, averaged along the other scan axis.

        Parameters
        ----------
        dataset : Dataset4dstem or ndarray
            (scan_0, scan_1, ky, kx) data.
        energy : float
            Beam energy in eV.
        average_axis : {0, 1}
            Scan axis averaged out (the direction parallel to the feature); the
            other scan axis becomes x.
        x_sampling : float, optional
            Scan step along x in A. Read from the dataset (nm converted) when not
            given.
        average_index : slice or sequence of int, optional
            Positions along ``average_axis`` that are averaged, e.g.
            ``slice(0, None, 2)`` and ``slice(1, None, 2)`` for two independent
            half sets (see ``l_curve(validation=...)``). Default all.
        """
        if average_axis not in (0, 1):
            raise ValueError("average_axis is 0 or 1")
        arr = dataset.array if hasattr(dataset, "array") else dataset
        keep = 1 - average_axis
        if x_sampling is None:
            if not hasattr(dataset, "sampling"):
                raise ValueError("x_sampling (A per scan step) is required for plain arrays.")
            x_sampling = _length_to_angstrom(
                float(np.asarray(dataset.sampling)[keep]), dataset.units[keep]
            )
        if average_index is not None:
            sel = [slice(None)] * 4
            sel[average_axis] = average_index
            arr = arr[tuple(sel)]
        stack = np.asarray(arr, dtype=np.float32).mean(axis=average_axis, dtype=np.float64)
        stack = stack.astype(np.float32)
        return cls(None, x_sampling, None, 0.0, None, energy, stack=stack, _token=cls._token)

    @classmethod
    def from_qx(
        cls,
        data,
        x_sampling: float | None = None,
        q_sampling: float | None = None,
        x_origin: float | None = None,
        q_origin: float | None = None,
        energy: float = 300e3,
    ) -> "DepthStrain":
        """An extracted qx plot, intensity (x, q).

        Parameters
        ----------
        data : Dataset2d or ndarray
            (n_x, n_q) qx plot. A Dataset2d supplies sampling, origin and units
            (nm and 1/nm are converted) for arguments that are not given.
        x_sampling, x_origin : float
            Position step and position of row 0, in A.
        q_sampling, q_origin : float
            Diffraction step and q of column 0, in 1/A (q = 0 at the direct beam).
        energy : float
            Beam energy in eV.
        """
        if hasattr(data, "array"):
            s, o, u = np.asarray(data.sampling), np.asarray(data.origin), data.units
            x_sampling = _length_to_angstrom(s[0], u[0]) if x_sampling is None else x_sampling
            x_origin = _length_to_angstrom(o[0], u[0]) if x_origin is None else x_origin
            q_sampling = _length_to_angstrom(s[1], u[1]) if q_sampling is None else q_sampling
            q_origin = _length_to_angstrom(o[1], u[1]) if q_origin is None else q_origin
            data = data.array
        if x_sampling is None or q_sampling is None:
            raise ValueError("x_sampling and q_sampling are required for plain arrays.")
        data = np.asarray(data, dtype=np.float64)
        if x_origin is None:
            x_origin = -0.5 * (data.shape[0] - 1) * x_sampling
        if q_origin is None:
            q_origin = -0.5 * (data.shape[1] - 1) * q_sampling
        return cls(data, x_sampling, q_sampling, x_origin, q_origin, energy, _token=cls._token)

    def set_crystal(
        self,
        crystal: Crystal | None = None,
        g_hkl: Sequence[int] = (0, 0, 1),
        cif_file: str | None = None,
        thermal_sigma: float | dict[str, float] = 0.05,
    ) -> "DepthStrain":
        """Crystal and the systematic-row reflection g.

        Parameters
        ----------
        crystal : Crystal, optional
            The matrix crystal (or give ``cif_file``).
        g_hkl : sequence of int
            The row's first reflection, (hkl) or Miller-Bravais (hkil); +q in
            the qx plot points along +g.
        cif_file : str, optional
            CIF of the crystal, when ``crystal`` is not given.
        thermal_sigma : float or dict
            RMS thermal displacement (A) for the absorptive structure factors.
        """
        if crystal is None:
            if cif_file is None:
                raise ValueError("give crystal or cif_file")
            crystal = Crystal.from_cif(cif_file)
        g_hkl = [int(v) for v in g_hkl]
        if len(g_hkl) == 4:
            g_hkl = [g_hkl[0], g_hkl[1], g_hkl[3]]
        self.crystal = crystal
        self.g_hkl = tuple(g_hkl)
        self.thermal_sigma = thermal_sigma
        g_vec = torch.as_tensor(g_hkl, dtype=torch.float64) @ crystal.lat_recip.to(torch.float64)
        self.g = float(torch.linalg.norm(g_vec))
        print(
            f"g = ({' '.join(str(v) for v in g_hkl)}): |g| = {self.g:.4f} 1/A, d = {1 / self.g:.4f} A"
        )
        return self

    # ----------------------------------------------------------------- geometry

    @staticmethod
    def _profile(image, center, direction, t, width: float) -> np.ndarray:
        """Profile of ``image`` along ``direction`` through ``center`` at offsets ``t`` (px),
        averaged over ``width`` px across it (bilinear interpolation)."""
        perp = np.array([-direction[1], direction[0]])
        n = max(int(round(width)), 1)
        s = np.arange(n) - 0.5 * (n - 1)
        rows = center[0] + t[:, None] * direction[0] + s[None] * perp[0]
        cols = center[1] + t[:, None] * direction[1] + s[None] * perp[1]
        vals = ndimage.map_coordinates(
            image, [rows.ravel(), cols.ravel()], order=1, mode="nearest"
        )
        return vals.reshape(rows.shape).mean(1)

    @staticmethod
    def _edge(t, slope, guess: float, window: float, sign: int) -> float:
        """Sub-pixel position of the steepest rising (sign 1) or falling (-1) edge near guess."""
        sel = np.flatnonzero(np.abs(t - guess) <= window)
        j = sel[np.argmax(sign * slope[sel])]
        if 0 < j < len(t) - 1:
            lo, mid, hi = sign * slope[j - 1], sign * slope[j], sign * slope[j + 1]
            den = lo - 2 * mid + hi
            if den < 0:
                return float(t[j] + np.clip(0.5 * (lo - hi) / den, -1, 1) * (t[1] - t[0]))
        return float(t[j])

    def fit_geometry(
        self,
        threshold: float = 0.5,
        n_beams: int = 2,
        centering: bool = True,
        width: float | None = None,
        sigma: float = 1.5,
        verbose: bool = True,
    ) -> "DepthStrain":
        """Direct beam, disk radius, row direction, q sampling and descan.

        The direct beam of the mean pattern comes from ``estimate_central_beam``.
        Candidate disks are the peaks (``ReverseMonteCarlo._find_peaks``) of the
        high-passed mean pattern correlated with a disk of that radius, and the
        systematic row is the neighbouring pair of candidates, opposite each
        other across the direct beam, with the strongest correlation, which
        gives a first row direction. Along the row the problem is
        one-dimensional: in a profile through the direct beam its two edges (the
        steepest rise and fall of the smoothed profile) give the center and the
        radius R. With ``centering`` this is done for every pattern, along and
        across the row, and a robust straight line along x models the descan.
        The profiles, aligned on those centers and averaged, have rising (and
        falling) disk edges every |g|, so the autocorrelation of the edge
        profile peaks at |g| in pixels; the direction is refined (within 5
        degrees) to make that peak sharpest. This gives the q sampling
        |g| / px and the semiconvergence R |g| / px.

        Parameters
        ----------
        threshold : float
            Fraction of the peak intensity that isolates the direct disk.
        n_beams : int
            Disks on each side of the direct beam in the profiles used for the
            spacing.
        centering : bool
            Measure and fit the direct-beam position of every pattern.
        width : float, optional
            Profile width in pixels. Default half the disk radius.
        sigma : float
            Smoothing (px) of the profiles before the edges are located.
        verbose : bool
            Print the result.
        """
        if self.stack is None:
            raise RuntimeError("fit_geometry needs CBED patterns (from_dataset4dstem).")
        if self.g is None:
            raise RuntimeError("set_crystal first")
        mean = self.stack.mean(0).astype(np.float64)
        (r0, c0), radius = estimate_central_beam(torch.as_tensor(mean), threshold=threshold)
        center = np.array([r0, c0])
        width = 0.5 * radius if width is None else width

        # row direction from candidate disks: disk template on the high-passed pattern
        m = int(np.ceil(radius)) + 2
        yy, xx = np.mgrid[-m : m + 1, -m : m + 1]
        disk = (np.hypot(yy, xx) <= radius).astype(np.float64)
        flat = mean - ndimage.gaussian_filter(mean, 2 * radius)
        corr = fftconvolve(flat, (disk - disk.mean())[::-1, ::-1], mode="same")
        pts, _, _ = ReverseMonteCarlo._find_peaks(corr, n_peaks=8 * n_beams + 12)
        rel = pts - center
        dist = np.hypot(*rel.T)
        near = np.flatnonzero((dist > 1.5 * radius) & (dist < 3.0 * radius))
        if near.size == 0:
            raise RuntimeError("no neighbouring disk found along the systematic row")

        def value(p):
            r, c = np.clip(np.round(p).astype(int), 0, np.array(corr.shape) - 1)
            return corr[r, c]

        score = [value(center + rel[i]) + value(center - rel[i]) for i in near]
        d = rel[near[int(np.argmax(score))]]
        g_guess = float(np.linalg.norm(d))
        direction = d / g_guess
        if direction[0] < 0 or (abs(direction[0]) < 1e-9 and direction[1] < 0):
            direction = -direction
        perp = np.array([-direction[1], direction[0]])

        def disk_center(image, c):
            """Direct-beam center and radius from its edges along and across the row."""
            t = np.arange(-radius - 15.0, radius + 15.5)
            out, rad = c.copy(), []
            for v in (direction, perp):
                slope = ndimage.gaussian_filter1d(
                    self._profile(image, c, v, t, width), sigma, order=1
                )
                rise = self._edge(t, slope, -radius, 12.0, 1)
                fall = self._edge(t, slope, radius, 12.0, -1)
                out = out + 0.5 * (rise + fall) * v
                rad.append(0.5 * (fall - rise))
            return out, rad[0]

        center, _ = disk_center(mean, center)
        nx = self.stack.shape[0]
        X = np.column_stack([np.ones(nx), np.arange(nx, dtype=np.float64)])
        pred = np.tile(center, (nx, 1))
        geo = dict(direction=direction)
        if centering:
            for _ in range(2):
                meas = np.array([disk_center(self.stack[i], pred[i])[0] for i in range(nx)])
                ok = np.all(np.isfinite(meas), axis=1)
                for _ in range(3):
                    coef, *_ = np.linalg.lstsq(X[ok], meas[ok], rcond=None)
                    res = np.linalg.norm(meas - X @ coef, axis=1)
                    mad = np.median(np.abs(res[ok])) + 1e-9
                    ok = ok & (res < 4.0 * 1.4826 * mad + 0.5)
                pred = X @ coef
            center = pred.mean(0)
            geo.update(centers=meas, centers_fit=pred, centers_used=ok)

        # direction and spacing: the rising and falling disk edges of the aligned profiles
        # repeat every |g|, so their autocorrelation peaks there (sharpest at the right angle)
        reach = (n_beams + 0.5) * g_guess + radius
        t = np.arange(-reach, reach + 0.5)

        def mean_profile(v, step=1):
            return np.mean(
                [self._profile(self.stack[i], pred[i], v, t, width) for i in range(0, nx, step)], 0
            )

        def spacing(prof):
            slope = ndimage.gaussian_filter1d(prof, sigma, order=1)
            ac = 0.0
            for part in (np.clip(slope, 0, None), np.clip(-slope, 0, None)):
                part = part - part.mean()
                ac = ac + np.correlate(part, part, "full")[len(part) - 1 :]
            lag = np.arange(len(ac), dtype=np.float64)
            sel = np.flatnonzero((lag > 1.6 * radius) & (lag < 3.0 * radius))
            j = sel[np.argmax(ac[sel])]
            lo, mid, hi = ac[j - 1], ac[j], ac[j + 1]
            sub = (
                np.clip(0.5 * (lo - hi) / (lo - 2 * mid + hi), -1, 1)
                if lo - 2 * mid + hi < 0
                else 0
            )
            return lag[j] + sub, ac[j] / ac[0]

        def rotate(v, a):
            return np.array(
                [v[0] * np.cos(a) - v[1] * np.sin(a), v[0] * np.sin(a) + v[1] * np.cos(a)]
            )

        angles = np.deg2rad(np.arange(-5.0, 5.01, 0.25))
        sharp = [
            spacing(mean_profile(rotate(direction, a), step=max(nx // 60, 1)))[1] for a in angles
        ]
        direction = rotate(direction, angles[int(np.argmax(sharp))])
        perp = np.array([-direction[1], direction[0]])
        prof = mean_profile(direction)
        g_px, _ = spacing(prof)
        slope = ndimage.gaussian_filter1d(prof, sigma, order=1)
        rise = self._edge(t, slope, -radius, 12.0, 1)
        fall = self._edge(t, slope, radius, 12.0, -1)
        radius = 0.5 * (fall - rise)
        n_idx = np.arange(-n_beams, n_beams + 1)
        q_sampling = self.g / g_px
        geo.update(
            direction=direction,
            center=center,
            g_px=float(g_px),
            radius_px=float(radius),
            q_sampling=float(q_sampling),
            semiconvergence=float(radius * q_sampling),
            disks=center[None] + (n_idx * g_px)[:, None] * direction[None],
            disk_index=n_idx,
            profile=(t, prof),
            mean=mean,
        )
        self.geometry = geo
        if verbose:
            if centering:
                drift = (pred[-1] - pred[0]) @ direction
                across = (pred[-1] - pred[0]) @ perp
                print(
                    f"descan over {nx} patterns: {drift:.2f} px along g, {across:.2f} px "
                    f"across, rms scatter {np.sqrt(np.mean(res[ok] ** 2)):.2f} px"
                )
            print(
                f"direct beam ({center[0]:.1f}, {center[1]:.1f}) px, disk radius {radius:.2f} px, "
                f"|g| = {g_px:.2f} px, "
                f"{q_sampling:.6f} 1/A per px, semiconvergence {geo['semiconvergence']:.4f} 1/A "
                f"({1e3 * geo['semiconvergence'] * self.wavelength:.2f} mrad)"
            )
        return self

    def extract_qx(
        self,
        width: float = 9.0,
        q_range: tuple[float, float] | None = None,
        x_range: tuple[int, int] | None = None,
        x_origin: float | None = None,
        centering: bool = True,
        flip: bool = False,
        geometry: dict | None = None,
    ) -> "DepthStrain":
        """The qx plot: a profile along the systematic row through every pattern.

        Each profile runs through that pattern's direct beam (the straight-line
        fit of ``fit_geometry`` when ``centering``, else the mean center) along
        the row, sampled every detector pixel and averaged over ``width`` pixels
        across the row (bilinear interpolation). q = 0 is the direct beam.

        Parameters
        ----------
        width : float
            Width across the row, in pixels.
        q_range : (float, float), optional
            q range in 1/A. Default: as far along the row as the detector allows.
        x_range : (int, int), optional
            Patterns kept, as a slice (start, stop) of the scan positions.
        x_origin : float, optional
            x of the first kept pattern, A. Default: x = 0 in the middle.
        centering : bool
            Follow the fitted direct-beam position (descan) along x.
        flip : bool
            Reverse +q (when +g points the other way on the detector).
        geometry : dict, optional
            The ``geometry`` of another DepthStrain of the same scan (e.g. the
            full data set, when this one is a half set), used instead of
            ``fit_geometry`` here, so that both qx plots share one calibration.
        """
        if geometry is not None:
            self.geometry = copy.deepcopy(geometry)
        geo = self.geometry
        if geo is None:
            raise RuntimeError("fit_geometry first")
        d = geo["direction"] * (-1.0 if flip else 1.0)
        perp = np.array([-d[1], d[0]])
        ny, nx_det = self.stack.shape[1:]
        if centering and "centers_fit" in geo:
            centers = np.asarray(geo["centers_fit"])
        else:
            centers = np.tile(geo["center"], (self.stack.shape[0], 1))
        i0, i1 = (0, self.stack.shape[0]) if x_range is None else x_range
        sel = np.arange(i0, i1)
        if q_range is None:
            # longest run of the row inside the detector for every pattern
            t = np.arange(-max(ny, nx_det), max(ny, nx_det) + 1, dtype=np.float64)
            inside = np.ones_like(t, dtype=bool)
            for c in centers[sel][[0, -1]]:
                p = c[None] + t[:, None] * d[None]
                inside &= (p[:, 0] >= 0) & (p[:, 0] <= ny - 1) & (p[:, 1] >= 0)
                inside &= p[:, 1] <= nx_det - 1
            t = t[inside]
        else:
            px = geo["q_sampling"]
            t = np.arange(np.ceil(q_range[0] / px), np.floor(q_range[1] / px) + 1)
        s = np.arange(int(round(width))) - 0.5 * (int(round(width)) - 1)
        rows = centers[sel, 0][:, None, None] + t[None, :, None] * d[0] + s[None, None] * perp[0]
        cols = centers[sel, 1][:, None, None] + t[None, :, None] * d[1] + s[None, None] * perp[1]
        frames = np.broadcast_to(sel[:, None, None], rows.shape).astype(np.float64)
        vals = ndimage.map_coordinates(
            self.stack, [frames.ravel(), rows.ravel(), cols.ravel()], order=1, cval=np.nan
        )
        self.data = vals.reshape(rows.shape).mean(-1)
        self.q_sampling = float(geo["q_sampling"])
        self.q_origin = float(t[0] * self.q_sampling)
        n = len(sel)
        self.x_origin = -0.5 * (n - 1) * self.x_sampling if x_origin is None else float(x_origin)
        geo.update(
            extract=dict(width=width, t=t, x_range=(i0, i1), flip=flip, centering=centering)
        )
        print(
            f"qx plot: {n} positions x {len(t)} q, q from {self.q_origin:.4f} to "
            f"{self.q_origin + (len(t) - 1) * self.q_sampling:.4f} 1/A"
        )
        return self

    # -------------------------------------------------------------------- model

    def set_model(
        self,
        thickness: float,
        tilt: float = 0.0,
        semiconvergence: float | None = None,
        shear: float = 0.0,
        strain_shape: tuple[int, int] = (9, 15),
        strain_x_range: tuple[float, float] | None = None,
        strain_z_range: tuple[float, float] | None = None,
        strain_gradient: tuple[float, float] = (0.0, 0.0),
        x_rows: tuple[int, int] | None = None,
        x_step: float = 20.0,
        x_range: tuple[float, float] | None = None,
        tilt_range: tuple[float, float] | None = None,
        tilt_step: float | None = None,
        dz: float = 5.0,
        coherent: bool = True,
        feather: float = 0.0,
        aperture_edge: float = 0.005,
        weight_edge: float = 0.01,
        g_max: float = 3.0,
        s_max: float = 0.05,
        scale: str = "global",
        device: str = "cpu",
    ) -> "DepthStrain":
        """Forward model, sampling, strain grid and the starting parameters.

        Parameters
        ----------
        thickness : float
            Foil thickness, A.
        tilt : float
            Incident tilt along g at the disk center, 1/A (residual tilt).
        semiconvergence : float, optional
            Aperture radius, 1/A. Default from ``fit_geometry``.
        shear : float
            Disk shift along q per unit x, 1/A^2 (descan or pivot error left
            after ``extract_qx`` centering).
        strain_shape : (int, int)
            Strain nodes (n_z, n_x).
        strain_x_range, strain_z_range : (float, float), optional
            Node extent in A. Default: the fitted x range and (0, thickness).
        strain_gradient : (float, float)
            Fixed linear background d eps / dx and d eps / dz (1/A), added to the
            fitted field.
        x_rows : (int, int), optional
            Data rows fitted, as a slice. Default all.
        x_step : float
            Position step of the simulated columns, A; the model is interpolated
            linearly onto the data positions.
        x_range : (float, float), optional
            Range of the simulated columns, A. Default: the fitted data rows.
        tilt_range : (float, float), optional
            Incident tilts simulated, 1/A. Default: the aperture about ``tilt``
            plus a margin of 0.05 1/A.
        tilt_step : float, optional
            Tilt step, 1/A. Default the data q sampling.
        dz : float
            Target slice thickness, A (the slice count is fixed here; the slice
            thickness follows the fitted thickness). The propagator of each slice
            is exact and only the strain phase is split (second order), so slices
            much thicker than a Runge-Kutta step work: in the three examples of
            the paper, 5 A stays within 0.1 % of the converged loss.
        coherent : bool
            Sum the beams coherently where disks overlap, else add intensities.
        feather : float
            The fit weights fall off this far (1/A) inside the aperture edge.
        aperture_edge, weight_edge : float
            Softness (1/A) of the model aperture and of the fit weights.
        g_max, s_max : float
            Beams n g with |n g| <= g_max and |s_n| <= s_max at zero tilt (1/A).
        scale : {"global", "row"}
            Intensity scale of the model, in closed form: one factor for the
            whole qx plot, or one per probe position, which absorbs probe current
            (dose) fluctuations between positions.
        device : str
            Torch device (float64).
        """
        if self.data is None:
            raise RuntimeError("no qx plot: extract_qx or from_qx first")
        if self.crystal is None:
            raise RuntimeError("set_crystal first")
        if semiconvergence is None:
            if self.geometry is None:
                raise ValueError("semiconvergence is required without fit_geometry")
            semiconvergence = self.geometry["semiconvergence"]
        if scale not in ("global", "row"):
            raise ValueError("scale is 'global' or 'row'")
        dev = torch.device(device)
        f64 = dict(dtype=torch.float64, device=dev)

        # beams of the systematic row
        g, lam = self.g, self.wavelength
        n_max = int(np.floor(g_max / g))
        n = np.arange(-n_max, n_max + 1)
        n = n[0.5 * lam * (n * g) ** 2 <= s_max]
        hkl = torch.as_tensor(np.outer(n, self.g_hkl), dtype=torch.long)
        crystal = self.crystal
        k_need = 2 * np.abs(n).max() * g * 1.01
        if (
            getattr(crystal, "U_dyn", None) is None
            or getattr(crystal, "dyn_energy_ev", None) != self.energy
            or getattr(crystal, "dyn_k_max", 0.0) < k_need
        ):
            crystal.calculate_dynamical_structure_factors(
                self.energy, thermal_sigma=self.thermal_sigma, k_max=k_need
            )
        U, u0_imag, _ = _coupling_matrix(crystal, hkl, relativistic_gamma(self.energy))

        # data rows and simulated columns
        n_rows = self.data.shape[0]
        r0, r1 = (0, n_rows) if x_rows is None else x_rows
        rows = np.arange(r0, r1)
        x_data = self.x_origin + rows * self.x_sampling
        if x_range is None:
            x_range = (x_data.min(), x_data.max())
        x0 = np.floor(x_range[0] / x_step) * x_step
        x1 = np.ceil(x_range[1] / x_step) * x_step
        x_sim = np.arange(x0, x1 + 0.5 * x_step, x_step)
        if tilt_step is None:
            tilt_step = self.q_sampling
        if tilt_range is None:
            reach = semiconvergence + 0.05
            tilt_range = (tilt - reach, tilt + reach)
        K = np.arange(tilt_range[0], tilt_range[1] + 0.5 * tilt_step, tilt_step)

        # strain nodes
        if strain_x_range is None:
            strain_x_range = (x_data.min(), x_data.max())
        if strain_z_range is None:
            strain_z_range = (0.0, thickness)
        z_nodes = torch.linspace(*strain_z_range, strain_shape[0], **f64)
        x_nodes = torch.linspace(*strain_x_range, strain_shape[1], **f64)
        x_sim_t = torch.as_tensor(x_sim, **f64)

        data = np.nan_to_num(self.data, nan=0.0)
        valid = np.isfinite(self.data).astype(np.float64)
        self.model = dict(
            n=torch.as_tensor(n, device=dev),
            gn=torch.as_tensor(n * g, **f64),
            i0=int(np.flatnonzero(n == 0)[0]),
            U=U.to(dev),
            u0_imag=float(u0_imag),
            x_sim=x_sim_t,
            K=torch.as_tensor(K, **f64),
            n_slices=max(int(round(thickness / dz)), 1),
            z_nodes=z_nodes,
            x_nodes=x_nodes,
            wx=_hat(x_sim_t, x_nodes).T.contiguous(),
            grad_x=float(strain_gradient[0]),
            grad_z=float(strain_gradient[1]),
            rows=torch.as_tensor(rows, device=dev),
            x_index=torch.arange(n_rows, **f64),
            q_index=torch.arange(self.data.shape[1], **f64),
            data=torch.as_tensor(data, **f64),
            valid=torch.as_tensor(valid, **f64),
            coherent=bool(coherent),
            feather=float(feather),
            aperture_edge=float(aperture_edge),
            weight_edge=float(weight_edge),
            scale=scale,
            device=str(dev),
        )
        self.params = dict(
            strain=torch.zeros(strain_shape, **f64),
            thickness=torch.tensor(float(thickness), **f64),
            tilt=torch.tensor(float(tilt), **f64),
            shear=torch.tensor(float(shear), **f64),
            q_offset=torch.tensor(float(self.q_origin), **f64),
            q_scale=torch.tensor(float(self.q_sampling), **f64),
            x_offset=torch.tensor(float(self.x_origin), **f64),
            semiconvergence=torch.tensor(float(semiconvergence), **f64),
        )
        self.loss_history = []
        self._set_weights()
        print(
            f"{len(n)} beams (n = {n.min()}..{n.max()}), {len(x_sim)} columns x {len(K)} tilts x "
            f"{self.model['n_slices']} slices, {len(rows)} data rows, "
            f"{strain_shape[0]} x {strain_shape[1]} strain nodes"
        )
        return self

    def set_strain_grid(
        self,
        strain_shape: tuple[int, int],
        strain_x_range: tuple[float, float] | None = None,
        strain_z_range: tuple[float, float] | None = None,
    ) -> "DepthStrain":
        """Change the strain nodes, keeping the current field (linear interpolation).

        For coarse-to-fine fits: fit a coarse grid, then refine it here and fit
        again. The extents default to the current ones.
        """
        m, p = self.model, self.params
        f64 = dict(dtype=torch.float64, device=p["strain"].device)
        zr = (float(m["z_nodes"][0]), float(m["z_nodes"][-1]))
        xr = (float(m["x_nodes"][0]), float(m["x_nodes"][-1]))
        z_nodes = torch.linspace(*(strain_z_range or zr), strain_shape[0], **f64)
        x_nodes = torch.linspace(*(strain_x_range or xr), strain_shape[1], **f64)
        with torch.no_grad():
            strain = _hat(z_nodes, m["z_nodes"]) @ p["strain"] @ _hat(x_nodes, m["x_nodes"]).T
        m.update(z_nodes=z_nodes, x_nodes=x_nodes, wx=_hat(m["x_sim"], x_nodes).T.contiguous())
        p["strain"] = strain
        print(f"{strain_shape[0]} x {strain_shape[1]} strain nodes")
        return self

    def _strain_field(self, p: dict, h: torch.Tensor) -> torch.Tensor:
        """eps at the slice midpoints of every simulated column, (n_slices, n_columns)."""
        m = self.model
        z = (torch.arange(m["n_slices"], dtype=h.dtype, device=h.device) + 0.5) * h
        eps = _hat(z, m["z_nodes"]) @ p["strain"] @ m["wx"]
        return eps + m["grad_x"] * m["x_sim"][None, :] + m["grad_z"] * z[:, None]

    def _propagate(self, p: dict) -> torch.Tensor:
        """Exit amplitudes (n_tilts, n_beams, n_columns) by Strang split steps through z."""
        m = self.model
        ns = m["n_slices"]
        h = p["thickness"] / ns
        K, gn = m["K"], m["gn"]
        diag = -gn[None, :] * (gn[None, :] + 2 * K[:, None]) + 1j * m["u0_imag"]
        A = m["U"][None] + torch.diag_embed(diag)
        e_half = torch.linalg.matrix_exp((0.5j * np.pi * self.wavelength) * h * A)
        e_full = e_half @ e_half
        eps = self._strain_field(p, h)
        # the strain phase 2 pi h (K + n g) eps of a slice is a beam phase 2 pi h n g eps
        # times a phase 2 pi h K eps common to all beams, which commutes with the
        # propagators: it adds up to 2 pi K u at the exit, u = int eps dz the displacement
        beam_phase = torch.exp((2j * np.pi) * h * gn[None, :, None] * eps[:, None, :])
        phi = e_half[:, :, m["i0"], None].expand(-1, -1, eps.shape[1])
        for k in range(ns):
            phi = torch.bmm(e_full if k < ns - 1 else e_half, phi * beam_phase[k])
        u = h * eps.sum(0)
        return phi * torch.exp((2j * np.pi) * K[:, None, None] * u[None, None, :])

    def _kappa(self, p: dict, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Position x of the rows and offset kappa of every data pixel from each disk center."""
        m = self.model
        x = p["x_offset"] + m["x_index"][rows] * self.x_sampling
        q = p["q_offset"] + m["q_index"] * p["q_scale"]
        return x, q[None, None, :] - m["gn"][:, None, None] + p["shear"] * x[None, :, None]

    def _set_weights(self) -> None:
        """Fit weights (n_rows, n_q) from the current calibration: inside the disks, falling
        off ``feather`` inside the aperture edge. Held fixed during a fit, as the loss could
        otherwise shrink them."""
        m, p = self.model, self.params
        rows = torch.arange(self.data.shape[0], device=m["rows"].device)
        with torch.no_grad():
            _, kappa = self._kappa(p, rows)
            rim, wt = p["semiconvergence"] - m["feather"], m["weight_edge"]
            w = (torch.sigmoid((rim - kappa) / wt) * torch.sigmoid((rim + kappa) / wt)).sum(0)
        m["weight"] = w * m["valid"]

    def _render(self, phi: torch.Tensor, p: dict, rows: torch.Tensor) -> torch.Tensor:
        """Model intensity on the data grid, (n_rows, n_q)."""
        m = self.model
        x, kappa = self._kappa(p, rows)
        xs, K = m["x_sim"], m["K"]
        gx = 2 * (x - xs[0]) / (xs[-1] - xs[0]) - 1 if len(xs) > 1 else torch.zeros_like(x)
        gk = 2 * (kappa + p["tilt"] - K[0]) / (K[-1] - K[0]) - 1
        grid = torch.stack(torch.broadcast_tensors(gx[None, :, None], gk), dim=-1)
        a = phi.permute(1, 0, 2)
        alpha, w = p["semiconvergence"], m["aperture_edge"]
        S = torch.sigmoid((kappa + alpha) / w) * torch.sigmoid((alpha - kappa) / w)
        sample = dict(mode="bilinear", padding_mode="border", align_corners=True)
        if m["coherent"]:
            out = F.grid_sample(torch.stack((a.real, a.imag), dim=1), grid, **sample)
            re, im = (S * out[:, 0]).sum(0), (S * out[:, 1]).sum(0)
            inten = re**2 + im**2
        else:
            out = F.grid_sample((a.real**2 + a.imag**2)[:, None], grid, **sample)
            inten = (S * out[:, 0]).sum(0)
        return inten

    def _loss(self, p: dict, rows: torch.Tensor, reg: float = 0.0, reg_type: str = "l2"):
        m = self.model
        inten = self._render(self._propagate(p), p, rows)
        if "weight" not in m:
            self._set_weights()
        weight = m["weight"][rows]
        wi, wd = weight * inten, weight * m["data"][rows]
        dims = (1,) if m.get("scale", "global") == "row" else (0, 1)
        scale = (wi * wd).sum(dims, keepdim=True) / ((wi * wi).sum(dims, keepdim=True) + 1e-30)
        residual = ((scale * wi - wd) ** 2).sum()
        penalty = torch.zeros((), dtype=residual.dtype, device=residual.device)
        if reg > 0:
            s = p["strain"]
            if reg_type == "l2":
                penalty = reg * (s**2).mean()
            elif reg_type == "l1":
                penalty = reg * s.abs().mean()
            elif reg_type == "tv":
                # gradient in physical units times the depth range of the nodes, so that
                # the weight does not depend on the node spacing
                zn, xn = m["z_nodes"], m["x_nodes"]
                depth = float(zn[-1] - zn[0]) if zn.numel() > 1 else 1.0
                hz = float(zn[1] - zn[0]) if zn.numel() > 1 else 1.0
                hx = float(xn[1] - xn[0]) if xn.numel() > 1 else 1.0
                dz = F.pad(torch.diff(s, dim=0), (0, 0, 0, 1)) * (depth / hz)
                dx = F.pad(torch.diff(s, dim=1), (0, 1, 0, 0)) * (depth / hx)
                penalty = reg * torch.sqrt(dz**2 + dx**2 + 1e-12).mean()
            else:
                raise ValueError("reg_type is 'l2', 'l1' or 'tv'")
        return residual + penalty, dict(residual=residual, penalty=penalty, scale=scale)

    # ---------------------------------------------------------------------- fit

    def _rows_in(self, x_range: tuple[float, float] | None) -> torch.Tensor:
        rows = self.model["rows"]
        if x_range is None:
            return rows
        x = self.params["x_offset"].item() + rows.to(torch.float64) * self.x_sampling
        keep = (x >= x_range[0]) & (x <= x_range[1])
        if not bool(keep.any()):
            raise ValueError("no data rows inside x_range")
        return rows[keep]

    def _fit(self, names, rows, reg, reg_type, max_iter, method, gtol, verbose, label):
        if self.params is None:
            raise RuntimeError("set_model first")
        unknown = [k for k in names if k not in _SCALES]
        if unknown:
            raise ValueError(f"unknown parameters {unknown}; choose from {list(_SCALES)}")
        self._set_weights()
        p0 = {k: v.detach().clone() for k, v in self.params.items()}
        dev = p0["thickness"].device
        sizes = [p0[k].numel() for k in names]
        x0 = np.concatenate([(p0[k] * _SCALES[k]).reshape(-1).cpu().numpy() for k in names])

        def unpack(x: torch.Tensor) -> dict:
            out, i = dict(p0), 0
            for k, s in zip(names, sizes):
                out[k] = (x[i : i + s] / _SCALES[k]).reshape(p0[k].shape)
                i += s
            return out

        with torch.no_grad():
            f0 = float(self._loss(p0, rows, reg, reg_type)[0])
        f0 = f0 if f0 > 0 else 1.0
        state = dict(best=np.inf, x=x0.copy(), n_eval=0)

        def fun(xn):
            x = torch.tensor(xn, dtype=torch.float64, device=dev, requires_grad=True)
            loss, _ = self._loss(unpack(x), rows, reg, reg_type)
            loss.backward()
            f = float(loss.detach())
            state["n_eval"] += 1
            if f < state["best"]:
                state["best"], state["x"] = f, xn.copy()
            return f / f0, x.grad.cpu().numpy() / f0

        bar = tqdm(total=max_iter, desc=label, disable=not verbose)

        def callback(intermediate_result):
            self.loss_history.append(float(intermediate_result.fun) * f0)
            bar.update(1)
            bar.set_postfix(loss=f"{self.loss_history[-1]:.6g}")

        res = optimize.minimize(
            fun,
            x0,
            jac=True,
            method=method,
            callback=callback,
            options=dict(maxiter=max_iter, gtol=gtol),
        )
        bar.close()
        best = unpack(torch.as_tensor(state["x"], dtype=torch.float64, device=dev))
        self.params = {k: v.detach().clone() for k, v in best.items()}
        with torch.no_grad():
            loss, parts = self._loss(self.params, rows, reg, reg_type)
        self.result = dict(
            loss=float(loss),
            residual=float(parts["residual"]),
            penalty=float(parts["penalty"]),
            scale=float(parts["scale"].mean()),
            initial_loss=f0,
            n_iter=int(res.nit),
            n_eval=state["n_eval"],
            message=str(res.message),
            fitted=tuple(names),
        )
        if verbose:
            print(
                f"{label}: loss {f0:.6g} -> {float(loss):.6g} (residual {self.result['residual']:.6g}, "
                f"penalty {self.result['penalty']:.6g}), {res.nit} iterations"
            )
        return self

    def fit_calibration(
        self,
        names: Sequence[str] = ("thickness", "tilt"),
        x_range: tuple[float, float] | None = None,
        max_iter: int = 100,
        method: str = "BFGS",
        gtol: float = 1e-6,
        verbose: bool = True,
    ) -> "DepthStrain":
        """Fit calibration parameters with the strain held fixed.

        Parameters
        ----------
        names : sequence of str
            Any of thickness, tilt, shear, q_offset, q_scale, x_offset,
            semiconvergence.
        x_range : (float, float), optional
            Fit only the data rows in this x window (A), e.g. a strain-free region.
        max_iter, method, gtol
            Passed to ``scipy.optimize.minimize`` (loss normalized to 1 at start).
        """
        names = tuple(k for k in names if k != "strain")
        rows = self._rows_in(x_range)
        self._fit(names, rows, 0.0, "l2", max_iter, method, gtol, verbose, "calibration")
        if verbose:
            print(", ".join(f"{k} {float(self.params[k]):.6g}" for k in names))
        return self

    def reconstruct(
        self,
        reg: float = 0.0,
        reg_type: str = "l2",
        also: Sequence[str] = (),
        max_iter: int = 300,
        method: str = "BFGS",
        gtol: float = 1e-6,
        shrinkage: float = 0.0,
        shrinkage_target: str = "strain",
        optimizer: dict | None = None,
        scheduler: dict | None = None,
        verbose: bool = True,
    ) -> "DepthStrain":
        """Fit the strain nodes.

        ``method`` "BFGS" or "L-BFGS-B" runs ``scipy.optimize.minimize`` (the
        paper's choice). "adam", "adamw" or "sgd" runs a torch optimizer
        (``OptimizerMixin``) on the same variables, which allows the shrinkage
        step after every update.

        Parameters
        ----------
        reg : float
            Penalty weight lambda on mean(eps_nodes^2) ("l2", the paper's
            Tikhonov term), mean(|eps_nodes|) ("l1") or the mean magnitude of
            the strain gradient at the nodes times the depth range of the nodes
            ("tv", total variation). All three are averages over the nodes, so
            a weight carries over to a finer grid (``set_strain_grid``).
        reg_type : {"l2", "l1", "tv"}
            Penalty type.
        also : sequence of str
            Calibration parameters fitted jointly with the strain.
        max_iter : int
            Iterations.
        method : str
            "BFGS", "L-BFGS-B" (scipy) or "adam", "adamw", "sgd" (torch).
        gtol : float
            scipy gradient tolerance (loss normalized to 1 at start).
        shrinkage : float
            Torch methods only: soft threshold applied after every step,
            v <- sign(v) max(|v| - shrinkage, 0), to the strain nodes
            (``shrinkage_target="strain"``, threshold in strain) or to the
            displacement u = int_0^z eps dz' at the nodes ("displacement",
            threshold in A). The threshold follows the learning-rate schedule.
        shrinkage_target : {"strain", "displacement"}
            What is shrunk.
        optimizer : dict, optional
            Torch optimizer, e.g. ``{"name": "adam", "lr": 0.02}``; the
            variables are the strain times 1e3 (and the calibration scaled as
            in ``_SCALES``). Default Adam with lr 0.02.
        scheduler : dict, optional
            Torch scheduler, e.g. ``{"name": "cosine_annealing"}``. Default none.
        """
        names = ("strain",) + tuple(k for k in also if k != "strain")
        rows = self.model["rows"] if self.model is not None else None
        if method.lower() in ("adam", "adamw", "sgd"):
            if optimizer is None:
                optimizer = {"name": method.lower(), "lr": 0.02}
            return self._fit_torch(
                names,
                rows,
                reg,
                reg_type,
                max_iter,
                optimizer,
                scheduler,
                shrinkage,
                shrinkage_target,
                verbose,
            )
        if shrinkage > 0:
            raise ValueError("shrinkage needs a torch method (adam, adamw or sgd)")
        return self._fit(names, rows, reg, reg_type, max_iter, method, gtol, verbose, "strain")

    def l_curve(
        self,
        regs: Sequence[float],
        reg_type: str = "l2",
        max_iter: int = 300,
        validation=None,
        select: str = "corner",
        verbose: bool = True,
    ) -> dict:
        """Pick the penalty weight from the data: the corner of the L-curve.

        The strain is fitted (BFGS) for every weight in ``regs``, from the
        largest down, each fit starting from the previous one. The corner is
        the fit farthest from the chord joining the two ends of the curve of
        log(residual) against log(penalty / weight); it is kept in ``params``.

        With ``validation``, an independent qx plot of the same scan (the other
        half set, see ``from_dataset4dstem(average_index=...)``), every fit is
        also scored against those data. A validation residual that rises again
        toward weak penalties shows where the fit starts to follow the noise;
        fitting of model errors, which both half sets share, or of directions
        the data do not constrain (such as the depth twins) does not show.

        Parameters
        ----------
        regs : sequence of float
            Penalty weights.
        reg_type : {"l2", "l1", "tv"}
            Penalty type.
        max_iter : int
            BFGS iterations per weight.
        validation : DepthStrain or ndarray, optional
            Independent qx plot on the same grid as ``data``.
        select : {"corner", "validation"}
            The fit kept in ``params``: the L-curve corner, or the one with the
            smallest validation residual.

        Returns
        -------
        dict
            "regs", "residual", "norm" (penalty / weight), "corner" (index),
            "strain" (the node values of every fit), and with ``validation``
            "validation" (residual against it) and "best" (index); also stored
            as ``self.l_curve_result``.
        """
        regs = sorted((float(r) for r in regs), reverse=True)
        out = dict(regs=regs, residual=[], norm=[], strain=[])
        if validation is not None:
            other = validation.data if hasattr(validation, "data") else validation
            other = np.asarray(other, dtype=np.float64)
            if other.shape != self.data.shape:
                raise ValueError("validation data must have the shape of data")
            out["validation"] = []
            held = self.model["data"].clone()
        for r in tqdm(regs, desc="L-curve", disable=not verbose):
            self.reconstruct(reg=r, reg_type=reg_type, max_iter=max_iter, verbose=False)
            out["residual"].append(self.result["residual"])
            out["norm"].append(self.result["penalty"] / r)
            out["strain"].append(self.params["strain"].detach().clone())
            if validation is not None:
                self.model["data"] = torch.as_tensor(np.nan_to_num(other)).to(held)
                with torch.no_grad():
                    _, parts = self._loss(self.params, self.model["rows"])
                self.model["data"] = held
                out["validation"].append(float(parts["residual"]))
        u = np.log(np.maximum(out["residual"], 1e-300))
        v = np.log(np.maximum(out["norm"], 1e-300))
        u = (u - u.min()) / max(np.ptp(u), 1e-12)
        v = (v - v.min()) / max(np.ptp(v), 1e-12)
        a = np.array([u[-1] - u[0], v[-1] - v[0]])
        dist = np.abs(a[0] * (v - v[0]) - a[1] * (u - u[0])) / max(np.hypot(*a), 1e-12)
        corner = int(np.argmax(dist))
        out["corner"] = corner
        pick = corner
        if validation is not None:
            out["best"] = int(np.argmin(out["validation"]))
            if select == "validation":
                pick = out["best"]
        elif select == "validation":
            raise ValueError("select='validation' needs validation data")
        self.params["strain"] = out["strain"][pick].clone()
        with torch.no_grad():
            loss, parts = self._loss(self.params, self.model["rows"], regs[pick], reg_type)
        self.result.update(
            loss=float(loss),
            residual=float(parts["residual"]),
            penalty=float(parts["penalty"]),
            reg=regs[pick],
            reg_type=reg_type,
        )
        self.l_curve_result = out
        if verbose:
            print(
                f"L-curve corner: reg = {regs[corner]:.3g} ({reg_type}), residual "
                f"{out['residual'][corner]:.6g}"
            )
            if validation is not None:
                b = out["best"]
                print(
                    f"best validation: reg = {regs[b]:.3g}, validation residual "
                    f"{out['validation'][b]:.6g}"
                )
            print(f"kept: reg = {regs[pick]:.3g}")
        return out

    def plot_l_curve(self, figsize=None):
        """Residual against penalty norm of the last ``l_curve`` scan, corner marked, and the
        validation residual against the weight when the scan had validation data."""
        out = self.l_curve_result
        has_val = "validation" in out
        if figsize is None:
            figsize = (9, 4) if has_val else (4.5, 4)
        fig, axs = plt.subplots(1, 2 if has_val else 1, figsize=figsize, layout="constrained")
        ax = axs[0] if has_val else axs
        ax.loglog(out["norm"], out["residual"], "o-", ms=4)
        c = out["corner"]
        ax.loglog(out["norm"][c], out["residual"][c], "o", ms=10, mfc="none", color="tab:red")
        for r, x, y in zip(out["regs"], out["norm"], out["residual"]):
            ax.annotate(f"{r:.0e}", (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("penalty / weight")
        ax.set_ylabel("residual")
        ax.set_title("L-curve (red: corner)", loc="left", fontsize=10)
        if has_val:
            ax = axs[1]
            ax.semilogx(out["regs"], out["validation"], "o-", ms=4)
            b = out["best"]
            ax.semilogx(
                out["regs"][b], out["validation"][b], "o", ms=10, mfc="none", color="tab:red"
            )
            ax.set_xlabel("penalty weight")
            ax.set_ylabel("validation residual")
            ax.set_title("independent data (red: best)", loc="left", fontsize=10)
        return fig, axs

    def get_optimization_parameters(self) -> dict[str, list[torch.Tensor]]:
        """OptimizerMixin hook: the scaled variables of the current torch fit."""
        return {} if self._opt_vars is None else {self.DEFAULT_OPTIMIZER_KEY: [self._opt_vars]}

    def _shrink(self, strain: torch.Tensor, threshold: float, target: str) -> torch.Tensor:
        """Soft threshold of the strain nodes, or of the displacement they integrate to."""
        if target == "strain":
            return torch.sign(strain) * torch.clamp(strain.abs() - threshold, min=0.0)
        if target != "displacement":
            raise ValueError("shrinkage_target is 'strain' or 'displacement'")
        zn = self.model["z_nodes"]
        dz = float(zn[1] - zn[0]) if zn.numel() > 1 else 1.0
        u = torch.cumsum(strain, dim=0) * dz
        u = torch.sign(u) * torch.clamp(u.abs() - threshold, min=0.0)
        return torch.diff(u, dim=0, prepend=torch.zeros_like(u[:1])) / dz

    def _fit_torch(
        self,
        names,
        rows,
        reg,
        reg_type,
        max_iter,
        optimizer,
        scheduler,
        shrinkage,
        target,
        verbose,
    ):
        if self.params is None:
            raise RuntimeError("set_model first")
        self._set_weights()
        p0 = {k: v.detach().clone() for k, v in self.params.items()}
        sizes = [p0[k].numel() for k in names]
        n_strain = p0["strain"].numel()
        self._opt_vars = torch.cat([(p0[k] * _SCALES[k]).reshape(-1) for k in names]).detach()

        def unpack(x: torch.Tensor) -> dict:
            out, i = dict(p0), 0
            for k, s in zip(names, sizes):
                out[k] = (x[i : i + s] / _SCALES[k]).reshape(p0[k].shape)
                i += s
            return out

        self.set_optimizer(optimizer)
        self.set_scheduler(scheduler or {"name": "none"}, num_iter=max_iter)
        lr0 = self.get_current_lr()
        with torch.no_grad():
            f0 = float(self._loss(p0, rows, reg, reg_type)[0])
        f0 = f0 if f0 > 0 else 1.0
        bar = tqdm(range(max_iter), desc="strain", disable=not verbose)
        for _ in bar:
            self.zero_optimizer_grad()
            loss, _ = self._loss(unpack(self._opt_vars), rows, reg, reg_type)
            (loss / f0).backward()
            self.step_optimizer()
            if shrinkage > 0:
                with torch.no_grad():
                    thr = shrinkage * self.get_current_lr() / lr0
                    x = self._opt_vars[:n_strain].view(p0["strain"].shape) / _SCALES["strain"]
                    x = self._shrink(x, thr, target)
                    self._opt_vars[:n_strain] = x.reshape(-1) * _SCALES["strain"]
            f = float(loss.detach())
            self.step_scheduler(f)
            self.loss_history.append(f)
            bar.set_postfix(loss=f"{self.loss_history[-1]:.6g}")
        self.params = {k: v.detach().clone() for k, v in unpack(self._opt_vars.detach()).items()}
        self._opt_vars = None
        self.remove_optimizer()
        with torch.no_grad():
            loss, parts = self._loss(self.params, rows, reg, reg_type)
        self.result = dict(
            loss=float(loss),
            residual=float(parts["residual"]),
            penalty=float(parts["penalty"]),
            scale=float(parts["scale"].mean()),
            initial_loss=f0,
            n_iter=max_iter,
            n_eval=max_iter,
            message="torch " + str(optimizer),
            fitted=tuple(names),
        )
        if verbose:
            zero = float((self.params["strain"] == 0).double().mean())
            print(
                f"strain: loss {f0:.6g} -> {float(loss):.6g} (residual "
                f"{self.result['residual']:.6g}, penalty {self.result['penalty']:.6g}), "
                f"{100 * zero:.0f}% of the nodes zero"
            )
        return self

    def align_twins(
        self,
        width: float | None = None,
        surface: float = 0.1,
        x_range: tuple[float, float] | None = None,
        slope_range: tuple[float, float] | None = None,
        detrend: bool = True,
        n_grid: int = 121,
        verbose: bool = True,
    ) -> np.ndarray:
        """Choose the depth twin of every column of strain nodes so the strain follows one line.

        In the column approximation each column of the field can be mirrored,
        eps(x, z) -> eps(x, t - z), with almost no change of its qx plot, so a
        fit can mix the two twins column by column. Mixing them where a feature
        is away from the midplane leaves steps in the field, which the penalties
        remove; where it crosses the midplane the twins meet, so one inclined
        layer comes out folded there, as a V or an inverted V.

        Straight lines z(x) are searched on a grid of their depths at the two
        ends of the node range (``n_grid`` each, from -t/2 to 3t/2). A column's
        cost is the second moment of eps^2 about the line, with distances capped
        at ``width`` (default t/4) so that strain far from the line counts the
        same anywhere, for the column or its mirror, whichever is smaller. The
        columns whose mirror fits the best line are mirrored in
        ``params["strain"]``. Refit afterwards (``reconstruct``); the global
        twin (the whole field mirrored) is still undetermined.

        Parameters
        ----------
        width : float, optional
            Distance cap of the cost, A. Default t/4.
        surface : float
            Nodes within this fraction of t of either surface are left out:
            surface strain mirrors onto the other surface and would favour lines
            along the surfaces.
        x_range : (float, float), optional
            Only the node columns inside this x range (A) are used and mirrored.
        slope_range : (float, float), optional
            Limits on |dz/dx| of the line, e.g. from a known inclination of the
            feature.
        detrend : bool
            Remove the median over x of every row of nodes first. Strain that is
            the same at every x cannot be told apart from calibration errors
            (thickness, tilt), and such bands favour horizontal lines.
        n_grid : int
            Line end depths searched.
        verbose : bool
            Print the chosen line.

        Returns
        -------
        np.ndarray
            Boolean mask of the mirrored node columns.
        """
        m, p = self.model, self.params
        t = float(p["thickness"])
        zn, xn = m["z_nodes"], m["x_nodes"]
        s = p["strain"].detach()
        cols = np.ones(len(xn), dtype=bool)
        if x_range is not None:
            xv = xn.cpu().numpy()
            cols = (xv >= x_range[0]) & (xv <= x_range[1])
        sc = s.cpu().numpy()
        if detrend:
            sc = sc - np.median(sc[:, cols], axis=1, keepdims=True)
        interior = ((zn >= surface * t) & (zn <= (1.0 - surface) * t)).cpu().numpy()
        w = sc**2 * interior[:, None] * cols[None]
        z = zn.cpu().numpy()[:, None]
        ends = np.linspace(-0.5 * t, 1.5 * t, n_grid)
        frac = ((xn - xn[0]) / max(float(xn[-1] - xn[0]), 1e-12)).cpu().numpy()
        lines = ends[:, None, None] + (ends[None, :, None] - ends[:, None, None]) * frac
        # sum_z w min((z - line)^2, width^2) for the column (z) and its mirror (t - z)
        cap = (0.25 * t if width is None else width) ** 2
        cost = [
            (w[None, None] * np.minimum((zz[None, None] - lines[:, :, None, :]) ** 2, cap)).sum(2)
            for zz in (z, t - z)
        ]
        best = np.minimum(cost[0], cost[1]).sum(-1)
        if slope_range is not None:
            slope = np.abs(ends[None, :] - ends[:, None]) / max(float(xn[-1] - xn[0]), 1e-12)
            best[(slope < slope_range[0]) | (slope > slope_range[1])] = np.inf
        i, j = np.unravel_index(np.argmin(best), best.shape)
        flip = (cost[1][i, j] < cost[0][i, j]) & cols
        mirrored = _hat((t - zn).clamp(min=0.0), zn) @ s
        mask = torch.as_tensor(flip, device=s.device)
        new = s.clone()
        new[:, mask] = mirrored[:, mask]
        self.params["strain"] = new
        if verbose:
            print(
                f"mirrored {int(flip.sum())} of {len(flip)} node columns; line from z = "
                f"{ends[i]:.0f} A at x = {float(xn[0]):.0f} A to z = {ends[j]:.0f} A at "
                f"x = {float(xn[-1]):.0f} A"
            )
        return flip

    # ------------------------------------------------------------------ results

    def simulate(self, rows: torch.Tensor | None = None) -> np.ndarray:
        """Model qx plot on the data grid, scaled to the data, (n_rows, n_q)."""
        rows = self.model["rows"] if rows is None else rows
        with torch.no_grad():
            _, parts = self._loss(self.params, rows)
            inten = self._render(self._propagate(self.params), self.params, rows)
        return (parts["scale"] * inten).cpu().numpy()

    def strain_map(
        self,
        shape: tuple[int, int] = (110, 200),
        mirror: bool = False,
        nodes: np.ndarray | None = None,
    ) -> Dataset2d:
        """eps(z, x) on a regular grid over the strain nodes and the foil, as a Dataset2d.

        Includes the fixed ``strain_gradient``. ``mirror`` returns the twin
        eps(x, t - z), which gives the same qx plot. ``nodes`` (n_z, n_x)
        replaces the fitted node values, e.g. to show another reconstruction on
        the same grid.
        """
        m, p = self.model, self.params
        strain = p["strain"] if nodes is None else torch.as_tensor(nodes).to(p["strain"])
        t = float(p["thickness"])
        x0, x1 = float(m["x_nodes"][0]), float(m["x_nodes"][-1])
        z = torch.linspace(0.0, t, shape[0], dtype=torch.float64, device=p["strain"].device)
        x = torch.linspace(x0, x1, shape[1], dtype=torch.float64, device=p["strain"].device)
        with torch.no_grad():
            eps = _hat(z, m["z_nodes"]) @ strain @ _hat(x, m["x_nodes"]).T
            eps = eps + m["grad_x"] * x[None, :] + m["grad_z"] * z[:, None]
        eps = eps.cpu().numpy()
        if mirror:
            eps = eps[::-1].copy()
        dz = t / max(shape[0] - 1, 1)
        dx = (x1 - x0) / max(shape[1] - 1, 1)
        return Dataset2d.from_array(
            eps,
            name="strain (mirrored)" if mirror else "strain",
            origin=(0.0, x0),
            sampling=(dz, dx),
            units=("A", "A"),
            signal_units="d u_g / d z",
        )

    # -------------------------------------------------------------------- plots

    @staticmethod
    def _display_units(units: str) -> tuple[float, str, str]:
        """Factor from A to the display length unit, and the length and q labels."""
        if units == "nm":
            return 0.1, "nm", "1/nm"
        return 1.0, "Å", "1/Å"

    def _qx_extent(self, rows: np.ndarray, f: float) -> list[float]:
        """[x0, x1, q0, q1] of the fitted rows, in display units."""
        x = float(self.params["x_offset"]) + rows * self.x_sampling
        q0 = float(self.params["q_offset"])
        q1 = q0 + (self.data.shape[1] - 1) * float(self.params["q_scale"])
        h = 0.5 * self.x_sampling
        return [f * (x[0] - h), f * (x[-1] + h), q0 / f, q1 / f]

    @staticmethod
    def _show(ax, image, x_ext, y_ext, transpose, **kwargs):
        """Image with x along the horizontal axis (or downward with ``transpose``).

        ``image`` is indexed (x, other); ``y_ext`` spans the other coordinate.
        """
        if transpose:
            return ax.imshow(
                image, extent=[y_ext[0], y_ext[1], x_ext[1], x_ext[0]], aspect="auto", **kwargs
            )
        return ax.imshow(
            image.T, extent=[x_ext[0], x_ext[1], y_ext[0], y_ext[1]], aspect="auto", **kwargs
        )

    def plot_geometry(self, power: float = 0.5, figsize=(11, 4.5)):
        """Mean CBED pattern with the fitted disks and extraction band, and the descan."""
        geo = self.geometry
        if geo is None:
            raise RuntimeError("fit_geometry first")
        has_centers = "centers_fit" in geo
        fig, axs = plt.subplots(
            1,
            2 if has_centers else 1,
            figsize=figsize,
            squeeze=False,
            layout="constrained",
            width_ratios=[1.0, 1.2] if has_centers else None,
        )
        ax = axs[0, 0]
        im = np.clip(geo["mean"] - geo["mean"].min(), 0, None) ** power
        ax.imshow(im, cmap="gray")
        d, c, R = geo["direction"], geo["center"], geo["radius_px"]
        n = np.arange(geo["disk_index"].min(), geo["disk_index"].max() + 1)
        ends = c[None] + np.array([n[0] - 1, n[-1] + 1])[:, None] * geo["g_px"] * d[None]
        for k in n:
            p = c + k * geo["g_px"] * d
            ax.add_patch(plt.Circle((p[1], p[0]), R, fill=False, color="tab:red", lw=1))
        ax.plot(*geo["disks"][:, ::-1].T, "+", color="tab:orange", ms=8)
        if "extract" in geo:
            ex = geo["extract"]
            dd = d * (-1.0 if ex["flip"] else 1.0)
            perp = np.array([-dd[1], dd[0]])
            t = ex["t"]
            for s in (-0.5 * ex["width"], 0.5 * ex["width"]):
                pts = c[None] + t[[0, -1], None] * dd[None] + s * perp[None]
                ax.plot(pts[:, 1], pts[:, 0], color="tab:cyan", lw=0.8)
            ends = c[None] + t[[0, -1], None] * dd[None]
            ax.annotate(
                "",
                xy=(ends[1, 1], ends[1, 0]),
                xytext=(c[1], c[0]),
                arrowprops=dict(arrowstyle="->", color="tab:cyan", lw=1.2),
            )
            ax.text(ends[1, 1], ends[1, 0], " +q", color="tab:cyan", va="center")
        lo = np.clip(ends.min(0) - 1.6 * R, 0, None)
        hi = np.minimum(ends.max(0) + 1.6 * R, np.array(im.shape) - 1)
        ax.set_xlim(lo[1], hi[1])
        ax.set_ylim(hi[0], lo[0])
        ax.set_title(f"mean pattern: |g| = {geo['g_px']:.1f} px, R = {R:.1f} px", loc="left")
        ax.set_axis_off()
        if has_centers:
            ax = axs[0, 1]
            perp = np.array([-d[1], d[0]])
            idx = np.arange(len(geo["centers"]))
            ok = geo["centers_used"]
            for vec, color, name in ((d, "tab:blue", "along g"), (perp, "tab:green", "across")):
                meas = (geo["centers"] - c) @ vec
                fit = (geo["centers_fit"] - c) @ vec
                ax.plot(idx[ok], meas[ok], ".", ms=2.5, color=color, alpha=0.6)
                ax.plot(idx[~ok], meas[~ok], ".", ms=2.5, color="0.75")
                ax.plot(idx, fit, color=color, lw=1.5, label=name)
            ax.set_xlabel("scan position")
            ax.set_ylabel("direct-beam shift (px)")
            ax.legend(frameon=False)
            ax.set_title("descan: measured (dots) and line fit", loc="left")
        return fig, axs

    def plot_qx(
        self,
        power: float = 1.0,
        cmap: str = "gray",
        units: str = "nm",
        transpose: bool = False,
        figsize=None,
    ):
        """The qx plot: x horizontal and q vertical, or the transpose."""
        f, lu, qu = self._display_units(units)
        if self.model is None:
            rows = np.arange(self.data.shape[0])
            x = self.x_origin + rows * self.x_sampling
            h = 0.5 * self.x_sampling
            q0 = self.q_origin
            q1 = q0 + (self.data.shape[1] - 1) * self.q_sampling
            ext = [f * (x[0] - h), f * (x[-1] + h), q0 / f, q1 / f]
        else:
            rows = self.model["rows"].cpu().numpy()
            ext = self._qx_extent(rows, f)
        data = np.clip(np.nan_to_num(self.data[rows]), 0, None) ** power
        figsize = figsize or ((4.5, 7) if transpose else (8, 3.5))
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        self._show(ax, data, ext[:2], ext[2:], transpose, cmap=cmap, origin="lower")
        xl, ql = f"x ({lu})", f"q ({qu})"
        ax.set_xlabel(ql if transpose else xl)
        ax.set_ylabel(xl if transpose else ql)
        return fig, ax

    def plot_fit(
        self,
        strain: bool = True,
        mirror: bool = False,
        residual: bool = True,
        transpose: bool = False,
        power: float = 1.0,
        cmap: str = "gray",
        units: str = "nm",
        strain_range: float | None = None,
        figsize=None,
    ):
        """Measured and model qx plots, their weighted residual and the strain, sharing x.

        Panels are stacked with x horizontal, or side by side with x running
        down when ``transpose`` (better for long scans). Data and model share
        one intensity scale; the strain panel has z (depth from the entrance
        surface) as its other axis and its own colorbar, in units of 1e-3.

        Parameters
        ----------
        strain, mirror : bool
            Add the strain map, and its depth-mirrored twin.
        residual : bool
            Add the weighted residual (model - data).
        transpose : bool
            Side-by-side panels with x vertical.
        power : float
            Display power of the qx intensities.
        units : {"nm", "A"}
            Display length unit (q in the reciprocal unit).
        strain_range : float, optional
            Strain color limit, in 1e-3. Default the largest |eps|.
        """
        if self.params is None:
            raise RuntimeError("set_model first")
        f, lu, qu = self._display_units(units)
        rows_t = self.model["rows"]
        rows = rows_t.cpu().numpy()
        data = np.nan_to_num(self.data[rows])
        model = self.simulate(rows_t)
        weight = self.model["weight"][rows_t].cpu().numpy()
        ext = self._qx_extent(rows, f)
        vmax = np.percentile(data, 99.9) ** power

        panels = [("data", "qx"), ("model", "qx")]
        if residual:
            panels.append(("weighted residual", "res"))
        if strain:
            panels.append(("strain", "eps"))
            if mirror:
                panels.append(("strain, mirrored twin", "twin"))
        t = float(self.params["thickness"])
        span = abs(ext[1] - ext[0])
        sizes = [
            1.0 if kind in ("qx", "res") else float(np.clip(f * t / span * 2.0, 0.6, 1.6))
            for _, kind in panels
        ]
        if figsize is None:
            figsize = (2.6 * sum(sizes) + 1.0, 7.5) if transpose else (8.5, 2.2 * sum(sizes) + 0.6)
        fig, axs = plt.subplots(
            1 if transpose else len(panels),
            len(panels) if transpose else 1,
            figsize=figsize,
            sharey=transpose,
            sharex=not transpose,
            layout="constrained",
            **({"width_ratios": sizes} if transpose else {"height_ratios": sizes}),
        )
        axs = np.atleast_1d(axs)
        xl, ql, zl = f"x ({lu})", f"q ({qu})", f"z ({lu})"
        cbar = dict(pad=0.01, fraction=0.04, location="bottom" if transpose else "right")
        for ax, (title, kind) in zip(axs, panels):
            if kind == "qx":
                img = data if title == "data" else model
                im = self._show(
                    ax,
                    np.clip(img, 0, None) ** power,
                    ext[:2],
                    ext[2:],
                    transpose,
                    cmap=cmap,
                    vmin=0,
                    vmax=vmax,
                    origin="lower",
                )
                fig.colorbar(im, ax=ax, **cbar)
                other = ql
            elif kind == "res":
                res = weight * (model - data)
                lim = np.percentile(np.abs(res), 99.5)
                im = self._show(
                    ax,
                    res,
                    ext[:2],
                    ext[2:],
                    transpose,
                    cmap="RdBu_r",
                    vmin=-lim,
                    vmax=lim,
                    origin="lower",
                )
                fig.colorbar(im, ax=ax, **cbar)
                other = ql
            else:
                ds = self.strain_map(mirror=kind == "twin")
                eps = 1e3 * ds.array
                lim = np.abs(eps).max() if strain_range is None else strain_range
                x0 = float(ds.origin[1])
                x1 = x0 + (eps.shape[1] - 1) * float(ds.sampling[1])
                if transpose:
                    im = ax.imshow(
                        eps.T,
                        extent=[0, f * t, f * x1, f * x0],
                        aspect="auto",
                        cmap="RdBu_r",
                        vmin=-lim,
                        vmax=lim,
                        interpolation="bilinear",
                    )
                else:
                    im = ax.imshow(
                        eps,
                        extent=[f * x0, f * x1, f * t, 0],
                        aspect="auto",
                        cmap="RdBu_r",
                        vmin=-lim,
                        vmax=lim,
                        interpolation="bilinear",
                    )
                fig.colorbar(im, ax=ax, label=r"$\partial u_g/\partial z$ ($10^{-3}$)", **cbar)
                other = zl
            ax.set_title(title, loc="left", fontsize=10)
            if transpose:
                ax.set_xlabel(other)
            else:
                ax.set_ylabel(other)
        if transpose:
            axs[0].set_ylabel(xl)
            axs[0].set_ylim(ext[1], ext[0])
        else:
            axs[-1].set_xlabel(xl)
            axs[0].set_xlim(ext[0], ext[1])
        return fig, axs

    def plot_strain(
        self,
        mirror: bool = False,
        strain_range: float | None = None,
        units: str = "nm",
        show_nodes: bool = True,
        equal_aspect: bool = True,
        nodes: np.ndarray | None = None,
        title: str | None = None,
        figsize=None,
    ):
        """eps(x, z) in units of 1e-3, entrance surface at the top, at true aspect ratio.

        ``nodes`` (n_z, n_x) shows other node values on the same grid (see
        ``strain_map``).
        """
        f, lu, _ = self._display_units(units)
        ds = self.strain_map(mirror=mirror, nodes=nodes)
        eps = 1e3 * ds.array
        x0 = float(ds.origin[1])
        x1 = x0 + (eps.shape[1] - 1) * float(ds.sampling[1])
        t = (eps.shape[0] - 1) * float(ds.sampling[0])
        lim = np.abs(eps).max() if strain_range is None else strain_range
        if figsize is None:
            aspect = t / (x1 - x0) if equal_aspect else 0.5
            figsize = (5.5 / aspect + 1.6, 5.5) if aspect > 0.75 else (8.0, 6.4 * aspect + 1.2)
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        im = ax.imshow(
            eps,
            extent=[f * x0, f * x1, f * t, 0.0],
            cmap="RdBu_r",
            vmin=-lim,
            vmax=lim,
            aspect="equal" if equal_aspect else "auto",
            interpolation="bilinear",
        )
        if show_nodes:
            xn = self.model["x_nodes"].cpu().numpy()
            zn = self.model["z_nodes"].cpu().numpy()
            zn = zn[zn <= t]
            zn = t - zn if mirror else zn
            xx, zz = np.meshgrid(xn, zn)
            ax.plot(f * xx.ravel(), f * zz.ravel(), ".", color="k", ms=1.5)
            ax.set_xlim(f * x0, f * x1)
            ax.set_ylim(f * t, 0.0)
        fig.colorbar(im, ax=ax, label=r"$\partial u_g/\partial z$ ($10^{-3}$)", shrink=0.9)
        ax.set_xlabel(f"x ({lu})")
        ax.set_ylabel(f"z ({lu})")
        if title is None:
            title = "strain, mirrored twin" if mirror else "strain"
        ax.set_title(title, loc="left")
        return fig, ax

    def plot_loss(self, figsize=(5, 3.5)):
        """Loss per optimizer iteration, over all fits so far."""
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        ax.semilogy(self.loss_history)
        ax.set_xlabel("iteration")
        ax.set_ylabel("loss")
        return fig, ax
