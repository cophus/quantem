from __future__ import annotations

import itertools
from os import PathLike
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from quantem.core.datastructures import Dataset2d, Dataset4dstem
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.validators import ensure_valid_array
from quantem.core.visualization import show_2d
from quantem.diffraction.bragg_vectors import BraggVectors
from quantem.diffraction.calibration import _ellipse_matrix
from quantem.diffraction.orientation_visualization import phase_color_cycle

# reciprocal length units converted to 1/Angstroms
_RECIPROCAL_UNITS = {"1/a": 1.0, "a^-1": 1.0, "1/nm": 0.1, "nm^-1": 0.1, "1/pm": 100.0}


class DiffractionMoire(AutoSerialize):
    """Lattices and moire spots in a single diffraction pattern.

    A pattern of two or more overlapping crystals, such as a twisted bilayer,
    contains the Bragg spots of every lattice plus double diffraction spots at
    sums of reflections from different lattices. This class detects the spots,
    finds the pattern center from Friedel symmetry, discovers the reciprocal
    lattice basis of each crystal, indexes every spot as an integer combination
    ``q = center + sum_l (h_l g1_l + k_l g2_l)``, and refines the center and all
    basis vectors together by weighted linear least squares over every indexed
    spot. The double diffraction spots then constrain the relative geometry of
    the lattices directly.

    Use :meth:`from_file` or :meth:`from_data`, then :meth:`find_peaks` and
    :meth:`find_lattices`. Positions are in pixels as ``(row, col)``; lengths
    are also reported in 1/Angstroms when the pattern is calibrated. In-plane
    angles follow the quantEM convention, from the row axis toward the column
    axis.

    Parameters
    ----------
    diffraction : Dataset2d
        The diffraction pattern.
    """

    _token = object()

    def __init__(self, diffraction: Dataset2d, _token: object | None = None):
        if _token is not self._token:
            raise RuntimeError(
                "Use DiffractionMoire.from_file() or .from_data() to instantiate this class."
            )
        super().__init__()
        self.diffraction = diffraction
        self.peaks = np.zeros((0, 3))
        self.center: NDArray | None = None
        self.lattices: NDArray | None = None
        self.indices: NDArray | None = None
        self.indexed: NDArray | None = None
        self.ellipse: NDArray | None = None
        self.rms: float | None = None
        self.metadata: dict[str, Any] = {}

    @classmethod
    def from_file(cls, file_path: str | PathLike, file_type: str | None = None) -> Self:
        """Read a diffraction pattern with :func:`~quantem.core.io.file_readers.read_2d`."""
        from quantem.core.io.file_readers import read_2d

        return cls.from_data(read_2d(file_path, file_type=file_type))

    @classmethod
    def from_data(cls, diffraction: Dataset2d | NDArray, sampling: float | None = None) -> Self:
        """Wrap a diffraction pattern.

        Parameters
        ----------
        diffraction : Dataset2d | np.ndarray
            The pattern.
        sampling : float, optional
            Reciprocal pixel size in 1/Angstroms, for an array or to override
            the calibration of a Dataset2d.
        """
        if not isinstance(diffraction, Dataset2d):
            diffraction = Dataset2d.from_array(ensure_valid_array(diffraction, ndim=2))
        if sampling is not None:
            diffraction.sampling = np.full(2, float(sampling))
            diffraction.units = ["1/A", "1/A"]
        return cls(diffraction=diffraction, _token=cls._token)

    @property
    def pixel_size(self) -> float | None:
        """Reciprocal pixel size in 1/Angstroms, None when uncalibrated."""
        unit = str(self.diffraction.units[0]).replace(" ", "").lower().replace("å", "a")
        scale = _RECIPROCAL_UNITS.get(unit)
        return None if scale is None else float(self.diffraction.sampling[0]) * scale

    def find_peaks(
        self,
        min_abs_intensity: float = 10.0,
        radius: float = 1.5,
        min_spacing: float = 6.0,
        edge_boundary: int = 8,
        background_sigma: float | None = 8.0,
        max_num_peaks: int = 2000,
        plot: bool = True,
    ) -> Self:
        """Detect the diffraction spots with :meth:`BraggVectors.detect_disks`.

        Parameters
        ----------
        min_abs_intensity : float, default=10.0
            Drop spots below this template correlation intensity.
        radius : float, default=1.5
            Radius of the synthetic spot template in pixels.
        min_spacing : float, default=6.0
            Minimum distance between spots in pixels.
        edge_boundary : int, default=8
            Border width in pixels in which spots are ignored.
        background_sigma : float, optional
            Width in pixels of the Gaussian background subtracted before
            detection, which removes the diffuse halo around the direct beam.
        max_num_peaks : int, default=2000
            Largest number of spots kept.
        plot : bool, default=True
            Show the detected spots.
        """
        array = np.asarray(self.diffraction.array, dtype=np.float32)
        bv = BraggVectors.from_dataset(Dataset4dstem.from_array(array[None, None]), device="cpu")
        bv.make_template_synthetic(radius=radius)
        peaks = bv.detect_disks(
            positions=[(0, 0)],
            min_abs_intensity=min_abs_intensity,
            min_spacing=min_spacing,
            edge_boundary=edge_boundary,
            background_sigma=background_sigma,
            subpixel="parabolic",
            max_num_peaks=max_num_peaks,
            progressbar=False,
        )
        self.peaks = np.asarray(peaks[0].numpy(), dtype=float).reshape(-1, 3)
        self.center = self.lattices = self.indices = self.indexed = self.rms = None
        self.ellipse = None
        if plot:
            self.plot()
        return self

    def find_lattices(
        self,
        num_lattices: int = 2,
        tol: float = 5.0,
        num_candidates: int = 40,
        min_radius: float = 20.0,
        plot: bool = True,
    ) -> Self:
        """Find the center and the basis of every lattice, then refine them.

        The center is found from Friedel symmetry: every pair ``+g, -g``
        shares its midpoint, so the midpoints of all pairs of strong spots pile
        up at the center, and the candidate piles are ranked by the intensity
        of the spots with an inversion partner. Each lattice basis is then picked from pairs of the strongest
        unassigned spots, by the intensity of the spots it explains within
        `tol`, minus the fraction a lattice of the same cell area would explain
        by chance; this penalty keeps a short moire vector from winning by
        explaining every spot. The spots on a chosen lattice are removed before
        the next one is picked. Finally :meth:`refine` indexes and fits
        everything.

        Parameters
        ----------
        num_lattices : int, default=2
            Number of crystal lattices.
        tol : float, default=5.0
            Largest distance in pixels between a spot and a lattice node.
        num_candidates : int, default=40
            Number of strongest unassigned spots tried as basis vectors.
        min_radius : float, default=20.0
            Spots closer than this to the center (pixels) are not used as basis
            vectors.
        plot : bool, default=True
            Show the indexed spots and the basis vectors.
        """
        if self.peaks.shape[0] < 3:
            raise RuntimeError("Run find_peaks() first.")
        q, inten = self.peaks[:, :2], self.peaks[:, 2]
        center = _friedel_center(q, inten, self.diffraction.shape, tol)
        v = q - center
        weight = np.clip(inten, 0.0, None)
        free = np.hypot(v[:, 0], v[:, 1]) > min_radius
        lattices = []
        for _ in range(int(num_lattices)):
            idx = np.flatnonzero(free)
            idx = idx[np.argsort(-inten[idx])][: int(num_candidates)]
            a, b = np.triu_indices(idx.size, 1)
            g1, g2 = v[idx[a]], v[idx[b]]
            cos = np.sum(g1 * g2, axis=1) / (
                np.hypot(g1[:, 0], g1[:, 1]) * np.hypot(g2[:, 0], g2[:, 1])
            )
            keep = np.abs(cos) < np.cos(np.deg2rad(30.0))
            if not keep.any():
                raise RuntimeError(f"Found only {len(lattices)} lattices.")
            bases = np.stack([g1[keep], g2[keep]], axis=1)
            score, _ = _lattice_score(v[free], weight[free], bases, tol)
            basis = _reduce_basis(bases[int(np.argmax(score))])
            lattices.append(basis)
            _, on = _lattice_score(v, weight, basis[None], tol)
            free &= ~on[0]
        self.metadata["find_lattices"] = dict(
            num_lattices=int(num_lattices),
            tol=float(tol),
            num_candidates=int(num_candidates),
            min_radius=float(min_radius),
        )
        return self.refine(center=center, lattices=np.stack(lattices), tol=tol, plot=plot)

    def refine(
        self,
        center: NDArray | None = None,
        lattices: NDArray | None = None,
        max_terms: int = 2,
        tol: float = 5.0,
        num_iterations: int = 3,
        ellipse: bool = False,
        plot: bool = True,
    ) -> Self:
        """Index every spot and refine the center and the lattice bases.

        The predicted spots are all integer combinations of the reflections of
        up to `max_terms` different lattices within the detected area: the
        reflections of each lattice for ``max_terms=1``, plus the double
        diffraction spots ``g_a + g_b`` for ``max_terms=2``. Each spot is
        assigned to the lowest order prediction within `tol` (orders count
        ``sum |h| + |k|``), and ``q = center + sum_l (h_l g1_l + k_l g2_l)`` is
        then fit by least squares weighted by the square root of the spot
        intensity, as in :meth:`BraggVectors.fit_lattice`. Spots with more than
        one distinct prediction within `tol` are indexed but left out of the
        fit, since overlapping spots merge into one peak between the
        predictions; near commensurate twists many double diffraction spots
        fall close to others. Indexing and fitting
        alternate `num_iterations` times.

        With `ellipse`, every layer is taken as the same crystal lattice rotated
        about the beam, as in a twisted homobilayer, and a shared elliptic
        distortion of the pattern is refined with the lattice, the rotation of
        each layer and the center, by nonlinear least squares. The distortion is
        stored in :attr:`ellipse` as ``[e11, e12]``, the correction matrix
        ``[[1 + e11, e12], [e12, 1 - e11]]`` of the quantEM calibration
        (:func:`~quantem.diffraction.calibration.calibrate_ellipse`), and
        :meth:`summary` reports the corrected lattices. Layers with different
        lattices leave the distortion inseparable from their shapes.

        Parameters
        ----------
        center : np.ndarray, optional
            Starting ``(row, col)`` center in pixels; defaults to the current one.
        lattices : np.ndarray, optional
            Starting ``(num_lattices, 2, 2)`` bases ``[g1, g2]`` in pixels, each
            vector as ``(row, col)``; defaults to the current ones.
        max_terms : int, default=2
            Largest number of lattices combined in one predicted spot.
        tol : float, default=5.0
            Largest distance in pixels between a spot and its prediction.
        num_iterations : int, default=3
            Number of index and fit cycles.
        ellipse : bool, default=False
            Refine a shared elliptic distortion, with one lattice for all layers.
        plot : bool, default=True
            Show the indexed spots and the basis vectors.
        """
        center = self.center if center is None else np.asarray(center, dtype=float)
        lattices = self.lattices if lattices is None else np.asarray(lattices, dtype=float)
        if center is None or lattices is None:
            raise RuntimeError("Run find_lattices() first, or pass center and lattices.")
        q, inten = self.peaks[:, :2], self.peaks[:, 2]
        num = lattices.shape[0]
        e = np.zeros(2)
        if ellipse:
            # layer l is R(theta_l) applied to the basis of layer 0, all seen
            # through the inverse of the correction matrix
            lattices = np.stack(
                [lattices[0]] + [_closest_basis(g, lattices[0]) for g in lattices[1:]]
            )
            theta = [_polar(lattices[0], g)[0] for g in lattices[1:]]
            params = np.concatenate([center, e, lattices[0].ravel(), theta])
        for _ in range(int(num_iterations)):
            indices, indexed, overlap = _index_spots(q - center, lattices, tol, max_terms)
            fit = indexed & ~overlap
            if fit.sum() < 2 + 2 * num:
                raise RuntimeError(
                    "Too few spots indexed to fit the lattices; for closely spaced spots, "
                    "lower tol so that fewer predictions overlap."
                )
            w = np.sqrt(np.clip(inten[fit], 0.0, None))[:, None]
            if ellipse:
                hk, target = indices[fit], q[fit]

                def residual(p):
                    c, g, _ = _ellipse_model(p, num)
                    return ((c + hk @ g.reshape(-1, 2) - target) * w).ravel()

                params = least_squares(residual, params).x
                center, lattices, e = _ellipse_model(params, num)
            else:
                design = np.concatenate([np.ones((fit.sum(), 1)), indices[fit]], axis=1)
                beta = np.linalg.lstsq(design * w, q[fit] * w, rcond=None)[0]
                center = beta[0]
                lattices = beta[1:].reshape(num, 2, 2)
        resid = q[fit] - center - indices[fit] @ lattices.reshape(-1, 2)
        self.center, self.lattices = center, lattices
        self.ellipse = e if ellipse else None
        self.indices, self.indexed = indices, indexed
        self.rms = float(np.sqrt(np.average(np.sum(resid**2, axis=1), weights=w[:, 0] ** 2)))
        self.metadata["refine"] = dict(
            max_terms=int(max_terms), tol=float(tol), ellipse=bool(ellipse)
        )
        if plot:
            self.plot()
        return self

    @property
    def spot_terms(self) -> NDArray:
        """Number of lattices contributing to each spot (0 for the center)."""
        num = self.lattices.shape[0]
        return np.any(self.indices.reshape(-1, num, 2) != 0, axis=2).sum(axis=1)

    def summary(self) -> dict[str, Any]:
        """Lattice spacings, relative rotations and moire periods.

        For each pair of lattices ``(a, b)``, the basis of ``b`` is replaced by
        the integer change of basis closest to the basis of ``a``, and the
        transform ``T`` with ``g_b = T g_a`` is split into a rotation and a
        symmetric stretch (polar decomposition). The rotation is the twist from
        ``a`` to ``b`` in the quantEM in-plane convention, the stretch
        eigenvalues minus one are the lattice mismatch, and the moire reciprocal
        vectors are ``g_b - g_a``. Spot positions cannot tell a twist ``t``
        from its symmetry equivalents, such as ``t + 60`` degrees for a
        hexagonal lattice, so the smallest one is reported. After
        ``refine(ellipse=True)`` every quantity is computed from the lattices
        with the elliptic distortion removed, and the mismatch is zero by
        construction.

        Returns
        -------
        dict
            ``lattices``: per lattice ``k`` (the lengths of ``g1``, ``g2``),
            ``d = 1 / k`` and the ``g1``-``g2`` angle; ``pairs``: per pair of
            lattices the twist (degrees), mismatch, and moire ``k`` and period;
            ``ellipse``: ``[e11, e12]`` or None. Lengths are in 1/Angstroms and
            Angstroms when the pattern is calibrated, otherwise in pixels.
        """
        if self.lattices is None:
            raise RuntimeError("Run find_lattices() first.")
        scale = self.pixel_size or 1.0
        out: dict[str, Any] = {
            "units": "1/A" if self.pixel_size else "px",
            "ellipse": self.ellipse,
        }
        lattices = self.lattices
        if self.ellipse is not None:
            lattices = lattices @ _ellipse_matrix(self.ellipse).T
        out["lattices"] = []
        for g in lattices:
            length = np.hypot(g[:, 0], g[:, 1])
            angle = np.degrees(np.arccos(np.dot(g[0], g[1]) / np.prod(length)))
            out["lattices"].append(
                dict(k=length * scale, d=1.0 / (length * scale), angle_deg=angle)
            )
        out["pairs"] = []
        for a, b in itertools.combinations(range(lattices.shape[0]), 2):
            ga = lattices[a]
            gb = _closest_basis(lattices[b], ga)
            twist, stretch = _polar(ga, gb)
            k_moire = np.hypot(*(gb - ga).T) * scale
            out["pairs"].append(
                dict(
                    lattices=(a, b),
                    twist_deg=float(np.degrees(twist)),
                    mismatch=stretch - 1.0,
                    k_moire=k_moire,
                    period_moire=1.0 / k_moire,
                )
            )
        return out

    def plot(
        self,
        sigma: float = 2.0,
        power: float = 0.3,
        ring_radius: float = 20.0,
        axsize: tuple[float, float] = (8, 8),
    ):
        """Pattern with the detected spots, or with the indexed spots and bases.

        Markers are thin rings drawn outside the spots, so the spots stay
        visible. Before :meth:`find_lattices` every detected spot is ringed in
        red. After it, spots of a single lattice are ringed in that lattice's
        color, double diffraction spots in white, unindexed spots in gray, and
        the basis vectors are drawn as arrows from the center that stop at the
        ring.

        Parameters
        ----------
        sigma : float, default=2.0
            Gaussian smoothing of the displayed pattern in pixels, which keeps
            spots a few pixels wide visible when a large pattern is shown small.
        power : float, default=0.3
            Display power of the intensity.
        ring_radius : float, default=20.0
            Radius of the marker rings in pixels.
        axsize : tuple, default=(8, 8)
            Figure size in inches.

        Returns
        -------
        fig, ax
        """
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Circle

        image = np.clip(np.asarray(self.diffraction.array, dtype=float), 0.0, None)
        if sigma > 0:
            image = gaussian_filter(image, sigma)
        fig, ax = show_2d(
            image,
            norm={"power": power, "lower_quantile": 0.5, "upper_quantile": 0.9995},
            axsize=axsize,
        )
        q = self.peaks

        def rings(mask, color, lw=1.0):
            circles = [Circle((c, r), ring_radius) for r, c in q[mask, :2]]
            ax.add_collection(
                PatchCollection(circles, facecolor="none", edgecolor=color, linewidth=lw),
                autolim=False,
            )

        if self.lattices is None:
            rings(np.ones(q.shape[0], dtype=bool), "r")
            ax.set_title(f"{q.shape[0]} spots")
            return fig, ax
        colors = phase_color_cycle(self.lattices.shape[0])
        terms = self.spot_terms
        rings(~self.indexed, "0.5", lw=0.8)
        rings(self.indexed & (terms >= 2), "w", lw=0.8)
        for i, basis in enumerate(self.lattices):
            m = self.indexed & (terms == 1) & np.any(self.indices[:, 2 * i : 2 * i + 2] != 0, 1)
            rings(m, colors[i], lw=1.5)
            for g in basis:
                tip = self.center + g * (1.0 - ring_radius / np.hypot(*g))
                ax.annotate(
                    "",
                    xy=(tip[1], tip[0]),
                    xytext=(self.center[1], self.center[0]),
                    arrowprops=dict(arrowstyle="-|>", color=colors[i], lw=1.5),
                )
        ax.set_title(f"{self.indexed.sum()} of {q.shape[0]} spots indexed, rms {self.rms:.2f} px")
        return fig, ax

    def __repr__(self) -> str:
        lines = [f"DiffractionMoire, {self.peaks.shape[0]} spots"]
        if self.lattices is None:
            return "\n".join(lines)
        res = self.summary()
        k_unit = res["units"]
        d_unit = "A" if k_unit == "1/A" else "1/px"
        lines.append(
            f"  center ({self.center[0]:.2f}, {self.center[1]:.2f}) px, "
            f"{self.indexed.sum()} spots indexed, rms {self.rms:.2f} px"
        )
        if self.ellipse is not None:
            e11, e12 = self.ellipse
            lines.append(
                f"  ellipse        e11 {e11:.2e}, e12 {e12:.2e} "
                f"({200 * np.hypot(e11, e12):.2f}% ellipticity), corrected below"
            )
        for i, lat in enumerate(res["lattices"]):
            k, d = lat["k"], lat["d"]
            lines.append(
                f"  lattice {i}      |g| {k[0]:.4f}, {k[1]:.4f} {k_unit}, "
                f"d {d[0]:.4f}, {d[1]:.4f} {d_unit}, angle {lat['angle_deg']:.2f} deg"
            )
        for p in res["pairs"]:
            mis, per = 100 * p["mismatch"], p["period_moire"]
            lines.append(
                f"  lattices {p['lattices'][0]}-{p['lattices'][1]}  twist {p['twist_deg']:.2f} deg, "
                f"mismatch {mis[0]:.2f}%, {mis[1]:.2f}%, moire period {per[0]:.3f}, "
                f"{per[1]:.3f} {d_unit}"
            )
        return "\n".join(lines)


def _friedel_center(
    q: NDArray, inten: NDArray, shape, tol: float, num: int = 150, num_candidates: int = 20
) -> NDArray:
    """Pattern center from Friedel symmetry.

    Every pair ``+g, -g`` shares its midpoint, so the midpoints of all pairs of
    the `num` strongest spots pile up at the center. They also pile up at half
    lattice vectors, and inversion through those maps every integer
    combination of the lattices onto another one, so positions alone barely
    separate them from the center. Friedel mates also share their intensity:
    each of the `num_candidates` largest piles is scored by
    ``sum sqrt(min(I, I'))`` over the spots whose inversion lands within `tol`
    of a spot of intensity ``I'``. The best candidate is then refined as the
    weighted mean midpoint of its Friedel pairs.
    """
    order = np.argsort(-inten)[:num]
    p = q[order]
    i, j = np.triu_indices(p.shape[0], 1)
    mid = 0.5 * (p[i] + p[j])
    H, _, _ = np.histogram2d(
        mid[:, 0], mid[:, 1], bins=shape, range=[[0, shape[0]], [0, shape[1]]]
    )
    H = gaussian_filter(H, 1.0)
    peaks = np.argwhere((H == maximum_filter(H, size=2 * int(np.ceil(tol)) + 1)) & (H > 0))
    peaks = peaks[np.argsort(-H[peaks[:, 0], peaks[:, 1]])][:num_candidates] + 0.5

    tree = cKDTree(q)
    inten = np.clip(inten, 0.0, None)

    def friedel_pairs(center):
        dist, partner = tree.query(2.0 * center - q)
        keep = (dist < tol) & (partner != np.arange(q.shape[0]))
        return keep, partner, np.sqrt(np.minimum(inten, inten[partner]))

    def support(center):
        keep, _, w = friedel_pairs(center)
        return w[keep].sum()

    center = peaks[int(np.argmax([support(c) for c in peaks]))]
    for _ in range(3):
        keep, partner, w = friedel_pairs(center)
        center = np.average(0.5 * (q[keep] + q[partner[keep]]), axis=0, weights=w[keep])
    return center


def _lattice_score(v: NDArray, weight: NDArray, bases: NDArray, tol: float):
    """Chance-corrected weighted fraction of spots within `tol` of each lattice.

    Returns the score of each of the ``(C, 2, 2)`` bases and the ``(C, N)``
    mask of the spots on it. A lattice with cell area ``A`` lands within `tol`
    of a random spot with probability ``pi tol**2 / A``, which is subtracted.
    """
    B = np.swapaxes(bases, 1, 2)  # columns g1, g2
    det = np.linalg.det(B)
    ok = np.abs(det) > 1e-6
    f = np.einsum("cij,nj->cni", np.linalg.inv(np.where(ok[:, None, None], B, np.eye(2))), v)
    d = np.einsum("cij,cnj->cni", B, f - np.round(f))
    on = np.hypot(d[..., 0], d[..., 1]) < tol
    chance = np.minimum(1.0, np.pi * tol**2 / np.maximum(np.abs(det), 1e-6))
    score = np.sum(weight * (on - chance[:, None]), axis=1) / weight.sum()
    return np.where(ok, score, -np.inf), on


def _reduce_basis(basis: NDArray) -> NDArray:
    """Shortest right-handed basis of the same lattice (Lagrange-Gauss reduction)."""
    g1, g2 = basis
    while True:
        if g1 @ g1 > g2 @ g2:
            g1, g2 = g2, g1
        m = np.round((g1 @ g2) / (g1 @ g1))
        if m == 0:
            break
        g2 = g2 - m * g1
    if g1 @ g2 < 0:
        g2 = -g2
    if g1[0] * g2[1] - g1[1] * g2[0] < 0:
        g1, g2 = g2, g1
    return np.stack([g1, g2])


def _polar(ga: NDArray, gb: NDArray) -> tuple[float, NDArray]:
    """Rotation angle (radians) and sorted stretches of ``T`` with ``g_b = T g_a``."""
    T = np.linalg.solve(ga, gb).T  # vectors as columns
    U, s, Vt = np.linalg.svd(T)
    R = U @ Vt
    return float(np.arctan2(R[1, 0], R[0, 0])), np.sort(s)


def _ellipse_model(params: NDArray, num: int):
    """Center, measured lattice bases and ellipse from the ellipse fit parameters.

    ``params`` holds the center, ``[e11, e12]``, the corrected basis of layer 0
    and the rotation of every other layer from row toward column (radians).
    """
    center, e, g0, theta = params[:2], params[2:4], params[4:8].reshape(2, 2), params[8:]
    distort = np.linalg.inv(_ellipse_matrix(e))
    lattices = [g0]
    for t in theta:
        rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        lattices.append(g0 @ rot.T)
    return center, np.stack(lattices) @ distort.T, e


def _closest_basis(basis: NDArray, reference: NDArray) -> NDArray:
    """The basis of the same lattice (unimodular, same handedness) closest to `reference`."""
    best, best_err = basis, np.inf
    for u in itertools.product((-1, 0, 1), repeat=4):
        U = np.array(u, dtype=float).reshape(2, 2)
        if round(np.linalg.det(U)) == 1:
            err = np.sum((U @ basis - reference) ** 2)
            if err < best_err:
                best, best_err = U @ basis, err
    return best


def _index_spots(v: NDArray, lattices: NDArray, tol: float, max_terms: int):
    """Lowest order integer combination within `tol` of each spot.

    Returns the ``(N, 2 * num_lattices)`` indices, the mask of indexed spots,
    and the mask of spots with more than one distinct prediction within `tol`.
    """
    num = lattices.shape[0]
    k_max = float(np.hypot(v[:, 0], v[:, 1]).max()) + tol
    reflections = []
    for g in lattices:
        n = int(np.ceil(k_max / np.hypot(g[:, 0], g[:, 1]).min())) + 1
        hk = np.stack(np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1)), -1).reshape(-1, 2)
        hk = hk[np.any(hk != 0, axis=1)]
        pos = hk @ g
        keep = np.hypot(pos[:, 0], pos[:, 1]) <= k_max
        reflections.append((hk[keep], pos[keep]))
    idx_all = [np.zeros((1, 2 * num), dtype=int)]
    pos_all = [np.zeros((1, 2))]
    for terms in range(1, int(max_terms) + 1):
        for combo in itertools.combinations(range(num), terms):
            grids = np.meshgrid(*[np.arange(reflections[i][0].shape[0]) for i in combo])
            sel = [gr.ravel() for gr in grids]
            pos = sum(reflections[i][1][s] for i, s in zip(combo, sel))
            idx = np.zeros((pos.shape[0], 2 * num), dtype=int)
            for i, s in zip(combo, sel):
                idx[:, 2 * i : 2 * i + 2] = reflections[i][0][s]
            keep = np.hypot(pos[:, 0], pos[:, 1]) <= k_max
            idx_all.append(idx[keep])
            pos_all.append(pos[keep])
    idx = np.concatenate(idx_all)
    pos = np.concatenate(pos_all)
    order = np.abs(idx).sum(axis=1)
    indices = np.zeros((v.shape[0], 2 * num), dtype=int)
    indexed = np.zeros(v.shape[0], dtype=bool)
    overlap = np.zeros(v.shape[0], dtype=bool)
    for n, hits in enumerate(cKDTree(pos).query_ball_point(v, tol)):
        if hits:
            hits = np.asarray(hits)
            dist = np.hypot(*(pos[hits] - v[n]).T)
            best = hits[np.lexsort((dist, order[hits]))[0]]
            indices[n], indexed[n] = idx[best], True
            # distinct predictions within tol (combinations at the same position count once)
            overlap[n] = np.any(np.hypot(*(pos[hits] - pos[best]).T) > 0.5)
    return indices, indexed, overlap
