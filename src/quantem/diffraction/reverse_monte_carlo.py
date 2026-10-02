"""Reverse Monte Carlo fitting of diffuse electron scattering from several zone axes.

One periodic supercell of a disordered crystal is fitted to every pattern at
once. The supercell's diffuse amplitude ``G(h) = sum_j (sigma_j - c)
exp(-2 pi i h.x_j / N)`` lives on the FFT grid of its sites; the diffuse
intensity of each pattern is ``|f_B - f_A|^2 |G|^2`` read off where the
pattern's Ewald sphere (with that pattern's fitted tilt) cuts the grid,
averaged over the cubic rotations so the model is as symmetric as the
(statistically cubic) foil. Swapping two atoms changes ``G`` by two phase
factors, so every move is scored exactly without recomputing the supercell.
Bragg peaks are excluded by a sigmoid weight that is 0 on each reflection and
1 away from it; a smooth background (constant, two Gaussians about the direct
beam, an Einstein thermal diffuse term and the Bragg cores blurred by the
detector's point spread) and one scale per pattern are solved in closed form.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import permutations, product

import numpy as np
import torch
from scipy import ndimage, optimize, sparse
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

from quantem.core.io.serialize import AutoSerialize
from quantem.diffraction.crystal import Crystal, electron_scattering_factor


def electron_wavelength(energy_ev: float) -> float:
    """Relativistic electron wavelength in Angstroms."""
    return 12.2642598 / np.sqrt(energy_ev * (1.0 + 0.97847573e-6 * energy_ev))


def cubic_rotations() -> np.ndarray:
    """The 24 proper rotations of the cube as signed permutation matrices (24, 3, 3)."""
    ops = []
    for perm in permutations(range(3)):
        for signs in product((1, -1), repeat=3):
            m = np.zeros((3, 3), dtype=int)
            m[range(3), perm] = signs
            if round(np.linalg.det(m)) == 1:
                ops.append(m)
    return np.stack(ops)


def _zone_frame(zone_axis) -> np.ndarray:
    """Orthonormal crystal-frame basis (e1, e2, z) with z along the zone axis."""
    z = np.asarray(zone_axis, dtype=float)
    z /= np.linalg.norm(z)
    trial = np.eye(3)[np.argmin(np.abs(z))]
    e1 = trial - (trial @ z) * z
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(z, e1), z])


def _rot2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ReverseMonteCarlo(AutoSerialize):
    """Reverse Monte Carlo fit of one supercell to diffraction patterns along several zone axes.

    Build with :meth:`from_images`, then ``set_crystal`` -> ``fit_geometry`` ->
    ``set_mask`` -> ``build_supercell`` -> ``fit_background`` -> ``run``.
    """

    _token = object()

    def __init__(self, images, zone_axes, sampling, energy, names, bin_factor, _token=None):
        if _token is not self._token:
            raise RuntimeError("Use ReverseMonteCarlo.from_images().")
        self.images = [np.asarray(im, dtype=np.float32) for im in images]
        self.zone_axes = [tuple(int(v) for v in z) for z in zone_axes]
        self.sampling = float(sampling)
        self.energy = float(energy)
        self.wavelength = electron_wavelength(self.energy)
        self.names = list(names)
        self.bin_factor = int(bin_factor)
        self.crystal: Crystal | None = None
        self.geometry: dict | None = None
        self.mask: dict | None = None
        self.loss_history: list[float] = []

    @classmethod
    def from_images(
        cls,
        images: Sequence,
        zone_axes: Sequence[Sequence[int]],
        energy: float = 200e3,
        bin_factor: int = 8,
        sampling: float | None = None,
        names: Sequence[str] | None = None,
    ) -> "ReverseMonteCarlo":
        """Patterns (Dataset2d or arrays) and the zone axis of each.

        Parameters
        ----------
        images : sequence of Dataset2d or ndarray
            One diffraction pattern per zone axis.
        zone_axes : sequence of (u, v, w)
            Nominal zone axis of each pattern, in the crystal's lattice
            indices; the tilt off it is fitted.
        energy : float
            Beam energy in eV.
        bin_factor : int
            Detector binning for the diffuse fit (geometry uses full resolution).
        sampling : float, optional
            Detector pixel size in 1/Angstrom. Read from the first Dataset2d
            (1/nm is converted) when not given.
        names : sequence of str, optional
            Panel titles; default the zone axes.
        """
        arrays = []
        for im in images:
            if hasattr(im, "array"):
                if sampling is None:
                    s = float(np.asarray(im.sampling)[0])
                    units = str(im.units[0]).lower()
                    sampling = s * 0.1 if "nm" in units else s
                arrays.append(np.asarray(im.array))
            else:
                arrays.append(np.asarray(im))
        if sampling is None:
            raise ValueError("sampling (1/Angstrom per pixel) is required for plain arrays.")
        if len(arrays) != len(zone_axes):
            raise ValueError("one zone axis per image")
        if names is None:
            names = ["[" + "".join(str(v) for v in z) + "]" for z in zone_axes]
        return cls(arrays, zone_axes, sampling, energy, names, bin_factor, _token=cls._token)

    def _binned(self, a: np.ndarray, reduce: str = "mean") -> np.ndarray:
        b = self.bin_factor
        ny, nx = (a.shape[0] // b) * b, (a.shape[1] // b) * b
        out = a[:ny, :nx].reshape(ny // b, b, nx // b, b).sum(axis=(1, 3))
        return out / b**2 if reduce == "mean" else out

    # ------------------------------------------------------------------ crystal

    def set_crystal(
        self,
        crystal: Crystal | None = None,
        cif_file: str | None = None,
        merge: dict[str, str] | None = None,
    ) -> "ReverseMonteCarlo":
        """Crystal whose mixed-occupancy sites are fitted.

        Parameters
        ----------
        crystal, cif_file : Crystal or path
            The average structure, with fractional occupancies on shared sites.
        merge : dict, optional
            Species to relabel before fitting, e.g. ``{"Zr": "Nb"}`` folds Zr
            into Nb so the shared site becomes a binary V-Nb site.
        """
        from ase.data import atomic_numbers, chemical_symbols

        if crystal is None:
            if cif_file is None:
                raise ValueError("give crystal or cif_file")
            crystal = Crystal.from_cif(cif_file, verbose=False)
        self.crystal = crystal
        cell = crystal.lat_real.numpy()
        a = float(np.linalg.norm(cell[0]))
        if not np.allclose(cell, a * np.eye(3), atol=1e-3 * a):
            raise NotImplementedError("Only cubic cells are supported so far.")
        merge = merge or {}
        frac = np.mod(crystal.positions_frac.numpy(), 1.0)
        symbols = [chemical_symbols[int(z)] for z in crystal.numbers]
        symbols = [merge.get(s, s) for s in symbols]
        occ = crystal.occupancy.numpy()

        # group species by site
        sites: list[tuple[np.ndarray, dict[str, float]]] = []
        for f, s, o in zip(frac, symbols, occ):
            for site in sites:
                if np.allclose(site[0], f, atol=1e-4):
                    site[1][s] = site[1].get(s, 0.0) + float(o)
                    break
            else:
                sites.append((f, {s: float(o)}))
        mixed = [(f, comp) for f, comp in sites if len(comp) > 1]
        if not mixed:
            raise ValueError("The crystal has no mixed-occupancy site to fit.")
        species = sorted({s for _, comp in mixed for s in comp})
        if len(species) != 2:
            raise NotImplementedError(f"Binary sites only so far; got {species}. Use merge=.")
        comps = {tuple(sorted(comp.items())) for _, comp in mixed}
        if len(comps) != 1:
            raise NotImplementedError("All mixed sites must share one composition.")
        comp = dict(next(iter(comps)))
        total = sum(comp.values())

        # smallest grid divisor that puts every mixed site on an integer grid
        fr = np.stack([f for f, _ in mixed])
        for d in range(1, 13):
            if np.allclose(fr * d, np.round(fr * d), atol=1e-4):
                break
        else:
            raise ValueError("Mixed sites are not on a rational grid with denominator <= 12.")

        self._a_crystal = a
        self.lattice_parameter = a
        self.species = species  # [A, B]; sigma = 1 marks B
        self.numbers = [atomic_numbers[s] for s in species]
        self.concentration = comp[species[1]] / total
        self.site_grid = np.round(fr * d).astype(int)
        self.grid_divisor = d
        print(
            f"{len(mixed)} mixed site(s) per cell, "
            + ", ".join(f"{s} {comp[s] / total:.3f}" for s in species)
            + f", a = {a:.4f} A"
        )
        return self

    def _zone_reflections(self, zone_axis, k_max: float):
        """Allowed reflections in a zone: hkl (n, 3), zone-frame coords at the CIF a (n, 2), |F|^2."""
        crystal = self.crystal
        crystal.calculate_structure_factors(k_max)
        hkl = crystal.hkl.numpy()
        inten = crystal.struct_factors_int.numpy()
        keep = (hkl @ np.asarray(zone_axis) == 0) & (inten > 1e-4 * inten.max())
        hkl, inten = hkl[keep], inten[keep]
        frame = _zone_frame(zone_axis)
        g = hkl / self._a_crystal
        return hkl, g @ frame[:2].T, inten

    # ----------------------------------------------------------------- geometry

    @staticmethod
    def _find_peaks(im, n_peaks: int = 150):
        smooth = ndimage.gaussian_filter(im, 2.0)
        prom = smooth - ndimage.gaussian_filter(im, 25.0)
        local = (prom == ndimage.maximum_filter(prom, 15)) & (prom > 0)
        r, c = np.nonzero(local)
        order = np.argsort(prom[r, c])[::-1][:n_peaks]
        r, c = r[order], c[order]
        pts = []
        for ri, ci in zip(r, c):
            r0, r1 = max(ri - 4, 0), min(ri + 5, im.shape[0])
            c0, c1 = max(ci - 4, 0), min(ci + 5, im.shape[1])
            w = np.clip(prom[r0:r1, c0:c1] - 0.3 * prom[ri, ci], 0, None)
            rr, cc = np.mgrid[r0:r1, c0:c1]
            pts.append([(w * rr).sum() / w.sum(), (w * cc).sum() / w.sum()])
        return np.asarray(pts), prom[r, c], prom

    @staticmethod
    def _halo_center(im) -> np.ndarray:
        """Center of the broad inelastic halo, which sits on the direct beam."""
        small = ndimage.median_filter(im[::4, ::4].astype(np.float64), size=9)
        b = ndimage.gaussian_filter(small, 10)
        return np.asarray(np.unravel_index(np.argmax(b), b.shape), dtype=float) * 4 + 1.5

    def fit_geometry(
        self,
        scale_range: tuple[float, float] = (0.85, 1.2),
        k_max: float = 1.6,
        centers: Sequence | None = None,
        fit_tilt: bool = True,
        verbose: bool = True,
    ) -> "ReverseMonteCarlo":
        """Index every pattern, fit its detector distortion and its tilt off the zone axis.

        The direct beam is the detected peak nearest the center of the broad
        inelastic halo (a tilted pattern can have diffracted beams brighter
        than the direct beam), or nearest ``centers[i]`` (row, col). Each
        pattern then gets its own center and 2x2 detector matrix (rotation,
        scale, ellipticity), fitted to the matched peaks. The lattice
        parameter is the mean over patterns at the nominal pixel size.

        The tilt (beam direction off the zone axis, small-angle vector in the
        zone frame) is fitted to the Bragg intensities: each reflection's
        excitation error is ``s = -(|g|^2 / 2K + tilt . g)`` and its intensity
        ``|F|^2 exp(-s^2 / 2 sigma^2)``. Zero tilt puts the Laue circle on the
        direct beam.
        """
        if self.crystal is None:
            raise RuntimeError("set_crystal first")
        pix = self.sampling
        k_wave = 1.0 / self.wavelength
        geo = dict(centers=[], matrices=[], tilts=[], a=[], rms_px=[], n_matched=[], peaks=[])
        geo.update(bragg_hkl=[], bragg_g=[], bragg_intensity=[], bragg_px=[], excitation_width=[])
        for i, (im, zone) in enumerate(zip(self.images, self.zone_axes)):
            pts, heights, prom = self._find_peaks(im)
            guess = (
                np.asarray(centers[i], dtype=float)
                if centers is not None and centers[i] is not None
                else self._halo_center(im)
            )
            center = pts[np.argmin(np.linalg.norm(pts - guess, axis=1))]
            hkl, g2, inten = self._zone_reflections(zone, k_max)
            nz = np.linalg.norm(hkl, axis=1) > 0
            g2_nz, inten_nz = g2[nz], inten[nz]

            # coarse search: in-plane rotation and scale
            score_img = ndimage.gaussian_filter(np.clip(prom, 0, None), 3.0)
            wts = np.sqrt(inten_nz)
            best = (-np.inf, 0.0, 1.0)
            for scale in np.arange(scale_range[0], scale_range[1] + 1e-9, 0.004):
                for th in np.deg2rad(np.arange(0.0, 360.0, 0.5)):
                    p = center + (g2_nz @ _rot2(th).T) / (pix * scale)
                    ok = (
                        (p[:, 0] >= 0)
                        & (p[:, 0] < im.shape[0] - 1)
                        & (p[:, 1] >= 0)
                        & (p[:, 1] < im.shape[1] - 1)
                    )
                    if ok.sum() < 4:
                        continue
                    pi = np.round(p[ok]).astype(int)
                    s = (wts[ok] * score_img[pi[:, 0], pi[:, 1]]).sum() / wts[ok].sum()
                    if s > best[0]:
                        best = (s, th, scale)
            A = _rot2(best[1]) / (pix * best[2])
            c = center.copy()

            # refine center + 2x2 matrix on matched peaks, tightening the match
            for tol in (12.0, 8.0, 5.0, 5.0):
                p = c + g2_nz @ A.T
                dist, j = cKDTree(pts).query(p)
                ok = dist < tol
                obs = pts[j[ok]]
                X = np.column_stack([np.ones(ok.sum()), g2_nz[ok]])
                coef, *_ = np.linalg.lstsq(X, obs, rcond=None)
                c, A = coef[0], coef[1:].T
            p = c + g2_nz @ A.T
            dist, j = cKDTree(pts).query(p)
            ok = dist < 5.0
            rms = float(np.sqrt(np.mean(dist[ok] ** 2)))
            a_i = self._a_crystal / (pix * np.sqrt(abs(np.linalg.det(A))))
            sv = np.linalg.svd(A, compute_uv=False)

            # Bragg intensities at the fitted positions
            p_all = c + g2 @ A.T
            r_core = 0.03 / pix
            inten_meas = _integrate_spots(im, p_all, r_core)

            geo["centers"].append(c)
            geo["matrices"].append(A)
            geo["a"].append(float(a_i))
            geo["rms_px"].append(rms)
            geo["n_matched"].append(int(ok.sum()))
            geo["peaks"].append(pts)
            geo["bragg_hkl"].append(hkl)
            geo["bragg_g"].append(g2)  # zone frame, at the CIF lattice parameter
            geo["bragg_px"].append(p_all)
            geo["bragg_intensity"].append(inten_meas)
            geo["_inten_kin"] = geo.get("_inten_kin", []) + [inten]
            if verbose:
                print(
                    f"{self.names[i]}: center ({c[0]:.1f}, {c[1]:.1f}), a = {a_i:.4f} A, "
                    f"anisotropy {100 * (sv[0] / sv[1] - 1):.2f}%, "
                    f"{int(ok.sum())} peaks, rms {rms:.2f} px"
                )

        self.lattice_parameter = float(np.mean(geo["a"]))
        self.geometry = geo
        for i in range(len(self.images)):
            tilt, width = (np.zeros(2), np.nan)
            if fit_tilt:
                tilt, width = self._fit_tilt(i, k_wave)
            geo["tilts"].append(tilt)
            geo["excitation_width"].append(width)
            if verbose and fit_tilt:
                ang = np.rad2deg(np.linalg.norm(tilt))
                print(f"{self.names[i]}: tilt {ang:.2f} deg off the zone axis")
        if verbose:
            print(f"lattice parameter {self.lattice_parameter:.4f} A at {pix:.6f} 1/A per pixel")
        return self

    def _fit_tilt(self, i: int, k_wave: float, max_tilt_deg: float = 3.5, prior_deg: float = 2.0):
        """Tilt vector (zone frame, radians) from the Bragg intensities of pattern i, with a
        Gaussian prior of ``prior_deg`` so patterns whose intensities barely constrain it stay
        near the zone axis."""
        geo = self.geometry
        hkl = geo["bragg_hkl"][i]
        nz = np.linalg.norm(hkl, axis=1) > 0
        g = geo["bragg_g"][i][nz] * self._a_crystal / self.lattice_parameter
        p_px = geo["bragg_px"][i][nz]
        ny, nx = self.images[i].shape
        inside = (
            (p_px[:, 0] > 20) & (p_px[:, 0] < ny - 20) & (p_px[:, 1] > 20) & (p_px[:, 1] < nx - 20)
        )
        g = g[inside]
        meas = geo["bragg_intensity"][i][nz][inside]
        kin = geo["_inten_kin"][i][nz][inside]
        ok = np.isfinite(meas)
        g, meas, kin = g[ok], np.clip(meas[ok], 0, None), kin[ok]
        y = np.sqrt(meas / meas.max())
        g2 = (g**2).sum(1) / (2 * k_wave)

        def model(x):
            tilt, log_w, log_a = x[:2], x[2], x[3]
            s = -(g2 + g @ tilt)
            return (
                np.exp(log_a) * np.sqrt(kin / kin.max()) * np.exp(-0.25 * (s / np.exp(log_w)) ** 2)
            )

        best = None
        for tx in np.linspace(-0.06, 0.06, 25):
            for ty in np.linspace(-0.06, 0.06, 25):
                for lw in (np.log(0.01), np.log(0.03)):
                    x = np.array([tx, ty, lw, 0.0])
                    r = ((model(x) - y) ** 2).sum() + ((x[:2] / np.deg2rad(prior_deg)) ** 2).sum()
                    if best is None or r < best[0]:
                        best = (r, x)
        lim = np.deg2rad(max_tilt_deg)
        lo = np.array([-lim, -lim, np.log(0.003), -5.0])
        hi = np.array([lim, lim, np.log(0.05), 5.0])
        prior = np.deg2rad(prior_deg)

        def resid(x):
            return np.concatenate([model(x) - y, x[:2] / prior])

        sol = optimize.least_squares(
            resid, np.clip(best[1], lo + 1e-9, hi - 1e-9), bounds=(lo, hi)
        )
        return sol.x[:2], float(np.exp(sol.x[2]))

    def bragg_positions(self, i: int, k_max: float = 3.0) -> np.ndarray:
        """Detector positions (row, col) of every zone reflection, direct beam included."""
        _, g2, _ = self._zone_reflections(self.zone_axes[i], k_max)
        return self.geometry["centers"][i] + g2 @ self.geometry["matrices"][i].T

    def _q_zone(self, i: int, rows, cols) -> np.ndarray:
        """Pixel coordinates -> in-plane scattering vector (..., 2) in the zone frame, 1/A."""
        p = np.stack([rows, cols], axis=-1) - self.geometry["centers"][i]
        g = p @ np.linalg.inv(self.geometry["matrices"][i]).T
        return g * self._a_crystal / self.lattice_parameter

    def _q_crystal(self, i: int, q2: np.ndarray) -> np.ndarray:
        """In-plane zone-frame q (n, 2) -> crystal-frame q (n, 3) on the tilted Ewald sphere."""
        k_wave = 1.0 / self.wavelength
        tilt = self.geometry["tilts"][i]
        qz = -((q2**2).sum(1) / (2 * k_wave) + q2 @ tilt)
        return np.column_stack([q2, qz]) @ _zone_frame(self.zone_axes[i])

    # --------------------------------------------------------------------- mask

    def set_mask(
        self,
        bragg_radius: float = 0.08,
        softness: float = 0.01,
        q_max: float = 1.2,
        center_radius: float = 0.25,
        edge_px: int = 8,
        tail_widths: tuple[float, ...] = (0.02, 0.06, 0.15),
    ) -> "ReverseMonteCarlo":
        """Diffuse-scattering weight, binned data, Bragg intensities and PSF tails.

        The weight is ``sigmoid((d - bragg_radius) / softness)``, with ``d``
        the distance (1/A) to the nearest reflection, the direct beam
        included: 0 on every Bragg peak, 1 between them. Pixels beyond
        ``q_max`` or within ``edge_px`` of the detector edge are dropped, and
        a second sigmoid removes the direct beam's bloom out to
        ``center_radius``. Each ``bin_factor`` square is reduced to its
        weighted mean.

        Each reflection's integrated intensity weights the diffuse envelope.
        The Bragg cores blurred by ``(1 + (r / width)^2)^-1.5`` kernels, one
        per ``tail_widths`` (1/A), are background terms for the detector's
        point-spread tails.
        """
        from scipy.signal import fftconvolve

        b = self.bin_factor
        out = dict(
            bragg_radius=bragg_radius, softness=softness, q_max=q_max, center_radius=center_radius
        )
        out.update(y=[], w=[], k=[], data=[], scale=[], bragg_k=[], bragg_intensity=[], tails=[])
        for i, im in enumerate(self.images):
            ny, nx = (im.shape[0] // b) * b, (im.shape[1] // b) * b
            rows, cols = np.mgrid[0:ny, 0:nx].astype(np.float64)
            k = self._q_zone(i, rows, cols)
            bragg = self.bragg_positions(i)
            g_q = self._q_zone(i, bragg[:, 0], bragg[:, 1])
            d, j = cKDTree(g_q).query(k.reshape(-1, 2))
            d = d.reshape(ny, nx)
            j = j.reshape(ny, nx)
            q = np.linalg.norm(k, axis=-1)
            w = (
                _sigmoid((d - bragg_radius) / softness)
                * _sigmoid((q - center_radius) / softness)
                * (q < q_max)
            )
            w[:edge_px] = w[-edge_px:] = 0
            w[:, :edge_px] = w[:, -edge_px:] = 0
            y = im[:ny, :nx].astype(np.float64)

            # integrated Bragg intensities over the local ring median
            n_g = len(g_q)
            core = d < bragg_radius
            ring = (d >= bragg_radius) & (d < 1.6 * bragg_radius)
            ring_med = np.zeros(n_g)
            jr, yr = j[ring], y[ring]
            order = np.argsort(jr, kind="stable")
            jr, yr = jr[order], yr[order]
            starts = np.searchsorted(jr, np.arange(n_g))
            ends = np.searchsorted(jr, np.arange(n_g), side="right")
            for g in np.nonzero(ends > starts)[0]:
                ring_med[g] = np.median(yr[starts[g] : ends[g]])
            core_sig = np.where(core, np.clip(y - ring_med[j], 0, None), 0.0)
            inten = np.bincount(j[core], weights=core_sig[core], minlength=n_g)
            n_core = np.bincount(j[core], minlength=n_g)
            seen = n_core > 0.5 * n_core.max()  # whole spot on the detector

            wb = self._binned(w, "sum")
            yb = np.where(wb > 0, self._binned(w * y, "sum") / np.maximum(wb, 1e-12), 0.0)
            rb, cb = np.mgrid[0 : ny // b, 0 : nx // b].astype(np.float64) * b + (b - 1) / 2
            norm = (wb * yb).sum() / wb.sum()
            tails = []
            for width in tail_widths:
                gpx = width / self.sampling
                half = int(min(6 * gpx, 200))
                rr = np.hypot(*np.mgrid[-half : half + 1, -half : half + 1])
                kern = (1 + (rr / gpx) ** 2) ** -1.5
                t = fftconvolve(core_sig, kern / kern.sum(), mode="same")
                tails.append(
                    np.where(wb > 0, self._binned(w * t, "sum") / np.maximum(wb, 1e-12), 0.0)
                    / norm
                )
            out["y"].append(yb / norm)
            out["w"].append(wb / b**2)
            out["k"].append(self._q_zone(i, rb, cb))
            out["data"].append(self._binned(y) / norm)
            out["scale"].append(norm)
            out["bragg_k"].append(g_q[seen])
            out["bragg_intensity"].append(inten[seen] / norm)
            out["tails"].append(np.stack(tails))
        self.mask = out
        return self

    # ---------------------------------------------------------------- supercell

    def build_supercell(
        self,
        cells: int = 16,
        seed: int | None = 0,
        symmetrize: bool = True,
        debye_waller: float = 0.5,
        envelope: str = "measured",
        resolution: float = 0.75,
        shared_scale: bool = True,
        device: str | None = None,
    ) -> "ReverseMonteCarlo":
        """Random supercell of ``cells^3`` unit cells at the crystal's composition.

        Parameters
        ----------
        cells : int
            Unit cells along each cube edge. The diffuse model is sampled
            every ``1 / (cells a)`` in reciprocal space.
        symmetrize : bool
            Average every pattern over the 24 cubic rotations of the supercell.
        debye_waller : float
            Isotropic B (A^2) damping the diffuse intensity.
        envelope : {"measured", "kinematic"}
            Chemical diffuse intensity is periodic in the reciprocal lattice,
            so diffuse scattering out of every Bragg beam g lands on the same
            |G|^2 and only its form factor changes: the intensity is
            ``|G(q)|^2 sum_g P_g |f_B - f_A|^2(q - g)``. "measured" takes P_g
            from the integrated Bragg intensities of each pattern (dynamical
            redistribution, tilt and thickness included); "kinematic" keeps
            only the direct beam.
        resolution : float
            Gaussian sigma, in supercell reciprocal-grid steps, with which
            each pixel reads the diffuse grid (27 nearest points). A finite
            supercell's |G|^2 is speckle; reading it through a kernel of about
            one step damps the speckle and stops the fit chasing single grid
            points (most visibly on mirror planes, where fewer symmetry images
            average).
        shared_scale : bool
            One diffuse scale for every pattern. The envelope carries each
            pattern's absolute Bragg intensities, so diffuse scattering out of
            a beam is proportional to that beam; a per-pattern scale lets a
            pattern's smooth background swallow its diffuse intensity.
        device : str, optional
            torch device; default cuda, then mps, then cpu.
        """
        if self.mask is None:
            raise RuntimeError("set_mask first")
        rng = np.random.default_rng(seed)
        self.rng = rng
        d = self.grid_divisor
        n = cells * d
        m = np.stack(np.meshgrid(*(np.arange(cells),) * 3, indexing="ij"), -1).reshape(-1, 3)
        x = (m[:, None, :] * d + self.site_grid[None]).reshape(-1, 3)
        n_sites = len(x)
        n_b = int(round(self.concentration * n_sites))
        sigma = np.zeros(n_sites, dtype=bool)
        sigma[rng.choice(n_sites, n_b, replace=False)] = True
        self.cells, self.grid_size = cells, n
        self.site_x = x
        self.sigma = sigma
        self.debye_waller = float(debye_waller)
        self.symmetrize = bool(symmetrize)
        self.envelope = envelope
        self.resolution = float(resolution)
        self.shared_scale = bool(shared_scale)
        self.device = torch.device(device or _default_device())
        self._setup_forward()
        print(f"{n_sites} sites, {n}^3 grid, {int(self._fit.sum())} fitted pixels, {self.device}")
        return self

    def _setup_forward(self):
        """Ewald-sphere sampling of the supercell grid for every binned pixel."""
        n = self.grid_size
        dev = self.device
        cols_w, vals_w, u_all, basis_all, pix_image = [], [], [], [], []
        stencil = np.array(list(product((-1, 0, 1), repeat=3)))
        for i, kk in enumerate(self.mask["k"]):
            q2 = kk.reshape(-1, 2)
            h = self._q_crystal(i, q2) * self.lattice_parameter * self.cells
            h0 = np.round(h).astype(np.int64)
            f = h - h0
            wts = np.exp(
                -0.5
                * ((f[:, None, :] - stencil[None]) ** 2).sum(-1)
                / max(self.resolution, 0.3) ** 2
            )
            wts /= wts.sum(1, keepdims=True)
            idx = np.mod(h0[:, None, :] + stencil[None], n)
            cols_w.append((idx[..., 0] * n + idx[..., 1]) * n + idx[..., 2])
            vals_w.append(wts)
            u, tds = self._envelope(i, q2)
            u_all.append(u)
            tails = self.mask["tails"][i].reshape(len(self.mask["tails"][i]), -1).T
            qm = np.linalg.norm(q2, axis=1)
            halos = self._phonon_halos(i, q2)
            rings = self._powder_rings(qm)
            basis_all.append(np.column_stack([np.ones_like(qm), qm, tds, halos, rings, tails]))
            pix_image.append(np.full(len(h), i))
        cols = np.concatenate(cols_w)
        vals = np.concatenate(vals_w)
        used, inv = np.unique(cols, return_inverse=True)
        cols_used = inv.reshape(cols.shape)
        n_pix = len(cols)
        self._W_all = sparse.csr_matrix(
            (vals.ravel(), (np.repeat(np.arange(n_pix), cols.shape[1]), cols_used.ravel())),
            shape=(n_pix, len(used)),
        )
        # symmetry: model reads mean_k I(S_k h) at each used h
        ops = cubic_rotations() if self.symmetrize else np.eye(3, dtype=int)[None]
        hh = np.stack(np.unravel_index(used, (n,) * 3), -1)
        sym_flat = np.stack(
            [(lambda s: (s[:, 0] * n + s[:, 1]) * n + s[:, 2])(np.mod(hh @ op.T, n)) for op in ops]
        )
        needed, inv_s = np.unique(sym_flat, return_inverse=True)
        self._needed = needed
        self._sym_index = torch.as_tensor(inv_s.reshape(sym_flat.shape), device=dev)
        self._u_all = np.concatenate(u_all)
        self._basis_all = np.concatenate(basis_all)
        self._pix_image = np.concatenate(pix_image)
        y = np.concatenate([a.ravel() for a in self.mask["y"]])
        w = np.concatenate([a.ravel() for a in self.mask["w"]])
        self._y_all, self._w_all = y, w
        fit = w > 1e-3
        self._fit = fit
        self._Wc = torch.as_tensor(cols_used[fit], device=dev)
        self._Wv = torch.as_tensor(vals[fit], dtype=torch.float32, device=dev)
        self._y = torch.as_tensor(y[fit], dtype=torch.float32, device=dev)
        self._w = torch.as_tensor(w[fit], dtype=torch.float32, device=dev)
        self._img = torch.as_tensor(self._pix_image[fit], device=dev)
        self._u = torch.as_tensor(self._u_all[fit], dtype=torch.float32, device=dev)
        hn = np.stack(np.unravel_index(needed, (n,) * 3), -1)
        self._h_needed = torch.as_tensor(hn, device=dev)
        ang = 2 * np.pi * np.arange(n) / n
        self._cos = torch.as_tensor(np.cos(ang), dtype=torch.float32, device=dev)
        self._sin = torch.as_tensor(np.sin(ang), dtype=torch.float32, device=dev)
        self._x = torch.as_tensor(self.site_x, device=dev)
        self._recompute_G()
        # direct-beam Lorentzian half width and wide Gaussian sigma, 1/A
        self.background_sigmas = [(0.1, 0.8) for _ in self.images]
        self._sigma_bounds = (np.array([0.01, 0.3]), np.array([1.0, 3.0]))
        self.coefficients = None

    def _read(self, i_used: torch.Tensor) -> torch.Tensor:
        """Grid values on the used points (..., n_used) -> kernel-weighted values on fitted pixels."""
        return (i_used[..., self._Wc] * self._Wv).sum(-1)

    def _envelope(self, i: int, q2: np.ndarray, max_beams: int = 40):
        """Diffuse and thermal-diffuse envelopes summed over the Bragg beams of pattern i."""
        if self.envelope == "measured":
            g = self.mask["bragg_k"][i]
            p = self.mask["bragg_intensity"][i]
            order = np.argsort(p)[::-1][:max_beams]
            g, p = g[order], p[order]
            keep = p > 0.002 * p.max()
            g, p = g[keep], p[keep]
        elif self.envelope == "kinematic":
            g, p = np.zeros((1, 2)), np.array([self.mask["bragg_intensity"][i].sum()])
        else:
            raise ValueError(f"unknown envelope {self.envelope!r}")
        z = torch.tensor(self.numbers)
        c = self.concentration
        u = np.zeros(len(q2))
        tds = np.zeros(len(q2))
        for gi, pi in zip(g, p):
            qm = np.linalg.norm(q2 - gi, axis=1)
            fe = electron_scattering_factor(z, torch.as_tensor(qm, dtype=torch.float64)).numpy()
            dw = np.exp(-0.5 * self.debye_waller * qm**2)
            u += pi * (fe[1] - fe[0]) ** 2 * dw
            tds += pi * ((1 - c) * fe[0] ** 2 + c * fe[1] ** 2) * (1 - dw)
        return u, tds

    def _powder_rings(self, q: np.ndarray, width: float = 0.025, k_max: float = 1.5) -> np.ndarray:
        """Powder rings of the average crystal about the direct beam, ``sum_g |F_g|^2 / g^2``
        broadened by ``width`` (1/A): misoriented grains or a damaged surface layer."""
        self.crystal.calculate_structure_factors(k_max)
        g = self.crystal.g_len.numpy() * self._a_crystal / self.lattice_parameter
        f2 = self.crystal.struct_factors_int.numpy()
        keep = g > 1e-6
        g, f2 = g[keep], f2[keep]
        out = np.zeros_like(q)
        for gi, fi in zip(g, f2):
            out += fi / gi**2 * np.exp(-0.5 * ((q - gi) / width) ** 2)
        return out / out.max()

    def _phonon_halos(self, i: int, q2: np.ndarray, widths=(0.04, 0.12)) -> np.ndarray:
        """Thermal diffuse halos about every Bragg spot: ``sum_g P_g k^2 / (|q - g|^2 + k^2)``
        per width ``k`` (1/A), acoustic phonons concentrating TDS next to each reflection."""
        g = self.mask["bragg_k"][i]
        p = self.mask["bragg_intensity"][i]
        nz = np.linalg.norm(g, axis=1) > 1e-6
        g, p = g[nz], p[nz] / max(p[nz].sum(), 1e-12)
        out = np.zeros((len(q2), len(widths)))
        for gi, pi in zip(g, p):
            d2 = ((q2 - gi) ** 2).sum(1)
            for n, k in enumerate(widths):
                out[:, n] += pi * k**2 / (d2 + k**2)
        return out

    def _phases(self, sites: torch.Tensor):
        """cos and sin of 2 pi h.x / N for the given sites on every needed grid point."""
        idx = (self._x[sites] @ self._h_needed.T) % self.grid_size  # (B, n_needed)
        return self._cos[idx], self._sin[idx]

    def _recompute_G(self):
        n = self.grid_size
        a = np.zeros((n,) * 3)
        xs = self.site_x
        a[xs[:, 0], xs[:, 1], xs[:, 2]] = self.sigma - self.concentration
        G = np.fft.fftn(a).ravel()[self._needed]
        self._Gr = torch.as_tensor(G.real, dtype=torch.float32, device=self.device)
        self._Gi = torch.as_tensor(G.imag, dtype=torch.float32, device=self.device)

    def _diffuse_used(self, Gr=None, Gi=None) -> torch.Tensor:
        """Symmetrized |G|^2 per site on the used grid points."""
        Gr = self._Gr if Gr is None else Gr
        Gi = self._Gi if Gi is None else Gi
        inten = (Gr**2 + Gi**2) / len(self.sigma)
        return inten[..., self._sym_index].mean(dim=-2)

    # --------------------------------------------------------------- background

    def _bg_basis(self, i: int, sel: np.ndarray) -> np.ndarray:
        q = self._basis_all[sel, 1]
        s1, s2 = self.background_sigmas[i]
        return np.column_stack(
            [
                np.ones_like(q),
                1.0 / (1.0 + (q / s1) ** 2),
                np.exp(-0.5 * (q / s2) ** 2),
                self._basis_all[sel, 2:],
            ]
        )

    def _solve_linear(self, diffuse_fit: np.ndarray, refit_sigmas: bool = False):
        """Scale, background amplitudes (and optionally widths) per image, weighted least squares."""
        y = self._y_all[self._fit]
        w = self._w_all[self._fit]
        img = self._pix_image[self._fit]
        sel_all = np.nonzero(self._fit)[0]
        coefs, loss = [], 0.0
        for i in range(len(self.images)):
            m = img == i
            sel = sel_all[m]
            sw = np.sqrt(w[m])

            def solve(sig):
                sig = np.clip(sig, *self._sigma_bounds)
                self.background_sigmas[i] = tuple(float(v) for v in sig)
                X = np.column_stack([diffuse_fit[m], self._bg_basis(i, sel)])
                lo = np.zeros(X.shape[1])
                lo[1] = -np.inf
                x = _lsq_bounded(X, y[m], sw, lo)
                return x, float(((X @ x - y[m]) ** 2 * w[m]).sum())

            if refit_sigmas:
                r = optimize.minimize(
                    lambda ls: solve(np.exp(ls))[1],
                    np.log(self.background_sigmas[i]),
                    method="Nelder-Mead",
                    options=dict(xatol=1e-3, fatol=1e-6, maxiter=200),
                )
                solve(np.exp(r.x))
            c, l_i = solve(self.background_sigmas[i])
            coefs.append(c)
            loss += l_i
        self.coefficients = np.stack(coefs)
        if not getattr(self, "shared_scale", False):
            return loss

        # one diffuse scale for all patterns, backgrounds per pattern
        blocks = [self._bg_basis(i, sel_all[img == i]) for i in range(len(self.images))]
        nb = blocks[0].shape[1]
        X = np.zeros((len(y), 1 + nb * len(blocks)))
        X[:, 0] = diffuse_fit
        lo = np.zeros(X.shape[1])
        for i, bl in enumerate(blocks):
            X[img == i, 1 + nb * i : 1 + nb * (i + 1)] = bl
            lo[1 + nb * i] = -np.inf
        sw = np.sqrt(w)
        x = _lsq_bounded(X, y, sw, lo)
        if x[0] <= 0:
            # a random supercell explains nothing yet: start the scale from the background residual
            r = y - X[:, 1:] @ x[1:]
            x[0] = max((w * r * diffuse_fit).sum() / max((w * diffuse_fit**2).sum(), 1e-30), 0)
        for i in range(len(blocks)):
            self.coefficients[i, 0] = x[0]
            self.coefficients[i, 1:] = x[1 + nb * i : 1 + nb * (i + 1)]
        return float(((X @ x - y) ** 2 * w).sum())

    def fit_background(self) -> float:
        """Fit the diffuse scale and, per pattern, a constant, a direct-beam Lorentzian, a wide
        Gaussian, Einstein thermal diffuse, phonon halos about each reflection, powder rings of
        the average crystal and Bragg tails."""
        loss = self._solve_linear(self._model_diffuse(), refit_sigmas=True)
        self._update_residual()
        print(
            "background widths (1/A): "
            + ", ".join(
                f"{n} {s[0]:.3f}/{s[1]:.3f}" for n, s in zip(self.names, self.background_sigmas)
            )
        )
        return loss

    def _model_diffuse(self) -> np.ndarray:
        """u(q) * (W I_sym) on the fitted pixels (before scale)."""
        return (self._u * self._read(self._diffuse_used())).cpu().numpy().astype(np.float64)

    def _update_residual(self):
        dev = self.device
        img = self._pix_image[self._fit]
        sel = np.nonzero(self._fit)[0]
        bg = np.zeros(len(sel))
        for i in range(len(self.images)):
            m = img == i
            bg[m] = self._bg_basis(i, sel[m]) @ self.coefficients[i, 1:]
        self._bg = torch.as_tensor(bg, dtype=torch.float32, device=dev)
        c = torch.as_tensor(self.coefficients[:, 0], dtype=torch.float32, device=dev)
        self._scale = c[self._img]
        model = self._scale * self._u * self._read(self._diffuse_used()) + self._bg
        self._r = self._y - model
        return float((self._w * self._r**2).sum())

    # ---------------------------------------------------------------------- RMC

    def run(
        self,
        n_sweeps: int = 20,
        batch: int = 64,
        temperature: float = 0.05,
        refit_every: int = 2,
        progress: bool = True,
    ) -> "ReverseMonteCarlo":
        """Composition-conserving swaps until the diffuse fit converges.

        Each batch proposes ``batch`` disjoint swaps against the current
        supercell and scores each exactly. Metropolis acceptance at
        ``temperature`` (a fraction of the median score change, falling
        linearly to 0); the accepted swaps are applied together, capped at a
        number that adapts so the joint step never raises the loss. A sweep is
        one proposal per site. Scale and background are refit every
        ``refit_every`` sweeps.
        """
        if self.coefficients is None:
            self.fit_background()
        n_sites = len(self.sigma)
        loss = self._update_residual()
        if not self.loss_history:
            self.loss_history.append(loss)
        cap = max(batch // 8, 1)
        t0 = None
        n_batches = max(n_sites // (2 * batch), 1)
        sweeps = tqdm(range(n_sweeps), desc="RMC sweeps", disable=not progress)
        su = None
        for sweep in sweeps:
            accepted = 0
            for _ in range(n_batches):
                if su is None:
                    su = self._scale * self._u
                on = np.nonzero(~self.sigma)[0]
                off = np.nonzero(self.sigma)[0]
                j_on = self.rng.choice(on, batch, replace=False)
                j_off = self.rng.choice(off, batch, replace=False)
                c1, s1 = self._phases(torch.as_tensor(j_on, device=self.device))
                c2, s2 = self._phases(torch.as_tensor(j_off, device=self.device))
                dGr = c1 - c2
                dGi = s2 - s1
                d_int = (2 * (self._Gr * dGr + self._Gi * dGi) + dGr**2 + dGi**2) / n_sites
                d_used = d_int[:, self._sym_index].mean(dim=1)  # (B, n_used)
                dm = su * self._read(d_used)  # (B, n_fit)
                dL = (self._w * (dm**2 - 2 * self._r * dm)).sum(-1)
                dl = dL.cpu().numpy()
                if not t0:
                    t0 = float(np.median(np.abs(dl)))
                temp = temperature * t0 * (1 - sweep / n_sweeps)
                if temp > 0:
                    ok = (dl < 0) | (self.rng.random(batch) < np.exp(-np.clip(dl / temp, 0, 50)))
                else:
                    ok = dl < 0
                pick = np.nonzero(ok)[0]
                pick = pick[np.argsort(dl[pick])][:cap]
                if len(pick) == 0:
                    continue
                pk = torch.as_tensor(pick, device=self.device)
                Gr_new = self._Gr + dGr[pk].sum(0)
                Gi_new = self._Gi + dGi[pk].sum(0)
                r_new = self._y - su * self._read(self._diffuse_used(Gr_new, Gi_new)) - self._bg
                loss_new = float((self._w * r_new**2).sum())
                if loss_new > loss + max(temp, 0.0) * len(pick) and len(pick) > 1:
                    cap = max(cap // 2, 1)
                    continue
                self._Gr, self._Gi, self._r = Gr_new, Gi_new, r_new
                self.sigma[j_on[pick]] = True
                self.sigma[j_off[pick]] = False
                loss = loss_new
                accepted += len(pick)
                cap = min(int(cap * 1.25) + 1, batch)
            if (sweep + 1) % refit_every == 0:
                self._recompute_G()
                self._solve_linear(self._model_diffuse())
                loss = self._update_residual()
                su = None
            self.loss_history.append(loss)
            sweeps.set_postfix(loss=f"{loss:.4g}", accepted=accepted)
        self._recompute_G()
        self._solve_linear(self._model_diffuse())
        self.loss_history[-1] = self._update_residual()
        return self

    # ---------------------------------------------------------------- analysis

    def model_images(self, diffuse_only: bool = False) -> list[np.ndarray]:
        """Model on the binned grid of every pattern (data units): scaled diffuse + background,
        or the scaled diffuse term alone."""
        i_used = self._diffuse_used().cpu().numpy().astype(np.float64)
        diffuse = self._u_all * (self._W_all @ i_used)
        out = []
        for i, k in enumerate(self.mask["k"]):
            m = self._pix_image == i
            sel = np.nonzero(m)[0]
            c = self.coefficients[i]
            img = c[0] * diffuse[m]
            if not diffuse_only:
                img = img + self._bg_basis(i, sel) @ c[1:]
            out.append(img.reshape(k.shape[:2]))
        return out

    def background_images(self) -> list[np.ndarray]:
        """Fitted background (constant, Gaussians, thermal diffuse, Bragg tails) per pattern."""
        full = self.model_images()
        diffuse = self.model_images(diffuse_only=True)
        return [f - d for f, d in zip(full, diffuse)]

    def warren_cowley(self, n_shells: int = 6) -> dict:
        """Warren-Cowley alpha of the B species about B for the first neighbour shells."""
        n = self.grid_size
        d = self.grid_divisor
        occ = np.zeros((n,) * 3)
        xs = self.site_x
        occ[xs[:, 0], xs[:, 1], xs[:, 2]] = self.sigma
        site = np.zeros((n,) * 3, dtype=bool)
        site[xs[:, 0], xs[:, 1], xs[:, 2]] = True
        fo = np.fft.fftn(occ)
        fs = np.fft.fftn(site)
        bb = np.real(np.fft.ifftn(fo * np.conj(fo)))
        ss = np.real(np.fft.ifftn(fs * np.conj(fs)))
        v = np.stack(np.meshgrid(*(np.fft.fftfreq(n, 1 / n),) * 3, indexing="ij"), -1)
        r = np.linalg.norm(v, axis=-1) / d * self.lattice_parameter
        valid = ss > 0.5
        radii = np.unique(np.round(r[valid], 4))[1 : n_shells + 1]
        c = self.concentration
        alpha = []
        for rad in radii:
            m = valid & (np.abs(r - rad) < 1e-3)
            p_bb = bb[m].sum() / ss[m].sum() / c  # P(B neighbour | B)
            alpha.append((p_bb - c) / (1 - c))
        return dict(
            radius=radii, alpha=np.asarray(alpha), pair=f"{self.species[1]}-{self.species[1]}"
        )

    def diffuse_section(self, normal=(0, 0, 1), extent: float = 2.0, smooth: bool = True):
        """Symmetrized supercell diffuse intensity on a reciprocal-lattice plane, in Laue units.

        Parameters
        ----------
        normal : (h, k, l)
            Plane normal; the plane passes through the origin.
        extent : float
            Half width in units of the cubic reciprocal lattice vector 1/a.
        smooth : bool
            Blur by the fit's ``resolution`` kernel.

        Returns
        -------
        image, (u, v) in-plane axes (crystal frame, unit vectors), distance of each pixel from
        the nearest reciprocal-lattice node (1/a units)
        """
        n = self.grid_size
        a = np.zeros((n,) * 3)
        xs = self.site_x
        a[xs[:, 0], xs[:, 1], xs[:, 2]] = self.sigma - self.concentration
        inten = np.abs(np.fft.fftn(a)) ** 2 / len(self.sigma)
        ops = cubic_rotations() if self.symmetrize else np.eye(3, dtype=int)[None]
        idx = np.stack(np.meshgrid(*(np.arange(n),) * 3, indexing="ij"), -1).reshape(-1, 3)
        sym = np.zeros(n**3)
        for op in ops:
            s = np.mod(idx @ op.T, n)
            sym += inten[s[:, 0], s[:, 1], s[:, 2]]
        sym = (sym / len(ops)).reshape((n,) * 3)
        if smooth and self.resolution > 0:
            sym = ndimage.gaussian_filter(sym, self.resolution, mode="wrap")
        laue = self.concentration * (1 - self.concentration)
        frame = _zone_frame(normal)
        u, v = frame[0], frame[1]
        steps = np.arange(-extent * self.cells, extent * self.cells + 1)
        su, sv = np.meshgrid(steps, steps, indexing="ij")
        pts = su[..., None] * u + sv[..., None] * v  # grid units (cells per 1/a)
        img = ndimage.map_coordinates(
            sym, np.moveaxis(pts, -1, 0).reshape(3, -1), order=1, mode="grid-wrap"
        ).reshape(su.shape)
        frac = pts / self.cells
        node_dist = np.linalg.norm(frac - np.round(frac), axis=-1)  # 1/a units
        return img / laue, (u, v), node_dist

    # ----------------------------------------------------------------- plotting

    def plot_images(
        self, quantiles: tuple[float, float] = (0.86, 0.98), cmap: str = "turbo_black", **kwargs
    ):
        """Binned patterns on a linear scale. Once a mask is set, the scale spans min to max of the
        diffuse region of each pattern; before that, ``quantiles`` of the whole pattern."""
        from quantem.core.visualization import show_2d

        arrays, norms = [], []
        for i, im in enumerate(self.images):
            b = self._binned(im)
            if self.mask is not None:
                q = np.linalg.norm(self.mask["k"][i], axis=-1)
                sel = (self.mask["w"][i] > 0.5) & (q < self.mask["q_max"])
                vals = b[sel]
                norms.append(dict(interval_type="manual", vmin=vals.min(), vmax=vals.max()))
            else:
                lo, hi = np.quantile(b, quantiles)
                norms.append(dict(interval_type="manual", vmin=lo, vmax=hi))
            arrays.append(b)
        return show_2d(
            arrays,
            title=self.names,
            norm=norms,
            cmap=cmap,
            axsize=kwargs.pop("axsize", (6, 4)),
            **kwargs,
        )

    def plot_geometry(self, power: float = 0.3, q_view: float = 1.4, **kwargs):
        """Each pattern with its fitted reflections (red), direct beam (cyan) and Laue circle (yellow)."""
        import matplotlib.patches as mpatches

        from quantem.core.visualization import show_2d

        fig, axs = show_2d(
            self.images,
            title=self.names,
            norm={
                "stretch_type": "power",
                "power": power,
                "lower_quantile": 0.05,
                "upper_quantile": 0.999,
            },
            axsize=kwargs.pop("axsize", (5, 5)),
            **kwargs,
        )
        axs = np.atleast_1d(axs).ravel()
        r = 0.03 / self.sampling
        k_wave = 1.0 / self.wavelength
        for i, ax in enumerate(axs):
            p = self.bragg_positions(i)
            c = self.geometry["centers"][i]
            for pr, pc in p:
                ax.add_patch(mpatches.Circle((pc, pr), r, fill=False, color="tab:red", lw=1.0))
            ax.add_patch(mpatches.Circle((c[1], c[0]), 1.5 * r, fill=False, color="cyan", lw=1.5))
            tilt = self.geometry["tilts"][i]
            if np.linalg.norm(tilt) > 0:
                th = np.linspace(0, 2 * np.pi, 361)
                circ = -k_wave * tilt + k_wave * np.linalg.norm(tilt) * np.column_stack(
                    [np.cos(th), np.sin(th)]
                )
                px = (
                    c
                    + (circ * self.lattice_parameter / self._a_crystal)
                    @ self.geometry["matrices"][i].T
                )
                ax.plot(px[:, 1], px[:, 0], "--", color="yellow", lw=1.0)
            half = q_view / self.sampling
            ax.set_xlim(c[1] - half, c[1] + half)
            ax.set_ylim(c[0] + half, c[0] - half)
        return fig, axs

    def plot_mask(self, **kwargs):
        """Binned diffuse weight of every pattern."""
        from quantem.core.visualization import show_2d

        return show_2d(
            self.mask["w"],
            title=self.names,
            cmap="gray",
            axsize=kwargs.pop("axsize", (4, 2.7)),
            **kwargs,
        )

    def plot_fit(
        self,
        diffuse_only: bool = False,
        sigma: float = 0.7,
        quantiles: tuple[float, float] = (0.01, 0.99),
        cmap: str = "turbo_black",
        **kwargs,
    ):
        """Experiment (left) and model (right) for every zone on one linear scale per row.

        Default: the binned pattern (blurred by ``sigma`` binned pixels) and
        the full model. ``diffuse_only``: the experiment minus the fitted
        background next to the supercell's diffuse term, then their
        difference on a diverging map (white = no difference). The scale
        spans ``quantiles`` of the experiment inside the diffuse mask (weight
        > 0.5), which excludes the Bragg peaks and the direct-beam bloom;
        masked pixels are black in the diffuse view. Panels are cropped to
        ``q_max``.
        """
        import matplotlib

        from quantem.core.visualization import show_2d

        model = self.model_images(diffuse_only=diffuse_only)
        background = self.background_images() if diffuse_only else None
        cmap_obj = matplotlib.colormaps[cmap].with_extremes(bad="black")
        diverging = matplotlib.colormaps["RdBu_r"].with_extremes(bad="black")
        rows, titles, norms, cmaps = [], [], [], []
        for i in range(len(self.images)):
            q = np.linalg.norm(self.mask["k"][i], axis=-1)
            inside = q < self.mask["q_max"]
            w = self.mask["w"][i]
            sel = (w > 0.5) & inside
            rr, cc = np.nonzero(inside)
            crop = (slice(rr.min(), rr.max() + 1), slice(cc.min(), cc.max() + 1))
            if diffuse_only:
                exp = self.mask["y"][i] - background[i]
                if sigma:
                    exp = _nan_blur(np.where(w > 0.2, exp, np.nan), sigma)
                exp = np.where(sel, exp, np.nan)
                mod = np.where(sel, model[i], np.nan)
                lo, hi = np.nanquantile(exp, quantiles)
                h = 0.5 * (hi - lo)
                rows.append([exp[crop], mod[crop], (exp - mod)[crop]])
                titles.append(
                    [
                        f"{self.names[i]} experiment - background",
                        f"{self.names[i]} model",
                        "experiment - background - model",
                    ]
                )
                norms.append(
                    [dict(interval_type="manual", vmin=lo, vmax=hi)] * 2
                    + [dict(interval_type="manual", vmin=-h, vmax=h)]
                )
                cmaps.append([cmap_obj, cmap_obj, diverging])
            else:
                exp = self.mask["data"][i]
                if sigma:
                    exp = ndimage.gaussian_filter(exp, sigma)
                lo, hi = np.quantile(exp[sel], quantiles)
                rows.append(
                    [np.where(inside, exp, np.nan)[crop], np.where(inside, model[i], np.nan)[crop]]
                )
                titles.append([f"{self.names[i]} experiment", f"{self.names[i]} model"])
                norms.append([dict(interval_type="manual", vmin=lo, vmax=hi)] * 2)
                cmaps.append([cmap_obj, cmap_obj])
        return show_2d(
            rows,
            title=titles,
            norm=norms,
            cmap=cmaps,
            axsize=kwargs.pop("axsize", (4, 4)),
            **kwargs,
        )

    def plot_sro(self, n_shells: int = 8, extent: float = 2.0, layer: int = 0):
        """Short-range order three ways.

        Left: Warren-Cowley alpha against neighbour distance (alpha < 0
        prefers unlike neighbours, > 0 like). Middle: the symmetrized diffuse
        intensity of the supercell in Laue units (1 = random alloy) on the
        (001) and (1-10) reciprocal planes, where ordering shows as diffuse
        maxima at special points, e.g. 100 (B2-type) or 1/2 1/2 1/2 (D0_3); the color
        scale is set away from the reciprocal-lattice nodes, so clustering (small-q
        intensity on the nodes) saturates.
        Right: one (001) layer of the supercell, B atoms dark, and its local
        B concentration over the first two shells.
        """
        import matplotlib.pyplot as plt

        sro = self.warren_cowley(n_shells)
        sec_001, _, d_001 = self.diffuse_section((0, 0, 1), extent)
        sec_110, _, d_110 = self.diffuse_section((1, -1, 0), extent)
        n = self.grid_size
        d = self.grid_divisor
        occ = np.full((n,) * 3, np.nan)
        xs = self.site_x
        occ[xs[:, 0], xs[:, 1], xs[:, 2]] = self.sigma
        site = ~np.isnan(occ)
        local = ndimage.gaussian_filter(np.nan_to_num(occ), d * 0.6, mode="wrap") / np.maximum(
            ndimage.gaussian_filter(site.astype(float), d * 0.6, mode="wrap"), 1e-9
        )
        z = layer * d
        sl = occ[:, :, z]
        sl_sites = site[:, :, z]

        fig, axs = plt.subplots(1, 5, figsize=(22, 4.4))
        ax = axs[0]
        ax.axhline(0, color="0.6", lw=0.8)
        ax.stem(sro["radius"], sro["alpha"], basefmt=" ")
        ax.set_xlabel("neighbour distance (A)")
        ax.set_ylabel(f"Warren-Cowley alpha ({sro['pair']})")
        lim = max(0.1, 1.2 * np.abs(sro["alpha"]).max())
        ax.set_ylim(-lim, lim)
        ext = [-extent, extent, -extent, extent]
        # scale to the diffuse between the nodes, not the small-q peaks on them
        between = np.concatenate([sec_001[d_001 > 0.2], sec_110[d_110 > 0.2]])
        vmax = np.quantile(between, 0.995)
        for ax, sec, title, xl, yl in (
            (axs[1], sec_001, "(001) section", "h", "k"),
            (axs[2], sec_110, "(1-10) section", "[001]", "[110]/sqrt2"),
        ):
            im = ax.imshow(sec.T, origin="lower", extent=ext, cmap="magma", vmin=0, vmax=vmax)
            ax.set_title(f"{title}, Laue units")
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            fig.colorbar(im, ax=ax, fraction=0.046)
        rr, cc = np.nonzero(sl_sites)
        axs[3].scatter(
            cc / d, rr / d, c=sl[rr, cc], cmap="gray_r", s=6, vmin=-0.2, vmax=1.2, marker="s"
        )
        axs[3].set_aspect("equal")
        axs[3].set_title(f"(001) layer, {self.species[1]} dark")
        axs[3].set_xlabel("cells")
        im = axs[4].imshow(
            local[:, :, z],
            cmap="RdBu_r",
            vmin=self.concentration - 0.3,
            vmax=self.concentration + 0.3,
            origin="lower",
            extent=[0, self.cells, 0, self.cells],
        )
        axs[4].set_title(f"local {self.species[1]} fraction")
        fig.colorbar(im, ax=axs[4], fraction=0.046)
        fig.tight_layout()
        return fig, axs

    def plot_loss(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(self.loss_history, "k.-")
        ax.set_xlabel("sweep")
        ax.set_ylabel("weighted loss")
        ax.set_yscale("log")
        fig.tight_layout()
        return fig, ax


def _lsq_bounded(X: np.ndarray, y: np.ndarray, sw: np.ndarray, lo: np.ndarray) -> np.ndarray:
    """Weighted least squares with lower bounds, columns normalized for conditioning."""
    Xw = X * sw[:, None]
    norm = np.linalg.norm(Xw, axis=0)
    norm[norm == 0] = 1.0
    res = optimize.lsq_linear(Xw / norm, y * sw, bounds=(lo * norm, np.inf))
    return res.x / norm


def _integrate_spots(im: np.ndarray, positions: np.ndarray, radius: float) -> np.ndarray:
    """Integrated intensity inside ``radius`` px of each position, over the median of the ring
    out to 1.6 radius; NaN for spots off the detector."""
    out = np.full(len(positions), np.nan)
    r_out = int(np.ceil(1.6 * radius)) + 1
    yy, xx = np.mgrid[-r_out : r_out + 1, -r_out : r_out + 1]
    for n, (pr, pc) in enumerate(positions):
        r0, c0 = int(round(pr)), int(round(pc))
        if (
            r0 - r_out < 0
            or c0 - r_out < 0
            or r0 + r_out >= im.shape[0]
            or c0 + r_out >= im.shape[1]
        ):
            continue
        win = im[r0 - r_out : r0 + r_out + 1, c0 - r_out : c0 + r_out + 1]
        dist = np.hypot(yy + r0 - pr, xx + c0 - pc)
        bg = np.median(win[(dist >= radius) & (dist < 1.6 * radius)])
        out[n] = (win[dist < radius] - bg).sum()
    return out


def _nan_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur that ignores NaNs."""
    ok = np.isfinite(a)
    num = ndimage.gaussian_filter(np.where(ok, a, 0.0), sigma)
    den = ndimage.gaussian_filter(ok.astype(float), sigma)
    return np.where(ok, num / np.maximum(den, 1e-12), np.nan)
