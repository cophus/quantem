from __future__ import annotations

from os import PathLike
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import curve_fit

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.validators import ensure_valid_array
from quantem.core.visualization import show_2d


class DiffractionMoire(AutoSerialize):
    _token = object()

    def __init__(
        self,
        diffraction: Dataset2d,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError("Use DiffractionMoire.from_data() or .from_file() to instantiate this class.")
        self._diffraction = diffraction
        self.peaks = np.zeros((0, 2), dtype=float)

    @classmethod
    def from_file(
        cls,
        file_path: str | PathLike,
        file_type: str | None = None,
    ) -> Self:
        diffraction = Dataset2d.from_file(str(file_path), file_type=file_type)
        return cls.from_data(diffraction)

    @classmethod
    def from_data(
        cls,
        diffraction: Dataset2d | NDArray,
        *,
        name: str | None = None,
        origin: NDArray | float | int | None = None,
        sampling: NDArray | float | int | None = None,
        units: list[str] | None = None,
        signal_units: str = "arb. units",
    ) -> Self:
        if isinstance(diffraction, Dataset2d):
            ds = diffraction
            if name is not None:
                ds.name = name
        else:
            arr = ensure_valid_array(diffraction, ndim=2)
            ds = Dataset2d.from_array(
                arr,
                name=name if name is not None else "diffraction",
                origin=origin if origin is not None else np.zeros(2),
                sampling=sampling if sampling is not None else np.ones(2),
                units=units if units is not None else ["pixels"] * 2,
                signal_units=signal_units,
            )

        return cls(
            diffraction=ds,
            _token=cls._token,
        )

    @property
    def diffraction(self) -> Dataset2d:
        return self._diffraction

    def find_peaks(
        self,
        *,
        threshold: float | None = None,
        min_distance_px: float | None = None,
        max_peaks: int | None = None,
        sigma: float = 0.0,
        mask_diffraction: NDArray | None = None,
        refine_subpixel: bool = True,
        refine_radius_px: int = 4,
        refine_maxfev: int = 500,
        plot_result: bool = True,
        plot_radius_min_px: float = 2.0,
        plot_radius_max_px: float = 10.0,
        plot_radius_power: float = 0.5,
        plot_edgecolor: str = "r",
        plot_linewidth: float = 1.0,
        plot_alpha: float = 0.5,
        plot_labels: bool = False,
        plot_label_color: Any = (0, 0.9, 1.0),
        plot_label_size: float = 6.0,
        plot_label_alpha: float = 0.9,
        plot_label_offset_px: tuple[float, float] = (0.0, -12.0),
        plot_label_bbox: bool = True,
        plot_label_bbox_facecolor: Any = (0.0, 0.0, 0.0, 0.35),
        plot_label_bbox_edgecolor: Any = "none",
        plot_label_bbox_pad: float = 0.15,
        plot_label_top_k: int | None = None,
        return_peaks: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        im = np.asarray(self.diffraction.array)
        if im.ndim != 2:
            raise ValueError("DiffractionMoire.find_peaks requires a 2D diffraction array.")

        im_s = gaussian_filter(im, sigma=float(sigma)) if float(sigma) > 0 else im

        if threshold is None:
            mu = float(np.nanmean(im_s))
            sig = float(np.nanstd(im_s))
            threshold = mu + 5.0 * sig

        if min_distance_px is None or float(min_distance_px) <= 0:
            win = 3
            r2_min = None
        else:
            win = int(np.ceil(float(min_distance_px)) * 2 + 1)
            win = max(3, win)
            r2_min = float(min_distance_px) ** 2

        mx = maximum_filter(im_s, size=win, mode="nearest")
        cand = (im_s == mx) & (im_s >= float(threshold))

        if mask_diffraction is not None:
            m = np.asarray(mask_diffraction).astype(bool)
            if m.shape != im.shape:
                raise ValueError("mask_diffraction must have the same shape as the diffraction array.")
            cand &= m

        peaks_rc_int = np.argwhere(cand)
        if peaks_rc_int.size == 0:
            self.peaks = np.zeros((0, 2), dtype=float)
            out = {
                "peaks": self.peaks,
                "I": np.zeros((0,), dtype=float),
                "peaks_refined": None,
                "I_refined": None,
                "gaussian_fit_params": None,
                "gaussian_fit_success": None,
                "threshold": float(threshold),
                "sigma": float(sigma),
                "min_distance_px": None if min_distance_px is None else float(min_distance_px),
            }
            if plot_result:
                fig, ax = show_2d(im_s, **kwargs)
                ax0 = ax.flat[0] if hasattr(ax, "flat") else ax
                ax0.figure.canvas.draw_idle()
            if return_peaks:
                return out
            return out

        peak_I = im[peaks_rc_int[:, 0], peaks_rc_int[:, 1]].astype(float, copy=False)
        order = np.argsort(-peak_I)
        peaks_rc_int = peaks_rc_int[order]
        peak_I = peak_I[order]

        if r2_min is not None and peaks_rc_int.shape[0] > 1:
            del_mask = np.zeros((peaks_rc_int.shape[0],), dtype=bool)
            row = peaks_rc_int[:, 0].astype(float, copy=False)
            col = peaks_rc_int[:, 1].astype(float, copy=False)
            for i in range(peaks_rc_int.shape[0] - 1):
                if del_mask[i]:
                    continue
                dr = row[i] - row[(i + 1) :]
                dc = col[i] - col[(i + 1) :]
                d2 = dr * dr + dc * dc
                j = np.flatnonzero(d2 < r2_min)
                if j.size:
                    del_mask[(i + 1) + j] = True
            keep = ~del_mask
            peaks_rc_int = peaks_rc_int[keep]
            peak_I = peak_I[keep]

        if max_peaks is not None and peaks_rc_int.shape[0] > int(max_peaks):
            peaks_rc_int = peaks_rc_int[: int(max_peaks)]
            peak_I = peak_I[: int(max_peaks)]

        peaks_rc = peaks_rc_int.astype(float, copy=False)
        self.peaks = peaks_rc

        out: dict[str, Any] = {
            "peaks": peaks_rc,
            "I": peak_I,
            "peaks_refined": None,
            "I_refined": None,
            "gaussian_fit_params": None,
            "gaussian_fit_success": None,
            "threshold": float(threshold),
            "sigma": float(sigma),
            "min_distance_px": None if min_distance_px is None else float(min_distance_px),
        }

        if refine_subpixel and peaks_rc.shape[0] > 0:
            r = int(refine_radius_px)
            if r < 1:
                raise ValueError("refine_radius_px must be >= 1.")

            def _gauss2d(coords: NDArray, A: float, row0: float, col0: float, s: float, c: float) -> NDArray:
                rr, cc = coords
                return A * np.exp(-((rr - row0) ** 2 + (cc - col0) ** 2) / (2.0 * s * s)) + c

            peaks_rc_ref = peaks_rc.copy()
            I_ref = peak_I.copy()
            fit_params = np.full((peaks_rc.shape[0], 5), np.nan, dtype=float)
            fit_success = np.zeros((peaks_rc.shape[0],), dtype=bool)

            for k in range(peaks_rc.shape[0]):
                row0_i = float(peaks_rc[k, 0])
                col0_i = float(peaks_rc[k, 1])
                row_i = int(np.round(row0_i))
                col_i = int(np.round(col0_i))

                r0 = max(0, row_i - r)
                r1 = min(im_s.shape[0], row_i + r + 1)
                c0 = max(0, col_i - r)
                c1 = min(im_s.shape[1], col_i + r + 1)
                if (r1 - r0) < 3 or (c1 - c0) < 3:
                    continue

                patch = im_s[r0:r1, c0:c1].astype(float, copy=False)
                rr, cc = np.mgrid[r0:r1, c0:c1]
                xdata = np.vstack([rr.ravel(), cc.ravel()])
                ydata = patch.ravel()

                c_init = float(np.median(ydata))
                A_init = float(np.max(ydata) - c_init)
                if not np.isfinite(A_init) or A_init <= 0:
                    continue
                s_init = max(0.75, float(r) / 2.0)

                row_min = float(r0)
                row_max = float(r1 - 1)
                col_min = float(c0)
                col_max = float(c1 - 1)

                p0 = (A_init, row0_i, col0_i, s_init, c_init)
                lb = (0.0, row_min, col_min, 0.5, float(np.min(ydata) - abs(A_init)))
                ub = (
                    float(np.inf),
                    row_max,
                    col_max,
                    float(max(0.75, 2.0 * r)),
                    float(np.max(ydata) + abs(A_init)),
                )

                try:
                    popt, _ = curve_fit(
                        _gauss2d,
                        xdata,
                        ydata,
                        p0=p0,
                        bounds=(lb, ub),
                        maxfev=int(refine_maxfev),
                    )
                except Exception:
                    continue

                A_fit, row_fit, col_fit, s_fit, c_fit = (
                    float(popt[0]),
                    float(popt[1]),
                    float(popt[2]),
                    float(popt[3]),
                    float(popt[4]),
                )
                if not (np.isfinite(row_fit) and np.isfinite(col_fit) and np.isfinite(s_fit) and s_fit > 0):
                    continue

                peaks_rc_ref[k, 0] = row_fit
                peaks_rc_ref[k, 1] = col_fit
                I_ref[k] = max(A_fit + c_fit, float(im[row_i, col_i]))
                fit_params[k, :] = np.array([A_fit, row_fit, col_fit, s_fit, c_fit], dtype=float)
                fit_success[k] = True

            order_ref = np.argsort(-I_ref)
            peaks_rc_ref = peaks_rc_ref[order_ref]
            I_ref = I_ref[order_ref]
            fit_params = fit_params[order_ref]
            fit_success = fit_success[order_ref]

            out["peaks_refined"] = peaks_rc_ref
            out["I_refined"] = I_ref
            out["gaussian_fit_params"] = fit_params
            out["gaussian_fit_success"] = fit_success
            out["refine_radius_px"] = int(refine_radius_px)

            self.peaks = peaks_rc_ref
            out["peaks"] = peaks_rc_ref
            out["I"] = I_ref

        if plot_result:
            from matplotlib.patches import Circle

            fig, ax = show_2d(im_s, **kwargs)
            ax0 = ax.flat[0] if hasattr(ax, "flat") else ax

            peaks_plot = np.asarray(out["peaks"], dtype=float)
            I_plot = np.asarray(out["I"], dtype=float)

            if peaks_plot.size:
                I_min = float(np.nanmin(I_plot)) if I_plot.size else 0.0
                I_max = float(np.nanmax(I_plot)) if I_plot.size else 1.0
                denom = I_max - I_min
                if not np.isfinite(denom) or denom <= 0:
                    t = np.ones_like(I_plot, dtype=float)
                else:
                    t = np.clip((I_plot - I_min) / denom, 0.0, 1.0)

                t = np.power(t, float(plot_radius_power))
                radii = float(plot_radius_min_px) + (float(plot_radius_max_px) - float(plot_radius_min_px)) * t

                for (row, col), rad in zip(peaks_plot, radii):
                    ax0.add_patch(
                        Circle(
                            (col, row),
                            radius=float(rad),
                            facecolor="none",
                            edgecolor=plot_edgecolor,
                            linewidth=float(plot_linewidth),
                            alpha=float(plot_alpha),
                        )
                    )

                if plot_labels:
                    dcol, drow = float(plot_label_offset_px[0]), float(plot_label_offset_px[1])
                    bbox_kw = None
                    if plot_label_bbox:
                        bbox_kw = dict(
                            boxstyle=f"round,pad={float(plot_label_bbox_pad)}",
                            facecolor=plot_label_bbox_facecolor,
                            edgecolor=plot_label_bbox_edgecolor,
                        )

                    nlab = peaks_plot.shape[0]
                    if plot_label_top_k is not None:
                        nlab = min(nlab, int(plot_label_top_k))

                    for idx in range(nlab):
                        row, col = float(peaks_plot[idx, 0]), float(peaks_plot[idx, 1])
                        ax0.text(
                            float(col + dcol),
                            float(row + drow),
                            str(idx),
                            color=plot_label_color,
                            fontsize=float(plot_label_size),
                            alpha=float(plot_label_alpha),
                            ha="center",
                            va="center",
                            bbox=bbox_kw,
                        )

            ax0.figure.canvas.draw_idle()

        if return_peaks:
            return out

    def lattice_vectors(
        self,
        ind_origin=None,
        ind_u1=None,
        ind_v1=None,
        ind_u2=None,
        ind_v2=None,
        max_tile=None,
        moire_draw: bool = False,
        moire_exclude_radius_px: float = 2.0,
        moire_parent_maxlen_px: float | None = None,
        moire_round_decimals: int = 3,
        return_peaks: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        if not hasattr(self, "peaks"):
            raise RuntimeError("No peaks found. Run .find_peaks() first.")
        peaks = np.asarray(self.peaks, dtype=float)
        if peaks.ndim != 2 or peaks.shape[1] != 2:
            raise RuntimeError("self.peaks must have shape (N, 2) with columns [row, col].")
        if peaks.shape[0] == 0:
            raise RuntimeError("No peaks available in self.peaks.")

        im = np.asarray(self.diffraction.array)
        if im.ndim != 2:
            raise ValueError("DiffractionMoire.lattice_vectors requires a 2D diffraction array.")
        nrow, ncol = int(im.shape[0]), int(im.shape[1])

        def _as_int(x, name: str) -> int:
            if x is None:
                raise ValueError(f"{name} must be provided (int index into self.peaks).")
            xi = int(x)
            if xi < 0 or xi >= peaks.shape[0]:
                raise IndexError(f"{name}={xi} out of range for self.peaks with N={peaks.shape[0]}.")
            return xi

        def _pair_ok(a, b) -> bool:
            return (a is not None) and (b is not None)

        def _pair_half(a, b) -> bool:
            return (a is None) ^ (b is None)

        if _pair_half(ind_u1, ind_v1):
            raise ValueError("Provide both ind_u1 and ind_v1, or neither.")
        if _pair_half(ind_u2, ind_v2):
            raise ValueError("Provide both ind_u2 and ind_v2, or neither.")

        have_lattice1 = _pair_ok(ind_u1, ind_v1)
        have_lattice2 = _pair_ok(ind_u2, ind_v2)
        if not (have_lattice1 or have_lattice2):
            raise ValueError("Provide lattice indices for at least one lattice (u/v).")

        ind_origin_i = _as_int(ind_origin, "ind_origin")
        r0 = peaks[ind_origin_i].astype(float, copy=False)

        if have_lattice1:
            ind_u1_i = _as_int(ind_u1, "ind_u1")
            ind_v1_i = _as_int(ind_v1, "ind_v1")
            u1 = peaks[ind_u1_i].astype(float, copy=False) - r0
            v1 = peaks[ind_v1_i].astype(float, copy=False) - r0
            if np.linalg.norm(u1) == 0 or np.linalg.norm(v1) == 0:
                raise ValueError("Lattice 1 vectors u1/v1 must be non-zero (origin and vector peaks must differ).")
        else:
            ind_u1_i = None
            ind_v1_i = None
            u1 = None
            v1 = None

        if have_lattice2:
            ind_u2_i = _as_int(ind_u2, "ind_u2")
            ind_v2_i = _as_int(ind_v2, "ind_v2")
            u2 = peaks[ind_u2_i].astype(float, copy=False) - r0
            v2 = peaks[ind_v2_i].astype(float, copy=False) - r0
            if np.linalg.norm(u2) == 0 or np.linalg.norm(v2) == 0:
                raise ValueError("Lattice 2 vectors u2/v2 must be non-zero (origin and vector peaks must differ).")
        else:
            ind_u2_i = None
            ind_v2_i = None
            u2 = None
            v2 = None

        if moire_draw and not (have_lattice1 and have_lattice2):
            raise ValueError("moire_draw=True requires both lattices (u1/v1 and u2/v2).")

        def _default_max_tile(u: NDArray, v: NDArray) -> int:
            du = float(np.linalg.norm(u))
            dv = float(np.linalg.norm(v))
            dmin = min(du, dv) if (du > 0 and dv > 0) else max(du, dv)
            if not np.isfinite(dmin) or dmin <= 0:
                return 10
            diag = float(np.hypot(nrow, ncol))
            mt = int(np.ceil(diag / dmin)) + 2
            return int(np.clip(mt, 3, 200))

        if max_tile is None:
            mt_list: list[int] = []
            if have_lattice1:
                mt_list.append(_default_max_tile(u1, v1))
            if have_lattice2:
                mt_list.append(_default_max_tile(u2, v2))
            max_tile_use = max(mt_list) if mt_list else 10
        else:
            max_tile_use = int(max_tile)
            if max_tile_use < 0:
                raise ValueError("max_tile must be >= 0.")

        def _tile_points_bounded(r0: NDArray, u: NDArray, v: NDArray, mt: int) -> NDArray:
            if mt == 0:
                return r0[None, :].copy()
            a = np.arange(-mt, mt + 1, dtype=float)
            b = np.arange(-mt, mt + 1, dtype=float)
            aa, bb = np.meshgrid(a, b, indexing="ij")
            pts = r0[None, None, :] + aa[..., None] * u[None, None, :] + bb[..., None] * v[None, None, :]
            pts = pts.reshape(-1, 2)
            keep = (
                (pts[:, 0] >= 0.0)
                & (pts[:, 0] <= (nrow - 1))
                & (pts[:, 1] >= 0.0)
                & (pts[:, 1] <= (ncol - 1))
            )
            return pts[keep]

        pts1 = _tile_points_bounded(r0, u1, v1, max_tile_use) if have_lattice1 else None
        pts2 = _tile_points_bounded(r0, u2, v2, max_tile_use) if have_lattice2 else None

        pts_moire = None
        moire_info: dict[str, Any] | None = None

        if moire_draw:
            maxlen = float(moire_parent_maxlen_px) if moire_parent_maxlen_px is not None else float(0.5 * np.hypot(nrow, ncol))
            if not np.isfinite(maxlen) or maxlen <= 0:
                raise ValueError("moire_parent_maxlen_px must be a positive finite number.")

            def _disp_set(u: NDArray, v: NDArray, maxlen: float) -> tuple[NDArray, NDArray]:
                dmin = float(min(np.linalg.norm(u), np.linalg.norm(v)))
                if not np.isfinite(dmin) or dmin <= 0:
                    raise ValueError("Invalid lattice vectors for moire generation.")
                mt = int(np.ceil(maxlen / dmin)) + 1
                mt = int(np.clip(mt, 1, 200))
                a = np.arange(-mt, mt + 1, dtype=float)
                b = np.arange(-mt, mt + 1, dtype=float)
                aa, bb = np.meshgrid(a, b, indexing="ij")
                aa = aa.ravel()
                bb = bb.ravel()
                disp = aa[:, None] * u[None, :] + bb[:, None] * v[None, :]
                dn = np.sqrt(np.sum(disp**2, axis=1))
                keep = dn <= maxlen + 1e-9
                aa = aa[keep]
                bb = bb[keep]
                disp = disp[keep]
                return disp, np.stack([aa, bb], axis=1)

            disp1, ab1 = _disp_set(u1, v1, maxlen)
            disp2, ab2 = _disp_set(u2, v2, maxlen)

            if disp1.size and disp2.size:
                n1 = int(disp1.shape[0])
                n2 = int(disp2.shape[0])
                if n1 * n2 > 2_000_000:
                    raise ValueError(
                        f"Moire candidate count too large (N1*N2={n1*n2}). "
                        "Reduce moire_parent_maxlen_px to limit the search space."
                    )

                pts = r0[None, None, :] + disp1[:, None, :] + disp2[None, :, :]
                pts = pts.reshape(-1, 2)

                keep_combo = ~(
                    ((ab1[:, 0] == 0) & (ab1[:, 1] == 0))[:, None]
                    | ((ab2[:, 0] == 0) & (ab2[:, 1] == 0))[None, :]
                ).ravel()
                pts = pts[keep_combo]

                keep_bounds = (
                    (pts[:, 0] >= 0.0)
                    & (pts[:, 0] <= (nrow - 1))
                    & (pts[:, 1] >= 0.0)
                    & (pts[:, 1] <= (ncol - 1))
                )
                pts = pts[keep_bounds]

                if pts.size:
                    pts = np.unique(np.round(pts, int(moire_round_decimals)), axis=0)

                    r_excl = float(moire_exclude_radius_px)
                    if r_excl > 0:
                        parent_pts = np.vstack((r0[None, :] + disp1, r0[None, :] + disp2))
                        parent_keep = (
                            (parent_pts[:, 0] >= 0.0)
                            & (parent_pts[:, 0] <= (nrow - 1))
                            & (parent_pts[:, 1] >= 0.0)
                            & (parent_pts[:, 1] <= (ncol - 1))
                        )
                        parent_pts = parent_pts[parent_keep]
                        if parent_pts.size:
                            try:
                                from scipy.spatial import cKDTree

                                tree = cKDTree(parent_pts)
                                d, _ = tree.query(pts, k=1, workers=-1)
                                pts = pts[d >= r_excl]
                            except Exception:
                                r2 = r_excl * r_excl
                                keep = np.ones(pts.shape[0], dtype=bool)
                                for i in range(pts.shape[0]):
                                    d2 = np.sum((parent_pts - pts[i]) ** 2, axis=1)
                                    if np.any(d2 < r2):
                                        keep[i] = False
                                pts = pts[keep]

                    pts_moire = pts if pts.size else None

            moire_info = {
                "moire_parent_maxlen_px": maxlen,
                "moire_exclude_radius_px": float(moire_exclude_radius_px),
                "moire_round_decimals": int(moire_round_decimals),
                "moire_n_disp1": int(disp1.shape[0]),
                "moire_n_disp2": int(disp2.shape[0]),
            }

        data: dict[str, Any] = {
            "origin_index": ind_origin_i,
            "origin_rc": r0.copy(),
            "max_tile": max_tile_use,
            "lattice1_points_rc": pts1,
            "lattice2_points_rc": pts2,
            "moire_points_rc": pts_moire,
            "moire_info": moire_info,
        }

        if have_lattice1:
            data.update(
                {
                    "u1_index": ind_u1_i,
                    "v1_index": ind_v1_i,
                    "u1_rc": u1.copy(),
                    "v1_rc": v1.copy(),
                }
            )

        if have_lattice2:
            data.update(
                {
                    "u2_index": ind_u2_i,
                    "v2_index": ind_v2_i,
                    "u2_rc": u2.copy(),
                    "v2_rc": v2.copy(),
                }
            )

        self.lattice_vectors_data = data

        fig, ax = show_2d(im, **kwargs)
        ax0 = ax.flat[0] if hasattr(ax, "flat") else ax

        c1 = (1.0, 0.0, 0.0)
        c2 = (0.0, 0.8, 1.0)
        cm = (0.8, 0.0, 1.0)
        c0 = (0.0, 1.0, 0.0)

        lw_vec = 1.0
        lw_pts = 1.0

        ax0.scatter([r0[1]], [r0[0]], s=90, marker="x", c=[c0], linewidths=lw_vec, zorder=8)

        def _draw_vec(vec: NDArray, color):
            p = r0 + vec
            ax0.plot([r0[1], p[1]], [r0[0], p[0]], color=color, linewidth=lw_vec, zorder=7)
            ax0.scatter([p[1]], [p[0]], s=45, marker="o", facecolors="none", edgecolors=[color], linewidths=lw_vec, zorder=8)

        if have_lattice1:
            _draw_vec(u1, c1)
            _draw_vec(v1, c1)

        if have_lattice2:
            _draw_vec(u2, c2)
            _draw_vec(v2, c2)

        if pts1 is not None and pts1.size:
            ax0.scatter(
                pts1[:, 1],
                pts1[:, 0],
                s=22,
                marker="o",
                facecolors="none",
                edgecolors=[c1],
                linewidths=lw_pts,
                alpha=0.8,
                zorder=5,
            )

        if pts2 is not None and pts2.size:
            ax0.scatter(
                pts2[:, 1],
                pts2[:, 0],
                s=22,
                marker="o",
                facecolors="none",
                edgecolors=[c2],
                linewidths=lw_pts,
                alpha=0.8,
                zorder=5,
            )

        if pts_moire is not None and pts_moire.size:
            ax0.scatter(
                pts_moire[:, 1],
                pts_moire[:, 0],
                s=14,
                marker="o",
                facecolors="none",
                edgecolors=[cm],
                linewidths=lw_pts,
                alpha=0.9,
                zorder=5,
            )

        ax0.figure.canvas.draw_idle()

        if return_peaks:
            return {"data": data, "fig": fig, "ax": ax0}
        return {}
