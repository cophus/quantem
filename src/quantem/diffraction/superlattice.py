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
                ub = (float(np.inf), row_max, col_max, float(max(0.75, 2.0 * r)), float(np.max(ydata) + abs(A_init)))

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

            out["peaks_refined"] = peaks_rc_ref
            out["I_refined"] = I_ref
            out["gaussian_fit_params"] = fit_params
            out["gaussian_fit_success"] = fit_success
            out["refine_radius_px"] = int(refine_radius_px)

            self.peaks = peaks_rc_ref

        if plot_result:
            from matplotlib.patches import Circle

            fig, ax = show_2d(im_s, **kwargs)
            ax0 = ax.flat[0] if hasattr(ax, "flat") else ax

            peaks_plot = out["peaks_refined"] if out["peaks_refined"] is not None else out["peaks"]
            I_plot = out["I_refined"] if out["I_refined"] is not None else out["I"]

            peaks_plot = np.asarray(peaks_plot, dtype=float)
            I_plot = np.asarray(I_plot, dtype=float)

            if peaks_plot.size:
                I_min = float(np.nanmin(I_plot)) if I_plot.size else 0.0
                I_max = float(np.nanmax(I_plot)) if I_plot.size else 1.0
                denom = I_max - I_min
                if not np.isfinite(denom) or denom <= 0:
                    t = np.ones_like(I_plot, dtype=float)
                else:
                    t = (I_plot - I_min) / denom
                    t = np.clip(t, 0.0, 1.0)

                t = np.power(t, float(plot_radius_power))
                radii = float(plot_radius_min_px) + (float(plot_radius_max_px) - float(plot_radius_min_px)) * t

                for (row, col), rad in zip(peaks_plot, radii, strict=False):
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

            ax0.figure.canvas.draw_idle()

        if return_peaks:
            return out

    def lattice_vectors(
        self,
        peaks: NDArray | None = None,
        *,
        max_order: int | None = None,
        enforce_two_vectors: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        raise NotImplementedError
