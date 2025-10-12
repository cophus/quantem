from collections.abc import Sequence
from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from quantem.core.io.serialize import AutoSerialize
from quantem.core.datastructures.dataset4d import Dataset4d
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.visualization import show_2d


class DPC(AutoSerialize):
    """
    DPC reconstruction class

    TODO - merge utils / processing with direct ptycho branch
    For now just implement simple CoM
    """


    _token = object()

    def __init__(
        self,
        dataset: Dataset4dstem,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError(
                "Use DriftCorrection.from_data() or .from_file() to instantiate this class."
            )
        self._dataset = dataset

    @classmethod
    def from_file(
        cls,
        file_path: Sequence[str],
        file_type: str | None = None,
    ) -> "DPC":
        dataset = Dataset4dstem.from_file(file_path, file_type=file_type)
        return cls.from_data(
            dataset,
        )

    @classmethod
    def from_data(
        cls,
        dataset: Union[Dataset4dstem, Dataset4d, NDArray],
    ) -> "DPC":

        return cls(
            dataset=dataset,
            _token=cls._token,
        )

    # --- Properties ---
    @property
    def dataset(self) -> Dataset4dstem:
        return self._dataset

    @dataset.setter
    def dataset(self, value: Dataset4dstem):
        self._dataset = Dataset4dstem


    # Preprocessing center of mass
    def preprocess(
        self,
        mask_diffraction: NDArray | None = None,
        rotation_steps = 180,
        normalize_zero_com: bool = True,
        print_optimization = True,
        plot_optimization = True,
        plot_com_raw = False,
        plot_com = True,
        **kwargs,
    ) -> "DPC":
        """
        Compute raw center-of-mass (CoM) for each probe position.

        Let I(rx, ry, kx, ky) be the diffraction intensities and x,y be the pixel
        coordinates along the diffraction axes. For each (rx, ry):

            com_x = sum[I * x] / sum[I]
            com_y = sum[I * y] / sum[I]

        If `mask_diffraction` is provided (bool, shape (Nkx, Nky)), all sums are
        restricted to mask==True.

        Stores
        ------
        self.com_x_raw : (Nx, Ny) float64
        self.com_y_raw : (Nx, Ny) float64
        self.intensity_sum : (Nx, Ny) float64   # denominator (masked or unmasked)
        """

        I = self._dataset.array  # shape: (Nx, Ny, Nkx, Nky)
        Nx, Ny, Nkx, Nky = I.shape
        x = np.arange(Nkx, dtype=np.float64)          # (Nkx,)
        y = np.arange(Nky, dtype=np.float64)          # (Nky,)

        if mask_diffraction is not None:
            mask = np.asarray(mask_diffraction, dtype=bool)
            if mask.shape != (Nkx, Nky):
                raise ValueError("`mask_diffraction` must have shape (Nkx, Nky).")

            m = mask.astype(np.float64)               # (Nkx, Nky)
            wx = (x[:, None] * m)                     # (Nkx, Nky)
            wy = (y[None, :] * m)                     # (Nkx, Nky)

            # Denominator and numerators via 2D weights; no large temporaries
            denom = np.tensordot(I, m,  axes=([2, 3], [0, 1]))  # (Nx, Ny)
            num_x = np.tensordot(I, wx, axes=([2, 3], [0, 1]))  # (Nx, Ny)
            num_y = np.tensordot(I, wy, axes=([2, 3], [0, 1]))  # (Nx, Ny)

        else:
            # Unmasked: use separable reductions to minimize memory
            denom = I.sum(axis=(2, 3))                          # (Nx, Ny)

            Iy = I.sum(axis=3)                                   # (Nx, Ny, Nkx)
            num_x = np.tensordot(Iy, x, axes=([2], [0]))         # (Nx, Ny)

            Ix = I.sum(axis=2)                                   # (Nx, Ny, Nky)
            num_y = np.tensordot(Ix, y, axes=([2], [0]))         # (Nx, Ny)

        with np.errstate(divide="ignore", invalid="ignore"):
            com_x = np.divide(num_x, denom, out=np.full_like(denom, np.nan, dtype=np.float64), where=denom != 0)
            com_y = np.divide(num_y, denom, out=np.full_like(denom, np.nan, dtype=np.float64), where=denom != 0)

        self.com_x_raw = com_x
        self.com_y_raw = com_y
        self.intensity_sum = denom
        self._mask_diffraction = mask_diffraction

        # Refine rotation using curl minimization
        self._rotation_angles_deg = np.linspace(0,180,rotation_steps,endpoint=False)
        self._rotation_angles = np.deg2rad(self._rotation_angles_deg)
        cosa = np.cos(self._rotation_angles)
        sina = np.sin(self._rotation_angles)

        X = self.com_x_raw
        Y = self.com_y_raw

        # Central differences on interior (skip outermost edges)
        Xx = 0.5 * (X[2:, 1:-1] - X[:-2, 1:-1])
        Xy = 0.5 * (X[1:-1, 2:] - X[1:-1, :-2])
        Yx = 0.5 * (Y[2:, 1:-1] - Y[:-2, 1:-1])
        Yy = 0.5 * (Y[1:-1, 2:] - Y[1:-1, :-2])

        # Precompute terms used in curl after rotation
        D  = Xx + Yy       # divergence(X, Y)
        C  = Yx - Xy       # curl(X, Y)
        D2 = Yx + Xy
        C2 = Xx - Yy

        metric = np.empty_like(self._rotation_angles, dtype=np.float64)
        metric_transpose = np.empty_like(self._rotation_angles, dtype=np.float64)

        for i in range(self._rotation_angles.size):
            s, c = sina[i], cosa[i]
            curl0 = s * D  + c * C
            curl1 = s * D2 + c * C2
            metric[i] = np.mean(curl0**2)
            metric_transpose[i] = np.mean(curl1**2)

        i0 = int(np.argmin(metric))
        i1 = int(np.argmin(metric_transpose))
        use_transpose = metric_transpose[i1] < metric[i0]
        best_i = i1 if use_transpose else i0

        self._curl_metric = metric
        self._curl_metric_transpose = metric_transpose

        self._rotation_best_deg = float(self._rotation_angles_deg[best_i])
        self._rotation_best_rad = float(self._rotation_angles[best_i])
        self._rotation_best_transpose = bool(use_transpose)

        c, s = cosa[best_i], sina[best_i]
        if not use_transpose:
            self.com_x = c * X - s * Y
            self.com_y = s * X + c * Y
        else:
            self.com_x = c * Y - s * X
            self.com_y = s * Y + c * X

        if normalize_zero_com:
            self.com_x -= np.mean(self.com_x)
            self.com_y -= np.mean(self.com_y)
            
        # print the best fit optimization
        if print_optimization:
            print(f"Best rotation angle: {self._rotation_best_deg:.3f} deg; transpose: {self._rotation_best_transpose}")

        # Plot the best fit rotation
        if plot_optimization:
            angles = self._rotation_angles_deg
            metric = self._curl_metric
            metric_transpose = self._curl_metric_transpose

            # Periodic extension (duplicate the 0° point at 180°)
            angles_p = np.r_[angles, 180.0]
            metric_p = np.r_[metric, metric[0]]
            metric_t_p = np.r_[metric_transpose, metric_transpose[0]]

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(angles_p, metric_p, color=(1, 0, 0), label="not transposed")
            ax.plot(angles_p, metric_t_p, color=(0, 0.7, 1), label="transposed")

            # Green marker at global minimum
            if self._rotation_best_transpose:
                i_best = int(np.argmin(metric_transpose))
                y_best = metric_transpose[i_best]
            else:
                i_best = int(np.argmin(metric))
                y_best = metric[i_best]
            ax.plot(
                self._rotation_best_deg, 
                y_best, "o", 
                markerfacecolor=(0, 1, 0), 
                markeredgecolor=(0, 0, 0), 
                markersize=6,
            )

            ax.set_xlim(0, 180)
            ax.set_xlabel("rotation angle")
            ax.set_ylabel("mean squared curl")
            ax.legend()
            fig.tight_layout()

        # plotting images
        if plot_com_raw:
            show_2d(
                [
                    self.com_x_raw,
                    self.com_y_raw,
                ],
                title=[
                    'Raw CoM x',
                    'Raw CoM y',
                ],
                cmap = 'RdBu_r',
                **kwargs,
            )
        if plot_com:
            show_2d(
                [
                    self.com_x,
                    self.com_y,
                ],
                title=[
                    'Optmized CoM x',
                    'Optmized CoM y',
                ],
                cmap = 'RdBu_r',
                **kwargs,
            )

        return self


    def reconstruct(
        self,
        method: str = "fft_padded",
        dx: float = 1.0,
        dy: float = 1.0,
        pad_factor: float = 2.0,
        zero_mean: bool = True,
        plot_phase: bool = True,
        iter_max: int = 128,
        step: float = 0.5,
        stopping_criterion: float = 1e-6,
        backtrack: bool = True,
        q_lowpass: float | None = None,
        q_highpass: float | None = None,
        butterworth_order: float = 2.0,
        **kwargs,
    ) -> "DPC":
        """
        Reconstruct phase from CoM fields (self.com_x, self.com_y).

        Parameters
        ----------
        method : {"dct","fft","fft_padded"}, default "fft_padded"
            "dct"         : Neumann (natural) boundaries via DCT Poisson solve.
            "fft"         : Periodic boundaries via direct Fourier integration.
            "fft_padded"  : Lower-right zero pad, iterative Fourier update, crop.
        dx, dy : float
            Sampling along axis-0 and axis-1.
        pad_factor : float
            Padding factor for "fft_padded".
        zero_mean : bool
            Subtract mean from final phase.
        plot_phase : bool
            Display result with show_2d.

        Iterative controls (fft_padded)
        -------------------------------
        iter_max : int
        step : float
        stopping_criterion : float
        backtrack : bool

        Butterworth constraints (applied after reconstruction)
        ------------------------------------------------------
        q_lowpass : float | None
        q_highpass : float | None
        butterworth_order : float

        Stores
        ------
        self.phase : ndarray (H, W)
        """
        if not hasattr(self, "com_x") or not hasattr(self, "com_y"):
            raise RuntimeError("Run preprocess() to populate self.com_x / self.com_y first.")

        gx = np.asarray(self.com_x, dtype=np.float64)
        gy = np.asarray(self.com_y, dtype=np.float64)
        H, W = gx.shape

        if method == "fft":
            # Direct Fourier integration (periodic BCs)
            kx = np.fft.fftfreq(H, d=dx)[:, None]
            ky = np.fft.fftfreq(W, d=dy)[None, :]
            k2 = kx**2 + ky**2
            k2[0, 0] = np.inf
            op_x = (-1j * 0.25) * (kx / k2)
            op_y = (-1j * 0.25) * (ky / k2)
            phase = np.real(np.fft.ifft2(np.fft.fft2(gx) * op_x + np.fft.fft2(gy) * op_y))

        elif method == "fft_padded":
            # Lower-right padding; iterative integration on padded grid, then crop.
            ph = int(np.round(H * pad_factor))
            pw = int(np.round(W * pad_factor))

            pad_x = np.zeros((ph, pw), dtype=np.float64); pad_x[:H, :W] = gx
            pad_y = np.zeros((ph, pw), dtype=np.float64); pad_y[:H, :W] = gy

            mask = np.zeros((ph, pw), dtype=bool)
            mask[:H, :W] = True

            kx = np.fft.fftfreq(ph, d=dx)[:, None]
            ky = np.fft.fftfreq(pw, d=dy)[None, :]
            k2 = kx**2 + ky**2
            k2[0, 0] = np.inf
            op_x = (-1j * 0.25) * (kx / k2)
            op_y = (-1j * 0.25) * (ky / k2)

            phase_pad = np.zeros((ph, pw), dtype=np.float64)

            denom = np.mean(pad_x[mask] ** 2 + pad_y[mask] ** 2)
            if not np.isfinite(denom) or denom <= 0.0:
                denom = 1.0

            err = np.inf
            s = float(step)
            for _ in range(int(iter_max)):
                prev_phase = phase_pad.copy()

                # Centered finite-difference gradient
                grad_x = (np.roll(phase_pad, 1, axis=0) - np.roll(phase_pad, -1, axis=0)) / (2.0 * dx)
                grad_y = (np.roll(phase_pad, 1, axis=1) - np.roll(phase_pad, -1, axis=1)) / (2.0 * dy)

                # Residual in the valid (unpadded) region
                rx = np.zeros_like(grad_x); ry = np.zeros_like(grad_y)
                rx[mask] = pad_x[mask] - grad_x[mask]
                ry[mask] = pad_y[mask] - grad_y[mask]

                # Fourier integration update
                update = np.real(np.fft.ifft2(np.fft.fft2(rx) * op_x + np.fft.fft2(ry) * op_y))
                phase_pad = phase_pad + s * update

                # Error & backtracking
                grad_x = (np.roll(phase_pad, 1, axis=0) - np.roll(phase_pad, -1, axis=0)) / (2.0 * dx)
                grad_y = (np.roll(phase_pad, 1, axis=1) - np.roll(phase_pad, -1, axis=1)) / (2.0 * dy)
                err_new = np.mean((pad_x[mask] - grad_x[mask]) ** 2 + (pad_y[mask] - grad_y[mask]) ** 2) / denom

                if backtrack and (err_new > err):
                    phase_pad = prev_phase
                    s *= 0.5
                    if s < stopping_criterion:
                        break
                    continue

                err = err_new
                if s < stopping_criterion:
                    break

            phase = phase_pad[:H, :W]

        elif method == "dct":
            # Poisson solve with Neumann BCs via DCT. Sign fixed to match FFT convention.
            from scipy.fft import dctn, idctn

            ddx = np.empty_like(gx); ddy = np.empty_like(gy)
            ddx[1:-1, :] = (gx[2:, :] - gx[:-2, :]) / (2 * dx)
            ddx[0, :]    = (gx[1, :] - gx[0, :]) / dx
            ddx[-1, :]   = (gx[-1, :] - gx[-2, :]) / dx

            ddy[:, 1:-1] = (gy[:, 2:] - gy[:, :-2]) / (2 * dy)
            ddy[:, 0]    = (gy[:, 1] - gy[:, 0]) / dy
            ddy[:, -1]   = (gy[:, -1] - gy[:, -2]) / dy

            # Use negative divergence so DCT result matches FFT method orientation.
            rhs = -(ddx + ddy)

            rhs_hat = dctn(rhs, type=2, norm="ortho")
            m = np.arange(H, dtype=np.float64)[:, None]
            n = np.arange(W, dtype=np.float64)[None, :]
            lam = (2.0 - 2.0 * np.cos(np.pi * m / H)) / (dx * dx) + \
                  (2.0 - 2.0 * np.cos(np.pi * n / W)) / (dy * dy)
            lam[0, 0] = np.inf  # remove DC
            phi_hat = rhs_hat / lam
            phi_hat[0, 0] = 0.0
            phase = idctn(phi_hat, type=3, norm="ortho")

        else:
            raise ValueError("method must be one of {'dct','fft','fft_padded'}")

        # Optional Butterworth envelope on the reconstructed phase (all methods)
        if (q_lowpass is not None) or (q_highpass is not None):
            kx = np.fft.fftfreq(H, d=dx)[:, None]
            ky = np.fft.fftfreq(W, d=dy)[None, :]
            kr = np.sqrt(kx**2 + ky**2)
            env = np.ones_like(kr)
            n = float(butterworth_order)
            if q_highpass is not None:
                env *= 1.0 - 1.0 / (1.0 + (kr / float(q_highpass)) ** (2.0 * n))
            if q_lowpass is not None:
                env *= 1.0 / (1.0 + (kr / float(q_lowpass)) ** (2.0 * n))
            phase = np.real(np.fft.ifft2(np.fft.fft2(phase) * env))

        if zero_mean:
            phase = phase - np.mean(phase)

        self.phase = phase
        self._reconstruction_method = method
        self._dx = dx
        self._dy = dy
        self._pad_factor = pad_factor
        self._zero_mean = bool(zero_mean)
        self._q_lowpass = q_lowpass
        self._q_highpass = q_highpass
        self._butterworth_order = butterworth_order

        if plot_phase:
            label = {"dct": "DCT", "fft": "FFT", "fft_padded": "Padded FFT"}[method]
            show_2d(self.phase, title=f"CoM-DPC Phase ({label})", cbar=True, **kwargs)

        return self



