from collections.abc import Sequence
from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray

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
        plot_rotation_optimization = True,
        plot_com_raw = False,
        plot_com_True = False,
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
        if I.ndim != 4:
            raise ValueError("Expected 4D dataset array with shape (Nx, Ny, Nkx, Nky).")

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




        # plotting
        if plot_com_raw:
            show_2d(
                [
                    self.com_x_raw,
                    self.com_y_raw,
                ],
                cmap = 'RdBu_r',
                **kwargs,
            )



        return self









