from __future__ import annotations
from typing import Optional, Sequence
import numpy as np
from numpy.typing import NDArray

from quantem.core.io.serialize import AutoSerialize
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.datastructures.dataset3d import Dataset3d


class Parallax(AutoSerialize):
    """
    Virtual parallax (tilted/shifted BF) builder.

    Attributes set by preprocess
    ----------------------------
    bf_images : Dataset3d
        Stack of virtual BF images with shape (Nimg, Nx, Ny).
    bf_pixel_indices : np.ndarray
        Integer (Nimg, 2) array of (kx, ky) pixel indices used for each image.
    dp_mask : np.ndarray
        Boolean mask (Nkx, Nky) selecting BF pixels.
    """

    _token = object()

    def __init__(self, dataset: Dataset4dstem, _token: object | None = None) -> None:
        if _token is not self._token:
            raise RuntimeError("Use Parallax.from_data() or .from_file() to instantiate.")
        self._dataset = dataset
        self.bf_images: Optional[Dataset3d] = None
        self.bf_pixel_indices: Optional[NDArray[np.int_]] = None
        self.dp_mask: Optional[NDArray[np.bool_]] = None

    @classmethod
    def from_file(
        cls,
        file_path: Sequence[str],
        file_type: str | None = None,
    ) -> "Parallax":
        ds = Dataset4dstem.from_file(file_path, file_type=file_type)
        return cls.from_data(ds)

    @classmethod
    def from_data(cls, dataset: Dataset4dstem) -> "Parallax":
        return cls(dataset=dataset, _token=cls._token)

    # --- properties ---
    @property
    def dataset(self) -> Dataset4dstem:
        return self._dataset

    @dataset.setter
    def dataset(self, value: Dataset4dstem) -> None:
        self._dataset = value

    # --- core ---
    def preprocess(
        self,
        dp_mask: Optional[NDArray[np.bool_]] = None,
        threshold_fraction: float = 0.8,
    ) -> "Parallax":
        """
        Build virtual BF images by selecting detector pixels and stacking
        their scan images.

        Parameters
        ----------
        dp_mask : ndarray[bool], optional
            Boolean mask over diffraction plane (Nkx, Nky). If None, it is
            derived from the scan-mean DP using `threshold_fraction`.
        threshold_fraction : float, default 0.8
            Pixels with mean intensity >= fraction * max(mean DP) are selected
            when `dp_mask` is None.

        Returns
        -------
        self
        """
        arr = self._dataset.array  # (Nx, Ny, Nkx, Nky)
        if arr.ndim != 4:
            raise ValueError("dataset.array must have shape (Nx, Ny, Nkx, Nky).")

        Nx, Ny, Nkx, Nky = arr.shape

        if dp_mask is None:
            mean_dp = arr.mean(axis=(0, 1))  # (Nkx, Nky)
            if not np.isfinite(mean_dp).any():
                raise ValueError("Mean diffraction pattern contains no finite values.")
            thr = float(threshold_fraction) * float(np.nanmax(mean_dp))
            dp_mask = mean_dp >= thr

        dp_mask = np.asarray(dp_mask, dtype=bool)
        if dp_mask.shape != (Nkx, Nky):
            raise ValueError("dp_mask must have shape (Nkx, Nky).")

        k_indices = np.argwhere(dp_mask)  # (Nimg, 2) with columns [kx, ky]
        if k_indices.size == 0:
            raise ValueError("No BF pixels selected; check threshold_fraction or dp_mask.")

        # Gather images without looping: reshape to (Nx, Ny, Nkx*Nky) then index columns.
        flat = arr.reshape(Nx, Ny, Nkx * Nky)
        flat_ids = np.ravel_multi_index((k_indices[:, 0], k_indices[:, 1]), (Nkx, Nky))
        stack = np.transpose(flat[:, :, flat_ids], (2, 0, 1))  # (Nimg, Nx, Ny)

        # Store outputs
        self.dp_mask = dp_mask
        self.bf_pixel_indices = k_indices.astype(np.int32, copy=False)
        self.bf_images = Dataset3d.from_array(stack)

        return self
