from abc import abstractmethod

import numpy as np
import torch

from quantem.core.io.serialize import AutoSerialize


class DetectorBase(AutoSerialize):
    @abstractmethod
    def forward(self, exit_waves: torch.Tensor) -> torch.Tensor:
        """
        Exit waves to measured intensities
        """
        raise NotImplementedError


class DetectorPixelated(DetectorBase):
    """
    A detector model that simulates pixelated detectors.

    The forward model can use a real-space box (``roi_shape``) larger than the detector. The
    real-space sampling is fixed by the detector extent in reciprocal space, so a larger box
    samples reciprocal space more finely, and each detector pixel then integrates the far-field
    intensity over the model pixels it overlaps. This reduces wrap-around of the wavefield
    during multislice propagation and models the integration over finite detector pixels.

    Parameters
    ----------
    roi_shape : tuple[int, int] | None, optional
        Shape of the real-space box used by the forward model, in pixels. ``None`` (default)
        uses the detector shape.
    psf : np.ndarray | None, optional
        Point spread function of the detector on the detector pixel grid, an odd-sized 2D
        kernel. The predicted intensities are convolved with it after binning. It is normalized
        to sum to one, so it only redistributes counts. ``None`` (default) means no blur.
        ``detector_psf_from_noise`` estimates it from the data.
    """

    def __init__(
        self,
        roi_shape: tuple[int, int] | np.ndarray | None = None,
        psf: np.ndarray | None = None,
    ):
        self.roi_shape = roi_shape
        self.psf = psf
        self._detector_shape: tuple[int, int] | None = None
        self._binning: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def roi_shape(self) -> tuple[int, int] | None:
        """Shape of the real-space box of the forward model, or None for the detector shape."""
        return getattr(self, "_roi_shape", None)

    @roi_shape.setter
    def roi_shape(self, shape: tuple[int, int] | np.ndarray | None) -> None:
        if shape is None:
            self._roi_shape = None
            return
        shape = tuple(int(s) for s in np.atleast_1d(shape))
        if len(shape) == 1:
            shape = (shape[0], shape[0])
        if len(shape) != 2 or min(shape) <= 0 or any(s % 2 for s in shape):
            raise ValueError(f"roi_shape must be two positive even integers, got {shape}")
        self._roi_shape = shape
        self._binning = None

    @property
    def psf(self) -> np.ndarray | None:
        """Detector point spread function on the detector grid, normalized to sum to one."""
        return getattr(self, "_psf", None)

    @psf.setter
    def psf(self, kernel: np.ndarray | None) -> None:
        self._psf_tensor = None
        if kernel is None:
            self._psf = None
            return
        kernel = np.asarray(kernel, dtype=np.float64)
        if kernel.ndim != 2 or any(s % 2 == 0 for s in kernel.shape):
            raise ValueError(f"psf must be a 2D kernel with odd side lengths, got {kernel.shape}")
        if np.any(kernel < 0) or kernel.sum() <= 0:
            raise ValueError("psf must be non-negative with a positive sum")
        self._psf = kernel / kernel.sum()

    @property
    def detector_shape(self) -> tuple[int, int] | None:
        """Shape of the measured diffraction patterns, set by the reconstruction."""
        return getattr(self, "_detector_shape", None)

    @detector_shape.setter
    def detector_shape(self, shape: tuple[int, int] | np.ndarray) -> None:
        self._detector_shape = tuple(int(s) for s in shape)
        self._binning = None

    def model_roi_shape(self, detector_shape: tuple[int, int] | np.ndarray) -> np.ndarray:
        """Real-space box of the forward model for a detector of shape ``detector_shape``."""
        if self.roi_shape is None:
            return np.array(detector_shape).astype("int")
        return np.array(self.roi_shape).astype("int")

    def forward(self, exit_waves: torch.Tensor) -> torch.Tensor:
        """
        Exit waves to measured intensities
        """
        # exit_waves shape: (nprobes, batch_size, roi_shape[0], roi_shape[1])
        # incoherent sum of all probe components
        exit_fft = torch.fft.fft2(exit_waves, norm="ortho")
        intensities = torch.sum(torch.abs(exit_fft) ** 2, dim=0)
        intensities = torch.fft.fftshift(intensities, dim=(-2, -1))  # detector centering
        if self.detector_shape is not None and tuple(intensities.shape[-2:]) != self.detector_shape:
            bin_r, bin_c = self._get_binning(tuple(intensities.shape[-2:]), intensities)
            intensities = bin_r @ intensities @ bin_c.T
        if self.psf is not None:
            intensities = self._apply_psf(intensities)
        return intensities

    def _apply_psf(self, intensities: torch.Tensor) -> torch.Tensor:
        """Convolve the detector intensities with the psf, zero beyond the detector edge."""
        kernel = getattr(self, "_psf_tensor", None)
        if kernel is None or kernel.device != intensities.device or kernel.dtype != intensities.dtype:
            kernel = torch.as_tensor(self.psf, dtype=intensities.dtype, device=intensities.device)
            kernel = torch.flip(kernel, dims=(0, 1))[None, None]  # conv2d correlates
            self._psf_tensor = kernel
        shape = intensities.shape
        blurred = torch.nn.functional.conv2d(
            intensities.reshape(-1, 1, *shape[-2:]),
            kernel,
            padding=(kernel.shape[-2] // 2, kernel.shape[-1] // 2),
        )
        return blurred.reshape(shape)

    def _get_binning(
        self, model_shape: tuple[int, int], like: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Overlap matrices mapping the model far field onto the detector pixels."""
        binning = getattr(self, "_binning", None)
        if (
            binning is None
            or binning[0].shape[1] != model_shape[0]
            or binning[1].shape[1] != model_shape[1]
            or binning[0].device != like.device
            or binning[0].dtype != like.dtype
        ):
            assert self.detector_shape is not None
            binning = tuple(
                torch.as_tensor(
                    pixel_overlap_matrix(n_det, n_model), dtype=like.dtype, device=like.device
                )
                for n_det, n_model in zip(self.detector_shape, model_shape)
            )
            self._binning = binning
        return binning


def pixel_overlap_matrix(num_detector: int, num_model: int) -> np.ndarray:
    """
    Fraction of each model pixel that falls inside each detector pixel, along one axis.

    Both grids span the same reciprocal-space extent and are fftshift-centered, with zero
    frequency at index ``n // 2``. Each column sums to one inside the detector, so the total
    intensity is conserved.

    Parameters
    ----------
    num_detector : int
        Number of detector pixels.
    num_model : int
        Number of model pixels.

    Returns
    -------
    np.ndarray
        Overlap matrix with shape ``(num_detector, num_model)``.
    """
    # pixel edges in units of detector pixels
    edges_det = np.arange(num_detector + 1) - num_detector // 2 - 0.5
    edges_model = (np.arange(num_model + 1) - num_model // 2 - 0.5) * num_detector / num_model
    low = np.maximum(edges_det[:-1, None], edges_model[None, :-1])
    high = np.minimum(edges_det[1:, None], edges_model[None, 1:])
    return np.clip(high - low, 0, None) * num_model / num_detector


DetectorModelType = DetectorPixelated  # | DetectorPixelatedDIP
