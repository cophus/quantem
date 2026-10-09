from dataclasses import dataclass
from math import ceil
from typing import Literal, Union, overload

import numpy as np
import torch
from scipy.optimize import curve_fit

from quantem.core.utils import array_funcs as af

ArrayLike = Union[np.ndarray, "torch.Tensor"]


@dataclass
class OptimizationParameter:
    """Specification for a parameter to optimize.

    Shared by the iterative and direct hyperparameter searches, which both test candidate
    specifications with ``isinstance``. It lives here, rather than in either of them, so
    that there is only ever one class and a value built for one search is accepted by the
    other. It is re-exported from ``direct_ptychography`` and ``optimize_hyperparameters``,
    which are the paths callers already use.
    """

    low: float
    high: float
    log: bool = False
    n_points: int | None = None

    def grid_values(self):
        """Return an array of grid values for this parameter."""
        if self.n_points is None:
            raise ValueError("n_points must be specified for grid search parameters.")
        if self.log:
            return np.geomspace(self.low, self.high, self.n_points)
        else:
            return np.linspace(self.low, self.high, self.n_points)


# TODO: figure out what here should be put into ptycho base vs kept in a utilities file


class SimpleBatcher:
    def __init__(
        self,
        num: int,
        batch_size: int | None,
        shuffle: bool = True,
        rng: np.random.Generator | int | None = None,
        val_ratio: float = 0.0,
        val_mode: Literal["grid", "random"] = "grid",
        train_indices: np.ndarray | None = None,
        val_indices: np.ndarray | None = None,
    ):
        self.batch_size = batch_size if batch_size is not None else num
        self.shuffle = shuffle
        self.rng = rng

        # Train/validation split (fixed for the lifetime of this batcher)
        if train_indices is not None or val_indices is not None:
            if train_indices is None or val_indices is None:
                raise ValueError("Both train_indices and val_indices must be provided together.")
            self.train_indices = np.asarray(train_indices, dtype=int)
            self.val_indices = np.asarray(val_indices, dtype=int)
        else:
            self.train_indices, self.val_indices = compute_train_val_split(
                num, val_ratio, val_mode, self.rng
            )

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    @rng.setter
    def rng(self, rng: np.random.Generator | int | None):
        if rng is None:
            rng = np.random.default_rng()
        elif isinstance(rng, (int, float)):
            rng = np.random.default_rng(rng)
        elif not isinstance(rng, np.random.Generator):
            raise TypeError(f"rng should be a np.random.Generator or a seed, got {type(rng)}")
        self._rng = rng

    def __iter__(self):
        train_order = (
            self.rng.permutation(self.train_indices) if self.shuffle else self.train_indices
        )
        for i in range(0, len(train_order), self.batch_size):
            yield train_order[i : i + self.batch_size]

    def __len__(self):
        return int(ceil(len(self.train_indices) / self.batch_size))

    def iter_val(self):
        if len(self.val_indices) == 0:
            return iter(())

        # Do not shuffle validation by default
        def _gen():
            for i in range(0, len(self.val_indices), self.batch_size):
                yield self.val_indices[i : i + self.batch_size]

        return _gen()

    @property
    def has_validation(self) -> bool:
        return len(self.val_indices) > 0

    def val_len(self) -> int:
        return int(ceil(len(self.val_indices) / self.batch_size)) if self.has_validation else 0


def compute_train_val_split(
    num: int,
    val_ratio: float,
    val_mode: Literal["grid", "random"],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the train/validation index split.

    Returns ``(train_indices, val_indices)`` as int numpy arrays. ``val_mode="grid"``
    selects every k-th index (with ``k = round(1/val_ratio)``, inverted when
    ``val_ratio > 0.5``); ``"random"`` selects a seeded ``rng.permutation`` slice.
    """
    indices = np.arange(num)
    if val_ratio < 0 or val_ratio >= 1:
        val_ratio = 0.0
    n_val = int(round(len(indices) * val_ratio))
    if n_val <= 0:
        return indices, np.asarray([], dtype=int)

    if val_mode == "random":
        # Random unique selection for validation
        perm = rng.permutation(indices)
        val_indices = perm[:n_val]
        train_indices = np.setdiff1d(indices, val_indices, assume_unique=False)
    else:  # grid/regular selection: every k-th index
        if val_ratio <= 0.5:
            k = max(1, int(round(1.0 / val_ratio)))
            invert = False
        else:
            k = max(1, int(round(1.0 / (1.0 - val_ratio))))
            invert = True

        grid_sel = indices[::k]
        if len(grid_sel) > n_val:
            grid_sel = grid_sel[:n_val]
        if invert:
            train_indices = grid_sel
            val_indices = np.setdiff1d(indices, grid_sel, assume_unique=False)
        else:
            val_indices = grid_sel
            train_indices = np.setdiff1d(indices, val_indices, assume_unique=False)

    return np.asarray(train_indices, dtype=int), np.asarray(val_indices, dtype=int)


def add_input_noise(
    model_input: torch.Tensor,
    noise_std: float,
    dtype: torch.dtype,
    device: "torch.device | str | int",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add gaussian noise to a DIP model input when noise_std > 0."""
    if noise_std > 0.0:
        noise = (
            torch.randn(
                model_input.shape,
                dtype=dtype,
                device=device,
                generator=generator,
            )
            * noise_std
        )
        return model_input + noise
    return model_input


@overload
def fourier_shift_expand(
    array: np.ndarray, positions: np.ndarray, expand_dim: bool = True
) -> np.ndarray: ...
@overload
def fourier_shift_expand(
    array: "torch.Tensor", positions: "torch.Tensor", expand_dim: bool = True
) -> "torch.Tensor": ...
def fourier_shift_expand(
    array: ArrayLike, positions: ArrayLike, expand_dim: bool = True
) -> ArrayLike:
    """Fourier-shift array by flat array of positions."""
    dtype = array.dtype if af.is_complex(array) else None
    phase = fourier_translation_operator(positions, array.shape, expand_dim, dtype=dtype)
    fourier_array = af.fft2(array)
    shifted_fourier_array = fourier_array * phase
    shifted_array = af.ifft2(shifted_fourier_array)
    if af.is_complex(array):
        return shifted_array
    else:
        return shifted_array.real  # type:ignore ## will be numeric so this should be safe


@overload
def fourier_translation_operator(
    positions: np.ndarray,
    shape: tuple,
    expand_dim: bool = True,
    dtype: "str|torch.dtype|np.dtype|None" = None,
) -> np.ndarray: ...
@overload
def fourier_translation_operator(
    positions: "torch.Tensor",
    shape: tuple,
    expand_dim: bool = True,
    dtype: "str|torch.dtype|np.dtype|None" = None,
) -> "torch.Tensor": ...
def fourier_translation_operator(
    positions: ArrayLike,
    shape: tuple,
    expand_dim: bool = True,
    dtype: "str|torch.dtype|np.dtype|None" = None,
) -> ArrayLike:
    """Returns phase ramp for fourier-shifting array of shape `shape`."""
    nr, nc = shape[-2:]
    r = positions[..., 0][:, None, None]
    c = positions[..., 1][:, None, None]
    kr = af.match_device(np.fft.fftfreq(nr, d=1.0).astype(np.float32), positions)
    kc = af.match_device(np.fft.fftfreq(nc, d=1.0).astype(np.float32), positions)
    ramp_r = af.exp(-2.0j * np.pi * kr[None, :, None] * r)
    ramp_c = af.exp(-2.0j * np.pi * kc[None, None, :] * c)
    ramp = ramp_r * ramp_c
    if expand_dim:
        for _ in range(len(shape) - 2):
            ramp = ramp[:, None, ...]
    if dtype is not None:
        ramp = af.as_type(ramp, dtype)
    return ramp


def sum_patches_base(
    patches: torch.Tensor, indices: torch.Tensor, obj_shape: tuple
) -> torch.Tensor:
    flat_weights = patches.reshape(-1)
    flat_indices = indices.reshape(-1)
    out = af.match_device(
        torch.zeros(
            int(torch.prod(torch.tensor(obj_shape))), dtype=patches.dtype, device=patches.device
        ),
        patches,
    )
    out.index_add_(0, flat_indices, flat_weights)
    return out.reshape(obj_shape)


def sum_patches(patches: torch.Tensor, indices: torch.Tensor, obj_shape: tuple) -> torch.Tensor:
    if torch.is_complex(patches):
        real = sum_patches_base(patches.real, indices, obj_shape)
        imag = sum_patches_base(patches.imag, indices, obj_shape)
        return real + 1.0j * imag
    else:
        return sum_patches_base(patches, indices, obj_shape)


def shift_array(
    ar: np.ndarray,
    rshift: np.ndarray,
    cshift: np.ndarray,
    periodic: bool = True,
    bilinear: bool = False,
):
    """
        Shifts array ar by the shift vector (rshift, cshift), using the either
    the Fourier shift theorem (i.e. with sinc interpolation), or bilinear
    resampling. Boundary conditions can be periodic or not.

    Args:
            ar (float): input array
            rshift (float): shift along axis 0 (rows) in pixels
            cshift (float): shift along axis 1 (columns) in pixels
            periodic (bool): flag for periodic boundary conditions
            bilinear (bool): flag for bilinear image shifts
            device(str): calculation device will be perfomed on. Must be 'cpu' or 'gpu'
        Returns:
            (array) the shifted array
    """
    xp = af.get_xp_module(ar)

    # Apply image shift
    if bilinear is False:
        nr, nc = xp.shape(ar)
        qr, qc = make_Fourier_coords2D(nr, nc, 1)
        qr = xp.asarray(qr)
        qc = xp.asarray(qc)

        p = xp.exp(-(2j * xp.pi) * ((cshift * qc) + (rshift * qr)))
        shifted_ar = xp.real(xp.fft.ifft2((xp.fft.fft2(ar)) * p))

    else:
        rF = xp.floor(rshift).astype(int).item()
        cF = xp.floor(cshift).astype(int).item()
        wr = rshift - rF
        wc = cshift - cF

        shifted_ar = (
            xp.roll(ar, (rF, cF), axis=(0, 1)) * ((1 - wr) * (1 - wc))
            + xp.roll(ar, (rF + 1, cF), axis=(0, 1)) * ((wr) * (1 - wc))
            + xp.roll(ar, (rF, cF + 1), axis=(0, 1)) * ((1 - wr) * (wc))
            + xp.roll(ar, (rF + 1, cF + 1), axis=(0, 1)) * ((wr) * (wc))
        )

    if periodic is False:
        # Rounded coordinates for boundaries
        rR = (xp.round(rshift)).astype(int)
        cR = (xp.round(cshift)).astype(int)

        if rR > 0:
            shifted_ar[0:rR, :] = 0
        elif rR < 0:
            shifted_ar[rR:, :] = 0
        if cR > 0:
            shifted_ar[:, 0:cR] = 0
        elif cR < 0:
            shifted_ar[:, cR:] = 0

    return shifted_ar


def make_Fourier_coords2D(
    Nr: int, Nc: int, pixelSize: float | tuple[float, float] = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates Fourier coordinates for a (Nr,Nc)-shaped 2D array.
        Specifying the pixelSize argument sets a unit size.
    """
    if isinstance(pixelSize, (tuple, list)):
        assert len(pixelSize) == 2, "pixelSize must either be a scalar or have length 2"
        pixelSize_r = pixelSize[0]
        pixelSize_c = pixelSize[1]
    else:
        pixelSize_r = pixelSize
        pixelSize_c = pixelSize

    qr = np.fft.fftfreq(Nr, pixelSize_r)
    qc = np.fft.fftfreq(Nc, pixelSize_c)
    qc, qr = np.meshgrid(qc, qr)
    return qr, qc


######## Fitting


def _plane(xy, mx, my, b):
    return mx * xy[0] + my * xy[1] + b


def _parabola(xy, c0, cx1, cx2, cy1, cy2, cxy):
    return (
        c0 + cx1 * xy[0] + cy1 * xy[1] + cx2 * xy[0] ** 2 + cy2 * xy[1] ** 2 + cxy * xy[0] * xy[1]
    )


def _bezier_two(xy, c00, c01, c02, c10, c11, c12, c20, c21, c22):
    return (
        c00 * ((1 - xy[0]) ** 2) * ((1 - xy[1]) ** 2)
        + c10 * 2 * (1 - xy[0]) * xy[0] * ((1 - xy[1]) ** 2)
        + c20 * (xy[0] ** 2) * ((1 - xy[1]) ** 2)
        + c01 * 2 * ((1 - xy[0]) ** 2) * (1 - xy[1]) * xy[1]
        + c11 * 4 * (1 - xy[0]) * xy[0] * (1 - xy[1]) * xy[1]
        + c21 * 2 * (xy[0] ** 2) * (1 - xy[1]) * xy[1]
        + c02 * ((1 - xy[0]) ** 2) * (xy[1] ** 2)
        + c12 * 2 * (1 - xy[0]) * xy[0] * (xy[1] ** 2)
        + c22 * (xy[0] ** 2) * (xy[1] ** 2)
    )


# TODO -- testing this
def fit_origin(
    data: np.ndarray | tuple[np.ndarray, np.ndarray],
    mask: np.ndarray | None = None,
    fit_function: Literal["plane", "parabola", "bezier_two", "constant"] = "plane",
    robust=False,
    robust_steps=3,
    robust_thresh=2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fits the origin of diffraction space using the specified method."""

    qr0_meas, qc0_meas = data

    if fit_function == "plane":
        f = _plane
    elif fit_function == "parabola":
        f = _parabola
    elif fit_function == "bezier_two":
        f = _bezier_two
    elif fit_function == "constant":
        # only average over the masked-in (and finite) positions; otherwise NaN/masked-out
        # positions (e.g. zeroed diffraction patterns) would poison the mean
        qr0_sel = qr0_meas[mask] if mask is not None else qr0_meas
        qc0_sel = qc0_meas[mask] if mask is not None else qc0_meas
        qr0_fit = np.nanmean(qr0_sel) * np.ones_like(qr0_meas)
        qc0_fit = np.nanmean(qc0_sel) * np.ones_like(qc0_meas)
        qr0_residuals = qr0_meas - qr0_fit
        qc0_residuals = qc0_meas - qc0_fit
        return qr0_fit, qc0_fit, qr0_residuals, qc0_residuals
    else:
        raise ValueError(
            "fit_function must be one of 'plane', 'parabola', 'bezier_two', 'constant'"
        )
    shape = qr0_meas.shape
    r, c = np.indices(shape)
    r1D = r.reshape(1, np.prod(shape))
    c1D = c.reshape(1, np.prod(shape))
    rc = np.vstack((r1D, c1D))

    if mask is not None:
        qr0_meas_masked = qr0_meas[mask]
        qc0_meas_masked = qc0_meas[mask]
        mask1D = mask.reshape(1, np.prod(shape))
        rc_masked = np.vstack((r1D[mask1D], c1D[mask1D]))
        # old failed for zero-valued DPs
        # rc_masked = np.vstack((r1D * mask1D, c1D * mask1D))

        popt_r, _ = curve_fit(f, rc_masked, qr0_meas_masked)
        popt_c, _ = curve_fit(f, rc_masked, qc0_meas_masked)

        if robust:
            popt_r = perform_robust_fitting(
                f, rc_masked, qr0_meas_masked, popt_r, robust_steps, robust_thresh
            )
            popt_c = perform_robust_fitting(
                f, rc_masked, qc0_meas_masked, popt_c, robust_steps, robust_thresh
            )
    else:
        popt_r, _ = curve_fit(f, rc, qr0_meas)
        popt_c, _ = curve_fit(f, rc, qc0_meas)

        if robust:
            popt_r = perform_robust_fitting(f, rc, qr0_meas, popt_r, robust_steps, robust_thresh)
            popt_c = perform_robust_fitting(f, rc, qc0_meas, popt_c, robust_steps, robust_thresh)

    qr0_fit = f(rc, *popt_r).reshape(shape)
    qc0_fit = f(rc, *popt_c).reshape(shape)
    qr0_residuals = qr0_meas - qr0_fit
    qc0_residuals = qc0_meas - qc0_fit

    return qr0_fit, qc0_fit, qr0_residuals, qc0_residuals


def perform_robust_fitting(func, rc, data, initial_guess, robust_steps, robust_thresh):
    """Performs robust fitting by iteratively rejecting outliers."""
    popt = initial_guess
    for k in range(robust_steps):
        fit_vals = func(rc, *popt)
        rmse = np.sqrt(np.mean((fit_vals - data) ** 2))
        mask = np.abs(fit_vals - data) <= robust_thresh * rmse
        rc = np.vstack((rc[0][mask], rc[1][mask]))
        data = data[mask]
        popt, _ = curve_fit(func, rc, data, p0=popt)
    return popt


class AffineTransform:
    """
    Affine Transform Class.

    Simplified version of AffineTransform from tike:
    https://github.com/AdvancedPhotonSource/tike/blob/f9004a32fda5e49fa63b987e9ffe3c8447d59950/src/tike/ptycho/position.py

    AffineTransform() -> Identity

    Parameters
    ----------
    scale0: float
        x-scaling
    scale1: float
        y-scaling
    shear1: float
        \\gamma shear
    angle: float
        \\theta rotation angle
    t0: float
        x-translation
    t1: float
        y-translation
    dilation: float
        Isotropic expansion (multiplies scale0 and scale1)
    """

    def __init__(
        self,
        scale0: float = 1.0,
        scale1: float = 1.0,
        shear1: float = 0.0,
        angle: float = 0.0,
        t0: float = 0.0,
        t1: float = 0.0,
        dilation: float = 1.0,
    ):
        self.scale0 = scale0 * dilation
        self.scale1 = scale1 * dilation
        self.shear1 = shear1
        self.angle = angle
        self.t0 = t0
        self.t1 = t1

    @classmethod
    def from_array(cls, T: np.ndarray):
        """
        Return an Affine Transfrom from a 2x2 matrix.
        Use decomposition method from Graphics Gems 2 Section 7.1
        """
        R = T[:2, :2].copy()
        scale0 = np.linalg.norm(R[0])
        if scale0 <= 0:
            return cls()
        R[0] /= scale0
        shear1 = R[0] @ R[1]
        R[1] -= shear1 * R[0]
        scale1 = np.linalg.norm(R[1])
        if scale1 <= 0:
            return cls()
        R[1] /= scale1
        shear1 /= scale1
        angle = np.arctan2(-R[0, 1], R[0, 0])

        if T.shape[0] > 2:
            t0, t1 = T[2]
        else:
            t0 = t1 = 0.0

        return cls(
            scale0=float(scale0),
            scale1=float(scale1),
            shear1=float(shear1),
            angle=float(angle),
            t0=t0,
            t1=t1,
        )

    def asarray(self):
        """
        Return an 2x2 matrix of scale, shear, rotation.
        This matrix is scale @ shear @ rotate from left to right.
        """
        cosx = np.cos(self.angle)
        sinx = np.sin(self.angle)
        return (
            np.array(
                [
                    [self.scale0, 0.0],
                    [0.0, self.scale1],
                ],
                dtype="float32",
            )
            @ np.array(
                [
                    [1.0, 0.0],
                    [self.shear1, 1.0],
                ],
                dtype="float32",
            )
            @ np.array(
                [
                    [+cosx, -sinx],
                    [+sinx, +cosx],
                ],
                dtype="float32",
            )
        )

    def asarray3(self):
        """
        Return an 3x2 matrix of scale, shear, rotation, translation.
        This matrix is scale @ shear @ rotate from left to right.
        Expects a homogenous (z) coordinate of 1.
        """
        T = np.empty((3, 2), dtype="float32")
        T[2] = (self.t0, self.t1)
        T[:2, :2] = self.asarray()
        return T

    def astuple(self):
        """Return the constructor parameters in a tuple."""
        return (
            self.scale0,
            self.scale1,
            self.shear1,
            self.angle,
            self.t0,
            self.t1,
        )

    def __call__(self, x: np.ndarray, origin=(0, 0), xp=np) -> np.ndarray:
        origin = xp.asarray(origin, dtype=xp.float32)
        tf_matrix = self.asarray()
        tf_matrix = xp.asarray(tf_matrix, dtype=xp.float32)
        tf_translation = xp.array((self.t0, self.t1)) + origin
        return ((x - origin) @ tf_matrix) + tf_translation

    def __str__(self):
        return (
            "AffineTransform( \n"
            f"  scale0 = {self.scale0:.4f}, scale1 = {self.scale1:.4f}, \n"
            f"  shear1 = {self.shear1:.4f}, angle = {self.angle:.4f}, \n"
            f"  t0 = {self.t0:.4f}, t1 = {self.t1:.4f}, \n"
            ")"
        )

    def __repr__(self):
        return (
            "AffineTransform( \n"
            f"  scale0 = {self.scale0:.4f}, scale1 = {self.scale1:.4f}, \n"
            f"  shear1 = {self.shear1:.4f}, angle = {self.angle:.4f}, \n"
            f"  t0 = {self.t0:.4f}, t1 = {self.t1:.4f}, \n"
            ")"
        )


def center_crop_arr(
    arr: np.ndarray, shape: tuple[int, ...], pad_if_needed: bool = False
) -> np.ndarray:
    """
    Crop an array to a given shape, centered along all axes.

    Parameters
    ----------
    arr : np.ndarray
        The input n-dimensional array to be cropped.
    shape : tuple[int, ...]
        The desired output shape. Must have the same number of dimensions as arr,
        and each dimension must be less than or equal to the corresponding dimension of arr.
    """
    if len(shape) != arr.ndim:
        raise ValueError(
            f"Shape must have the same number of dimensions as arr. "
            f"Got shape with {len(shape)} dimensions and arr with {arr.ndim} dimensions."
        )

    pad = [[0, 0]] * len(shape)
    for i, (s, a) in enumerate(zip(shape, arr.shape)):
        if s > a:
            if not pad_if_needed:
                raise ValueError(
                    f"Dimension {i} of shape ({s}) is larger than dimension {i} of arr ({a})."
                )
            pad[i] = [(s - a) // 2, ceil((s - a) / 2)]

    if any(p != [0, 0] for p in pad):
        arr = np.pad(arr, pad_width=pad, mode="constant")

    slices = []
    for i, (s, a) in enumerate(zip(shape, arr.shape)):
        start = (a - s) // 2
        end = start + s
        slices.append(slice(start, end))

    # Return the cropped array
    return arr[tuple(slices)]


def split_counts(dset, fraction: float = 0.5, rng: np.random.Generator | int | None = None):
    """
    Split each recorded count between two datasets by binomial thinning.

    Each detector count is assigned to the first dataset with probability ``fraction`` and to
    the second otherwise. For Poisson-distributed counts the two datasets are independent
    measurements of the same diffraction patterns at doses ``fraction`` and ``1 - fraction``,
    so a reconstruction from one can be validated against the other at every probe position
    and detector pixel. Thinning commutes with binning and cropping, and so it can be applied
    before or after either. Detectors where one electron produces counts in neighboring pixels
    give weakly correlated halves.

    Parameters
    ----------
    dset : Dataset4dstem | np.ndarray
        Integer counts.
    fraction : float, optional
        Probability that a count is assigned to the first dataset, by default 0.5.
    rng : np.random.Generator | int | None, optional
        Random generator or seed.

    Returns
    -------
    tuple
        ``(first, second)``, of the same type as ``dset``, with ``first + second == dset``.
    """
    if not 0 < fraction < 1:
        raise ValueError(f"fraction must be between 0 and 1, got {fraction}")
    rng = np.random.default_rng(rng)
    array = np.asarray(dset.array if hasattr(dset, "array") else dset)
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.mod(array, 1) == 0):
            raise ValueError("split_counts requires integer counts")
        array = array.astype(np.int64)
    first = np.empty_like(array)
    for index in np.ndindex(array.shape[: max(array.ndim - 3, 1)]):
        counts = np.maximum(array[index].astype(np.int64), 0)  # chunked for memory
        first[index] = rng.binomial(counts, fraction)
    second = array - first
    if hasattr(dset, "array"):
        return tuple(
            type(dset).from_array(
                array=half,
                name=f"{dset.name} {label}",
                origin=dset.origin,
                sampling=dset.sampling,
                units=dset.units,
                signal_units=dset.signal_units,
            )
            for half, label in ((first, "split A"), (second, "split B"))
        )
    return first, second


def refine_slices(
    obj: np.ndarray, obj_type: Literal["potential", "pure_phase", "complex"]
) -> np.ndarray:
    """
    Insert a slice midway between each pair of slices of a multislice object.

    The new slices are linear interpolations of their neighbors along the beam direction,
    giving ``2 * num_slices - 1`` slices at half the spacing. All slices are then rescaled so
    that the sum over slices of every pixel, the projected potential or phase, is unchanged.
    Complex transmission functions are interpolated as ``log(t)``, which assumes that the phase
    of each slice is below pi.

    Parameters
    ----------
    obj : np.ndarray
        Object with shape ``(num_slices, H, W)``.
    obj_type : {"potential", "pure_phase", "complex"}
        Representation of ``obj``.

    Returns
    -------
    np.ndarray
        Object with shape ``(2 * num_slices - 1, H, W)``.
    """

    def _refine(field: np.ndarray) -> np.ndarray:
        refined = np.empty((2 * field.shape[0] - 1, *field.shape[1:]), dtype=field.dtype)
        refined[0::2] = field
        refined[1::2] = 0.5 * (field[:-1] + field[1:])
        total, total_refined = field.sum(0), refined.sum(0)
        small = np.abs(total_refined) <= 1e-6 * max(np.abs(total_refined).max(), 1e-30)
        ratio = np.where(small, 0.5, total / np.where(small, 1.0, total_refined))
        return refined * ratio

    if obj.shape[0] < 2:
        raise ValueError("refine_slices requires at least two slices")
    if obj_type == "complex":
        log_amplitude = _refine(np.log(np.maximum(np.abs(obj), 1e-12)))
        phase = _refine(np.angle(obj))
        return np.exp(log_amplitude + 1j * phase).astype(obj.dtype)
    return _refine(obj)


def shear_slices(
    obj: np.ndarray,
    tilt_mrad: tuple[float, float],
    slice_thicknesses: np.ndarray | float,
    sampling: tuple[float, float],
) -> np.ndarray:
    """
    Shift each slice of a multislice object laterally in proportion to its depth.

    A crystal tilted by ``tilt_mrad`` relative to the reconstruction axis displaces its columns by
    ``z * tan(tilt)`` at depth ``z``. Shifting each slice back by this amount, about the center of
    the stack, aligns the columns so that the sum over slices is the projection along the
    crystal axis. The correction is accurate when the slices are thin compared with the depth
    over which a column moves by about one pixel.

    Parameters
    ----------
    obj : np.ndarray
        Real object with shape ``(num_slices, H, W)``, for example a potential or phase field.
    tilt_mrad : tuple[float, float]
        Tilt along rows and columns, in mrad.
    slice_thicknesses : np.ndarray | float
        Distances between consecutive slices in A.
    sampling : tuple[float, float]
        Real-space sampling in A.

    Returns
    -------
    np.ndarray
        Sheared object with the same shape as ``obj``.
    """
    num_slices = obj.shape[0]
    thick = np.broadcast_to(np.asarray(slice_thicknesses, dtype=np.float64), (num_slices - 1,))
    depth = np.concatenate([[0.0], np.cumsum(thick)])
    depth -= depth.mean()
    kr = np.fft.fftfreq(obj.shape[-2])[:, None]
    kc = np.fft.fftfreq(obj.shape[-1])[None, :]
    out = np.empty_like(obj)
    for s in range(num_slices):
        shift_r = depth[s] * np.tan(tilt_mrad[0] * 1e-3) / sampling[0]
        shift_c = depth[s] * np.tan(tilt_mrad[1] * 1e-3) / sampling[1]
        ramp = np.exp(2j * np.pi * (kr * shift_r + kc * shift_c))
        out[s] = np.fft.ifft2(np.fft.fft2(obj[s]) * ramp).real
    return out


def estimate_tilt(
    obj: np.ndarray,
    slice_thicknesses: np.ndarray | float,
    sampling: tuple[float, float],
    max_tilt_mrad: float = 10.0,
    num_steps: int = 21,
) -> tuple[tuple[float, float], np.ndarray]:
    """
    Tilt that maximizes the variance of the projected object, from a grid search.

    Columns that run through all slices give the sharpest projection when they are aligned, so
    the projected variance peaks at the crystal tilt. The search uses ``shear_slices`` on a grid
    of ``num_steps`` x ``num_steps`` tilts within ``max_tilt_mrad`` and refines on a grid twice
    as fine around the best value.

    Returns
    -------
    tilt_mrad : tuple[float, float]
        Estimated tilt along rows and columns, in mrad.
    scores : np.ndarray
        Projected variance on the coarse grid.
    """

    thick = np.broadcast_to(np.asarray(slice_thicknesses, dtype=np.float64), (obj.shape[0] - 1,))
    max_shift = thick.sum() / 2 * np.tan(max_tilt_mrad * 2e-3) / np.min(sampling)
    m = int(np.ceil(max_shift)) + 2  # exclude the edges that the Fourier shifts wrap around

    def score(tilt: tuple[float, float]) -> float:
        projection = shear_slices(obj, tilt, slice_thicknesses, sampling).sum(0)
        return float(projection[m:-m, m:-m].var())

    grid = np.linspace(-max_tilt_mrad, max_tilt_mrad, num_steps)
    scores = np.array([[score((tr, tc)) for tc in grid] for tr in grid])
    ir, ic = np.unravel_index(np.argmax(scores), scores.shape)
    step = grid[1] - grid[0]
    fine = np.linspace(-step, step, 9)
    best = max(
        (
            (score((grid[ir] + dr, grid[ic] + dc)), grid[ir] + dr, grid[ic] + dc)
            for dr in fine
            for dc in fine
        )
    )
    return (float(best[1]), float(best[2])), scores


def detector_noise_response(
    dset,
    bin_factor: int = 1,
    inner_radius: float | None = None,
    num_rows: int = 64,
) -> dict:
    """
    Estimate the detector response to one electron from the shot noise in the counts.

    Hybrid pixel detectors at high voltage register one electron as several counts spread over
    neighboring pixels. Shot noise is then correlated between neighboring pixels, while the
    diffraction signal hardly changes between neighboring scan positions. The covariance of
    pixels within one pattern, minus the covariance between patterns at neighboring positions
    along the fast scan axis, is the detector part alone. A symmetric 3x3 response is fitted to
    it at nonzero lags in the dark field, where the counts are sparse. The summed covariance over the mean count
    is the noise gain, the variance over the mean of counts summed over a large area. It equals
    the counts per electron when every electron gives the same number of counts, and is larger
    when that number varies.

    Parameters
    ----------
    dset : Dataset4dstem | np.ndarray
        Unbinned counts with shape ``(scan_rows, scan_cols, ky, kx)``.
    bin_factor : int, optional
        Detector binning used for the reconstruction, by default 1. The returned ``psf`` is on
        the binned grid.
    inner_radius : float | None, optional
        Inner radius of the dark-field annulus in pixels. ``None`` uses 1.25 times the radius
        of the bright-field disk.
    num_rows : int, optional
        Number of scan rows used, evenly spaced, by default 64.

    Returns
    -------
    dict
        ``psf``: detector point spread on the binned grid, summing to one, for
        ``DetectorPixelated(psf=...)``. ``response``: the unbinned response with its center set
        to one. ``noise_gain``: variance over mean of large-area count sums (1 for Poisson
        counts).
        ``correlation``: measured detector noise correlation (5x5 lags).
    """
    from scipy.optimize import least_squares
    from scipy.signal import correlate2d

    array = np.asarray(dset.array if hasattr(dset, "array") else dset)
    rows = np.linspace(0, array.shape[0] - 1, min(num_rows, array.shape[0])).astype(int)
    cols = np.arange(0, array.shape[1] - 1, 3)
    first = array[rows][:, cols].reshape(-1, *array.shape[-2:]).astype(np.float64)
    second = array[rows][:, cols + 1].reshape(-1, *array.shape[-2:]).astype(np.float64)
    mean = 0.5 * (first.mean(0) + second.mean(0))

    bright = mean > 0.5 * mean.max()
    ky, kx = np.indices(mean.shape)
    center = (ky[bright].mean(), kx[bright].mean())
    if inner_radius is None:
        inner_radius = 1.25 * np.sqrt(bright.sum() / np.pi)
    radius = np.hypot(ky - center[0], kx - center[1])
    region = radius > inner_radius
    region[:3] = region[-3:] = False
    region[:, :3] = region[:, -3:] = False
    if region.sum() < 100:
        raise ValueError("too few dark-field pixels; lower inner_radius")

    d1, d2 = first - mean, second - mean
    lags = np.zeros((5, 5))
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            both = region & np.roll(region, (-dy, -dx), (0, 1))

            def shifted(x):
                return np.roll(x, (-dy, -dx), (1, 2))

            same = 0.5 * ((d1 * shifted(d1))[:, both].mean() + (d2 * shifted(d2))[:, both].mean())
            cross = 0.5 * ((d1 * shifted(d2))[:, both].mean() + (d2 * shifted(d1))[:, both].mean())
            lags[dy + 2, dx + 2] = same - cross
    correlation = lags / lags[2, 2]

    def kernel(p):
        a, b = np.abs(p)  # nearest and diagonal neighbors
        return np.array([[b, a, b], [a, 1, a], [b, a, b]])

    off_center = np.ones((5, 5), dtype=bool)
    off_center[2, 2] = False

    def residual(p):
        # the zero lag also holds the spread in counts per electron, so it is left out
        h = kernel(p[1:])
        auto = correlate2d(h, h, mode="full")
        return (np.abs(p[0]) * auto - correlation)[off_center]

    response = np.pad(kernel(least_squares(residual, [0.5, 0.1, 0.03]).x[1:]), 1)

    # average the binned response over the landing positions of an electron inside one bin
    b = int(bin_factor)
    half = 2 // b + 1
    size = 2 * half + 1
    psf = np.zeros((size, size))
    for oy in range(b):
        for ox in range(b):
            canvas = np.zeros((size * b, size * b))
            y0, x0 = half * b + oy - 2, half * b + ox - 2
            canvas[y0 : y0 + 5, x0 : x0 + 5] = response
            psf += canvas.reshape(size, b, size, b).sum((1, 3))
    psf /= psf.sum()

    return {
        "psf": psf,
        "response": response,
        "noise_gain": float(lags.sum() / mean[region].mean()),
        "correlation": correlation,
    }
