from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from matplotlib.patches import Circle, Ellipse
from scipy.optimize import least_squares

from quantem.core.utils.filter import otsu_threshold


def fit_probe_circle(
    img: np.ndarray, threshold: Optional[float] = None, show: bool = True
) -> Tuple[float, float, float]:
    """
    Fit a circle to the probe shape in an image.

    Args:
        img (np.ndarray): Input image containing the probe.
        threshold (Optional[float]): Threshold for binarization. If None, Otsu's method is used.
        show (bool): Whether to display the fitted circle. Default is True.

    Returns:
        Tuple[float, float, float]: Center coordinates (xc, yc) and radius R of the fitted circle.
    """
    if threshold is None:
        threshold = otsu_threshold(img)
    binary = ndi.binary_closing(img > threshold, iterations=2)
    binary: np.ndarray = ndi.binary_fill_holes(binary)  # type: ignore

    smoothed = ndi.gaussian_filter(binary.astype(float), sigma=1)
    grad_mag = np.hypot(ndi.sobel(smoothed, axis=1), ndi.sobel(smoothed, axis=0))

    edge_points = np.argwhere(grad_mag > grad_mag.mean())
    y, x = edge_points[:, 0], edge_points[:, 1]

    def calc_R(xc: float, yc: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def f_2(c: Tuple[float, float], x: np.ndarray, y: np.ndarray) -> np.ndarray:
        Ri = calc_R(*c, x, y)
        return Ri - Ri.mean()

    center_estimate = np.mean(x), np.mean(y)
    result = least_squares(f_2, center_estimate, args=(x, y))
    xc, yc = result.x
    R = calc_R(xc, yc, x, y).mean()

    if show:
        _fig, ax = plt.subplots()
        ax.imshow(img, cmap="gray")
        ax.add_patch(Circle((xc, yc), R, color="red", alpha=0.3, linewidth=2))
        plt.show()

    return yc, xc, R


def fit_probe_ellipse(
    img: np.ndarray, threshold: Optional[float] = None, show: bool = True
) -> Tuple[float, float, float, float, float]:
    """
    Fit an ellipse to the probe shape in an image using the direct least squares fitting method for
    ellipses, which solves the general conic equation Ax² + Bxy + Cy² + Dx + Ey + F = 0 subject to
    the constraint B² - 4AC < 0 (ellipse condition).

    Args:
        img (np.ndarray): Input image containing the probe.
        threshold (Optional[float]): Threshold for binarization. If None, Otsu's method is used.
        show (bool): Whether to display the fitted ellipse. Default is True.

    Returns:
        Tuple[float, float, float, float, float]: Center coordinates (xc, yc),
        semi-major axis (a_axis), semi-minor axis (b_axis), and rotation angle (theta) in radians.
    """
    # Threshold the image to create a binary mask
    if threshold is None:
        threshold = otsu_threshold(img)
    binary: np.ndarray = ndi.binary_fill_holes(img > threshold)  # type: ignore

    # Find edge points using gradient magnitude
    smoothed = ndi.gaussian_filter(binary.astype(float), sigma=1)
    grad_x = ndi.sobel(smoothed, axis=1)
    grad_y = ndi.sobel(smoothed, axis=0)
    grad_mag = np.hypot(grad_x, grad_y)
    edge_points = np.argwhere(grad_mag > grad_mag.mean())
    y, x = edge_points[:, 0], edge_points[:, 1]

    # Set up the design matrix for the conic equation Ax² + Bxy + Cy² + Dx + Ey + F = 0
    # Each row represents one edge point [x², xy, y², x, y, 1]
    D = np.vstack([x**2, x * y, y**2, x, y, np.ones_like(x)]).T
    S = np.dot(D.T, D)  # Scatter matrix

    # Constraint matrix to enforce ellipse condition (B² - 4AC < 0)
    C = np.zeros([6, 6])
    C[0, 2] = C[2, 0] = 2  # Constraint: 4AC
    C[1, 1] = -1  # Constraint: -B²

    # Solve the generalized eigenvalue problem to find ellipse coefficients
    eigvals, eigvecs = np.linalg.eig(np.linalg.inv(S).dot(C))
    cond = np.logical_and(np.isreal(eigvals), eigvals > 0)
    a = eigvecs[:, cond][:, 0].real  # type: ignore

    # Extract coefficients from the conic equation
    # General form: a0*x² + a1*xy + a2*y² + a3*x + a4*y + a5 = 0
    b, c, d, f, g, a0 = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]

    # Calculate ellipse center coordinates
    num = b * b - a0 * c
    xc = (c * d - b * f) / num
    yc = (a0 * f - b * d) / num

    # Calculate semi-major and semi-minor axes lengths
    up = 2 * (a0 * f * f + c * d * d + g * b * b - 2 * b * d * f - a0 * c * g)
    down1 = (b * b - a0 * c) * ((c - a0) * np.sqrt(1 + 4 * b * b / ((a0 - c) ** 2)) - (c + a0))
    down2 = (b * b - a0 * c) * ((a0 - c) * np.sqrt(1 + 4 * b * b / ((a0 - c) ** 2)) - (c + a0))
    a_axis = np.sqrt(up / down1)  # Semi-major axis
    b_axis = np.sqrt(up / down2)  # Semi-minor axis

    # Calculate rotation angle of the ellipse
    theta = 0.5 * np.arctan(2 * b / (a0 - c))

    if show:
        fig, ax = plt.subplots()
        ax.imshow(img, cmap="gray")
        ellipse = Ellipse(
            (xc, yc),
            2 * a_axis,
            2 * b_axis,
            angle=np.degrees(theta),
            color="red",
            alpha=0.3,
            linewidth=2,
        )
        ax.add_patch(ellipse)
        plt.show()

    return yc, xc, a_axis, b_axis, theta


def correct_ellipticity(
    array: np.ndarray,
    ellipse: Tuple[float, float, float, float, float],
    batch_size: int = 512,
) -> np.ndarray:
    """
    Resample diffraction patterns so that an elliptical bright-field disk becomes circular.

    Projector-lens distortion makes the image of the round condenser aperture elliptical, so the
    detector pixels do not sample scattering angles uniformly. Each pattern is resampled
    (bilinear) onto a grid on which the fitted ellipse is a circle with the same area, about the
    same center, and rescaled to keep its total counts.

    Parameters
    ----------
    array : np.ndarray
        Diffraction patterns with shape ``(..., H, W)``.
    ellipse : tuple
        ``(row_center, col_center, a_axis, b_axis, theta)`` as returned by ``fit_probe_ellipse``
        for the mean pattern.
    batch_size : int, optional
        Number of patterns resampled at a time.

    Returns
    -------
    np.ndarray
        Resampled patterns, float32, same shape as ``array``.
    """
    import torch

    yc, xc, a_axis, b_axis, theta = (float(v) for v in ellipse)
    H, W = array.shape[-2:]
    radius = np.sqrt(a_axis * b_axis)
    rows, cols = np.meshgrid(
        np.arange(H, dtype=np.float64) - yc, np.arange(W, dtype=np.float64) - xc, indexing="ij"
    )
    cos, sin = np.cos(theta), np.sin(theta)
    u = (cos * cols + sin * rows) * a_axis / radius  # along the a axis
    v = (-sin * cols + cos * rows) * b_axis / radius
    col_in = cos * u - sin * v + xc
    row_in = sin * u + cos * v + yc
    grid = torch.as_tensor(
        np.stack([2 * col_in / (W - 1) - 1, 2 * row_in / (H - 1) - 1], -1), dtype=torch.float32
    )[None]

    frames = array.reshape(-1, H, W)
    out = np.empty(frames.shape, dtype=np.float32)
    for start in range(0, len(frames), batch_size):
        x = torch.as_tensor(np.asarray(frames[start : start + batch_size], dtype=np.float32))[
            :, None
        ]
        y = torch.nn.functional.grid_sample(
            x, grid.expand(len(x), -1, -1, -1), mode="bilinear", align_corners=True
        )[:, 0]
        y = y * (x.sum((-1, -2, -3)) / y.sum((-1, -2)).clamp(min=1e-6))[:, None, None]
        out[start : start + batch_size] = y.numpy()
    return out.reshape(array.shape)
