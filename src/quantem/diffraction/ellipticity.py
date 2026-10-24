import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset4dstem import Dataset4dstem


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


def fit_elliptical_distortion(
    data: Dataset4dstem | Dataset2d,
    center: tuple[float, float],
    fit_radii: tuple[float, float],
    p0: torch.Tensor | None = None,
    mask: NDArray | torch.Tensor | None = None,
    device: str = "cpu",
    max_iter: int = 500,
    lr: float = 0.5,
    write_metadata: bool = True,
) -> dict:
    """Fit elliptical distortion from the mean diffraction pattern.

    Fits a central-beam Gaussian + a two-sided ("Janus") elliptical ring + a constant background to the
    mean diffraction pattern. The ellipse is parameterized in
    **canonical-conic** form ``(A, B, C)`` — which stays smooth through
    the circular point (no theta degeneracy at low ellipticity).

    Model
    -----
    Let ``u_col = col - col0`` and ``u_row = row - row0`` be the pixel
    offsets from the fitted center. The ring is the level set of the
    form:

        A·u_col² + B·u_col·u_row + C·u_row² = 1

    with semiaxes ``a`` and ``b`` derived from (A, B, C) (see
    :func:`convert_ellipse_params`) and mean ring radius
    ``R = (a + b) / 2``. Define an ellipse-adapted radial coordinate ρ
    that is 0 at the center, R on the ring, and grows linearly along every ray from
    the center. Iso-ρ contours are concentric ellipses similar to
    the ring. In pixel units:

        ρ = R · √(A·u_col² + B·u_col·u_row + C·u_row²)

    For a circle, ρ reduces to the ordinary Euclidean radius
    ``√(u_col² + u_row²)``. The model intensity is:

        f = I0 · exp(-ρ² / (2·σ0²))                # central beam
          + I1 · exp(-(ρ - R)² / (2·σ1²)) · Θ(R - ρ)   # inner ring half
          + I1 · exp(-(ρ - R)² / (2·σ2²)) · Θ(ρ - R)   # outer ring half
          + c_bkgd

    where Θ is the Heaviside step. Inner and outer halves share the same
    amplitude I1 and meet continuously at ρ = R, but with different
    half-widths σ1 (inside the ring) and σ2 (outside). The 11 fit
    parameters ``(I0, I1, σ0, σ1, σ2, c_bkgd, row0, col0, A, B, C)`` are
    described under :func:`_amorphous_ring_model`. The central beam
    shares the elliptical metric, so its iso-intensity contours are
    ellipses concentric with the ring.

    Algorithm
    ---------
    The mean DP is reduced once over all scan positions (or used
    directly for 2D input). The fit minimizes the loss over pixels in
    the annulus ``fit_radii``, using torch Adam. Softplus + conic rescale gives us GPU-compatible gradients
    and built-in positivity.

    Parameters
    ----------
    data : Dataset4dstem or Dataset2d
        ``Dataset4dstem`` is averaged over scan
        positions to produce the mean DP. ``Dataset2d`` (e.g. SAED) is
        used directly. 
    center : (float, float)
        Approximate (row, col) center; refined to sub-pixel by the fit
        via the ``row0, col0`` offset parameters.
    fit_radii : (float, float)
        Inner and outer radii in pixels of the annular fit region. The
        annulus should bracket the ring and exclude the central beam.
    p0 : Tensor or None
        Optional initial guess for all 11 model parameters in the order
        ``(I0, I1, σ0, σ1, σ2, c_bkgd, row0, col0, A, B, C)`` (see
        :func:`_amorphous_ring_model`). If ``None``, all 11 are
        auto-estimated.
    mask : ndarray, Tensor, or None
        Optional boolean mask of shape ``(n_row, n_col)``.
        ``True`` = exclude pixel.
    device : str
        Torch device for the fit.
    max_iter : int
        Number of Adam iterations.
    lr : float
        Adam learning rate.
    write_metadata : bool
        If True (default), stamp ``data.metadata["ellipticity"]`` with
        the fitted ``(a, b, theta_deg)``.

    Returns
    -------
    dict with keys:
        "ellipse_params" : (a, b, theta_deg)
            Semimajor axis, semiminor axis (``a ≥ b``), and tilt of the
            semimajor axis measured from the column axis toward the row
            axis, in degrees, in the range ``[-90, 90)``. Usable
            directly as the ``ellipse_params`` argument of
            :func:`polar_transform`.
        "center" : (row, col)
            Refined center = ``center + (row0, col0)``.
        "fit_params" : Tensor of shape (11,)
            All 11 fitted parameters in canonical-conic form. Recover
            ``(a, b, θ_rad)`` from the last three with
            :func:`convert_ellipse_params`.
        "cost" : float
            Sum of squared residuals over the fit region at the
            best-loss iterate.
    """
    # Compute mean DP over all scan positions 
    data_t = (
        torch.from_numpy(np.ascontiguousarray(data.array))
        if data.array is not None
        else data.tensor
    )
    if data_t.ndim == 4:
        mean_dp = data_t.mean(dim=(0, 1), dtype=torch.float32)
    elif data_t.ndim == 2:
        mean_dp = data_t.to(torch.float32)
    else:
        raise ValueError(
            f"Got data with shape {tuple(data.shape)}. Expected a 2D or 4D-STEM dataset."
        )

    dp_t = mean_dp.to(device)
    n_row, n_col = dp_t.shape
    center_row, center_col = float(center[0]), float(center[1])
    r_inner, r_outer = float(fit_radii[0]), float(fit_radii[1])
    # Build coordinate grids relative to center
    row_offsets = torch.arange(n_row, dtype=torch.float32, device=device) - center_row
    col_offsets = torch.arange(n_col, dtype=torch.float32, device=device) - center_col
    offset_row_grid, offset_col_grid = torch.meshgrid(row_offsets, col_offsets, indexing="ij")
    r_grid = torch.sqrt(offset_col_grid**2 + offset_row_grid**2)
    # Select pixels and data in annular region
    annular_mask = (r_grid > r_inner) & (r_grid < r_outer)
    if mask is not None:
        if isinstance(mask, np.ndarray):
            mask_t = torch.from_numpy(mask.astype(bool)).to(device)
        else:
            mask_t = mask.bool().to(device)
        annular_mask = annular_mask & ~mask_t
    offset_row_fit = offset_row_grid[annular_mask]
    offset_col_fit = offset_col_grid[annular_mask]
    val_fit = dp_t[annular_mask]
    if val_fit.numel() == 0:
        raise ValueError("No pixels in the fitting annulus. Check center and fit_radii.")

    # Auto-estimate initial parameters if not provided
    if p0 is None:
        # Radial integral (r-weighted) to find ring peak.
        # Weighting by r suppresses the central beam contribution.
        r_flat = r_grid[annular_mask]
        # make a binned 1d hist to find ring peak
        nbins = max(int(r_outer - r_inner), 10)
        bin_step = (r_outer - r_inner) / nbins
        bin_idx = ((r_flat - r_inner) / bin_step).long().clamp(0, nbins - 1)
        radial_integral = torch.zeros(nbins, device=device).scatter_add_(0, bin_idx, val_fit)
        # get ring peak's radius in pixels (bin center)
        peak_idx = radial_integral.argmax().item()
        R_init = r_inner + (peak_idx + 0.5) * bin_step
        # set init param guesses
        c_bkgd_init = max(float(val_fit.min()), 0.0)
        near_ring = val_fit[(r_flat - R_init).abs() < 3.0]
        if near_ring.numel() == 0:
            raise ValueError(
                f"No annulus pixels within 3 px of R_init={R_init:.1f}. "
                "Annulus is too narrow or fit_radii is misaligned with the ring."
            )
        I1_init = max(float(near_ring.mean()) - c_bkgd_init, 1.0)
        I0_init = max(float(dp_t[int(center_row), int(center_col)]) - c_bkgd_init, 1.0)
        # All sigmas in pixel units
        sigma0_init = R_init * 0.3
        sigma1_init = R_init * 0.15
        sigma2_init = R_init * 0.2

        A_init, B_init, C_init = convert_ellipse_params_r(R_init, R_init, 0.0)
        p0 = torch.tensor(
            [
                I0_init,
                I1_init,
                sigma0_init,
                sigma1_init,
                sigma2_init,
                c_bkgd_init,
                0.0,  # row0 offset (relative to provided center)
                0.0,  # col0 offset
                A_init,  # conic A (init as circle of radius R_init)
                B_init,  # conic B
                C_init,  # conic C
            ],
            dtype=torch.float32,
            device=device,
        )
    else:
        if isinstance(p0, np.ndarray):
            p0 = torch.from_numpy(p0.astype(np.float32)).to(device)
        p0 = p0.float().to(device)

    def _inv_softplus(x: float) -> float:
        """Inverse of softplus: log(exp(x) - 1)."""
        if x > 20.0:
            return x
        return float(np.log(np.expm1(x)))

    # Optimize an unconstrained raw, the live param is softplus(raw) > 0 to 
    # ensure positivity. Adam can step a param through zero, so this guard is needed.
    _POS_IDX = [0, 1, 2, 3, 4, 5, 8, 10]  # I0, I1, sigma0-2, c_bkgd, A, C
    _CONIC_IDX = [8, 9, 10]  # A, B, C
    # Conic rescale by R0^2. The conic equation A·u²+B·uv+C·v² = 1 forces
    # A,B,C ~ 1/R² physically, but LR is tuned for the larger params, so without
    # this rescale (A,B,C) would barely move. We pick R0 from the initial
    # circle init and hold it fixed during optimization to avoid feedback loops.
    a0, b0, _ = convert_ellipse_params(
        float(p0[8]),
        float(p0[9]),
        float(p0[10]),
    )
    R2 = ((a0 + b0) / 2.0) ** 2
    raw = p0.clone()
    for i in _CONIC_IDX:
        raw[i] = raw[i] * R2
    for i in _POS_IDX:
        raw[i] = _inv_softplus(max(float(raw[i]), 1e-6))
    raw = raw.requires_grad_(True)

    def _to_physical(raw_params: torch.Tensor) -> torch.Tensor:
        """Map unconstrained raw params to physical params."""
        phys = raw_params.clone()
        for i in _POS_IDX:
            phys[i] = F.softplus(raw_params[i])
        for i in _CONIC_IDX:
            phys[i] = phys[i] / R2
        return phys

    optimizer = torch.optim.Adam([raw], lr=lr)
    # Track the best iterate seen
    best_loss = float("inf")
    best_raw = raw.detach().clone()
    for _ in range(max_iter):
        optimizer.zero_grad()
        phys = _to_physical(raw)
        model_vals = _amorphous_ring_model(phys, offset_row_fit, offset_col_fit)
        loss = ((model_vals - val_fit) ** 2).mean()
        lv = loss.item()
        if lv < best_loss:
            best_loss = lv
            best_raw = raw.detach().clone()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            # the canonical conic is an ellipse if B² < 4·A·C.
            # A and C stay > 0 from softplus; clamping |B| < 2·√(A·C)·0.999
            # enforces B² < 4·A·C strictly. 
            A_hat = F.softplus(raw[8])
            C_hat = F.softplus(raw[10])
            B_hat_max = 2.0 * torch.sqrt(A_hat * C_hat) * 0.999
            raw[9].clamp_(-B_hat_max, B_hat_max)
    # Extract final parameters
    with torch.no_grad():
        final_params = _to_physical(best_raw)
    final_loss = best_loss * val_fit.numel()
    a_fit, b_fit, theta_fit_rad = convert_ellipse_params(
        float(final_params[8]),
        float(final_params[9]),
        float(final_params[10]),
    )
    theta_fit_deg = float(np.degrees(theta_fit_rad))
    # Normalize angle to [-90, 90)
    theta_fit_deg = ((theta_fit_deg + 90.0) % 180.0) - 90.0
    # Refined center = nominal center + fitted offset
    refined_row = center_row + float(final_params[6])
    refined_col = center_col + float(final_params[7])
    if write_metadata:
        data.metadata["ellipticity"] = (a_fit, b_fit, theta_fit_deg)
    return {
        "ellipse_params": (a_fit, b_fit, theta_fit_deg),
        "center": (refined_row, refined_col),
        "fit_params": final_params,
        "cost": final_loss,
    }


def _amorphous_ring_model(
    params: torch.Tensor,
    offset_row: torch.Tensor,
    offset_col: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a parametric amorphous ring diffraction model.

    The model consists of a central-beam Gaussian, an asymmetric Gaussian ring
    (with separate inner and outer half-widths), and a constant background.
    Both the central beam and the ring share an elliptical radial metric
    given by ``A*x^2 + B*x*y + C*y^2 = 1`` (x = column offset, y = row offset 
    from the fitted center).

    Parameters
    ----------
    params : torch.Tensor
        11 model parameters: (I0, I1, sigma0, sigma1, sigma2, c_bkgd,
        row0, col0, A, B, C) where:
        - I0: central beam peak intensity
        - I1: ring peak intensity
        - sigma0: central beam width in pixels
        - sigma1: ring inner half-width in pixels
        - sigma2: ring outer half-width in pixels
        - c_bkgd: constant background
        - row0, col0: center offsets from nominal center in pixels
        - A, B, C: canonical conic parameters; the ring lies on
          ``A*u_col^2 + B*u_col*u_row + C*u_row^2 = 1``
    offset_row : torch.Tensor
        Row pixel offsets from the nominal center.
    offset_col : torch.Tensor
        Column pixel offsets from the nominal center.

    Returns
    -------
    torch.Tensor
        Model intensity values at each (offset_row, offset_col) position.
    """
    I0, I1, sigma0, sigma1, sigma2, c_bkgd, row0, col0, A, B, C = params
    # Shift coordinates by fitted center offset
    u_col = offset_col - col0
    u_row = offset_row - row0
    # Ring radius in pixels = mean semiaxis; rescale the conic by R^2 so the
    # elliptical radial coordinate r is in pixel units (r = R on the ring)
    a, b = _ellipse_semiaxes(A, B, C)
    ring_radius = (a + b) / 2.0
    r2 = ring_radius**2 * (A * u_col**2 + B * u_col * u_row + C * u_row**2)
    r_elliptical = torch.sqrt(torch.clamp(r2, min=1e-12))
    # Central beam Gaussian in the shared elliptical metric
    central = I0 * torch.exp(-r2 / (2.0 * sigma0**2))
    # Asymmetric ring: deviation from the ring radius, in pixel units
    dr_pixels = r_elliptical - ring_radius
    inner_mask = (dr_pixels < 0).float()
    outer_mask = 1.0 - inner_mask
    ring = I1 * (
        inner_mask * torch.exp(-(dr_pixels**2) / (2.0 * sigma1**2))
        + outer_mask * torch.exp(-(dr_pixels**2) / (2.0 * sigma2**2))
    )
    return central + ring + c_bkgd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def convert_ellipse_params(A: float, B: float, C: float) -> tuple[float, float, float]:
    """Convert canonical-conic ellipse parameters to semiaxes and tilt."""
    val = np.sqrt((A - C) ** 2 + B**2)
    b4a = B**2 - 4.0 * A * C
    if B == 0:
        theta = 0.0 if A < C else np.pi / 2.0
    else:
        theta = np.arctan2((C - A - val), B)
    a = -np.sqrt(-2.0 * b4a * (A + C + val)) / b4a
    b = -np.sqrt(-2.0 * b4a * (A + C - val)) / b4a
    a, b = max(a, b), min(a, b)
    return float(a), float(b), float(theta)


def convert_ellipse_params_r(a: float, b: float, theta: float) -> tuple[float, float, float]:
    """Convert ellipse semiaxes and tilt to canonical-conic parameters.
    Inverse of :func:`convert_ellipse_params`.
    """
    sin2, cos2 = np.sin(theta) ** 2, np.cos(theta) ** 2
    a2, b2 = a**2, b**2
    A = sin2 / b2 + cos2 / a2
    C = cos2 / b2 + sin2 / a2
    B = 2.0 * (b2 - a2) * np.sin(theta) * np.cos(theta) / (a2 * b2)
    return float(A), float(B), float(C)


def _ellipse_semiaxes(
    A: torch.Tensor, B: torch.Tensor, C: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable semiaxis lengths of the conic ``A*x^2 + B*x*y + C*y^2 = 1``.

    Torch counterpart of the (a, b) part of :func:`convert_ellipse_params`,
    used inside the fit model. The clamps guard the sqrt at the circular
    point (where ``(A - C)**2 + B**2 == 0``) and keep the discriminant
    strictly negative so gradients stay finite.
    """
    val = torch.sqrt(torch.clamp((A - C) ** 2 + B**2, min=1e-24))
    b4a = torch.clamp(B**2 - 4.0 * A * C, max=-1e-24)
    a = -torch.sqrt(torch.clamp(-2.0 * b4a * (A + C + val), min=1e-24)) / b4a
    b = -torch.sqrt(torch.clamp(-2.0 * b4a * (A + C - val), min=1e-24)) / b4a
    return a, b
