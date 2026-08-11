import matplotlib.pyplot as plt
import numpy as np


def _range_mask(x, xmin, xmax):
    """Return a boolean mask selecting `x` within [xmin, xmax], defaulting to all True."""
    if xmin is None and xmax is None:
        return np.ones_like(x, dtype=bool)
    xmin_eff = x.min() if xmin is None else xmin
    xmax_eff = x.max() if xmax is None else xmax
    if xmax_eff <= xmin_eff:
        raise ValueError(f"xmax must be > xmin (got xmin={xmin_eff}, xmax={xmax_eff}).")
    mask = (x >= xmin_eff) & (x <= xmax_eff)
    if not np.any(mask):
        raise ValueError("Requested plot range contains no data.")
    return mask


def _nearest_index(x, value):
    return int(np.argmin(np.abs(x - value)))


def plot_diagonal(
    padf,
    rmin: float | None = None,
    rmax: float | None = None,
    figsize: tuple[float, float] = (7, 5),
    returnfig: bool = False,
):
    """
    Plot the r = r' diagonal of the PADF as a 2D map with r on the x-axis
    (Angstrom) and theta on the y-axis (degrees).

    Equivalent view to PairAngleDistributionFunction.simple_plot(), but with
    physical-unit axes taken from padf.r / padf.theta_deg instead of raw
    array indices.
    """
    r = padf.r
    theta_deg = padf.theta_deg

    diag = np.einsum("iik->ik", padf.padf.numpy() if hasattr(padf.padf, "numpy") else np.asarray(padf.padf))  # (Nr, Ntheta)

    r_mask = _range_mask(r, rmin, rmax)
    r_sel = r[r_mask]
    diag = diag[r_mask, :].T  # (Ntheta, Nr_sel)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(r_sel, theta_deg, diag, shading="auto")
    fig.colorbar(im, ax=ax, label=r"$\Theta(r, r, \theta)$")

    ax.set_xlabel("r = r' (Å)")
    ax.set_ylabel(r"$\theta$ (degrees)")
    ax.set_title("PADF diagonal")
    fig.tight_layout()

    if returnfig:
        return fig
    plt.show()


def plot_theta0_map(
    padf,
    rmin: float | None = None,
    rmax: float | None = None,
    figsize: tuple[float, float] = (6, 5),
    returnfig: bool = False,
):
    """
    Plot Theta(r, r', theta=0) as a 2D map over the full (r, r') grid,
    reproducing the top row of Fig. 2 in Martin (2017).
    """
    return plot_rrp_slice(padf, theta_deg=0.0, rmin=rmin, rmax=rmax, figsize=figsize, returnfig=returnfig)


def plot_rrp_slice(
    padf,
    theta_deg: float,
    rmin: float | None = None,
    rmax: float | None = None,
    figsize: tuple[float, float] = (6, 5),
    returnfig: bool = False,
):
    """
    Plot Theta(r, r', theta) as a 2D map over the full (r, r') grid at a
    fixed angle theta (degrees), generalizing plot_theta0_map to any theta.
    """
    r = padf.r
    theta_axis = padf.theta_deg

    padf_arr = padf.padf.numpy() if hasattr(padf.padf, "numpy") else np.asarray(padf.padf)
    theta_idx = _nearest_index(theta_axis, theta_deg)
    slice_2d = padf_arr[:, :, theta_idx]  # (Nr, Nr')

    r_mask = _range_mask(r, rmin, rmax)
    r_sel = r[r_mask]
    slice_2d = slice_2d[np.ix_(r_mask, r_mask)]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(r_sel, r_sel, slice_2d.T, shading="auto")
    fig.colorbar(im, ax=ax, label=r"$\Theta(r, r', \theta)$")

    ax.set_xlabel("r (Å)")
    ax.set_ylabel("r' (Å)")
    ax.set_title(rf"$\Theta(r, r', \theta={theta_axis[theta_idx]:.1f}\degree)$")
    fig.tight_layout()

    if returnfig:
        return fig
    plt.show()


def plot_angular_slice(
    padf,
    r_shell: float | list[float],
    figsize: tuple[float, float] = (6, 4),
    returnfig: bool = False,
):
    """
    Plot Theta(r, r, theta) vs theta (degrees) at one or more fixed radial
    shells r = r', reproducing the top row of Fig. 3 in Martin (2017).

    r_shell : float or list of float
        Radial distance(s) in Angstrom (Angstrom) at which to take the r = r' shell.
        Nearest available r bin is used.
    """
    r = padf.r
    theta_deg = padf.theta_deg
    padf_arr = padf.padf.numpy() if hasattr(padf.padf, "numpy") else np.asarray(padf.padf)

    shells = [r_shell] if np.isscalar(r_shell) else list(r_shell)

    fig, ax = plt.subplots(figsize=figsize)
    for shell in shells:
        idx = _nearest_index(r, shell)
        ax.plot(theta_deg, padf_arr[idx, idx, :], label=f"r = r' = {r[idx]:.2f} Å")

    ax.set_xlabel(r"$\theta$ (degrees)")
    ax.set_ylabel(r"$\Theta(r, r, \theta)$")
    ax.set_title("Angular dependence of PADF")
    ax.legend()
    fig.tight_layout()

    if returnfig:
        return fig
    plt.show()
