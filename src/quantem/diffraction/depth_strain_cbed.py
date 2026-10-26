"""Depth-resolved strain from whole 4DSTEM CBED patterns: many beams, 2D disks.

The extension of ``DepthStrain`` from the qx plot of one systematic row to the
full diffraction pattern. Every incident direction t inside the aperture
(2D) is propagated through every column x as a Bloch state of all beams g near
the Ewald sphere (``select_dynamical_beams``), with the strain phase
2 pi h g_x eps(x, z) per slice (eps = d u_x / d z, x along the row of
``g_hkl``), a Strang split step with one exact propagator per tilt
(``matrix_exp``). The amplitudes are placed in their disks on the detector,
summed coherently where disks overlap, and compared with the measured patterns
pixel by pixel. Off-row disks are modelled explicitly instead of being folded
into effective structure factors of the row.

Measured effects in the forward model, each with parameters fitted to the data:

- crystal: orientation (indexed from the off-row disks and their Bragg lines,
  ``fit_orientation``), incident tilt (2D), thickness, optionally the top
  and bottom surface profiles (``add_surfaces``), a thickness per column, and
  the composition of a second phase on the same lattice, which follows the
  strain it causes (``set_composition``, e.g. AlN in GaN for an AlGaN layer);
- probe: aperture radius and edge, a non-uniform illumination across the
  aperture (radial profile with first and second angular harmonics), and the
  probe aberrations as ray displacements (defocus, two-fold astigmatism,
  spherical aberration);
- detector: pixel size along and across the row (elliptic distortion) and the
  orientation of the pattern (rotation by 180 degrees or mirrors, which a
  two-beam model cannot tell apart but the dynamical intensities of a polar
  crystal can);
- background: per pattern, linear in the Bragg intensity, two diffuse halos
  around the disks (the Bragg pattern blurred: thermal diffuse scattering
  follows the beams) and a smooth radial term, all with non-negative
  coefficients;
- noise: weights from the two independent half sets of the scan (even and odd
  lines along the feature), counting noise plus a multiplicative floor for
  the specimen texture the model cannot describe.

The forward model runs in single precision on the quantEM device
(``config.set_device``, e.g. "mps" or "cuda"), the parameters stay double
precision on the CPU. ``fit`` and ``fit_dip`` use the quantEM optimizers
(``OptimizerMixin``, Adam with cosine annealing by default); ``duty`` < 1
pauses after every step to share the GPU. The deep image prior uses the
network of ``DepthStrain``.

Workflow: ``DepthStrain.fit_geometry`` on the 4DSTEM data (direct beam,
row direction, q sampling, descan) -> ``from_dataset4dstem`` (cosmic rays,
averages along the feature, descan alignment, binning) -> ``set_crystal`` ->
``fit_orientation`` -> ``set_model`` -> ``fit`` -> ``plot_patterns``,
``plot_qx``, ``plot_strain``.

Units are Angstroms and 1/Angstroms (no 2 pi); the beam energy is in eV.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch import nn

from quantem.core import config
from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.io.serialize import AutoSerialize
from quantem.core.ml.cnn import CNN2d
from quantem.core.ml.optimizer_mixin import OptimizerMixin
from quantem.core.visualization.visualization import show_2d
from quantem.diffraction.bloch import _coupling_matrix, relativistic_gamma, select_dynamical_beams
from quantem.diffraction.crystal import Crystal
from quantem.diffraction.depth_strain import _hat, _StrainNet
from quantem.diffraction.reverse_monte_carlo import electron_wavelength
from quantem.diffraction.rotations import quat_from_matrix

# optimizer variable = parameter * scale, so that a unit step is a typical change
_SCALES = dict(
    strain=1e3,
    thickness=0.1,
    tilt=100.0,
    rot_z=1e3,
    q_scale=100.0,
    alpha=100.0,
    edge=1e3,
    illum=1.0,
    background=1.0,
    aberrations=1e-3,
    surface=0.1,
    debye_waller=10.0,
    absorption=10.0,
    vegard=100.0,
)


def _interp(t: torch.Tensor, nodes: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Linear interpolation of node values at t, held constant outside the nodes."""
    step = nodes[1] - nodes[0]
    u = ((t - nodes[0]) / step).clamp(0, len(nodes) - 1 - 1e-6)
    i = u.floor().long()
    f = u - i
    return values[..., i] * (1 - f) + values[..., i + 1] * f


def _nnls(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Non-negative least squares for a few basis functions, batched: A (P, K, K) normal
    matrices, b (P, K). The solution is the unconstrained one on the subset of the basis
    with all coefficients positive that lowers the residual most (2^K subsets)."""
    K = b.shape[-1]
    masks = torch.tensor(
        [[(s >> k) & 1 for k in range(K)] for s in range(2**K)], dtype=A.dtype, device=A.device
    )
    eye = torch.eye(K, dtype=A.dtype, device=A.device)
    As = (
        A[None] * (masks[:, None, :, None] * masks[:, None, None, :])
        + eye * (1 - masks)[:, None, None, :]
    )
    x = torch.linalg.solve(As, (b[None] * masks[:, None])[..., None])[..., 0]
    gain = (x * b[None]).sum(-1)
    gain = torch.where((x >= 0).all(-1), gain, torch.full_like(gain, -torch.inf))
    best = gain.argmax(0)
    return x.gather(0, best[None, :, None].expand(1, *b.shape))[0]


def _rotation(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def _device(device=None) -> torch.device:
    """The quantEM device (``config``), or ``device``."""
    name, _ = config.validate_device(config.get("device") if device is None else device)
    return torch.device(name)


def _sync(dev: torch.device) -> None:
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


class _UNetDIP(nn.Module):
    """The ``CNN2d`` U-Net of the ptychography DIP for one field: a fixed random input on a grid
    rounded up to a multiple of ``2**levels``, the output cropped to the node grid."""

    def __init__(self, out_shape, start_filters=16, levels=3, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.out_shape = tuple(int(n) for n in out_shape)
        m = 2**levels
        work = tuple(int(np.ceil(n / m)) * m for n in self.out_shape)
        self.register_buffer("code", 0.1 * torch.randn((1, 1) + work, generator=gen))
        torch.manual_seed(seed)
        self.net = CNN2d(
            in_channels=1, out_channels=1, start_filters=start_filters, num_layers=levels
        )

    def forward(self, noise: float = 0.0) -> torch.Tensor:
        x = self.code if noise <= 0 else self.code + noise * torch.randn_like(self.code)
        return self.net(x)[0, 0, : self.out_shape[0], : self.out_shape[1]]


class DepthStrainCBED(AutoSerialize, OptimizerMixin):
    """Depth-resolved strain eps(x, z) from whole CBED patterns along a line scan.

    The patterns are ``Dataset3d`` (x, q along the row, q across), all the
    lines along the feature averaged (``patterns``) and the even and the odd
    lines (``even``, ``odd``). Parameters live in ``params`` (double
    precision torch tensors): ``strain`` (nz, nx) node values of d u_x / d z,
    ``thickness`` (A), ``tilt`` (2,) incident tilt of the crystal (1/A),
    ``rot_z`` rotation of the crystal about the beam (rad), ``q_scale`` (2,)
    detector scale along and across the row, ``alpha`` aperture radius (1/A),
    ``edge`` aperture edge width (1/A), ``illum`` (5, n) log amplitude of the
    illumination across the aperture (radial profiles from the center to the
    edge of the harmonics 1, cos, sin, cos 2, sin 2 of the azimuth), ``background`` (n,) log of the radial
    background, ``debye_waller`` (a change Delta B of the Debye-Waller factor, A^2,
    on the elastic structure factors) and ``absorption`` (a scale of the absorptive ones):
    the physical way to correct the structure factors of ``set_crystal``, ``aberrations``
    (C1, A1 along, A1 diagonal, all in A, and C3
    in mm), and with ``add_surfaces`` the ``surface`` (2, n) heights of the
    top and bottom surfaces (A).
    """

    _token = object()
    _SKIP = (
        "_m",
        "_dip",
        "_opt_vars",
        "_optimizer",
        "_scheduler",
        "_optimizer_params",
        "_scheduler_params",
    )

    def __init__(self, patterns, even, odd, energy, geometry, binning, _token=None):
        if _token is not self._token:
            raise RuntimeError("Use DepthStrainCBED.from_dataset4dstem().")
        self.patterns: Dataset3d = patterns
        self.even: Dataset3d = even
        self.odd: Dataset3d = odd
        self.energy = float(energy)
        self.wavelength = electron_wavelength(self.energy)
        self.geometry = geometry
        self.binning = int(binning)
        self.crystal: Crystal | None = None
        self.crystal_alt: Crystal | None = None
        self.g_hkl: tuple[int, int, int] | None = None
        self.orientation: dict | None = None
        self.config: dict | None = None
        self.params: dict | None = None
        self.history: list[float] = []
        self.result: dict | None = None
        self._m: dict | None = None
        self._dip: dict | None = None
        self._opt_vars: dict | None = None
        self.x = self._positions_x()  # probe positions along x (A)
        OptimizerMixin.__init__(self)

    def save(self, path, mode="w", store="auto", skip=(), compression_level=4) -> None:
        """Save with ``AutoSerialize`` (zarr), without the device copies of the model (rebuilt from
        ``config`` on first use), the optimizer state and the DIP network (its strain nodes are
        saved)."""
        skip = (skip,) if isinstance(skip, str) else tuple(skip)
        super().save(path, mode, store, skip + self._SKIP, compression_level)

    def _post_load(self) -> None:
        self._m = None
        self._dip = None
        self._opt_vars = None
        if not hasattr(self, "x"):
            self.x = self._positions_x()
        if not hasattr(self, "_optimizer_params"):
            OptimizerMixin.__init__(self)

    # -------------------------------------------------------------------- input

    @classmethod
    def from_dataset4dstem(
        cls,
        dataset,
        geometry: dict,
        energy: float,
        x_range: tuple[int, int] | None = None,
        x_origin: float | None = None,
        average_axis: int = 1,
        binning: int = 2,
        cosmic_ray_sigma: float | None = 10.0,
        verbose: bool = True,
    ) -> "DepthStrainCBED":
        """Patterns along x from a 4DSTEM scan, averaged along the feature.

        Pixels more than ``cosmic_ray_sigma`` robust standard deviations above
        the median of the positions along ``average_axis`` (cosmic rays) are
        replaced by that median. The patterns are averaged along that axis,
        all positions and, as independent half sets for the noise and for
        validation, the even and the odd ones. Every pattern is shifted
        (bilinear) so that its direct beam (``geometry["centers_fit"]``, the
        descan of ``DepthStrain.fit_geometry``) sits at the mean position, and
        binned ``binning`` x ``binning``.

        Parameters
        ----------
        dataset : Dataset4dstem
            (scan_0, scan_1, ky, kx), with the scan sampling in A or nm.
        geometry : dict
            ``DepthStrain.geometry`` of the same scan (``fit_geometry``).
        energy : float
            Beam energy in eV.
        x_range : (int, int), optional
            Scan positions kept along x, as (start, stop).
        x_origin : float, optional
            x of the first kept position (A). Default: x = 0 in the middle.
        """
        if average_axis not in (0, 1):
            raise ValueError("average_axis is 0 or 1")
        keep = 1 - average_axis
        units = str(dataset.units[keep]).lower()
        x_sampling = float(np.asarray(dataset.sampling)[keep]) * (10.0 if units == "nm" else 1.0)
        arr = dataset.array
        n = arr.shape[keep]
        i0, i1 = (0, n) if x_range is None else x_range
        sel = [slice(None)] * 4
        sel[keep] = slice(i0, i1)
        a = np.asarray(arr[tuple(sel)], dtype=np.float32)
        if average_axis == 0:
            a = np.swapaxes(a, 0, 1)
        if cosmic_ray_sigma:
            med = np.median(a, axis=1, keepdims=True)
            mad = 1.4826 * np.median(np.abs(a - med), axis=1, keepdims=True)
            bad = (a - med) > cosmic_ray_sigma * mad + 3 * float(np.median(mad))
            if verbose:
                print(f"cosmic rays: {int(bad.sum())} pixels replaced ({100 * bad.mean():.4f} %)")
            a = np.where(bad, np.broadcast_to(med, a.shape), a)
            del med, mad, bad
        centers = np.asarray(geometry["centers_fit"])[i0:i1]
        c0 = centers.mean(0)
        H, W = (s // binning for s in a.shape[2:])
        n_kept = i1 - i0
        if x_origin is None:
            x_origin = -0.5 * (n_kept - 1) * x_sampling
        q_px = binning * float(geometry["q_sampling"])
        center = (c0 - 0.5 * (binning - 1)) / binning

        def prepare(stack, name):
            out = np.empty((stack.shape[0], H, W), np.float32)
            for i in range(stack.shape[0]):
                im = ndimage.shift(
                    stack[i].astype(np.float64), c0 - centers[i], order=1, mode="nearest"
                )
                im = im[: H * binning, : W * binning]
                out[i] = im.reshape(H, binning, W, binning).sum((1, 3))
            return Dataset3d.from_array(
                out,
                name=name,
                origin=(x_origin, -center[0] * q_px, -center[1] * q_px),
                sampling=(x_sampling, q_px, q_px),
                units=("A", "A^-1", "A^-1"),
                signal_units="counts",
            )

        patterns = prepare(a.mean(1), "all lines")
        even = prepare(a[:, 0::2].mean(1), "even lines")
        odd = prepare(a[:, 1::2].mean(1), "odd lines")
        geo = dict(
            direction=np.asarray(geometry["direction"], dtype=np.float64),
            q_sampling=float(geometry["q_sampling"]),
            radius_px=float(geometry["radius_px"]),
            radius_across_px=float(geometry.get("radius_across_px", geometry["radius_px"])),
            center=center,
        )
        if verbose:
            print(
                f"{n_kept} positions ({x_sampling:.3f} A steps), patterns {H} x {W} after "
                f"{binning} x {binning} binning, {a.shape[1]} lines averaged (half sets of "
                f"{a[:, 0::2].shape[1]} and {a[:, 1::2].shape[1]})"
            )
        return cls(patterns, even, odd, energy, geo, binning, _token=cls._token)

    def _positions_x(self) -> np.ndarray:
        """Probe positions along x (A)."""
        n = self.patterns.shape[0]
        return float(self.patterns.origin[0]) + np.arange(n) * float(self.patterns.sampling[0])

    def set_crystal(
        self, crystal: Crystal, g_hkl=(0, 1, 2), k_max: float = 3.2
    ) -> "DepthStrainCBED":
        """Crystal (absorptive Weickenmeier-Kohl factors to ``k_max``) and the row reflection g,
        which defines x (the strain is d u_x / d z with x along g)."""
        crystal.calculate_dynamical_structure_factors(self.energy, k_max=k_max)
        self.crystal = crystal
        self.g_hkl = tuple(int(v) for v in g_hkl)
        self.k_max_dyn = float(k_max)
        return self

    # ------------------------------------------------------------- orientation

    def _detector_q(self, sx: int = 1, sy: int = 1):
        """Lab q (along, across the row) of every detector pixel for a detector orientation, and
        the pixel coordinates along and across the row."""
        geo = self.geometry
        H, W = self.patterns.shape[1:]
        rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        d = geo["direction"]
        p = np.array([-d[1], d[0]])
        rel = np.stack([rr - geo["center"][0], cc - geo["center"][1]], -1)
        s_par = self.binning * geo["q_sampling"]
        s_perp = s_par * geo["radius_px"] / geo["radius_across_px"]
        return sx * (rel @ d) * s_par, sy * (rel @ p) * s_perp, (rel @ d), (rel @ p)

    def _frame(self, psi_deg: float, rot_y: float = 0.0, rot_z: float = 0.0) -> np.ndarray:
        """Crystal-to-lab rotation: g along lab x, the beam along lab -z, rotated by psi about g."""
        B = self.crystal.lat_recip.numpy()
        g1 = np.asarray(self.g_hkl, dtype=np.float64) @ B
        ex = g1 / np.linalg.norm(g1)
        ref = np.array([0.0, 0.0, 1.0]) if abs(ex[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e2 = np.cross(ex, ref)
        e2 /= np.linalg.norm(e2)
        e3 = np.cross(ex, e2)
        psi = np.deg2rad(psi_deg)
        bz = np.cos(psi) * e2 + np.sin(psi) * e3
        R0 = np.stack([ex, np.cross(ex, bz), -bz])
        return _rotation([0, 0, 1], rot_z) @ _rotation([0, 1, 0], rot_y) @ R0

    def fit_orientation(
        self,
        detectors: Sequence[tuple[int, int]] = ((1, 1), (-1, -1), (1, -1), (-1, 1)),
        psi_step: float = 0.1,
        thickness: float = 500.0,
        alpha: float | None = None,
        device=None,
        verbose: bool = True,
    ) -> dict:
        """Rotation about g from the off-row disks and their Bragg lines.

        With g along x and the beam perpendicular to it, one angle psi about g
        remains. Every psi (and every detector orientation, a sign along and
        across the row) is scored by the correlation of a two-beam pattern of
        all reflections near the Ewald sphere (disks with their excess Bragg
        lines, extinction from the structure factors) with the mean measured
        pattern away from the systematic row, both high-passed the same way so
        that disk edges and lines count, not the diffuse background. The best
        psi is refined with small rotations about the lab y (the tilt along
        the row) and z (in-plane) axes. A two-beam model obeys Friedel's law,
        so detector orientations related by a 180 degree rotation score the
        same; ``set_model(detector=...)`` and a dynamical fit decide.

        Returns and stores ``orientation``: the ``detectors``, the ``scan``
        (n_detectors, n_psi) of correlations over ``psi``, and per detector the
        ``best`` psi, refined rotation matrix ``R`` and correlation.
        """
        dev = _device(device)
        f32 = dict(dtype=torch.float32, device=dev)
        cr = self.crystal
        cr.calculate_structure_factors(k_max=2.6)
        k0 = 1 / self.wavelength
        gamma = relativistic_gamma(self.energy)
        G_c = cr.hkl.numpy() @ cr.lat_recip.numpy()
        Fh = np.abs(cr.struct_factors.numpy())
        gl = np.linalg.norm(G_c, axis=1)
        keep = (gl > 0.3) & (gl < 2.3) & (Fh > 0.01 * Fh.max())
        Gc = torch.as_tensor(G_c[keep], **f32)
        w_ext = torch.as_tensor(gamma * Fh[keep] / np.pi / (2 * k0), **f32)
        alpha = alpha or self.geometry["radius_px"] * self.geometry["q_sampling"]
        img = self.patterns.array.mean(0)
        H, W = img.shape
        qpar, qperp, _, _ = self._detector_q()
        rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        sel = ((np.abs(qperp) > alpha + 0.03) | (np.abs(qpar) > 1.35)) & (rr > 2) & (rr < H - 3)
        sel &= (cc > 2) & (cc < W - 3)
        dimg = np.sqrt(np.clip(img, 0, None))
        dimg = dimg - ndimage.gaussian_filter(dimg, 5)
        dv = torch.as_tensor(dimg[sel], **f32)
        dv = (dv - dv.mean()) / dv.std()
        ky, kx = np.fft.fftfreq(H)[:, None], np.fft.fftfreq(W)[None, :]
        lp = torch.as_tensor(np.exp(-2 * (np.pi * 5) ** 2 * (kx**2 + ky**2)), **f32)
        qp = torch.as_tensor(qpar.ravel(), **f32)
        qq = torch.as_tensor(qperp.ravel(), **f32)
        sel_t = torch.as_tensor(sel.ravel(), device=dev)

        def score(R, sx, sy):
            Gl = Gc @ torch.as_tensor(R.T, **f32)
            s0 = (2 * k0 * Gl[:, 2] - (Gl**2).sum(1)) / (2 * k0)
            near = torch.abs(s0) < 0.042
            G, w = Gl[near], w_ext[near]
            tx = sx * qp[:, None] - G[None, :, 0]
            ty = sy * qq[:, None] - G[None, :, 1]
            inside = (tx**2 + ty**2) < alpha**2
            s = s0[near][None, :] - (tx * G[None, :, 0] + ty * G[None, :, 1]) / k0
            se = torch.sqrt(s**2 + w[None] ** 2)
            inten = (w[None] ** 2) * torch.sin(np.pi * thickness * se) ** 2 / se**2
            mi = torch.sqrt((inten * inside).sum(1).reshape(H, W) + 1e-12)
            mi = mi - torch.fft.ifft2(torch.fft.fft2(mi) * lp).real
            mv = mi.reshape(-1)[sel_t]
            mv = (mv - mv.mean()) / (mv.std() + 1e-12)
            return float((mv * dv).mean())

        psis = np.arange(0.0, 360.0, psi_step)
        scans, best_all = [], []
        for sx, sy in detectors:
            sc = np.array([score(self._frame(ps), sx, sy) for ps in psis])
            scans.append(sc)
            ps0 = psis[int(np.argmax(sc))]
            best = (-np.inf, None)
            for ps in np.arange(ps0 - 2 * psi_step, ps0 + 2 * psi_step + 1e-9, psi_step / 10):
                for ay in np.arange(-0.0015, 0.00151, 0.0005):
                    for az in np.arange(-0.005, 0.00501, 0.00125):
                        v = score(self._frame(ps, ay, az), sx, sy)
                        if v > best[0]:
                            best = (v, (ps, ay, az))
            far = np.abs(((psis - ps0 + 180) % 360) - 180) > 1.0
            best_all.append(
                dict(
                    detector=(int(sx), int(sy)),
                    correlation=float(best[0]),
                    psi=float(best[1][0]),
                    rot_y=float(best[1][1]),
                    rot_z=float(best[1][2]),
                    R=self._frame(*best[1]),
                    runner_up=float(sc[far].max()),
                )
            )
            if verbose:
                print(
                    f"detector ({sx:+d}, {sy:+d}): psi {best[1][0]:.3f} deg, rot_y {best[1][1]:+.4f}, "
                    f"rot_z {best[1][2]:+.4f} rad, correlation {best[0]:.3f} "
                    f"(best more than 1 deg away {best_all[-1]['runner_up']:.3f})"
                )
        self.orientation = dict(
            psi=psis, detectors=[tuple(d) for d in detectors], scan=np.array(scans), best=best_all
        )
        return self.orientation

    def _rotation_for(self, detector) -> np.ndarray:
        for b in self.orientation["best"]:
            if tuple(b["detector"]) == tuple(detector):
                return np.asarray(b["R"])
        raise ValueError(f"no orientation for detector {detector}: fit_orientation first")

    # -------------------------------------------------------------------- model

    def set_model(
        self,
        detector: tuple[int, int] = (1, 1),
        R: np.ndarray | None = None,
        thickness: float = 500.0,
        tilt: tuple[float, float] = (0.0, 0.0),
        alpha: float | None = None,
        data: str = "all",
        positions: tuple[int, int] | None = None,
        average: bool = False,
        strain_shape: tuple[int, int] = (36, 60),
        strain_x_range: tuple[float, float] | None = None,
        strain_z_range: tuple[float, float] | None = None,
        x_step: float = 20.0,
        dz: float = 10.0,
        t_step: float | None = None,
        t_margin: float = 0.03,
        sg_max: float = 0.015,
        k_max: float = 2.1,
        n_illum: int = 24,
        n_background: int = 24,
        q_max: float = 1.6,
        halo_sigmas: tuple[float, ...] = (4.0, 12.0),
        texture: float = 0.05,
        device=None,
        verbose: bool = True,
    ) -> "DepthStrainCBED":
        """Beams, detector pixels, tilt grid, columns, strain grid and the starting parameters.

        Parameters of an earlier model whose shapes agree are kept.

        Parameters
        ----------
        detector : (int, int)
            Detector orientation: sign along and across the row ((-1, -1) is a
            rotation by 180 degrees).
        R : ndarray, optional
            Crystal-to-lab rotation; default the ``fit_orientation`` result for
            ``detector``.
        data : {"all", "even", "odd"}
            The patterns fitted; the others can be held out (``validate``).
        positions, average : (int, int), bool
            Fit only these positions, or their mean pattern (one column at their
            mean x, e.g. a strain-free region for the calibration).
        t_step : float, optional
            Incident tilt grid step (1/A); default 1.5 detector pixels.
        sg_max, k_max : float
            Beams with |s_g| < sg_max + alpha |g| and |g| < k_max, plus the row.
        texture : float
            Multiplicative noise floor for texture both half sets share.
        device : str, optional
            Torch device of the forward model; default the quantEM device.
        """
        if self.crystal is None:
            raise RuntimeError("set_crystal first")
        if R is None:
            R = self._rotation_for(detector)
        self.config = dict(
            detector=tuple(int(v) for v in detector),
            R=np.asarray(R),
            thickness=float(thickness),
            tilt=tuple(float(v) for v in tilt),
            alpha=alpha,
            data=data,
            positions=None if positions is None else tuple(positions),
            average=bool(average),
            strain_shape=tuple(strain_shape),
            strain_x_range=strain_x_range,
            strain_z_range=strain_z_range,
            x_step=float(x_step),
            dz=float(dz),
            t_step=t_step,
            t_margin=float(t_margin),
            sg_max=float(sg_max),
            k_max=float(k_max),
            n_illum=int(n_illum),
            n_background=int(n_background),
            q_max=float(q_max),
            halo_sigmas=tuple(halo_sigmas),
            texture=float(texture),
            device=device,
        )
        p0 = self.params or {}
        self._build(verbose=verbose)
        cfg = self.config
        alpha = cfg["alpha"] or self.geometry["radius_px"] * self.geometry["q_sampling"]
        f64 = dict(dtype=torch.float64)
        self.params = dict(
            strain=torch.zeros(cfg["strain_shape"], **f64),
            thickness=torch.tensor(cfg["thickness"], **f64),
            tilt=torch.tensor(cfg["tilt"], **f64),
            rot_z=torch.tensor(0.0, **f64),
            q_scale=torch.tensor([1.0, 1.0], **f64),
            alpha=torch.tensor(float(alpha), **f64),
            edge=torch.tensor(0.0046, **f64),
            illum=torch.zeros(5, cfg["n_illum"], **f64),
            background=torch.full((cfg["n_background"],), -6.0, **f64),
            aberrations=torch.zeros(4, **f64),
            debye_waller=torch.tensor(0.0, **f64),
            absorption=torch.tensor(1.0, **f64),
        )
        for k, v in p0.items():
            if k in self.params and tuple(v.shape) == tuple(self.params[k].shape):
                self.params[k] = v.detach().clone()
        if "surface" in p0:
            self.add_surfaces(p0["surface"].shape[1])
            self.params["surface"] = p0["surface"].detach().clone()
        if "vegard" in p0:
            self.params["vegard"] = p0["vegard"].detach().clone()
        return self

    def _build(self, verbose: bool = False) -> None:
        """Device copies and index tables of the model (from ``config``)."""
        cfg = self.config
        dev = _device(cfg["device"])
        f32 = dict(dtype=torch.float32, device=dev)
        lam = self.wavelength
        sx, sy = cfg["detector"]
        R = np.asarray(cfg["R"])
        cr = self.crystal
        alpha = cfg["alpha"] or self.geometry["radius_px"] * self.geometry["q_sampling"]
        # beams: near the Ewald sphere for every tilt in the aperture, plus the row
        b = select_dynamical_beams(
            cr,
            quat_from_matrix(torch.as_tensor(R)),
            self.energy,
            alpha_max_rad=1.1 * alpha * lam,
            sg_max=cfg["sg_max"],
            k_max=cfg["k_max"],
        )
        g = torch.as_tensor(self.g_hkl)
        row = torch.stack([n * g for n in range(-4, 5) if n != 0])
        hkl = torch.cat([b, row])
        _, ui = np.unique(hkl.numpy(), axis=0, return_index=True)
        hkl = hkl[np.sort(ui)]
        # elastic and absorptive parts of the coupling (the absorptive part and the Debye-Waller
        # factor of the elastic part can be refined: ``absorption``, ``debye_waller``)
        gamma = relativistic_gamma(self.energy)
        k_dyn = getattr(self, "k_max_dyn", 3.2)
        cr.calculate_dynamical_structure_factors(
            self.energy, k_max=k_dyn, include_core=False, include_phonon=False
        )
        U_el, _, _ = _coupling_matrix(cr, hkl, gamma)
        cr.calculate_dynamical_structure_factors(self.energy, k_max=k_dyn)
        U, u0i, _ = _coupling_matrix(cr, hkl, gamma)
        g_cryst = hkl.to(torch.float64) @ cr.lat_recip
        g_diff2 = ((g_cryst[:, None, :] - g_cryst[None, :, :]) ** 2).sum(-1)
        # data
        stack = dict(all=self.patterns.array, even=self.even.array, odd=self.odd.array)
        pos = slice(None) if cfg["positions"] is None else slice(*cfg["positions"])
        D = {k: np.asarray(v[pos]) for k, v in stack.items()}
        x_pos = self.x[pos]
        if cfg["average"]:
            D = {k: v.mean(0, keepdims=True) for k, v in D.items()}
            x_pos = np.array([x_pos.mean()])
        n_pos = len(x_pos)
        H, W = self.patterns.shape[1:]
        qpar, qperp, upar, uperp = self._detector_q(sx, sy)
        s_par = self.binning * self.geometry["q_sampling"]
        G0 = (R @ g_cryst.numpy().T).T
        reach = alpha + 3 * s_par
        lists, vis = [], []
        for i in range(len(hkl)):
            on = np.flatnonzero((np.hypot(qpar - G0[i, 0], qperp - G0[i, 1]) < reach).ravel())
            if len(on) > 0.15 * np.pi * (alpha / s_par) ** 2:
                lists.append(on)
                vis.append(i)
        union = np.unique(np.concatenate(lists))
        where = {j: i for i, j in enumerate(union)}
        P = max(len(s) for s in lists)
        idx = np.zeros((len(vis), P), np.int64)
        valid = np.zeros((len(vis), P), np.float32)
        for k, s in enumerate(lists):
            idx[k, : len(s)] = [where[j] for j in s]
            valid[k, : len(s)] = 1
        # incident tilts (relative to the optic axis) inside the aperture
        t_step = cfg["t_step"] or 1.5 * s_par
        ext = alpha + cfg["t_margin"]
        nT = 2 * int(np.ceil(ext / t_step)) + 1
        t_ext = (nT - 1) / 2 * t_step
        tv = np.linspace(-t_ext, t_ext, nT)
        TX, TY = np.meshgrid(tv, tv, indexing="xy")
        inside = np.hypot(TX, TY) <= ext + 1e-9
        # columns, slices and strain nodes
        x_step = cfg["x_step"] if not cfg["average"] else 200.0
        x_sim = torch.arange(
            np.floor(x_pos.min() / x_step) * x_step - x_step,
            np.ceil(x_pos.max() / x_step) * x_step + 1.5 * x_step,
            x_step,
            dtype=torch.float64,
        )
        xr = cfg["strain_x_range"] or (float(self.x.min()), float(self.x.max()))
        zr = cfg["strain_z_range"] or (0.0, 1.04 * cfg["thickness"])
        z_nodes = torch.linspace(*zr, cfg["strain_shape"][0], dtype=torch.float64)
        x_nodes = torch.linspace(*xr, cfg["strain_shape"][1], dtype=torch.float64)
        self._m = dict(
            dev=dev,
            hkl=hkl,
            U=U.to(torch.complex64).to(dev),
            U_cpu=U,
            U_el=U_el.to(torch.complex64).to(dev),
            U_abs=(U - U_el).to(torch.complex64).to(dev),
            g_diff2=g_diff2.to(torch.float32).to(dev),
            u0i=float(u0i),
            g_cryst=g_cryst,
            R=torch.as_tensor(R, dtype=torch.float64),
            sx=sx,
            sy=sy,
            x_pos=torch.as_tensor(x_pos),
            n_pos=n_pos,
            union=union,
            union_t=torch.as_tensor(union, device=dev),
            vis=torch.as_tensor(vis),
            pix_index=torch.as_tensor(idx, device=dev),
            pix_valid=torch.as_tensor(valid, device=dev),
            pix_index_cpu=torch.as_tensor(idx),
            u_par=torch.as_tensor(upar.ravel()[union]),
            u_perp=torch.as_tensor(uperp.ravel()[union]),
            s_par=s_par,
            s_perp=s_par * self.geometry["radius_px"] / self.geometry["radius_across_px"],
            q_abs=torch.as_tensor(np.hypot(qpar, qperp).ravel()[union]),
            nT=nT,
            t_ext=t_ext,
            t_in=torch.as_tensor(np.stack([TX[inside], TY[inside]], -1)),
            t_in_idx=torch.as_tensor(np.flatnonzero(inside.ravel()), device=dev),
            x_sim=x_sim,
            n_slices=max(int(round(cfg["thickness"] / cfg["dz"])), 1),
            z_nodes=z_nodes,
            x_nodes=x_nodes,
            wx=_hat(x_sim, x_nodes).T.contiguous(),
            illum_r=torch.linspace(0, 1, cfg["n_illum"], dtype=torch.float64),
            bg_q=torch.linspace(0, cfg["q_max"], cfg["n_background"], dtype=torch.float64),
            D={k: torch.as_tensor(v.reshape(n_pos, -1)[:, union], **f32) for k, v in D.items()},
            shape=(H, W),
        )
        ky, kx = np.fft.fftfreq(H)[:, None], np.fft.fftfreq(W)[None, :]
        self._m["halo"] = [
            torch.as_tensor(
                np.exp(-2 * (np.pi * s) ** 2 * (kx**2 + ky**2))[:, : W // 2 + 1], **f32
            )
            for s in cfg["halo_sigmas"]
        ]
        self._set_noise()
        if getattr(self, "crystal_alt", None) is not None:
            self._set_alt()
        if "surface" in (self.params or {}):
            n = self.params["surface"].shape[1]
            self._m["s_nodes"] = torch.linspace(-560.0, 560.0, n, dtype=torch.float64)
        if verbose:
            print(
                f"{len(hkl)} beams ({len(vis)} disks on the detector), {len(union)} pixels per "
                f"pattern, {n_pos} patterns ({cfg['data']}), {len(self._m['t_in'])} incident tilts, "
                f"{len(x_sim)} columns x {self._m['n_slices']} slices, "
                f"{cfg['strain_shape'][0]} x {cfg['strain_shape'][1]} strain nodes on {dev}"
            )

    def _model(self) -> dict:
        if self._m is None:
            self._build()
        return self._m

    def _set_noise(self) -> None:
        """Weights 1 / sigma: var = a I + b I^2 from (even - odd) / 2 binned by intensity, plus the
        texture floor; ``noise`` is the expected weighted residual of the fitted data."""
        m, cfg = self._m, self.config
        Dn = m["D"][cfg["data"]].cpu().numpy()
        H, W = m["shape"]
        img = np.zeros((Dn.shape[0], H * W))
        img[:, m["union"]] = Dn
        inten = np.clip(ndimage.gaussian_filter(img.reshape(-1, H, W), (0, 1, 1)), 0, None)
        inten = inten.reshape(Dn.shape[0], -1)[:, m["union"]]
        diff = ((m["D"]["even"] - m["D"]["odd"]) / 2).cpu().numpy()
        bins = np.quantile(inten, np.linspace(0, 1, 31))
        xs, ys = [], []
        for b0, b1 in zip(bins[:-1], bins[1:]):
            s = (inten >= b0) & (inten < b1)
            if s.any():
                xs.append(inten[s].mean())
                ys.append((diff[s] ** 2).mean())
        xs, ys = np.array(xs), np.array(ys)
        (a, b2), *_ = np.linalg.lstsq(np.stack([xs, xs**2], 1), ys, rcond=None)
        var = a * inten + (max(b2, 0) + cfg["texture"] ** 2) * inten**2 + a
        w = 1 / np.sqrt(var)
        w = w / w.mean()
        # a half set has twice the variance of the full set
        half = 2.0 if cfg["data"] in ("even", "odd") else 1.0
        m["weight"] = torch.as_tensor(w, dtype=torch.float32, device=m["dev"])
        m["noise"] = half * float(((w * diff) ** 2).sum())
        m["noise_model"] = (float(a), float(np.sqrt(max(b2, 0))))

    def noise_level(self) -> float:
        """Expected weighted residual of pure noise (the residual is reported in these units)."""
        return self._model()["noise"]

    def illumination_profile(self, n: int = 221) -> tuple[np.ndarray, np.ndarray]:
        """Azimuthal mean of the illumination intensity (aperture edge included) against the
        incident angle (mrad), out to 1.1 times the aperture radius."""
        p, m = self.params, self._model()
        a = float(p["alpha"])
        r = torch.linspace(0, 1.1 * a, n, dtype=torch.float64)[:, None]
        th = torch.linspace(0, 2 * np.pi, 72, dtype=torch.float64)[None, :-1]
        L, nodes = p["illum"].detach(), m["illum_r"]
        harmonics = (1, torch.cos(th), torch.sin(th), torch.cos(2 * th), torch.sin(2 * th))
        logA = sum(_interp(r / a, nodes, L[i]) * h for i, h in enumerate(harmonics))
        amp = torch.sigmoid((a - r) / float(p["edge"])) * torch.exp(logA)
        return 1e3 * self.wavelength * r[:, 0].numpy(), (amp**2).mean(1).numpy()

    def set_composition(
        self, crystal_alt: Crystal, vegard: float, k_max: float = 3.2
    ) -> "DepthStrainCBED":
        """A second phase on the same lattice, with the composition that follows the strain.

        The structure matrix of a slice becomes (1 - c) U + c U_alt (virtual crystal), with
        U_alt the absorptive structure factors of ``crystal_alt`` on the lattice of the crystal of
        ``set_crystal`` (lattice changes are the strain), e.g. AlN at the GaN lattice: c is then the
        Al fraction of AlGaN. The mean absorption changes with c as well. The composition causes
        the strain (lattice mismatch), so it is not a field of its own: c = strain / ``vegard``,
        clamped to [0, 1], with ``vegard`` (parameter ``vegard``, fitted like the others; this is
        its start) the strain per unit fraction of the second phase.
        """
        crystal_alt.calculate_dynamical_structure_factors(self.energy, k_max=k_max)
        self.crystal_alt = crystal_alt
        self._model()
        self._set_alt()
        self.params["vegard"] = torch.tensor(float(vegard), dtype=torch.float64)
        return self

    @staticmethod
    def _composition(p: dict) -> torch.Tensor | None:
        """Composition nodes (``set_composition``): the strain over ``vegard``."""
        return p["strain"] / p["vegard"] if "vegard" in p else None

    def _set_alt(self) -> None:
        m = self._m
        U_alt, u0i_alt, _ = _coupling_matrix(
            self.crystal_alt, m["hkl"], relativistic_gamma(self.energy)
        )
        dU = (U_alt - m["U_cpu"]) + torch.diag(
            torch.full((len(m["hkl"]),), 1j * (u0i_alt - m["u0i"]), dtype=torch.complex128)
        )
        m["dU"] = dU.to(torch.complex64).to(m["dev"])

    def add_surfaces(self, n_nodes: int = 8) -> "DepthStrainCBED":
        """Top and bottom surface heights (A) on ``n_nodes`` positions along x: a thickness per
        column. The strain nodes stay at their depths; slices beyond them take the nearest nodes."""
        m = self._model()
        m["s_nodes"] = torch.linspace(-560.0, 560.0, n_nodes, dtype=torch.float64)
        self.params["surface"] = torch.zeros(2, n_nodes, dtype=torch.float64)
        return self

    # ------------------------------------------------------------------ physics

    def _lab_g(self, p: dict) -> torch.Tensor:
        m = self._m
        c, s = torch.cos(p["rot_z"]), torch.sin(p["rot_z"])
        z, o = 0 * c, 1 + 0 * c
        Rz = torch.stack([torch.stack([c, -s, z]), torch.stack([s, c, z]), torch.stack([z, z, o])])
        return (Rz @ m["R"] @ m["g_cryst"].T).T

    def _excitation(self, p: dict):
        m = self._m
        G = self._lab_g(p)
        T = m["t_in"] + p["tilt"][None, :]
        k0 = 1 / self.wavelength
        kz = torch.sqrt(k0**2 - (T**2).sum(1))
        num = 2 * kz[:, None] * G[None, :, 2] - 2 * (T @ G[:, :2].T) - (G**2).sum(1)[None]
        return G, num / (2 * (kz[:, None] - G[None, :, 2]))

    def _propagate(self, p: dict) -> torch.Tensor:
        """Exit amplitudes (n_col, nT * nT, n_beams) on the incident tilt grid."""
        m = self._m
        dev = m["dev"]
        G, s = self._excitation(p)
        S = m["n_slices"]
        xs = m["x_sim"]
        if "surface" in p:
            top = _hat(xs, m["s_nodes"]) @ p["surface"][0]
            bot = p["thickness"] + _hat(xs, m["s_nodes"]) @ p["surface"][1]
        else:
            top = torch.zeros_like(xs)
            bot = p["thickness"] + torch.zeros_like(xs)
        h = (bot - top) / S
        h0 = h.mean()
        k0 = 1 / self.wavelength
        U = m["U"]
        a_abs = p.get("absorption")
        dB = p.get("debye_waller")
        if a_abs is not None and dB is not None:
            # a change Delta B of the Debye-Waller factor on the elastic part,
            # exp(-Delta B |g - h|^2 / 4), and a scale of the absorptive part and mean absorption
            dw = torch.exp(-dB.to(torch.float32).to(dev) * m["g_diff2"] / 4)
            a32 = a_abs.to(torch.float32).to(dev)
            U = m["U_el"] * dw + m["U_abs"] * a32
            u0i = (m["u0i"] * a32).to(torch.complex64)
        else:
            u0i = torch.tensor(m["u0i"], dtype=torch.complex64, device=dev)
        iu = torch.tensor(1j, dtype=torch.complex64, device=dev)
        diag = (2 * k0 * s).to(torch.float32).to(dev).to(torch.complex64) + iu * u0i
        A = U[None] + torch.diag_embed(diag)
        coef = torch.tensor(0.5j * np.pi * self.wavelength, dtype=torch.complex64, device=dev)
        E_half = torch.linalg.matrix_exp(
            coef * h0.to(torch.float32).to(dev).to(torch.complex64) * A
        )
        if "surface" in p:
            # a column's slice is the common one times exp(i pi lambda (h - h0) A / 2), which
            # commutes with it and is close to the identity (cubic series)
            dh = (h - h0).to(torch.float32).to(dev).to(torch.complex64)
            X = coef * dh[:, None, None, None] * A[None]
            X2 = X @ X
            eye = torch.eye(A.shape[-1], dtype=torch.complex64, device=dev)
            E_half = E_half[None] @ (eye + X + X2 / 2 + X2 @ X / 6)
        else:
            E_half = E_half[None]
        E_full = E_half @ E_half
        z = top[None, :] + (torch.arange(S, dtype=torch.float64)[:, None] + 0.5) * h[None, :]
        Wz = _hat(z.reshape(-1), m["z_nodes"]).reshape(S, len(xs), -1)
        eps = torch.einsum("kcz,zc->kc", Wz, p["strain"] @ m["wx"])
        ph = (2 * np.pi) * (h[None, :, None] * eps[:, :, None]) * G[None, None, :, 0]
        ph = ph.to(torch.float32).to(dev)
        phase = torch.polar(torch.ones_like(ph), ph)[:, :, None, :]
        comp = self._composition(p)
        if comp is not None:
            # the change of the structure matrix with composition, at the slice midpoints: one
            # matrix per slice and column, the same for every incident direction
            c = torch.einsum("kcz,zc->kc", Wz, comp @ m["wx"]).clamp(0.0, 1.0)
            cc = (h[None, :] * c).to(torch.float32).to(dev).to(torch.complex64)
            nb = m["dU"].shape[0]
            X = (2 * coef * cc.reshape(-1)[:, None, None] * m["dU"][None]).contiguous()
            Cmat = torch.linalg.matrix_exp(X).reshape(S, len(xs), nb, nb)
        phi = E_half[..., 0].expand(len(xs), -1, -1)
        for k in range(S):
            Ek = E_full if k < S - 1 else E_half
            v = phi * phase[k]
            if comp is not None:
                v = torch.einsum("cij,ctj->cti", Cmat[k], v)
            phi = (Ek @ v[..., None])[..., 0]
        out = torch.zeros(len(xs), m["nT"] ** 2, A.shape[-1], dtype=torch.complex64, device=dev)
        out[:, m["t_in_idx"]] = phi
        return out

    def _render(self, p: dict, amp: torch.Tensor) -> torch.Tensor:
        """Bragg intensity (n_pos, n_pix) on the detector pixels of the disks."""
        m = self._m
        dev = m["dev"]
        G = self._lab_g(p)
        vis = m["vis"]
        Gv = G[vis, :2]
        qx = m["sx"] * m["u_par"] * m["s_par"] * p["q_scale"][0]
        qy = m["sy"] * m["u_perp"] * m["s_perp"] * p["q_scale"][1]
        idx = m["pix_index_cpu"]
        tx = qx[idx] - Gv[:, 0, None]
        ty = qy[idx] - Gv[:, 1, None]
        # probe aberrations as ray displacements: the ray of tilt t crosses the specimen at x + d(t)
        C1, A1a, A1b, C3 = p["aberrations"]
        lam = self.wavelength
        dx = lam * ((C1 + A1a) * tx + A1b * ty) + lam**3 * (C3 * 1e7) * (tx**2 + ty**2) * tx
        xs = m["x_sim"]
        gxn = 2 * (m["x_pos"][None, :, None] + dx[:, None, :] - xs[0]) / (xs[-1] - xs[0]) - 1
        gtx = (tx / m["t_ext"])[:, None, :].expand_as(gxn)
        gty = (ty / m["t_ext"])[:, None, :].expand_as(gxn)
        grid = torch.stack([gtx, gty, gxn], -1)[:, :, :, None, :].to(torch.float32).to(dev)
        nT = m["nT"]
        a = amp[:, :, vis].reshape(len(xs), nT, nT, len(vis)).permute(3, 0, 1, 2)
        out = F.grid_sample(
            torch.stack([a.real, a.imag], 1),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[..., 0]
        r = torch.hypot(tx, ty)
        th = torch.atan2(ty, tx)
        L = p["illum"]
        rn = r / p["alpha"]
        nodes = m["illum_r"]
        logA = (
            _interp(rn, nodes, L[0])
            + _interp(rn, nodes, L[1]) * torch.cos(th)
            + _interp(rn, nodes, L[2]) * torch.sin(th)
            + _interp(rn, nodes, L[3]) * torch.cos(2 * th)
            + _interp(rn, nodes, L[4]) * torch.sin(2 * th)
        )
        ap = torch.sigmoid((p["alpha"] - r) / p["edge"]) * torch.exp(logA)
        out = out * (ap.to(torch.float32).to(dev) * m["pix_valid"])[:, None, None, :]
        n_u = len(m["union"])
        re = torch.zeros(m["n_pos"], n_u, dtype=torch.float32, device=dev)
        im = torch.zeros_like(re)
        for k in range(len(vis)):
            re = re.index_add(1, m["pix_index"][k], out[k, 0])
            im = im.index_add(1, m["pix_index"][k], out[k, 1])
        return re**2 + im**2

    def _basis(self, p: dict, bragg: torch.Tensor) -> torch.Tensor:
        """Bragg intensity, diffuse halos around the disks, radial background: (K, n_pos, n_pix)."""
        m = self._m
        H, W = m["shape"]
        full = torch.zeros(m["n_pos"], H * W, dtype=torch.float32, device=m["dev"])
        full[:, m["union_t"]] = bragg
        Fb = torch.fft.rfft2(full.reshape(m["n_pos"], H, W))
        out = [bragg]
        for lp in m["halo"]:
            blur = torch.fft.irfft2(Fb * lp, s=(H, W)).reshape(m["n_pos"], -1)
            out.append(blur[:, m["union_t"]])
        bg = torch.exp(_interp(m["q_abs"], m["bg_q"], p["background"]))
        out.append(bg.to(torch.float32).to(m["dev"])[None, :].expand(m["n_pos"], -1))
        return torch.stack(out, 0)

    def _loss(
        self,
        p: dict,
        data: str | None = None,
        reg: float = 0.0,
        surface_reg: float = 0.0,
    ):
        m = self._model()
        Bk = self._basis(p, self._render(p, self._propagate(p)))
        D = m["D"][data or self.config["data"]]
        w = m["weight"]
        wb, wd = w[None] * Bk, w * D
        A = torch.einsum("kpj,lpj->pkl", wb, wb)
        b = torch.einsum("kpj,pj->pk", wb, wd)
        ridge = 1e-6 * torch.diag_embed(torch.diagonal(A, dim1=1, dim2=2))
        coef = _nnls(A + ridge, b)
        M = torch.einsum("pk,kpj->pj", coef, Bk)
        res = ((w * M - wd) ** 2).sum()
        pen = torch.zeros((), dtype=torch.float64)
        if reg > 0:
            pen = pen + reg * self._tv(p["strain"])
        if surface_reg > 0 and "surface" in p:
            pen = pen + surface_reg * (torch.diff(p["surface"], n=2, dim=1) ** 2).mean()
        pen = pen.to(torch.float32).to(m["dev"])
        return res + pen, dict(residual=res, penalty=pen, coef=coef, fit=M)

    def _tv(self, s: torch.Tensor) -> torch.Tensor:
        """Total variation of the strain nodes as in ``DepthStrain``: the mean gradient magnitude in
        physical units times the depth range of the nodes (independent of the node spacing)."""
        m = self._m
        zn, xn = m["z_nodes"], m["x_nodes"]
        depth = float(zn[-1] - zn[0])
        dz = F.pad(torch.diff(s, dim=0), (0, 0, 0, 1)) * (depth / float(zn[1] - zn[0]))
        dx = F.pad(torch.diff(s, dim=1), (0, 1, 0, 0)) * (depth / float(xn[1] - xn[0]))
        return torch.sqrt(dz**2 + dx**2 + 1e-12).mean()

    def tv_norm(self) -> float:
        """Total variation of the current strain nodes (the penalty per unit ``reg``)."""
        self._model()
        return float(self._tv(self.params["strain"]))

    # ---------------------------------------------------------------------- fit

    def get_optimization_parameters(self) -> dict[str, list[torch.Tensor]]:
        """OptimizerMixin hook: the scaled parameters of the current fit, or the DIP network."""
        if self._dip is not None and self._dip.get("active"):
            return {self.DEFAULT_OPTIMIZER_KEY: list(self._dip["net"].parameters())}
        if self._opt_vars is None:
            return {}
        return {self.DEFAULT_OPTIMIZER_KEY: list(self._opt_vars.values())}

    def _step(self, loss: torch.Tensor, t0: float, duty: float) -> None:
        self.zero_optimizer_grad()
        loss.backward()
        self.step_optimizer()
        self.step_scheduler(float(loss.detach()))
        if duty < 1:
            _sync(self._m["dev"])
            time.sleep((time.time() - t0) * (1 - duty) / duty)

    def fit(
        self,
        names: Sequence[str] = ("strain",),
        iters: int = 300,
        lr: float = 0.01,
        reg: float = 0.0,
        surface_reg: float = 0.0,
        optimizer: dict | None = None,
        scheduler: dict | None = None,
        duty: float = 1.0,
        verbose: int = 50,
    ) -> "DepthStrainCBED":
        """Fit the named parameters (scaled as in ``_SCALES``) with a quantEM optimizer.

        ``optimizer`` and ``scheduler`` are ``OptimizerMixin`` dicts (default
        Adam with learning rate ``lr`` and cosine annealing). ``reg`` is the TV
        weight on the strain nodes (as in ``DepthStrain``), ``surface_reg`` a
        curvature penalty on the surface profiles, and ``duty`` < 1 pauses after
        every step so that the GPU is busy only that fraction of the time. The
        residual is reported in units of ``noise``.
        """
        m = self._model()
        p = self.params
        self._opt_vars = {
            k: (p[k] * _SCALES[k]).detach().clone().requires_grad_(True) for k in names
        }
        self.set_optimizer(optimizer or {"name": "adam", "lr": lr})
        self.set_scheduler(scheduler or {"name": "cosine_annealing"}, num_iter=iters)
        try:
            for i in range(iters):
                t0 = time.time()
                q = dict(p)
                for k in names:
                    q[k] = self._opt_vars[k] / _SCALES[k]
                loss, aux = self._loss(
                    q,
                    reg=reg,
                    surface_reg=surface_reg,
                )
                self.history.append(float(aux["residual"].detach()) / m["noise"])
                self._step(loss, t0, duty)
                if verbose and (i % verbose == 0 or i == iters - 1):
                    print(f"  step {i:4d}: residual {self.history[-1]:.3f} x noise", flush=True)
            for k in names:
                p[k] = (self._opt_vars[k] / _SCALES[k]).detach()
        finally:
            self._opt_vars = None
            self.remove_optimizer()
        self.result = dict(residual=self.residual(), fitted=tuple(names), reg=reg)
        return self

    def residual(self, data: str | None = None, params: dict | None = None) -> float:
        """Weighted residual of the current fit in units of ``noise``, against the fitted
        patterns or another set (``data="odd"``: held out when the model fits the even ones)."""
        with torch.no_grad():
            _, aux = self._loss(params or self.params, data=data)
        return float(aux["residual"]) / self._m["noise"]

    def validate(self, data: str = "odd", params: dict | None = None) -> float:
        """Residual against the other half set (see ``residual``)."""
        return self.residual(data, params)

    # ---------------------------------------------------------------------- DIP

    def set_dip(
        self,
        network: str = "strainnet",
        channels: int = 32,
        levels: int = 3,
        scale: float = 1e-2,
        pretrain_iters: int = 3000,
        seed: int = 0,
        verbose: bool = True,
    ) -> "DepthStrainCBED":
        """Strain nodes as ``scale`` times the output of a network for a fixed random input (deep
        image prior).

        ``network`` is ``"strainnet"`` (``_StrainNet`` of ``DepthStrain``) or ``"unet"`` (the
        ``CNN2d`` U-Net of the ptychography DIP, ``levels`` levels and ``channels`` start
        filters). With ``pretrain_iters`` > 0 it is first trained to reproduce the current nodes
        (a fine-tune of that field); with 0 it starts from its random weights, so that the network
        is the only prior.
        """
        shape = tuple(self.params["strain"].shape)
        if network == "strainnet":
            net = _StrainNet(shape, channels=channels, levels=levels, seed=seed)
        elif network == "unet":
            net = _UNetDIP(shape, start_filters=channels, levels=levels, seed=seed)
        else:
            raise ValueError("network is 'strainnet' or 'unet'")
        self._dip = dict(net=net, scale=float(scale), active=True)
        if pretrain_iters > 0:
            target = self.params["strain"].float()
            norm = max(float((target**2).mean()), 1e-12)
            self.set_optimizer({"name": "adam", "lr": 1e-3})
            try:
                for _ in range(pretrain_iters):
                    self.zero_optimizer_grad()
                    loss = ((scale * net() - target) ** 2).mean() / norm
                    loss.backward()
                    self.step_optimizer()
            finally:
                self.remove_optimizer()
            if verbose:
                print(
                    f"DIP ({network}): pretrained to the current nodes, rms error "
                    f"{(loss.item() * norm) ** 0.5:.2e}"
                )
        elif verbose:
            print(f"DIP ({network}): random start, no pretraining")
        self._dip["active"] = False
        return self

    def fit_dip(
        self,
        iters: int = 1000,
        lr: float = 3e-4,
        reg: float = 0.0,
        validation: str | None = "odd",
        every: int = 50,
        duty: float = 1.0,
        verbose: bool = True,
    ) -> "DepthStrainCBED":
        """Train the network on the fitted patterns; with ``validation`` the nodes that predict
        those patterns best (scored every ``every`` steps, and at the start) are kept."""
        m = self._model()
        net, scale = self._dip["net"], self._dip["scale"]
        p = dict(self.params)

        def nodes():
            return (scale * net()).double()

        def score():
            with torch.no_grad():
                return self.residual(validation, {**p, "strain": nodes()})

        curve = []
        best = (np.inf, 0, nodes().detach().clone())
        if validation:
            best = (score(), 0, best[2])
            curve.append((0, best[0]))
        self._dip["active"] = True
        self.set_optimizer({"name": "adam", "lr": lr})
        self.set_scheduler({"name": "cosine_annealing"}, num_iter=iters)
        try:
            for i in range(iters):
                t0 = time.time()
                loss, aux = self._loss({**p, "strain": nodes()}, reg=reg)
                self.history.append(float(aux["residual"].detach()) / m["noise"])
                self._step(loss, t0, duty)
                if validation and ((i + 1) % every == 0 or i == iters - 1):
                    v = score()
                    curve.append((i + 1, v))
                    if v < best[0]:
                        best = (v, i + 1, nodes().detach().clone())
                    if verbose:
                        print(
                            f"  step {i + 1:4d}: fit {self.history[-1]:.3f}, {validation} lines "
                            f"{v:.3f} x noise (best {best[0]:.3f} at step {best[1]})",
                            flush=True,
                        )
        finally:
            self._dip["active"] = False
            self.remove_optimizer()
        if not validation:
            best = (np.nan, iters, nodes().detach().clone())
        self.params["strain"] = best[2]
        self.result = dict(
            residual=self.residual(), fitted=("dip",), reg=reg, best_step=best[1], validation=curve
        )
        return self

    # -------------------------------------------------------------------- plots

    def model_patterns(self) -> np.ndarray:
        """Fitted model on the disk pixels, (n_pos, n_pix)."""
        with torch.no_grad():
            _, aux = self._loss(self.params)
        return aux["fit"].cpu().numpy()

    def _image(self, v: np.ndarray, fill: float = 0.0) -> np.ndarray:
        """Values on the disk pixels as a detector image."""
        m = self._m
        H, W = m["shape"]
        out = np.full(H * W, fill)
        out[m["union"]] = v
        return out.reshape(H, W)

    def _positions(self, positions) -> list[int]:
        xs = self._model()["x_pos"].numpy()
        return [int(np.argmin(np.abs(xs - xp))) for xp in positions]

    def plot_patterns(
        self, positions=(-400, -120, 0, 150, 400), power: float = 0.4, axsize=(4, 4)
    ):
        """Measured (top row) and model (bottom row) patterns at the positions x (A), on the disk
        pixels, each column with one intensity scale (power-law stretch); ``show_2d``."""
        m = self._model()
        M = self.model_patterns()
        D = m["D"][self.config["data"]].cpu().numpy()
        xs = m["x_pos"].numpy()
        idx = self._positions(positions)
        norms = [
            dict(
                interval_type="manual",
                vmin=0.0,
                vmax=float(np.percentile(D[i], 99.7)),
                stretch_type="power",
                power=power,
            )
            for i in idx
        ]
        return show_2d(
            [
                [self._image(np.clip(D[i], 0, None)) for i in idx],
                [self._image(np.clip(M[i], 0, None)) for i in idx],
            ],
            norm=[norms, norms],
            cmap="inferno",
            title=[[f"experiment, x = {xs[i] / 10:+.0f} nm" for i in idx], ["model"] * len(idx)],
            axsize=axsize,
        )

    def plot_residuals(
        self, positions=(-400, -120, 0, 150, 400), vrange: float = 10.0, axsize=(4, 4)
    ):
        """Residual (data - model) / sigma at the positions x (A); ``show_2d``."""
        m = self._model()
        M = self.model_patterns()
        D = m["D"][self.config["data"]].cpu().numpy()
        Wn = m["weight"].cpu().numpy() / np.sqrt(m["noise"] / D.size)
        xs = m["x_pos"].numpy()
        idx = self._positions(positions)
        return show_2d(
            [self._image(Wn[i] * (D[i] - M[i])) for i in idx],
            norm=dict(interval_type="centered", half_range=vrange),
            cmap="RdBu_r",
            cbar=True,
            title=[f"(data - model) / sigma, x = {xs[i] / 10:+.0f} nm" for i in idx],
            axsize=axsize,
        )

    def residual_regions(self) -> dict:
        """Misfit per pixel (units of the noise) and its share in the direct disk, the first and
        higher orders of the row, and the off-row disks and gaps."""
        m = self._model()
        M = self.model_patterns()
        D = m["D"][self.config["data"]].cpu().numpy()
        R2 = (m["weight"].cpu().numpy() * (D - M)) ** 2 / (m["noise"] / D.size)
        qp = (m["u_par"] * m["s_par"]).numpy()
        qq = (m["u_perp"] * m["s_perp"]).numpy()
        g = float(
            np.linalg.norm(
                np.asarray(self.g_hkl, dtype=np.float64) @ self.crystal.lat_recip.numpy()
            )
        )
        a = float(self.params["alpha"])
        row = np.abs(qq) < a
        regions = {
            "direct disk": row & (np.abs(qp) < a),
            "row, first order": row & (np.abs(np.abs(qp) - g) < a),
            "row, higher orders": row & (np.abs(qp) > g + a),
            "off-row disks and gaps": ~row,
        }
        return {
            k: (float(R2[:, s].mean()), float(R2[:, s].sum() / R2.sum()))
            for k, s in regions.items()
        }

    def plot_qx(self, width_px: float = 2.5, power: float = 0.5, figsize=(17, 6.5)):
        """A cut along the row through measured and model patterns, and the residual / sigma."""
        m = self._model()
        M = self.model_patterns()
        D = m["D"][self.config["data"]].cpu().numpy()
        Wn = m["weight"].cpu().numpy() / np.sqrt(m["noise"] / D.size)
        sel = np.abs(m["u_perp"].numpy()) < width_px
        bins = np.round(m["u_par"].numpy()[sel]).astype(int)
        ub = np.unique(bins)

        def cut(A):
            a = A[:, sel]
            return np.stack([a[:, bins == b].mean(1) for b in ub], 1)

        dc, mc = cut(D), cut(M)
        xs = m["x_pos"].numpy()
        q = ub * m["s_par"] * m["sx"]
        ext = [q[0] * 10, q[-1] * 10, xs[-1] / 10, xs[0] / 10]
        v = np.nanpercentile(dc, 99.5) ** power
        fig, axs = plt.subplots(1, 3, figsize=figsize, sharey=True, layout="constrained")
        axs[0].imshow(
            np.clip(dc, 0, None) ** power, aspect="auto", extent=ext, cmap="gray", vmax=v
        )
        axs[1].imshow(
            np.clip(mc, 0, None) ** power, aspect="auto", extent=ext, cmap="gray", vmax=v
        )
        axs[2].imshow(
            cut(Wn * (D - M)), aspect="auto", extent=ext, cmap="RdBu_r", vmin=-10, vmax=10
        )
        for ax, t in zip(axs, ("experiment", "model", "(data - model) / sigma")):
            ax.set_title(t)
            ax.set_xlabel("q along g (1/nm)")
        axs[0].set_ylabel("x (nm)")
        return fig, axs

    def plot_strain(
        self,
        panels: dict | None = None,
        strain_range: float = 15.0,
        figsize=None,
        key: str = "strain",
    ):
        """The strain field d u_x / d z (x along g) inside the foil (between the surfaces of this
        fit), and further fields on the same axes: ``panels`` maps titles to None (this fit), node
        values on the grid of this fit, or callables f(z, x) (A). ``key="composition"`` shows the
        composition (``set_composition``) instead."""
        m = self._model()
        zn, xn = m["z_nodes"].numpy(), m["x_nodes"].numpy()
        x = np.linspace(xn[0], xn[-1], 401)[None, :]
        t = float(self.params["thickness"])
        xx = torch.as_tensor(x.ravel())
        if "surface" in self.params:
            top = (_hat(xx, m["s_nodes"]) @ self.params["surface"][0]).numpy()
            bot = t + (_hat(xx, m["s_nodes"]) @ self.params["surface"][1]).numpy()
        else:
            top, bot = np.zeros(x.size), np.full(x.size, t)
        # the whole foil; beyond the nodes the field is that of the nearest nodes, as in the model
        z0, z1 = min(zn[0], top.min()), max(zn[-1], bot.max())
        z = np.linspace(z0, z1, 221)[:, None]
        Wz = _hat(torch.as_tensor(z.ravel()), m["z_nodes"]).numpy()
        Wx = _hat(torch.as_tensor(x.ravel()), m["x_nodes"]).numpy()
        nodes = self._composition(self.params) if key == "composition" else self.params[key]
        own = Wz @ nodes.detach().numpy() @ Wx.T
        if key == "composition":
            own = np.clip(own, 0, 1)
            scale, cmap, vmin, vmax, label = 1.0, "magma", 0.0, 1.0, "composition c"
        else:
            scale, cmap, vmin, vmax = 1e3, "RdBu_r", -strain_range, strain_range
            label = r"$\partial u_x/\partial z$ ($10^{-3}$)"
        outside = (z < top[None, :]) | (z > bot[None, :])  # no crystal, no data
        panels = panels or {"this fit": None}
        n = len(panels)
        fig, axs = plt.subplots(
            n, 1, figsize=figsize or (8, 3.3 * n + 0.6), sharex=True, layout="constrained"
        )
        axs = np.atleast_1d(axs)
        ext = [xn[0] / 10, xn[-1] / 10, z1 / 10, z0 / 10]
        for ax, (title, f) in zip(axs, panels.items()):
            if f is None:
                field = own
            elif callable(f):
                field = f(z, x)
            else:  # node values on the grid of this fit
                field = Wz @ np.asarray(torch.as_tensor(f).detach().cpu()) @ Wx.T
            field = np.where(outside, np.nan, field)
            im = ax.imshow(
                scale * field,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
                extent=ext,
            )
            ax.plot(x.ravel() / 10, top / 10, "k", lw=0.8)
            ax.plot(x.ravel() / 10, bot / 10, "k", lw=0.8)
            ax.set_title(title, loc="left")
            ax.set_ylabel("z (nm)")
        axs[-1].set_xlabel("x (nm)")
        fig.colorbar(im, ax=axs, label=label, shrink=0.8)
        return fig, axs
