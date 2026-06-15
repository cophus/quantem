"""show3d_atoms: interactive 3D volume slice viewer with an atomic-site overlay.

Shows a single orthogonal slice (xy / xz / yz) through a 3D volume in one panel,
with a movable slice position, an intensity histogram with an adjustable color
range, and the traced atomic sites overlaid.  Marker size scales with site
intensity and marker opacity falls off with distance from the slice, so you can
judge tracing quality and pick thresholds slice by slice.

All coordinates are in voxel / array-index space so the sites register exactly
with the volume.
"""

import pathlib

import anywidget
import numpy as np
import traitlets


class Show3DAtoms(anywidget.AnyWidget):
    """Interactive orthogonal-slice viewer with atomic-site overlay.

    Parameters
    ----------
    volume : ndarray or Dataset3d
        3D scalar volume ``(n0, n1, n2)``.
    sites : ndarray, optional
        ``(N, >=3)`` array of sites in voxel coordinates; columns are
        ``[a0, a1, a2, intensity, sigma]`` (intensity/sigma optional, padded
        with zeros).
    sampling : sequence of float, optional
        Voxel size per axis (for axis labels). Default ``(1, 1, 1)``.
    title : str, optional
        Title shown above the panel.
    cmap : str, default "gray"
        Colormap name.
    """

    _esm = pathlib.Path(__file__).parent / "static" / "show3d_atoms.js"

    # Data (voxel/array-index space).
    volume_bytes = traitlets.Bytes(b"").tag(sync=True)
    sites_bytes = traitlets.Bytes(b"").tag(sync=True)
    n0 = traitlets.Int(0).tag(sync=True)
    n1 = traitlets.Int(0).tag(sync=True)
    n2 = traitlets.Int(0).tag(sync=True)
    num_sites = traitlets.Int(0).tag(sync=True)
    sampling = traitlets.List(traitlets.Float(), default_value=[1.0, 1.0, 1.0]).tag(sync=True)

    # Display / interaction state.
    title = traitlets.Unicode("").tag(sync=True)
    cmap = traitlets.Unicode("gray").tag(sync=True)
    plane = traitlets.Unicode("xy").tag(sync=True)  # "xy" | "xz" | "yz"
    slice_index = traitlets.Int(0).tag(sync=True)
    slice_thickness = traitlets.Float(3.0).tag(sync=True)  # voxels at full opacity
    opacity_falloff = traitlets.Float(3.0).tag(sync=True)  # extra voxels fading to 0
    vmin_pct = traitlets.Float(0.0).tag(sync=True)
    vmax_pct = traitlets.Float(100.0).tag(sync=True)
    marker_scale = traitlets.Float(1.0).tag(sync=True)
    marker_linewidth = traitlets.Float(1.0).tag(sync=True)
    marker_filled = traitlets.Bool(True).tag(sync=True)
    show_sites = traitlets.Bool(True).tag(sync=True)
    show_slice = traitlets.Bool(True).tag(sync=True)
    canvas_size = traitlets.Int(520).tag(sync=True)

    def __init__(
        self,
        volume,
        sites=None,
        *,
        sampling=None,
        title="",
        cmap="gray",
        **kwargs,
    ):
        super().__init__(**kwargs)
        vol = self._to_numpy3d(volume)
        self._volume = vol
        self.n0, self.n1, self.n2 = (int(s) for s in vol.shape)
        self.volume_bytes = np.ascontiguousarray(vol, dtype=np.float32).tobytes()

        if sites is None:
            sites = np.zeros((0, 5), dtype=np.float32)
        sites = np.asarray(sites, dtype=np.float32)
        if sites.ndim != 2 or sites.shape[1] < 3:
            raise ValueError(
                f"sites must be (N, >=3) [a0,a1,a2,(intensity),(sigma)], got {sites.shape}"
            )
        if sites.shape[1] < 5:
            sites = np.pad(sites, ((0, 0), (0, 5 - sites.shape[1])), constant_values=0.0)
        self._sites = sites[:, :5]
        self.num_sites = int(sites.shape[0])
        self.sites_bytes = np.ascontiguousarray(self._sites, dtype=np.float32).tobytes()

        if sampling is not None:
            self.sampling = [float(s) for s in sampling]
        self.title = title
        self.cmap = cmap
        # xy plane fixes axis 2; start in the middle.
        self.slice_index = int(vol.shape[2] // 2)

    @staticmethod
    def _to_numpy3d(volume):
        if isinstance(volume, np.ndarray):
            arr = volume
        else:
            arr = getattr(volume, "array", None)
            if arr is None:
                if hasattr(volume, "detach"):  # torch tensor
                    arr = volume.detach().cpu().numpy()
                elif hasattr(volume, "numpy"):  # Dataset
                    arr = volume.numpy()
                else:
                    arr = volume
        arr = np.asarray(arr)
        if arr.ndim != 3:
            raise ValueError(f"volume must be 3D, got shape {arr.shape}")
        return arr

    def __repr__(self) -> str:
        return (
            f"Show3DAtoms(shape=({self.n0}, {self.n1}, {self.n2}), "
            f"{self.num_sites} sites, plane={self.plane})"
        )
