"""3D atomic model analysis: calibration, neighbor finding, structure classification.

:class:`AtomicModel` holds the sites of a 3D atomic model (for example traced
from an atomic electron tomography reconstruction) in a
:class:`~quantem.core.datastructures.Vector` and provides the analysis
pipeline:

1. :meth:`AtomicModel.compute_pdf` - radial distribution function and first
   nearest-neighbor (NN) peak fit.
2. :meth:`AtomicModel.calibrate` - set the physical size of one voxel from
   the measured NN distance of a reference crystal.
3. :meth:`AtomicModel.find_neighbors` - neighbor lists, coordination and bond
   lengths.
4. :meth:`AtomicModel.match_templates` - fast polyhedral template matching
   against ``fcc``, ``hcp``, ``bcc``, ``diamond`` ... environments.
5. :meth:`AtomicModel.classify` / :meth:`AtomicModel.segment_grains` /
   :meth:`AtomicModel.compute_strain` - per-site structure, grain (sector)
   labels via orientation clustering, and local strain.

Every per-site result is stored as a named *channel* (a field of the sites
``Vector``) so it can be plotted with :meth:`AtomicModel.plot` or explored
interactively with :meth:`AtomicModel.show`.

Coordinate convention
---------------------
Fields ``x, y, z`` are the positions along array axes 0, 1, 2 of the source
volume, stored in their native (typically voxel) units.  ``sampling`` converts
them to physical units; :attr:`AtomicModel.positions` returns calibrated
coordinates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from quantem.atoms import measurements as meas
from quantem.atoms.matching import TemplateMatch, match_template
from quantem.atoms.pdf import (
    find_neighbors,
    fit_first_peak,
    nn_distance_from_lattice,
    radial_distribution,
)
from quantem.atoms.templates import PolyhedralTemplate, get_template
from quantem.atoms.visualization import PLOT_REGISTRY
from quantem.core.datastructures import Vector
from quantem.core.io.serialize import AutoSerialize

__all__ = ["AtomicModel"]

_POSITION_FIELDS = ("x", "y", "z")


class AtomicModel(AutoSerialize):
    """A 3D atomic model with per-site measurement channels.

    Use the ``from_*`` constructors rather than ``__init__``.

    Parameters
    ----------
    sites : Vector
        0-D ``Vector`` whose cell holds one row per site with at least the
        fields ``x, y, z``.
    sampling : ndarray
        ``(3,)`` physical size of one native coordinate unit along each axis.
    units : str
        Physical length unit after calibration (e.g. ``"A"``).
    name : str
        Model name.
    metadata : dict
        Free-form metadata.
    """

    _token = object()

    def __init__(
        self,
        sites: Vector,
        sampling: NDArray,
        units: str,
        name: str,
        metadata: dict[str, Any] | None = None,
        _token: object | None = None,
    ) -> None:
        if _token is not self._token:
            raise RuntimeError("Use AtomicModel.from_array() or another from_* constructor.")
        self._sites = sites
        self._sampling = np.asarray(sampling, dtype=float).reshape(3)
        self._units = str(units)
        self._name = str(name)
        self._metadata: dict[str, Any] = dict(metadata or {})
        self._pdf: dict[str, Any] | None = None
        self._nn_fit: dict[str, Any] | None = None
        self._neighbor_distances: NDArray | None = None
        self._neighbor_indices: NDArray | None = None
        self._matches: dict[str, TemplateMatch] = {}
        self._template_specs: dict[str, dict[str, Any]] = {}
        self._structure_names: list[str] = []
        self._categories: dict[str, list[str]] = {}

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_array(
        cls,
        xyz: NDArray,
        sampling: float | Sequence[float] = 1.0,
        units: str = "voxels",
        name: str | None = None,
        channels: dict[str, NDArray] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "AtomicModel":
        """Create a model from an ``(N, 3)`` coordinate array.

        Parameters
        ----------
        xyz : ndarray
            ``(N, 3)`` positions along axes 0, 1, 2.  A ``(3, N)`` array is
            transposed automatically.
        sampling : float or sequence of float
            Physical size of one coordinate unit (scalar or per axis).
        units : str
            Physical length unit (``"voxels"`` if uncalibrated).
        name : str, optional
            Model name.
        channels : dict, optional
            Extra per-site arrays stored as channels, e.g. ``{"species": ...}``.
        metadata : dict, optional
            Free-form metadata.
        """
        xyz = np.asarray(xyz, dtype=float)
        if xyz.ndim != 2 or 3 not in xyz.shape:
            raise ValueError(f"xyz must be (N, 3) or (3, N), got {xyz.shape}")
        if xyz.shape[1] != 3:
            xyz = xyz.T
        sites = Vector.from_shape(
            shape=(),
            fields=list(_POSITION_FIELDS),
            units=[units] * 3,
            name="sites",
        )
        sites[...] = np.ascontiguousarray(xyz)
        sampling_arr = np.broadcast_to(np.asarray(sampling, dtype=float), (3,)).copy()
        model = cls(
            sites=sites,
            sampling=sampling_arr,
            units=units,
            name=name or "atomic model",
            metadata=metadata,
            _token=cls._token,
        )
        for key, values in (channels or {}).items():
            model.set_channel(key, values)
        return model

    @classmethod
    def from_mat(
        cls,
        path: str | Path,
        key: str | None = None,
        sampling: float | Sequence[float] = 1.0,
        units: str = "voxels",
        name: str | None = None,
        one_based: bool = False,
    ) -> "AtomicModel":
        """Load coordinates from a MATLAB ``.mat`` file.

        Parameters
        ----------
        path : str or Path
            File path (v5/v7 or v7.3 HDF5).
        key : str, optional
            Variable name.  If omitted, the first numeric ``(N, 3)`` / ``(3, N)``
            array is used.
        sampling, units, name
            See :meth:`from_array`.
        one_based : bool
            Subtract 1 from the coordinates (MATLAB 1-based voxel indices).
        """
        path = Path(path)
        arrays: dict[str, NDArray] = {}
        try:
            import scipy.io as sio

            raw = sio.loadmat(path)
            arrays = {
                k: np.asarray(v)
                for k, v in raw.items()
                if not k.startswith("__") and isinstance(v, np.ndarray) and v.dtype.kind in "fiu"
            }
        except NotImplementedError:  # v7.3
            import h5py

            with h5py.File(path, "r") as f:

                def _collect(g, prefix=""):
                    for k, v in g.items():
                        if isinstance(v, h5py.Dataset) and v.dtype.kind in "fiu":
                            arrays[prefix + k] = np.asarray(v[()]).T
                        elif isinstance(v, h5py.Group):
                            _collect(v, prefix + k + "/")

                _collect(f)
        if key is None:
            candidates = [k for k, v in arrays.items() if v.ndim == 2 and 3 in v.shape]
            if not candidates:
                raise ValueError(
                    f"No (N, 3) coordinate array found in {path.name}: {list(arrays)}"
                )
            key = candidates[0]
        xyz = arrays[key].astype(float)
        if one_based:
            xyz = xyz - 1.0
        return cls.from_array(
            xyz,
            sampling=sampling,
            units=units,
            name=name or path.stem,
            metadata={"source": str(path), "key": key},
        )

    @classmethod
    def from_xyz(
        cls,
        path: str | Path,
        sampling: float | Sequence[float] = 1.0,
        units: str = "A",
        name: str | None = None,
    ) -> "AtomicModel":
        """Load an ``.xyz`` text file (``element x y z`` rows after a 2-line header)."""
        path = Path(path)
        lines = path.read_text().strip().splitlines()
        try:
            count = int(lines[0].split()[0])
            body = lines[2 : 2 + count]
        except (ValueError, IndexError):
            body = lines
        symbols, coords = [], []
        for line in body:
            parts = line.split()
            if len(parts) < 4:
                continue
            symbols.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        xyz = np.asarray(coords)
        names = sorted(set(symbols))
        species = np.array([names.index(s) for s in symbols], dtype=float)
        model = cls.from_array(xyz, sampling=sampling, units=units, name=name or path.stem)
        model.set_channel("species", species, categories=names)
        return model

    @classmethod
    def from_atoms(cls, atoms: Any, name: str | None = None) -> "AtomicModel":
        """Create a model from a :class:`quantem.tomography.Atoms` tracing result.

        Positions are taken in voxel units together with the traced intensity
        and Gaussian width, and ``sampling`` is copied from the source volume.
        """
        sites = atoms.sites.array
        sampling = np.asarray(atoms._sampling, dtype=float)
        xyz = (sites[:, :3] - np.asarray(atoms._origin)[None, :]) / sampling[None, :]
        model = cls.from_array(
            xyz,
            sampling=sampling,
            units=str(atoms._units[0]),
            name=name or f"{atoms._source.name} atoms",
        )
        model.set_channel("intensity", sites[:, 3])
        model.set_channel("sigma", sites[:, 4] / float(sampling.mean()))
        return model

    # ------------------------------------------------------------------ #
    # Basic properties
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        """Model name."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = str(value)

    @property
    def metadata(self) -> dict[str, Any]:
        """Free-form metadata dictionary."""
        return self._metadata

    @property
    def sites(self) -> Vector:
        """Per-site table (0-D ``Vector``); fields ``x, y, z`` plus channels."""
        return self._sites

    @property
    def num_sites(self) -> int:
        """Number of atomic sites."""
        return int(self._sites.array.shape[0])

    @property
    def sampling(self) -> NDArray:
        """``(3,)`` physical size of one native coordinate unit per axis."""
        return self._sampling

    @property
    def units(self) -> str:
        """Physical length unit of :attr:`positions`."""
        return self._units

    @property
    def positions_native(self) -> NDArray:
        """``(N, 3)`` positions in native (uncalibrated) units."""
        return np.array(self._sites.select_fields(*_POSITION_FIELDS).array, dtype=float)

    @positions_native.setter
    def positions_native(self, value: NDArray) -> None:
        value = np.asarray(value, dtype=float)
        if value.shape != (self.num_sites, 3):
            raise ValueError(f"positions must have shape {(self.num_sites, 3)}")
        self._sites.select_fields(*_POSITION_FIELDS)[...] = value
        self._invalidate()

    @property
    def positions(self) -> NDArray:
        """``(N, 3)`` calibrated positions (native * sampling)."""
        return self.positions_native * self._sampling[None, :]

    @property
    def center(self) -> NDArray:
        """``(3,)`` mean calibrated position."""
        return self.positions.mean(0)

    @property
    def channels(self) -> list[str]:
        """Names of all per-site channels (fields other than ``x, y, z``)."""
        return [f for f in self._sites.fields if f not in _POSITION_FIELDS]

    @property
    def categories(self) -> dict[str, list[str]]:
        """Label names for categorical channels, e.g. ``{"structure": ["fcc", "hcp"]}``."""
        return self._categories

    @property
    def structure_names(self) -> list[str]:
        """Template names indexed by the ``structure`` channel value."""
        return list(self._structure_names)

    @property
    def templates(self) -> dict[str, PolyhedralTemplate]:
        """Templates used in the last :meth:`match_templates` call."""
        return {
            name: PolyhedralTemplate(
                name=name,
                vectors=np.asarray(spec["vectors"]),
                shells=tuple(float(x) for x in spec["shells"]),
                shell_counts=tuple(int(x) for x in spec["shell_counts"]),
                symmetry=np.asarray(spec["symmetry"]),
            )
            for name, spec in self._template_specs.items()
        }

    @property
    def matches(self) -> dict[str, TemplateMatch]:
        """Raw per-template matching results (see :class:`TemplateMatch`)."""
        return self._matches

    @property
    def pdf(self) -> dict[str, Any] | None:
        """Result of :meth:`compute_pdf` (native units), or ``None``."""
        return self._pdf

    @property
    def nn_fit(self) -> dict[str, Any] | None:
        """First-peak fit from :meth:`compute_pdf` (native units), or ``None``."""
        return self._nn_fit

    @property
    def nn_distance(self) -> float:
        """Mean nearest-neighbor distance in native units (requires :meth:`compute_pdf`)."""
        if self._nn_fit is None:
            self.compute_pdf()
        assert self._nn_fit is not None
        return float(self._nn_fit["r_nn"])

    @property
    def bond_length(self) -> float:
        """Mean nearest-neighbor distance in calibrated units."""
        return self.nn_distance * float(self._sampling.mean())

    @property
    def neighbor_indices(self) -> NDArray:
        """``(N, K)`` neighbor indices sorted by distance (``-1`` = missing)."""
        if self._neighbor_indices is None:
            self.find_neighbors()
        assert self._neighbor_indices is not None
        return self._neighbor_indices

    @property
    def neighbor_distances(self) -> NDArray:
        """``(N, K)`` neighbor distances in native units."""
        if self._neighbor_distances is None:
            self.find_neighbors()
        assert self._neighbor_distances is not None
        return self._neighbor_distances

    def _invalidate(self) -> None:
        self._pdf = None
        self._nn_fit = None
        self._neighbor_distances = None
        self._neighbor_indices = None
        self._matches = {}

    # ------------------------------------------------------------------ #
    # Channels
    # ------------------------------------------------------------------ #
    def get_channel(self, name: str) -> NDArray:
        """Return a per-site channel as a ``(N,)`` array."""
        if name in _POSITION_FIELDS:
            return self.positions[:, _POSITION_FIELDS.index(name)]
        if name not in self._sites.fields:
            raise KeyError(f"Unknown channel {name!r}; available: {self.channels}")
        return np.array(self._sites.select_fields(name).array[:, 0], dtype=float)

    def set_channel(
        self,
        name: str,
        values: NDArray,
        units: str = "none",
        categories: Sequence[str] | None = None,
    ) -> None:
        """Add or overwrite a per-site channel.

        Parameters
        ----------
        name : str
            Channel name.
        values : ndarray
            ``(N,)`` values (cast to float; use integer codes for categories).
        units : str
            Units label.
        categories : sequence of str, optional
            Names for integer codes ``0, 1, ...``; marks the channel categorical.
        """
        if name in _POSITION_FIELDS:
            raise ValueError("Use positions_native to modify coordinates.")
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.shape[0] != self.num_sites:
            raise ValueError(f"values must have length {self.num_sites}, got {values.shape[0]}")
        if name in self._sites.fields:
            self._sites.select_fields(name)[...] = values[:, None]
        else:
            self._sites.add_fields(name, values[:, None], units)
        if categories is not None:
            self._categories[name] = [str(c) for c in categories]
        elif name in self._categories:
            del self._categories[name]

    def remove_channel(self, name: str) -> None:
        """Delete a channel."""
        self._sites.remove_fields(name)
        self._categories.pop(name, None)

    def __getitem__(self, name: str) -> NDArray:
        return self.get_channel(name)

    # ------------------------------------------------------------------ #
    # Pair distribution function and calibration
    # ------------------------------------------------------------------ #
    def compute_pdf(
        self,
        r_max: float | None = None,
        dr: float | None = None,
        sigma: float | None = None,
        fit_radius: float = 1.25,
        cutoff_sigma: float = 2.0,
        r_min: float | None = None,
    ) -> dict[str, Any]:
        """Compute the radial distribution function and fit the first NN peak.

        All radii are in native units.

        Parameters
        ----------
        r_max : float, optional
            Maximum radius; default 4x an initial NN estimate.
        dr : float, optional
            Bin width; default ``r_max / 600``.
        sigma : float, optional
            Smoothing width; default ``2 * dr``.
        fit_radius, cutoff_sigma, r_min
            See :func:`quantem.atoms.pdf.fit_first_peak`.

        Returns
        -------
        dict
            RDF arrays (``r``, ``g``, ``g_smooth``, ``counts``) plus the fit.
        """
        xyz = self.positions_native
        if r_max is None:
            d1, _ = find_neighbors(xyz, 1)
            r_max = 4.0 * float(np.median(d1))
        if dr is None:
            dr = r_max / 600.0
        pdf = radial_distribution(xyz, r_max=r_max, dr=dr, sigma=sigma)
        fit = fit_first_peak(
            pdf["r"],
            pdf["g_smooth"],
            fit_radius=fit_radius,
            cutoff_sigma=cutoff_sigma,
            r_min=r_min,
        )
        self._pdf = pdf
        self._nn_fit = fit
        return {**pdf, **{k: v for k, v in fit.items()}}

    def calibrate(
        self,
        structure: str | None = None,
        lattice_constant: float | None = None,
        nn_distance: float | None = None,
        units: str = "A",
    ) -> float:
        """Set ``sampling`` so the measured NN distance matches a reference.

        Provide either ``structure`` + ``lattice_constant`` or ``nn_distance``.

        Parameters
        ----------
        structure : str, optional
            Reference crystal (``"fcc"``, ``"bcc"``, ``"hcp"`` ...).
        lattice_constant : float, optional
            Lattice constant of the reference crystal in ``units``.
        nn_distance : float, optional
            Target NN distance in ``units``.
        units : str
            Physical unit of the reference.

        Returns
        -------
        float
            The isotropic scale (physical units per native unit).
        """
        if nn_distance is None:
            if structure is None or lattice_constant is None:
                raise ValueError("Give structure and lattice_constant, or nn_distance.")
            nn_distance = nn_distance_from_lattice(structure, lattice_constant)
        scale = float(nn_distance) / self.nn_distance
        self._sampling = np.full(3, scale)
        self._units = units
        return scale

    # ------------------------------------------------------------------ #
    # Neighbors and bonds
    # ------------------------------------------------------------------ #
    def find_neighbors(self, num_neighbors: int = 24, cutoff: float | None = None) -> None:
        """Build neighbor lists and per-site bond statistics.

        Adds channels ``num_neighbors`` (first-shell coordination),
        ``bond_mean`` and ``bond_std`` (calibrated units).

        Parameters
        ----------
        num_neighbors : int
            Neighbors stored per site; must exceed the largest template.
        cutoff : float, optional
            First-shell radial cutoff in native units.  Default: upper cutoff
            from the RDF first-peak fit.
        """
        dist, idx = find_neighbors(self.positions_native, num_neighbors)
        self._neighbor_distances = dist
        self._neighbor_indices = idx
        if cutoff is None:
            cutoff = self.first_shell_cutoff
        first = dist <= cutoff
        scale = float(self._sampling.mean())
        d_first = np.where(first, dist, np.nan) * scale
        with np.errstate(invalid="ignore"):
            self.set_channel("num_neighbors", first.sum(1))
            self.set_channel("bond_mean", np.nanmean(d_first, axis=1), self._units)
            self.set_channel("bond_std", np.nanstd(d_first, axis=1), self._units)

    @property
    def first_shell_cutoff(self) -> float:
        """Upper radial cutoff of the first shell (native units) from the RDF fit."""
        if self._nn_fit is None:
            self.compute_pdf()
        assert self._nn_fit is not None
        return float(self._nn_fit["cutoff"][1])

    def neighbor_vectors(self, normalize: bool = True) -> tuple[NDArray, NDArray]:
        """Neighbor displacement vectors.

        Parameters
        ----------
        normalize : bool
            Divide by the NN distance so bonds have length ~1.

        Returns
        -------
        dxyz, dist : ndarray
            ``(N, K, 3)`` vectors and ``(N, K)`` lengths (native units, or NN
            units when normalized).  Missing neighbors are ``inf``.
        """
        idx = self.neighbor_indices
        dist = self.neighbor_distances.copy()
        xyz = self.positions_native
        safe = np.where(idx >= 0, idx, 0)
        dxyz = xyz[safe] - xyz[:, None, :]
        dxyz[idx < 0] = np.inf
        if normalize:
            r_nn = self.nn_distance
            return dxyz / r_nn, dist / r_nn
        return dxyz, dist

    def bond_angles(self) -> NDArray:
        """All first-shell bond angles per site, ``(N, K(K-1)/2)`` degrees with ``nan`` padding."""
        dxyz, dist = self.neighbor_vectors(normalize=False)
        valid = dist <= self.first_shell_cutoff
        dxyz = np.where(np.isfinite(dxyz), dxyz, 0.0)
        return meas.bond_angles(dxyz, valid)

    # ------------------------------------------------------------------ #
    # Template matching and classification
    # ------------------------------------------------------------------ #
    def match_templates(
        self,
        templates: Sequence[str | PolyhedralTemplate] = ("fcc", "hcp"),
        score_radius: float = 0.5,
        cutoff_factor: float = 1.15,
        angle_tolerance: float = 30.0,
        num_refine: int = 2,
        chunk_size: int = 512,
        device: str | None = None,
        progress: bool = True,
    ) -> dict[str, TemplateMatch]:
        """Match polyhedral templates to every site.

        Adds channels ``score_<name>``, ``rmsd_<name>`` and ``matched_<name>``
        for each template, then calls :meth:`classify` with default settings.

        Parameters
        ----------
        templates : sequence of str or PolyhedralTemplate
            Template names (see :data:`quantem.atoms.TEMPLATE_NAMES`) or objects.
        score_radius : float
            Matching radius in NN units; see :func:`quantem.atoms.matching.match_template`.
        cutoff_factor : float
            Neighbors farther than ``cutoff_factor * template.max_radius`` (NN
            units) are ignored for that template.
        angle_tolerance, num_refine, chunk_size, device, progress
            Forwarded to :func:`quantem.atoms.matching.match_template`.

        Returns
        -------
        dict
            ``{name: TemplateMatch}``.
        """
        dxyz, dist = self.neighbor_vectors(normalize=True)
        dxyz = np.where(np.isfinite(dxyz), dxyz, 1e3)
        self._matches = {}
        self._template_specs = {}
        for item in templates:
            template = get_template(item) if isinstance(item, str) else item
            name = template.name
            valid = dist <= cutoff_factor * template.max_radius
            if valid.shape[1] < template.num_neighbors:
                raise ValueError(
                    f"Template {name!r} has {template.num_neighbors} neighbors but only "
                    f"{valid.shape[1]} are stored; call find_neighbors(num_neighbors=...)."
                )
            result = match_template(
                dxyz,
                valid,
                template,
                score_radius=score_radius,
                angle_tolerance=angle_tolerance,
                num_refine=num_refine,
                chunk_size=chunk_size,
                device=device,
                progress=progress,
            )
            self._matches[name] = result
            self._template_specs[name] = {
                "vectors": np.asarray(template.vectors),
                "shells": list(template.shells),
                "shell_counts": list(template.shell_counts),
                "symmetry": np.asarray(template.symmetry),
            }
            self.set_channel(f"score_{name}", result["score"])
            self.set_channel(f"rmsd_{name}", result["rmsd"])
            self.set_channel(f"matched_{name}", result["num_matched"])
        self.classify()
        return self._matches

    def classify(
        self, threshold: float = 0.5, smooth: bool = False, use_strained: bool = False
    ) -> NDArray:
        """Assign each site to its best-scoring template.

        Adds channels ``structure`` (categorical: template index, ``-1`` for
        unclassified), ``score_max`` and, when exactly two templates were
        matched, ``score_diff`` (first minus second).

        Parameters
        ----------
        threshold : float
            Minimum score for a site to be classified.
        smooth : bool
            Average each site's scores with its first-shell neighbors before
            deciding (more robust for noisy models).
        use_strained : bool
            Use the affine-fit scores (``score_strained``) instead.

        Returns
        -------
        ndarray
            ``(N,)`` structure codes.
        """
        if not self._matches:
            raise RuntimeError("Call match_templates() first.")
        names = list(self._matches)
        key = "score_strained" if use_strained else "score"
        scores = np.stack([self._matches[n][key] for n in names], axis=1)
        if smooth:
            idx = self.neighbor_indices
            first = self.neighbor_distances <= self.first_shell_cutoff
            safe = np.where(idx >= 0, idx, 0)
            nb = scores[safe] * first[..., None]
            scores = (scores + nb.sum(1)) / (1.0 + first.sum(1))[:, None]
        best = scores.argmax(1)
        score_max = scores.max(1)
        structure = np.where(score_max >= threshold, best, -1)
        self._structure_names = names
        self.set_channel("structure", structure, categories=names)
        self.set_channel("score_max", score_max)
        if len(names) == 2:
            self.set_channel("score_diff", scores[:, 0] - scores[:, 1])
        return structure

    def rotations(self, template: str | None = None) -> NDArray:
        """``(N, 3, 3)`` fitted orientations (lab <- crystal) for a template.

        With ``template=None`` the rotation from each site's classified
        structure is returned (identity for unclassified sites).
        """
        if template is not None:
            return self._matches[template]["rotation"]
        structure = self.get_channel("structure").astype(int)
        out = np.tile(np.eye(3), (self.num_sites, 1, 1))
        for i, name in enumerate(self._structure_names):
            sel = structure == i
            out[sel] = self._matches[name]["rotation"][sel]
        return out

    def segment_grains(
        self,
        structure: str = "fcc",
        angle_threshold: float = 5.0,
        min_size: int = 20,
        min_score: float | None = None,
    ) -> NDArray:
        """Cluster sites of one structure into grains by local orientation.

        Neighboring sites of the given structure whose disorientation is below
        ``angle_threshold`` are connected; connected components become grains.
        Adds channels ``grain`` (``-1`` = none) and ``misorientation`` (largest
        disorientation to any first-shell neighbor of the same structure).

        Parameters
        ----------
        structure : str
            Template name to segment (e.g. ``"fcc"``).
        angle_threshold : float
            Maximum disorientation (degrees) inside a grain.
        min_size : int
            Grains with fewer sites are discarded.
        min_score : float, optional
            Only sites with ``score_<structure>`` above this take part;
            default: sites classified as ``structure``.

        Returns
        -------
        ndarray
            ``(N,)`` grain labels sorted by decreasing size.
        """
        if structure not in self._matches:
            raise KeyError(f"No match for {structure!r}; run match_templates first.")
        template = self.templates[structure]
        rot = self._matches[structure]["rotation"]
        idx = self.neighbor_indices
        first = self.neighbor_distances <= self.first_shell_cutoff
        if min_score is None:
            member = self.get_channel("structure").astype(int) == self._structure_names.index(
                structure
            )
        else:
            member = self._matches[structure]["score"] >= min_score
        ang = meas.misorientation(rot, idx, template.symmetry)
        safe = np.where(idx >= 0, idx, 0)
        same = member[:, None] & member[safe] & first & (idx >= 0)
        edge = same & (ang < angle_threshold)
        labels = meas.segment_grains(idx, edge, min_size=min_size)
        labels[~member] = -1
        worst = np.where(same, ang, -1.0).max(axis=1).clip(0.0)
        num_grains = int(labels.max()) + 1 if labels.size else 0
        self.set_channel("grain", labels, categories=[str(i) for i in range(num_grains)])
        self.set_channel("misorientation", worst, "deg")
        return labels

    def compute_strain(
        self, template: str | None = None, frame: str = "lab"
    ) -> dict[str, NDArray]:
        """Local strain from the affine template fit.

        Adds channels ``strain_xx, strain_yy, strain_zz, strain_xy, strain_xz,
        strain_yz, strain_dilation, strain_equivalent``.  Strain is relative to
        the mean NN distance of the model.

        Parameters
        ----------
        template : str, optional
            Template whose fit to use; default: each site's classified structure.
        frame : {"lab", "crystal"}
            Frame of the strain tensor.
        """
        if template is not None:
            f = self._matches[template]["deformation"]
            r = self._matches[template]["rotation"]
        else:
            structure = self.get_channel("structure").astype(int)
            f = np.tile(np.eye(3), (self.num_sites, 1, 1))
            r = f.copy()
            for i, name in enumerate(self._structure_names):
                sel = structure == i
                f[sel] = self._matches[name]["deformation"][sel]
                r[sel] = self._matches[name]["rotation"][sel]
        strain = meas.strain_from_deformation(f, r, frame=frame)
        for key in ("e_xx", "e_yy", "e_zz", "e_xy", "e_xz", "e_yz"):
            self.set_channel("strain_" + key[2:], strain[key])
        self.set_channel("strain_dilation", strain["dilation"])
        self.set_channel("strain_equivalent", strain["equivalent"])
        return strain

    # ------------------------------------------------------------------ #
    # Other measurements
    # ------------------------------------------------------------------ #
    def surface_distance(self) -> NDArray:
        """Distance of each site to the convex hull (calibrated units); channel ``surface_distance``."""
        d = meas.convex_hull_distance(self.positions)
        self.set_channel("surface_distance", d, self._units)
        return d

    def sample_volume(self, volume: Any, radius: float = 1.5, name: str = "intensity") -> NDArray:
        """Mean reconstruction intensity around each site; stored as a channel.

        Parameters
        ----------
        volume : ndarray or Dataset3d
            Source volume indexed like the native coordinates.
        radius : float
            Sphere radius in voxels.
        name : str
            Channel name.
        """
        arr = getattr(volume, "array", volume)
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        values = meas.sample_volume(np.asarray(arr), self.positions_native, radius=radius)
        self.set_channel(name, values)
        return values

    def classify_species(
        self,
        channel: str = "intensity",
        num_species: int = 2,
        names: Sequence[str] | None = None,
    ) -> NDArray:
        """Split a channel (e.g. intensity) into species with 1D k-means; channel ``species``."""
        labels, centers = meas.kmeans_1d(self.get_channel(channel), num_species)
        if names is None:
            names = [f"species_{i}" for i in range(num_species)]
        self.set_channel("species", labels, categories=names)
        self._metadata["species_centers"] = centers.tolist()
        return labels

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #
    def rotate(self, rotation: NDArray, about_center: bool = True) -> None:
        """Rotate all positions in place with a ``(3, 3)`` matrix (``x' = R x``)."""
        rotation = np.asarray(rotation, dtype=float)
        xyz = self.positions_native
        c = xyz.mean(0) if about_center else np.zeros(3)
        self.positions_native = (xyz - c) @ rotation.T + c

    def select(self, mask: NDArray) -> "AtomicModel":
        """Return a new model containing only the sites where ``mask`` is True."""
        mask = np.asarray(mask, dtype=bool)
        table = self._sites.array[mask]
        sites = Vector.from_shape(
            shape=(), fields=list(self._sites.fields), units=list(self._sites.units), name="sites"
        )
        sites[...] = np.ascontiguousarray(table)
        model = AtomicModel(
            sites=sites,
            sampling=self._sampling.copy(),
            units=self._units,
            name=self._name,
            metadata=dict(self._metadata),
            _token=self._token,
        )
        model._categories = dict(self._categories)
        model._structure_names = list(self._structure_names)
        return model

    # ------------------------------------------------------------------ #
    # Visualization
    # ------------------------------------------------------------------ #
    def plot(self, kind: str = "slab", show_docstring: bool = False, **kwargs):
        """Static matplotlib plots; see :mod:`quantem.atoms.visualization`.

        Parameters
        ----------
        kind : str
            One of ``"pdf"``, ``"histogram"``, ``"slab"``, ``"slices"``,
            ``"template"``.
        show_docstring : bool
            Print the plot function's docstring instead of plotting.
        **kwargs
            Forwarded to the plot function.
        """
        if kind not in PLOT_REGISTRY:
            raise ValueError(f"Unknown plot kind {kind!r}; choose from {list(PLOT_REGISTRY)}")
        fn = PLOT_REGISTRY[kind]
        if show_docstring:
            print(fn.__doc__)
            return None
        return fn(self, **kwargs)

    def show(self, **kwargs):
        """Open the interactive 3D viewer (:class:`quantem.atoms.ShowAtoms3D`)."""
        from quantem.atoms.show_atoms import ShowAtoms3D

        return ShowAtoms3D(self, **kwargs)

    def __repr__(self) -> str:
        return (
            f"AtomicModel(name={self._name!r}, num_sites={self.num_sites}, "
            f"units={self._units!r}, channels={self.channels})"
        )
