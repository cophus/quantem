"""Orientation and phase mapping over one or more candidate crystals.

CrystalMap is the standard entry point for ACOM. It owns one
:class:`~quantem.diffraction.orientation.OrientationMap` per candidate
crystal, fans the matching and refinement stages out over all of them, and
holds the :class:`~quantem.diffraction.phase.PhaseMap` that decides which
crystal sits at each probe position::

    cm = CrystalMap.from_vectors(peaks, [Cu_metal, Cu2O], energy_ev=300e3)
    cm.build_plan(**plan_params)
    cm.match_orientations(**match_params)
    cm.refine_orientations(**refine_params)
    cm.fit()
    cm.plot_phase()
    cm.plot_orientation()

The individual maps stay available for anything asymmetric or per-crystal --
``cm["Cu metal"]``, ``cm[0]`` or ``cm.orientation_maps`` -- and every method
on OrientationMap works there exactly as before.
"""

from __future__ import annotations

import numpy as np

from quantem.core.io.serialize import AutoSerialize
from quantem.diffraction.crystal import Crystal
from quantem.diffraction.orientation import OrientationMap
from quantem.diffraction.phase import PhaseMap


class CrystalMap(AutoSerialize):
    """Per-position crystal orientation and phase over a scan.

    Parameters
    ----------
    orientation_maps : list of OrientationMap
        One matched (or unmatched) map per candidate crystal.

    Attributes
    ----------
    orientation_maps : list of OrientationMap
        The per-crystal maps, in the order they were given.
    names : list of str
        Crystal names, used for indexing and plot labels.
    phases : PhaseMap or None
        The phase decision, populated by :meth:`fit`.
    """

    _token = object()

    def __init__(self, orientation_maps: list[OrientationMap], _token=None):
        if _token is not self._token:
            raise RuntimeError(
                "Use CrystalMap.from_vectors() or CrystalMap.from_orientation_maps()."
            )
        if len(orientation_maps) == 0:
            raise ValueError("CrystalMap needs at least one crystal.")
        names = [om.crystal.name for om in orientation_maps]
        if len(set(names)) != len(names):
            raise ValueError(
                f"crystal names must be unique for indexing by name, got {names}. "
                "Set Crystal(name=...) to tell them apart."
            )
        shapes = {tuple(om.peaks.shape[:2]) for om in orientation_maps}
        if len(shapes) != 1:
            raise ValueError(f"all crystals must share one scan shape, got {shapes}")
        self.orientation_maps = orientation_maps
        self.phases: PhaseMap | None = None
        self.metadata: dict = {}

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def from_vectors(
        cls,
        peaks,
        crystals: Crystal | list[Crystal],
        energy_ev: float = 300e3,
        precession_deg: float = 0.0,
        semiconv_mrad: float = 0.0,
    ) -> "CrystalMap":
        """Build one OrientationMap per crystal from a shared peak table.

        Parameters
        ----------
        peaks : Vector
            Calibrated Bragg peaks, shared by every crystal.
        crystals : Crystal or list of Crystal
            Candidate phases. A single Crystal is accepted, so single-phase
            work uses the same entry point.
        energy_ev, precession_deg, semiconv_mrad
            Passed to :meth:`OrientationMap.from_vectors`.
        """
        xtls = list(crystals) if isinstance(crystals, (list, tuple)) else [crystals]
        oms = [
            OrientationMap.from_vectors(
                peaks,
                xtl,
                energy_ev=energy_ev,
                precession_deg=precession_deg,
                semiconv_mrad=semiconv_mrad,
            )
            for xtl in xtls
        ]
        return cls(oms, _token=cls._token)

    @classmethod
    def from_orientation_maps(cls, orientation_maps: list[OrientationMap]) -> "CrystalMap":
        """Wrap maps that were built and matched by hand."""
        return cls(list(orientation_maps), _token=cls._token)

    # ------------------------------------------------------------------
    # access
    # ------------------------------------------------------------------

    @property
    def names(self) -> list[str]:
        return [om.crystal.name for om in self.orientation_maps]

    @property
    def peaks(self):
        return self.orientation_maps[0].peaks

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.orientation_maps[0].peaks.shape[:2])

    def __len__(self) -> int:
        return len(self.orientation_maps)

    def __iter__(self):
        return iter(self.orientation_maps)

    def __getitem__(self, key) -> OrientationMap:
        """`cm[0]` or `cm["Cu metal"]` -> the OrientationMap of that crystal."""
        if isinstance(key, str):
            try:
                return self.orientation_maps[self.names.index(key)]
            except ValueError:
                raise KeyError(f"no crystal named {key!r}; have {self.names}") from None
        return self.orientation_maps[key]

    def __repr__(self) -> str:
        R, C = self.shape
        stage = "unmatched"
        if self.orientation_maps[0].quats is not None:
            stage = "matched"
            if "refine" in self.orientation_maps[0].metadata:
                stage = "refined"
        if self.phases is not None and self.phases.phase_index is not None:
            stage += ", phase fit"
        return "CrystalMap(%d x %d, %s, [%s])" % (R, C, stage, ", ".join(self.names))

    # ------------------------------------------------------------------
    # staged workflow, fanned out over the crystals
    # ------------------------------------------------------------------

    def build_plan(self, overrides: dict | None = None, **kwargs) -> "CrystalMap":
        """Build the correlation plan for every crystal.

        Parameters
        ----------
        overrides : dict, optional
            Per-crystal keyword overrides, keyed by crystal name, e.g.
            ``overrides={"Cu2O": dict(angle_step_zone_axis_deg=2.0)}``.
        **kwargs
            Passed to :meth:`OrientationMap.build_plan` for every crystal.
        """
        return self._fanout("build_plan", overrides, **kwargs)

    def match_orientations(self, overrides: dict | None = None, **kwargs) -> "CrystalMap":
        """Match every crystal against the measured peaks.

        Each crystal is correlated against its own plan, so the scores are
        comparable: the library slices are unit vectors and the measured
        polar image is divided by its norm, making the correlation a cosine
        similarity in [0, 1]. With `num_matches` above 1 each further match
        is fitted to what the earlier ones leave unexplained, so a probe
        straddling two grains indexes both. Every match is still scored
        against the pattern as measured, so comparing the matches tells the
        two cases apart: two grains score alike, while a spurious second
        match on a single grain scores well below the first.

        Parameters
        ----------
        overrides : dict, optional
            Per-crystal keyword overrides, keyed by crystal name.
        **kwargs
            Passed to :meth:`OrientationMap.match_orientations` for every
            crystal. The ones usually set are `num_matches`,
            `min_number_peaks` and `positions`, the last restricting the
            match to a few probe positions for a staged test run.

        Returns
        -------
        CrystalMap
            Self, so stages chain.

        Notes
        -----
        The correlation saturates on sparse patterns: a position carrying
        the direct beam and two noise peaks scores about as well as a real
        grain, because some library orientation almost always has a
        reflection at that radius and angle. Judge a match by
        :meth:`signal_confidence` or the peak count, not by the correlation
        alone.
        """
        return self._fanout("match_orientations", overrides, **kwargs)

    def refine_orientations(self, overrides: dict | None = None, **kwargs) -> "CrystalMap":
        """Refine every crystal off the library grid.

        Least squares on the paired peak positions removes the quantization
        of the plan: the in-plane rotation comes from the pairing, the zone
        axis tilt from the intensity envelope. A second pass rescues
        positions whose answer disagrees with all of their neighbours.

        Parameters
        ----------
        overrides : dict, optional
            Per-crystal keyword overrides, keyed by crystal name.
        **kwargs
            Passed to :meth:`OrientationMap.refine_orientations` for every
            crystal, commonly `num_iterations` and `zone_search_deg`.

        Returns
        -------
        CrystalMap
            Self, so stages chain.
        """
        return self._fanout("refine_orientations", overrides, **kwargs)

    def _fanout(self, method: str, overrides: dict | None, **kwargs) -> "CrystalMap":
        overrides = overrides or {}
        unknown = set(overrides) - set(self.names)
        if unknown:
            raise KeyError(f"overrides name unknown crystals {sorted(unknown)}; have {self.names}")
        for om in self.orientation_maps:
            kw = {**kwargs, **overrides.get(om.crystal.name, {})}
            getattr(om, method)(**kw)
        return self

    def fit(self, **kwargs) -> "CrystalMap":
        """Decide the phase at every position; see :meth:`PhaseMap.fit`.

        With a single crystal there is nothing to choose between, but the fit
        still runs: it applies the null hypothesis, so positions with no
        diffracted signal come out unindexed and the maps below fade them.
        """
        self.phases = PhaseMap.from_orientation_maps(self.orientation_maps)
        self.phases.fit(**kwargs)
        return self

    # ------------------------------------------------------------------
    # derived quantities
    # ------------------------------------------------------------------

    def _require_fit(self, what: str) -> PhaseMap:
        if self.phases is None or self.phases.phase_index is None:
            raise ValueError(f"run fit() before {what}.")
        return self.phases

    @property
    def phase_index(self) -> np.ndarray:
        """Winning crystal at each probe position.

        Returns
        -------
        np.ndarray
            ``(scan_row, scan_col)`` index into :attr:`names`, or -1 where
            the null hypothesis in :meth:`fit` found too little diffracted
            signal to name a crystal.
        """
        return self._require_fit("phase_index").phase_index.numpy()

    def signal_confidence(self, signal_range="auto") -> np.ndarray:
        """Confidence in [0, 1] that a crystal is present, from the data alone.

        The measured intensity beyond the direct beam, scaled to [0, 1].
        Vacuum and amorphous support diffract nothing, so they score zero
        however well some orientation happens to correlate -- which the
        correlation itself cannot tell you, since it saturates on sparse
        patterns.

        Parameters
        ----------
        signal_range : tuple or "auto", default="auto"
            Diffracted intensity mapped to 0 ... 1. "auto" spans zero to the
            95th percentile over the indexed positions.

        Returns
        -------
        np.ndarray
            ``(scan_row, scan_col)`` confidence in [0, 1].
        """
        return self._require_fit("signal_confidence()").signal_confidence(signal_range)

    def mask(self, phase=None, signal_range="auto") -> np.ndarray:
        """Display mask in [0, 1] for one crystal, or for all indexed positions.

        The phase decision times the diffracted-signal confidence: positions
        of another crystal, and positions with nothing there, are zero.

        Parameters
        ----------
        phase : int or str, optional
            Crystal index or name. None (default) keeps every indexed
            position, whichever crystal won.
        signal_range : tuple or "auto"
            Passed to :meth:`signal_confidence`.
        """
        conf = self.signal_confidence(signal_range)
        if phase is None:
            return conf
        i = self.names.index(phase) if isinstance(phase, str) else int(phase)
        return (self.phase_index == i) * conf

    def phase_fractions(self) -> dict[str, float]:
        """Fraction of the scan won by each crystal, plus the unindexed share."""
        ph = self.phase_index
        out = {"unindexed": float((ph == -1).mean())}
        for i, n in enumerate(self.names):
            out[n] = float((ph == i).mean())
        return out

    # ------------------------------------------------------------------
    # plotting
    # ------------------------------------------------------------------

    def plot_phase(self, **kwargs):
        """Map of which crystal won at each probe position.

        Color gives the crystal, brightness gives the evidence. Positions
        the null hypothesis left unindexed in :meth:`fit` -- vacuum,
        amorphous support, anything that diffracts nothing -- are black.

        Parameters
        ----------
        shade_by : {"signal", "reliability", "none"}, default="signal"
            What the brightness means. "signal" fades by the measured
            diffracted intensity, so the map shows where crystals are.
            "reliability" uses the cost gap to the best model without the
            winning crystal, which answers which phase rather than whether
            there is one.
        shade_range : tuple or "auto", default="auto"
            Values mapped to black ... full color.
        phase_colors : np.ndarray, optional
            One RGB color per crystal.
        scalebar : dict, "auto" or None, default="auto"
            Real-space scale bar; "auto" takes the scan step carried by the
            peaks.
        **kwargs
            Further arguments of :meth:`PhaseMap.plot_phase`.

        Returns
        -------
        tuple
            ``(fig, ax)``.

        Raises
        ------
        ValueError
            If :meth:`fit` has not been run.
        """
        return self._require_fit("plot_phase()").plot_phase(**kwargs)

    def plot_orientation(self, direction=("z", "r"), phase=None, mask=None, **kwargs):
        """Inverse pole figure maps of every crystal, masked by the phase decision.

        Parameters
        ----------
        direction : str or sequence of str, default=("z", "r")
            Out-of-plane, in-plane, or both.
        phase : int or str, optional
            Restrict to one crystal. None (default) plots all of them.
        mask : np.ndarray, optional
            Overrides the automatic phase-and-signal mask.
        **kwargs
            Passed to :meth:`OrientationMap.plot_orientation`.

        Returns
        -------
        list of tuple
            One ``(fig, ax)`` per crystal and direction.
        """
        dirs = [direction] if isinstance(direction, str) else list(direction)
        out = []
        for i in self._phase_indices(phase):
            m = mask if mask is not None else self.mask(i)
            for d in dirs:
                out.append(
                    self.orientation_maps[i].plot_orientation(direction=d, mask=m, **kwargs)
                )
        return out

    def plot_pole_figure(self, pole=(0, 0, 1), phase=None, mask=None, **kwargs):
        """Stereographic pole figure of each crystal, masked by the phase decision.

        Parameters
        ----------
        pole : tuple of int, default=(0, 0, 1)
            Crystal direction plotted, in Miller indices.
        phase : int or str, optional
            Restrict to one crystal. None (default) plots all of them.
        mask : np.ndarray, optional
            Overrides the automatic phase-and-signal mask.
        **kwargs
            Passed to :meth:`OrientationMap.plot_pole_figure`, e.g.
            `color_by`, `int_range` and `overlay`.

        Returns
        -------
        list of tuple
            One ``(fig, ax)`` per crystal.
        """
        out = []
        for i in self._phase_indices(phase):
            m = mask if mask is not None else self.mask(i)
            out.append(self.orientation_maps[i].plot_pole_figure(pole=pole, mask=m, **kwargs))
        return out

    def plot_matches(self, positions, phase=None, **kwargs):
        """Matched patterns at a few probe positions, over the measured peaks.

        One panel per candidate, with the measured peaks as gray disks and
        the simulated pattern as colored markers, both sized by intensity.
        Passing a `dataset` puts the recorded pattern behind them instead;
        the pixel size and the fitted origins then come from the peaks, which
        carry them from the dataset through the calibration, so neither needs
        passing. A position matched by nothing is drawn with its peaks alone
        and labelled "no match".

        Parameters
        ----------
        positions : list of tuple of int
            ``(row, col)`` probe positions, one panel row each.
        phase : int or str, optional
            Restrict to one crystal. None (default) shows all of them.
        matches : tuple of int, default=(0, 1)
            Which matches of each crystal to draw. With `num_matches` of 2,
            (0, 1) shows the best and the residual match side by side, which
            is how a probe straddling two grains shows itself.
        dataset : Dataset4dstem, optional
            Show the recorded diffraction pattern behind the overlay.
        measured_scale, measured_power : float, optional
            Size and intensity compression of the gray measured peaks.
        transpose_plots : bool, default=False
            Panel layout only: rows are positions unless this is True.
        **kwargs
            Further arguments of
            :func:`~quantem.diffraction.orientation_visualization.plot_pattern_matches`.

        Returns
        -------
        tuple
            ``(fig, axs)``.
        """
        from quantem.diffraction.orientation_visualization import plot_pattern_matches

        oms = [self.orientation_maps[i] for i in self._phase_indices(phase)]
        md = self.peaks.metadata or {}
        if kwargs.get("dataset") is not None:
            if md.get("pixel_size") is not None:
                kwargs.setdefault("pixel_size", float(md["pixel_size"]))
            if md.get("origins") is not None:
                kwargs.setdefault("origins", np.asarray(md["origins"]))
        return plot_pattern_matches(oms, positions=positions, **kwargs)

    def plot_correlation(self, **kwargs):
        """Correlation and reliability of every crystal, one panel each.

        The top row is the best correlation of each crystal, the bottom row
        its reliability. Read them with care: the correlation is a cosine
        similarity, so it saturates on patterns carrying only a few peaks
        and stays high on the substrate, and the reliability compares
        crystals rather than testing whether one is there at all. Use
        :meth:`plot_phase` or :meth:`signal_confidence` for that.

        Parameters
        ----------
        mask : bool, default=False
            If True, multiply every panel by :meth:`signal_confidence`, so
            positions with no diffracted signal go to zero.
        shared_scale : bool, default=True
            Put every crystal on one scale, so the panels can be compared
            directly. Each row keeps its own range, since correlation and
            reliability are different quantities. False lets each panel
            autoscale, which shows the structure within a weak crystal at
            the cost of comparability. Passing `norm` overrides both.
        **kwargs
            Passed to :func:`~quantem.core.visualization.show_2d`.

        Returns
        -------
        tuple
            ``(fig, axs)``.
        """
        from quantem.core.visualization import show_2d

        mask = kwargs.pop("mask", False)
        shared_scale = kwargs.pop("shared_scale", True)
        corr = [om.corr[..., 0].numpy() for om in self.orientation_maps]
        rel = [om.reliability.numpy() for om in self.orientation_maps]
        if mask:
            conf = self.signal_confidence()
            corr = [c * conf for c in corr]
            rel = [r * conf for r in rel]
        if shared_scale and "norm" not in kwargs:
            # one scale per row, so the crystals are directly comparable;
            # correlation and reliability keep their own ranges
            kwargs["norm"] = [
                [
                    {
                        "interval_type": "manual",
                        "vmin": float(min(np.nanmin(a) for a in row)),
                        "vmax": float(max(np.nanmax(a) for a in row)),
                    }
                ]
                * len(row)
                for row in (corr, rel)
            ]
        kwargs.setdefault("cbar", True)
        kwargs.setdefault(
            "title",
            [
                [f"{n} correlation" for n in self.names],
                [f"{n} reliability" for n in self.names],
            ],
        )
        sb = self.orientation_maps[0].scan_scalebar
        if sb is not None:
            kwargs.setdefault("scalebar", [[sb] + [False] * (len(corr) - 1), [False] * len(corr)])
        return show_2d([corr, rel], **kwargs)

    def _phase_indices(self, phase) -> list[int]:
        if phase is None:
            return list(range(len(self.orientation_maps)))
        i = self.names.index(phase) if isinstance(phase, str) else int(phase)
        return [i]

    # ------------------------------------------------------------------
    # checkpointing
    # ------------------------------------------------------------------

    def save(self, path, mode: str = "w", include_plan: bool = False, **kwargs):
        """Save the whole analysis to one file.

        Parameters
        ----------
        include_plan : bool, default=False
            The correlation plan dominates the file size and is rebuilt in
            seconds by :meth:`build_plan`, so it is dropped by default. Pass
            True to keep it and reload a map ready to match again.
        """
        if include_plan:
            return AutoSerialize.save(self, path, mode=mode, **kwargs)
        stash = [(om, om.plan_fft) for om in self.orientation_maps]
        try:
            for om, _ in stash:
                om.plan_fft = None
            return AutoSerialize.save(self, path, mode=mode, **kwargs)
        finally:
            for om, plan in stash:
                om.plan_fft = plan
