"""
DiffractionSim: interactive crystal and diffraction simulation for teaching.

Left panel: the unit cell in 3D, rotated by dragging (mouse or touch) or by
buttons about the screen axes. Right panel: the diffraction pattern of the
same orientation, updated live: nanobeam (kinematical markers, or Bloch wave
intensities that follow the thickness slider), CBED disks, or Kossel /
LACBED lines. Every simulation runs in the browser, so the widget can be
exported as a single HTML file and embedded in a web page without Python.
"""

from __future__ import annotations

import base64
import json
import pathlib
import re

import anywidget
import numpy as np
import torch
import traitlets

_STATIC = pathlib.Path(__file__).parent / "static" / "diffsim.js"

# ASE bulk structures offered in the crystal menu
PRESETS: dict[str, dict] = {
    "Si (diamond cubic)": dict(name="Si", crystalstructure="diamond", a=5.431, cubic=True),
    "Ge (diamond cubic)": dict(name="Ge", crystalstructure="diamond", a=5.658, cubic=True),
    "Al (fcc)": dict(name="Al", crystalstructure="fcc", a=4.05, cubic=True),
    "Cu (fcc)": dict(name="Cu", crystalstructure="fcc", a=3.615, cubic=True),
    "Au (fcc)": dict(name="Au", crystalstructure="fcc", a=4.078, cubic=True),
    "Fe (bcc)": dict(name="Fe", crystalstructure="bcc", a=2.866, cubic=True),
    "W (bcc)": dict(name="W", crystalstructure="bcc", a=3.165, cubic=True),
    "Ti (hcp)": dict(name="Ti", crystalstructure="hcp", a=2.9505, c=4.6855),
    "Mg (hcp)": dict(name="Mg", crystalstructure="hcp", a=3.209, c=5.211),
    "GaAs (zincblende)": dict(name="GaAs", crystalstructure="zincblende", a=5.653, cubic=True),
    "NaCl (rocksalt)": dict(name="NaCl", crystalstructure="rocksalt", a=5.64, cubic=True),
    "SrTiO3 (perovskite)": dict(
        symbols=["Sr", "Ti", "O"],
        basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
        spacegroup=221,
        cellpar=[3.905, 3.905, 3.905, 90, 90, 90],
    ),
    "Al2O3 (corundum)": dict(
        symbols=["Al", "O"],
        basis=[(0, 0, 0.3523), (0.3064, 0, 0.25)],
        spacegroup=167,
        cellpar=[4.7602, 4.7602, 12.9933, 90, 90, 120],
    ),
    "SiO2 (alpha quartz)": dict(
        symbols=["Si", "O"],
        basis=[(0.4697, 0, 0), (0.4135, 0.2669, 0.1191)],
        spacegroup=154,
        cellpar=[4.9134, 4.9134, 5.4052, 90, 90, 120],
    ),
    "alpha-Mn (58 atoms)": dict(
        symbols=["Mn", "Mn", "Mn", "Mn"],
        basis=[
            (0, 0, 0),
            (0.3175, 0.3175, 0.3175),
            (0.3570, 0.3570, 0.0348),
            (0.0896, 0.0896, 0.2820),
        ],
        spacegroup=217,
        cellpar=[8.911, 8.911, 8.911, 90, 90, 90],
    ),
}


def _preset_atoms(key: str):
    from ase.build import bulk

    spec = PRESETS[key]
    if "spacegroup" in spec:
        from ase.spacegroup import crystal as ase_crystal

        return ase_crystal(
            spec["symbols"],
            basis=spec["basis"],
            spacegroup=spec["spacegroup"],
            cellpar=spec["cellpar"],
        )
    return bulk(**spec)


def _f32_b64(a) -> str:
    return base64.b64encode(
        np.ascontiguousarray(np.asarray(a, dtype=np.float32)).tobytes()
    ).decode()


def prepare_crystal(crystal, energy_ev: float, k_max: float) -> dict:
    """Everything the browser needs to draw the cell and simulate patterns.

    One reflection list: the points of the primitive reciprocal lattice
    within k_max (glide-forbidden reflections such as Si 200 and 222 are
    present with zero kinematical intensity and fill by multiple scattering
    in the Bloch calculation). Per reflection the kinematical |F_g|^2
    (Lobato) and the Bloch coupling U_g (absorptive Weickenmeier-Kohl
    factors at this energy when available, else gamma F_g / pi). The browser
    builds the structure matrix from the same list, so couplings between
    beams further apart than k_max are taken as zero; at the default
    k_max = 4 1/A those factors are below 2% of U_000 for every element.
    Indices travel as int16, g is rebuilt from the reciprocal cell.
    """
    from ase.data import chemical_symbols, covalent_radii
    from ase.data.colors import jmol_colors

    from quantem.core.utils.utils import electron_wavelength_angstrom
    from quantem.diffraction import bloch

    if crystal.g_vec is None or crystal.k_max is None or crystal.k_max < k_max:
        crystal.calculate_structure_factors(k_max=k_max)
    have_dyn = (
        getattr(crystal, "U_dyn", None) is not None
        and abs(getattr(crystal, "dyn_energy_ev", -1) - energy_ev) < 1
        and getattr(crystal, "dyn_k_max", 0) >= k_max
    )
    if not have_dyn:
        try:
            crystal.calculate_dynamical_structure_factors(energy_ev=energy_ev, k_max=k_max)
            have_dyn = True
        except Exception:
            have_dyn = False
    hkl_u, g_u = bloch._beam_universe(crystal)
    keep = torch.linalg.norm(g_u, dim=1) <= k_max
    hkl = hkl_u[keep]
    gamma_rel = bloch.relativistic_gamma(energy_ev)
    U_g, u0_imag = _coupling_vector(crystal, hkl, gamma_rel)
    # kinematical |F|^2 (Lobato) on the same list
    lut = {tuple(h): i for i, h in enumerate(crystal.hkl.tolist())}
    F2 = torch.zeros(hkl.shape[0], dtype=torch.float64)
    for i, h in enumerate(hkl.tolist()):
        j = lut.get(tuple(h))
        if j is not None:
            F2[i] = crystal.struct_factors_int[j]
    numbers = crystal.numbers.numpy()
    hkl_i16 = np.ascontiguousarray(hkl.numpy().astype(np.int16))
    return {
        "name": crystal.name,
        "spacegroup": getattr(crystal, "spacegroup", ""),
        "pointgroup": getattr(crystal, "pointgroup", ""),
        "cell": crystal.lat_real.numpy().tolist(),
        "recip": crystal.lat_recip.numpy().tolist(),
        "positions_frac": crystal.positions_frac.numpy().tolist(),
        "numbers": numbers.tolist(),
        "symbols": [chemical_symbols[int(z)] for z in numbers],
        "colors": [jmol_colors[int(z)].tolist() for z in numbers],
        "radii": [float(covalent_radii[int(z)]) for z in numbers],
        "hkl_i16": base64.b64encode(hkl_i16.tobytes()).decode(),
        "F2": _f32_b64(F2.numpy()),
        "U_re": _f32_b64(U_g.real.numpy()),
        "U_im": _f32_b64(U_g.imag.numpy()),
        "u0_imag": float(u0_imag),
        "absorptive": bool(have_dyn),
        "n_reflections": int(hkl.shape[0]),
        "energy_ev": float(energy_ev),
        "wavelength": float(electron_wavelength_angstrom(energy_ev)),
        "k_max": float(k_max),
        "hexagonal": bool(getattr(crystal, "hexagonal_matching", False)),
    }


def _coupling_vector(crystal, hkl: torch.Tensor, gamma_rel: float) -> tuple[torch.Tensor, float]:
    """U_g for a list of hkl (zero where no factor is stored) and the mean
    absorption U_000''. Same factor choice as bloch._coupling_matrix, but a
    vector lookup instead of the (N, N) difference matrix."""
    if getattr(crystal, "U_dyn", None) is not None:
        hkl_all, U_all = crystal.hkl_dyn, crystal.U_dyn
    else:
        hkl_all, U_all = crystal.hkl, crystal.struct_factors * (gamma_rel / np.pi)
    lut = {tuple(h): i for i, h in enumerate(hkl_all.tolist())}
    idx = torch.tensor([lut.get(tuple(h), -1) for h in hkl.tolist()], dtype=torch.long)
    U = torch.zeros(hkl.shape[0], dtype=torch.complex128)
    has = idx >= 0
    U[has] = U_all[idx[has]]
    i0 = lut.get((0, 0, 0), -1)
    u0_imag = (
        float(U_all[i0].imag) if (i0 >= 0 and getattr(crystal, "U_dyn", None) is not None) else 0.0
    )
    return U, u0_imag


def prepare_kossel_reference(
    crystal, energy_ev: float, thicknesses_A, angle_step_mrad=3.0, k_max=1.0
):
    """Bright field Kossel reference on the Lambert grid, for the pixel
    rendering of the Kossel / LACBED mode (a lookup in the browser)."""
    from quantem.diffraction import bloch

    master = bloch.calculate_kossel_reference(
        crystal,
        list(thicknesses_A),
        energy_ev=energy_ev,
        angle_step_mrad=angle_step_mrad,
        sg_max=0.05,
        k_max=k_max,
        progress_bar=False,
    )
    lam = np.nan_to_num(master["lambert"], nan=float(np.nanmax(master["lambert"])))
    return {
        "shape": list(lam.shape),
        "step": float(master["step"]),
        "thicknesses": [float(t) for t in master["thicknesses"]],
        "data": _f32_b64(lam),
    }


class DiffractionSim(anywidget.AnyWidget):
    """Interactive unit cell and diffraction pattern simulator.

    Parameters
    ----------
    crystal : Crystal | ase.Atoms | str | None
        A quantem Crystal, an ASE Atoms object, a CIF path, or the name of a
        preset (see DiffractionSim.presets_available()). None starts with silicon.
    energy_ev : float, default=200e3
        Beam energy.
    k_max : float, default=4.0
        Largest scattering vector in the pattern (1/Angstroms).
    zone_axis : sequence of 3 | None
        Initial zone axis along the beam; None keeps the identity
        orientation (c axis along the beam).
    thickness_A, semiconv_mrad, sigma_excitation : float
        Initial values of the thickness, convergence semiangle and
        excitation envelope sliders.
    pattern_range : float | None
        Scattering vector at the edge of the nanobeam / CBED panel
        (1/Angstroms); None shows everything out to k_max.
    field_mrad : float, default=50
        Half angle of the Kossel / LACBED field of view.
    sg_max : float, default=0.05
        Excitation error cutoff (1/Angstroms) selecting the Bloch beams;
        reflections outside it take thin-slab intensities.
    quality : {"fast", "medium", "fine"}
        Bloch beam cap and CBED tilt sampling.
    show_kikuchi : bool
        Overlay the Kikuchi line pairs on the nanobeam pattern.
    scaling : {"linear", "power", "log"}
        Intensity scaling of the pixel renderings; "power" raises the
        intensities to `power` (default 0.5).
    cmap : str
        Colormap of the pixel renderings, e.g. "inferno", "turbo_black", "gray".
    view_from : {"detector", "gun"}
        Viewpoint shared by both panels (a launch argument, no UI control). "detector" looks up the column from
        the detector side: the exit face of the cell is nearest you and tilts
        together with the Laue circle and Kikuchi pattern. "gun" is the
        operator's view down the column; there the entrance face is nearest
        and tilts opposite to the pattern (the Laue center marks where the
        zone axis exits toward the detector).
    mode : {"nanobeam", "cbed", "kossel"}
    render : {"markers", "pixels"}
        Nanobeam: markers sized by intensity, or a pixelated pattern.
        Kossel: vector lines, or the pixel lookup of the reference pattern
        (compute_kossel_reference()).
    n_cells : sequence of 3 int, default=(1, 1, 1)
        Block of cells drawn in the left panel (up to 6 per axis).
    polyhedra : bool
        Draw coordination polyhedra (convex hull of the nearest neighbours)
        around every species except the most numerous one; around every
        atom of an elemental crystal.
    size : int
        Height of the panels in CSS pixels.

    Examples
    --------
    >>> from quantem.widget import DiffractionSim
    >>> w = DiffractionSim("Si (diamond cubic)", zone_axis=[1, 1, 0])
    >>> w
    >>> w.export_html("si_110.html")   # standalone page, no Python needed
    """

    _esm = _STATIC

    crystal_json = traitlets.Unicode("{}").tag(sync=True)
    presets = traitlets.List(trait=traitlets.Unicode(), default_value=list(PRESETS)).tag(sync=True)
    preset = traitlets.Unicode("").tag(sync=True)
    energy_ev = traitlets.Float(200e3).tag(sync=True)
    k_max = traitlets.Float(4.0).tag(sync=True)
    orientation = traitlets.List(trait=traitlets.Float(), default_value=[1.0, 0.0, 0.0, 0.0]).tag(
        sync=True
    )
    mode = traitlets.Unicode("nanobeam").tag(sync=True)
    render = traitlets.Unicode("markers").tag(sync=True)
    dynamical = traitlets.Bool(True).tag(sync=True)
    thickness_A = traitlets.Float(500.0).tag(sync=True)
    semiconv_mrad = traitlets.Float(2.0).tag(sync=True)
    sigma_excitation = traitlets.Float(0.02).tag(sync=True)
    rotation_step_deg = traitlets.Float(15.0).tag(sync=True)
    pattern_range = traitlets.Float(4.0).tag(sync=True)
    field_mrad = traitlets.Float(50.0).tag(sync=True)
    sg_max = traitlets.Float(0.05).tag(sync=True)
    quality = traitlets.Unicode("medium").tag(sync=True)
    show_kikuchi = traitlets.Bool(False).tag(sync=True)
    view_from = traitlets.Unicode("detector").tag(sync=True)
    scaling = traitlets.Unicode("linear").tag(sync=True)
    power = traitlets.Float(0.5).tag(sync=True)
    cmap = traitlets.Unicode("inferno").tag(sync=True)
    vmin_pct = traitlets.Float(0.0).tag(sync=True)
    vmax_pct = traitlets.Float(100.0).tag(sync=True)
    show_labels = traitlets.Bool(True).tag(sync=True)
    show_cell_axes = traitlets.Bool(True).tag(sync=True)
    n_cells = traitlets.List(trait=traitlets.Int(), default_value=[1, 1, 1]).tag(sync=True)
    polyhedra = traitlets.Bool(False).tag(sync=True)
    size = traitlets.Int(420).tag(sync=True)
    kossel_json = traitlets.Unicode("{}").tag(sync=True)
    status = traitlets.Unicode("").tag(sync=True)
    widget_version = traitlets.Unicode("0.1").tag(sync=True)

    def __init__(self, crystal=None, zone_axis=None, pattern_range=None, **kwargs):
        if "n_cells" in kwargs:
            kwargs["n_cells"] = [int(n) for n in kwargs["n_cells"]]
        super().__init__(**kwargs)
        self.pattern_range = float(pattern_range) if pattern_range is not None else self.k_max
        self._crystal = None
        self._kossel_cache: dict = {}
        if crystal is None:
            crystal = "Si (diamond cubic)"
        self.set_crystal(crystal)
        if zone_axis is not None:
            self.set_zone_axis(zone_axis)
        self.observe(self._on_preset, names="preset")
        self.observe(self._on_physics, names=["energy_ev", "k_max"])
        self.on_msg(self._on_message)

    # ------------------------------------------------------------------
    @staticmethod
    def presets_available() -> list[str]:
        return list(PRESETS)

    @property
    def crystal(self):
        return self._crystal

    def set_crystal(self, crystal) -> "DiffractionSim":
        """Load a Crystal, ASE Atoms, CIF path or preset name."""
        from ase import Atoms

        from quantem.diffraction.crystal import Crystal

        preset_name = ""
        if isinstance(crystal, str):
            if crystal in PRESETS:
                preset_name = crystal
                xtl = Crystal.from_ase(
                    _preset_atoms(crystal), name=crystal.split(" (")[0], verbose=False
                )
            else:
                xtl = Crystal.from_cif(crystal, verbose=False)
        elif isinstance(crystal, Atoms):
            xtl = Crystal.from_ase(crystal, verbose=False)
        else:
            xtl = crystal
        self._crystal = xtl
        self.crystal_json = json.dumps(prepare_crystal(xtl, self.energy_ev, self.k_max))
        self.kossel_json = "{}"
        if preset_name and self.preset != preset_name:
            self.preset = preset_name
        return self

    def set_zone_axis(self, zone_axis, in_plane_deg: float = 0.0) -> "DiffractionSim":
        """Put a crystal direction [uvw] along the beam."""
        from quantem.diffraction.rotations import quat_from_zone_axis

        d = torch.as_tensor(zone_axis, dtype=torch.float64) @ self._crystal.lat_real
        q = quat_from_zone_axis(d[None], in_plane_deg)[0]
        self.orientation = [float(v) for v in q]
        return self

    def compute_kossel_reference(
        self, thicknesses_A=(300.0, 600.0, 1000.0), angle_step_mrad=3.0, k_max=1.0
    ):
        """Precompute the Kossel reference pattern for the pixel rendering of
        the Kossel / LACBED mode (about a minute for silicon at 3 mrad)."""
        key = (
            round(self.energy_ev),
            tuple(float(t) for t in thicknesses_A),
            angle_step_mrad,
            k_max,
        )
        if key not in self._kossel_cache:
            self.status = "computing Kossel reference pattern..."
            self._kossel_cache[key] = prepare_kossel_reference(
                self._crystal, self.energy_ev, thicknesses_A, angle_step_mrad, k_max
            )
            self.status = ""
        self.kossel_json = json.dumps(self._kossel_cache[key])
        return self

    # ------------------------------------------------------------------
    def _on_preset(self, change):
        name = change["new"]
        if (
            name in PRESETS
            and self._crystal is not None
            and self._crystal.name != name.split(" (")[0]
        ):
            self.set_crystal(name)

    def _on_physics(self, change):
        if change["name"] == "k_max" and self.pattern_range > self.k_max:
            self.pattern_range = self.k_max
        if self._crystal is not None:
            self.crystal_json = json.dumps(
                prepare_crystal(self._crystal, self.energy_ev, self.k_max)
            )
            self.kossel_json = "{}"

    def _on_message(self, widget, content, buffers):
        if isinstance(content, dict) and content.get("type") == "kossel_reference":
            self.compute_kossel_reference()

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        keys = [
            "crystal_json",
            "presets",
            "preset",
            "energy_ev",
            "k_max",
            "orientation",
            "mode",
            "render",
            "dynamical",
            "thickness_A",
            "semiconv_mrad",
            "sigma_excitation",
            "rotation_step_deg",
            "pattern_range",
            "field_mrad",
            "sg_max",
            "quality",
            "show_kikuchi",
            "view_from",
            "scaling",
            "power",
            "cmap",
            "vmin_pct",
            "vmax_pct",
            "show_labels",
            "show_cell_axes",
            "n_cells",
            "polyhedra",
            "size",
            "kossel_json",
            "status",
            "widget_version",
        ]
        return {k: getattr(self, k) for k in keys}

    def export_html(
        self, path, title: str | None = None, presets: list[str] | None = None
    ) -> pathlib.Path:
        """Write a standalone HTML page of the widget with its current state.

        The page carries the compiled widget, the crystal data and (when
        computed) the Kossel reference, and runs entirely in the browser:
        rotating the cell, changing thickness or mode needs no Python.
        `presets` lists additional crystals to embed so the crystal menu
        works offline (each adds its reflection list to the file).
        """
        state = self.state_dict()
        embedded = {}
        for name in presets or []:
            from quantem.diffraction.crystal import Crystal

            xtl = Crystal.from_ase(_preset_atoms(name), name=name.split(" (")[0], verbose=False)
            embedded[name] = prepare_crystal(xtl, self.energy_ev, self.k_max)
        if self.preset and self.preset not in embedded:
            embedded[self.preset] = json.loads(self.crystal_json)
        state["embedded_presets"] = embedded
        bundle = _STATIC.read_text()
        title = title or f"quantEM diffraction simulator: {self._crystal.name}"
        html = _standalone_html(bundle, state, title)
        out = pathlib.Path(path)
        out.write_text(html)
        return out


def _standalone_html(bundle_js: str, state: dict, title: str) -> str:
    state_json = json.dumps(state).replace("</", "<\\/")
    bundle_b64 = base64.b64encode(bundle_js.encode()).decode()
    safe_title = re.sub(r"[<>&]", "", title)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>{safe_title}</title>
<style>
  html, body {{ margin: 0; padding: 0; background: #ffffff; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
  #root {{ padding: 8px; }}
  @media (prefers-color-scheme: dark) {{ html, body {{ background: #1e1e1e; }} }}
</style>
</head>
<body>
<div id="root"></div>
<script type="module">
const state = {state_json};
class Model {{
  constructor(s) {{ this.s = s; this.cb = {{}}; }}
  get(k) {{ return this.s[k]; }}
  set(k, v) {{ this.s[k] = v; (this.cb["change:" + k] || []).forEach((f) => f()); }}
  save_changes() {{}}
  on(ev, f) {{ (this.cb[ev] = this.cb[ev] || []).push(f); }}
  off(ev, f) {{ if (!this.cb[ev]) return; this.cb[ev] = f ? this.cb[ev].filter((g) => g !== f) : []; }}
  send(msg) {{ if (msg && msg.type === "kossel_reference") this.set("status", "Kossel reference not available offline"); }}
}}
const bytes = Uint8Array.from(atob("{bundle_b64}"), (c) => c.charCodeAt(0));
const url = URL.createObjectURL(new Blob([bytes], {{ type: "text/javascript" }}));
const mod = await import(url);
const model = new Model(state);
model.set("standalone", true);
mod.render({{ model, el: document.getElementById("root") }});
</script>
</body>
</html>
"""
