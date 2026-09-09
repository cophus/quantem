/**
 * DiffractionSim: unit cell (left) and its diffraction pattern (right), both
 * following the same orientation. Drag the cell to tilt the crystal and the
 * pattern follows live. Every calculation (kinematical and Bloch wave
 * intensities, CBED disks, Kossel lines) runs here in the browser, so the
 * widget also works as a standalone HTML page.
 */

import * as React from "react";
import { createRender, useModel, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Switch from "@mui/material/Switch";
import Slider from "@mui/material/Slider";
import Button from "@mui/material/Button";
import TextField from "@mui/material/TextField";
import ToggleButton from "@mui/material/ToggleButton";
import ToggleButtonGroup from "@mui/material/ToggleButtonGroup";
import Tooltip from "@mui/material/Tooltip";
import { useTheme } from "../theme";
import { COLORMAP_NAMES } from "../colormaps";
import { downloadBlob } from "../format";
import { Quat, Vec3, directionIndices, matTVec, qmult, qnormalize, quatFromAxisAngle, quatFromZoneAxis, quatToMatrix } from "./math";
import {
  Reflection, blochIntensities, blochSolve, kinematicalTilted, kosselLines, kosselLookup, labReflections,
  parseCrystal, parseKossel, hybridBeams, slabIntensities,
} from "./physics";
import { cellGeometry, drawCell } from "./crystal3d";
import {
  Frame, cbedImage, drawImage, drawKikuchiOverlay, drawKosselLines, drawMarkers, histogramBins, nanobeamImage,
  setupCanvas, tiltGrid, toPx,
} from "./pattern";

const DIRECT: Reflection = { index: -1, hkl: [0, 0, 0], g: [0, 0, 0], gLen: 0, s: 0 };
const QUALITY: Record<string, { grid: number; beams: number; nanobeam: number }> = {
  fast: { grid: 5, beams: 24, nanobeam: 40 },
  medium: { grid: 7, beams: 36, nanobeam: 64 },
  fine: { grid: 9, beams: 56, nanobeam: 96 },
};

// ---------------------------------------------------------------------------
function Histogram({ bins, vminPct, vmaxPct, onRangeChange, width = 130, height = 40, dark, lo, hi }: {
  bins: number[]; vminPct: number; vmaxPct: number; onRangeChange: (a: number, b: number) => void;
  width?: number; height?: number; dark: boolean; lo: number; hi: number;
}) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const c = dark ? { bg: "#1a1a1a", on: "#888", off: "#444", border: "#333" } : { bg: "#f0f0f0", on: "#666", off: "#bbb", border: "#ccc" };
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr; canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    ctx.fillStyle = c.bg; ctx.fillRect(0, 0, width, height);
    const nb = 64, ratio = Math.floor(bins.length / nb);
    const red: number[] = [];
    for (let i = 0; i < nb; i++) { let s = 0; for (let j = 0; j < ratio; j++) s += bins[i * ratio + j] || 0; red.push(s); }
    const mx = Math.max(...red.map((v) => Math.log1p(v)), 1e-3);
    const bw = width / nb;
    const b0 = Math.floor((vminPct / 100) * nb), b1 = Math.floor((vmaxPct / 100) * nb);
    for (let i = 0; i < nb; i++) {
      const h = (Math.log1p(red[i]) / mx) * (height - 2);
      ctx.fillStyle = i >= b0 && i <= b1 ? c.on : c.off;
      ctx.fillRect(i * bw + 0.5, height - h, Math.max(1, bw - 1), h);
    }
  }, [bins, vminPct, vmaxPct, width, height, dark]);
  const fmt = (pct: number) => { const v = lo + (pct / 100) * (hi - lo); return Math.abs(v) >= 1000 || (Math.abs(v) < 0.01 && v !== 0) ? v.toExponential(1) : v.toFixed(2); };
  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 0.25 }}>
      <canvas ref={canvasRef} style={{ width, height, border: `1px solid ${c.border}` }} />
      <Slider
        value={[vminPct, vmaxPct]}
        onChange={(_, v) => { const [a, b] = v as number[]; onRangeChange(Math.min(a, b - 1), Math.max(b, a + 1)); }}
        min={0} max={100} size="small" valueLabelDisplay="auto" valueLabelFormat={fmt}
        sx={{ width, py: 0, "& .MuiSlider-thumb": { width: 8, height: 8 }, "& .MuiSlider-rail": { height: 2 }, "& .MuiSlider-track": { height: 2 }, "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px" } }}
      />
      <Box sx={{ display: "flex", justifyContent: "space-between", width }}>
        <Typography sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}>{fmt(vminPct)}</Typography>
        <Typography sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}>{fmt(vmaxPct)}</Typography>
      </Box>
    </Box>
  );
}

function LabeledSlider({ label, value, onChange, min, max, step, fmt, width = 200, disabled }: {
  label: string; value: number; onChange: (v: number) => void; min: number; max: number; step: number;
  fmt: (v: number) => string; width?: number; disabled?: boolean;
}) {
  return (
    <Box sx={{ width }}>
      <Box sx={{ display: "flex", justifyContent: "space-between" }}>
        <Typography sx={{ fontSize: 11, opacity: disabled ? 0.4 : 0.8 }}>{label}</Typography>
        <Typography sx={{ fontSize: 11, fontFamily: "monospace", opacity: disabled ? 0.4 : 0.9 }}>{fmt(value)}</Typography>
      </Box>
      <Slider value={value} min={min} max={max} step={step} size="small" disabled={disabled}
        onChange={(_, v) => onChange(v as number)}
        sx={{ py: 0.5, "& .MuiSlider-thumb": { width: 12, height: 12 } }} />
    </Box>
  );
}

function parseZoneAxis(text: string): Vec3 | null {
  const t = text.trim().replace(/[\[\]()]/g, "");
  let parts: string[];
  if (/[\s,]/.test(t)) parts = t.split(/[\s,]+/).filter(Boolean);
  else parts = t.match(/-?\d/g) || [];
  if (parts.length !== 3) return null;
  const v = parts.map(Number);
  if (v.some((x) => !isFinite(x)) || v.every((x) => x === 0)) return null;
  return v as Vec3;
}

function fmtIndices(v: [number, number, number] | null, brackets = "[]"): string {
  if (!v) return "—";
  return brackets[0] + v.map((h) => (h < 0 ? `${-h}̅` : `${h}`)).join("") + brackets[1];
}

// ---------------------------------------------------------------------------
function DiffSim() {
  const model = useModel();
  const { themeInfo, colors } = useTheme();
  const dark = themeInfo.theme === "dark";
  const standalone = !!model.get("standalone");
  const embedded: Record<string, unknown> = (model.get("embedded_presets") as Record<string, unknown>) || {};

  const [crystalJson, setCrystalJson] = useModelState<string>("crystal_json");
  const [kosselJson] = useModelState<string>("kossel_json");
  const [presets] = useModelState<string[]>("presets");
  const [preset, setPreset] = useModelState<string>("preset");
  const [energy, setEnergy] = useModelState<number>("energy_ev");
  const [orientation, setOrientation] = useModelState<number[]>("orientation");
  const [mode, setMode] = useModelState<string>("mode");
  const [render, setRender] = useModelState<string>("render");
  const [dynamical, setDynamical] = useModelState<boolean>("dynamical");
  const [thickness, setThickness] = useModelState<number>("thickness_A");
  const [semiconv, setSemiconv] = useModelState<number>("semiconv_mrad");
  const [sigma, setSigma] = useModelState<number>("sigma_excitation");
  const [stepDeg, setStepDeg] = useModelState<number>("rotation_step_deg");
  const [scaling, setScaling] = useModelState<string>("scaling");
  const [power, setPower] = useModelState<number>("power");
  const [cmap, setCmap] = useModelState<string>("cmap");
  const [vminPct, setVminPct] = useModelState<number>("vmin_pct");
  const [vmaxPct, setVmaxPct] = useModelState<number>("vmax_pct");
  const [showLabels, setShowLabels] = useModelState<boolean>("show_labels");
  const [showCellAxes, setShowCellAxes] = useModelState<boolean>("show_cell_axes");
  const [nCells, setNCells] = useModelState<number[]>("n_cells");
  const [polyhedra, setPolyhedra] = useModelState<boolean>("polyhedra");
  const [sizePref] = useModelState<number>("size");
  const [status] = useModelState<string>("status");

  const crystal = React.useMemo(() => parseCrystal(crystalJson), [crystalJson]);
  const kossel = React.useMemo(() => parseKossel(kosselJson), [kosselJson]);
  const nCellsSafe: [number, number, number] = [nCells?.[0] || 1, nCells?.[1] || 1, nCells?.[2] || 1];
  const geom = React.useMemo(() => (crystal ? cellGeometry(crystal, nCellsSafe, polyhedra) : null), [crystal, nCellsSafe.join(","), polyhedra]);

  // local view state
  const [patternRange, setPatternRange] = useModelState<number>("pattern_range");
  const [fieldMrad, setFieldMrad] = useModelState<number>("field_mrad");
  const [SG_MAX] = useModelState<number>("sg_max");
  const [quality, setQuality] = useModelState<string>("quality");
  const [kikuchi, setKikuchi] = useModelState<boolean>("show_kikuchi");
  const [viewFrom] = useModelState<string>("view_from");
  const viewX = viewFrom === "gun" ? 1 : -1;
  const qMaxDisp = Math.min(Math.max(patternRange || 0, 0.2), crystal?.k_max ?? 4);
  const setQMaxDisp = setPatternRange;
  const [zoneText, setZoneText] = React.useState("");
  const [dragging, setDragging] = React.useState(false);
  const [winW, setWinW] = React.useState(typeof window !== "undefined" ? window.innerWidth : 1200);
  React.useEffect(() => {
    const f = () => setWinW(window.innerWidth);
    window.addEventListener("resize", f);
    return () => window.removeEventListener("resize", f);
  }, []);
  const S = Math.max(220, Math.min(sizePref, winW - 40));

  // orientation: local quaternion for smooth dragging, pushed to the model with a throttle
  const [quat, setQuatLocal] = React.useState<Quat>(orientation as Quat);
  const quatRef = React.useRef<Quat>(quat);
  React.useEffect(() => { const q = orientation as Quat; quatRef.current = q; setQuatLocal(q); }, [orientation.join(",")]);
  const pushTimer = React.useRef<number | null>(null);
  const setQuat = React.useCallback((q: Quat, immediate = false) => {
    quatRef.current = q;
    setQuatLocal(q);
    const push = () => { pushTimer.current = null; setOrientation([...quatRef.current]); };
    if (immediate) { if (pushTimer.current) window.clearTimeout(pushTimer.current); push(); }
    else if (!pushTimer.current) pushTimer.current = window.setTimeout(push, 200);
  }, [setOrientation]);

  // axis given in SCREEN coordinates (x right, y up, z toward the viewer); mapped to the lab frame by the view
  const rotateLab = React.useCallback((axis: Vec3, deg: number, immediate = true) => {
    const dq = quatFromAxisAngle([viewX * axis[0], axis[1], viewX * axis[2]], (deg * Math.PI) / 180);
    setQuat(qnormalize(qmult(dq, quatRef.current)), immediate);
  }, [setQuat, viewX]);

  // ---- pointer handling on the cell canvas (mouse and touch) -------------
  const cellRef = React.useRef<HTMLCanvasElement>(null);
  const pointers = React.useRef<Map<number, [number, number]>>(new Map());
  const onPointerDown = (e: React.PointerEvent) => {
    (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
    pointers.current.set(e.pointerId, [e.clientX, e.clientY]);
    setDragging(true);
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const prev = pointers.current.get(e.pointerId);
    if (!prev) return;
    const cur: [number, number] = [e.clientX, e.clientY];
    if (pointers.current.size >= 2) {
      // two fingers: twist about the beam axis
      const other = [...pointers.current.entries()].find(([id]) => id !== e.pointerId);
      if (other) {
        const [ox, oy] = other[1];
        const a0 = Math.atan2(prev[1] - oy, prev[0] - ox);
        const a1 = Math.atan2(cur[1] - oy, cur[0] - ox);
        let da = a1 - a0;
        if (da > Math.PI) da -= 2 * Math.PI;
        if (da < -Math.PI) da += 2 * Math.PI;
        rotateLab([0, 0, 1], (-da * 180) / Math.PI, false);
      }
    } else {
      const dx = cur[0] - prev[0], dy = cur[1] - prev[1];
      const degPerPx = 180 / S;
      const ang = Math.hypot(dx, dy) * degPerPx;
      if (ang > 0) rotateLab([dy, dx, 0], ang, false); // trackball: the face nearest the viewer follows the pointer
    }
    pointers.current.set(e.pointerId, cur);
  };
  const onPointerUp = (e: React.PointerEvent) => {
    pointers.current.delete(e.pointerId);
    if (pointers.current.size === 0) { setDragging(false); setQuat(quatRef.current, true); }
  };

  // ---- pointer handling on the pattern canvas ---------------------------------
  // Dragging the pattern by dq (1/A, or rad in Kossel mode) moves the zone
  // axis so the pattern follows: the Laue circle centre sits at q = -k0 delta
  // for a zone axis tilted by delta, so the crystal tilts by -dq/k0.
  const shiftPattern = React.useCallback((dqx: number, dqy: number, inverseAngstrom: boolean, immediate: boolean) => {
    const k0 = crystal ? 1 / crystal.wavelength : 1;
    const ax = inverseAngstrom ? dqx / k0 : dqx, ay = inverseAngstrom ? dqy / k0 : dqy;
    const ang = Math.hypot(ax, ay);
    if (ang <= 0) return;
    const dq = quatFromAxisAngle([ay, -ax, 0], ang);
    setQuat(qnormalize(qmult(dq, quatRef.current)), immediate);
  }, [crystal, setQuat]);
  const patPointers = React.useRef<Map<number, [number, number]>>(new Map());
  const patScale = React.useRef(1); // px per unit of the current frame
  const onPatDown = (e: React.PointerEvent) => {
    (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
    patPointers.current.set(e.pointerId, [e.clientX, e.clientY]);
    setDragging(true);
  };
  const onPatMove = (e: React.PointerEvent) => {
    const prev = patPointers.current.get(e.pointerId);
    if (!prev) return;
    const cur: [number, number] = [e.clientX, e.clientY];
    if (patPointers.current.size >= 2) {
      const other = [...patPointers.current.entries()].find(([id]) => id !== e.pointerId);
      if (other) {
        const [ox, oy] = other[1];
        let da = Math.atan2(cur[1] - oy, cur[0] - ox) - Math.atan2(prev[1] - oy, prev[0] - ox);
        if (da > Math.PI) da -= 2 * Math.PI;
        if (da < -Math.PI) da += 2 * Math.PI;
        rotateLab([0, 0, 1], (-da * 180) / Math.PI, false);
      }
    } else {
      const dx = (viewX * (cur[0] - prev[0])) / patScale.current, dy = -(cur[1] - prev[1]) / patScale.current;
      shiftPattern(dx, dy, mode !== "kossel", false);
    }
    patPointers.current.set(e.pointerId, cur);
  };
  const onPatUp = (e: React.PointerEvent) => {
    patPointers.current.delete(e.pointerId);
    if (patPointers.current.size === 0) { setDragging(false); setQuat(quatRef.current, true); }
  };
  const onPatDoubleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left, y = e.clientY - rect.top;
    const qx = (viewX * (x - rect.width / 2)) / patScale.current, qy = -(y - rect.height / 2) / patScale.current;
    shiftPattern(-qx, -qy, mode !== "kossel", true);
  };

  // ---- derived geometry ---------------------------------------------------
  const R = React.useMemo(() => quatToMatrix(quat), [quat]);
  const zoneAxis = React.useMemo(() => {
    if (!crystal) return null;
    const dc = matTVec(R, [0, 0, 1]);
    return directionIndices(crystal.cell, dc);
  }, [crystal, R]);
  const k0 = crystal ? 1 / crystal.wavelength : 0;
  const qual = QUALITY[quality] || QUALITY.medium;

  // ---- nanobeam -----------------------------------------------------------
  const nb = React.useMemo(() => {
    if (!crystal || mode !== "nanobeam") return { beams: [] as Reflection[], nDyn: 0 };
    if (dynamical) return hybridBeams(crystal, quat, qMaxDisp, SG_MAX, dragging ? Math.min(qual.nanobeam, 40) : qual.nanobeam);
    return { beams: [DIRECT, ...labReflections(crystal, quat, qMaxDisp)], nDyn: 0 };
  }, [crystal, quat, qMaxDisp, mode, dynamical, dragging, qual, SG_MAX]);
  const nbBeams = nb.beams;
  const nbSol = React.useMemo(() => (crystal && mode === "nanobeam" && dynamical && nb.nDyn ? blochSolve(crystal, nbBeams.slice(0, nb.nDyn)) : null), [crystal, nbBeams, nb.nDyn, mode, dynamical]);
  const nbInten = React.useMemo(() => {
    if (!crystal || mode !== "nanobeam") return new Float64Array(0);
    if (dynamical && nbSol) {
      const out = new Float64Array(nbBeams.length);
      out.set(blochIntensities(nbSol, thickness));
      slabIntensities(crystal, nbBeams, nb.nDyn, [0, 0], thickness, out);
      return out;
    }
    return kinematicalTilted(crystal, nbBeams, [0, 0], sigma);
  }, [crystal, nbBeams, nb.nDyn, nbSol, mode, dynamical, thickness, sigma]);

  // ---- CBED ---------------------------------------------------------------
  const alpha = semiconv * 1e-3;
  const cbed = React.useMemo(() => {
    if (!crystal || mode !== "cbed") return null;
    const Rk = k0 * Math.sin(alpha);
    const grid = tiltGrid(Rk, dragging ? 5 : qual.grid);
    if (!dynamical) return { grid, beams: [DIRECT, ...labReflections(crystal, quat, qMaxDisp)], nDyn: 0, sols: null };
    const { beams, nDyn } = hybridBeams(crystal, quat, qMaxDisp, SG_MAX, dragging ? Math.min(qual.beams, 24) : qual.beams, Math.sin(alpha));
    const dyn = beams.slice(0, nDyn);
    const sols = grid.tilts.map((t) => blochSolve(crystal, dyn, t));
    return { grid, beams, nDyn, sols };
  }, [crystal, quat, qMaxDisp, mode, dynamical, alpha, k0, dragging, qual, SG_MAX]);
  const cbedInten = React.useMemo(() => {
    if (!crystal || !cbed) return null;
    if (cbed.sols) {
      return cbed.sols.map((sol, i) => {
        const out = new Float64Array(cbed.beams.length);
        out.set(blochIntensities(sol, thickness));
        slabIntensities(crystal, cbed.beams, cbed.nDyn, cbed.grid.tilts[i], thickness, out);
        return out;
      });
    }
    return cbed.grid.tilts.map((t) => kinematicalTilted(crystal, cbed.beams, t, sigma));
  }, [crystal, cbed, thickness, sigma]);

  // ---- Kossel -------------------------------------------------------------
  const fieldRad = fieldMrad * 1e-3;
  const lines = React.useMemo(() => {
    if (!crystal || (mode !== "kossel" && !(mode === "nanobeam" && kikuchi))) return [];
    const fov = mode === "kossel" ? fieldRad : qMaxDisp / k0;
    return kosselLines(crystal, quat, Math.min(crystal.k_max, 2.5), fov);
  }, [crystal, quat, mode, fieldRad, kikuchi, qMaxDisp, k0]);

  // ---- pixel image of the current mode --------------------------------------
  const frame: Frame = React.useMemo(() => ({ size: S, qMax: mode === "kossel" ? fieldRad : qMaxDisp, viewX }), [S, mode, fieldRad, qMaxDisp, viewX]);
  patScale.current = (0.5 * S * 0.92) / frame.qMax;
  const pixelMode = (mode === "nanobeam" && render === "pixels") || mode === "cbed" || (mode === "kossel" && render === "pixels");
  const image = React.useMemo<Float32Array | null>(() => {
    if (!crystal || !pixelMode) return null;
    if (mode === "nanobeam") return nanobeamImage(frame, nbBeams, nbInten, Math.max(1.5, S / 200));
    if (mode === "cbed" && cbed && cbedInten) return cbedImage(frame, cbed.beams, cbed.grid, cbedInten);
    if (mode === "kossel" && kossel) return kosselLookup(kossel, quat, fieldRad, S, thickness, viewX);
    return null;
  }, [crystal, pixelMode, mode, frame, nbBeams, nbInten, cbed, cbedInten, kossel, quat, fieldRad, S, thickness, viewX]);
  const display = React.useMemo(() => {
    if (!image) return null;
    let data = image;
    if (scaling === "log") {
      let mx = 0;
      for (let i = 0; i < image.length; i++) if (isFinite(image[i])) mx = Math.max(mx, image[i]);
      const eps = 1e-4 * (mx || 1);
      data = new Float32Array(image.length);
      for (let i = 0; i < image.length; i++) data[i] = isFinite(image[i]) ? Math.log10(Math.max(image[i], 0) + eps) : NaN;
    } else if (scaling === "power") {
      const pw = Math.min(Math.max(power || 0.5, 0.05), 1);
      data = new Float32Array(image.length);
      for (let i = 0; i < image.length; i++) data[i] = isFinite(image[i]) ? Math.pow(Math.max(image[i], 0), pw) : NaN;
    }
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < data.length; i++) { const v = data[i]; if (isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } }
    if (!(hi > lo)) { lo = 0; hi = 1; }
    return { data, lo, hi, bins: histogramBins(data, lo, hi) };
  }, [image, scaling, power]);

  // ---- drawing --------------------------------------------------------------
  React.useEffect(() => {
    const canvas = cellRef.current;
    if (!canvas || !geom) return;
    drawCell(canvas, geom, quat, S, { dark, showAxes: showCellAxes, showLabels: showLabels, atomScale: 0.45, viewX });
  }, [geom, quat, S, dark, showCellAxes, showLabels, viewX]);

  const patRef = React.useRef<HTMLCanvasElement>(null);
  React.useEffect(() => {
    const canvas = patRef.current;
    if (!canvas || !crystal) return;
    const ctx = setupCanvas(canvas, S);
    if (!ctx) return;
    if (display) {
      const vmin = display.lo + (vminPct / 100) * (display.hi - display.lo);
      const vmax = display.lo + (vmaxPct / 100) * (display.hi - display.lo);
      drawImage(ctx, frame, display.data, cmap, vmin, vmax, dark, mode === "kossel" ? "rad" : "Å⁻¹", mode === "kossel" ? 0.01 : 1);
      if (mode === "nanobeam" && kikuchi) drawKikuchiOverlay(ctx, frame, lines, k0, true);
      if (mode === "nanobeam" && showLabels) labelBeams(ctx, frame, nbBeams, nbInten, dark, true);
    } else if (mode === "nanobeam") {
      drawMarkers(ctx, frame, nbBeams, nbInten, dark, showLabels, !dynamical);
      if (kikuchi) drawKikuchiOverlay(ctx, frame, lines, k0, dark);
    } else if (mode === "kossel") {
      if (render === "pixels" && !kossel) {
        ctx.fillStyle = dark ? "#000" : "#fff"; ctx.fillRect(0, 0, S, S);
        ctx.fillStyle = dark ? "#ccc" : "#333"; ctx.font = "13px sans-serif"; ctx.textAlign = "center";
        ctx.fillText("no Kossel reference pattern loaded", S / 2, S / 2 - 10);
        ctx.fillText(standalone ? "(export the page after compute_kossel_reference)" : "press “compute reference” below", S / 2, S / 2 + 10);
      } else {
        drawKosselLines(ctx, frame, lines, dark, showLabels, 0.02);
      }
    } else if (mode === "cbed") {
      ctx.fillStyle = dark ? "#000" : "#fff"; ctx.fillRect(0, 0, S, S);
    }
  }, [crystal, display, frame, mode, render, dark, cmap, vminPct, vmaxPct, nbBeams, nbInten, showLabels, dynamical, kikuchi, lines, k0, kossel, S, standalone]);

  // ---- actions ---------------------------------------------------------------
  const goZoneAxis = () => {
    const uvw = parseZoneAxis(zoneText);
    if (!uvw || !crystal) return;
    const c = crystal.cell;
    const d: Vec3 = [
      uvw[0] * c[0][0] + uvw[1] * c[1][0] + uvw[2] * c[2][0],
      uvw[0] * c[0][1] + uvw[1] * c[1][1] + uvw[2] * c[2][1],
      uvw[0] * c[0][2] + uvw[1] * c[1][2] + uvw[2] * c[2][2],
    ];
    setQuat(quatFromZoneAxis(d), true);
  };
  const choosePreset = (name: string) => {
    if (standalone) {
      const data = embedded[name];
      if (data) { setCrystalJson(JSON.stringify(data)); setPreset(name); }
    } else {
      setPreset(name);
    }
  };
  const savePng = () => {
    const a = cellRef.current, b = patRef.current;
    if (!a || !b) return;
    const off = document.createElement("canvas");
    off.width = a.width + b.width + 8; off.height = Math.max(a.height, b.height);
    const ctx = off.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = dark ? "#1e1e1e" : "#fff"; ctx.fillRect(0, 0, off.width, off.height);
    ctx.drawImage(a, 0, 0); ctx.drawImage(b, a.width + 8, 0);
    off.toBlob((blob) => { if (blob) downloadBlob(blob, `${crystal?.name || "crystal"}_${mode}.png`); });
  };
  const exportHtml = async () => {
    const res = await fetch(import.meta.url);
    const bundle = await res.text();
    const keys = ["crystal_json", "presets", "preset", "energy_ev", "k_max", "orientation", "mode", "render", "dynamical", "thickness_A",
      "semiconv_mrad", "sigma_excitation", "rotation_step_deg", "pattern_range", "field_mrad", "sg_max", "quality", "show_kikuchi", "view_from",
      "scaling", "power", "cmap", "vmin_pct", "vmax_pct", "show_labels",
      "show_cell_axes", "n_cells", "polyhedra", "size", "kossel_json", "status", "widget_version"];
    const state: Record<string, unknown> = {};
    for (const k of keys) state[k] = model.get(k);
    state.orientation = [...quatRef.current];
    const emb: Record<string, unknown> = { ...embedded };
    if (crystal && crystalJson) emb[preset || crystal.name] = JSON.parse(crystalJson);
    state.embedded_presets = emb;
    state.presets = Object.keys(emb);
    if (!state.preset) state.preset = crystal?.name || "";
    downloadBlob(new Blob([standaloneHtml(bundle, state, `quantEM diffraction simulator: ${crystal?.name || ""}`)], { type: "text/html" }),
      `${crystal?.name || "crystal"}_diffsim.html`);
  };

  const presetNames = standalone ? Object.keys(embedded) : presets;
  const ctl = {
    fontSize: 12, height: 30, bgcolor: colors.controlBg, color: colors.text,
    "& .MuiSelect-select": { py: 0.4, fontSize: 12 },
    "& .MuiSvgIcon-root": { color: colors.textMuted },
    "& .MuiOutlinedInput-notchedOutline": { borderColor: colors.border },
    "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: colors.accent },
    "&.Mui-disabled": { color: colors.textMuted, "& .MuiSelect-select": { WebkitTextFillColor: colors.textMuted } },
  };
  const menuProps = { PaperProps: { sx: { bgcolor: colors.controlBg, color: colors.text, border: `1px solid ${colors.border}` } }, sx: { zIndex: 9999 } };
  const tbg = {
    "& .MuiToggleButton-root": {
      px: 1, py: 0.3, fontSize: 11, textTransform: "none", color: colors.textMuted, borderColor: colors.border, bgcolor: colors.controlBg,
      "&.Mui-selected": { color: colors.accent, bgcolor: dark ? "#2e3a48" : "#e3eefc" },
      "&:hover": { bgcolor: dark ? "#333" : "#e8e8e8" },
    },
  };
  const tf = {
    "& input": { fontSize: 12, py: 0.6, color: colors.text },
    "& input::placeholder": { color: colors.textMuted, opacity: 1 },
    "& .MuiOutlinedInput-notchedOutline": { borderColor: colors.border },
    "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: colors.accent },
    bgcolor: colors.controlBg,
  };
  const btn = { fontSize: 11, height: 30, color: colors.accent, borderColor: colors.border, textTransform: "none" as const, "&:hover": { borderColor: colors.accent } };
  const sw = { "& .MuiSwitch-track": { bgcolor: dark ? "#777" : undefined } };
  const panelW = S;
  const nDyn = mode === "nanobeam" ? nb.nDyn : mode === "cbed" && cbed ? cbed.nDyn : 0;
  const nBeams = mode === "nanobeam" ? nbBeams.length : mode === "cbed" && cbed ? cbed.beams.length : lines.length;

  if (!crystal || !geom) {
    return <Box sx={{ p: 2, color: colors.text, bgcolor: colors.bg }}>loading crystal…</Box>;
  }

  return (
    <Box sx={{ bgcolor: colors.bg, color: colors.text, p: 1.25, borderRadius: 1, border: `1px solid ${colors.border}`, width: "fit-content", maxWidth: "100%", fontFamily: "system-ui, sans-serif", boxSizing: "border-box" }}>
      {/* top bar */}
      <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap sx={{ mb: 1 }}>
        <Typography sx={{ fontSize: 13, fontWeight: 600, mr: 0.5 }}>Diffraction simulator</Typography>
        <Select size="small" value={presetNames.includes(preset) ? preset : ""} displayEmpty onChange={(e) => choosePreset(e.target.value as string)} sx={{ ...ctl, minWidth: 170 }} MenuProps={menuProps}>
          {!presetNames.includes(preset) && <MenuItem value="" sx={{ fontSize: 12 }}>{crystal.name}</MenuItem>}
          {presetNames.map((p) => <MenuItem key={p} value={p} sx={{ fontSize: 12 }}>{p}</MenuItem>)}
        </Select>
        <Tooltip title={standalone ? "The exported page carries one energy" : "Beam energy; the reflection list is recomputed in Python"}>
          <Select size="small" value={energy} disabled={standalone} onChange={(e) => setEnergy(Number(e.target.value))} sx={{ ...ctl, minWidth: 90 }} MenuProps={menuProps}>
            {[60e3, 80e3, 100e3, 120e3, 200e3, 300e3].filter((v) => v !== energy).concat([energy]).sort((a, b) => a - b).map((v) => (
              <MenuItem key={v} value={v} sx={{ fontSize: 12 }}>{(v / 1e3).toFixed(0)} keV</MenuItem>
            ))}
          </Select>
        </Tooltip>
        <TextField size="small" placeholder="zone axis, e.g. 1 1 0" value={zoneText} onChange={(e) => setZoneText(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") goZoneAxis(); }}
          sx={{ width: 150, ...tf }} />
        <Button size="small" variant="outlined" onClick={goZoneAxis} sx={{ ...btn, minWidth: 0, px: 1 }}>go</Button>
        <Box sx={{ flex: 1 }} />
        <Button size="small" variant="outlined" onClick={savePng} sx={btn}>save PNG</Button>
        <Button size="small" variant="outlined" onClick={exportHtml} sx={btn}>export HTML</Button>
      </Stack>

      <Stack direction="row" spacing={1.5} flexWrap="wrap" useFlexGap alignItems="flex-start">
        {/* left: unit cell */}
        <Box sx={{ width: panelW }}>
          <canvas ref={cellRef} style={{ width: S, height: S, touchAction: "none", cursor: dragging ? "grabbing" : "grab", borderRadius: 4, background: dark ? "#141414" : "#fafafa", border: `1px solid ${colors.border}`, display: "block" }}
            onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp} onPointerCancel={onPointerUp} onPointerLeave={onPointerUp} />
          <Stack direction="row" spacing={0.5} alignItems="center" flexWrap="wrap" useFlexGap sx={{ mt: 0.75 }}>
            {(["x", "y", "z"] as const).map((ax, i) => (
              <ToggleButtonGroup key={ax} size="small" exclusive value={null} sx={tbg}>
                <ToggleButton value="-" onClick={() => rotateLab([+(i === 0), +(i === 1), +(i === 2)], -stepDeg)}>{ax} −</ToggleButton>
                <ToggleButton value="+" onClick={() => rotateLab([+(i === 0), +(i === 1), +(i === 2)], stepDeg)}>{ax} +</ToggleButton>
              </ToggleButtonGroup>
            ))}
            <TextField size="small" type="number" value={stepDeg} onChange={(e) => setStepDeg(Math.max(0.01, Number(e.target.value) || 0.01))}
              inputProps={{ step: 1, min: 0.01, max: 180, style: { fontSize: 11, padding: "4px 6px", width: 42 } }} sx={tf} />
            <Typography sx={{ fontSize: 11, opacity: 0.7 }}>°</Typography>
            <Button size="small" onClick={() => setQuat([1, 0, 0, 0], true)} sx={{ ...btn, height: 26, minWidth: 0, px: 1 }}>reset</Button>
          </Stack>
          <Stack direction="row" spacing={1.5} alignItems="center" sx={{ mt: 0.5 }}>
            <Typography sx={{ fontSize: 11, opacity: 0.75 }}>{crystal.name} · {crystal.spacegroup || crystal.pointgroup}</Typography>
            <Typography sx={{ fontSize: 11, fontFamily: "monospace" }}>zone axis {fmtIndices(zoneAxis)}</Typography>
          </Stack>
          <Stack direction="row" spacing={1} alignItems="center">
            <Switch size="small" sx={sw} checked={showCellAxes} onChange={(e) => setShowCellAxes(e.target.checked)} />
            <Typography sx={{ fontSize: 11 }}>cell axes</Typography>
            <Switch size="small" sx={sw} checked={showLabels} onChange={(e) => setShowLabels(e.target.checked)} />
            <Typography sx={{ fontSize: 11 }}>labels</Typography>
            <Switch size="small" sx={sw} checked={polyhedra} onChange={(e) => setPolyhedra(e.target.checked)} />
            <Typography sx={{ fontSize: 11 }}>polyhedra</Typography>
          </Stack>
          <Stack direction="row" spacing={0.5} alignItems="center" sx={{ mt: 0.25 }}>
            <Typography sx={{ fontSize: 11, mr: 0.5 }}>cells</Typography>
            {[0, 1, 2].map((i) => (
              <TextField key={i} size="small" type="number" value={nCellsSafe[i]}
                onChange={(e) => { const v = [...nCellsSafe]; v[i] = Math.max(1, Math.min(6, Math.round(Number(e.target.value) || 1))); setNCells(v); }}
                inputProps={{ min: 1, max: 6, step: 1, style: { fontSize: 11, padding: "3px 4px", width: 26 } }} sx={tf} />
            ))}
            <Typography sx={{ fontSize: 11, opacity: 0.7 }}>along a, b, c</Typography>
          </Stack>
          <Typography sx={{ fontSize: 10.5, opacity: 0.6, mt: 0.5 }}>drag the cell (near face follows) or the pattern (tilt map follows) · double-click a point of the pattern to centre it · two fingers twist · buttons rotate about the screen axes</Typography>
        </Box>

        {/* right: pattern */}
        <Box sx={{ width: panelW }}>
          <canvas ref={patRef} style={{ width: S, height: S, touchAction: "none", cursor: dragging ? "grabbing" : "grab", borderRadius: 4, border: `1px solid ${colors.border}`, display: "block" }}
            onPointerDown={onPatDown} onPointerMove={onPatMove} onPointerUp={onPatUp} onPointerCancel={onPatUp} onPointerLeave={onPatUp} onDoubleClick={onPatDoubleClick} />
          <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap sx={{ mt: 0.75 }}>
            <ToggleButtonGroup size="small" exclusive value={mode} onChange={(_, v) => v && setMode(v)} sx={tbg}>
              <ToggleButton value="nanobeam">nanobeam</ToggleButton>
              <ToggleButton value="cbed">CBED</ToggleButton>
              <ToggleButton value="kossel">Kossel / LACBED</ToggleButton>
            </ToggleButtonGroup>
            {mode !== "cbed" && (
              <ToggleButtonGroup size="small" exclusive value={render} onChange={(_, v) => v && setRender(v)} sx={tbg}>
                <ToggleButton value="markers">{mode === "kossel" ? "lines" : "markers"}</ToggleButton>
                <ToggleButton value="pixels">pixels</ToggleButton>
              </ToggleButtonGroup>
            )}
            {mode !== "kossel" && (
              <Stack direction="row" alignItems="center">
                <Switch size="small" sx={sw} checked={dynamical} onChange={(e) => setDynamical(e.target.checked)} />
                <Typography sx={{ fontSize: 11 }}>dynamical</Typography>
              </Stack>
            )}
          </Stack>
          <Stack direction="row" spacing={2} flexWrap="wrap" useFlexGap sx={{ mt: 0.75 }}>
            <LabeledSlider label="thickness" value={thickness} onChange={setThickness} min={10} max={2000} step={5} fmt={(v) => `${v.toFixed(0)} Å`}
              disabled={mode !== "kossel" ? !dynamical : render !== "pixels"} />
            {mode === "cbed" && <LabeledSlider label="convergence semiangle" value={semiconv} onChange={setSemiconv} min={0.2} max={30} step={0.1} fmt={(v) => `${v.toFixed(1)} mrad`} />}
            {mode === "kossel" && <LabeledSlider label="field of view (half angle)" value={fieldMrad} onChange={setFieldMrad} min={10} max={250} step={5} fmt={(v) => `${v.toFixed(0)} mrad`} />}
            {mode !== "kossel" && <LabeledSlider label="pattern range" value={qMaxDisp} onChange={setQMaxDisp} min={0.2} max={crystal.k_max} step={0.05} fmt={(v) => `${v.toFixed(2)} Å⁻¹`} />}
            {mode !== "kossel" && !dynamical && <LabeledSlider label="excitation error σ" value={sigma} onChange={setSigma} min={0.002} max={0.1} step={0.001} fmt={(v) => `${v.toFixed(3)} Å⁻¹`} />}
          </Stack>
          <Stack direction="row" spacing={1.5} alignItems="flex-start" flexWrap="wrap" useFlexGap sx={{ mt: 0.75 }}>
            {display && (
              <Histogram bins={display.bins} vminPct={vminPct} vmaxPct={vmaxPct} onRangeChange={(a, b) => { setVminPct(a); setVmaxPct(b); }} dark={dark} lo={display.lo} hi={display.hi} />
            )}
            {pixelMode && (
              <Stack spacing={0.25}>
                <Select size="small" value={COLORMAP_NAMES.includes(cmap) ? cmap : COLORMAP_NAMES[0]} onChange={(e) => setCmap(e.target.value as string)} sx={{ ...ctl, minWidth: 110 }} MenuProps={menuProps}>
                  {COLORMAP_NAMES.map((n) => <MenuItem key={n} value={n} sx={{ fontSize: 12 }}>{n}</MenuItem>)}
                </Select>
                <Select size="small" value={["linear", "power", "log"].includes(scaling) ? scaling : "linear"} onChange={(e) => setScaling(e.target.value as string)} sx={{ ...ctl, minWidth: 110 }} MenuProps={menuProps}>
                  <MenuItem value="linear" sx={{ fontSize: 12 }}>linear</MenuItem>
                  <MenuItem value="power" sx={{ fontSize: 12 }}>power law</MenuItem>
                  <MenuItem value="log" sx={{ fontSize: 12 }}>log</MenuItem>
                </Select>
                {scaling === "power" && (
                  <LabeledSlider label="exponent" value={power} onChange={setPower} min={0.1} max={1} step={0.05} fmt={(v) => v.toFixed(2)} width={110} />
                )}
              </Stack>
            )}
            {mode === "nanobeam" && (
              <Stack direction="row" alignItems="center">
                <Switch size="small" sx={sw} checked={kikuchi} onChange={(e) => setKikuchi(e.target.checked)} />
                <Typography sx={{ fontSize: 11 }}>Kikuchi lines</Typography>
              </Stack>
            )}
            {mode !== "kossel" && dynamical && (
              <Stack direction="row" alignItems="center" spacing={0.5}>
                <Typography sx={{ fontSize: 11, opacity: 0.8 }}>quality</Typography>
                <Select size="small" value={quality} onChange={(e) => setQuality(e.target.value as string)} sx={{ ...ctl, minWidth: 90 }} MenuProps={menuProps}>
                  {Object.keys(QUALITY).map((n) => <MenuItem key={n} value={n} sx={{ fontSize: 12 }}>{n}</MenuItem>)}
                </Select>
              </Stack>
            )}
            {mode === "kossel" && render === "pixels" && !kossel && !standalone && (
              <Button size="small" variant="outlined" onClick={() => model.send({ type: "kossel_reference" })} sx={btn}>compute reference</Button>
            )}
          </Stack>
          <Typography sx={{ fontSize: 10.5, opacity: 0.6, mt: 0.5 }}>
            {(energy / 1e3).toFixed(0)} keV · λ = {(crystal.wavelength * 100).toFixed(3)} pm · {nBeams} {mode === "kossel" ? "lines" : "beams"}
            {mode !== "kossel" && dynamical ? ` · ${nDyn} Bloch beams (|s| < ${SG_MAX} Å⁻¹${crystal.absorptive ? ", absorptive" : ""}), thin-slab intensities for the rest` : ""}
            {mode === "cbed" ? " · disks summed incoherently where they overlap" : ""}
            {status ? ` · ${status}` : ""}
          </Typography>
        </Box>
      </Stack>
    </Box>
  );
}

function labelBeams(ctx: CanvasRenderingContext2D, f: Frame, beams: Reflection[], inten: Float64Array, dark: boolean, onImage: boolean) {
  let iMax = 0;
  for (let i = 0; i < beams.length; i++) iMax = Math.max(iMax, inten[i]);
  if (iMax <= 0) return;
  ctx.font = `${Math.max(9, Math.round(f.size / 38))}px sans-serif`;
  ctx.textAlign = "center"; ctx.textBaseline = "bottom";
  ctx.fillStyle = onImage ? "#ffd54f" : dark ? "#ffd54f" : "#c62828";
  let count = 0;
  for (let i = 0; i < beams.length && count < 40; i++) {
    if (inten[i] / iMax < 0.08) continue;
    const [x, y] = toPx(f, beams[i].g[0], beams[i].g[1]);
    ctx.fillText(beams[i].hkl.map((h) => (h < 0 ? `${-h}̅` : `${h}`)).join(""), x, y - 6);
    count++;
  }
}

function standaloneHtml(bundle: string, state: Record<string, unknown>, title: string): string {
  const bytes = new TextEncoder().encode(bundle);
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  const b64 = btoa(bin);
  const stateJson = JSON.stringify(state).replace(/<\//g, "<\\/");
  const safeTitle = title.replace(/[<>&]/g, "");
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>${safeTitle}</title>
<style>
  html, body { margin: 0; padding: 0; background: #ffffff; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
  #root { padding: 8px; }
  @media (prefers-color-scheme: dark) { html, body { background: #1e1e1e; } }
</style>
</head>
<body>
<div id="root"></div>
<script type="module">
const state = ${stateJson};
class Model {
  constructor(s) { this.s = s; this.cb = {}; }
  get(k) { return this.s[k]; }
  set(k, v) { this.s[k] = v; (this.cb["change:" + k] || []).forEach((f) => f()); }
  save_changes() {}
  on(ev, f) { (this.cb[ev] = this.cb[ev] || []).push(f); }
  off(ev, f) { if (!this.cb[ev]) return; this.cb[ev] = f ? this.cb[ev].filter((g) => g !== f) : []; }
  send(msg) { if (msg && msg.type === "kossel_reference") this.set("status", "Kossel reference not available offline"); }
}
const bytes = Uint8Array.from(atob("${b64}"), (c) => c.charCodeAt(0));
const url = URL.createObjectURL(new Blob([bytes], { type: "text/javascript" }));
const mod = await import(url);
const model = new Model(state);
model.set("standalone", true);
mod.render({ model, el: document.getElementById("root") });
</script>
</body>
</html>
`;
}

export const render = createRender(DiffSim);
