/**
 * Show3DAtoms - orthogonal-slice viewer of a 3D volume with an atomic-site overlay.
 *
 * One panel shows an xy / xz / yz slice (movable along its normal), an intensity
 * histogram (percentile-clipped so outliers don't dominate) with an adjustable
 * color range, and the traced sites overlaid: marker size scales with intensity,
 * marker opacity fades with distance from the slice.
 *
 * Interaction: left-drag = box zoom, middle-drag = pan, double-click = reset view
 * and all settings.  All site coordinates are in voxel/array-index space.
 */

import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import Slider from "@mui/material/Slider";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Switch from "@mui/material/Switch";
import { useTheme } from "../theme";
import { extractFloat32 } from "../format";
import { COLORMAPS, COLORMAP_NAMES, applyColormap } from "../colormaps";
import { findDataRange, sliderRange, percentileClip } from "../stats";

const sliderStyles = {
  py: 0,
  "& .MuiSlider-thumb": { width: 11, height: 11 },
  "& .MuiSlider-rail": { height: 2 },
  "& .MuiSlider-track": { height: 2 },
};

const MARKER_BASE = 1.5;   // half the previous default size
const MIN_ZOOM_VOX = 4;    // smallest viewport extent (voxels)
const AXIS_LABELS = ["x", "y", "z"];

type PlaneInfo = {
  normalAxis: number; rowAxis: number; colAxis: number;
  rows: number; cols: number; depth: number; normalLabel: string;
};
type Viewport = { row0: number; row1: number; col0: number; col1: number };

function planeInfo(plane: string, n0: number, n1: number, n2: number): PlaneInfo {
  let normalAxis: number, rowAxis: number, colAxis: number;
  if (plane === "yz") { normalAxis = 0; rowAxis = 1; colAxis = 2; }
  else if (plane === "xz") { normalAxis = 1; rowAxis = 0; colAxis = 2; }
  else { normalAxis = 2; rowAxis = 0; colAxis = 1; }  // xy
  const dims = [n0, n1, n2];
  return {
    normalAxis, rowAxis, colAxis,
    rows: dims[rowAxis], cols: dims[colAxis], depth: dims[normalAxis],
    normalLabel: AXIS_LABELS[normalAxis],
  };
}

function extractSlice(vol: Float32Array, n1: number, n2: number, info: PlaneInfo, k: number): Float32Array {
  const { rows, cols, normalAxis } = info;
  const out = new Float32Array(rows * cols);
  const stride0 = n1 * n2;
  if (normalAxis === 2) {
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) out[r * cols + c] = vol[r * stride0 + c * n2 + k];
  } else if (normalAxis === 1) {
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) out[r * cols + c] = vol[r * stride0 + k * n2 + c];
  } else {
    const base = k * stride0;
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) out[r * cols + c] = vol[base + r * n2 + c];
  }
  return out;
}

/** Histogram binned over a fixed [lo, hi] range, clipping outliers into the end bins. */
function histogramInRange(data: Float32Array, lo: number, hi: number, nbins = 96): number[] {
  const bins = new Array(nbins).fill(0);
  const range = hi > lo ? hi - lo : 1;
  const scale = nbins / range;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (!isFinite(v)) continue;
    let b = Math.floor((v - lo) * scale);
    if (b < 0) b = 0; else if (b >= nbins) b = nbins - 1;
    bins[b]++;
  }
  const mx = Math.max(...bins, 1e-9);
  for (let i = 0; i < nbins; i++) bins[i] /= mx;
  return bins;
}

function Show3DAtoms() {
  const { colors: tc } = useTheme();

  const [volumeBytes] = useModelState<DataView>("volume_bytes");
  const [sitesBytes] = useModelState<DataView>("sites_bytes");
  const [n0] = useModelState<number>("n0");
  const [n1] = useModelState<number>("n1");
  const [n2] = useModelState<number>("n2");
  const [numSites] = useModelState<number>("num_sites");
  const [title] = useModelState<string>("title");
  const [cmap, setCmap] = useModelState<string>("cmap");
  const [plane, setPlane] = useModelState<string>("plane");
  const [sliceIndex, setSliceIndex] = useModelState<number>("slice_index");
  const [thickness, setThickness] = useModelState<number>("slice_thickness");
  const [falloff, setFalloff] = useModelState<number>("opacity_falloff");
  const [vminPct, setVminPct] = useModelState<number>("vmin_pct");
  const [vmaxPct, setVmaxPct] = useModelState<number>("vmax_pct");
  const [markerScale, setMarkerScale] = useModelState<number>("marker_scale");
  const [markerLinewidth, setMarkerLinewidth] = useModelState<number>("marker_linewidth");
  const [markerFilled, setMarkerFilled] = useModelState<boolean>("marker_filled");
  const [showSites, setShowSites] = useModelState<boolean>("show_sites");
  const [showSlice, setShowSlice] = useModelState<boolean>("show_slice");
  const [canvasSize] = useModelState<number>("canvas_size");

  const volume = React.useMemo(() => extractFloat32(volumeBytes), [volumeBytes]);
  const sites = React.useMemo(() => extractFloat32(sitesBytes), [sitesBytes]);
  const info = React.useMemo(() => planeInfo(plane, n0, n1, n2), [plane, n0, n1, n2]);
  const k = Math.max(0, Math.min(info.depth - 1, sliceIndex));

  // Robust intensity range (percentile-clipped) for histogram + color mapping.
  const baseRange = React.useMemo(() => {
    if (!volume) return { lo: 0, hi: 1 };
    const { vmin, vmax, min, max } = percentileClip(volume, 0.5, 99.5);
    return vmax > vmin ? { lo: vmin, hi: vmax } : findDataRange(volume).max > findDataRange(volume).min
      ? { lo: min, hi: max } : { lo: min, hi: min + 1 };
  }, [volume]);
  const histBins = React.useMemo(
    () => (volume ? histogramInRange(volume, baseRange.lo, baseRange.hi) : null),
    [volume, baseRange],
  );
  const maxIntensity = React.useMemo(() => {
    if (!sites || numSites === 0) return 1;
    let m = 0;
    for (let i = 0; i < numSites; i++) m = Math.max(m, sites[i * 5 + 3]);
    return m > 0 ? m : 1;
  }, [sites, numSites]);

  const scale = canvasSize / Math.max(info.rows, info.cols, 1);
  const canvasW = Math.round(info.cols * scale);
  const canvasH = Math.round(info.rows * scale);

  const [viewport, setViewport] = React.useState<Viewport | null>(null);
  const [drag, setDrag] = React.useState<{ mode: "zoom" | "pan"; sx: number; sy: number; startVp: Viewport } | null>(null);
  const [dragBox, setDragBox] = React.useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null);

  const fullVp = React.useCallback((): Viewport => ({ row0: 0, row1: info.rows, col0: 0, col1: info.cols }), [info]);
  const vp = viewport ?? fullVp();

  // Reset viewport + clamp slice when the plane changes.
  const prevPlane = React.useRef(plane);
  React.useEffect(() => {
    if (prevPlane.current !== plane) {
      prevPlane.current = plane;
      setViewport(null);
      if (sliceIndex > info.depth - 1) setSliceIndex(Math.floor(info.depth / 2));
    }
  }, [plane, info.depth, sliceIndex, setSliceIndex]);

  // ---- Effect A: colormap the full slice into an offscreen canvas (heavy) ----
  const offRef = React.useRef<HTMLCanvasElement | null>(null);
  const [sliceVersion, setSliceVersion] = React.useState(0);
  React.useEffect(() => {
    if (!volume) return;
    let off = offRef.current;
    if (!off) { off = document.createElement("canvas"); offRef.current = off; }
    off.width = info.cols; off.height = info.rows;
    const octx = off.getContext("2d")!;
    if (showSlice) {
      const sliceData = extractSlice(volume, n1, n2, info, k);
      const { vmin, vmax } = sliderRange(baseRange.lo, baseRange.hi, vminPct, vmaxPct);
      const rgba = new Uint8ClampedArray(info.rows * info.cols * 4);
      applyColormap(sliceData, rgba, COLORMAPS[cmap] || COLORMAPS.gray, vmin, vmax);
      const img = octx.createImageData(info.cols, info.rows);
      img.data.set(rgba);
      octx.putImageData(img, 0, 0);
    } else {
      octx.fillStyle = "#000";
      octx.fillRect(0, 0, info.cols, info.rows);
    }
    setSliceVersion((v) => v + 1);
  }, [volume, n1, n2, info, k, cmap, baseRange, vminPct, vmaxPct, showSlice]);

  // ---- Effect B: draw offscreen (viewport crop) + atom overlay + rubber band (light) ----
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  React.useEffect(() => {
    const canvas = canvasRef.current;
    const off = offRef.current;
    if (!canvas || !off) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvasW * dpr; canvas.height = canvasH * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, canvasW, canvasH);

    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, vp.col0, vp.row0, vp.col1 - vp.col0, vp.row1 - vp.row0, 0, 0, canvasW, canvasH);

    if (showSites && sites && numSites > 0) {
      const half = thickness / 2;
      const fade = Math.max(falloff, 1e-6);
      const sxv = canvasW / (vp.col1 - vp.col0);
      const syv = canvasH / (vp.row1 - vp.row0);
      for (let i = 0; i < numSites; i++) {
        const o = i * 5;
        const d = Math.abs(sites[o + info.normalAxis] - k);
        if (d > half + fade) continue;
        const opacity = d <= half ? 1 : Math.max(0, 1 - (d - half) / fade);
        if (opacity <= 0.01) continue;
        // +0.5 voxel so markers sit on pixel centers (drawImage maps voxel i to [i, i+1)).
        const x = (sites[o + info.colAxis] + 0.5 - vp.col0) * sxv;
        const y = (sites[o + info.rowAxis] + 0.5 - vp.row0) * syv;
        const r = MARKER_BASE * markerScale * (0.4 + 1.6 * Math.sqrt(Math.min(1, sites[o + 3] / maxIntensity)));
        if (x < -r || x > canvasW + r || y < -r || y > canvasH + r) continue;
        ctx.beginPath();
        ctx.arc(x, y, r, 0, 2 * Math.PI);
        ctx.lineWidth = markerLinewidth;
        if (markerFilled) {
          ctx.fillStyle = `rgba(255, 70, 70, ${0.85 * opacity})`;
          ctx.fill();
          if (markerLinewidth > 0) {
            ctx.strokeStyle = `rgba(255, 255, 255, ${0.5 * opacity})`;
            ctx.stroke();
          }
        } else {
          ctx.strokeStyle = `rgba(255, 70, 70, ${opacity})`;
          ctx.stroke();
        }
      }
    }

    if (dragBox) {
      ctx.lineWidth = 1;
      ctx.strokeStyle = tc.accent;
      ctx.setLineDash([4, 3]);
      ctx.strokeRect(
        Math.min(dragBox.x0, dragBox.x1), Math.min(dragBox.y0, dragBox.y1),
        Math.abs(dragBox.x1 - dragBox.x0), Math.abs(dragBox.y1 - dragBox.y0),
      );
      ctx.setLineDash([]);
    }
  }, [sliceVersion, viewport, vp, sites, numSites, info, k, showSites, thickness, falloff,
      markerScale, markerLinewidth, markerFilled, maxIntensity, canvasW, canvasH, dragBox, tc]);

  // ---- Mouse interaction: left=box zoom, middle=pan, dblclick=reset ----
  const relPx = (e: { clientX: number; clientY: number }) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };
  const onMouseDown = (e: React.MouseEvent) => {
    const p = relPx(e);
    if (e.button === 0) {
      setDrag({ mode: "zoom", sx: p.x, sy: p.y, startVp: vp });
      setDragBox({ x0: p.x, y0: p.y, x1: p.x, y1: p.y });
    } else if (e.button === 1) {
      e.preventDefault();
      setDrag({ mode: "pan", sx: p.x, sy: p.y, startVp: vp });
    }
  };
  React.useEffect(() => {
    if (!drag) return;
    const onMove = (e: MouseEvent) => {
      const p = relPx(e);
      if (drag.mode === "zoom") {
        setDragBox((b) => (b ? { ...b, x1: p.x, y1: p.y } : b));
      } else {
        const ext = drag.startVp;
        const dCol = ((p.x - drag.sx) / canvasW) * (ext.col1 - ext.col0);
        const dRow = ((p.y - drag.sy) / canvasH) * (ext.row1 - ext.row0);
        let c0 = ext.col0 - dCol, r0 = ext.row0 - dRow;
        const cw = ext.col1 - ext.col0, ch = ext.row1 - ext.row0;
        c0 = Math.max(0, Math.min(info.cols - cw, c0));
        r0 = Math.max(0, Math.min(info.rows - ch, r0));
        setViewport({ col0: c0, col1: c0 + cw, row0: r0, row1: r0 + ch });
      }
    };
    const onUp = () => {
      if (drag.mode === "zoom") {
        setDragBox((b) => {
          if (b && Math.abs(b.x1 - b.x0) > 4 && Math.abs(b.y1 - b.y0) > 4) {
            const toData = (x: number, y: number) => ({
              col: vp.col0 + (x / canvasW) * (vp.col1 - vp.col0),
              row: vp.row0 + (y / canvasH) * (vp.row1 - vp.row0),
            });
            const a = toData(b.x0, b.y0), c = toData(b.x1, b.y1);
            let col0 = Math.min(a.col, c.col), col1 = Math.max(a.col, c.col);
            let row0 = Math.min(a.row, c.row), row1 = Math.max(a.row, c.row);
            // Enforce min size + match canvas aspect (no pixel distortion).
            if (col1 - col0 < MIN_ZOOM_VOX) { const m = (col0 + col1) / 2; col0 = m - MIN_ZOOM_VOX / 2; col1 = m + MIN_ZOOM_VOX / 2; }
            if (row1 - row0 < MIN_ZOOM_VOX) { const m = (row0 + row1) / 2; row0 = m - MIN_ZOOM_VOX / 2; row1 = m + MIN_ZOOM_VOX / 2; }
            const aspect = canvasW / canvasH;
            let w = col1 - col0, h = row1 - row0;
            if (w / h > aspect) { const nh = w / aspect, m = (row0 + row1) / 2; row0 = m - nh / 2; row1 = m + nh / 2; }
            else { const nw = h * aspect, m = (col0 + col1) / 2; col0 = m - nw / 2; col1 = m + nw / 2; }
            col0 = Math.max(0, col0); col1 = Math.min(info.cols, col1);
            row0 = Math.max(0, row0); row1 = Math.min(info.rows, row1);
            setViewport({ row0, row1, col0, col1 });
          }
          return null;
        });
      }
      setDrag(null);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
  }, [drag, canvasW, canvasH, info, vp]);

  const resetAll = () => {
    setViewport(null);
    setPlane("xy");
    setSliceIndex(Math.floor(n2 / 2));
    setVminPct(0); setVmaxPct(100);
    setThickness(3); setFalloff(3);
    setMarkerScale(1); setMarkerLinewidth(1); setMarkerFilled(true);
    setShowSites(true); setShowSlice(true);
  };

  const fmt = (v: number) => (Math.abs(v) >= 1000 ? v.toExponential(1) : v.toFixed(1));
  const labelSx = { fontSize: 11, color: tc.text, whiteSpace: "nowrap" } as const;
  const valSx = { fontSize: 10, fontFamily: "monospace", color: tc.textMuted } as const;
  const selSx = { fontSize: 11, color: tc.text, bgcolor: tc.controlBg, "& .MuiSelect-select": { py: 0.4 } };
  const cr = sliderRange(baseRange.lo, baseRange.hi, vminPct, vmaxPct);

  return (
    <Box sx={{ p: 1.5, bgcolor: tc.bg, color: tc.text, width: "fit-content" }}>
      {title && <Typography sx={{ fontSize: 13, fontWeight: 600, mb: 0.5 }}>{title}</Typography>}
      <Box sx={{ display: "flex", gap: "12px", alignItems: "flex-start" }}>
        <Box>
          <canvas
            ref={canvasRef}
            onMouseDown={onMouseDown}
            onDoubleClick={resetAll}
            onContextMenu={(e) => e.preventDefault()}
            style={{ width: canvasW, height: canvasH, display: "block", cursor: drag?.mode === "pan" ? "grabbing" : "crosshair",
                     border: `1px solid ${tc.border}`, background: "#000" }}
          />
          <Box sx={{ mt: 0.5 }}>
            <canvas
              ref={(el) => {
                if (!el || !histBins) return;
                const ctx = el.getContext("2d"); if (!ctx) return;
                const W = canvasW, H = 46, dpr = window.devicePixelRatio || 1;
                el.width = W * dpr; el.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
                ctx.clearRect(0, 0, W, H);
                const nb = histBins.length, bw = W / nb;
                const lo = (vminPct / 100) * nb, hi = (vmaxPct / 100) * nb;
                for (let i = 0; i < nb; i++) {
                  const bh = histBins[i] * (H - 2);
                  ctx.fillStyle = i >= lo && i <= hi ? tc.accent : tc.textMuted;
                  ctx.fillRect(i * bw + 0.5, H - bh, Math.max(1, bw - 1), bh);
                }
              }}
              style={{ width: canvasW, height: 46, display: "block", border: `1px solid ${tc.border}` }}
            />
            <Slider
              value={[vminPct, vmaxPct]}
              onChange={(_, v) => { const [a, b] = v as number[]; setVminPct(Math.min(a, b - 1)); setVmaxPct(Math.max(b, a + 1)); }}
              min={0} max={100} size="small" sx={{ ...sliderStyles, width: canvasW }}
            />
            <Box sx={{ display: "flex", justifyContent: "space-between", width: canvasW }}>
              <Typography sx={valSx}>{fmt(cr.vmin)}</Typography>
              <Typography sx={valSx}>color range</Typography>
              <Typography sx={valSx}>{fmt(cr.vmax)}</Typography>
            </Box>
          </Box>
          <Typography sx={{ ...valSx, mt: 0.25 }}>drag: zoom · middle-drag: pan · double-click: reset</Typography>
        </Box>

        <Stack spacing={1.25} sx={{ minWidth: 200 }}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <Typography sx={labelSx}>plane</Typography>
            <Select value={plane} onChange={(e) => setPlane(e.target.value)} size="small" sx={selSx}>
              <MenuItem value="xy">xy</MenuItem><MenuItem value="xz">xz</MenuItem><MenuItem value="yz">yz</MenuItem>
            </Select>
            <Typography sx={labelSx}>cmap</Typography>
            <Select value={cmap} onChange={(e) => setCmap(e.target.value)} size="small" sx={selSx}>
              {COLORMAP_NAMES.map((nm) => <MenuItem key={nm} value={nm}>{nm}</MenuItem>)}
            </Select>
          </Box>
          <Box>
            <Typography sx={labelSx}>{info.normalLabel} slice: {k} / {info.depth - 1}</Typography>
            <Slider value={k} onChange={(_, v) => setSliceIndex(v as number)} min={0} max={Math.max(0, info.depth - 1)} size="small" sx={sliderStyles} />
          </Box>
          <Box>
            <Typography sx={labelSx}>slice thickness: {thickness.toFixed(1)} voxels</Typography>
            <Slider value={thickness} onChange={(_, v) => setThickness(v as number)} min={0.5} max={20} step={0.5} size="small" sx={sliderStyles} />
          </Box>
          <Box>
            <Typography sx={labelSx}>opacity falloff: {falloff.toFixed(1)} voxels</Typography>
            <Slider value={falloff} onChange={(_, v) => setFalloff(v as number)} min={0} max={20} step={0.5} size="small" sx={sliderStyles} />
          </Box>
          <Box>
            <Typography sx={labelSx}>marker scale: {markerScale.toFixed(2)}×</Typography>
            <Slider value={markerScale} onChange={(_, v) => setMarkerScale(v as number)} min={0.1} max={5} step={0.1} size="small" sx={sliderStyles} />
          </Box>
          <Box>
            <Typography sx={labelSx}>line width: {markerLinewidth.toFixed(2)}</Typography>
            <Slider value={markerLinewidth} onChange={(_, v) => setMarkerLinewidth(v as number)} min={0} max={4} step={0.25} size="small" sx={sliderStyles} />
          </Box>
          <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, flexWrap: "wrap" }}>
            <Switch checked={showSites} onChange={(e) => setShowSites(e.target.checked)} size="small" />
            <Typography sx={labelSx}>sites ({numSites})</Typography>
            <Switch checked={showSlice} onChange={(e) => setShowSlice(e.target.checked)} size="small" />
            <Typography sx={labelSx}>slice</Typography>
          </Box>
          <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
            <Switch checked={markerFilled} onChange={(e) => setMarkerFilled(e.target.checked)} size="small" />
            <Typography sx={labelSx}>filled markers ({markerFilled ? "filled" : "hollow"})</Typography>
          </Box>
        </Stack>
      </Box>
    </Box>
  );
}

export const render = createRender(Show3DAtoms);
