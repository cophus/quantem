/**
 * Pattern renderers for the simulator: nanobeam markers or pixels, CBED
 * disks, Kossel lines and the Kossel reference lookup. Every renderer works
 * in canvas coordinates with q_x to the right and q_y up.
 */

import { COLORMAPS, applyColormap } from "../colormaps";
import type { Reflection } from "./physics";
import type { KosselLine } from "./physics";

export interface Frame {
  size: number; // canvas CSS px (square)
  qMax: number; // 1/A at the edge (nanobeam / CBED) or rad (Kossel)
  viewX: number; // +1: seen from the gun side (lab x to the right); -1: from the detector side (mirrored)
}

export function scaleOf(f: Frame): number {
  return (0.5 * f.size * 0.92) / f.qMax; // px per unit
}

/** Lab (x, y) in pattern units to canvas px. */
export function toPx(f: Frame, x: number, y: number): [number, number] {
  const s = scaleOf(f);
  return [f.size / 2 + f.viewX * x * s, f.size / 2 - y * s];
}

/** Canvas px to lab (x, y) in pattern units. */
export function fromPx(f: Frame, px: number, py: number): [number, number] {
  const s = scaleOf(f);
  return [(f.viewX * (px - f.size / 2)) / s, -(py - f.size / 2) / s];
}

export function setupCanvas(canvas: HTMLCanvasElement, size: number): CanvasRenderingContext2D | null {
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== size * dpr || canvas.height !== size * dpr) {
    canvas.width = size * dpr;
    canvas.height = size * dpr;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

/** Nanobeam pattern as markers with area ~ sqrt(intensity). */
export function drawMarkers(
  ctx: CanvasRenderingContext2D, f: Frame, beams: Reflection[], inten: Float64Array, dark: boolean,
  labels: boolean, kinematic: boolean,
) {
  const s = scaleOf(f);
  ctx.fillStyle = dark ? "#000" : "#fff";
  ctx.fillRect(0, 0, f.size, f.size);
  let iMax = 0;
  for (let i = 0; i < beams.length; i++) if (beams[i].index >= 0 || !kinematic) iMax = Math.max(iMax, inten[i]);
  if (iMax <= 0) iMax = 1;
  // marker radius capped so neighbouring spots of the densest net do not merge
  let gMin = Infinity;
  for (const b of beams) if (b.index >= 0 && b.gLen > 1e-6) gMin = Math.min(gMin, b.gLen);
  const rMax = Math.min(0.055 * f.size, isFinite(gMin) ? 0.42 * gMin * s : Infinity);
  const fg = dark ? "#fff" : "#000";
  const strong: { x: number; y: number; r: number; hkl: number[] }[] = [];
  for (let i = 0; i < beams.length; i++) {
    const b = beams[i];
    const rel = inten[i] / iMax;
    const [x, y] = toPx(f, b.g[0], b.g[1]);
    if (b.index < 0 && kinematic) {
      ctx.strokeStyle = fg; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(x, y, rMax * 0.9, 0, 2 * Math.PI); ctx.stroke();
      strong.push({ x, y, r: rMax * 0.9, hkl: b.hkl });
      continue;
    }
    if (rel < 1e-6) continue;
    const r = rMax * Math.pow(rel, 0.25);
    ctx.fillStyle = fg;
    ctx.globalAlpha = 0.9;
    ctx.beginPath(); ctx.arc(x, y, Math.max(r, 0.6), 0, 2 * Math.PI); ctx.fill();
    ctx.globalAlpha = 1;
    if (rel > 0.08) strong.push({ x, y, r, hkl: b.hkl });
  }
  if (labels) {
    ctx.font = `${Math.max(9, Math.round(f.size / 38))}px sans-serif`;
    ctx.textAlign = "center"; ctx.textBaseline = "bottom";
    ctx.fillStyle = dark ? "#ffd54f" : "#c62828";
    for (const p of strong.slice(0, 40)) {
      ctx.fillText(hklText(p.hkl), p.x, p.y - p.r - 2);
    }
  }
  // scale bar of 1 1/A
  drawScaleBar(ctx, f, s, "Å⁻¹", dark, 1);
}

export function hklText(hkl: number[]): string {
  return hkl.map((h) => (h < 0 ? `${-h}̅` : `${h}`)).join("");
}

function drawScaleBar(ctx: CanvasRenderingContext2D, f: Frame, s: number, unit: string, dark: boolean, value: number) {
  let v = value;
  while (v * s > 0.4 * f.size) v /= 2;
  while (v * s < 0.12 * f.size) v *= 2;
  const L = v * s;
  const x0 = f.size - L - 14, y0 = f.size - 14;
  ctx.strokeStyle = dark ? "#eee" : "#222";
  ctx.fillStyle = dark ? "#eee" : "#222";
  ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0 + L, y0); ctx.stroke();
  ctx.font = "11px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "bottom";
  const label = v >= 1 ? `${+v.toFixed(2)} ${unit}` : `${+(v * 1000).toFixed(0)} m${unit}`;
  ctx.fillText(label, x0 + L / 2, y0 - 4);
}

/** Splat beams as Gaussian spots into a float image (canvas order). */
export function nanobeamImage(f: Frame, beams: Reflection[], inten: Float64Array, spotPx: number): Float32Array {
  const n = f.size;
  const img = new Float32Array(n * n);
  const sig = spotPx;
  const w = Math.ceil(3.5 * sig);
  for (let i = 0; i < beams.length; i++) {
    if (inten[i] <= 0) continue;
    const [x, y] = toPx(f, beams[i].g[0], beams[i].g[1]);
    const amp = inten[i] / (2 * Math.PI * sig * sig);
    const x0 = Math.max(0, Math.floor(x - w)), x1 = Math.min(n - 1, Math.ceil(x + w));
    const y0 = Math.max(0, Math.floor(y - w)), y1 = Math.min(n - 1, Math.ceil(y + w));
    for (let py = y0; py <= y1; py++) {
      const dy = py + 0.5 - y;
      for (let px = x0; px <= x1; px++) {
        const dx = px + 0.5 - x;
        img[py * n + px] += amp * Math.exp(-(dx * dx + dy * dy) / (2 * sig * sig));
      }
    }
  }
  return img;
}

export interface TiltGrid {
  n: number; // grid points per side
  h: number; // spacing, 1/A
  R: number; // disk radius, 1/A
  tilts: [number, number][]; // grid points within R + h (row-major over the n x n grid, NaN-free)
  index: Int32Array; // n*n -> position in tilts or -1
}

export function tiltGrid(R: number, n: number): TiltGrid {
  const h = (2 * R) / (n - 1);
  const tilts: [number, number][] = [];
  const index = new Int32Array(n * n).fill(-1);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const tx = -R + i * h, ty = -R + j * h;
      if (Math.hypot(tx, ty) <= R + 1.01 * h) {
        index[j * n + i] = tilts.length;
        tilts.push([tx, ty]);
      }
    }
  }
  return { n, h, R, tilts, index };
}

/**
 * CBED image: for every beam a disk of radius R centered on g, with the
 * intensity at each pixel interpolated bilinearly from the tilt grid.
 * intensities[t][b] is the intensity of beam b at tilt t.
 */
export function cbedImage(f: Frame, beams: Reflection[], grid: TiltGrid, intensities: Float64Array[]): Float32Array {
  const n = f.size;
  const img = new Float32Array(n * n);
  const s = scaleOf(f);
  const Rpx = grid.R * s;
  const lookup = (b: number, tx: number, ty: number): number => {
    const fx = (tx + grid.R) / grid.h, fy = (ty + grid.R) / grid.h;
    const i0 = Math.min(Math.max(Math.floor(fx), 0), grid.n - 2);
    const j0 = Math.min(Math.max(Math.floor(fy), 0), grid.n - 2);
    const wx = Math.min(Math.max(fx - i0, 0), 1), wy = Math.min(Math.max(fy - j0, 0), 1);
    const v = (i: number, j: number) => {
      const k = grid.index[j * grid.n + i];
      return k < 0 ? 0 : intensities[k][b];
    };
    return v(i0, j0) * (1 - wx) * (1 - wy) + v(i0 + 1, j0) * wx * (1 - wy) + v(i0, j0 + 1) * (1 - wx) * wy + v(i0 + 1, j0 + 1) * wx * wy;
  };
  for (let b = 0; b < beams.length; b++) {
    const [cx, cy] = toPx(f, beams[b].g[0], beams[b].g[1]);
    if (cx < -Rpx || cy < -Rpx || cx > n + Rpx || cy > n + Rpx) continue;
    let anyInt = 0;
    for (const arr of intensities) if (arr[b] > 1e-7) { anyInt = 1; break; }
    if (!anyInt) continue;
    const x0 = Math.max(0, Math.floor(cx - Rpx - 1)), x1 = Math.min(n - 1, Math.ceil(cx + Rpx + 1));
    const y0 = Math.max(0, Math.floor(cy - Rpx - 1)), y1 = Math.min(n - 1, Math.ceil(cy + Rpx + 1));
    for (let py = y0; py <= y1; py++) {
      const dy = (py + 0.5 - cy);
      for (let px = x0; px <= x1; px++) {
        const dx = (px + 0.5 - cx);
        const rr = Math.hypot(dx, dy);
        if (rr > Rpx + 0.5) continue;
        const edge = Math.min(1, Rpx + 0.5 - rr); // anti-aliased rim
        const tx = (f.viewX * dx) / s, ty = -dy / s;
        img[py * n + px] += edge * lookup(b, tx, ty);
      }
    }
  }
  return img;
}

/** Colormapped float image onto the canvas, with a percentile contrast window. */
export function drawImage(
  ctx: CanvasRenderingContext2D, f: Frame, img: Float32Array, cmap: string, vmin: number, vmax: number,
  dark: boolean, unit: string, barValue: number,
) {
  const n = f.size;
  const lut = COLORMAPS[cmap] || COLORMAPS[Object.keys(COLORMAPS)[0]];
  const rgba = new Uint8ClampedArray(n * n * 4);
  const clean = new Float32Array(n * n);
  for (let i = 0; i < n * n; i++) clean[i] = isFinite(img[i]) ? img[i] : vmin;
  applyColormap(clean, rgba, lut, vmin, vmax);
  for (let i = 0; i < n * n; i++) if (!isFinite(img[i])) rgba[4 * i + 3] = 0;
  const off = document.createElement("canvas");
  off.width = n; off.height = n;
  const octx = off.getContext("2d");
  if (!octx) return;
  octx.putImageData(new ImageData(rgba, n, n), 0, 0);
  ctx.fillStyle = dark ? "#000" : "#fff";
  ctx.fillRect(0, 0, n, n);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, n, n);
  drawScaleBar(ctx, f, scaleOf(f), unit, dark, barValue);
}

/** Bright field Kossel pattern as vector lines (deficient lines dark). */
export function drawKosselLines(
  ctx: CanvasRenderingContext2D, f: Frame, lines: KosselLine[], dark: boolean, labels: boolean, minStrength: number,
) {
  const n = f.size, s = scaleOf(f);
  const cx = n / 2, cy = n / 2, Rpx = f.qMax * s;
  ctx.fillStyle = dark ? "#000" : "#fff";
  ctx.fillRect(0, 0, n, n);
  ctx.save();
  ctx.beginPath(); ctx.arc(cx, cy, Rpx, 0, 2 * Math.PI); ctx.clip();
  ctx.fillStyle = dark ? "#bdbdbd" : "#e0e0e0";
  ctx.fill();
  const sorted = [...lines].filter((l) => l.strength >= minStrength).sort((a, b) => a.strength - b.strength);
  const L = 2 * f.qMax;
  for (const l of sorted) {
    const [nx, ny] = l.normal;
    // line p . n = distance, direction t = (-ny, nx); drawn between two lab points
    const [x0, y0] = toPx(f, nx * l.distance - ny * L, ny * l.distance + nx * L);
    const [x1, y1] = toPx(f, nx * l.distance + ny * L, ny * l.distance - nx * L);
    ctx.strokeStyle = dark ? `rgba(20,20,20,${0.25 + 0.75 * l.strength})` : `rgba(30,30,30,${0.2 + 0.8 * l.strength})`;
    ctx.lineWidth = Math.max(1, l.width * s);
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
  }
  if (labels) {
    ctx.font = `${Math.max(9, Math.round(n / 40))}px sans-serif`;
    ctx.fillStyle = dark ? "#ffd54f" : "#c62828";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    const strong = sorted.filter((l) => l.strength > 0.3 && Math.abs(l.distance) < f.qMax * 0.95).slice(-24);
    for (const l of strong) {
      const [nx, ny] = l.normal;
      // label near the rim along the line
      const t = Math.sqrt(Math.max(f.qMax * f.qMax * 0.8 - l.distance * l.distance, 0));
      const [x, y] = toPx(f, nx * l.distance - ny * t, ny * l.distance + nx * t);
      ctx.fillText(hklText(l.hkl), x, y);
    }
  }
  ctx.restore();
  drawScaleBar(ctx, f, s, "rad", dark, 0.01);
}

/** Kikuchi line pairs overlaid on a nanobeam pattern (deficient dark, excess bright). */
export function drawKikuchiOverlay(ctx: CanvasRenderingContext2D, f: Frame, lines: KosselLine[], k0: number, dark: boolean) {
  const L = 3 * f.qMax;
  for (const l of lines) {
    if (l.strength < 0.25) continue;
    const d = l.distance * k0; // 1/A
    const [nx, ny] = l.normal;
    const draw = (dist: number, color: string) => {
      const [x0, y0] = toPx(f, nx * dist - ny * L, ny * dist + nx * L);
      const [x1, y1] = toPx(f, nx * dist + ny * L, ny * dist - nx * L);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    };
    draw(d, dark ? `rgba(120,170,255,${0.3 + 0.5 * l.strength})` : `rgba(30,90,200,${0.3 + 0.5 * l.strength})`);
    draw(d + l.gxy, dark ? `rgba(255,140,120,${0.3 + 0.5 * l.strength})` : `rgba(200,60,40,${0.3 + 0.5 * l.strength})`);
  }
}

/** 256-bin histogram of the finite values. */
export function histogramBins(img: Float32Array, lo: number, hi: number): number[] {
  const bins = new Array(256).fill(0);
  const range = hi > lo ? hi - lo : 1;
  for (let i = 0; i < img.length; i++) {
    const v = img[i];
    if (!isFinite(v)) continue;
    const b = Math.min(255, Math.max(0, Math.floor(((v - lo) / range) * 255)));
    bins[b]++;
  }
  return bins;
}
