"""Interactive 3D viewer for :class:`~quantem.atoms.AtomicModel`.

:class:`ShowAtoms3D` is a self-contained `anywidget <https://anywidget.dev>`_
(no JavaScript build step) that renders all sites of a model in a notebook:

* drag to rotate, scroll to zoom, shift-drag (or right-drag) to pan,
  double-click to reset;
* a **channel** dropdown colors sites by any per-site channel, with a live
  histogram whose two handles set the color range;
* **clip** controls cut a slab along the view direction or a model axis so
  flat atomic planes can be isolated, and "hide outside range" keeps only sites
  whose channel value lies inside the histogram range (e.g. only twin sites);
* the current view is synced back to Python (:attr:`ShowAtoms3D.view_matrix`)
  so the same projection can be reproduced with ``model.plot("slab", normal=...)``.

The widget is optional: it requires the ``anywidget`` package.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    import anywidget
    import traitlets
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "ShowAtoms3D requires the 'anywidget' package: pip install anywidget"
    ) from exc

from quantem.atoms.visualization import _CATEGORICAL_COLORS

__all__ = ["ShowAtoms3D"]

_DEFAULT_CMAPS = [
    "viridis",
    "turbo",
    "inferno",
    "magma",
    "plasma",
    "cividis",
    "coolwarm",
    "RdBu_r",
    "gray",
]

_ESM = r"""
function mat3mul(a, b) {
  const o = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    o[3*i+j] = a[3*i]*b[j] + a[3*i+1]*b[3+j] + a[3*i+2]*b[6+j];
  }
  return o;
}
function rotX(t){const c=Math.cos(t),s=Math.sin(t);return [1,0,0,0,c,-s,0,s,c];}
function rotY(t){const c=Math.cos(t),s=Math.sin(t);return [c,0,s,0,1,0,-s,0,c];}
function orthonormalize(m){
  let r=[m[0],m[1],m[2]], u=[m[3],m[4],m[5]];
  const nr=Math.hypot(...r); r=r.map(v=>v/nr);
  const d=u[0]*r[0]+u[1]*r[1]+u[2]*r[2]; u=[u[0]-d*r[0],u[1]-d*r[1],u[2]-d*r[2]];
  const nu=Math.hypot(...u); u=u.map(v=>v/nu);
  const n=[r[1]*u[2]-r[2]*u[1], r[2]*u[0]-r[0]*u[2], r[0]*u[1]-r[1]*u[0]];
  return [...r,...u,...n];
}

function render({ model, el }) {
  // ---------------- data ----------------
  const N = model.get("num_sites");
  const pos = new Float32Array(model.get("positions").buffer.slice(0));
  const channelNames = model.get("channel_names");
  const chanData = new Float32Array(model.get("channel_data").buffer.slice(0));
  const categories = model.get("channel_categories");
  const palettes = model.get("channel_palettes");
  const cmapNames = model.get("cmap_names");
  const luts = new Uint8Array(model.get("cmap_luts").buffer.slice(0));
  const radius0 = model.get("model_radius");
  const units = model.get("units");

  const proj = new Float32Array(N * 3);
  const order = new Uint32Array(N);
  const depthKey = new Float32Array(N);
  const visible = new Uint8Array(N);
  let spriteCache = new Map();
  let spriteRadiusPx = -1;
  let dragging = false, dragMode = 0, lastX = 0, lastY = 0, moved = false;
  let hist = null, histDrag = 0;

  // ---------------- layout ----------------
  el.innerHTML = "";
  const root = document.createElement("div"); root.className = "qa-root"; el.appendChild(root);
  const left = document.createElement("div"); left.className = "qa-left"; root.appendChild(left);
  const title = document.createElement("div"); title.className = "qa-title"; left.appendChild(title);
  const canvas = document.createElement("canvas"); canvas.className = "qa-canvas"; left.appendChild(canvas);
  const status = document.createElement("div"); status.className = "qa-status"; left.appendChild(status);
  const panel = document.createElement("div"); panel.className = "qa-panel"; root.appendChild(panel);
  const ctx = canvas.getContext("2d");

  function row(label) {
    const r = document.createElement("div"); r.className = "qa-row";
    const l = document.createElement("label"); l.textContent = label; r.appendChild(l);
    panel.appendChild(r); return r;
  }
  function select(label, options, key, fmt) {
    const r = row(label); const s = document.createElement("select");
    options.forEach(o => { const op = document.createElement("option"); op.value = o; op.textContent = fmt ? fmt(o) : o; s.appendChild(op); });
    s.value = model.get(key);
    s.onchange = () => { model.set(key, s.value); model.save_changes(); };
    r.appendChild(s); return s;
  }
  function slider(label, key, min, max, step, live) {
    const r = row(label); const s = document.createElement("input"); s.type = "range";
    s.min = min; s.max = max; s.step = step; s.value = model.get(key);
    const v = document.createElement("span"); v.className = "qa-val"; v.textContent = Number(model.get(key)).toFixed(2);
    s.oninput = () => { v.textContent = Number(s.value).toFixed(2); model.set(key, Number(s.value)); if (live) draw(); };
    s.onchange = () => { model.set(key, Number(s.value)); model.save_changes(); };
    r.appendChild(s); r.appendChild(v); return { s, v };
  }
  function checkbox(label, key) {
    const r = row(label); const c = document.createElement("input"); c.type = "checkbox"; c.checked = model.get(key);
    c.onchange = () => { model.set(key, c.checked); model.save_changes(); };
    r.appendChild(c); return c;
  }
  function button(parent, text, fn) {
    const b = document.createElement("button"); b.className = "qa-btn"; b.textContent = text; b.onclick = fn; parent.appendChild(b); return b;
  }

  const channelSel = select("channel", channelNames, "channel");
  const cmapSel = select("colormap", cmapNames, "cmap");
  const histCanvas = document.createElement("canvas"); histCanvas.className = "qa-hist"; histCanvas.width = 240; histCanvas.height = 96; panel.appendChild(histCanvas);
  const hctx = histCanvas.getContext("2d");
  const rangeRow = row("range");
  const vminIn = document.createElement("input"); vminIn.type = "number"; vminIn.className = "qa-num"; vminIn.step = "any";
  const vmaxIn = document.createElement("input"); vmaxIn.type = "number"; vmaxIn.className = "qa-num"; vmaxIn.step = "any";
  rangeRow.appendChild(vminIn); rangeRow.appendChild(vmaxIn);
  vminIn.onchange = () => { model.set("vmin", Number(vminIn.value)); model.set("range_auto", false); model.save_changes(); };
  vmaxIn.onchange = () => { model.set("vmax", Number(vmaxIn.value)); model.set("range_auto", false); model.save_changes(); };
  const rr = row(""); button(rr, "auto range", () => { model.set("range_auto", true); model.save_changes(); autoRange(); });
  const hideChk = checkbox("hide outside range", "hide_outside_range");
  const legend = document.createElement("div"); legend.className = "qa-legend"; panel.appendChild(legend);

  const sep1 = document.createElement("div"); sep1.className = "qa-sep"; sep1.textContent = "clip"; panel.appendChild(sep1);
  const clipChk = checkbox("enable clip", "clip_enabled");
  const clipAxis = select("axis", ["view", "x", "y", "z"], "clip_axis");
  const clipCenter = slider("center", "clip_center", -radius0, radius0, radius0 / 400, true);
  const clipThick = slider("thickness", "clip_thickness", 0.1, 2 * radius0, radius0 / 400, true);

  const sep2 = document.createElement("div"); sep2.className = "qa-sep"; sep2.textContent = "display"; panel.appendChild(sep2);
  const sizeSl = slider("marker", "marker_size", 0.05, 2.0, 0.01, true);
  const cueSl = slider("depth cue", "depth_cue", 0, 1, 0.01, true);
  const edgeChk = checkbox("edges", "show_edges");
  const bgChk = checkbox("dark background", "dark_background");
  const viewRow = row("view");
  button(viewRow, "x", () => setView([0,1,0, 0,0,1, 1,0,0]));
  button(viewRow, "y", () => setView([0,0,1, 1,0,0, 0,1,0]));
  button(viewRow, "z", () => setView([1,0,0, 0,1,0, 0,0,1]));
  button(viewRow, "reset", () => { setView([1,0,0, 0,1,0, 0,0,1]); model.set("zoom", 1.0); model.set("pan", [0, 0]); model.save_changes(); });
  const info = document.createElement("div"); info.className = "qa-info"; panel.appendChild(info);

  function setView(m) { model.set("rotation", Array.from(m)); model.save_changes(); draw(); }

  // ---------------- channel helpers ----------------
  function channelIndex() { const i = channelNames.indexOf(model.get("channel")); return i < 0 ? 0 : i; }
  function channelValues() { const i = channelIndex(); return chanData.subarray(i * N, (i + 1) * N); }
  function isCategorical() { return categories[model.get("channel")] !== undefined; }
  function autoRange() {
    const v = channelValues(); const name = model.get("channel");
    if (isCategorical()) { model.set("vmin", -0.5); model.set("vmax", categories[name].length - 0.5); model.save_changes(); return; }
    const arr = Array.from(v).filter(Number.isFinite).sort((a, b) => a - b);
    if (!arr.length) return;
    const lo = arr[Math.floor(0.01 * (arr.length - 1))], hi = arr[Math.floor(0.99 * (arr.length - 1))];
    model.set("vmin", lo); model.set("vmax", hi === lo ? lo + 1e-9 : hi); model.save_changes();
  }
  function computeHist() {
    const v = channelValues(); const nb = 64;
    let lo, hi;
    if (isCategorical()) { lo = -1.5; hi = categories[model.get("channel")].length - 0.5; }
    else {
      let mn = Infinity, mx = -Infinity;
      for (let i = 0; i < N; i++) { const x = v[i]; if (Number.isFinite(x)) { if (x < mn) mn = x; if (x > mx) mx = x; } }
      lo = mn; hi = mx === mn ? mn + 1e-9 : mx;
    }
    const counts = new Float64Array(nb);
    for (let i = 0; i < N; i++) { const x = v[i]; if (!Number.isFinite(x)) continue; let b = Math.floor((x - lo) / (hi - lo) * nb); if (b >= nb) b = nb - 1; if (b < 0) b = 0; counts[b]++; }
    hist = { lo, hi, counts, nb };
  }
  function lutColor(t) {
    const ci = Math.max(0, cmapNames.indexOf(model.get("cmap")));
    const k = Math.max(0, Math.min(255, Math.round(t * 255)));
    const o = (ci * 256 + k) * 3; return [luts[o], luts[o + 1], luts[o + 2]];
  }
  function drawHist() {
    if (!hist) computeHist();
    const W = histCanvas.width, H = histCanvas.height, pad = 4, hh = H - 22;
    const dark = model.get("dark_background");
    hctx.fillStyle = dark ? "#1e1e1e" : "#f4f4f4"; hctx.fillRect(0, 0, W, H);
    const vmin = model.get("vmin"), vmax = model.get("vmax");
    const cmax = Math.max(1, ...hist.counts);
    const cat = isCategorical(); const name = model.get("channel");
    for (let b = 0; b < hist.nb; b++) {
      const x0 = pad + (W - 2 * pad) * b / hist.nb, w = (W - 2 * pad) / hist.nb;
      const val = hist.lo + (b + 0.5) / hist.nb * (hist.hi - hist.lo);
      let t = (val - vmin) / (vmax - vmin);
      let col;
      if (cat) { const code = Math.round(val); const p = palettes[name]; col = code < 0 || !p ? [140,140,140] : p.slice(3*(code % (p.length/3)), 3*(code % (p.length/3))+3); }
      else col = lutColor(Math.max(0, Math.min(1, t)));
      const h = Math.log1p(hist.counts[b]) / Math.log1p(cmax) * hh;
      hctx.fillStyle = `rgb(${col[0]},${col[1]},${col[2]})`;
      hctx.globalAlpha = (val < vmin || val > vmax) ? 0.3 : 1.0;
      hctx.fillRect(x0, pad + hh - h, Math.max(w - 1, 1), h);
    }
    hctx.globalAlpha = 1;
    // colorbar
    for (let i = 0; i < W - 2 * pad; i++) {
      const val = hist.lo + i / (W - 2 * pad) * (hist.hi - hist.lo);
      const t = Math.max(0, Math.min(1, (val - vmin) / (vmax - vmin)));
      let col;
      if (cat) { const code = Math.round(val); const p = palettes[name]; col = code < 0 || !p ? [140,140,140] : p.slice(3*(code % (p.length/3)), 3*(code % (p.length/3))+3); }
      else col = lutColor(t);
      hctx.fillStyle = `rgb(${col[0]},${col[1]},${col[2]})`; hctx.fillRect(pad + i, pad + hh + 4, 1, 8);
    }
    // handles
    const xs = [vmin, vmax].map(v => pad + (W - 2 * pad) * (v - hist.lo) / (hist.hi - hist.lo));
    hctx.strokeStyle = dark ? "#fff" : "#000"; hctx.lineWidth = 1.5;
    xs.forEach(x => { hctx.beginPath(); hctx.moveTo(x, pad); hctx.lineTo(x, H - 2); hctx.stroke(); });
    vminIn.value = Number(vmin.toPrecision(5)); vmaxIn.value = Number(vmax.toPrecision(5));
    // legend
    legend.innerHTML = "";
    if (cat) {
      const p = palettes[name] || [];
      categories[name].forEach((lab, i) => {
        const d = document.createElement("div"); d.className = "qa-legend-item";
        const sw = document.createElement("span"); sw.className = "qa-swatch"; sw.style.background = `rgb(${p[3*i]},${p[3*i+1]},${p[3*i+2]})`;
        d.appendChild(sw); d.appendChild(document.createTextNode(`${i}: ${lab}`)); legend.appendChild(d);
      });
      const d = document.createElement("div"); d.className = "qa-legend-item";
      const sw = document.createElement("span"); sw.className = "qa-swatch"; sw.style.background = "rgb(140,140,140)";
      d.appendChild(sw); d.appendChild(document.createTextNode("-1: none")); legend.appendChild(d);
    }
  }
  function histPointer(ev) {
    const rect = histCanvas.getBoundingClientRect(); const pad = 4, W = histCanvas.width;
    const x = (ev.clientX - rect.left) * (W / rect.width);
    return hist.lo + (x - pad) / (W - 2 * pad) * (hist.hi - hist.lo);
  }
  histCanvas.onpointerdown = (ev) => {
    if (!hist) computeHist();
    const v = histPointer(ev); const vmin = model.get("vmin"), vmax = model.get("vmax");
    histDrag = Math.abs(v - vmin) < Math.abs(v - vmax) ? 1 : 2; histCanvas.setPointerCapture(ev.pointerId);
  };
  histCanvas.onpointermove = (ev) => {
    if (!histDrag) return; const v = histPointer(ev);
    if (histDrag === 1) model.set("vmin", Math.min(v, model.get("vmax") - 1e-9)); else model.set("vmax", Math.max(v, model.get("vmin") + 1e-9));
    model.set("range_auto", false); drawHist(); draw();
  };
  histCanvas.onpointerup = (ev) => { histDrag = 0; histCanvas.releasePointerCapture(ev.pointerId); model.save_changes(); };

  // ---------------- sprites ----------------
  function sprite(r, g, b, shade, radiusPx, edges) {
    const key = ((r << 16) | (g << 8) | b) * 32 + shade;
    let s = spriteCache.get(key); if (s) return s;
    const R = Math.max(1, Math.ceil(radiusPx)); const size = 2 * R + 2;
    s = document.createElement("canvas"); s.width = size; s.height = size;
    const c = s.getContext("2d");
    const f = 1 - 0.85 * shade / 31;
    const cx = R + 1, cy = R + 1;
    const grad = c.createRadialGradient(cx - 0.35 * R, cy - 0.35 * R, 0.1 * R, cx, cy, R);
    grad.addColorStop(0, `rgb(${Math.min(255, r*f+90*f)|0},${Math.min(255, g*f+90*f)|0},${Math.min(255, b*f+90*f)|0})`);
    grad.addColorStop(1, `rgb(${(r*f*0.75)|0},${(g*f*0.75)|0},${(b*f*0.75)|0})`);
    c.fillStyle = grad; c.beginPath(); c.arc(cx, cy, R, 0, 2 * Math.PI); c.fill();
    if (edges && R > 2) { c.strokeStyle = "rgba(0,0,0,0.6)"; c.lineWidth = Math.max(0.5, R * 0.12); c.stroke(); }
    spriteCache.set(key, s); return s;
  }

  // ---------------- main draw ----------------
  function draw() {
    const W = model.get("canvas_size"), H = W;
    if (canvas.width !== W) { canvas.width = W; canvas.height = H; }
    const dark = model.get("dark_background");
    ctx.fillStyle = dark ? "#111" : "#fff"; ctx.fillRect(0, 0, W, H);
    const m = model.get("rotation"); const zoom = model.get("zoom"); const pan = model.get("pan");
    const scale = zoom * (0.5 * W * 0.92) / radius0;
    const cx = W / 2 + pan[0], cy = H / 2 + pan[1];
    const vals = channelValues(); const vmin = model.get("vmin"), vmax = model.get("vmax");
    const cat = isCategorical(); const name = model.get("channel"); const pal = palettes[name];
    const hide = model.get("hide_outside_range");
    const clipOn = model.get("clip_enabled"), clipAx = model.get("clip_axis");
    const cc = model.get("clip_center"), ct = model.get("clip_thickness");
    let count = 0;
    let dmin = Infinity, dmax = -Infinity;
    for (let i = 0; i < N; i++) {
      const x = pos[3*i], y = pos[3*i+1], z = pos[3*i+2];
      const u = m[0]*x + m[1]*y + m[2]*z, v = m[3]*x + m[4]*y + m[5]*z, d = m[6]*x + m[7]*y + m[8]*z;
      proj[3*i] = cx + u * scale; proj[3*i+1] = cy - v * scale; proj[3*i+2] = d;
      let ok = 1;
      if (clipOn) { const c = clipAx === "view" ? d : (clipAx === "x" ? x : (clipAx === "y" ? y : z)); if (Math.abs(c - cc) > ct / 2) ok = 0; }
      if (ok && hide) { const val = vals[i]; if (!(val >= vmin && val <= vmax)) ok = 0; }
      visible[i] = ok;
      if (ok) { if (d < dmin) dmin = d; if (d > dmax) dmax = d; order[count++] = i; }
    }
    const sub = order.subarray(0, count);
    for (let k = 0; k < count; k++) depthKey[sub[k]] = proj[3*sub[k]+2];
    sub.sort((a, b) => depthKey[a] - depthKey[b]);
    const radiusPx = Math.max(0.6, model.get("marker_size") * model.get("bond_length") * 0.5 * scale);
    if (radiusPx !== spriteRadiusPx) { spriteCache = new Map(); spriteRadiusPx = radiusPx; }
    const edges = model.get("show_edges"); const cue = model.get("depth_cue");
    const dr = Math.max(dmax - dmin, 1e-9);
    const inv = 1 / Math.max(vmax - vmin, 1e-12);
    const ci = Math.max(0, cmapNames.indexOf(model.get("cmap")));
    for (let k = 0; k < count; k++) {
      const i = sub[k]; const val = vals[i];
      let r, g, b;
      if (cat) { const code = Math.round(val); if (code < 0 || !pal || !Number.isFinite(val)) { r = 140; g = 140; b = 140; } else { const j = 3 * (code % (pal.length / 3)); r = pal[j]; g = pal[j+1]; b = pal[j+2]; } }
      else if (!Number.isFinite(val)) { r = 140; g = 140; b = 140; }
      else { let t = (val - vmin) * inv; t = t < 0 ? 0 : (t > 1 ? 1 : t); const o = (ci * 256 + Math.round(t * 255)) * 3; r = luts[o]; g = luts[o+1]; b = luts[o+2]; }
      const t = (proj[3*i+2] - dmin) / dr;
      const shade = Math.round(cue * (1 - t) * 31);
      const s = sprite(r, g, b, shade, radiusPx, edges);
      ctx.drawImage(s, proj[3*i] - s.width / 2, proj[3*i+1] - s.height / 2);
    }
    status.textContent = `${count} / ${N} sites shown`;
  }

  // ---------------- interaction ----------------
  canvas.oncontextmenu = (e) => e.preventDefault();
  canvas.onpointerdown = (ev) => {
    dragging = true; moved = false; lastX = ev.clientX; lastY = ev.clientY;
    dragMode = (ev.button === 2 || ev.shiftKey) ? 2 : 1; canvas.setPointerCapture(ev.pointerId);
  };
  canvas.onpointermove = (ev) => {
    if (!dragging) return;
    const dx = ev.clientX - lastX, dy = ev.clientY - lastY; lastX = ev.clientX; lastY = ev.clientY;
    if (Math.abs(dx) + Math.abs(dy) > 0) moved = true;
    if (dragMode === 2) { const p = model.get("pan"); model.set("pan", [p[0] + dx, p[1] + dy]); }
    else {
      const m = model.get("rotation"); const k = 0.008;
      const rs = mat3mul(rotX(-dy * k), rotY(dx * k));
      model.set("rotation", Array.from(orthonormalize(mat3mul(rs, m))));
    }
    draw();
  };
  canvas.onpointerup = (ev) => {
    dragging = false; canvas.releasePointerCapture(ev.pointerId);
    if (!moved) pick(ev); model.save_changes();
  };
  canvas.onwheel = (ev) => { ev.preventDefault(); const z = model.get("zoom") * Math.exp(-ev.deltaY * 0.0015); model.set("zoom", Math.max(0.05, Math.min(50, z))); draw(); model.save_changes(); };
  canvas.ondblclick = () => { model.set("zoom", 1.0); model.set("pan", [0, 0]); model.save_changes(); draw(); };
  function pick(ev) {
    const rect = canvas.getBoundingClientRect();
    const x = (ev.clientX - rect.left) * canvas.width / rect.width, y = (ev.clientY - rect.top) * canvas.height / rect.height;
    let best = -1, bd = 1e9;
    for (let i = 0; i < N; i++) { if (!visible[i]) continue; const dx = proj[3*i] - x, dy = proj[3*i+1] - y; const d2 = dx*dx + dy*dy - 0.02 * proj[3*i+2]; if (d2 < bd) { bd = d2; best = i; } }
    if (best >= 0 && Math.sqrt(bd + 1) < 20) {
      model.set("selected", best); model.save_changes();
      const v = channelValues()[best]; const name = model.get("channel");
      const lab = isCategorical() ? (v >= 0 ? categories[name][Math.round(v)] : "none") : v.toPrecision(4);
      info.textContent = `site ${best}: ${name} = ${lab}  (${pos[3*best].toFixed(2)}, ${pos[3*best+1].toFixed(2)}, ${pos[3*best+2].toFixed(2)}) ${units}`;
    }
  }

  // ---------------- model listeners ----------------
  model.on("change:channel", () => { channelSel.value = model.get("channel"); hist = null; if (model.get("range_auto")) autoRange(); drawHist(); draw(); });
  model.on("change:cmap", () => { cmapSel.value = model.get("cmap"); spriteCache = new Map(); drawHist(); draw(); });
  ["vmin", "vmax"].forEach(k => model.on("change:" + k, () => { drawHist(); draw(); }));
  ["rotation", "zoom", "pan", "marker_size", "depth_cue", "show_edges", "clip_enabled", "clip_axis", "clip_center", "clip_thickness", "hide_outside_range", "canvas_size"].forEach(k => model.on("change:" + k, () => {
    if (k === "clip_axis") clipAxis.value = model.get(k);
    if (k === "clip_enabled") clipChk.checked = model.get(k);
    if (k === "hide_outside_range") hideChk.checked = model.get(k);
    if (k === "show_edges") edgeChk.checked = model.get(k);
    if (k === "clip_center") { clipCenter.s.value = model.get(k); clipCenter.v.textContent = Number(model.get(k)).toFixed(2); }
    if (k === "clip_thickness") { clipThick.s.value = model.get(k); clipThick.v.textContent = Number(model.get(k)).toFixed(2); }
    if (k === "marker_size") { sizeSl.s.value = model.get(k); sizeSl.v.textContent = Number(model.get(k)).toFixed(2); }
    if (k === "depth_cue") { cueSl.s.value = model.get(k); cueSl.v.textContent = Number(model.get(k)).toFixed(2); }
    draw();
  }));
  model.on("change:dark_background", () => { bgChk.checked = model.get("dark_background"); root.classList.toggle("qa-dark", model.get("dark_background")); drawHist(); draw(); });
  model.on("change:title", () => { title.textContent = model.get("title"); });

  title.textContent = model.get("title");
  root.classList.toggle("qa-dark", model.get("dark_background"));
  if (model.get("range_auto")) autoRange();
  drawHist(); draw();
}
export default { render };
"""

_CSS = r"""
.qa-root { display: flex; gap: 10px; font-family: system-ui, sans-serif; font-size: 12px; color: #222; background: #fafafa; padding: 6px; border-radius: 6px; }
.qa-root.qa-dark { color: #ddd; background: #222; }
.qa-left { display: flex; flex-direction: column; gap: 4px; }
.qa-title { font-weight: 600; font-size: 13px; }
.qa-canvas { border: 1px solid #888; cursor: grab; touch-action: none; }
.qa-status, .qa-info { font-size: 11px; opacity: 0.8; min-height: 14px; }
.qa-panel { display: flex; flex-direction: column; gap: 4px; width: 250px; }
.qa-row { display: flex; align-items: center; gap: 6px; }
.qa-row label { width: 74px; flex: none; }
.qa-row select { flex: 1; min-width: 0; }
.qa-row input[type=range] { flex: 1; min-width: 0; }
.qa-val { width: 44px; text-align: right; font-variant-numeric: tabular-nums; }
.qa-num { width: 80px; }
.qa-hist { border: 1px solid #888; touch-action: none; cursor: col-resize; }
.qa-sep { margin-top: 6px; font-weight: 600; border-bottom: 1px solid #888; }
.qa-btn { padding: 2px 8px; font-size: 11px; }
.qa-legend { display: flex; flex-wrap: wrap; gap: 4px 10px; }
.qa-legend-item { display: flex; align-items: center; gap: 4px; }
.qa-swatch { width: 10px; height: 10px; border-radius: 50%; border: 1px solid #555; display: inline-block; }
"""


def _colormap_luts(names: list[str]) -> bytes:
    import matplotlib.pyplot as plt

    out = np.zeros((len(names), 256, 3), dtype=np.uint8)
    for i, n in enumerate(names):
        rgba = plt.get_cmap(n)(np.linspace(0, 1, 256))
        out[i] = (rgba[:, :3] * 255).round().astype(np.uint8)
    return out.tobytes()


class ShowAtoms3D(anywidget.AnyWidget):
    """Interactive 3D site viewer for an :class:`~quantem.atoms.AtomicModel`.

    Parameters
    ----------
    model : AtomicModel
        Model to display; all channels are sent to the browser as float32.
    channel : str, optional
        Initial color channel (default: ``"structure"`` if present).
    cmap : str
        Initial colormap for continuous channels.
    cmaps : sequence of str, optional
        Colormaps offered in the dropdown (matplotlib names).
    canvas_size : int
        Canvas width and height in pixels.
    marker_size : float
        Marker diameter as a fraction of the bond length.
    dark_background : bool
        Dark theme.
    title : str, optional
        Title shown above the canvas (default: model name).

    Attributes
    ----------
    rotation : list of float
        Row-major ``(3, 3)`` view matrix (rows: right, up, toward-viewer).
    view_matrix : ndarray
        Same as ``rotation`` as an array; ``view_matrix[2]`` is the viewing
        direction usable as ``model.plot("slab", normal=...)``.
    selected : int
        Index of the last clicked site (``-1`` if none).
    """

    _esm = _ESM
    _css = _CSS

    positions = traitlets.Bytes(b"").tag(sync=True)
    num_sites = traitlets.Int(0).tag(sync=True)
    channel_names = traitlets.List(traitlets.Unicode()).tag(sync=True)
    channel_data = traitlets.Bytes(b"").tag(sync=True)
    channel_categories = traitlets.Dict().tag(sync=True)
    channel_palettes = traitlets.Dict().tag(sync=True)
    cmap_names = traitlets.List(traitlets.Unicode()).tag(sync=True)
    cmap_luts = traitlets.Bytes(b"").tag(sync=True)
    model_radius = traitlets.Float(1.0).tag(sync=True)
    bond_length = traitlets.Float(1.0).tag(sync=True)
    units = traitlets.Unicode("").tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)

    channel = traitlets.Unicode("").tag(sync=True)
    cmap = traitlets.Unicode("viridis").tag(sync=True)
    vmin = traitlets.Float(0.0).tag(sync=True)
    vmax = traitlets.Float(1.0).tag(sync=True)
    range_auto = traitlets.Bool(True).tag(sync=True)
    hide_outside_range = traitlets.Bool(False).tag(sync=True)
    rotation = traitlets.List(traitlets.Float(), default_value=[1, 0, 0, 0, 1, 0, 0, 0, 1]).tag(
        sync=True
    )
    zoom = traitlets.Float(1.0).tag(sync=True)
    pan = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0]).tag(sync=True)
    clip_enabled = traitlets.Bool(False).tag(sync=True)
    clip_axis = traitlets.Unicode("view").tag(sync=True)
    clip_center = traitlets.Float(0.0).tag(sync=True)
    clip_thickness = traitlets.Float(1.0).tag(sync=True)
    marker_size = traitlets.Float(0.7).tag(sync=True)
    depth_cue = traitlets.Float(0.5).tag(sync=True)
    show_edges = traitlets.Bool(True).tag(sync=True)
    dark_background = traitlets.Bool(True).tag(sync=True)
    canvas_size = traitlets.Int(640).tag(sync=True)
    selected = traitlets.Int(-1).tag(sync=True)

    def __init__(
        self,
        model: Any,
        channel: str | None = None,
        cmap: str = "viridis",
        cmaps: list[str] | None = None,
        canvas_size: int = 640,
        marker_size: float = 0.7,
        dark_background: bool = True,
        title: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model = model
        xyz = np.asarray(model.positions, dtype=np.float32)
        center = xyz.mean(0)
        xyz = xyz - center
        self._center = center
        self.positions = np.ascontiguousarray(xyz, dtype=np.float32).tobytes()
        self.num_sites = int(xyz.shape[0])
        self.model_radius = float(np.linalg.norm(xyz, axis=1).max()) if xyz.shape[0] else 1.0
        try:
            self.bond_length = float(model.bond_length)
        except Exception:
            self.bond_length = float(self.model_radius / 20)
        self.units = str(model.units)
        self.title = title if title is not None else str(model.name)
        self.cmap_names = list(cmaps or _DEFAULT_CMAPS)
        self.cmap_luts = _colormap_luts(self.cmap_names)
        self.cmap = cmap if cmap in self.cmap_names else self.cmap_names[0]
        self.canvas_size = int(canvas_size)
        self.marker_size = float(marker_size)
        self.dark_background = bool(dark_background)
        self.clip_thickness = float(3 * self.bond_length)
        self.update_channels()
        if channel is None:
            channel = "structure" if "structure" in self.channel_names else self.channel_names[0]
        self.channel = channel

    # ------------------------------------------------------------------ #
    def update_channels(self) -> None:
        """Re-send all channels of the model (call after adding new channels)."""
        model = self._model
        names = ["x", "y", "z"] + list(model.channels)
        data = np.stack([model.get_channel(n) for n in names], axis=0).astype(np.float32)
        cats: dict[str, list[str]] = {}
        palettes: dict[str, list[int]] = {}
        for name, labels in model.categories.items():
            cats[name] = list(labels)
            n_cat = max(
                len(labels), int(np.nanmax(model.get_channel(name))) + 1 if model.num_sites else 1
            )
            if name == "grain" or n_cat > len(_CATEGORICAL_COLORS):
                rng = np.random.default_rng(0)
                colors = rng.uniform(0.15, 0.95, (n_cat, 3))
            else:
                colors = _CATEGORICAL_COLORS[np.arange(n_cat) % len(_CATEGORICAL_COLORS)]
            palettes[name] = (colors * 255).round().astype(int).ravel().tolist()
        for name in ("grain",):
            if name in model.channels and name not in cats:
                codes = model.get_channel(name)
                n_cat = int(np.nanmax(codes)) + 1 if codes.size else 1
                cats[name] = [str(i) for i in range(n_cat)]
                rng = np.random.default_rng(0)
                palettes[name] = (
                    (rng.uniform(0.15, 0.95, (max(n_cat, 1), 3)) * 255)
                    .round()
                    .astype(int)
                    .ravel()
                    .tolist()
                )
        self.channel_names = names
        self.channel_data = np.ascontiguousarray(data).tobytes()
        self.channel_categories = cats
        self.channel_palettes = palettes

    @property
    def view_matrix(self) -> NDArray:
        """``(3, 3)`` current view matrix (rows: right, up, toward-viewer)."""
        return np.asarray(self.rotation, dtype=float).reshape(3, 3)

    @property
    def view_direction(self) -> NDArray:
        """``(3,)`` unit vector pointing from the model toward the viewer."""
        return self.view_matrix[2]

    def set_view(self, normal: str | NDArray, up: str | NDArray | None = None) -> None:
        """Look along ``normal`` (as in :func:`quantem.atoms.visualization.view_matrix`)."""
        from quantem.atoms.visualization import view_matrix

        self.rotation = view_matrix(normal, up).ravel().tolist()

    def set_range(self, vmin: float, vmax: float) -> None:
        """Set the color range and disable auto-ranging."""
        self.range_auto = False
        self.vmin, self.vmax = float(vmin), float(vmax)

    def clip(
        self, axis: str = "view", center: float = 0.0, thickness: float | None = None
    ) -> None:
        """Enable slab clipping along ``axis`` (``"view"``, ``"x"``, ``"y"`` or ``"z"``)."""
        self.clip_axis = axis
        self.clip_center = float(center)
        if thickness is not None:
            self.clip_thickness = float(thickness)
        self.clip_enabled = True

    def slab_kwargs(self) -> dict[str, Any]:
        """Keyword arguments reproducing the current view with ``model.plot("slab", ...)``."""
        vm = self.view_matrix
        out: dict[str, Any] = {"normal": vm[2], "up": vm[1], "channel": self.channel}
        if self.clip_enabled and self.clip_axis == "view":
            out.update(offset=self.clip_center, thickness=self.clip_thickness)
        if not self.range_auto:
            out.update(vmin=self.vmin, vmax=self.vmax)
        if self.hide_outside_range:
            out["hide"] = None
        return out

    def __repr__(self) -> str:
        return f"ShowAtoms3D({self.num_sites} sites, channel={self.channel!r}, clip={self.clip_enabled})"

    def _repr_json_state(self) -> str:  # pragma: no cover - debugging aid
        return json.dumps(
            {k: getattr(self, k) for k in ("channel", "cmap", "vmin", "vmax", "rotation", "zoom")}
        )
