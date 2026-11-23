"""Coarse forward with ground-truth parameters vs the fine multislice data."""
import json, sys, time
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from coarse2d import Coarse2D

kind = sys.argv[1] if len(sys.argv) > 1 else "few"
s_cut = float(sys.argv[2]) if len(sys.argv) > 2 else 0.04
torch.set_num_threads(4)
d = np.load(f"runs/sim_{kind}.npz")
cfg = json.loads(str(d["cfg"])); refl = json.loads(str(d["refl"]))
ang = d["angles"]; data = torch.as_tensor(d["I_coarse"], dtype=torch.float64)
m = Coarse2D(cfg, ang, s_cut=s_cut)
vbar = torch.as_tensor(d["vbar"], dtype=torch.float64); occ = torch.as_tensor(d["occ"], dtype=torch.float64)
g = torch.as_tensor([e["g"] for e in refl], dtype=torch.float64)
rho = torch.as_tensor([complex(*e["rho"]) for e in refl], dtype=torch.complex128)
st = torch.as_tensor([e["grain"] for e in refl])
t0 = time.time()
full, I0 = m.forward(vbar, occ, g, rho, st, return_parts=True)
t1 = time.time()
mean_only = m.forward(vbar, None)
crop = slice(m.n_out // 2 - 64, m.n_out // 2 + 64)
def err(a): return float(((a - data)[:, crop] ** 2).mean().sqrt())
flick_var = float(((data - mean_only)[:, crop] ** 2).mean())
print(f"forward {t1 - t0:.1f} s, s_cut {s_cut}")
print(f"rms(data - mean-only model) = {err(mean_only):.4f}")
print(f"rms(data - full model)      = {err(full):.4f}")
print(f"fraction of the beyond-mean signal explained = {1 - err(full)**2 / flick_var:.3f}")
cc = np.corrcoef((full - mean_only)[:, crop].ravel(), (data - mean_only)[:, crop].ravel())[0, 1]
print(f"corr(model flicker, data flicker) = {cc:.3f}")
np.savez(f"runs/check_{kind}.npz", full=full.numpy(), mean_only=mean_only.numpy())
x = (np.arange(m.n_out) - m.n_out // 2) * cfg["dx_coarse"]
ext = [x[0], x[-1], ang[-1], ang[0]]
fig, ax = plt.subplots(1, 4, figsize=(18, 6), sharey=True)
for a, im, t in zip(ax, [data, full, data - mean_only, full - mean_only],
                    ["data", "coarse model (GT params)", "data - mean-only model", "model - mean-only model"]):
    if "-" in t:
        a.imshow(im, aspect="auto", extent=ext, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    else:
        a.imshow(im, aspect="auto", extent=ext, cmap="gray", vmin=0.2, vmax=1.8)
    a.set_title(t); a.set_xlim(-130, 130)
plt.tight_layout(); plt.savefig(f"runs/check_{kind}.png", dpi=100)
