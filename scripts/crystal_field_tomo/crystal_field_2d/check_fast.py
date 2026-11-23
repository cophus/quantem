"""Fast (kinematic + saturation) forward with ground-truth parameters vs data."""
import json, sys, time
import numpy as np, torch
from fast2d import Fast2D

kind = sys.argv[1] if len(sys.argv) > 1 else "few"
torch.set_num_threads(4)
d = np.load(f"runs/sim_{kind}.npz")
cfg = json.loads(str(d["cfg"])); refl = json.loads(str(d["refl"]))
ang = d["angles"]; data = torch.as_tensor(d["I_coarse"])
m = Fast2D(cfg, ang, coherent=len(sys.argv) > 2 and sys.argv[2] == 'coh')
K = int(d["occ"].shape[0]); M = max(sum(e["grain"] == k for e in refl) for k in range(K))
th = torch.zeros(K, M); u = torch.full((K, M), 0.5); a = torch.zeros(K, M); phi = torch.zeros(K, M); cnt = [0] * K
for e in refl:
    k = e["grain"]; j = cnt[k]; cnt[k] += 1
    th[k, j], u[k, j], a[k, j] = e["theta_c"], e["u"], float(np.hypot(*e["rho"])) ** 2
    phi[k, j] = float(np.arctan2(e['rho'][1], e['rho'][0]))
vbar = torch.as_tensor(d["vbar"]); W = torch.as_tensor(d["occ"])
t0 = time.time()
tidx = torch.arange(len(ang))
full = m.forward(tidx, vbar, W, th, u, a, phi=phi); I0 = m.forward(tidx, vbar, W, th, u, a * 0)
print(f"fast forward {time.time() - t0:.2f} s")
chk = np.load(f"runs/check_{kind}.npz")
mean_only = torch.as_tensor(chk["mean_only"], dtype=torch.float32)
crop = slice(m.n_out // 2 - 64, m.n_out // 2 + 64)
def err(x): return float(((x - data)[:, crop] ** 2).mean().sqrt())
fv = float(((data - mean_only)[:, crop] ** 2).mean())
print(f"rms data - multislice mean-only {err(mean_only):.4f}; data - fast mean-only {err(I0):.4f}")
print(f"rms data - fast full {err(full):.4f}; explained {1 - err(full)**2 / fv:.3f}")
cc = np.corrcoef((full - I0)[:, crop].ravel(), (data - I0)[:, crop].ravel())[0, 1]
print(f"corr(fast model flicker, data flicker) = {cc:.3f}")
