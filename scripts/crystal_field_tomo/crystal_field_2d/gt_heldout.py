"""Hold true grains and reflection cells fixed; refit vbar and strengths only."""
import json, sys, numpy as np, torch
from fast2d import Fast2D
torch.set_num_threads(4)
kind = sys.argv[1] if len(sys.argv) > 1 else "fewA"
T = len(np.load(f"runs/sim_{kind}.npz")["angles"]); held = torch.arange(5, T, 10); train = torch.as_tensor([i for i in range(T) if i % 10 != 5])
d = np.load(f"runs/sim_{kind}.npz"); cfg = json.loads(str(d["cfg"])); refl = json.loads(str(d["refl"]))
ang = d["angles"]; data = torch.as_tensor(d["I_coarse"]); m = Fast2D(cfg, ang)
crop = slice(m.n_out // 2 - 70, m.n_out // 2 + 70); var = float(data[:, crop].var())
K = d["occ"].shape[0]; M = max(sum(e["grain"] == k for e in refl) for k in range(K))
th = torch.zeros(K, M); u = torch.full((K, M), 0.5); a0 = torch.zeros(K, M); cnt = [0]*K
for e in refl:
    k = e["grain"]; j = cnt[k]; cnt[k] += 1
    th[k, j], u[k, j], a0[k, j] = e["theta_c"], e["u"], float(np.hypot(*e["rho"]))**2
W = torch.as_tensor(d["occ"])
vbar = torch.as_tensor(d["vbar"]).clone().requires_grad_(True)
la = torch.log(a0 + 1e-9).requires_grad_(True)
opt = torch.optim.Adam([dict(params=[vbar], lr=1e-3), dict(params=[la], lr=0.02)])
g = torch.Generator().manual_seed(0)
for it in range(800):
    tidx = train[torch.randperm(len(train), generator=g)[:60]]
    pred = m.forward(tidx, vbar, W, th, u, la.exp())
    loss = ((pred - data[tidx])[:, crop]**2).mean() / var
    opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): vbar.clamp_(min=0)
with torch.no_grad():
    full = m.forward(torch.arange(len(ang)), vbar, W, th, u, la.exp())
    sc = lambda idx: float(((full[idx] - data[idx])[:, crop]**2).mean() / var)
    print(f"GT structure, vbar + strengths refit on train tilts: train nmse {sc(train):.4f}  held-out nmse {sc(held):.4f}")
    np.savez(f"runs/gtrefit_{kind}_held.npz", pred=full.numpy(), vbar=vbar.numpy(), a=la.exp().numpy())
