"""Lattice-free crystal-field inversion: a dense cell field per voxel with group TV.

Unknowns
  vbar (nv, nv) >= 0       mean phase density (rad / A), TV penalty
  a (K, nv, nv) >= 0       amplitude of every cell (theta_c, u) in every voxel; a cell is
                           one Friedel pair (flash tilt theta_c, in-plane |g| = u)
The crystal term uses min(vbar / vbar_ref, 1) (detached) as a mass gate, so crystal can
only sit where there is material. Cells are a fixed grid (1 deg x 0.05 1/A), each with the
same strength kappa^2 = A0, so a carries occupancy times strength.

Group TV, lam_g * mean over neighbour pairs of ||a(x, :) - a(x', :)||_2, makes a new
boundary expensive and an extra cell on an existing boundary cheap, so the cheapest
explanation puts all flashes that share footprints on one region. A small L1 keeps the
field sparse. No lattice, no grain count, no flash detection.

Only cells near Bragg at a tilt are evaluated there. Every tenth tilt is held out.
"""

import json
import sys
import time

import numpy as np
import torch

from fast2d import Fast2D
from invert_lattice import tv

A0 = 2e-4


def group_tv(a):
    dx = a[:, 1:, :] - a[:, :-1, :]
    dz = a[:, :, 1:] - a[:, :, :-1]
    return 0.5 * (torch.sqrt((dx**2).sum(0) + 1e-10).mean() + torch.sqrt((dz**2).sum(0) + 1e-10).mean())


def run(kind="fewAm", iters=1500, warm=300, batch=16, lam_tv=0.05, lam_g=0.3, lam_1=5e-4, a0=2e-3,
        lr_a=0.01, dth=1.0, du=0.05, u_min=0.25, u_max=1.40, s_cut=0.03, seed=0, tag="", log_every=100,
        threads=3):
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    d = np.load(f"runs/sim_{kind}.npz")
    cfg = json.loads(str(d["cfg"]))
    ang = d["angles"]
    data = torch.as_tensor(d["I_coarse"])
    m = Fast2D(cfg, ang, x_half=cfg["R"] + 16, s_cut=s_cut)
    T = len(ang)
    held = torch.arange(5, T, 10)
    train = torch.as_tensor([i for i in range(T) if i not in set(held.tolist())])
    crop = slice(m.n_out // 2 - 70, m.n_out // 2 + 70)
    var = float(data[:, crop].var())
    nv = cfg["n_vox"]

    th_g = np.arange(-90.0, 90.0, dth)
    u_g = np.arange(u_min, u_max + 1e-9, du)
    TH, U = np.meshgrid(th_g, u_g, indexing="ij")
    th_c = torch.as_tensor(TH.ravel(), dtype=torch.float32)
    u_c = torch.as_tensor(U.ravel(), dtype=torch.float32)
    K = len(th_c)
    # cells near Bragg at each tilt (either Friedel member within s_cut)
    act = []
    for t in range(T):
        dl = np.radians((ang[t] - TH.ravel() + 90.0) % 180.0 - 90.0)
        sp = U.ravel() * np.sin(dl) - 0.5 * m.lam * U.ravel() ** 2
        sm = -U.ravel() * np.sin(dl) - 0.5 * m.lam * U.ravel() ** 2
        act.append(torch.as_tensor(np.nonzero((np.abs(sp) < s_cut) | (np.abs(sm) < s_cut))[0]))
    print(f"{K} cells per voxel, {np.mean([len(x) for x in act]):.0f} near Bragg per tilt", flush=True)

    vbar = torch.zeros(nv, nv, requires_grad=True)
    a = torch.full((K, nv, nv), a0, requires_grad=True)
    opt = torch.optim.Adam([dict(params=[vbar], lr=1e-3), dict(params=[a], lr=lr_a)])
    ones = torch.full((K, 1), A0)

    def gate():
        v = vbar.detach()
        if not (v > 0).any():
            return torch.zeros_like(v)
        return (v / torch.quantile(v[v > 0], 0.95).clamp_min(1e-6)).clamp(max=1.0)

    def predict(tidx, crystal=True):
        g = gate()
        out = []
        for t in tidx.tolist():
            idx = act[t] if crystal else act[t][:1]  # crystal off: one cell at zero weight
            W = a[idx] * g if crystal else a[idx].detach() * 0
            out.append(m.forward(torch.tensor([t]), vbar, W, th_c[idx][:, None], u_c[idx][:, None], ones[idx]))
        return torch.cat(out)

    def score(idx, crystal=True):
        with torch.no_grad():
            pred = predict(idx, crystal)
            return float(((pred - data[idx])[:, crop] ** 2).mean() / var), pred

    gen = torch.Generator().manual_seed(seed + 1)
    t0 = time.time()
    hist = []
    for it in range(iters):
        tidx = train[torch.randperm(len(train), generator=gen)[:batch]]
        crystal = it >= warm
        pred = predict(tidx, crystal)
        mse = ((pred - data[tidx])[:, crop] ** 2).mean() / var
        loss = mse + lam_tv * tv(vbar) / nv
        if crystal:
            loss = loss + lam_g * group_tv(a) + lam_1 * a.sum(0).mean()
        opt.zero_grad()
        loss.backward()
        if not crystal:
            a.grad = None
        opt.step()
        with torch.no_grad():
            vbar.clamp_(min=0)
            a.clamp_(min=0)
        if it % log_every == 0 or it == iters - 1:
            hist.append(dict(it=it, nmse=float(mse.detach()), t=time.time() - t0))
            print(f"it {it:4d} nmse {float(mse.detach()):.4f}  sum a / voxel {float(a.detach().sum(0).mean()):.2f}"
                  f"  {time.time() - t0:.0f}s", flush=True)
    tr, _ = score(train)
    ho, _ = score(held)
    ho0, _ = score(held, crystal=False)
    print(f"final: train nmse {tr:.4f}  held-out nmse {ho:.4f}  (same vbar, crystal off: {ho0:.4f})", flush=True)
    with torch.no_grad():
        pred_all = predict(torch.arange(T))
        ag = (a * gate()).numpy()
    tot = ag.sum(0)
    np.savez_compressed(f"runs/cells_{kind}{tag}.npz", pred=pred_all.numpy(), vbar=vbar.detach().numpy(), held=held.numpy(),
             a_sum=tot, a_max=ag.max(0), a_sq=(ag**2).sum(0), a_arg=ag.argmax(0),
             a16=ag.astype(np.float16), th_c=th_c.numpy(), u_c=u_c.numpy(), hist=json.dumps(hist),
             scores=json.dumps(dict(train=tr, held=ho, held_crystal_off=ho0)))
    return tr, ho


if __name__ == "__main__":
    run(**(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}))


def true_field(sim, th_c, u_c, snap=True):
    """Diagnostic only: the cell field of the true structure. Each true reflection of a
    crystalline grain gets amplitude |rho| / sqrt(A0) on the grain's occupancy, either
    snapped to the nearest grid cell or as its own exact cell. Returns W, theta_c, u."""
    occ = torch.as_tensor(sim["occ"])
    refl = json.loads(str(sim["refl"]))
    amp = [occ[e["grain"]] * float(np.hypot(*e["rho"])) / np.sqrt(A0) for e in refl]
    if not snap:
        thc = torch.as_tensor([e["theta_c"] for e in refl], dtype=torch.float32)
        uc = torch.as_tensor([e["u"] for e in refl], dtype=torch.float32)
        return torch.stack(amp), thc, uc
    th_c, u_c = np.asarray(th_c), np.asarray(u_c)
    W = torch.zeros(len(th_c), *occ.shape[1:])
    for e, am in zip(refl, amp):
        j = int(np.argmin(((th_c - e["theta_c"] + 90) % 180 - 90) ** 2 + (25 * (u_c - e["u"])) ** 2))
        W[j] += am
    return W, torch.as_tensor(th_c, dtype=torch.float32), torch.as_tensor(u_c, dtype=torch.float32)


def predict_field(sim, vbar, W, th_c, u_c, s_cut=0.03, chunk=6):
    """All-tilt prediction of a fixed cell field (no gate)."""
    cfg = json.loads(str(sim["cfg"]))
    m = Fast2D(cfg, sim["angles"], x_half=cfg["R"] + 16, s_cut=s_cut)
    T = len(sim["angles"])
    vb = torch.as_tensor(vbar, dtype=torch.float32)
    with torch.no_grad():
        return torch.cat([m.forward(torch.arange(i, min(i + chunk, T)), vb, W, th_c[:, None], u_c[:, None],
                                    torch.full((len(th_c), 1), A0)) for i in range(0, T, chunk)]).numpy()
