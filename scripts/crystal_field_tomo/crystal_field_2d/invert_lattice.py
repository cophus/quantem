"""Crystal-field inversion with one learned lattice (2D toy).

Unknowns
  vbar (nv, nv) >= 0            mean phase density (rad / A), small TV penalty
  lattice |a1|, |a2|, gamma     one real-space metric for every crystal voxel
  p_n >= 0                      strength of each Friedel pair, a_n = A_SCALE p_n
  logits (2 n_phi + 1, nv, nv)  per-voxel softmax over a dictionary of lattice
                                orientations (n_phi proper + n_phi mirrored, at
                                180 / n_phi deg) and "empty"
Every orientation starts present at a small weight, so every reflection gets a
gradient at every tilt; an entropy penalty, ramped in late, sharpens each voxel's
orientation distribution. Every tenth tilt is held out and only scored.
"""

import json
import sys
import time

import numpy as np
import torch
from scipy.ndimage import median_filter

from fast2d import Fast2D

A_SCALE = 1e-4


def lattice_reflections(L1, L2, gam, hk):
    """Sample-frame angle (deg) and length of g_hk for the metric (L1, L2, gamma)."""
    A = torch.stack([torch.stack([L1, L2 * torch.cos(gam)]), torch.stack([L1 * 0, L2 * torch.sin(gam)])])
    B = torch.linalg.inv(A).T  # columns b1, b2
    g = hk.to(B.dtype) @ B.T
    return torch.rad2deg(torch.atan2(g[:, 1], g[:, 0])), torch.linalg.norm(g, dim=1)


def index_list(L1, L2, gam, g_min=0.25, g_max=1.4):
    n = 8
    hk = torch.tensor([(h, k) for h in range(0, n + 1) for k in range(-n, n + 1)
                       if h > 0 or (h == 0 and k > 0)])
    _, gl = lattice_reflections(torch.tensor(L1), torch.tensor(L2), torch.tensor(gam), hk)
    return hk[(gl > g_min) & (gl < g_max)]


def dominant_u(I, cfg, lam):
    """Typical in-plane |g| from the mean flicker autocorrelation: the dark and
    bright copies of a flash anti-correlate at the displacement lam df u."""
    f = I - median_filter(I, size=(7, 1), mode="wrap")
    Fq = np.fft.fft(f, axis=1)
    ac = np.fft.ifft(np.abs(Fq) ** 2, axis=1).real.mean(0)
    lag = np.arange(len(ac)) * cfg["dx_coarse"]
    sel = (lag > 4) & (lag < 120)
    d = lag[sel][np.argmin(ac[sel])]
    return d / (lam * cfg["defocus"])


def tv(v):
    dx = v[1:, :-1] - v[:-1, :-1]
    dz = v[:-1, 1:] - v[:-1, :-1]
    return torch.sqrt(dx**2 + dz**2 + 1e-8).sum()


def run(kind="fewA", n_phi=180, iters=1500, warm=300, batch=12, lam_tv=2e-3, lam_ent=3e-3,
        ent_from=0.5, gamma0=60.0, L_scale=1.0, u_start=None, lattice0=None, seed=0, tag="", log_every=100, lattice_lr=2e-3):
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    d = np.load(f"runs/sim_{kind}.npz")
    cfg = json.loads(str(d["cfg"]))
    ang = d["angles"]
    data = torch.as_tensor(d["I_coarse"])
    m = Fast2D(cfg, ang, x_half=cfg['R'] + 16)
    T = len(ang)
    held = torch.arange(5, T, 10)
    train = torch.as_tensor([i for i in range(T) if i not in set(held.tolist())])
    crop = slice(m.n_out // 2 - 70, m.n_out // 2 + 70)
    var = float(data[:, crop].var())
    nv = cfg["n_vox"]

    u_star = dominant_u(d["I_coarse"], cfg, m.lam) if u_start is None else u_start
    a0 = 2 / (np.sqrt(3) * u_star) * L_scale  # hexagonal guess with first shell u_star
    print(f"flicker autocorrelation -> u* = {u_star:.3f} 1/A, start |a1|=|a2|={a0:.3f} A, gamma {gamma0} deg")
    b0 = (a0, a0, gamma0) if lattice0 is None else tuple(lattice0)  # (|a1|, |a2|, gamma deg)
    if lattice0 is not None:
        print(f"lattice given: {b0}")
    hk = index_list(b0[0], b0[1], np.radians(b0[2]))
    n_ref = len(hk)
    L1 = torch.tensor(b0[0], dtype=torch.float32, requires_grad=True)
    L2 = torch.tensor(b0[1], dtype=torch.float32, requires_grad=True)
    gam = torch.tensor(np.radians(b0[2]), dtype=torch.float32, requires_grad=True)
    p = torch.full((n_ref,), 1.0, requires_grad=True)
    vbar = torch.zeros(nv, nv, requires_grad=True)
    K = 2 * n_phi
    logits = torch.zeros(K + 1, nv, nv)
    logits[0] = np.log(K)  # crystal fraction 1/2 everywhere at the start
    logits.requires_grad_(True)
    phi0 = torch.arange(n_phi, dtype=torch.float32) * (180.0 / n_phi)
    dphi = torch.zeros(K, requires_grad=True)  # per-entry offset, +- half a step
    opt = torch.optim.Adam([
        dict(params=[vbar], lr=1e-3),
        dict(params=[logits], lr=0.1),
        dict(params=[p], lr=0.02),
        dict(params=[L1, L2, gam], lr=lattice_lr),
        dict(params=[dphi], lr=0.02),
    ])

    def decode():
        alpha, u = lattice_reflections(L1, L2, gam, hk)
        ph = torch.cat([phi0, phi0]) + 0.5 * (180.0 / n_phi) * torch.tanh(dphi)
        th = torch.cat([ph[:n_phi, None] + alpha[None], ph[n_phi:, None] - alpha[None]])  # proper, mirrored
        th = (th + 90.0) % 180.0 - 90.0
        uu = u[None].expand(K, -1)
        aa = (A_SCALE * p.clamp_min(0))[None].expand(K, -1)
        W = torch.softmax(logits, 0)[1:]
        return W, th, uu, aa

    def predict(tidx, crystal=True):
        W, th, uu, aa = decode()
        return m.forward(tidx, vbar, W, th, uu, aa if crystal else aa * 0)

    def score(idx, crystal=True):
        with torch.no_grad():
            pred = torch.cat([predict(idx[i:i + 12], crystal) for i in range(0, len(idx), 12)])
            return float(((pred - data[idx])[:, crop] ** 2).mean() / var), pred

    g = torch.Generator().manual_seed(seed + 1)
    t0 = time.time()
    hist = []
    for it in range(iters):
        tidx = train[torch.randperm(len(train), generator=g)[:batch]]
        crystal = it >= warm
        pred = predict(tidx, crystal)
        mse = ((pred - data[tidx])[:, crop] ** 2).mean() / var
        loss = mse + lam_tv * tv(vbar) / nv
        if crystal and it >= ent_from * iters:
            Wc = torch.softmax(logits, 0)
            loss = loss + lam_ent * (-(Wc * torch.log(Wc + 1e-9)).sum(0)).mean()
        opt.zero_grad()
        loss.backward()
        if not crystal:
            for t_ in (logits, p, L1, L2, gam, dphi):
                t_.grad = None
        opt.step()
        with torch.no_grad():
            vbar.clamp_(min=0)
            p.clamp_(min=0)
        if it % log_every == 0 or it == iters - 1:
            hist.append(dict(it=it, nmse=float(mse.detach()), L1=float(L1.detach()), L2=float(L2.detach()),
                             gamma=float(torch.rad2deg(gam.detach())), t=time.time() - t0))
            print(f"it {it:4d} nmse {float(mse.detach()):.4f}  |a1| {float(L1):.3f} |a2| {float(L2):.3f} "
                  f"gamma {float(torch.rad2deg(gam)):.1f}  {time.time() - t0:.0f}s", flush=True)
    tr, _ = score(train)
    ho, pred_ho = score(held)
    ho0, _ = score(held, crystal=False)
    print(f"final: train nmse {tr:.4f}  held-out nmse {ho:.4f}  (same vbar, crystal off: {ho0:.4f})")
    with torch.no_grad():
        pred_all = torch.cat([predict(torch.arange(i, min(i + 12, T))) for i in range(0, T, 12)])
        W, th, uu, aa = decode()
    np.savez(f"runs/lat_{kind}{tag}.npz", pred=pred_all.numpy(), vbar=vbar.detach().numpy(),
             W_crys=W.sum(0).numpy(), W_arg=W.argmax(0).numpy(), held=held.numpy(),
             lattice=np.array([float(L1), float(L2), float(torch.rad2deg(gam))]),
             p=p.detach().numpy(), hk=hk.numpy(), hist=json.dumps(hist),
             scores=json.dumps(dict(train=tr, held=ho, held_crystal_off=ho0)))
    return tr, ho


if __name__ == "__main__":
    run(**(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}))
