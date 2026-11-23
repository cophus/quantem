"""Step A: per-tilt footprints.

Fit the mean phase density vbar (shared by all tilts) plus, at every tilt, S
free footprints F_tj(x) >= 0 (the column intensity a beam takes out of phi_0)
with a signed in-plane frequency u_tj (displacement direction and size). The
model per tilt is

    phi_0 = exp(i Phi) sqrt(1 - sum_j F_j),   psi_j = exp(i Phi) sqrt(F_j),

each propagated with its own transfer function, intensities summed. An L1
penalty on F keeps most slots empty. No grains, no cutoffs: this only splits
each tilt's flicker into displaced copies.
"""

import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import chi, crop_intensity
from fast2d import Fast2D


def project(m, maps, tidx):
    """Line integrals of maps (C, nv, nv) at tilts tidx -> (T, C, X_sub)."""
    T = len(tidx)
    rot = F.grid_sample(maps[None].expand(T, -1, -1, -1), m.grid(tidx), mode="bilinear",
                        padding_mode="zeros", align_corners=False)
    return rot.sum(-1) * m.dz


def model(m, vbar, Fp, u, tidx):
    T = len(tidx)
    phase = project(m, vbar[None], tidx)[:, 0]
    Fj = Fp[tidx].clamp_min(0)  # (T, S, X)
    D = Fj.sum(1).clamp(max=1 - 1e-4)
    ex = torch.exp(1j * phase)
    w0 = torch.ones(T, m.nw, dtype=m.cdt)
    w0[:, m.sub] = ex * torch.sqrt(1 - D)
    I = torch.fft.ifft(torch.fft.fft(w0) * m.ctf0).abs() ** 2
    S = Fj.shape[1]
    wn = torch.zeros(T, S, m.nw, dtype=m.cdt)
    wn[..., m.sub] = ex[:, None] * torch.sqrt(Fj + 1e-10)
    ctf = torch.exp(-1j * chi(m.q[None, None] + u[tidx][..., None], m.lam, m.cfg["defocus"],
                              m.cfg["cs"]))
    I = I + (torch.fft.ifft(torch.fft.fft(wn) * ctf).abs() ** 2).sum(1)
    return crop_intensity(I, m.n_out)


def run(kind="few", S=4, u0=(-0.9, -0.5, 0.5, 0.9), iters=1500, warm=300, batch=60,
        lam=3e-3, seed=0, tag="", vlr=3e-3):
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    d = np.load(f"runs/sim_{kind}.npz")
    cfg = json.loads(str(d["cfg"]))
    ang = d["angles"]
    data = torch.as_tensor(d["I_coarse"])
    m = Fast2D(cfg, ang)
    T, X = len(ang), len(m.sub)
    nv = cfg["n_vox"]
    crop = slice(m.n_out // 2 - 70, m.n_out // 2 + 70)
    var = float(data[:, crop].var())
    vbar = torch.zeros(nv, nv, requires_grad=True)
    Fp = torch.full((T, S, X), 1e-3, requires_grad=True)
    u = torch.as_tensor(u0, dtype=torch.float32).repeat(T, 1).requires_grad_(True)
    opt = torch.optim.Adam([dict(params=[vbar], lr=vlr), dict(params=[Fp], lr=3e-3),
                            dict(params=[u], lr=1e-2)])
    g = torch.Generator().manual_seed(seed + 1)
    t0 = time.time()
    for it in range(iters):
        tidx = torch.randperm(T, generator=g)[:batch]
        Fuse = Fp if it >= warm else Fp * 0
        pred = model(m, vbar, Fuse, u, tidx)
        mse = ((pred - data[tidx])[:, crop] ** 2).mean() / var
        loss = mse + (lam * Fp[tidx].clamp_min(0).sum() / batch if it >= warm else 0)
        opt.zero_grad()
        loss.backward()
        if it < warm:
            Fp.grad = None
            u.grad = None
        opt.step()
        with torch.no_grad():
            vbar.clamp_(min=0)
            Fp.clamp_(min=0)
            u.data = torch.sign(u) * u.abs().clamp(0.25, 1.4)
        if it % 250 == 0 or it == iters - 1:
            mass = Fp.detach().sum(-1)
            print(f"it {it:4d} nmse {float(mse.detach()):.4f}  footprint mass {float(mass.sum()):.1f}  "
                  f"slots >5% of max {int((mass > 0.05 * mass.max()).sum())}  {time.time() - t0:.0f}s",
                  flush=True)
    with torch.no_grad():
        pred = torch.cat([model(m, vbar, Fp, u, torch.arange(i, min(i + 60, T))) for i in range(0, T, 60)])
        nm = float(((pred - data)[:, crop] ** 2).mean() / var)
    print(f"final nmse (all tilts) {nm:.4f}")
    np.savez(f"runs/fp_{kind}{tag}.npz", vbar=vbar.detach().numpy(), F=Fp.detach().numpy(),
             u=u.detach().numpy(), pred=pred.numpy())


if __name__ == "__main__":
    run(**(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}))
