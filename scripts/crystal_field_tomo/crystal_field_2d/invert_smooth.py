"""Crystal-field inversion with one learned lattice and a smooth orientation field.

Same forward model and dictionary as invert_lattice.py; the per-voxel weights
are factored as W_k(x) = c(x) q_k(x):
  c(x) in [0, 1]   crystallinity, sigmoid of a blurred logit map
  q_k(x)           orientation distribution, softmax of blurred logits
The blur (Gaussian, sigma voxels) ties each voxel's orientation to its
neighbours, and a total-variation penalty on W makes the field piecewise
constant, so one orientation has to explain every tilt a region flashes in.
The mean potential carries a TV penalty scaled to matter at the true density.
Options:
  sharpen > 0   entropy of q (not weighted by c), ramped linearly from 0 at
                ent_from to `sharpen` at the end, so each voxel ends with one
                orientation and c alone says how crystalline it is
  mass_gate     W = c q min(vbar / vbar_ref, 1), vbar detached: crystal needs
                mass, so c is the crystalline fraction of the material present
  diffuse       the non-crystalline remainder (1 - c) scatters as an orientation
                average (uniform over the dictionary) at a learned global
                fraction s of the crystal strength: the amorphous halo
  lam_c > 0     L1 penalty on c: material counts as amorphous unless the data
                need it to be crystalline
  robust > 0    Cauchy loss c^2 log(1 + (r / c)^2) on residuals in units of the data
                std, c = robust; if robust < 0 the scale is |robust| x 1.4826 MAD of
                the current batch residual. Large, concentrated residuals (flashes) are
                discounted; small spread ones are fitted as with squared error
Every tenth tilt is held out and only scored.
"""

import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from fast2d import Fast2D
from invert_lattice import A_SCALE, index_list, lattice_reflections, tv


def gauss_blur(x, sigma):
    """Separable Gaussian blur over the last two axes of (C, nv, nv)."""
    if sigma <= 0:
        return x
    r = int(np.ceil(3 * sigma))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=x.dtype) / sigma) ** 2)
    k = k / k.sum()
    C = x.shape[0]
    y = F.conv2d(x[:, None], k.view(1, 1, -1, 1), padding=(r, 0))
    y = F.conv2d(y, k.view(1, 1, 1, -1), padding=(0, r))
    return y[:, 0].reshape(C, *x.shape[1:])


def tv_field(W):
    """Mean over neighbour pairs of sum_k |W_k(i) - W_k(j)|."""
    return 0.5 * ((W[:, 1:] - W[:, :-1]).abs().sum(0).mean() + (W[:, :, 1:] - W[:, :, :-1]).abs().sum(0).mean())


def run(kind="fewA", n_phi=90, iters=1500, warm=300, batch=8, lam_tv=0.05, lam_sm=0.05, sigma=2.0,
        lam_ent=3e-3, ent_from=0.5, sharpen=0.0, mass_gate=False, diffuse=False, lam_c=0.0, robust=0.0, gamma0=60.0, u_start=0.42, lattice0=None, lattice_lr=2e-3,
        crystal_on=True, seed=0, tag="", log_every=100, threads=3):
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    d = np.load(f"runs/sim_{kind}.npz")
    cfg = json.loads(str(d["cfg"]))
    ang = d["angles"]
    data = torch.as_tensor(d["I_coarse"])
    m = Fast2D(cfg, ang, x_half=cfg["R"] + 16)
    T = len(ang)
    held = torch.arange(5, T, 10)
    train = torch.as_tensor([i for i in range(T) if i not in set(held.tolist())])
    crop = slice(m.n_out // 2 - 70, m.n_out // 2 + 70)
    var = float(data[:, crop].var())
    nv = cfg["n_vox"]

    a0 = 2 / (np.sqrt(3) * u_start)  # hexagonal guess with first shell u_start
    b0 = (a0, a0, gamma0) if lattice0 is None else tuple(lattice0)  # (|a1|, |a2|, gamma deg)
    print(f"start lattice {np.round(b0, 3)}", flush=True)
    hk = index_list(b0[0], b0[1], np.radians(b0[2]))
    L1 = torch.tensor(b0[0], dtype=torch.float32, requires_grad=True)
    L2 = torch.tensor(b0[1], dtype=torch.float32, requires_grad=True)
    gam = torch.tensor(np.radians(b0[2]), dtype=torch.float32, requires_grad=True)
    p = torch.full((len(hk),), 1.0, requires_grad=True)
    vbar = torch.zeros(nv, nv, requires_grad=True)
    K = 2 * n_phi
    c_raw = torch.zeros(1, nv, nv, requires_grad=True)  # crystallinity 1/2 at the start
    z_raw = torch.zeros(K, nv, nv, requires_grad=True)  # uniform orientation distribution
    phi0 = torch.arange(n_phi, dtype=torch.float32) * (180.0 / n_phi)
    dphi = torch.zeros(K, requires_grad=True)  # per-entry offset, +- half a step
    s_raw = torch.zeros((), requires_grad=True)  # diffuse strength s = sigmoid(s_raw)
    opt = torch.optim.Adam([
        dict(params=[s_raw], lr=0.02),
        dict(params=[vbar], lr=1e-3),
        dict(params=[c_raw, z_raw], lr=0.1),
        dict(params=[p], lr=0.02),
        dict(params=[L1, L2, gam], lr=lattice_lr),
        dict(params=[dphi], lr=0.02),
    ])

    def fields():
        c = torch.sigmoid(gauss_blur(c_raw, sigma))[0]
        q = torch.softmax(gauss_blur(z_raw, sigma), 0)
        return c, q

    def decode():
        alpha, u = lattice_reflections(L1, L2, gam, hk)
        ph = torch.cat([phi0, phi0]) + 0.5 * (180.0 / n_phi) * torch.tanh(dphi)
        th = torch.cat([ph[:n_phi, None] + alpha[None], ph[n_phi:, None] - alpha[None]])  # proper, mirrored
        th = (th + 90.0) % 180.0 - 90.0
        c, q = fields()
        W = c[None] * q
        if diffuse:
            W = W + ((1 - c) * torch.sigmoid(s_raw) / K)[None]
        if mass_gate:  # the TV penalty acts on the ungated field c q, not on this
            v = vbar.detach()
            W = W * (v / torch.quantile(v[v > 0], 0.95).clamp_min(1e-6)).clamp(max=1.0) if (v > 0).any() else W * 0
        return W, th, u[None].expand(K, -1), (A_SCALE * p.clamp_min(0))[None].expand(K, -1)

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
        crystal = crystal_on and it >= warm
        pred = predict(tidx, crystal)
        mse = ((pred - data[tidx])[:, crop] ** 2).mean() / var
        if robust != 0:
            r = (pred - data[tidx])[:, crop] / np.sqrt(var)
            cs = robust if robust > 0 else -robust * 1.4826 * float((r.detach() - r.detach().median()).abs().median())
            fit = (cs**2 * torch.log1p((r / cs) ** 2)).mean()
        else:
            fit = mse
        loss = fit + lam_tv * tv(vbar) / nv
        if crystal:
            c, q = fields()
            loss = loss + lam_sm * tv_field(c[None] * q) + lam_c * c.mean()
            if it >= ent_from * iters:
                c, q = fields()
                ent = -(q * torch.log(q + 1e-9)).sum(0)
                if sharpen > 0:
                    ramp = (it - ent_from * iters) / ((1 - ent_from) * iters)
                    loss = loss + sharpen * ramp * ent.mean()
                else:
                    loss = loss + lam_ent * (c * ent).mean()
        opt.zero_grad()
        loss.backward()
        if not crystal:
            for t_ in (c_raw, z_raw, p, L1, L2, gam, dphi, s_raw):
                t_.grad = None
        opt.step()
        with torch.no_grad():
            vbar.clamp_(min=0)
            p.clamp_(min=0)
        if it % log_every == 0 or it == iters - 1:
            hist.append(dict(it=it, nmse=float(mse.detach()), L1=float(L1.detach()), L2=float(L2.detach()),
                             gamma=float(torch.rad2deg(gam.detach())), t=time.time() - t0))
            print(f"it {it:4d} nmse {float(mse.detach()):.4f}  |a1| {float(L1.detach()):.3f} |a2| {float(L2.detach()):.3f} "
                  f"gamma {float(torch.rad2deg(gam.detach())):.1f}  {time.time() - t0:.0f}s", flush=True)
    tr, _ = score(train)
    ho, _ = score(held)
    ho0, _ = score(held, crystal=False)
    print(f"final: train nmse {tr:.4f}  held-out nmse {ho:.4f}  (same vbar, crystal off: {ho0:.4f})", flush=True)
    with torch.no_grad():
        pred_all = torch.cat([predict(torch.arange(i, min(i + 12, T)), crystal_on) for i in range(0, T, 12)])
        c, q = fields()
    np.savez(f"runs/sm_{kind}{tag}.npz", pred=pred_all.numpy(), vbar=vbar.detach().numpy(),
             cryst=c.numpy(), q_max=q.max(0).values.numpy(), q_arg=q.argmax(0).numpy(), held=held.numpy(),
             lattice=np.array([float(L1.detach()), float(L2.detach()), float(torch.rad2deg(gam.detach()))]),
             p=p.detach().numpy(), hk=hk.numpy(), hist=json.dumps(hist), s_diffuse=float(torch.sigmoid(s_raw.detach())),
             scores=json.dumps(dict(train=tr, held=ho, held_crystal_off=ho0)))
    return tr, ho


if __name__ == "__main__":
    run(**(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}))
