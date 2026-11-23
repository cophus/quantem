"""Sparse per-voxel crystal-field inversion (2D toy).

Unknowns
  vbar (nv, nv) >= 0         mean phase density, rad / A
  logits (K + 1, nv, nv)     per-voxel softmax over K crystal sets + "empty"
  per set, per 1-degree azimuth bin, S slots:
     p >= 0   strength, a = A_SCALE * p (= kappa^2, intensity-like)
     u        in-plane |g| at the Bragg tilt (> 0.25 1/A)
     dth      sub-bin azimuth offset (+-0.5 deg)
Every slot exists from the start with a small strength, so every cell gets a
gradient (dense in azimuth, a few u starting points per bin). An L1 penalty on
p keeps the cell lists sparse; the softmax makes each voxel carry one set's
whole cell list, which is what localizes a cell seen in only one projection.
"""

import json
import sys
import time

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from fast2d import Fast2D

A_SCALE = 1e-4


def build(cfg, ang, K, S, u0, seed, dev="cpu", init=None):
    g = torch.Generator().manual_seed(seed)
    nv = cfg["n_vox"]
    noise = torch.randn(K + 1, nv, nv, generator=g).numpy()
    noise = np.stack([gaussian_filter(n, 6.0, mode="wrap") for n in noise])
    noise = noise / noise.std() * 1.5
    vbar0 = np.zeros((nv, nv), np.float32)
    if init is not None:  # cluster labels + support from the signature step
        lab, sup, vbar0 = init
        oh = np.stack([(1 - sup)] + [(lab == k) * sup for k in range(K)])
        noise = np.log(oh + 0.05)
    nb = len(ang)
    P = dict(
        vbar=torch.as_tensor(vbar0, dtype=torch.float32),
        logits=torch.as_tensor(noise, dtype=torch.float32),
        p=torch.full((K, nb, S), 0.05),
        u_raw=torch.log(torch.expm1(torch.as_tensor(u0, dtype=torch.float32) - 0.25))
        .view(1, 1, S).repeat(K, nb, 1),
        dth_raw=torch.zeros(K, nb, S),
    )
    for v in P.values():
        v.requires_grad_(True)
    return P


def decode(P, ang):
    W = torch.softmax(P["logits"], 0)[1:]
    K, nb, S = P["p"].shape
    th = torch.as_tensor(ang, dtype=torch.float32).view(1, nb, 1) + 0.5 * torch.tanh(P["dth_raw"])
    u = 0.25 + torch.nn.functional.softplus(P["u_raw"])
    a = A_SCALE * P["p"].clamp_min(0)
    return W, th.reshape(K, -1), u.reshape(K, -1), a.reshape(K, -1)


def match_cells(P, ang, refl, a_frac=0.15):
    """GT reflection recall: a strong GT cell counts as found if some slot with
    a > a_frac * a_true lies within 1 deg and 0.04 1/A of it."""
    W, th, u, a = decode(P, ang)
    th, u, a = th.detach().numpy(), u.detach().numpy(), a.detach().numpy()
    found, strong = 0, 0
    for e in refl:
        at = float(np.hypot(*e["rho"])) ** 2
        if at < 0.3 * (np.pi / 250) ** 2:
            continue
        strong += 1
        dth = np.abs((th - e["theta_c"] + 90) % 180 - 90)
        ok = (dth < 1.0) & (np.abs(u - e["u"]) < 0.04) & (a > a_frac * at)
        found += bool(ok.any())
    n_slots = int((a > 0.15 * (np.pi / 250) ** 2).sum())
    return found, strong, n_slots


def occupancy_score(P, occ_gt):
    """Best one-to-one set -> grain IoU (greedy) on hard assignments."""
    W = torch.softmax(P["logits"], 0).detach().numpy()
    lab = W.argmax(0) - 1  # -1 empty
    gt = np.where(occ_gt.max(0) > 0.5, occ_gt.argmax(0), -1)
    ious = []
    used = set()
    for k in range(occ_gt.shape[0]):
        best, bj = 0.0, None
        for j in range(W.shape[0] - 1):
            if j in used:
                continue
            inter = ((gt == k) & (lab == j)).sum()
            uni = ((gt == k) | (lab == j)).sum()
            iou = inter / max(uni, 1)
            if iou > best:
                best, bj = iou, j
        used.add(bj)
        ious.append(best)
    return ious


def run(kind="few", K=6, S=3, u0=(0.45, 0.75, 1.05), iters=600, batch=45, lam_l1=2e-4,
        lam_ent=0.0, seed=0, warm=60, tag="", log_every=50, init_from_sig=False):
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    d = np.load(f"runs/sim_{kind}.npz")
    cfg = json.loads(str(d["cfg"]))
    refl = json.loads(str(d["refl"]))
    ang = d["angles"]
    data = torch.as_tensor(d["I_coarse"])
    m = Fast2D(cfg, ang)
    crop = slice(m.n_out // 2 - 70, m.n_out // 2 + 70)
    init = None
    if init_from_sig:
        lab = np.load(f"runs/sig_{kind}.npz")["lab"]
        vb = np.load(f"runs/fp_{kind}_s10.npz")["vbar"]
        sm = gaussian_filter(vb, 2.0)
        sup = np.clip(sm / np.percentile(sm, 95), 0, 1)
        init = (lab, sup, vb)
    P = build(cfg, ang, K, S, u0, seed, init=init)
    opt = torch.optim.Adam(
        [
            dict(params=[P["vbar"]], lr=1e-3),
            dict(params=[P["logits"]], lr=0.05),
            dict(params=[P["p"]], lr=0.02),
            dict(params=[P["u_raw"]], lr=0.01),
            dict(params=[P["dth_raw"]], lr=0.05),
        ]
    )
    var = float(data[:, crop].var())
    g = torch.Generator().manual_seed(seed + 1)
    hist = []
    t0 = time.time()
    for it in range(iters):
        tidx = torch.randperm(len(ang), generator=g)[:batch]
        W, th, u, a = decode(P, ang)
        if it < warm:  # mean potential first, no crystal
            a = a * 0
        pred = m.forward(tidx, P["vbar"], W, th, u, a)
        mse = ((pred - data[tidx])[:, crop] ** 2).mean() / var
        l1 = lam_l1 * P["p"].clamp_min(0).sum() / P["p"].shape[0]
        Wall = torch.softmax(P["logits"], 0)
        ent = -(Wall * torch.log(Wall + 1e-9)).sum(0).mean()
        loss = mse + (l1 if it >= warm else 0) + lam_ent * min(1.0, it / iters * 2) * ent
        opt.zero_grad()
        loss.backward()
        if it < warm:
            for k in ("logits", "p", "u_raw", "dth_raw"):
                P[k].grad = None
        opt.step()
        with torch.no_grad():
            P["vbar"].clamp_(min=0)
            P["p"].clamp_(min=0)
        if it % log_every == 0 or it == iters - 1:
            f, s, ns = match_cells(P, ang, refl)
            ious = occupancy_score(P, d["occ"])
            hist.append(dict(it=it, mse=float(mse.detach()), found=f, strong=s, slots=ns, iou=ious,
                             t=time.time() - t0))
            print(f"it {it:4d}  nmse {float(mse):.4f}  cells {f}/{s} (active slots {ns})  "
                  f"IoU {' '.join(f'{x:.2f}' for x in ious)}  {time.time() - t0:.0f}s", flush=True)
    with torch.no_grad():
        W, th, u, a = decode(P, ang)
        full = torch.cat([m.forward(torch.arange(i, min(i + 45, len(ang))), P["vbar"], W, th, u, a)
                          for i in range(0, len(ang), 45)])
    np.savez(
        f"runs/inv_{kind}{tag}.npz",
        pred=full.numpy(),
        vbar=P["vbar"].detach().numpy(),
        W=W.numpy(),
        th=th.numpy(),
        u=u.numpy(),
        a=a.numpy(),
        hist=json.dumps(hist),
    )
    return P


if __name__ == "__main__":
    kw = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    run(**kw)
