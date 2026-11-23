"""Per-voxel sub-block multislice inversion, 3D.

Each 2 A voxel holds a sub^3 block of fine potential U (rad / A), indexed [x, y, z]. The tilt
axis is y, so the rotation is the 2D one applied to every y plane: U is passed to grid_sample
with y as channels and nothing is interpolated along y.

  rotate (bilinear in x, z) -> integrate 2 A slices -> multislice exp(i U dz) -> embed in the
  field of view -> defocus -> |psi|^2 -> Fourier crop to the 2 A pixel

Modes
  coarse    U band-limited to the voxel grid (Fourier upsampled), >= 0: mean potential only
  positive  U >= 0, or with floor = beta > 0, U >= -beta x local density (2 A band)
band = b > 0 fits band-limited images: sim3d.band_limit (pass below b x Nyquist, cos^2 taper to
Nyquist, corners removed) on data and model alike, so the pixel checkerboard is not fitted.
Every tenth tilt is held out and only scored.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
from sim3d import band_limit, chi, crop2, wavelength


def fourier_resample3(x, n_out):
    """Fourier crop or zero-pad the last three axes of a real field to n_out, mean kept.
    Sample 0 stays on sample 0."""
    n_in = x.shape[-1]
    ax = (-3, -2, -1)
    X = torch.fft.fftshift(torch.fft.fftn(x, dim=ax), dim=ax)
    c_in, c_out, h = n_in // 2, n_out // 2, min(n_in, n_out) // 2
    si, so = slice(c_in - h + 1, c_in + h), slice(c_out - h + 1, c_out + h)
    Y = torch.zeros(x.shape[:-3] + (n_out,) * 3, dtype=X.dtype)
    Y[..., so, so, so] = X[..., si, si, si]
    return (
        torch.fft.ifftn(torch.fft.ifftshift(Y, dim=ax), dim=ax).real
        * (n_out / n_in) ** 3
    )


def tv3(v):
    d = [
        v[1:, :-1, :-1] - v[:-1, :-1, :-1],
        v[:-1, 1:, :-1] - v[:-1, :-1, :-1],
        v[:-1, :-1, 1:] - v[:-1, :-1, :-1],
    ]
    return torch.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2 + 1e-8).sum()


class SubMS3:
    def __init__(self, cfg, angles_deg, nb=None, sub=7, dz=2.0):
        self.cfg = cfg
        nb = nb or cfg.get("nb", 20)
        self.lam = wavelength(cfg["energy_ev"])
        dv = cfg["dx_coarse"]
        self.nb, self.sub, self.dv = nb, sub, dv
        self.dxs = dv / sub
        self.half = nb * dv / 2
        self.n = n = nb * sub
        self.nsl = round(2 * self.half / dz)
        self.per = round(dz / self.dxs)
        assert self.per * self.nsl == n, (
            "slice thickness must be a whole number of sub-pixels"
        )
        self.ang = torch.as_tensor(angles_deg, dtype=torch.float32)
        # lab x on the detector grid, lab z at sub-pixel centres; y is the sample grid itself
        self.xl = (torch.arange(n, dtype=torch.float32) - n // 2) * self.dxs
        self.zl = -self.half + (torch.arange(n, dtype=torch.float32) + 0.5) * self.dxs
        q = torch.fft.fftfreq(n, self.dxs)
        Q = torch.sqrt(q[:, None] ** 2 + q[None, :] ** 2)
        self.prop = torch.exp(-1j * np.pi * self.lam * dz * Q**2) * (
            Q < 0.95 * q.abs().max()
        )
        self.N = N = round(cfg["fov"] / self.dxs)
        self.j0 = N // 2 - n // 2
        qf = torch.fft.fftfreq(N, self.dxs).double()
        Qf = torch.sqrt(qf[:, None] ** 2 + qf[None, :] ** 2).numpy()
        df = cfg["defocus"] - (
            self.half - (cfg["R"] + 4.0)
        )  # sim3d applies the CTF at its slab exit
        ctf = np.exp(-1j * chi(Qf, self.lam, df, cfg["cs"]))
        # the wave is [y, x]; its y samples sit half a sub-pixel above the detector grid
        ctf = ctf * np.exp(-2j * np.pi * qf.numpy()[:, None] * 0.5 * self.dxs)
        self.ctf = torch.as_tensor(ctf).to(torch.complex64)
        self.n_out = round(cfg["fov"] / dv)
        self.i0 = cfg["n_vox"] // 2 - nb // 2

    def grid(self, tidx):
        th = torch.deg2rad(self.ang[tidx])
        c, s = torch.cos(th)[:, None, None], torch.sin(th)[:, None, None]
        X, Z = self.xl[None, :, None], self.zl[None, None, :]
        xs, zs = c * X - s * Z, s * X + c * Z
        return torch.stack([zs / self.half, xs / self.half], -1)

    def forward(self, tidx, U):
        """Images (B, x, y) at tilts tidx of the fine object U [x, y, z]."""
        B, n = len(tidx), self.n
        obj = U.permute(1, 0, 2)[None].expand(
            B, -1, -1, -1
        )  # (B, y, x, z): y as channels
        lab = F.grid_sample(
            obj,
            self.grid(tidx),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        ph = (
            lab.view(B, n, n, self.nsl, self.per).sum(-1) * self.dxs
        )  # (B, y, x, slice)
        t = torch.exp(1j * ph)
        psi = torch.ones(B, n, n, dtype=torch.complex64)
        for m in range(self.nsl):
            psi = torch.fft.ifft2(torch.fft.fft2(psi * t[..., m]) * self.prop)
        full = torch.ones(B, self.N, self.N, dtype=torch.complex64)
        full[:, self.j0 : self.j0 + n, self.j0 : self.j0 + n] = psi
        img = torch.fft.ifft2(torch.fft.fft2(full) * self.ctf)
        return crop2(img.abs() ** 2, self.n_out).transpose(-1, -2)

    def coarse(self, U):
        """Mean potential on the n_vox grid (Fourier crop of the fine object onto voxel centres)."""
        k = self.sub // 2
        Uc = fourier_resample3(torch.roll(U, (-k, -k, -k), (0, 1, 2)), self.nb)
        nv = self.cfg["n_vox"]
        out = torch.zeros(nv, nv, nv)
        s = slice(self.i0, self.i0 + self.nb)
        out[s, s, s] = Uc
        return out

    def upsample(self, Vc):
        """Fine object from an nb^3 voxel field, band-limited to the voxel grid (inverse of coarse)."""
        k = self.sub // 2
        return torch.roll(fourier_resample3(Vc, self.n), (k, k, k), (0, 1, 2))

    def highpass(self, x):
        """Remove the voxel-grid band; coarse() of the result is 0."""
        if not hasattr(self, "_hp"):
            h, c = self.nb // 2, self.n // 2
            keep = torch.ones(self.n, self.n, self.n)
            keep[c - h + 1 : c + h, c - h + 1 : c + h, c - h + 1 : c + h] = 0
            self._hp = torch.fft.ifftshift(keep)
        return torch.fft.ifftn(torch.fft.fftn(x) * self._hp).real


def true_object(sim, m):
    """Diagnostic only: the true atoms as pixel-integrated 3D Gaussians on the fine grid."""
    from scipy.special import erf

    sig = json.loads(str(sim["fcc"]))["sigma"]
    e = -m.half + np.arange(m.n + 1) * m.dxs
    U = np.zeros((m.n,) * 3)
    w = np.arange(-4, 5)
    for a, p in zip(sim["atoms"].astype(np.float64), sim["P"].astype(np.float64)):
        g = []
        for ax in range(3):
            i = int(np.floor((a[ax] + m.half) / m.dxs)) + w
            i = i[(i >= 0) & (i < m.n)]
            g.append(
                (
                    i,
                    0.5
                    * (
                        erf((e[i + 1] - a[ax]) / (np.sqrt(2) * sig))
                        - erf((e[i] - a[ax]) / (np.sqrt(2) * sig))
                    ),
                )
            )
        U[np.ix_(g[0][0], g[1][0], g[2][0])] += (
            p * np.einsum("i,j,k->ijk", g[0][1], g[1][1], g[2][1]) / m.dxs**3
        )
    return torch.as_tensor(U, dtype=torch.float32)


def flicker_power(U, m, sigma=0.7):
    """Mean square of the high-band part of U in each voxel, Gaussian-smoothed, on the n_vox grid."""
    from scipy.ndimage import gaussian_filter

    hp = m.highpass(torch.as_tensor(U, dtype=torch.float32)).numpy()
    nb, s = m.nb, m.sub
    p = (hp**2).reshape(nb, s, nb, s, nb, s).mean((1, 3, 5))
    nv = m.cfg["n_vox"]
    out = np.zeros((nv,) * 3)
    sl = slice(m.i0, m.i0 + nb)
    out[sl, sl, sl] = gaussian_filter(p, sigma)
    return out


def run(
    kind="twinAm",
    mode="positive",
    iters=1500,
    batch=8,
    lr=3e-3,
    lr_end=0.1,
    init=0.01,
    lam_tv=0.05,
    floor=0.0,
    band=0.0,
    sub=7,
    nb=None,
    seed=0,
    tag="",
    log_every=100,
    threads=3,
):
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    d = np.load(HERE / "runs" / f"sim_{kind}.npz")
    cfg = json.loads(str(d["cfg"]))
    ang = d["angles"]
    data = torch.as_tensor(d["I_coarse"])
    if (
        band > 0
    ):  # fit band-limited images: the same smooth round band limit on data and model
        data = band_limit(data, band).float()
    m = SubMS3(cfg, ang, nb=nb, sub=sub)
    nb = m.nb
    T = len(ang)
    held = torch.arange(5, T, 10)
    train = torch.as_tensor([i for i in range(T) if i not in set(held.tolist())])
    c = slice(
        m.n_out // 2 - 3 * m.n_out // 8, m.n_out // 2 + 3 * m.n_out // 8
    )  # central 3/4 of the image
    var = float(data[:, c, c].var())
    nv = cfg["n_vox"]
    s = slice(m.i0, m.i0 + nb)

    g0 = torch.Generator().manual_seed(seed)
    if mode == "coarse":
        params = [torch.zeros(nb, nb, nb, requires_grad=True)]
    elif mode == "positive":
        params = [(init * torch.rand(m.n, m.n, m.n, generator=g0)).requires_grad_()]
    else:
        raise ValueError(mode)
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: lr_end ** (i / max(iters - 1, 1))
    )

    def obj():
        return m.upsample(params[0]) if mode == "coarse" else params[0]

    def predict(tidx, U):
        p = m.forward(tidx, U)
        return band_limit(p, band) if band > 0 else p

    def score(idx):
        with torch.no_grad():
            U = obj()
            pred = torch.cat(
                [predict(idx[i : i + 16], U) for i in range(0, len(idx), 16)]
            )
            return float(((pred - data[idx])[:, c, c] ** 2).mean() / var), pred

    gen = torch.Generator().manual_seed(seed + 1)
    t0 = time.time()
    hist = []
    for it in range(iters):
        tidx = train[torch.randperm(len(train), generator=gen)[:batch]]
        U = obj()
        pred = predict(tidx, U)
        mse = (((pred - data[tidx])[:, c, c]) ** 2).mean() / var
        # TV per voxel, scaled as in the 2D toy (mean over voxels x 128)
        loss = mse + lam_tv * tv3(m.coarse(U)) * 128 / nv**3
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            if mode == "positive" and floor > 0:  # U >= -floor x local density
                lo = m.upsample(m.coarse(params[0])[s, s, s]).clamp_min(0)
                params[0].copy_(torch.maximum(params[0], -floor * lo))
            else:
                params[0].clamp_(min=0)
        if it % log_every == 0 or it == iters - 1:
            ho_it, _ = score(held)
            hist.append(
                {
                    "it": it,
                    "nmse": float(mse.detach()),
                    "held": ho_it,
                    "t": time.time() - t0,
                }
            )
            print(
                f"it {it:4d} nmse {float(mse.detach()):.4f}  held-out {ho_it:.4f}  {time.time() - t0:.0f}s",
                flush=True,
            )
    tr, _ = score(train)
    ho, _ = score(held)
    print(f"final: train nmse {tr:.4f}  held-out nmse {ho:.4f}", flush=True)
    with torch.no_grad():
        _, pred_all = score(torch.arange(T))
        U = obj()
        vbar = m.coarse(U)
    np.savez_compressed(
        HERE / "runs" / f"ms_{kind}{tag}.npz",
        pred=pred_all.numpy(),
        vbar=vbar.numpy(),
        held=held.numpy(),
        U=U.detach().numpy().astype(np.float32),
        fpow=flicker_power(U.detach(), m).astype(np.float32),
        hist=json.dumps(hist),
        scores=json.dumps({"train": tr, "held": ho}),
        geom=json.dumps(
            {"nb": nb, "sub": sub, "dxs": m.dxs, "half": m.half, "i0": m.i0}
        ),
    )
    return tr, ho


if __name__ == "__main__":
    run(**(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}))


def peak_fraction(U, m, vox, w=29, q_lo=0.3, q_hi=1.4, top=60):
    """Crystallinity readout at voxels vox (n, 3), indices on the nb grid: share of the q_lo..q_hi
    power of the Hann-windowed local 3D spectrum (w^3 sub-pixels around the voxel centre) held by
    its `top` strongest bins. Bragg peaks are points in 3D and amorphous scattering a shell, so a
    crystal scores higher. Voxels whose window leaves the block are skipped."""
    U = np.asarray(U)
    hw = w // 2
    han = np.hanning(w)
    win = han[:, None, None] * han[None, :, None] * han[None, None, :]
    q = np.fft.fftfreq(w, m.dxs)
    Q = np.sqrt(q[:, None, None] ** 2 + q[None, :, None] ** 2 + q[None, None, :] ** 2)
    band = (Q > q_lo) & (Q < q_hi)
    out = []
    for v in vox:
        c = np.asarray(v) * m.sub + m.sub // 2
        p = U[
            c[0] - hw : c[0] + hw + 1,
            c[1] - hw : c[1] + hw + 1,
            c[2] - hw : c[2] + hw + 1,
        ]
        if p.shape != (w, w, w):
            continue
        p = (p - (p * win).sum() / win.sum()) * win
        s = np.abs(np.fft.fftn(p))[band] ** 2
        out.append(np.sort(s)[-top:].sum() / s.sum())
    return np.array(out)
