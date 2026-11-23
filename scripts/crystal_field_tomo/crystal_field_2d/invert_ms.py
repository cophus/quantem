"""Per-voxel sub-block multislice inversion.

Each 2 A voxel holds a sub x sub block of a fine potential U (rad / A), 0.29 A sub-pixels
for sub = 7. The tiled blocks are a fine object that can carry lattice fringes, amorphous
speckle and the mean density at once, so Bragg flicker is modelled by the same physics
that made the data rather than by an approximate beam model:

  rotate U to the lab frame (bilinear) -> integrate 2 A slices -> multislice with
  exp(i U dz) -> defocus transfer function -> |psi|^2 -> Fourier crop to the 2 A pixel

Modes
  complex   U = U_r + i U_i free, start 1 + small random complex per sub-pixel transmission
  positive  U real >= 0 (projected after each step), small random positive start; a
            positive object needs mass under every fringe, so crystal can only sit where
            there is material. floor = beta > 0 relaxes this to U >= -beta x local density
            (2 A band): a smooth fringe can then carry |V_g| up to (1 + beta) / 2 of the
            mean, as a crystal's atoms do, without having to form atoms
  coarse    U band-limited to the 2 A pixel (Fourier upsampled from nb x nb), real >= 0:
            the mean-potential-only baseline with the same multislice
  split     U = density + flicker, split by band: the density is band-limited to the 2 A
            pixel and >= 0; the flicker term is free (any sign) and high-passed, so it can
            reproduce the flashes without carrying or inflating the mean
  split_complex   as split, with a complex flicker term (start 1 + small random complex)
Regularisers
  lam_tv     TV on the mean potential (voxel grid)
  lam_spec   group TV on the local amplitude spectra (35 x 35 sub-pixel window per voxel):
             neighbours should hold the same lattice, so the flashes of one region share
             a support (lattice-free tie between flashes)
  spec_norm  use unit-norm spectra in that penalty (only the lattice counts, not its strength)
  lam_at     atomicity: mean sqrt(U + 1e-4), concave, so a fixed mass costs least when it is
             packed into few sharp peaks
  lam_l1     L1 on U
Every tenth tilt is held out and only scored.
"""

import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import chi, crop_intensity, wavelength
from invert_lattice import tv


def fourier_resample2(x, n_out):
    """Fourier crop or zero-pad the last two axes of a real field to n_out, mean kept.
    Sample 0 stays on sample 0."""
    n_in = x.shape[-1]
    X = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    c_in, c_out = n_in // 2, n_out // 2
    h = min(n_in, n_out) // 2
    Y = torch.zeros(x.shape[:-2] + (n_out, n_out), dtype=X.dtype)
    Y[..., c_out - h + 1 : c_out + h, c_out - h + 1 : c_out + h] = X[..., c_in - h + 1 : c_in + h, c_in - h + 1 : c_in + h]
    return torch.fft.ifft2(torch.fft.ifftshift(Y, dim=(-2, -1))).real * (n_out / n_in) ** 2


class SubMS:
    """Forward model on a fine grid of nb x nb voxels, sub x sub samples each, centred on the
    rotation axis. U is indexed [x, z] like the voxel maps."""

    def __init__(self, cfg, angles_deg, nb=88, sub=7, dz=2.0):
        self.cfg = cfg
        self.lam = wavelength(cfg["energy_ev"])
        dv = cfg["dx_coarse"]
        self.nb, self.sub, self.dv = nb, sub, dv
        self.dxs = dv / sub
        self.half = nb * dv / 2
        self.n = nb * sub
        self.nsl = int(round(2 * self.half / dz))
        self.per = int(round(dz / self.dxs))
        assert self.per * self.nsl == self.n, "slice thickness must be a whole number of sub-pixels"
        self.dz = dz
        self.ang = torch.as_tensor(angles_deg, dtype=torch.float32)
        # lab samples: x on the detector-aligned fine grid, z at sub-pixel centres
        self.xl = (torch.arange(self.n, dtype=torch.float32) - self.n // 2) * self.dxs
        self.zl = -self.half + (torch.arange(self.n, dtype=torch.float32) + 0.5) * self.dxs
        q = torch.fft.fftfreq(self.n, self.dxs)
        self.prop = torch.exp(-1j * np.pi * self.lam * dz * q**2) * (q.abs() < 0.95 * q.abs().max())
        self.N = int(round(cfg["fov"] / self.dxs))
        self.j0 = self.N // 2 - self.n // 2
        Q = torch.fft.fftfreq(self.N, self.dxs).double()
        z_exit_data = cfg["R"] + 4.0  # sim2d applies the CTF at the end of its slab
        df = cfg["defocus"] - (self.half - z_exit_data)
        self.ctf = torch.exp(-1j * torch.as_tensor(chi(Q.numpy(), self.lam, df, cfg["cs"]))).to(torch.complex64)
        self.n_out = int(round(cfg["fov"] / dv))
        # coarse voxel i of the n_vox grid that holds block voxel 0
        self.i0 = cfg["n_vox"] // 2 - nb // 2

    def grid(self, tidx):
        th = torch.deg2rad(self.ang[tidx])
        c, s = torch.cos(th)[:, None, None], torch.sin(th)[:, None, None]
        X, Z = self.xl[None, :, None], self.zl[None, None, :]
        xs, zs = c * X - s * Z, s * X + c * Z
        return torch.stack([zs / self.half, xs / self.half], -1)

    def forward(self, tidx, Ur, Ui=None):
        B = len(tidx)
        obj = Ur[None, None] if Ui is None else torch.stack([Ur, Ui])[None]
        lab = F.grid_sample(obj.expand(B, -1, -1, -1), self.grid(tidx), mode="bilinear", padding_mode="zeros",
                            align_corners=False)
        ph = lab.view(B, lab.shape[1], self.n, self.nsl, self.per).sum(-1) * self.dxs  # (B, C, x, slice)
        t = torch.exp(1j * ph[:, 0]) if Ui is None else torch.exp(1j * ph[:, 0] - ph[:, 1])
        psi = torch.ones(B, self.n, dtype=torch.complex64)
        for m in range(self.nsl):
            psi = torch.fft.ifft(torch.fft.fft(psi * t[:, :, m]) * self.prop)
        full = torch.ones(B, self.N, dtype=torch.complex64)
        full = torch.cat([full[:, : self.j0], psi, full[:, self.j0 + self.n :]], 1)
        img = torch.fft.ifft(torch.fft.fft(full) * self.ctf)
        return crop_intensity(img.abs() ** 2, self.n_out)

    def coarse(self, U):
        """Mean potential on the voxel grid (Fourier crop of the fine object, centred on voxels)."""
        k = self.sub // 2
        Uc = fourier_resample2(torch.roll(U, (-k, -k), (0, 1)), self.nb)
        out = torch.zeros(self.cfg["n_vox"], self.cfg["n_vox"])
        out[self.i0 : self.i0 + self.nb, self.i0 : self.i0 + self.nb] = Uc
        return out

    def highpass(self, x):
        """Remove the band the voxel grid holds (|q_x|, |q_z| < 1 / (2 dv)); coarse() of the result is 0."""
        if not hasattr(self, "_hp"):
            h = self.nb // 2
            c = self.n // 2
            keep = torch.ones(self.n, self.n)
            keep[c - h + 1 : c + h, c - h + 1 : c + h] = 0
            self._hp = torch.fft.ifftshift(keep)
        return torch.fft.ifft2(torch.fft.fft2(x) * self._hp).real

    def upsample(self, Vc):
        """Fine object from a voxel-grid field, band-limited to the 2 A pixel (inverse of coarse)."""
        k = self.sub // 2
        return torch.roll(fourier_resample2(Vc, self.n), (k, k), (0, 1))


def local_spectra(U, m, w=35, q_lo=0.3, q_hi=1.4):
    """Hann-windowed local amplitude spectra of U around every voxel centre, band
    q_lo..q_hi only, shape (nb, nb, n_band). Differentiable."""
    p = w // 2 - m.sub // 2
    patches = F.unfold(F.pad(U[None, None], (p, p, p, p)), w, stride=m.sub)[0].T.view(-1, w, w)
    han = torch.hann_window(w, periodic=False)
    win = han[:, None] * han[None, :]
    q = torch.fft.fftfreq(w, m.dxs)
    Q = torch.sqrt(q[:, None] ** 2 + q[None, :] ** 2)
    band = ((Q > q_lo) & (Q < q_hi)).view(-1)
    A = torch.fft.fft2(patches * win).abs().view(len(patches), -1)[:, band] / win.sum()
    return A.view(m.nb, m.nb, -1)


def spectral_group_tv(A):
    """Mean over neighbour voxel pairs of the l2 distance between local amplitude spectra:
    a region of one lattice is cheap, a boundary costs the same whatever crosses it."""
    dx = torch.sqrt(((A[1:] - A[:-1]) ** 2).sum(-1) + 1e-12).mean()
    dz = torch.sqrt(((A[:, 1:] - A[:, :-1]) ** 2).sum(-1) + 1e-12).mean()
    return 0.5 * (dx + dz)


def true_object(sim, m):
    """Diagnostic only: the true atoms as 2D Gaussians on the fine grid (rad / A)."""
    atoms = sim["atoms"].astype(np.float64)
    phases = json.loads(str(sim["phases"]))
    grains = json.loads(str(sim["grains"]))
    sig = phases["A"]["sigma"]
    strength = []
    for gr in grains:
        ph = phases["A" if gr["phase"] == "amorphous" else gr["phase"]]
        if len(ph["basis"]) != 1:
            raise NotImplementedError("multi-atom basis")
        strength.append(ph["basis"][0][1])
    P = np.asarray(strength)[sim["gid"]]
    from scipy.special import erf

    e = -m.half + np.arange(m.n + 1) * m.dxs  # pixel edges; atoms integrated over pixels
    U = np.zeros((m.n, m.n))
    w = np.arange(-4, 5)
    for (ax, az), p in zip(atoms, P):
        ix = int(np.floor((ax + m.half) / m.dxs)) + w
        iz = int(np.floor((az + m.half) / m.dxs)) + w
        ix, iz = ix[(ix >= 0) & (ix < m.n)], iz[(iz >= 0) & (iz < m.n)]
        gx = 0.5 * (erf((e[ix + 1] - ax) / (np.sqrt(2) * sig)) - erf((e[ix] - ax) / (np.sqrt(2) * sig)))
        gz = 0.5 * (erf((e[iz + 1] - az) / (np.sqrt(2) * sig)) - erf((e[iz] - az) / (np.sqrt(2) * sig)))
        U[np.ix_(ix, iz)] += p * np.outer(gx, gz) / m.dxs**2
    return torch.as_tensor(U, dtype=torch.float32)


def run(kind="fewAm", mode="positive", iters=2000, batch=16, lr=3e-3, lr_end=0.1, init=0.01, lam_tv=0.05,
        lam_spec=0.0, spec_norm=False, lam_at=0.0, lam_l1=0.0, robust=0.0, lr_density=1e-3, floor=0.0, sub=7, nb=88, seed=0, tag="", log_every=100, threads=3):
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    d = np.load(f"runs/sim_{kind}.npz")
    cfg = json.loads(str(d["cfg"]))
    ang = d["angles"]
    data = torch.as_tensor(d["I_coarse"])
    m = SubMS(cfg, ang, nb=nb, sub=sub)
    T = len(ang)
    held = torch.arange(5, T, 10)
    train = torch.as_tensor([i for i in range(T) if i not in set(held.tolist())])
    crop = slice(m.n_out // 2 - 70, m.n_out // 2 + 70)
    var = float(data[:, crop].var())
    nv = cfg["n_vox"]

    g0 = torch.Generator().manual_seed(seed)
    if mode == "coarse":
        params = [torch.zeros(nb, nb, requires_grad=True)]
    elif mode == "positive":
        params = [(init * torch.rand(m.n, m.n, generator=g0)).requires_grad_()]
    elif mode in ("split", "split_complex"):
        params = [torch.zeros(nb, nb, requires_grad=True), (init * torch.randn(m.n, m.n, generator=g0)).requires_grad_()]
        if mode == "split_complex":
            params.append((init * torch.randn(m.n, m.n, generator=g0)).requires_grad_())
    elif mode == "complex":
        params = [(init * torch.randn(m.n, m.n, generator=g0)).requires_grad_(),
                  (init * torch.randn(m.n, m.n, generator=g0)).requires_grad_()]
    else:
        raise ValueError(mode)
    groups = [dict(params=params, lr=lr)]
    if mode.startswith("split"):  # density at the coarse-baseline rate, flicker at lr
        groups = [dict(params=params[:1], lr=lr_density), dict(params=params[1:], lr=lr)]
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: lr_end ** (i / max(iters - 1, 1)))

    def obj():
        if mode == "coarse":
            return m.upsample(params[0]), None
        if mode == "split":
            return m.upsample(params[0]) + m.highpass(params[1]), None
        if mode == "split_complex":
            return m.upsample(params[0]) + m.highpass(params[1]), m.highpass(params[2])
        if mode == "positive":
            return params[0], None
        return params[0], params[1]

    def predict(tidx):
        Ur, Ui = obj()
        return m.forward(tidx, Ur, Ui)

    def score(idx):
        with torch.no_grad():
            pred = torch.cat([predict(idx[i : i + 16]) for i in range(0, len(idx), 16)])
            return float(((pred - data[idx])[:, crop] ** 2).mean() / var), pred

    gen = torch.Generator().manual_seed(seed + 1)
    t0 = time.time()
    hist = []
    for it in range(iters):
        tidx = train[torch.randperm(len(train), generator=gen)[:batch]]
        pred = predict(tidx)
        r = (pred - data[tidx])[:, crop] / np.sqrt(var)
        mse = (r**2).mean()
        if robust != 0:
            cs = robust if robust > 0 else -robust * 1.4826 * float((r.detach() - r.detach().median()).abs().median())
            fit = (cs**2 * torch.log1p((r / cs) ** 2)).mean()
        else:
            fit = mse
        Ur, _ = obj()
        loss = fit + lam_tv * tv(m.coarse(Ur)) / nv
        if lam_spec > 0:
            A = local_spectra(Ur, m)
            if spec_norm:
                A = A / (A.norm(dim=-1, keepdim=True) + 1e-3)
            loss = loss + lam_spec * spectral_group_tv(A)
        if lam_at > 0:
            loss = loss + lam_at * torch.sqrt(Ur.clamp_min(0) + 1e-4).mean()
        if lam_l1 > 0:
            loss = loss + lam_l1 * Ur.abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if mode in ("positive", "coarse", "split", "split_complex"):
            with torch.no_grad():
                if mode == "positive" and floor > 0:  # U >= -floor x local density
                    lo = m.upsample(m.coarse(params[0])[m.i0 : m.i0 + nb, m.i0 : m.i0 + nb]).clamp_min(0)
                    params[0].copy_(torch.maximum(params[0], -floor * lo))
                else:
                    params[0].clamp_(min=0)
        if it % log_every == 0 or it == iters - 1:
            ho_it, _ = score(held)
            hist.append(dict(it=it, nmse=float(mse.detach()), held=ho_it, t=time.time() - t0))
            print(f"it {it:4d} nmse {float(mse.detach()):.4f}  held-out {ho_it:.4f}  {time.time() - t0:.0f}s", flush=True)
    tr, _ = score(train)
    ho, _ = score(held)
    print(f"final: train nmse {tr:.4f}  held-out nmse {ho:.4f}", flush=True)
    with torch.no_grad():
        _, pred_all = score(torch.arange(T))
        Ur, Ui = obj()
        vbar = m.coarse(Ur)
    np.savez_compressed(f"runs/ms_{kind}{tag}.npz", pred=pred_all.numpy(), vbar=vbar.numpy(), held=held.numpy(),
                        U=Ur.detach().numpy().astype(np.float32),
                        flicker=(m.highpass(Ur.detach()).numpy().astype(np.float32)),
                        Ui=(Ui.detach().numpy().astype(np.float32) if Ui is not None else np.zeros(0)),
                        hist=json.dumps(hist), scores=json.dumps(dict(train=tr, held=ho)),
                        geom=json.dumps(dict(nb=nb, sub=sub, dxs=m.dxs, half=m.half, i0=m.i0)))
    return tr, ho


if __name__ == "__main__":
    run(**(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}))


def local_order(U, m, w=49, q_lo=0.3, q_hi=1.4, top=24):
    """Crystallinity readout from a fine object: for every voxel, the Hann-windowed local
    spectrum of U (w x w sub-pixels around the voxel centre, local mean removed). Returns
    voxel maps on the n_vox grid of the peak fraction (share of the q_lo..q_hi power in the
    `top` strongest bins: high for a few Bragg peaks, low for an amorphous ring) and of the
    band power relative to the local mean squared."""
    U = torch.as_tensor(U, dtype=torch.float32)
    p = w // 2 - m.sub // 2
    patches = F.unfold(F.pad(U[None, None], (p, p, p, p)), w, stride=m.sub)[0].T.view(-1, w, w)
    han = torch.hann_window(w, periodic=False)
    win = han[:, None] * han[None, :]
    mean = (patches * win).sum((1, 2)) / win.sum()
    S = torch.fft.fft2((patches - mean[:, None, None]) * win).abs() ** 2
    q = torch.fft.fftfreq(w, m.dxs)
    Q = torch.sqrt(q[:, None] ** 2 + q[None, :] ** 2)
    band = ((Q > q_lo) & (Q < q_hi)).view(-1)
    Sb = S.view(len(S), -1)[:, band]
    tot = Sb.sum(1)
    frac = Sb.topk(top, dim=1).values.sum(1) / tot.clamp_min(1e-20)
    power = tot / (mean**2 * win.sum() ** 2).clamp_min(1e-20)
    out = []
    for v in (frac, power):
        o = torch.zeros(m.cfg["n_vox"], m.cfg["n_vox"])
        o[m.i0 : m.i0 + m.nb, m.i0 : m.i0 + m.nb] = v.view(m.nb, m.nb)
        out.append(o.numpy())
    return out


def flicker_power(U, m, sigma=1.0):
    """Mean square of the high-band part of U (the part that makes the flicker) in each
    voxel, Gaussian-smoothed over sigma voxels, on the n_vox grid."""
    from scipy.ndimage import gaussian_filter

    hp = m.highpass(torch.as_tensor(U, dtype=torch.float32)).numpy()
    p = (hp**2).reshape(m.nb, m.sub, m.nb, m.sub).mean((1, 3))
    out = np.zeros((m.cfg["n_vox"], m.cfg["n_vox"]))
    out[m.i0 : m.i0 + m.nb, m.i0 : m.i0 + m.nb] = gaussian_filter(p, sigma)
    return out
