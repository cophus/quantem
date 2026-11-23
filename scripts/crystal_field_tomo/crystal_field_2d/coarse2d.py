"""Coarse forward model: mean potential plus Bragg-beam envelopes.

The wave is phi_0(x) + sum_n phi_n(x) exp(2 pi i g_x,n x), every envelope on a
coarse 1 A grid. Per 2 A slice:
  1. all envelopes pick up the mean phase exp(i vbar dz);
  2. phi_0 exchanges amplitude with every active beam through the local
     coupling c_n = rho_n W_n(x, z) exp(2 pi i g_z,n z) (exact exponential of
     the star-shaped coupling matrix, so intensity is conserved);
  3. phi_0 propagates with exp(-i pi lam dz q^2), beam n with
     exp(-i pi lam dz (q + g_x,n)^2), which carries the excitation error and
     the drift of the beam inside the sample.
At the exit each envelope gets its own transfer function exp(-i chi(q + g_x)),
which displaces it by chi'(g_x) / 2 pi. Intensities add incoherently (their
cross terms sit above the coarse Nyquist) and are Fourier-cropped like the data.

Beams are Friedel members of reflections: a reflection is (g in the sample
frame, rho_g complex, set index); the -g member uses conj(rho_g). Only members
with |s| < s_cut at a tilt are carried.
"""

import numpy as np
import torch
import torch.nn.functional as F

from common import chi, crop_intensity, wavelength


class Coarse2D:
    def __init__(self, cfg, angles_deg, n_wave=512, dz=2.0, s_cut=0.04, device="cpu",
                 dtype=torch.float64):
        self.cfg = cfg
        self.lam = wavelength(cfg["energy_ev"])
        self.dev, self.rdt = device, dtype
        self.cdt = torch.complex128 if dtype == torch.float64 else torch.complex64
        self.th = torch.as_tensor(np.radians(angles_deg), dtype=dtype, device=device)
        self.T = len(angles_deg)
        self.L = cfg["n_vox"] * cfg["dx_coarse"]  # reconstruction box (A)
        self.dxw = cfg["fov"] / n_wave
        self.nw = n_wave
        self.x = (torch.arange(n_wave, dtype=dtype, device=device) - n_wave // 2) * self.dxw
        self.q = torch.fft.fftfreq(n_wave, self.dxw, dtype=dtype, device=device)
        self.aa = (self.q.abs() < (2 / 3) * self.q.abs().max()).to(dtype)
        zr = cfg["R"] + 4.0
        self.dz = dz
        self.nz = int(np.ceil(2 * zr / dz))
        self.z = -zr + (torch.arange(self.nz, dtype=dtype, device=device) + 0.5) * dz
        self.s_cut = s_cut
        self.n_out = int(round(cfg["fov"] / cfg["dx_coarse"]))
        # lab sampling grid -> normalised sample coordinates for grid_sample
        c, s = torch.cos(self.th), torch.sin(self.th)
        X = self.x[None, :, None]
        Z = self.z[None, None, :]
        xs = c[:, None, None] * X - s[:, None, None] * Z  # sample x
        zs = s[:, None, None] * X + c[:, None, None] * Z  # sample z
        # map axis 0 (H) = sample x, axis 1 (W) = sample z
        self.grid = torch.stack([zs / (self.L / 2), xs / (self.L / 2)], -1)  # (T, X, Z, 2)

    def rotate(self, maps):
        """maps (C, nv, nv) in the sample frame -> (T, C, X, Z) on the lab grid."""
        inp = maps[None].expand(self.T, -1, -1, -1)
        return F.grid_sample(inp, self.grid, mode="bilinear", padding_mode="zeros",
                             align_corners=False)

    def active_beams(self, g, rho, set_idx):
        """Friedel members with |s| < s_cut per tilt, padded to a common count.

        g (P, 2) sample frame, rho (P,) complex, set_idx (P,) long.
        Returns gx, gz (T, Nb) lab components, rho (T, Nb), set (T, Nb).
        """
        gm = torch.cat([g, -g], 0)
        rm = torch.cat([rho, rho.conj()], 0)
        sm = torch.cat([set_idx, set_idx], 0)
        c, s = torch.cos(self.th)[:, None], torch.sin(self.th)[:, None]
        gx = c * gm[None, :, 0] + s * gm[None, :, 1]
        gz = -s * gm[None, :, 0] + c * gm[None, :, 1]
        sg = -gz - 0.5 * self.lam * (gm**2).sum(-1)[None]
        act = sg.abs() < self.s_cut
        nb = max(int(act.sum(1).max()), 1)
        order = torch.argsort((~act).to(torch.int8), dim=1, stable=True)[:, :nb]
        keep = torch.gather(act, 1, order)
        gx = torch.gather(gx, 1, order) * keep
        gz = torch.gather(gz, 1, order) * keep
        r = rm[order] * keep
        st = sm[order]
        return gx, gz, r, st

    def forward(self, vbar, occ, g=None, rho=None, set_idx=None, return_parts=False):
        """vbar (nv, nv) rad/A, occ (S, nv, nv). Returns coarse images (T, n_out)."""
        maps = torch.cat([vbar[None], occ], 0) if occ is not None else vbar[None]
        rot = self.rotate(maps.to(self.rdt))  # (T, 1+S, X, Z)
        V = rot[:, 0]
        W = rot[:, 1:]
        lam, dz = self.lam, self.dz
        T, X = self.T, self.nw
        phi0 = torch.ones(T, X, dtype=self.cdt, device=self.dev)
        prop0 = torch.exp(-1j * np.pi * lam * dz * self.q**2) * self.aa
        if g is not None and len(g):
            gx, gz, r, st = self.active_beams(g, rho, set_idx)
            nb = gx.shape[1]
            phin = torch.zeros(T, nb, X, dtype=self.cdt, device=self.dev)
            propn = torch.exp(-1j * np.pi * lam * dz * (self.q[None, None] + gx[..., None]) ** 2) * self.aa
            tix = torch.arange(T, device=self.dev)[:, None]
        else:
            nb = 0
        for j in range(self.nz):
            tr = torch.exp(1j * V[..., j] * dz)
            phi0 = phi0 * tr
            if nb:
                phin = phin * tr[:, None]
                Wb = W[..., j][tix, st]  # (T, nb, X)
                cpl = r[..., None] * Wb * torch.exp(2j * np.pi * gz * self.z[j])[..., None]
                K = torch.sqrt((cpl.abs() ** 2).sum(1))  # (T, X)
                a = (cpl.conj() * phin).sum(1)
                sinK = dz * torch.sinc(K * dz / np.pi)
                cm1 = -0.5 * dz**2 * torch.sinc(K * dz / (2 * np.pi)) ** 2
                phin = phin + cpl * (cm1 * a + 1j * sinK * phi0)[:, None]
                phi0 = torch.cos(K * dz) * phi0 + 1j * sinK * a
                phin = torch.fft.ifft(torch.fft.fft(phin) * propn)
            phi0 = torch.fft.ifft(torch.fft.fft(phi0) * prop0)
        cf = self.cfg
        I0 = torch.fft.ifft(torch.fft.fft(phi0) * torch.exp(-1j * chi(self.q, lam, cf["defocus"], cf["cs"]))).abs() ** 2
        I = I0
        if nb:
            ctfn = torch.exp(-1j * chi(self.q[None, None] + gx[..., None], lam, cf["defocus"], cf["cs"]))
            In = torch.fft.ifft(torch.fft.fft(phin) * ctfn).abs() ** 2
            I = I0 + In.sum(1)
        out = crop_intensity(I, self.n_out)
        if return_parts:
            return out, crop_intensity(I0, self.n_out)
        return out
