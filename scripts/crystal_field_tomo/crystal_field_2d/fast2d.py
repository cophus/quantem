"""Fast coarse forward for discovery: no slice loop.

Per tilt and beam, the column amplitude is kinematic,
    A_n(x) = kappa_n * sum_z W_k(x, z) exp(-2 pi i s_n z) dz,
and dynamical saturation is applied per column: with the summed kinematic
intensity I_k = sum_n |A_n|^2, the beams carry I_n = |A_n|^2 sin^2(sqrt(I_k)) / I_k
and phi_0 keeps sqrt(1 - sin^2 sqrt(I_k)) (two-beam result at the Bragg angle).
The mean potential enters as a phase object exp(i sum_z vbar dz). Each beam then
gets its own transfer function exp(-i chi(q + g_x)) and intensities add.

Beams come in slots: set k, slot m has a cell (theta_c [deg], u) and an
intensity-like strength a = kappa^2 >= 0 (linear in a before saturation, so the
gradient at a = 0 is finite). Both Friedel members of a slot are evaluated.
"""

import numpy as np
import torch
import torch.nn.functional as F

from common import chi, crop_intensity, wavelength


class Fast2D:
    def __init__(self, cfg, angles_deg, n_wave=512, dz=2.0, s_cut=0.03, device="cpu",
                 dtype=torch.float32, coherent=False, up=4, x_half=None):
        self.cfg = cfg
        self.lam = wavelength(cfg["energy_ev"])
        self.dev, self.rdt = device, dtype
        self.cdt = torch.complex64 if dtype == torch.float32 else torch.complex128
        self.ang = torch.as_tensor(angles_deg, dtype=dtype, device=device)
        self.L = cfg["n_vox"] * cfg["dx_coarse"]
        self.dxw = cfg["fov"] / n_wave
        self.nw = n_wave
        x = (torch.arange(n_wave, dtype=dtype, device=device) - n_wave // 2) * self.dxw
        xh = self.L / 2 if x_half is None else x_half
        self.sub = torch.nonzero(x.abs() <= xh).squeeze(1)  # wave samples that can hold crystal
        self.xs = x[self.sub]
        self.q = torch.fft.fftfreq(n_wave, self.dxw, dtype=dtype, device=device)
        zr = cfg["R"] + 4.0
        self.dz = dz
        self.nz = int(np.ceil(2 * zr / dz))
        self.z = -zr + (torch.arange(self.nz, dtype=dtype, device=device) + 0.5) * dz
        self.s_cut = s_cut
        self.n_out = int(round(cfg["fov"] / cfg["dx_coarse"]))
        self.ctf0 = torch.exp(-1j * chi(self.q, self.lam, cfg["defocus"], cfg["cs"]))
        self.z_exit = zr
        # coherent recombination: beams are put back on their carriers on a grid
        # fine enough for |g| + envelope band, summed as amplitudes, then cropped
        self.coherent = coherent
        self.up = up
        self.xf = (torch.arange(n_wave * up, dtype=dtype, device=device) - n_wave * up // 2) * self.dxw / up

    def grid(self, tidx):
        th = torch.deg2rad(self.ang[tidx])
        c, s = torch.cos(th)[:, None, None], torch.sin(th)[:, None, None]
        X, Z = self.xs[None, :, None], self.z[None, None, :]
        xs, zs = c * X - s * Z, s * X + c * Z
        return torch.stack([zs / (self.L / 2), xs / (self.L / 2)], -1)

    def members(self, tidx, theta_c, u):
        """Active Friedel members per (tilt, set). theta_c, u: (K, M).
        Returns idx into the 2M members (T, K, N), mask, s, gx (all (T, K, N))."""
        th = self.ang[tidx][:, None, None]
        d = torch.deg2rad((th - theta_c[None] + 90.0) % 180.0 - 90.0)  # (T, K, M)
        sd, cd = torch.sin(d), torch.cos(d)
        uu = u[None].expand_as(d)
        s = torch.cat([uu * sd, -uu * sd], -1) - 0.5 * self.lam * torch.cat([uu, uu], -1) ** 2
        gx = torch.cat([uu * cd, -uu * cd], -1)
        act = s.abs() < self.s_cut
        nb = max(int(act.sum(-1).max()), 1)
        order = torch.argsort((~act).to(torch.int8), dim=-1, stable=True)[..., :nb]
        mask = torch.gather(act, -1, order)
        return order, mask, torch.gather(s, -1, order), torch.gather(gx, -1, order)

    def upsample(self, spec):
        """Spectra on the wave grid (..., nw) -> real-space samples on the fine grid."""
        nw, nf = self.nw, self.nw * self.up
        h = nw // 2
        out = torch.zeros(spec.shape[:-1] + (nf,), dtype=spec.dtype, device=spec.device)
        out[..., :h] = spec[..., :h]
        out[..., nf - h :] = spec[..., h:]
        # the wave grid starts at x = -nw/2 dxw, as does the fine grid, so no phase shift
        return torch.fft.ifft(out) * self.up

    def forward(self, tidx, vbar, W, theta_c, u, a, return_parts=False, phi=None):
        """tidx: tilt indices; vbar (nv, nv); W (K, nv, nv) occupancies;
        theta_c, u, a: (K, M) slot cells and strengths; phi (K, M) optional phase
        of rho_g for the +u member (the -u member gets -phi). Returns (T, n_out)."""
        T = len(tidx)
        K = W.shape[0]
        maps = torch.cat([vbar[None], W], 0)[None].expand(T, -1, -1, -1)
        rot = F.grid_sample(maps, self.grid(tidx), mode="bilinear", padding_mode="zeros",
                            align_corners=False)  # (T, 1+K, X, Z)
        phase = rot[:, 0].sum(-1) * self.dz  # (T, X)
        order, mask, s, gx = self.members(tidx, theta_c, u)
        M = theta_c.shape[1]
        a2 = torch.cat([a, a], -1)  # (K, 2M)
        kap = torch.sqrt(torch.gather(a2[None].expand(T, -1, -1), -1, order).clamp_min(0) + 1e-12)
        kap = kap * mask
        arg = 2 * np.pi * s[..., None] * self.z  # (T, K, N, Z)
        Wr = rot[:, 1:]  # (T, K, X, Z), real
        sc = (kap[:, :, None, :] * self.dz)
        A = torch.complex(torch.matmul(Wr, torch.cos(arg).transpose(-1, -2)) * sc,
                          -torch.matmul(Wr, torch.sin(arg).transpose(-1, -2)) * sc)  # (T, K, X, N)
        Ik_n = A.real**2 + A.imag**2
        Ik = Ik_n.sum((1, 3))  # (T, X)
        sat = torch.where(Ik > 1e-8, torch.sin(torch.sqrt(Ik + 1e-12)) ** 2 / (Ik + 1e-12),
                          1.0 - Ik / 3.0)
        D = (Ik * sat).clamp(max=1.0 - 1e-6)
        ex = torch.exp(1j * phase)
        full0 = torch.zeros(T, self.nw, dtype=self.cdt, device=self.dev)
        full0[:, self.sub] = ex * torch.sqrt(1.0 - D) - 1.0
        full0 = full0 + 1.0
        I0 = torch.fft.ifft(torch.fft.fft(full0) * self.ctf0).abs() ** 2
        psi = A * (torch.sqrt(sat) * ex)[:, None, :, None]  # (T, K, X, N)
        psi = psi.permute(0, 1, 3, 2).reshape(T, -1, len(self.sub))
        fulln = torch.zeros(T, psi.shape[1], self.nw, dtype=self.cdt, device=self.dev)
        fulln[..., self.sub] = psi
        gxf = gx.reshape(T, -1)
        ctfn = torch.exp(-1j * chi(self.q[None, None] + gxf[..., None], self.lam,
                                    self.cfg["defocus"], self.cfg["cs"]))
        if not self.coherent:
            In = (torch.fft.ifft(torch.fft.fft(fulln) * ctfn).abs() ** 2).sum(1)
            out = crop_intensity(I0 + In, self.n_out)
        else:
            # constant phase of each member: i * rho/|rho| * exp(-i pi lam gx^2 z_exit)
            ph = -np.pi * self.lam * gxf**2 * self.z_exit + np.pi / 2
            if phi is not None:
                p2 = torch.cat([phi, -phi], -1)[None].expand(T, -1, -1)
                ph = ph + torch.gather(p2, -1, order).reshape(T, -1)
            spec_n = torch.fft.fft(fulln) * ctfn * torch.exp(1j * ph)[..., None]
            psi_f = self.upsample(spec_n) * torch.exp(2j * np.pi * gxf[..., None] * self.xf)
            psi0_f = self.upsample(torch.fft.fft(full0) * self.ctf0)
            If = (psi0_f + psi_f.sum(1)).abs() ** 2
            out = crop_intensity(If, self.n_out)
        if return_parts:
            return out, crop_intensity(I0, self.n_out), Ik
        return out
