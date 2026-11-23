"""Oracle data without Bragg scattering (diagnostic only).

Same sample, same tilts, same optics as sim2d, but each slice's potential is low-passed
to |q_x| < q_lp before the transmission function, so nothing scatters beyond the coarse
pixel's band: no Bragg beams, no high-angle amorphous scattering. What is left is the
image of the mean potential, so fitting it bounds how well any flicker model could
recover the density. The difference from the real data is the true flicker.

Run:  python oracle.py fewAm    -> runs/sim_fewAm_nobragg.npz
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

import sim2d
from common import chi, crop_intensity, wavelength

HERE = Path(__file__).parent


def multislice_lowpass(sample, angles_deg, cfg, q_lp, chunk=12):
    lam = wavelength(cfg["energy_ev"])
    dx, dz = cfg["dx_fine"], cfg["dz_fine"]
    n = int(round(cfg["fov"] / dx))
    q = np.fft.fftfreq(n, dx)
    aa = np.abs(q) < (2 / 3) * q.max()
    prop = torch.as_tensor(np.exp(-1j * np.pi * lam * dz * q**2) * aa)
    ctf = torch.as_tensor(np.exp(-1j * chi(q, lam, cfg["defocus"], cfg["cs"])))
    lp = torch.as_tensor(np.abs(q) < q_lp)
    z0 = -(sample["R"] + 4.0)
    n_sl = int(np.ceil(2 * (sample["R"] + 4.0) / dz))
    sig = sim2d.PHASES["A"]["sigma"]
    win = np.arange(-10, 11)
    atoms, P = sample["atoms"], sample["P"]
    out = np.zeros((len(angles_deg), n))
    for c0 in range(0, len(angles_deg), chunk):
        th = np.radians(angles_deg[c0 : c0 + chunk])
        T = len(th)
        xl = np.cos(th)[:, None] * atoms[:, 0] + np.sin(th)[:, None] * atoms[:, 1]
        zl = -np.sin(th)[:, None] * atoms[:, 0] + np.cos(th)[:, None] * atoms[:, 1]
        sl = np.floor((zl - z0) / dz).astype(np.int64)
        px = np.round(xl / dx).astype(np.int64)[..., None] + n // 2 + win
        val = P[None, :, None] * np.exp(-0.5 * ((px - n // 2) * dx - xl[..., None]) ** 2 / sig**2) / (
            np.sqrt(2 * np.pi) * sig
        )
        flat = (np.arange(T)[:, None, None] * n_sl + sl[..., None]) * n + px
        pot = torch.zeros(T * n_sl * n, dtype=torch.float64)
        pot.index_add_(0, torch.as_tensor(flat.ravel()), torch.as_tensor(val.ravel()))
        pot = torch.fft.ifft(torch.fft.fft(pot.view(T, n_sl, n)) * lp).real
        psi = torch.ones(T, n, dtype=torch.complex128)
        for s in range(n_sl):
            psi = torch.fft.ifft(torch.fft.fft(psi * torch.exp(1j * pot[:, s])) * prop)
        out[c0 : c0 + T] = (torch.fft.ifft(torch.fft.fft(psi) * ctf).abs() ** 2).numpy()
    return out


def main(kind, q_lp=0.25):
    cfg = dict(sim2d.CONFIG)
    rng = np.random.default_rng(cfg["seed"])
    sim2d.calibrate_phases()
    sample = sim2d.build_sample(kind, rng)
    angles = np.arange(-90.0, 90.0, cfg["tilt_step"])
    torch.set_num_threads(3)
    I_fine = multislice_lowpass(sample, angles, cfg, q_lp)
    n_coarse = int(round(cfg["fov"] / cfg["dx_coarse"]))
    I_coarse = crop_intensity(torch.as_tensor(I_fine), n_coarse).numpy()
    ref = dict(np.load(HERE / "runs" / f"sim_{kind}.npz"))
    assert np.allclose(ref["atoms"], sample["atoms"].astype(np.float32)), "sample differs from sim2d"
    ref["I_coarse"] = I_coarse.astype(np.float32)
    ref.pop("I_fine")
    np.savez_compressed(HERE / "runs" / f"sim_{kind}_nobragg.npz", **ref)
    print(f"{kind}: no-Bragg data -> runs/sim_{kind}_nobragg.npz")


if __name__ == "__main__":
    main(sys.argv[1])
