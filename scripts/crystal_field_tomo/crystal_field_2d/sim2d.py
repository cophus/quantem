"""Fine 2D multislice ground truth for the crystal-field tomography toy.

Builds a polycrystalline 2D particle out of Gaussian atoms (several phases,
grains, an optional twin), rotates it through a tilt series, runs multislice
on a fine grid (atoms resolved), applies the objective transfer function with
no aperture, and Fourier-crops the intensities to a coarse pixel that cannot
resolve any lattice plane.

Run:  python sim2d.py few      (twin pair + one grain of another phase)
      python sim2d.py many     (Voronoi polycrystal, three phases)
      python sim2d.py fewAm    (twin pair of phase A + an amorphous cap)
A suffix A makes every crystalline grain phase A; a suffix m makes the cap
(few) or every third grain (many) amorphous, at the density of phase A.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common import cell_of, chi, crop_intensity, lattice_matrix, phase_reflections, wavelength

HERE = Path(__file__).parent

PHASES = {
    # oblique lattice: no mirror along a1, so a mirror twin is a distinct grain
    "A": dict(a1=(2.70, 0.0), a2=(0.95, 2.45), basis=[((0, 0), 1.0)], sigma=0.20, xi=180.0),
    "B": dict(a1=(2.90, 0.0), a2=(1.45, 2.5115), basis=[((0, 0), 1.0)], sigma=0.20, xi=250.0),
    "C": dict(
        a1=(2.30, 0.0), a2=(0.0, 3.40), basis=[((0, 0), 1.0), ((0.5, 0.5), 0.5)], sigma=0.20, xi=220.0
    ),
}

CONFIG = dict(
    energy_ev=300e3,
    defocus=2500.0,  # Angstrom
    cs=0.0,
    fov=512.0,
    dx_fine=0.125,
    dz_fine=1.0,
    dx_coarse=2.0,
    n_vox=128,  # coarse reconstruction grid (2 A voxels, 256 A box)
    tilt_step=1.0,
    seed=0,
)


def calibrate_phases():
    """Scale each phase's atom strengths so its strongest reflection has the
    requested extinction distance xi = pi / |rho_g|."""
    for name, ph in PHASES.items():
        g, rho = phase_reflections(ph, 0.0)
        scale = (np.pi / ph["xi"]) / np.abs(rho).max()
        ph["basis"] = [(f, P * scale) for f, P in ph["basis"]]


def amorphous_atoms(R, rng, d_min=2.1, n_relax=80):
    """Dense random packing over the disk of radius R at the number density of
    phase A: uniform points pushed apart until no pair is closer than d_min."""
    from scipy.spatial import cKDTree

    A = lattice_matrix(PHASES["A"], 0.0)
    n = int(round(np.pi * R**2 / abs(np.linalg.det(A))))
    r = R * np.sqrt(rng.uniform(size=n))
    t = rng.uniform(0, 2 * np.pi, n)
    x = np.stack([r * np.cos(t), r * np.sin(t)], -1)
    for _ in range(n_relax):
        pr = cKDTree(x).query_pairs(d_min, output_type="ndarray")
        if len(pr) == 0:
            break
        dv = x[pr[:, 1]] - x[pr[:, 0]]
        dist = np.linalg.norm(dv, axis=1, keepdims=True) + 1e-9
        push = 0.5 * (d_min - dist) * dv / dist
        np.add.at(x, pr[:, 0], -push)
        np.add.at(x, pr[:, 1], push)
        rr = np.linalg.norm(x, axis=1, keepdims=True)
        x = np.where(rr > R, x * R / rr, x)
    return x


def build_sample(kind, rng):
    """Grains as (phase, orient, mirror, region-test). Returns atoms (n, 2),
    strengths (n,), grain id (n,), and the grain table. A kind ending in "A"
    (fewA, manyA) makes every grain phase A; a trailing "m" adds amorphous grains."""
    amorph = kind.endswith("m")
    kind = kind[:-1] if amorph else kind
    single = kind.endswith("A")
    kind = kind[:-1] if single else kind
    if kind == "few":
        R = 80.0
        beta = 35.0  # twin boundary through the origin
        tb = np.radians(beta)
        n_tw = np.array([-np.sin(tb), np.cos(tb)])
        cap = np.array([np.cos(np.radians(-60)), np.sin(np.radians(-60))])
        grains = [
            dict(phase="A", orient=beta, mirror=None),
            dict(phase="A", orient=beta, mirror=beta),  # mirror twin of grain 0
            dict(phase="C", orient=float(rng.uniform(0, 180)), mirror=None),
        ]

        def label(r):
            lab = np.where(r @ n_tw > 0, 0, 1)
            lab = np.where(r @ cap > 35.0, 2, lab)
            return np.where(np.linalg.norm(r, axis=-1) <= R, lab, -1)

    elif kind == "many":
        R = 100.0
        n_g = 12
        seeds = []
        while len(seeds) < n_g:
            p = rng.uniform(-R, R, 2)
            if np.linalg.norm(p) < R * 0.9:
                seeds.append(p)
        seeds = np.array(seeds)
        names = list(PHASES)
        grains = [
            dict(phase=names[rng.integers(len(names))], orient=float(rng.uniform(0, 180)), mirror=None)
            for _ in range(n_g)
        ]

        def label(r):
            d = np.linalg.norm(r[..., None, :] - seeds, axis=-1)
            lab = np.argmin(d, axis=-1)
            return np.where(np.linalg.norm(r, axis=-1) <= R, lab, -1)

    else:
        raise ValueError(kind)
    if single:
        for gr in grains:
            gr["phase"] = "A"
    if amorph:
        for k in ([2] if kind == "few" else range(2, len(grains), 3)):
            grains[k] = dict(phase="amorphous", orient=0.0, mirror=None)

    atoms, P, gid = [], [], []
    for k, gr in enumerate(grains):
        if gr["phase"] == "amorphous":
            r = amorphous_atoms(R, rng)
            keep = label(r) == k
            atoms.append(r[keep])
            P.append(np.full(keep.sum(), PHASES["A"]["basis"][0][1]))
            gid.append(np.full(keep.sum(), k))
            continue
        ph = PHASES[gr["phase"]]
        A = lattice_matrix(ph, gr["orient"], gr["mirror"])
        n = int(np.ceil(1.6 * R / min(np.linalg.norm(A, axis=0)))) + 2
        ij = np.stack(np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1), indexing="ij"), -1)
        ij = ij.reshape(-1, 2).astype(np.float64)
        for frac, Pj in ph["basis"]:
            r = (ij + np.asarray(frac)) @ A.T
            keep = label(r) == k
            atoms.append(r[keep])
            P.append(np.full(keep.sum(), Pj))
            gid.append(np.full(keep.sum(), k))
    atoms, P, gid = np.concatenate(atoms), np.concatenate(P), np.concatenate(gid)
    atoms = atoms + rng.normal(scale=0.03, size=atoms.shape)  # small static disorder
    return dict(atoms=atoms, P=P, gid=gid, grains=grains, R=R, label=label)


def multislice_tilts(sample, angles_deg, cfg, chunk=12, device="cpu"):
    """Fine multislice for every tilt. Returns fine intensities (n_tilt, n_fine)."""
    lam = wavelength(cfg["energy_ev"])
    dx, dz = cfg["dx_fine"], cfg["dz_fine"]
    n = int(round(cfg["fov"] / dx))
    x = (np.arange(n) - n // 2) * dx
    q = np.fft.fftfreq(n, dx)
    aa = np.abs(q) < (2 / 3) * q.max()
    prop = torch.as_tensor(np.exp(-1j * np.pi * lam * dz * q**2) * aa, device=device)
    ctf = torch.as_tensor(np.exp(-1j * chi(q, lam, cfg["defocus"], cfg["cs"])), device=device)

    z0 = -(sample["R"] + 4.0)
    n_sl = int(np.ceil(2 * (sample["R"] + 4.0) / dz))
    sig = PHASES["A"]["sigma"]
    win = np.arange(-10, 11)
    atoms = sample["atoms"]
    P = sample["P"]
    out = np.zeros((len(angles_deg), n), dtype=np.float64)
    for c0 in range(0, len(angles_deg), chunk):
        th = np.radians(angles_deg[c0 : c0 + chunk])
        T = len(th)
        xl = np.cos(th)[:, None] * atoms[:, 0] + np.sin(th)[:, None] * atoms[:, 1]
        zl = -np.sin(th)[:, None] * atoms[:, 0] + np.cos(th)[:, None] * atoms[:, 1]
        sl = np.floor((zl - z0) / dz).astype(np.int64)
        px0 = np.round(xl / dx).astype(np.int64) + n // 2
        px = px0[..., None] + win  # (T, N, 21)
        val = (
            P[None, :, None]
            * np.exp(-0.5 * ((px - n // 2) * dx - xl[..., None]) ** 2 / sig**2)
            / (np.sqrt(2 * np.pi) * sig)
        )
        flat = (np.arange(T)[:, None, None] * n_sl + sl[..., None]) * n + px
        pot = torch.zeros(T * n_sl * n, dtype=torch.float64, device=device)
        pot.index_add_(0, torch.as_tensor(flat.ravel(), device=device), torch.as_tensor(val.ravel()))
        pot = pot.view(T, n_sl, n)
        psi = torch.ones(T, n, dtype=torch.complex128, device=device)
        for s in range(n_sl):
            psi = torch.fft.ifft(torch.fft.fft(psi * torch.exp(1j * pot[:, s])) * prop)
        img = torch.fft.ifft(torch.fft.fft(psi) * ctf)
        out[c0 : c0 + T] = (img.abs() ** 2).cpu().numpy()
    return out, x


def coarse_truth(sample, cfg):
    """Ground truth on the coarse voxel grid: low-passed mean phase density
    (rad / A), grain occupancy fractions, and per-grain reflections (none for
    amorphous grains)."""
    nv, dv = cfg["n_vox"], cfg["dx_coarse"]
    L = nv * dv
    up = 8
    nf = nv * up
    df = dv / up
    dens = np.zeros((nf, nf))
    # fine sample 0 sits on the centre of coarse voxel 0, so the Fourier crop
    # lands on coarse voxel centres (i + 0.5) dv - L / 2
    ij = (sample["atoms"] - (0.5 * dv - L / 2)) / df
    i0 = np.floor(ij).astype(int)
    w = ij - i0
    for di in (0, 1):
        for dj in (0, 1):
            wt = (w[:, 0] if di else 1 - w[:, 0]) * (w[:, 1] if dj else 1 - w[:, 1])
            np.add.at(dens, ((i0[:, 0] + di) % nf, (i0[:, 1] + dj) % nf), sample["P"] * wt)
    dens /= df**2
    F = np.fft.fftshift(np.fft.fft2(dens))
    qf = np.fft.fftshift(np.fft.fftfreq(nf, df))
    Q2 = qf[:, None] ** 2 + qf[None, :] ** 2
    F *= np.exp(-2 * np.pi**2 * PHASES["A"]["sigma"] ** 2 * Q2)
    c = nf // 2
    Fc = F[c - nv // 2 : c + nv // 2, c - nv // 2 : c + nv // 2].copy()
    Fc[0, :] = 0
    Fc[:, 0] = 0  # drop Nyquist rows
    vbar = np.fft.ifft2(np.fft.ifftshift(Fc)).real * (nv * nv) / (nf * nf)

    # occupancy fractions by 4x4 supersampling
    ss = 4
    t = (np.arange(nv * ss) + 0.5) / ss * dv - L / 2
    X, Z = np.meshgrid(t, t, indexing="ij")
    lab = sample["label"](np.stack([X, Z], -1))
    n_g = len(sample["grains"])
    occ = np.stack([(lab == k).reshape(nv, ss, nv, ss).mean((1, 3)) for k in range(n_g)])

    refl = []
    for k, gr in enumerate(sample["grains"]):
        if gr["phase"] == "amorphous":
            continue
        g, rho = phase_reflections(PHASES[gr["phase"]], gr["orient"], gr["mirror"])
        th_c, u, sign = cell_of(g)
        for gi, ri, ti, ui, si in zip(g, rho, th_c, u, sign):
            if si > 0:  # one entry per Friedel pair, the +u member
                refl.append(dict(grain=k, g=gi.tolist(), rho=[ri.real, ri.imag], theta_c=ti, u=ui))
    return vbar, occ, refl


def main(kind):
    cfg = dict(CONFIG)
    rng = np.random.default_rng(cfg["seed"])
    calibrate_phases()
    sample = build_sample(kind, rng)
    angles = np.arange(-90.0, 90.0, cfg["tilt_step"])
    torch.set_num_threads(4)
    t0 = time.time()
    I_fine, x = multislice_tilts(sample, angles, cfg)
    t1 = time.time()
    n_coarse = int(round(cfg["fov"] / cfg["dx_coarse"]))
    I_coarse = crop_intensity(torch.as_tensor(I_fine), n_coarse).numpy()
    vbar, occ, refl = coarse_truth(sample, cfg)
    cfg["R"] = sample["R"]  # sets the slab and the exit (defocus reference) plane
    out = HERE / "runs" / f"sim_{kind}.npz"
    np.savez_compressed(
        out,
        angles=angles,
        I_coarse=I_coarse.astype(np.float32),
        I_fine=I_fine.astype(np.float16),
        vbar=vbar.astype(np.float32),
        occ=occ.astype(np.float32),
        cryst=occ[[k for k, gr in enumerate(sample["grains"]) if gr["phase"] != "amorphous"]].sum(0).astype(np.float32),
        atoms=sample["atoms"].astype(np.float32),
        gid=sample["gid"],
        refl=json.dumps(refl),
        grains=json.dumps(sample["grains"]),
        cfg=json.dumps(cfg),
        phases=json.dumps({k: {kk: vv for kk, vv in v.items()} for k, v in PHASES.items()}),
    )
    print(f"{kind}: {len(sample['atoms'])} atoms, {len(sample['grains'])} grains, "
          f"{len(refl)} Friedel pairs, multislice {t1 - t0:.1f} s -> {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "few")
