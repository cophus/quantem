"""Small 3D ground truth for crystal-field tomography.

A 3.2 nm FCC particle (mirror twin across a (111) plane, optional amorphous cap at the same
number density) is tilted about y in 1 degree steps; fine multislice (0.25 A, 1 A slices),
defocus, no aperture; intensities Fourier-cropped to 2 A pixels.

Geometry as in the 2D toy: sample frame (x, y, z), tilt axis y, beam along lab +z, at tilt
theta x_lab = cos x + sin z, z_lab = -sin x + cos z, y_lab = y. Volumes are indexed
[x, y, z], images [x, y].

Run:  python sim3d.py twinAm    (twin pair + amorphous cap)
      python sim3d.py twin      (no cap)
      python sim3d.py twinAm 2  (particle, field of view and grids 2x larger -> sim_twinAm_x2.npz)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import erf

HERE = Path(__file__).resolve().parent


def wavelength(energy_ev):
    """Relativistic electron wavelength in Angstrom."""
    return 12398.419843320026 / np.sqrt(energy_ev * (2 * 510998.95 + energy_ev))


def chi(q, lam, defocus, cs):
    """Aberration phase chi(q) = pi lam df q^2 + pi/2 Cs lam^3 q^4."""
    return np.pi * lam * defocus * q**2 + 0.5 * np.pi * cs * lam**3 * q**4


FCC = {
    "a": 4.08,
    "sigma": 0.3,
    "xi": 180.0,
}  # xi: extinction distance of 111, sets the atom strength

CONFIG = {
    "energy_ev": 300e3,
    "defocus": 1000.0,  # A, at the exit of the slab
    "cs": 0.0,
    "fov": 96.0,
    "dx_fine": 0.25,
    "dz_fine": 1.0,
    "dx_coarse": 2.0,
    "n_vox": 24,  # coarse truth grid (2 A voxels, 48 A box)
    "nb": 20,  # reconstruction block (voxels per side, centred in the n_vox grid)
    "R": 16.0,
    "tilt_step": 1.0,
    "seed": 0,
}


def atom_strength():
    """P (rad A^2) such that U_111 = 4 P / a^3 exp(-2 pi^2 sigma^2 g^2) = pi / xi."""
    a, s = FCC["a"], FCC["sigma"]
    g = np.sqrt(3) / a
    return np.pi / FCC["xi"] * a**3 / 4 / np.exp(-2 * np.pi**2 * s**2 * g**2)


def random_rotation(rng):
    w, x, y, z = (q := rng.normal(size=4)) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def fcc_points(Rot, R):
    a = FCC["a"]
    n = int(np.ceil(R / a)) + 2
    ijk = np.stack(np.meshgrid(*[np.arange(-n, n + 1)] * 3, indexing="ij"), -1).reshape(
        -1, 3
    )
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    r = ((ijk[:, None, :] + basis[None]).reshape(-1, 3) * a) @ Rot.T
    return r[np.linalg.norm(r, axis=1) <= R + 1.0]


def amorphous_atoms(R, rng, d_min=2.5, n_relax=400):
    """Random packing over the ball of radius R at the FCC number density: uniform points
    pushed apart until no pair is closer than d_min."""
    from scipy.spatial import cKDTree

    n = round(4 / 3 * np.pi * R**3 * 4 / FCC["a"] ** 3)
    x = rng.normal(size=(n, 3))
    x *= (
        R
        * rng.uniform(size=(n, 1)) ** (1 / 3)
        / np.linalg.norm(x, axis=1, keepdims=True)
    )
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


def build_sample(kind, rng, R):
    amorph = kind.endswith("m")
    Rot = random_rotation(rng)
    n_tw = Rot @ np.ones(3) / np.sqrt(3)  # (111) twin plane through the origin
    M = np.eye(3) - 2 * np.outer(n_tw, n_tw)
    cap = (c := rng.normal(size=3)) / np.linalg.norm(c)
    grains = [
        {"phase": "fcc", "rot": Rot.tolist()},
        {"phase": "fcc", "rot": (M @ Rot).tolist(), "mirror": n_tw.tolist()},
    ]
    if amorph:
        grains.append({"phase": "amorphous", "cap": cap.tolist()})

    def label(r):
        lab = np.where(r @ n_tw > -1e-6, 0, 1)
        if amorph:
            lab = np.where(r @ cap > 0.4 * R, 2, lab)
        return np.where(np.linalg.norm(r, axis=-1) <= R, lab, -1)

    lat = fcc_points(Rot, R)
    parts = [lat, lat @ M.T]
    if amorph:
        parts.append(
            amorphous_atoms(R + 4.0, rng)
        )  # crowding at the packing boundary stays outside
    atoms, gid = [], []
    for k, r in enumerate(parts):
        keep = label(r) == k
        atoms.append(r[keep])
        gid.append(np.full(keep.sum(), k))
    atoms, gid = np.concatenate(atoms), np.concatenate(gid)
    atoms = atoms + rng.normal(scale=0.03, size=atoms.shape)
    P = np.full(len(atoms), atom_strength())
    return {"atoms": atoms, "P": P, "gid": gid, "grains": grains, "label": label}


def crop2(I, n_out):
    """Fourier-crop the last two axes to n_out on the same field of view (Nyquist dropped,
    mean kept). Sample 0 stays on sample 0."""
    n_in = I.shape[-1]
    h = n_out // 2
    i_in = torch.cat([torch.arange(h), torch.arange(n_in - h + 1, n_in)])
    i_out = torch.cat([torch.arange(h), torch.arange(n_out - h + 1, n_out)])
    Fi = torch.fft.fft2(I)[..., i_in, :][..., i_in]
    out = torch.zeros(I.shape[:-2] + (n_out, n_out), dtype=Fi.dtype, device=Fi.device)
    out[..., i_out[:, None], i_out[None, :]] = Fi
    return torch.fft.ifft2(out).real * (n_out / n_in) ** 2


def band_limit(I, q_lo=0.5):
    """Smooth round band limit for images on the coarse pixel grid (last two axes): pass below
    q_lo x Nyquist, cos^2 taper to zero at Nyquist, nothing in the corners of the square band.
    Removes the pixel checkerboard; apply the same call to data and model."""
    I = torch.as_tensor(I)
    n = I.shape[-1]
    q = torch.fft.fftfreq(n, 1.0) / 0.5  # in units of Nyquist
    Q = torch.sqrt(q[:, None] ** 2 + q[None, :] ** 2)
    W = torch.cos(0.5 * np.pi * ((Q - q_lo) / (1 - q_lo)).clamp(0, 1)) ** 2
    return torch.fft.ifft2(torch.fft.fft2(I) * W).real


def gauss_px(x, n, dx, sig, win):
    """Pixel-integrated 1D Gaussian weights on pixels (i - n // 2) dx, window around each atom."""
    i = np.round(x / dx).astype(np.int64)[..., None] + win
    lo = (i - 0.5) * dx - x[..., None]
    w = 0.5 * (erf((lo + dx) / (np.sqrt(2) * sig)) - erf(lo / (np.sqrt(2) * sig)))
    return i + n // 2, w


def multislice_tilts(sample, angles_deg, cfg, chunk=6, q_lp=None):
    """Fine multislice for every tilt. Returns fine intensities (n_tilt, n, n) indexed [x, y].
    q_lp (diagnostic oracle): low-pass every slice to |q| < q_lp, so nothing Bragg scatters."""
    lam = wavelength(cfg["energy_ev"])
    dx, dz = cfg["dx_fine"], cfg["dz_fine"]
    n = round(cfg["fov"] / dx)
    q = np.fft.fftfreq(n, dx)
    aa = np.abs(q) < (2 / 3) * np.abs(q).max()
    Q2 = q[:, None] ** 2 + q[None, :] ** 2
    prop = torch.as_tensor(
        np.exp(-1j * np.pi * lam * dz * Q2) * (aa[:, None] & aa[None, :])
    ).to(torch.complex64)
    ctf = torch.as_tensor(
        np.exp(-1j * chi(np.sqrt(Q2), lam, cfg["defocus"], cfg["cs"]))
    ).to(torch.complex64)
    half = cfg["R"] + 4.0
    n_sl = int(np.ceil(2 * half / dz))
    sig = FCC["sigma"]
    win = np.arange(-4, 5)
    atoms, P = sample["atoms"], sample["P"]
    out = np.zeros((len(angles_deg), n, n), dtype=np.float32)
    for c0 in range(0, len(angles_deg), chunk):
        th = np.radians(angles_deg[c0 : c0 + chunk])
        T = len(th)
        xl = np.cos(th)[:, None] * atoms[:, 0] + np.sin(th)[:, None] * atoms[:, 2]
        zl = -np.sin(th)[:, None] * atoms[:, 0] + np.cos(th)[:, None] * atoms[:, 2]
        yl = np.broadcast_to(atoms[:, 1], xl.shape)
        sl = np.floor((zl + half) / dz).astype(np.int64)
        ix, wx = gauss_px(xl, n, dx, sig, win)
        iy, wy = gauss_px(yl, n, dx, sig, win)
        val = P[None, :, None, None] * wx[..., :, None] * wy[..., None, :] / dx**2
        flat = (
            (np.arange(T)[:, None, None, None] * n_sl + sl[..., None, None]) * n
            + ix[..., :, None]
        ) * n + iy[..., None, :]
        pot = torch.zeros(T * n_sl * n * n, dtype=torch.float32)
        pot.index_add_(
            0,
            torch.as_tensor(flat.ravel()),
            torch.as_tensor(val.ravel(), dtype=torch.float32),
        )
        pot = pot.view(T, n_sl, n, n)
        if q_lp is not None:
            pot = torch.fft.ifft2(
                torch.fft.fft2(pot) * torch.as_tensor(np.sqrt(Q2) < q_lp)
            ).real
        psi = torch.ones(T, n, n, dtype=torch.complex64)
        for s in range(n_sl):
            psi = torch.fft.ifft2(
                torch.fft.fft2(psi * torch.exp(1j * pot[:, s])) * prop
            )
        img = torch.fft.ifft2(torch.fft.fft2(psi) * ctf)
        out[c0 : c0 + T] = (img.abs() ** 2).numpy()
    return out


def coarse_truth(sample, cfg):
    """Mean potential (rad / A) low-passed to the 2 A voxel grid [x, y, z], voxel centres at
    (i + 0.5) dv - L / 2, and grain occupancy fractions."""
    nv, dv = cfg["n_vox"], cfg["dx_coarse"]
    L = nv * dv
    up = 4
    nf, df = nv * up, dv / up
    ijk = (sample["atoms"] - (0.5 * dv - L / 2)) / df
    i0 = np.floor(ijk).astype(int)
    w = ijk - i0
    dens = np.zeros((nf, nf, nf))
    for d in np.ndindex(2, 2, 2):
        wt = np.prod([w[:, a] if d[a] else 1 - w[:, a] for a in range(3)], axis=0)
        np.add.at(
            dens, tuple((i0[:, a] + d[a]) % nf for a in range(3)), sample["P"] * wt
        )
    dens /= df**3
    F = np.fft.fftshift(np.fft.fftn(dens))
    qf = np.fft.fftshift(np.fft.fftfreq(nf, df))
    Q2 = qf[:, None, None] ** 2 + qf[None, :, None] ** 2 + qf[None, None, :] ** 2
    F *= np.exp(-2 * np.pi**2 * FCC["sigma"] ** 2 * Q2)
    tent = np.sinc(qf * df) ** 2  # undo the trilinear splat
    F /= tent[:, None, None] * tent[None, :, None] * tent[None, None, :]
    c, h = nf // 2, nv // 2
    Fc = F[c - h : c + h, c - h : c + h, c - h : c + h].copy()
    Fc[0], Fc[:, 0], Fc[:, :, 0] = 0, 0, 0  # Nyquist planes
    vbar = np.fft.ifftn(np.fft.ifftshift(Fc)).real * nv**3 / nf**3

    ss = 3
    t = (np.arange(nv * ss) + 0.5) / ss * dv - L / 2
    X = np.stack(np.meshgrid(t, t, t, indexing="ij"), -1)
    lab = sample["label"](X)
    occ = np.stack(
        [
            (lab == k).reshape(nv, ss, nv, ss, nv, ss).mean((1, 3, 5))
            for k in range(len(sample["grains"]))
        ]
    )
    return vbar, occ


def main(kind, scale=1):
    cfg = dict(CONFIG)
    for key in ["fov", "R", "n_vox", "nb"]:
        cfg[key] = cfg[key] * scale
    rng = np.random.default_rng(cfg["seed"])
    sample = build_sample(kind, rng, cfg["R"])
    angles = np.arange(-90.0, 90.0, cfg["tilt_step"])
    torch.set_num_threads(3)
    t0 = time.time()
    I_fine = multislice_tilts(sample, angles, cfg)
    t1 = time.time()
    n_out = round(cfg["fov"] / cfg["dx_coarse"])
    I_coarse = crop2(torch.as_tensor(I_fine), n_out).numpy()
    # diagnostic: the same sample with every slice low-passed below 0.25 / A, so no Bragg flicker
    I_nobragg = crop2(
        torch.as_tensor(multislice_tilts(sample, angles, cfg, q_lp=0.25)), n_out
    ).numpy()
    vbar, occ = coarse_truth(sample, cfg)
    cryst = occ[
        [k for k, g in enumerate(sample["grains"]) if g["phase"] != "amorphous"]
    ].sum(0)
    (HERE / "runs").mkdir(exist_ok=True)
    out = (
        HERE
        / "runs"
        / (f"sim_{kind}.npz" if scale == 1 else f"sim_{kind}_x{scale}.npz")
    )
    np.savez_compressed(
        out,
        angles=angles,
        I_coarse=I_coarse.astype(np.float32),
        I_nobragg=I_nobragg.astype(np.float32),
        vbar=vbar.astype(np.float32),
        occ=occ.astype(np.float32),
        cryst=cryst.astype(np.float32),
        atoms=sample["atoms"].astype(np.float32),
        P=sample["P"].astype(np.float32),
        gid=sample["gid"],
        grains=json.dumps(sample["grains"]),
        cfg=json.dumps(cfg),
        fcc=json.dumps(FCC),
    )
    print(f"{kind}: {len(sample['atoms'])} atoms, multislice {t1 - t0:.1f} s -> {out}")


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else "twinAm",
        int(sys.argv[2]) if len(sys.argv) > 2 else 1,
    )
