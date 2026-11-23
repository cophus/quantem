"""Reproduce the simulated HRTEM tilt series of the 3D crystal-field toy.

FCC particle (a = 4.08 A, mirror twin across a (111) plane, optional amorphous cap at the same
number density), 300 kV, no aperture, no Cs, tilted about y from -90 to +89 degrees in 1 degree
steps. Fine multislice (0.25 A pixels, 1 A slices), then intensities Fourier-cropped to 2 A pixels,
so no lattice plane is resolved and the Bragg scattering shows up as flicker.

    python reproduce_tilt_series.py              # 6.4 nm particle, 192 A field of view (~3 min)
    python reproduce_tilt_series.py --scale 1    # 3.2 nm particle, 96 A field of view (~15 s)
    python reproduce_tilt_series.py --no-cap     # twin pair only

Geometry: beam along lab +z, tilt axis y through the origin; at tilt t a sample point (x, y, z)
goes to x_lab = cos(t) x + sin(t) z, z_lab = -sin(t) x + cos(t) z, y_lab = y. The defocus is
measured from the slab exit plane z_lab = R + 4 A, transfer exp(-i pi lambda defocus q^2).

Writes tilt_series_<kind>_x<scale>.npz with
  images       (n_tilt, nx, ny) intensities indexed [x, y], vacuum = 1, 2 A pixels,
               pixel i at (i - n / 2) * 2 A
  angles_deg   (n_tilt,)
  pixel_A, fov_A, energy_eV, defocus_A, cs_A
  vbar         true mean potential (rad / A) on 2 A voxels [x, y, z], centres at
               (i + 0.5) * 2 A - L / 2; for scoring a reconstruction
  cryst        true crystalline fraction of each voxel
  atoms_A      atom positions (x, y, z) in the sample frame
"""

import argparse
import json

import numpy as np
import torch

import sim3d


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--scale", type=int, default=2, help="particle and field-of-view size factor"
    )
    p.add_argument("--no-cap", action="store_true", help="leave out the amorphous cap")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()

    cfg = dict(sim3d.CONFIG)
    for key in ["fov", "R", "n_vox", "nb"]:
        cfg[key] *= args.scale
    kind = "twin" if args.no_cap else "twinAm"
    torch.set_num_threads(args.threads)

    sample = sim3d.build_sample(kind, np.random.default_rng(cfg["seed"]), cfg["R"])
    angles = np.arange(-90.0, 90.0, cfg["tilt_step"])
    fine = sim3d.multislice_tilts(sample, angles, cfg)
    images = sim3d.crop2(
        torch.as_tensor(fine), round(cfg["fov"] / cfg["dx_coarse"])
    ).numpy()
    vbar, occ = sim3d.coarse_truth(sample, cfg)
    crystalline = [
        k for k, g in enumerate(sample["grains"]) if g["phase"] != "amorphous"
    ]

    out = f"tilt_series_{kind}_x{args.scale}.npz"
    np.savez_compressed(
        out,
        images=images.astype(np.float32),
        angles_deg=angles,
        pixel_A=cfg["dx_coarse"],
        fov_A=cfg["fov"],
        energy_eV=cfg["energy_ev"],
        defocus_A=cfg["defocus"],
        cs_A=cfg["cs"],
        vbar=vbar.astype(np.float32),
        cryst=occ[crystalline].sum(0).astype(np.float32),
        atoms_A=sample["atoms"].astype(np.float32),
        config=json.dumps(cfg),
    )
    print(
        f"{len(sample['atoms'])} atoms, {images.shape[0]} tilts of {images.shape[1]} x {images.shape[2]} px -> {out}"
    )


if __name__ == "__main__":
    main()
