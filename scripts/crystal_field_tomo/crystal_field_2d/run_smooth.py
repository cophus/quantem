"""Produces the runs/sm_*.npz files for crystal_field_2d_03.ipynb.

python run_smooth.py 1   learned lattice on fewA, mean-only baselines
python run_smooth.py 2   learned lattice on fewAm, true-lattice diagnostic on fewA
python run_smooth.py 3, 4   sharpened orientation on fewAm, with and without the mass gate
python run_smooth.py 5, 6   sharpened + diffuse channel + mass gate on fewAm, learned and true lattice
python run_smooth.py 7, 8   same with an L1 prior on crystallinity, true lattice
The two groups run side by side. Mean-only baselines are the same fit with the
crystal term never switched on.
"""

import sys

import numpy as np

import invert_smooth

TRUE_A = (2.70, float(np.hypot(0.95, 2.45)), float(np.degrees(np.arctan2(2.45, 0.95))))

if sys.argv[1] == "1":
    invert_smooth.run(kind="fewA", tag="_learned")
    invert_smooth.run(kind="fewA", n_phi=1, crystal_on=False, tag="_meanonly", log_every=500)
    invert_smooth.run(kind="fewAm", n_phi=1, crystal_on=False, tag="_meanonly", log_every=500)
elif sys.argv[1] == "2":
    invert_smooth.run(kind="fewAm", tag="_learned")
    # diagnostic only: the true lattice of phase A held fixed (not part of the reconstruction route)
    invert_smooth.run(kind="fewA", lattice0=TRUE_A, lattice_lr=0.0, tag="_truelat")

# one orientation per voxel (sharpened) and crystal gated by mass: c is the crystalline fraction
if sys.argv[1] == "3":
    invert_smooth.run(kind="fewAm", sharpen=0.05, ent_from=0.4, mass_gate=True, tag="_sharpgate")
if sys.argv[1] == "4":
    invert_smooth.run(kind="fewAm", sharpen=0.05, ent_from=0.4, tag="_sharp")

# sharpened orientation + amorphous diffuse channel + mass gate on fewAm
FULL = dict(sharpen=0.2, ent_from=0.4, mass_gate=True, diffuse=True)
if sys.argv[1] == "5":
    invert_smooth.run(kind="fewAm", tag="_full", **FULL)
if sys.argv[1] == "6":
    # diagnostic only: the true lattice of phase A held fixed
    invert_smooth.run(kind="fewAm", lattice0=TRUE_A, lattice_lr=0.0, tag="_full_truelat", **FULL)

# diagnostic only, true lattice: L1 prior on crystallinity at two strengths
if sys.argv[1] == "7":
    invert_smooth.run(kind="fewAm", lattice0=TRUE_A, lattice_lr=0.0, lam_c=0.1, tag="_full_truelat_c01", **FULL)
if sys.argv[1] == "8":
    invert_smooth.run(kind="fewAm", lattice0=TRUE_A, lattice_lr=0.0, lam_c=0.3, tag="_full_truelat_c03", **FULL)
