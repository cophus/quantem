"""Produces runs/lat_fewA_main.npz and runs/lat_fewA_meanonly.npz for crystal_field_2d_02.ipynb.

The mean-only baseline is the same fit with the crystal term never switched on.
"""

import torch

import invert_lattice

torch.set_num_threads(3)
invert_lattice.run(n_phi=90, iters=1500, warm=300, batch=8, u_start=0.42, tag="_main", log_every=100)
invert_lattice.run(n_phi=1, iters=1500, warm=1500, batch=8, u_start=0.42, tag="_meanonly", log_every=500)

# diagnostic only: the true lattice of phase A held fixed (not part of the reconstruction route)
import numpy as np

invert_lattice.run(n_phi=90, iters=1500, warm=300, batch=8, lattice_lr=0.0, tag="_truelat", log_every=100,
                   lattice0=(2.70, float(np.hypot(0.95, 2.45)), float(np.degrees(np.arctan2(2.45, 0.95)))))
