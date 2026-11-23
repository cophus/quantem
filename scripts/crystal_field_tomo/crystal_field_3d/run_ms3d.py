"""Producer for invert_ms3d.py.

python run_ms3d.py <kind> <case> [iters]
  coarse     mean potential only (band-limited to the 2 A voxel), same multislice
  positive   sub-blocks, U >= 0
  floor08    sub-blocks, U >= -0.8 x local density
  floor08_long   the same, 4000 iterations
  floor08_sub5, floor08_sub3   the same with 5^3 / 3^3 sub-blocks per voxel
  floor08_band   floor08 fitted to band-limited images (sim3d.band_limit, taper from 0.5 Nyquist)
"""

import sys

import invert_ms3d

CASES = {
    "coarse": {"mode": "coarse", "lr": 1e-3, "iters": 1500},
    "positive": {"mode": "positive"},
    "floor08": {"mode": "positive", "floor": 0.8},
    "floor08_long": {"mode": "positive", "floor": 0.8, "iters": 4000},
    "floor08_sub5": {"mode": "positive", "floor": 0.8, "sub": 5},
    "floor08_sub3": {"mode": "positive", "floor": 0.8, "sub": 3},
    "floor08_band": {"mode": "positive", "floor": 0.8, "band": 0.5},
}
kind, case = sys.argv[1], sys.argv[2]
iters = int(sys.argv[3]) if len(sys.argv) > 3 else CASES[case].pop("iters", 2000)
CASES[case].pop("iters", None)
invert_ms3d.run(kind=kind, tag=f"_{case}", iters=iters, **CASES[case])
