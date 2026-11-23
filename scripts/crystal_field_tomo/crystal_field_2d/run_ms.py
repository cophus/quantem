"""Producer for invert_ms.py (per-voxel sub-block multislice).

python run_ms.py <kind> <case>
  positive   real potential >= 0, small random positive start
  complex    free complex potential, start 1 + small random complex transmission
  coarse     mean potential only (band-limited to the 2 A pixel), same multislice
  split      density (2 A band, >= 0) + free high-band flicker term, real
  split_complex   same with a complex flicker term
  floor08, floor05   positive with U >= -beta x local density, beta 0.8 / 0.5
"""

import sys

import invert_ms

CASES = {
    "positive": dict(mode="positive"),
    "complex": dict(mode="complex"),
    "coarse": dict(mode="coarse", lr=1e-3, iters=1500),
    "split": dict(mode="split"),
    "split_complex": dict(mode="split_complex"),
    "floor08": dict(mode="positive", floor=0.8),
    "floor05": dict(mode="positive", floor=0.5),
}
kind, case = sys.argv[1], sys.argv[2]
invert_ms.run(kind=kind, tag=f"_{case}", **CASES[case])
