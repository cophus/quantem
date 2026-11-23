"""Produces runs/cells_*.npz: lattice-free cell field with group TV (invert_cells.py).

python run_cells.py fewAm g03       amorphous-cap sample, default weights (lam_1 5e-4)
python run_cells.py fewA g03        twin sample without the cap, default weights
python run_cells.py fewAm l1_5e-3   stronger L1 sparsity
python run_cells.py fewAm l1_2e-2   stronger still
"""

import sys

import invert_cells

kind, case = sys.argv[1], sys.argv[2]
weights = {"g03": {}, "l1_5e-3": dict(lam_1=5e-3), "l1_2e-2": dict(lam_1=2e-2)}[case]
invert_cells.run(kind=kind, tag=f"_{case}", **weights)
