"""Diagnostic only: mean-potential fits on the no-Bragg oracle data (oracle.py) and on the
real data at two TV weights. Produces runs/sm_{kind}_meanonly_tv*.npz.

python run_oracle.py 1   fewAm: oracle and real data, TV 0.01
python run_oracle.py 2   fewAm oracle TV 0.05, fewA oracle TV 0.05 and 0.01, fewA real TV 0.01
python run_oracle.py 3, 4   robust (Cauchy) fits on real fewAm, fewA
"""

import sys

import invert_smooth

mean_only = dict(n_phi=1, crystal_on=False, log_every=500)
if sys.argv[1] == "1":
    invert_smooth.run(kind="fewAm_nobragg", lam_tv=0.01, tag="_meanonly_tv0.01", **mean_only)
    invert_smooth.run(kind="fewAm", lam_tv=0.01, tag="_meanonly_tv0.01", **mean_only)
elif sys.argv[1] == "2":
    invert_smooth.run(kind="fewAm_nobragg", lam_tv=0.05, tag="_meanonly_tv0.05", **mean_only)
    invert_smooth.run(kind="fewA_nobragg", lam_tv=0.05, tag="_meanonly_tv0.05", **mean_only)
    invert_smooth.run(kind="fewA_nobragg", lam_tv=0.01, tag="_meanonly_tv0.01", **mean_only)
    invert_smooth.run(kind="fewA", lam_tv=0.01, tag="_meanonly_tv0.01", **mean_only)

# robust (Cauchy) mean-potential fits on the real data, scale = k x 1.4826 MAD of the residual
if sys.argv[1] == "3":
    invert_smooth.run(kind="fewAm", robust=-1.0, tag="_meanonly_cauchy1", **mean_only)
    invert_smooth.run(kind="fewAm", robust=-0.5, tag="_meanonly_cauchy05", **mean_only)
if sys.argv[1] == "4":
    invert_smooth.run(kind="fewA", robust=-1.0, tag="_meanonly_cauchy1", **mean_only)
    invert_smooth.run(kind="fewA", robust=-0.5, tag="_meanonly_cauchy05", **mean_only)
