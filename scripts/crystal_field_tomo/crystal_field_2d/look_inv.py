import sys, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
tag = sys.argv[1]; kind = sys.argv[2] if len(sys.argv) > 2 else "few"
r = np.load(f"runs/inv_{kind}{tag}.npz"); d = np.load(f"runs/sim_{kind}.npz")
W = r["W"]; K = W.shape[0]
fig, ax = plt.subplots(2, max(K, 3) + 1, figsize=(3.2 * (max(K, 3) + 1), 6.4))
ax[0, 0].imshow(r["vbar"].T, origin="lower"); ax[0, 0].set_title("vbar recon")
ax[1, 0].imshow(d["vbar"].T, origin="lower"); ax[1, 0].set_title("vbar GT")
for k in range(K):
    ax[0, k + 1].imshow(W[k].T, origin="lower", vmin=0, vmax=1); ax[0, k + 1].set_title(f"W set {k}, sum a={r['a'][k].sum()*1e4:.1f}")
for k in range(d["occ"].shape[0]):
    ax[1, k + 1].imshow(d["occ"][k].T, origin="lower", vmin=0, vmax=1); ax[1, k + 1].set_title(f"GT grain {k}")
for a in ax.ravel(): a.axis("off")
plt.tight_layout(); plt.savefig(f"runs/look_inv_{kind}{tag}.png", dpi=80)
