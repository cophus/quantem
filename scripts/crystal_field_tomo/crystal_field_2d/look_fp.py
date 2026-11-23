import sys, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
kind = sys.argv[1] if len(sys.argv) > 1 else "few"; tag = sys.argv[2] if len(sys.argv) > 2 else ""
r = np.load(f"runs/fp_{kind}{tag}.npz"); d = np.load(f"runs/sim_{kind}.npz")
refl = json.loads(str(d["refl"])); ang = d["angles"]
F, u = r["F"], r["u"]
xs = np.arange(F.shape[-1]) - F.shape[-1] // 2
fig, ax = plt.subplots(1, 4, figsize=(20, 6))
ax[0].imshow(r["vbar"].T, origin="lower", vmin=0, vmax=np.percentile(d["vbar"], 99.5)); ax[0].set_title("vbar recon")
ax[1].imshow(d["vbar"].T, origin="lower", vmin=0, vmax=np.percentile(d["vbar"], 99.5)); print("vbar GT p99", np.percentile(d["vbar"],99), "recon p99", np.percentile(r["vbar"],99)); ax[1].set_title("vbar GT")
ax[2].imshow(F.sum(1), aspect="auto", extent=[xs[0], xs[-1], ang[-1], ang[0]], cmap="magma")
ax[2].set_title("sum of footprints"); ax[2].set_xlim(-110, 110)
mass = F.sum(-1)
for j in range(F.shape[1]):
    ax[3].scatter(u[:, j], ang, s=60 * mass[:, j] / mass.max(), c="k")
for e in refl:
    if np.hypot(*e["rho"]) > 0.3 * np.pi / 250:
        ax[3].scatter([e["u"], -e["u"]], [e["theta_c"]] * 2, marker="x", c="r", s=20)
ax[3].invert_yaxis(); ax[3].set_xlabel("u (signed)"); ax[3].set_title("slots (size = mass) vs GT cells (red x)")
plt.tight_layout(); plt.savefig(f"runs/look_fp_{kind}{tag}.png", dpi=80)
