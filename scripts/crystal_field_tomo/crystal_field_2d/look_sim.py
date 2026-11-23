"""Quick look at a simulated tilt series: sinogram, tilt-to-tilt flicker, GT cells."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

kind = sys.argv[1] if len(sys.argv) > 1 else "few"
d = np.load(f"runs/sim_{kind}.npz")
I = d["I_coarse"]; ang = d["angles"]; cfg = json.loads(str(d["cfg"]))
refl = json.loads(str(d["refl"]))
xc = (np.arange(I.shape[1]) - I.shape[1] // 2) * cfg["dx_coarse"]
flick = I - median_filter(I, size=(7, 1), mode="wrap")
print("intensity mean %.4f  min %.3f  max %.3f" % (I.mean(), I.min(), I.max()))
print("flicker rms %.4f, max |flicker| %.3f" % (flick.std(), np.abs(flick).max()))
# strongest reflections per grain
for k in range(len(json.loads(str(d["grains"])))):
    r = sorted([e for e in refl if e["grain"] == k], key=lambda e: -np.hypot(*e["rho"]))[:6]
    print(f"grain {k}: " + ", ".join(f"(th {e['theta_c']:.1f}, u {e['u']:.3f}, xi {np.pi/np.hypot(*e['rho']):.0f})" for e in r))
fig, ax = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
ext = [xc[0], xc[-1], ang[-1], ang[0]]
ax[0].imshow(I, aspect="auto", extent=ext, cmap="gray"); ax[0].set_title("coarse intensity")
v = np.percentile(np.abs(flick), 99.5)
ax[1].imshow(flick, aspect="auto", extent=ext, cmap="RdBu_r", vmin=-v, vmax=v); ax[1].set_title("flicker (minus 7-tilt median)")
ax[2].imshow(flick, aspect="auto", extent=ext, cmap="RdBu_r", vmin=-v, vmax=v)
cols = ["k", "m", "g", "c", "y", "orange"]
for e in refl:
    if np.hypot(*e["rho"]) > 0.3 * np.pi / 180:
        ax[2].axhline(e["theta_c"], color=cols[e["grain"] % 6], lw=0.6, alpha=0.7)
ax[2].set_title("GT cells (strong), colour = grain")
for a in ax: a.set_xlabel("x (A)"); a.set_xlim(-150, 150)
ax[0].set_ylabel("tilt (deg)")
plt.tight_layout(); plt.savefig(f"runs/look_{kind}.png", dpi=110)
# a few profiles at a strong flash
e = max(refl, key=lambda e: np.hypot(*e["rho"]))
it = int(np.argmin(np.abs(ang - e["theta_c"])))
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
for j in (it - 2, it - 1, it, it + 1, it + 2):
    ax.plot(xc, I[j % len(ang)], label=f"{ang[j % len(ang)]:.0f} deg")
ax.set_xlim(-150, 150); ax.legend(); ax.set_title(f"profiles around strongest cell th={e['theta_c']:.1f} u={e['u']:.3f}")
plt.tight_layout(); plt.savefig(f"runs/look_{kind}_profiles.png", dpi=110)
