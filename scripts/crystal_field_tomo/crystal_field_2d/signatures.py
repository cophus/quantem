"""Grain masks from flash signatures.

Flicker f_t(x) = I_t(x) - median over neighbouring tilts. Each voxel r traces
x_t(r) = cos(t) x + sin(t) z through the sinogram; its signature is the dark part
of the flicker along that path. Voxels of one grain share that grain's dips at
all of its flash tilts; voxels outside it see only the tilts at which their ray
crosses it. K-means on normalised signatures (cosine distance) gives masks.
"""
import json, sys
import numpy as np
from scipy.ndimage import median_filter, map_coordinates

def signatures(kind):
    d = np.load(f"runs/sim_{kind}.npz")
    cfg = json.loads(str(d["cfg"])); ang = np.radians(d["angles"]); I = d["I_coarse"]
    f = I - median_filter(I, size=(7, 1), mode="wrap")
    dark = np.minimum(f, 0)
    dark = dark / (np.sqrt((dark**2).sum(1, keepdims=True)) + 1e-3)  # equal energy per tilt
    nv, dv = cfg["n_vox"], cfg["dx_coarse"]
    t = (np.arange(nv) + 0.5) * dv - nv * dv / 2
    X, Z = np.meshgrid(t, t, indexing="ij")
    xp = np.cos(ang)[:, None, None] * X + np.sin(ang)[:, None, None] * Z  # (T, nv, nv)
    pix = xp / cfg["dx_coarse"] + I.shape[1] // 2
    rows = np.broadcast_to(np.arange(len(ang))[:, None, None], pix.shape)
    S = map_coordinates(dark, [rows.ravel(), pix.ravel()], order=1).reshape(len(ang), nv, nv)
    return S, d, cfg

def kmeans_cos(Xn, K, w, iters=50, seed=0):
    rng = np.random.default_rng(seed)
    # k-means++ seeding on weighted points
    c = [Xn[rng.choice(len(Xn), p=w / w.sum())]]
    for _ in range(K - 1):
        dist = 1 - np.max(np.stack([Xn @ ci for ci in c]), 0)
        p = w * dist.clip(0) ** 2
        c.append(Xn[rng.choice(len(Xn), p=p / p.sum())])
    C = np.stack(c)
    for _ in range(iters):
        lab = np.argmax(Xn @ C.T, 1)
        for k in range(K):
            sel = lab == k
            if sel.any():
                v = (Xn[sel] * w[sel, None]).sum(0)
                C[k] = v / (np.linalg.norm(v) + 1e-12)
    obj = float((w * np.max(Xn @ C.T, 1)).sum())
    return lab, C, obj

if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "few"; K = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    S, d, cfg = signatures(kind)
    T, nv, _ = S.shape
    Xs = -S.reshape(T, -1).T  # (voxels, tilts), dips positive
    energy = np.linalg.norm(Xs, axis=1)
    Xn = Xs / (energy[:, None] + 1e-12)
    vb = np.load(f"runs/fp_{kind}_s10.npz")["vbar"]  # support from the reconstructed mean potential
    from scipy.ndimage import gaussian_filter
    sup = np.clip(gaussian_filter(vb, 2.0) / np.percentile(gaussian_filter(vb, 2.0), 95), 0, 1).ravel()
    w = energy**2 * sup**4
    best = max((kmeans_cos(Xn, K, w, seed=s) for s in range(10)), key=lambda r: r[2])
    lab = best[0].reshape(nv, nv)
    occ = d["occ"]; gt = np.where(occ.max(0) > 0.5, occ.argmax(0), -1)
    inside = gt >= 0
    # confusion within the particle
    conf = np.zeros((occ.shape[0], K), int)
    for g in range(occ.shape[0]):
        for k in range(K):
            conf[g, k] = ((gt == g) & (lab == k)).sum()
    print("confusion (rows GT grain, cols cluster), particle voxels only:\n", conf)
    np.savez(f"runs/sig_{kind}.npz", lab=lab, energy=energy.reshape(nv, nv), C=best[1])
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(energy.reshape(nv, nv).T, origin="lower"); ax[0].set_title("signature energy")
    ax[1].imshow(np.where(energy.reshape(nv, nv) > 0, lab, -1).T, origin="lower", cmap="tab10"); ax[1].set_title("k-means clusters")
    ax[2].imshow(gt.T, origin="lower", cmap="tab10"); ax[2].set_title("GT grains")
    plt.tight_layout(); plt.savefig(f"runs/sig_{kind}.png", dpi=80)
