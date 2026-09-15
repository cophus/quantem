"""Per-site measurements derived from template matching and neighbor lists.

Functions here operate on plain arrays so they can be tested independently of
:class:`~quantem.atoms.AtomicModel`, which wraps them.  Everything is
vectorized with NumPy / SciPy; nothing loops over atoms in Python.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull

__all__ = [
    "misorientation",
    "segment_grains",
    "fill_labels",
    "strain_from_deformation",
    "bond_angles",
    "convex_hull_distance",
    "sample_volume",
    "kmeans_1d",
    "rotation_to_quaternion",
]


def misorientation(
    rotation: NDArray,
    neighbor_index: NDArray,
    symmetry: NDArray | None = None,
    chunk_size: int = 2048,
) -> NDArray:
    """Disorientation angle between every site and each of its neighbors.

    Parameters
    ----------
    rotation : ndarray
        ``(N, 3, 3)`` site orientations (lab <- crystal).
    neighbor_index : ndarray
        ``(N, K)`` neighbor indices; ``-1`` marks missing neighbors.
    symmetry : ndarray, optional
        ``(S, 3, 3)`` proper symmetry rotations of the crystal; the minimum
        angle over all symmetry-equivalent descriptions is returned.
    chunk_size : int
        Sites per batch.

    Returns
    -------
    ndarray
        ``(N, K)`` angles in degrees, ``nan`` for missing neighbors.
    """
    rotation = np.asarray(rotation, dtype=np.float64)
    n, k = neighbor_index.shape
    if symmetry is None:
        symmetry = np.eye(3)[None]
    out = np.full((n, k), np.nan)
    idx_safe = np.where(neighbor_index >= 0, neighbor_index, 0)
    for start in range(0, n, chunk_size):
        sl = slice(start, min(start + chunk_size, n))
        ra = rotation[sl]  # (n,3,3)
        rb = rotation[idx_safe[sl]]  # (n,k,3,3)
        delta = np.einsum("nji,nkjl->nkil", ra, rb)  # R_a^T R_b
        trace = np.einsum("nkij,sji->nks", delta, symmetry).max(-1)
        ang = np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))
        out[sl] = ang
    out[neighbor_index < 0] = np.nan
    return out


def segment_grains(
    neighbor_index: NDArray,
    edge_mask: NDArray,
    min_size: int = 1,
) -> NDArray:
    """Label connected components of a neighbor graph.

    Parameters
    ----------
    neighbor_index : ndarray
        ``(N, K)`` neighbor indices (``-1`` = missing).
    edge_mask : ndarray
        ``(N, K)`` boolean; ``True`` where site ``n`` and neighbor ``k`` belong
        to the same grain (e.g. same structure and small misorientation).
    min_size : int
        Components smaller than this are labelled ``-1``.

    Returns
    -------
    ndarray
        ``(N,)`` integer labels ordered by decreasing size (0 = largest),
        ``-1`` for sites in components smaller than ``min_size``.
    """
    n, k = neighbor_index.shape
    rows = np.repeat(np.arange(n), k)
    cols = neighbor_index.ravel()
    keep = edge_mask.ravel() & (cols >= 0)
    graph = coo_matrix((np.ones(keep.sum()), (rows[keep], cols[keep])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    order = np.argsort(-sizes, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    out = rank[labels]
    out[sizes[labels] < min_size] = -1
    return out


def fill_labels(
    labels: NDArray, neighbor_index: NDArray, edge_mask: NDArray, member: NDArray
) -> NDArray:
    """Assign unlabelled member sites to the majority label of their neighbors.

    Parameters
    ----------
    labels : ndarray
        ``(N,)`` labels, ``-1`` = unassigned.
    neighbor_index : ndarray
        ``(N, K)`` neighbor indices (``-1`` = missing).
    edge_mask : ndarray
        ``(N, K)`` votes are only counted along ``True`` edges.
    member : ndarray
        ``(N,)`` sites eligible for filling.

    Returns
    -------
    ndarray
        Updated copy of ``labels``.
    """
    labels = np.array(labels, copy=True)
    todo = np.where(member & (labels < 0))[0]
    if todo.size == 0:
        return labels
    safe = np.where(neighbor_index >= 0, neighbor_index, 0)
    votes = np.where(edge_mask & (neighbor_index >= 0), labels[safe], -1)[todo]
    n_max = int(labels.max()) + 2
    counts = np.zeros((todo.size, n_max), dtype=int)
    rows = np.repeat(np.arange(todo.size), votes.shape[1])
    valid = votes.ravel() >= 0
    np.add.at(counts, (rows[valid], votes.ravel()[valid]), 1)
    best = counts.argmax(axis=1)
    has_votes = counts.max(axis=1) > 0
    labels[todo[has_votes]] = best[has_votes]
    return labels


def strain_from_deformation(deformation: NDArray, rotation: NDArray, frame: str = "lab") -> dict:
    """Small-strain tensor components from a deformation gradient.

    With ``p ~ F t`` and polar decomposition ``F = V R = R U`` the stretch in
    the lab frame is ``V`` and in the crystal (template) frame ``U``.  The
    strain is ``sym(V) - I`` or ``sym(U) - I``.

    Parameters
    ----------
    deformation : ndarray
        ``(N, 3, 3)`` deformation gradients ``F``.
    rotation : ndarray
        ``(N, 3, 3)`` rotations ``R`` from the rigid fit.
    frame : {"lab", "crystal"}
        Frame in which to express the strain.

    Returns
    -------
    dict of ndarray
        ``e_xx, e_yy, e_zz, e_xy, e_xz, e_yz`` components, ``dilation``
        (mean normal strain) and ``equivalent`` (von Mises deviatoric strain),
        plus the full ``(N, 3, 3)`` tensor under ``"tensor"``.
    """
    f = np.asarray(deformation, dtype=np.float64)
    r = np.asarray(rotation, dtype=np.float64)
    if frame == "lab":
        stretch = f @ np.transpose(r, (0, 2, 1))
    elif frame == "crystal":
        stretch = np.transpose(r, (0, 2, 1)) @ f
    else:
        raise ValueError("frame must be 'lab' or 'crystal'")
    e = 0.5 * (stretch + np.transpose(stretch, (0, 2, 1))) - np.eye(3)[None]
    dil = np.trace(e, axis1=1, axis2=2) / 3.0
    dev = e - dil[:, None, None] * np.eye(3)[None]
    equivalent = np.sqrt((2.0 / 3.0) * np.einsum("nij,nij->n", dev, dev))
    return {
        "e_xx": e[:, 0, 0],
        "e_yy": e[:, 1, 1],
        "e_zz": e[:, 2, 2],
        "e_xy": e[:, 0, 1],
        "e_xz": e[:, 0, 2],
        "e_yz": e[:, 1, 2],
        "dilation": dil,
        "equivalent": equivalent,
        "tensor": e,
    }


def bond_angles(dxyz: NDArray, valid: NDArray) -> NDArray:
    """All bond angles at each site between pairs of valid neighbor vectors.

    Parameters
    ----------
    dxyz : ndarray
        ``(N, K, 3)`` neighbor vectors.
    valid : ndarray
        ``(N, K)`` mask of first-shell neighbors.

    Returns
    -------
    ndarray
        ``(N, K*(K-1)/2)`` angles in degrees, ``nan`` where either neighbor is
        invalid.
    """
    n, k, _ = dxyz.shape
    unit = dxyz / np.linalg.norm(dxyz, axis=-1, keepdims=True).clip(1e-12)
    iu, ju = np.triu_indices(k, 1)
    cos = np.einsum("nkd,nkd->nk", unit[:, iu], unit[:, ju])
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    ang[~(valid[:, iu] & valid[:, ju])] = np.nan
    return ang


def convex_hull_distance(xyz: NDArray) -> NDArray:
    """Signed distance from each point to the convex hull surface (positive inside)."""
    xyz = np.asarray(xyz, dtype=float)
    hull = ConvexHull(xyz)
    eq = hull.equations  # (F, 4): n . x + d = 0, outward normals
    d = -(xyz @ eq[:, :3].T + eq[:, 3][None, :])
    return d.min(axis=1)


def sample_volume(volume: NDArray, xyz: NDArray, radius: float = 1.5) -> NDArray:
    """Mean volume intensity inside a sphere around each site (voxel units).

    Parameters
    ----------
    volume : ndarray
        3D array indexed ``[x, y, z]`` matching the coordinate order of ``xyz``.
    xyz : ndarray
        ``(N, 3)`` site positions in voxel coordinates.
    radius : float
        Integration sphere radius in voxels.

    Returns
    -------
    ndarray
        ``(N,)`` mean intensity; sites whose sphere leaves the volume use only
        the in-bounds voxels.
    """
    volume = np.asarray(volume)
    xyz = np.asarray(xyz, dtype=float)
    r = int(np.ceil(radius))
    rng = np.arange(-r, r + 1)
    off = np.stack(np.meshgrid(rng, rng, rng, indexing="ij"), -1).reshape(-1, 3)
    off = off[np.linalg.norm(off, axis=1) <= radius]
    center = np.rint(xyz).astype(int)
    idx = center[:, None, :] + off[None, :, :]  # (N, V, 3)
    shape = np.array(volume.shape)
    inside = np.all((idx >= 0) & (idx < shape), axis=-1)
    idx = np.clip(idx, 0, shape - 1)
    vals = volume[idx[..., 0], idx[..., 1], idx[..., 2]].astype(float)
    vals[~inside] = 0.0
    counts = inside.sum(1).clip(1)
    return vals.sum(1) / counts


def kmeans_1d(
    values: NDArray, num_clusters: int = 2, num_iter: int = 50
) -> tuple[NDArray, NDArray]:
    """Simple 1D k-means with quantile initialization.

    Returns
    -------
    labels, centers : ndarray
        ``(N,)`` labels sorted so that cluster 0 has the smallest center, and
        ``(num_clusters,)`` sorted centers.
    """
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    q = np.linspace(0, 1, num_clusters + 2)[1:-1]
    centers = np.quantile(v[finite], q)
    labels = np.zeros(v.shape, dtype=int)
    for _ in range(num_iter):
        labels = np.argmin(np.abs(v[:, None] - centers[None, :]), axis=1)
        new = np.array(
            [
                v[finite & (labels == c)].mean() if np.any(finite & (labels == c)) else centers[c]
                for c in range(num_clusters)
            ]
        )
        if np.allclose(new, centers):
            break
        centers = new
    order = np.argsort(centers)
    remap = np.empty_like(order)
    remap[order] = np.arange(num_clusters)
    return remap[labels], centers[order]


def rotation_to_quaternion(rotation: NDArray) -> NDArray:
    """Convert ``(N, 3, 3)`` rotation matrices to ``(N, 4)`` unit quaternions (w, x, y, z)."""
    from scipy.spatial.transform import Rotation

    q = Rotation.from_matrix(np.asarray(rotation)).as_quat()  # x, y, z, w
    q = np.roll(q, 1, axis=1)
    q[q[:, 0] < 0] *= -1
    return q
