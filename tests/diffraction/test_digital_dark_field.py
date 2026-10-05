"""Digital dark field: apertures, polar selection and grain labels on synthetic peaks."""

import numpy as np
import pytest

from quantem.core.datastructures.vector import Vector
from quantem.diffraction import digital_dark_field as ddf


def _lattice_peaks(R=4, C=5, fields=("q_row", "q_col", "intensity")):
    """Square lattice g1=(10,0), g2=(0,10); cells with c >= 3 also carry (5,5)."""
    nested = []
    for r in range(R):
        row = []
        for c in range(C):
            pts = [[10 * i, 10 * j, 1.0 + r] for i in (-1, 0, 1) for j in (-1, 0, 1)]
            if c >= 3:
                pts.append([5.0, 5.0, 2.0])
            row.append(np.asarray(pts, dtype=float))
        nested.append(row)
    return Vector.from_data(nested, fields=list(fields))


def test_aperture_array_modes():
    g1, g2 = (10.0, 0.0), (0.0, 10.0)
    arr = ddf.aperture_array(g1, g2, n1_range=(-1, 1), n2_range=(-1, 1))
    assert arr.shape == (9, 2)
    no_center = ddf.aperture_array(
        g1, g2, n1_range=(-1, 1), n2_range=(-1, 1), radius_range=(1, np.inf)
    )
    assert no_center.shape == (8, 2)
    line = ddf.aperture_array(g1, mode="line", n1_range=(-2, 2), center=(50, 50))
    np.testing.assert_allclose(line[:, 1], 50.0)
    single = ddf.aperture_array(g1, g2, mode="single", shift=(1, 2))
    np.testing.assert_allclose(single, [[10.0, 20.0]])
    clipped = ddf.aperture_array(g1, g2, center=(15, 15), shape=(30, 30), edge=6)
    np.testing.assert_allclose(clipped, [[15.0, 15.0]])  # 5 and 25 lie within the edge
    with pytest.raises(ValueError):
        ddf.aperture_array(g1, g2, mode="bad")


def test_aperture_subtract_and_image():
    fine = ddf.aperture_array((5.0, 0.0), (0.0, 5.0), n1_range=(-2, 2), n2_range=(-2, 2))
    coarse = ddf.aperture_array((10.0, 0.0), (0.0, 10.0), n1_range=(-1, 1), n2_range=(-1, 1))
    super_only = ddf.aperture_array_subtract(fine, coarse, tol=1.0)
    assert super_only.shape == (25 - 9, 2)

    peaks = _lattice_peaks()
    image = ddf.aperture_ddf_image(peaks, super_only, radius=1.0)
    assert image.shape == (4, 5)
    assert np.all(image[:, :3] == 0)
    np.testing.assert_allclose(image[:, 3:], 2.0)

    # overlapping apertures count each peak once
    image_full = ddf.aperture_ddf_image(peaks, np.vstack([coarse, coarse]), radius=1.0)
    np.testing.assert_allclose(image_full[2, 0], 9 * 3.0)


def test_polar_fields_and_mask():
    peaks = _lattice_peaks(fields=("qx", "qy", "intensity"))
    polar = ddf.add_polar_fields(peaks)
    assert polar.fields[-2:] == ["qr", "qphi"]
    flat = polar.select_fields("qx", "qy", "qr", "qphi").numpy()
    np.testing.assert_allclose(flat[:, 2], np.hypot(flat[:, 0], flat[:, 1]), atol=1e-5)
    # (qx, qy) = (-10, 0) is straight up on screen: +90 degrees
    up = (flat[:, 0] == -10) & (flat[:, 1] == 0)
    np.testing.assert_allclose(flat[up, 3], 90.0)

    ring = ddf.polar_mask(peaks, 10.0, tol=0.5)
    assert ring.sum() == 4 * 20
    upper = ddf.polar_mask(peaks, 10.0, tol=0.5, phi_range=(45, 135))
    assert upper.sum() == 20
    wrapped = ddf.polar_mask(peaks, 10.0, tol=0.5, phi_range=(135, -135))  # 180 degrees
    assert wrapped.sum() == 20
    image = ddf.radial_ddf_image(peaks, 10.0, tol=0.5)
    np.testing.assert_allclose(image[1], 4 * 2.0)


def test_assign_grain_labels():
    peaks = _lattice_peaks(R=1, C=1, fields=("qx", "qy", "intensity"))
    labels = np.array([0, 0, 1, 1, -1, 2, 2, 2, -1])
    labeled = peaks.copy()
    labeled.add_fields("cluster", values=labels[:, None])
    out = ddf.assign_grain_labels(labeled, grain_labels=np.array([3, -1, 4]))
    grains = out.select_fields("grain_label").numpy()[:, 0]
    np.testing.assert_array_equal(grains, [3, 3, -1, -1, -2, 4, 4, 4, -2])


def test_fit_lattice_and_group_images():
    rng = np.random.default_rng(0)
    g1, g2 = np.array([20.0, 3.0]), np.array([4.0, 21.0])
    nested = []
    for r in range(3):
        row = []
        for c in range(3):
            n = np.array([[i, j] for i in (-2, -1, 0, 1, 2) for j in (-2, -1, 0, 1, 2)], float)
            q = n @ np.stack([g1, g2]) + rng.normal(0, 0.2, (len(n), 2))
            row.append(np.concatenate([q, np.ones((len(n), 1))], axis=1))
        nested.append(row)
    peaks = Vector.from_data(nested, fields=["q_row", "q_col", "intensity"])
    f1, f2 = ddf.fit_lattice(
        peaks, (19.0, 2.0), (5.0, 20.0), radius=4.0, n1_range=(-2, 2), n2_range=(-2, 2)
    )
    np.testing.assert_allclose(f1, g1, atol=0.1)
    np.testing.assert_allclose(f2, g2, atol=0.1)

    a = np.zeros((4, 4))
    a[:2] = 1
    b = np.zeros((4, 4))
    b[2:] = 1
    images = np.stack([a, 2 * a, a + 0.05 * b, b, 3 * b, np.eye(4)])
    labels = ddf.group_ddf_images(images, min_correlation=0.9)
    assert labels[0] == labels[1] == labels[2] >= 0
    assert labels[3] == labels[4] >= 0 and labels[3] != labels[0]
    assert labels[5] == -1


def test_cluster_centers_and_lattice_distance():
    peaks = _lattice_peaks(R=1, C=2)
    labeled = peaks.copy()
    n = labeled.total_rows
    labels = np.full(n, -1)
    labels[:9] = 0  # the 3x3 lattice in cell (0, 0)
    labeled.add_fields("cluster", values=labels[:, None])
    centers = ddf.cluster_centers(labeled)
    np.testing.assert_allclose(centers, [[0.0, 0.0]], atol=1e-6)

    d = ddf.lattice_distance([[10.0, 10.0], [5.0, 5.0], [11.0, 0.0]], (10.0, 0.0), (0.0, 10.0))
    np.testing.assert_allclose(d, [0.0, np.hypot(5, 5), 1.0])
