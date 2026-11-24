"""DiffractionMoire: lattice discovery and refinement on a synthetic twisted bilayer."""

import numpy as np

from quantem.diffraction import DiffractionMoire


def test_twisted_bilayer():
    # two hexagonal lattices, |g| = 90 px, twisted by 21 degrees; double
    # diffraction adds a first-order reflection of one lattice to every
    # reflection of the other, at a tenth of the intensity; the pattern has a
    # 1% elliptic distortion, the inverse of the correction matrix of [e11, e12]
    shape, center, k, sampling = (512, 512), np.array([250.3, 260.7]), 90.0, 0.004
    e_true = np.array([0.004, -0.003])
    distort = np.linalg.inv(np.array([[1 + e_true[0], e_true[1]], [e_true[1], 1 - e_true[0]]]))

    def basis(deg):
        t = np.deg2rad([deg, deg + 60.0])
        return k * np.stack([np.cos(t), np.sin(t)], axis=1)

    hk = np.stack(np.meshgrid(np.arange(-3, 4), np.arange(-3, 4)), -1).reshape(-1, 2)
    first = np.array([[1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1]])
    ga, gb = hk @ basis(5.0), hk @ basis(26.0)
    double = np.concatenate(
        [
            ((first @ basis(5.0))[:, None] + gb[None]).reshape(-1, 2),
            (ga[:, None] + (first @ basis(26.0))[None]).reshape(-1, 2),
        ]
    )
    rr, cc = np.mgrid[: shape[0], : shape[1]]
    image = np.zeros(shape)
    for pos, inten in [(ga, 100.0), (gb, 100.0), (np.unique(double.round(6), axis=0), 10.0)]:
        for r, c in pos @ distort.T + center:
            if 6 < r < shape[0] - 6 and 6 < c < shape[1] - 6:
                image += inten * np.exp(-((rr - r) ** 2 + (cc - c) ** 2) / 2.0)
    image += np.random.default_rng(0).normal(0.0, 0.3, shape)

    dm = DiffractionMoire.from_data(image, sampling=sampling)
    dm.find_peaks(min_abs_intensity=2.0, plot=False).find_lattices(plot=False)

    # without the ellipse the distortion shows up as a lattice mismatch
    assert np.abs(dm.summary()["pairs"][0]["mismatch"]).max() > 2e-3
    dm.refine(ellipse=True, plot=False)
    assert np.allclose(dm.ellipse, e_true, atol=2e-4), dm.ellipse
    assert np.allclose(dm.center, center, atol=0.05), dm.center
    res = dm.summary()
    for lattice in res["lattices"]:
        assert np.allclose(lattice["k"], k * sampling, rtol=1e-3), lattice["k"]
    pair = res["pairs"][0]
    assert np.isclose(abs(pair["twist_deg"]), 21.0, atol=0.05), pair["twist_deg"]
    assert np.all(np.abs(pair["mismatch"]) < 1e-3), pair["mismatch"]
    period = 1.0 / (2.0 * k * sampling * np.sin(np.deg2rad(10.5)))
    assert np.allclose(pair["period_moire"], period, rtol=5e-3), pair["period_moire"]
    assert (dm.spot_terms[dm.indexed] == 2).sum() > 100, "double diffraction spots must be indexed"
