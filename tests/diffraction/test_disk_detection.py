"""Correlation options of the Bragg disk detection."""

import numpy as np
import torch

from quantem.diffraction.disk_detection import (
    detect_disks,
    detect_disks_batch,
    template_fourier,
)

H = W = 64
_YY, _XX = np.mgrid[0:H, 0:W]


def _disk(cy, cx, radius, amp):
    return amp / (1 + np.exp((np.hypot(_YY - cy, _XX - cx) - radius) / 0.7))


def _pattern():
    """Four disks spanning three decades of brightness on a bright halo."""
    planted = [(32, 32, 1000.0), (32, 44, 60.0), (20, 32, 12.0), (44, 20, 4.0)]
    dp = sum(_disk(cy, cx, 3, a) for cy, cx, a in planted)
    dp = dp + 30 * np.exp(-((_YY - 32) ** 2 + (_XX - 32) ** 2) / (2 * 25.0**2))
    dp = dp + np.random.default_rng(0).normal(0, 0.5, (H, W))
    template = torch.as_tensor(np.fft.ifftshift(_disk(32, 32, 3, 1.0)), dtype=torch.float)
    return torch.as_tensor(dp, dtype=torch.float), template_fourier(template), planted


def _found(peaks, planted, tol=1.5):
    """How many planted disks a peak list recovers."""
    if peaks.shape[0] == 0:
        return 0
    return sum(
        bool((np.hypot(peaks[:, 0] - cy, peaks[:, 1] - cx) < tol).any()) for cy, cx, _ in planted
    )


def test_corr_power_finds_weak_disks():
    """Hybrid correlation recovers disks the plain cross-correlation misses."""
    dp, tft, planted = _pattern()
    common = dict(min_spacing=4.0, edge_boundary=2, max_num_peaks=50)
    plain = detect_disks(dp, tft, corr_power=1.0, **common)
    hybrid = detect_disks(dp, tft, corr_power=0.5, **common)
    assert _found(plain, planted) < len(planted)
    assert _found(hybrid, planted) == len(planted)


def test_batched_matches_single_with_correlation_options():
    """The batched path reproduces the per-pattern result for every option."""
    dp, tft, _ = _pattern()
    common = dict(min_spacing=4.0, edge_boundary=2, max_num_peaks=50)
    cases = [
        {},
        dict(corr_power=0.5),
        dict(corr_power=0.0),
        dict(sigma_cc=1.5),
        dict(corr_power=0.7, sigma_cc=1.0, background_sigma=2.0),
    ]
    for kw in cases:
        for subpixel in ("upsample", "parabolic"):
            single = detect_disks(dp, tft, subpixel=subpixel, **common, **kw)
            batch = detect_disks_batch(
                torch.stack([dp, dp, dp]), tft, subpixel=subpixel, **common, **kw
            )[1]
            assert single.shape == batch.shape, (kw, subpixel)
            # float32 round-off only: the half-spectrum and full-spectrum
            # products differ in summation order once corr_power != 1
            assert np.allclose(single, batch, rtol=1e-4, atol=1e-5), (kw, subpixel)


def test_defaults_are_plain_cross_correlation():
    """corr_power=1 with no smoothing leaves the old behaviour untouched."""
    dp, tft, _ = _pattern()
    common = dict(min_spacing=4.0, edge_boundary=2, max_num_peaks=50)
    a = detect_disks(dp, tft, **common)
    b = detect_disks(dp, tft, corr_power=1.0, sigma_cc=None, **common)
    assert np.allclose(a, b)
