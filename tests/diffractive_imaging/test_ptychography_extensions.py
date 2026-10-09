"""
Tests for the forward-model box larger than the detector, the scan affine transform, learned
slice thickness, count splitting, slice refinement, density-matrix mode orthogonalization, and
initialization of one reconstruction from another.
"""

import numpy as np
import pytest
import torch

from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.ml import OptimizerParams
from quantem.diffractive_imaging import (
    DetectorPixelated,
    ObjectPixelated,
    ProbePixelated,
    Ptychography,
    PtychographyDatasetRaster,
    refine_slices,
    split_counts,
)
from quantem.diffractive_imaging.detector_models import pixel_overlap_matrix

N = 32  # detector pixels
Q_MAX = 0.5  # inverse Angstroms
PROBE_ENERGY = 300e3  # eV
SEMIANGLE = 5.0  # mrad, aperture at half the detector edge
GPTS = 12  # scan positions per side
STEP = 2  # scan step in object pixels


def make_counts(seed: int = 0, dose: float = 2000.0) -> np.ndarray:
    """Poisson counts from a white-noise phase object and a defocused probe."""
    rng = np.random.default_rng(seed)
    sampling = 1 / (2 * Q_MAX)
    obj = np.exp(1j * 0.3 * rng.random((N + GPTS * STEP, N + GPTS * STEP)))
    q = np.fft.fftfreq(N, sampling)
    q = np.hypot(q[:, None], q[None, :])
    wavelength = 0.019687
    aperture = (q * wavelength * 1e3 <= SEMIANGLE).astype(float)
    probe = np.fft.ifft2(aperture * np.exp(-1j * np.pi * wavelength * 50 * q**2))
    counts = np.zeros((GPTS, GPTS, N, N))
    for i in range(GPTS):
        for j in range(GPTS):
            patch = obj[i * STEP : i * STEP + N, j * STEP : j * STEP + N]
            intensity = np.abs(np.fft.fft2(np.fft.fftshift(probe) * patch)) ** 2
            counts[i, j] = np.fft.fftshift(intensity)
    counts *= dose / counts.sum((-2, -1), keepdims=True)
    return rng.poisson(counts).astype(np.uint32)


def make_dset(counts: np.ndarray) -> Dataset4dstem:
    return Dataset4dstem.from_array(
        array=counts,
        sampling=(STEP / (2 * Q_MAX), STEP / (2 * Q_MAX), 2 * Q_MAX / N, 2 * Q_MAX / N),
        units=("A", "A", "A^-1", "A^-1"),
    )


def make_ptycho(
    dset: Dataset4dstem,
    num_slices: int = 3,
    num_probes: int = 2,
    roi_shape: tuple[int, int] | None = None,
    validation_dset: Dataset4dstem | None = None,
    learn_scan_affine: bool = False,
) -> Ptychography:
    pdset = PtychographyDatasetRaster.from_dataset4dstem(
        dset,
        verbose=0,
        learn_descan=False,
        learn_scan_positions=False,
        learn_scan_affine=learn_scan_affine,
        validation_dset=validation_dset,
    )
    pdset.preprocess(
        com_fit_function="constant",
        plot_rotation=False,
        plot_com=False,
        probe_energy=PROBE_ENERGY,
        force_com_rotation=0,
        force_com_transpose=False,
    )
    obj_model = ObjectPixelated.from_uniform(
        num_slices=num_slices, slice_thicknesses=10, obj_type="potential"
    )
    probe_model = ProbePixelated.from_params(
        probe_params={"energy": PROBE_ENERGY, "defocus": 50, "semiangle_cutoff": SEMIANGLE},
        num_probes=num_probes,
    )
    ptycho = Ptychography.from_models(
        dset=pdset,
        obj_model=obj_model,
        probe_model=probe_model,
        detector_model=DetectorPixelated(roi_shape=roi_shape),
        device="cpu",
        verbose=0,
        rng=0,
    )
    ptycho.preprocess(obj_padding_px=(8, 8), plot_rotation=False, plot_com=False)
    return ptycho


@pytest.fixture(scope="module")
def counts() -> np.ndarray:
    return make_counts()


class TestDetectorBox:
    @pytest.mark.parametrize("num_model", [32, 40, 48, 64])
    def test_overlap_matrix_conserves_intensity(self, num_model):
        matrix = pixel_overlap_matrix(32, num_model)
        assert matrix.shape == (32, num_model)
        interior = slice(num_model // 4, 3 * num_model // 4)
        np.testing.assert_allclose(matrix[:, interior].sum(0), 1.0, atol=1e-12)

    def test_overlap_matrix_identity_for_equal_grids(self):
        np.testing.assert_allclose(pixel_overlap_matrix(32, 32), np.eye(32), atol=1e-12)

    def test_larger_box_keeps_sampling_and_detector_shape(self, counts):
        reference = make_ptycho(make_dset(counts))
        ptycho = make_ptycho(make_dset(counts), roi_shape=(48, 48))
        np.testing.assert_allclose(ptycho.sampling, reference.sampling)
        np.testing.assert_allclose(
            ptycho.reciprocal_sampling * 48, reference.reciprocal_sampling * 32
        )
        assert tuple(ptycho.probe.shape[-2:]) == (48, 48)
        assert tuple(ptycho.dset.patch_indices.shape[-2:]) == (48, 48)

        idx = torch.arange(8)
        patches, _, fractional, _ = ptycho.dset.forward(idx, ptycho.obj_padding_px)
        _, exit_waves = ptycho.forward_operator(
            ptycho.obj_model.forward(patches), ptycho.probe_model.forward(fractional)
        )
        intensities = ptycho.detector_model.forward(exit_waves)
        assert tuple(intensities.shape) == (8, N, N)

        # the band-limited probe alone stays inside the detector, so its intensity is conserved
        probes = ptycho.probe_model.forward(fractional)
        total = (probes.abs() ** 2).sum((0, -2, -1))
        binned = ptycho.detector_model.forward(probes).sum((-2, -1))
        torch.testing.assert_close(binned, total, rtol=1e-4, atol=0)

        ptycho.reconstruct(
            num_iters=2,
            optimizer_params={"object": OptimizerParams.Adam(lr=1e-3)},
            batch_size=36,
        )
        assert np.isfinite(ptycho.iter_losses).all()


class TestScanAffine:
    def test_identity_returns_scan_positions(self, counts):
        ptycho = make_ptycho(make_dset(counts), learn_scan_affine=True)
        torch.testing.assert_close(ptycho.dset.positions_px, ptycho.dset.scan_positions_px)

    def test_affine_scales_about_center(self, counts):
        ptycho = make_ptycho(make_dset(counts), learn_scan_affine=True)
        ptycho.dset.scan_affine = 1.05 * torch.eye(2)
        positions = ptycho.dset.positions_px.detach()
        center = ptycho.dset.initial_scan_positions_px.mean(0)
        expected = center + 1.05 * (ptycho.dset.scan_positions_px.detach() - center)
        torch.testing.assert_close(positions, expected)

    def test_affine_is_learned(self, counts):
        ptycho = make_ptycho(make_dset(counts), learn_scan_affine=True)
        ptycho.reconstruct(
            num_iters=2,
            optimizer_params={
                "object": OptimizerParams.Adam(lr=1e-3),
                "dataset": {"scan_affine": OptimizerParams.Adam(lr=1e-3)},
            },
            batch_size=36,
        )
        assert not torch.equal(ptycho.dset.scan_affine.detach(), torch.eye(2))
        ptycho.reset_recon()
        torch.testing.assert_close(ptycho.dset.scan_affine.detach(), torch.eye(2))


class TestSliceThickness:
    def test_thickness_is_learned(self, counts):
        ptycho = make_ptycho(make_dset(counts))
        ptycho.obj_model.learn_slice_thickness = True
        ptycho.reconstruct(
            num_iters=2,
            optimizer_params={"object": OptimizerParams.Adam(lr=1e-2)},
            batch_size=36,
        )
        assert "slice_thickness" in ptycho.obj_model.optimizer_params
        assert not np.allclose(ptycho.slice_thicknesses, 10.0)
        assert ptycho.propagators.grad_fn is None

        # turning learning off keeps the learned value and drops the parameter group
        learned = ptycho.slice_thicknesses.copy()
        ptycho.obj_model.learn_slice_thickness = False
        ptycho.reconstruct(num_iters=1, optimizer_params={"object": OptimizerParams.Adam(lr=1e-3)})
        np.testing.assert_allclose(ptycho.slice_thicknesses, learned)
        assert "slice_thickness" not in ptycho.obj_model.optimizer_params


class TestSplitCounts:
    def test_halves_sum_to_counts(self, counts):
        first, second = split_counts(make_dset(counts), rng=1)
        np.testing.assert_array_equal(first.array + second.array, counts)
        assert abs(first.array.sum() / counts.sum() - 0.5) < 0.01

    def test_validation_loss_and_noise_floor(self, counts):
        first, second = split_counts(make_dset(counts), rng=1)
        ptycho = make_ptycho(first, validation_dset=second)
        assert ptycho.dset.has_validation_counts
        ptycho.reconstruct(
            num_iters=2,
            optimizer_params={"object": OptimizerParams.Adam(lr=1e-3)},
            batch_size=36,
        )
        assert len(ptycho.val_iter_losses) == 2
        assert ptycho.val_noise_floor is not None and ptycho.val_noise_floor > 0


class TestRefineSlices:
    @pytest.mark.parametrize("obj_type", ["potential", "complex"])
    def test_projection_conserved(self, obj_type):
        rng = np.random.default_rng(0)
        potential = rng.random((4, 8, 8)).astype(np.float32)
        if obj_type == "complex":
            obj = np.exp(1j * potential - 0.1 * potential).astype(np.complex64)
            refined = refine_slices(obj, "complex")
            np.testing.assert_allclose(np.angle(refined).sum(0), np.angle(obj).sum(0), rtol=1e-4)
        else:
            refined = refine_slices(potential, "potential")
            np.testing.assert_allclose(refined.sum(0), potential.sum(0), rtol=1e-5)
        assert refined.shape == (7, 8, 8)


class TestProbeOrthogonalization:
    def test_density_matrix_and_intensities_preserved(self, counts):
        ptycho = make_ptycho(make_dset(counts), num_probes=3)
        rng = np.random.default_rng(0)
        raw = torch.as_tensor(
            rng.standard_normal((3, N, N)) + 1j * rng.standard_normal((3, N, N)),
            dtype=torch.complex64,
        )
        modes = ptycho.probe_model._probe_orthogonalization_constraint(raw)
        flat = modes.reshape(3, -1)
        overlap = flat @ flat.conj().T
        torch.testing.assert_close(
            overlap, torch.diag(torch.diagonal(overlap)), atol=1e-3 * overlap.abs().max(), rtol=0
        )
        occupation = torch.diagonal(overlap).real
        assert torch.all(occupation[:-1] >= occupation[1:])
        raw_flat = raw.reshape(3, -1)
        torch.testing.assert_close(
            (raw_flat.T @ raw_flat.conj()).abs().sum(),
            (flat.T @ flat.conj()).abs().sum(),
            rtol=1e-4,
            atol=0,
        )
        intensity_raw = (torch.fft.fft2(raw).abs() ** 2).sum(0)
        intensity = (torch.fft.fft2(modes).abs() ** 2).sum(0)
        torch.testing.assert_close(intensity, intensity_raw, rtol=1e-3, atol=1e-2)


class TestInitializeFrom:
    def test_coarse_to_fine(self, counts):
        dset = make_dset(counts)
        coarse = make_ptycho(dset[..., 8:24, 8:24].copy(), num_slices=3, learn_scan_affine=True)
        coarse.reconstruct(
            num_iters=2,
            optimizer_params={
                "object": OptimizerParams.Adam(lr=1e-2),
                "dataset": {"scan_affine": OptimizerParams.Adam(lr=1e-3)},
            },
            batch_size=36,
        )
        # smooth, positive test object so interpolation and refinement are well defined
        shape = coarse.obj_shape_full
        rows, cols = np.meshgrid(np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
        smooth = 1 + np.cos(2 * np.pi * rows / shape[1]) * np.cos(2 * np.pi * cols / shape[2])
        profile = np.array([0.2, 1.0, 0.5])[:, None, None]
        coarse.obj_model._obj.data = torch.as_tensor(profile * smooth, dtype=torch.float32)

        fine = make_ptycho(dset, num_slices=5, roi_shape=(40, 40))
        fine.initialize_from(coarse, refine_slices=True)

        np.testing.assert_allclose(fine.slice_thicknesses, 5.0, rtol=1e-6)
        expected = (
            coarse.dset.positions_px.detach().numpy() - coarse.obj_padding_px
        ) * coarse.sampling / fine.sampling + fine.obj_padding_px
        np.testing.assert_allclose(fine.dset.positions_px.detach().numpy(), expected, rtol=1e-5)
        assert tuple(fine.probe.shape) == (2, 40, 40)
        np.testing.assert_allclose(
            (np.abs(fine.probe) ** 2).sum(), fine.dset.mean_diffraction_intensity, rtol=1e-4
        )
        # the resampled projection matches the analytic field at the mapped coordinates
        scale = coarse.sampling / fine.sampling
        fine_rows = np.arange(fine.obj_shape_full[1])
        fine_cols = np.arange(fine.obj_shape_full[2])
        source_rows = (fine_rows - fine.obj_padding_px[0]) / scale[0] + coarse.obj_padding_px[0]
        source_cols = (fine_cols - fine.obj_padding_px[1]) / scale[1] + coarse.obj_padding_px[1]
        r, c = np.meshgrid(source_rows, source_cols, indexing="ij")
        analytic = profile.sum() * (
            1 + np.cos(2 * np.pi * r / shape[1]) * np.cos(2 * np.pi * c / shape[2])
        )
        inside = (r > 2) & (r < shape[1] - 3) & (c > 2) & (c < shape[2] - 3)
        projection = fine.obj.sum(0)
        np.testing.assert_allclose(projection[inside], analytic[inside], rtol=0.01, atol=0.01)

    def test_pad_slices_with_vacuum(self, counts):
        dset = make_dset(counts)
        source = make_ptycho(dset, num_slices=3)
        source.reconstruct(
            num_iters=1, optimizer_params={"object": OptimizerParams.Adam(lr=1e-2)}, batch_size=36
        )
        padded = make_ptycho(dset, num_slices=6)
        with pytest.raises(ValueError):
            padded.initialize_from(source, pad_slices=(1, 1))
        padded.initialize_from(source, pad_slices=(2, 1))

        np.testing.assert_allclose(padded.slice_thicknesses, 10.0, rtol=1e-6)
        assert len(padded.slice_thicknesses) == 5
        np.testing.assert_allclose(padded.obj[:2], 0.0)
        np.testing.assert_allclose(padded.obj[-1], 0.0)
        np.testing.assert_allclose(padded.obj[2:5], source.obj, atol=1e-5)
        # the probe propagated through the 20 A of vacuum above returns to the source probe
        forward = padded.probe_model._compute_propagator_arrays(
            padded.sampling, 2, np.array([20.0])
        )[0]
        recovered = np.fft.ifft2(np.fft.fft2(padded.probe) * forward.numpy())
        for mode_recovered, mode_source in zip(recovered, source.probe):  # equal up to a phase
            overlap = np.abs(np.vdot(mode_recovered, mode_source))
            norm = np.linalg.norm(mode_recovered) * np.linalg.norm(mode_source)
            assert overlap / norm > 0.999


class TestPhaseAmplitude:
    def test_fields_combine_and_constrain(self, counts):
        ptycho = make_ptycho(make_dset(counts), num_slices=3)
        obj_model = ObjectPixelated.from_uniform(
            num_slices=3, slice_thicknesses=10, obj_type="phase_amplitude"
        )
        obj_model._initialize_obj((3, 16, 16), sampling=(0.1, 0.1))
        with torch.no_grad():
            obj_model._obj.data = 0.3 * torch.ones(3, 16, 16)
            obj_model._amplitude.data = torch.full((3, 16, 16), 1.4)
            obj_model._amplitude.data[0, 0, 0] = 0.5
        amplitude = obj_model.amplitude
        assert float(amplitude.max()) == pytest.approx(1.0)  # clamped to [0, 1]
        assert float(amplitude[0, 0, 0]) == pytest.approx(0.5)
        expected = amplitude * torch.exp(1j * obj_model.phase)
        torch.testing.assert_close(obj_model.obj, expected)
        assert len(obj_model.params) == 2
        del ptycho

    def test_reconstruct_and_promote_from_potential(self, counts):
        dset = make_dset(counts)
        coarse = make_ptycho(dset, num_slices=3)
        coarse.reconstruct(
            num_iters=1, optimizer_params={"object": OptimizerParams.Adam(lr=1e-3)}, batch_size=36
        )
        pdset = coarse.dset
        obj_model = ObjectPixelated.from_uniform(
            num_slices=3, slice_thicknesses=10, obj_type="phase_amplitude"
        )
        probe_model = ProbePixelated.from_params(
            probe_params={"energy": PROBE_ENERGY, "defocus": 50, "semiangle_cutoff": SEMIANGLE},
            num_probes=2,
        )
        fine = Ptychography.from_models(
            dset=pdset,
            obj_model=obj_model,
            probe_model=probe_model,
            detector_model=DetectorPixelated(),
            device="cpu",
            verbose=0,
            rng=0,
        )
        fine.preprocess(obj_padding_px=(8, 8), plot_rotation=False, plot_com=False)
        fine.initialize_from(coarse)
        np.testing.assert_allclose(fine.obj, coarse.obj, atol=1e-4)
        np.testing.assert_allclose(fine.obj_amplitude, 1.0)
        fine.reconstruct(
            num_iters=2,
            optimizer_params={"object": OptimizerParams.Adam(lr=1e-3)},
            batch_size=36,
        )
        assert np.isfinite(fine.iter_losses).all()
        assert fine.obj_amplitude.max() <= 1.0 + 1e-6
        assert not np.allclose(fine.obj_amplitude, 1.0)

    def test_dip_from_potential(self, counts):
        from quantem.core.ml import CNN2d
        from quantem.diffractive_imaging import ObjectDIP

        ptycho = make_ptycho(make_dset(counts), num_slices=2, num_probes=1)
        dip = ObjectDIP.from_pixelated(
            model=CNN2d(in_channels=4, dtype=torch.float32, final_activation="identity"),
            pixelated=ptycho.obj_model,
            obj_type="phase_amplitude",
        )
        assert dip.num_channels == 4
        assert dip.obj.is_complex() and tuple(dip.phase.shape) == tuple(ptycho.obj_model.obj.shape)


class TestAbsorption:
    def test_transmission_amplitude(self, counts):
        ptycho = make_ptycho(make_dset(counts), num_slices=2, num_probes=1)
        obj_model = ptycho.obj_model
        with torch.no_grad():
            obj_model._obj.data = 0.5 * torch.rand_like(obj_model._obj)
        obj_model.absorption = 0.2
        indices = ptycho.dset.patch_indices[:4]
        patches = obj_model.forward(indices)
        obj = obj_model.obj
        relative = obj - obj.mean(dim=(-2, -1), keepdim=True)
        expected = torch.exp(-0.2 * relative).reshape(obj_model.num_slices, -1)[:, indices]
        potential = obj.reshape(obj_model.num_slices, -1)[:, indices]
        torch.testing.assert_close(patches.abs(), expected)
        torch.testing.assert_close(torch.angle(patches), potential)
        np.testing.assert_allclose(
            ptycho.obj_amplitude, np.exp(-0.2 * relative.detach().numpy()), rtol=1e-5
        )

    def test_absorption_is_learned(self, counts):
        ptycho = make_ptycho(make_dset(counts))
        ptycho.obj_model.absorption = 0.1
        ptycho.obj_model.learn_absorption = True
        ptycho.reconstruct(
            num_iters=2,
            optimizer_params={"object": OptimizerParams.Adam(lr=1e-2)},
            batch_size=36,
        )
        assert "absorption" in ptycho.obj_model.optimizer_params
        assert ptycho.obj_model.absorption != pytest.approx(0.1)
        # reset restores the value set before reconstruction
        ptycho.obj_model.reset()
        assert ptycho.obj_model.absorption == pytest.approx(0.1)

    def test_absorbed_counts_return_as_background(self, counts):
        ptycho = make_ptycho(make_dset(counts), num_slices=2, num_probes=2)
        obj_model = ptycho.obj_model
        with torch.no_grad():
            obj_model._obj.data = 0.5 * torch.rand_like(obj_model._obj) - 0.1
        obj_model.absorption = 0.3
        obj_model.relative_absorption = False
        # amplitude only drops where the potential is positive
        patches = obj_model.forward(ptycho.dset.patch_indices[:4])
        assert patches.abs().max() <= 1.0 + 1e-6
        ptycho.dset.learn_background = True
        ptycho.dset.background_from_absorption = True
        indices = torch.arange(4)
        patch_data, _, fractional, _ = ptycho.dset.forward(indices, ptycho.obj_padding_px)
        probes = ptycho.probe_model.forward(fractional)
        _, overlap = ptycho.forward_operator(obj_model.forward(patch_data), probes)
        predicted = ptycho.predict_intensities(overlap, probes)
        incident = (probes.abs() ** 2).sum(dim=(0, -2, -1))
        coherent = ptycho.detector_model.forward(overlap).sum(dim=(-2, -1))
        assert torch.all(coherent < incident)
        torch.testing.assert_close(predicted.sum(dim=(-2, -1)), incident, rtol=1e-4, atol=1e-3)

    def test_validation_and_dip_transfer(self, counts):
        from quantem.core.ml import CNN2d
        from quantem.diffractive_imaging import ObjectDIP

        ptycho = make_ptycho(make_dset(counts), num_slices=2, num_probes=1)
        with pytest.raises(ValueError):
            ptycho.obj_model.absorption = -0.1
        complex_obj = ObjectPixelated.from_uniform(
            num_slices=2, slice_thicknesses=10, obj_type="complex"
        )
        with pytest.raises(ValueError):
            complex_obj.learn_absorption = True
        ptycho.obj_model.absorption = 0.15
        dip = ObjectDIP.from_pixelated(
            model=CNN2d(in_channels=2, dtype=torch.float32, final_activation="identity"),
            pixelated=ptycho.obj_model,
        )
        assert dip.absorption == pytest.approx(0.15)


class TestDIPSoftConstraints:
    def test_tv_z_reuses_forward_output(self, counts):
        from quantem.core.ml import CNN2d
        from quantem.diffractive_imaging import ObjectDIP, PtychoObjConstraintParams

        ptycho = make_ptycho(make_dset(counts), num_slices=3, num_probes=1)
        dip = ObjectDIP.from_pixelated(
            model=CNN2d(in_channels=3, dtype=torch.float32, final_activation="identity"),
            pixelated=ptycho.obj_model,
        )
        dip.constraints = PtychoObjConstraintParams.Raster(tv_weight_z=10.0)
        assert dip.has_soft_constraints
        calls = []
        dip.model.register_forward_hook(lambda *args: calls.append(1))
        dip.forward(ptycho.dset.patch_indices[:4])
        loss = dip.apply_soft_constraints(dip.soft_constraint_obj(), mask=dip.mask)
        assert len(calls) == 1  # the soft constraint reused the forward output
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in dip.model.parameters())

    def test_forward_hard_constraints(self, counts):
        from quantem.core.ml import CNN2d
        from quantem.diffractive_imaging import ObjectDIP, PtychoObjConstraintParams

        ptycho = make_ptycho(make_dset(counts), num_slices=3, num_probes=1)
        dip = ObjectDIP.from_pixelated(
            model=CNN2d(in_channels=3, dtype=torch.float32, final_activation="identity"),
            pixelated=ptycho.obj_model,
        )
        dip.constraints = PtychoObjConstraintParams.Raster(positivity=True, identical_slices=True)
        indices = ptycho.dset.patch_indices[:4]
        assert not dip.forward_hard_constraints
        dip.forward_hard_constraints = True
        patches = dip.forward(indices)
        phase = torch.angle(patches)
        assert float(phase.min()) >= -1e-6  # positivity in the forward model
        torch.testing.assert_close(phase[0], phase[-1])  # identical slices
        patches.abs().sum().backward()
        assert any(p.grad is not None for p in dip.model.parameters())


class TestShowObjSlices:
    def test_shared_scale_and_signed_values(self, counts, monkeypatch):
        import warnings

        from quantem.diffractive_imaging import ptychography_visualizations

        captured = {}

        def fake_show_2d(arrays, **kwargs):
            captured["arrays"] = arrays
            captured["norm"] = kwargs["norm"]
            return None, None

        monkeypatch.setattr(ptychography_visualizations, "show_2d", fake_show_2d)
        ptycho = make_ptycho(make_dset(counts), num_slices=3, num_probes=1)
        obj = np.stack([np.full((16, 16), v) for v in (-0.5, 0.0, 2.0)])
        obj[:, 0, 0] = [0.5, 1.0, -1.0]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ptycho.show_obj_slices(
                obj=obj, interval_scaling="all", lower_quantile=0.0, upper_quantile=1.0
            )
        # one manual color scale for every slice, from the signed values
        assert captured["norm"] == {"interval_type": "manual", "vmin": -1.0, "vmax": 2.0}
        assert np.min(captured["arrays"][0][0]) == pytest.approx(-0.5)


class TestDetectorPSF:
    def test_psf_blurs_and_conserves_counts(self):
        det = DetectorPixelated(psf=np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]]))
        assert np.isclose(det.psf.sum(), 1.0)
        waves = torch.ones((1, 2, 16, 16), dtype=torch.complex64)  # plane wave
        intensities = det.forward(waves)
        assert torch.isclose(intensities.sum(), torch.tensor(2.0 * 256), rtol=1e-4)
        center = intensities[0, 8, 8]
        assert torch.isclose(intensities[0, 8, 9] / center, torch.tensor(0.25))
        assert torch.isclose(intensities[0, 9, 9], torch.tensor(0.0))

    def test_psf_validation(self):
        with pytest.raises(ValueError):
            DetectorPixelated(psf=np.ones((2, 3)))
        with pytest.raises(ValueError):
            DetectorPixelated(psf=-np.ones((3, 3)))


class TestDetectorNoiseResponse:
    def test_recovers_known_response(self):
        from quantem.diffractive_imaging import detector_noise_response

        rng = np.random.default_rng(0)
        ky, kx = np.indices((40, 40))
        rate = np.where(np.hypot(ky - 20, kx - 20) < 6, 20.0, 0.3)  # electrons per pixel
        response = np.array([[0.0, 0.2, 0.0], [0.2, 1.0, 0.2], [0.0, 0.2, 0.0]])
        electrons = rng.poisson(rate, size=(16, 48, 40, 40))
        counts = np.empty_like(electrons)
        for index in np.ndindex(electrons.shape[:2]):
            # each electron gives one count in its pixel and one in each neighbor with p = 0.2
            e = electrons[index]
            spread = sum(
                np.roll(rng.binomial(e, p), shift, axis)
                for p, shift, axis in ((0.2, 1, 0), (0.2, -1, 0), (0.2, 1, 1), (0.2, -1, 1))
            )
            counts[index] = e + spread
        result = detector_noise_response(counts, bin_factor=1, num_rows=16)
        # E[M^2] / E[M] for M = 1 + Binomial(4, 0.2) counts per electron
        assert np.isclose(result["noise_gain"], (4 * 0.2 * 0.8 + 1.8**2) / 1.8, rtol=0.08)
        assert np.isclose(result["response"][2, 3], 0.2, atol=0.04)
        assert result["response"][3, 3] < 0.03
        assert np.isclose(result["psf"].sum(), 1.0)
        binned = detector_noise_response(counts, bin_factor=2, num_rows=16)["psf"]
        # an electron keeps itself and two of its four neighbor counts inside a 2x2 bin
        assert np.isclose(binned[binned.shape[0] // 2, binned.shape[1] // 2], 1.4 / 1.8, atol=0.02)
