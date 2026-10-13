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
    PtychoProbeConstraintParams,
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


class TestProbeFourier:
    def _ptycho(self, counts, probe_model, num_slices=1):
        from quantem.diffractive_imaging import ProbeFourier  # noqa: F401

        pdset = PtychographyDatasetRaster.from_dataset4dstem(
            make_dset(counts), verbose=0, learn_descan=False, learn_scan_positions=False
        )
        pdset.preprocess(
            com_fit_function="constant",
            plot_rotation=False,
            plot_com=False,
            probe_energy=PROBE_ENERGY,
            force_com_rotation=0,
            force_com_transpose=False,
        )
        ptycho = Ptychography.from_models(
            dset=pdset,
            obj_model=ObjectPixelated.from_uniform(
                num_slices=num_slices, slice_thicknesses=10, obj_type="potential"
            ),
            probe_model=probe_model,
            detector_model=DetectorPixelated(),
            device="cpu",
            verbose=0,
            rng=0,
        )
        ptycho.preprocess(obj_padding_px=(8, 8), plot_rotation=False, plot_com=False)
        return ptycho

    def test_matches_pixelated_probe(self, counts):
        from quantem.diffractive_imaging import ProbeFourier

        params = {
            "energy": PROBE_ENERGY,
            "defocus": 120.0,
            "semiangle_cutoff": SEMIANGLE,
            "C12": 30.0,
            "phi12": 0.4,
            "C21": 400.0,
            "phi21": -1.0,
        }
        probes = []
        for cls in (ProbePixelated, ProbeFourier):
            ptycho = self._ptycho(counts, cls.from_params(dict(params), num_probes=1))
            probes.append(ptycho.probe_model.probe.detach()[0])
        a, b = probes
        overlap = (a.conj() * b).sum().abs() / (
            a.abs().square().sum() * b.abs().square().sum()
        ).sqrt()
        assert overlap > 0.99999
        coefs = ptycho.probe_model.aberration_coefs
        assert coefs["C10"] == pytest.approx(-120.0, abs=1e-3)
        assert coefs["C12"] == pytest.approx(30.0, abs=1e-3)
        assert coefs["phi12"] == pytest.approx(0.4, abs=1e-4)

    def test_learns_defocus_and_fits_phase(self):
        from quantem.diffractive_imaging import ProbeFourier

        probe_model = ProbeFourier.from_params(
            {"energy": PROBE_ENERGY, "defocus": 0.0, "semiangle_cutoff": SEMIANGLE},
            num_probes=2,
            aberration_keys=("C10", "C12"),
        )
        ptycho = self._ptycho(make_counts(0, dose=5e4), probe_model)
        assert float(probe_model.probe.detach().abs().square().sum()) == pytest.approx(
            ptycho.dset.mean_diffraction_intensity, rel=1e-3
        )
        ptycho.reconstruct(
            num_iters=30,
            batch_size=36,
            optimizer_params={
                "object": OptimizerParams.Adam(lr=3e-2),
                "probe": {
                    "default": OptimizerParams.Adam(lr=1e-3),
                    "aberrations": OptimizerParams.Adam(lr=2e-2),
                },
            },
        )
        # the data were simulated with C10 = +50 A (a phase of pi * wavelength * 50 * q^2)
        assert probe_model.aberration_coefs["C10"] == pytest.approx(50.0, abs=8.0)
        before = probe_model.probe.detach().clone()
        coefs = probe_model.fit_aberrations()
        assert coefs["C10"] == pytest.approx(50.0, abs=5.0)
        torch.testing.assert_close(probe_model.probe.detach(), before, rtol=1e-4, atol=1e-3)

    def test_initialize_from_shifts_defocus(self, counts):
        from quantem.diffractive_imaging import ProbeFourier

        params = {"energy": PROBE_ENERGY, "defocus": 50.0, "semiangle_cutoff": SEMIANGLE}
        source = self._ptycho(counts, ProbeFourier.from_params(dict(params), num_probes=2), 2)
        target = self._ptycho(counts, ProbeFourier.from_params(dict(params), num_probes=2), 5)
        target.initialize_from(source, pad_slices=(2, 1))
        assert target.probe_model.aberration_coefs["C10"] == pytest.approx(-50.0 - 20.0, abs=1e-3)
        target.probe_model.reset()  # the transferred probe is the new starting point
        assert target.probe_model.aberration_coefs["C10"] == pytest.approx(-70.0, abs=1e-3)


class TestLowpassDepthTV:
    def test_tilted_columns_cost_less_after_lowpass(self, counts):
        from quantem.diffractive_imaging import PtychoObjConstraintParams

        ptycho = make_ptycho(make_dset(counts), num_slices=4, num_probes=1)
        obj_model = ptycho.obj_model
        rows, cols = obj_model.obj.shape[-2:]
        dy, dx = obj_model.sampling
        yy, xx = np.meshgrid(np.arange(rows) * dy, np.arange(cols) * dx, indexing="ij")
        # a lattice of sharp columns that moves 0.3 A sideways per slice, as for a tilted crystal
        tilted = np.stack(
            [
                np.cos(2 * np.pi * (xx - 0.3 * s) / 2.0) * np.cos(2 * np.pi * yy / 2.0)
                for s in range(4)
            ]
        )
        array = torch.as_tensor(tilted, dtype=torch.float32)
        costs = {}
        for sigma in (0.0, 1.0):
            obj_model.constraints = PtychoObjConstraintParams.Raster(
                tv_weight_z=1.0, tv_z_sigma=sigma, positivity=False
            )
            costs[sigma] = float(obj_model.get_tv_loss(array))
        assert costs[1.0] < 0.2 * costs[0.0]
        # a depth change of the slowly varying part is still penalized
        ramp = torch.as_tensor(np.stack([np.full((rows, cols), float(s)) for s in range(4)]))
        assert float(obj_model.get_tv_loss(ramp.float())) == pytest.approx(1.0, rel=1e-4)


class TestProbeFourierReuse:
    def test_new_reconstruction_keeps_the_learned_probe(self, counts):
        from quantem.diffractive_imaging import ProbeFourier

        params = {"energy": PROBE_ENERGY, "defocus": 50.0, "semiangle_cutoff": SEMIANGLE}
        probe_model = ProbeFourier.from_params(params, num_probes=2)
        first = TestProbeFourier()._ptycho(counts, probe_model)
        probe_model.shift_defocus(25.0)
        probe_model._initial_aberrations = probe_model._aberrations.detach().clone()
        before = probe_model.probe.detach().clone()
        TestProbeFourier()._ptycho(counts, probe_model)  # preprocess runs set_initial_probe again
        torch.testing.assert_close(probe_model.probe.detach(), before)
        assert first.probe_model is probe_model

    def test_save_and_reload(self, counts, tmp_path):
        from quantem.diffractive_imaging import ProbeFourier

        params = {"energy": PROBE_ENERGY, "defocus": 50.0, "semiangle_cutoff": SEMIANGLE}
        probe_model = ProbeFourier.from_params(params, num_probes=2)
        ptycho = TestProbeFourier()._ptycho(counts, probe_model)
        probe_model.shift_defocus(13.0)
        before = probe_model.probe.detach().clone()
        path = tmp_path / "probe_fourier.zip"
        ptycho.save(path, mode="o", verbose=False)
        pdset = PtychographyDatasetRaster.from_dataset4dstem(
            make_dset(counts), verbose=0, learn_descan=False, learn_scan_positions=False
        )
        pdset.preprocess(
            com_fit_function="constant",
            plot_rotation=False,
            plot_com=False,
            probe_energy=PROBE_ENERGY,
            force_com_rotation=0,
            force_com_transpose=False,
        )
        loaded = Ptychography.from_file(path, dset=pdset)
        assert loaded.probe_model.aberration_coefs["C10"] == pytest.approx(-37.0, abs=1e-3)
        torch.testing.assert_close(loaded.probe_model.probe.detach(), before)

    def test_field_tv_penalizes_noisy_fields(self, counts):
        from quantem.diffractive_imaging import ProbeFourier, PtychoProbeConstraintParams

        params = {"energy": PROBE_ENERGY, "defocus": 50.0, "semiangle_cutoff": SEMIANGLE}
        probe_model = ProbeFourier.from_params(params, num_probes=1)
        TestProbeFourier()._ptycho(counts, probe_model)
        probe_model.constraints = PtychoProbeConstraintParams.Raster(tv_weight=1.0)
        smooth = float(probe_model.apply_soft_constraints(probe_model.probe))
        with torch.no_grad():
            probe_model._phase.data = 0.5 * torch.randn_like(probe_model._phase)
        noisy = float(probe_model.apply_soft_constraints(probe_model.probe))
        assert noisy > smooth + 0.3

    def test_physical_probe_learns_only_coefficients(self):
        from quantem.diffractive_imaging import ProbeFourier, PtychoProbeConstraintParams

        probe_model = ProbeFourier.from_params(
            {"energy": PROBE_ENERGY, "defocus": 0.0, "semiangle_cutoff": SEMIANGLE},
            num_probes=2,
            aberration_keys=("C10", "C12"),
            learn_fields=False,
            learn_amplitude=True,
            learn_probe_tilt=True,
        )
        ptycho = TestProbeFourier()._ptycho(make_counts(0, dose=5e4), probe_model)
        assert set(probe_model.get_optimization_parameters()) == {"default", "aberrations", "tilt"}
        ptycho.reconstruct(
            num_iters=30,
            batch_size=36,
            optimizer_params={
                "object": OptimizerParams.Adam(lr=3e-2),
                "probe": {
                    "default": OptimizerParams.Adam(lr=1e-3),
                    "aberrations": OptimizerParams.Adam(lr=2e-2),
                },
            },
            constraints={"probe": PtychoProbeConstraintParams.Raster(tv_weight=1.0)},
        )
        assert probe_model.aberration_coefs["C10"] == pytest.approx(50.0, abs=8.0)
        # the phase stays the aberration surface
        assert float(probe_model._phase.detach().abs().max()) == 0.0

    def test_frozen_fields_learn_mode_weights(self, counts):
        from quantem.diffractive_imaging import ProbeFourier

        params = {"energy": PROBE_ENERGY, "defocus": 50.0, "semiangle_cutoff": SEMIANGLE}
        probe_model = ProbeFourier.from_params(params, num_probes=3, learn_fields=False)
        TestProbeFourier()._ptycho(counts, probe_model)
        assert set(probe_model.get_optimization_parameters()) == {"aberrations", "modes"}

    def test_higher_order_basis_matches_aberration_surface(self, counts):
        from quantem.diffractive_imaging import ProbeFourier
        from quantem.diffractive_imaging.complex_probe import (
            aberration_surface,
            polar_spatial_frequencies,
        )

        coefs = {"C10": 40.0, "C32": 3e4, "phi32": 0.3, "C34": 2e4, "phi34": -0.2, "C41": 4e5}
        coefs.update({"phi41": 0.7, "C50": 2e7})
        probe_model = ProbeFourier.from_params(
            {"energy": PROBE_ENERGY, "semiangle_cutoff": SEMIANGLE, **coefs},
            num_probes=1,
            aberration_keys=("C10", "C32", "C34", "C41", "C50"),
        )
        TestProbeFourier()._ptycho(counts, probe_model)
        shape = tuple(int(v) for v in probe_model.roi_shape)
        sampling = tuple(
            float(v) for v in 1 / (probe_model.roi_shape * probe_model.reciprocal_sampling)
        )
        k, phi = polar_spatial_frequencies(shape, sampling)
        wavelength = 0.019687
        reference = aberration_surface(
            k * wavelength, phi, wavelength, {key: torch.tensor(v) for key, v in coefs.items()}
        )
        inside = probe_model._aperture > 0.5
        difference = (probe_model.chi().detach() - reference)[inside].abs().max()
        assert float(difference) < 1e-4
        for key, value in coefs.items():
            assert probe_model.aberration_coefs[key] == pytest.approx(value, rel=1e-4)


class TestProbeRealSpaceWindow:
    def test_modes_vanish_at_the_box_edge(self, counts):
        params = {"energy": PROBE_ENERGY, "defocus": 400.0, "semiangle_cutoff": SEMIANGLE}
        probe_model = ProbePixelated.from_params(params, num_probes=2)
        TestProbeFourier()._ptycho(counts, probe_model)
        probe_model.constraints = PtychoProbeConstraintParams.Raster(real_space_window=0.5)
        raw_probe = probe_model._probe.detach()
        raw = raw_probe.abs().square().sum(0).numpy()
        probe = probe_model._probe_orthogonalization_constraint(raw_probe)
        probe = probe_model.apply_hard_constraints(probe)
        intensity = probe.abs().square().sum(0).numpy()
        rows, cols = intensity.shape
        r = np.hypot(np.fft.fftfreq(rows)[:, None] * 2, np.fft.fftfreq(cols)[None, :] * 2)
        assert raw[r >= 1].sum() > 0
        assert np.all(intensity[r >= 1] < 1e-12)
        np.testing.assert_allclose(intensity[r < 0.45], raw[r < 0.45], rtol=1e-4, atol=1e-8)


class TestScanCalibration:
    def test_correct_ellipticity_makes_the_disk_round(self):
        from quantem.core.utils.diffractive_imaging_utils import (
            correct_ellipticity,
            fit_probe_ellipse,
        )

        rows, cols = np.indices((96, 96)).astype(float)
        theta = 0.4
        u = np.cos(theta) * (cols - 47.3) + np.sin(theta) * (rows - 48.6)
        v = -np.sin(theta) * (cols - 47.3) + np.cos(theta) * (rows - 48.6)
        disk = ((u / 30) ** 2 + (v / 26) ** 2 < 1).astype(float) * 100
        corrected = correct_ellipticity(disk[None], fit_probe_ellipse(disk, show=False))[0]
        _, _, a_axis, b_axis, _ = fit_probe_ellipse(corrected, show=False)
        assert abs(a_axis / b_axis - 1) < 0.01
        assert corrected.sum() == pytest.approx(disk.sum(), rel=1e-5)

    def test_scan_transform_stretches_the_raster(self, counts):
        def positions(transform):
            pdset = PtychographyDatasetRaster.from_dataset4dstem(
                make_dset(counts), verbose=0, learn_descan=False, learn_scan_positions=False
            )
            pdset.scan_transform = transform
            pdset.preprocess(
                com_fit_function="constant",
                plot_rotation=False,
                plot_com=False,
                probe_energy=PROBE_ENERGY,
                force_com_rotation=0,
                force_com_transpose=False,
            )
            ptycho = Ptychography.from_models(
                dset=pdset,
                obj_model=ObjectPixelated.from_uniform(num_slices=1, obj_type="potential"),
                probe_model=ProbePixelated.from_params(
                    {"energy": PROBE_ENERGY, "defocus": 50.0, "semiangle_cutoff": SEMIANGLE}
                ),
                detector_model=DetectorPixelated(),
                device="cpu",
                verbose=0,
                rng=0,
            )
            ptycho.preprocess(obj_padding_px=(8, 8), plot_rotation=False, plot_com=False)
            ptycho.dset.reset()
            p = ptycho.dset.positions_px.detach().numpy().astype(np.float64)
            return p - p.mean(0)

        plain = positions(None)
        transform = np.array([[1.02, 0.01], [0.01, 0.99]])
        stretched = positions(transform)
        np.testing.assert_allclose(stretched, plain @ transform.T, atol=1e-3)


class TestPositionTilts:
    def _ptycho(self, counts, probe_tilt=(0.0, 0.0)):
        pdset = PtychographyDatasetRaster.from_dataset4dstem(
            make_dset(counts), verbose=0, learn_descan=False, learn_scan_positions=False
        )
        pdset.preprocess(
            com_fit_function="constant",
            plot_rotation=False,
            plot_com=False,
            probe_energy=PROBE_ENERGY,
            force_com_rotation=0,
            force_com_transpose=False,
        )
        ptycho = Ptychography.from_models(
            dset=pdset,
            obj_model=ObjectPixelated.from_uniform(
                num_slices=4, slice_thicknesses=20, obj_type="potential"
            ),
            probe_model=ProbePixelated.from_params(
                {"energy": PROBE_ENERGY, "defocus": 50.0, "semiangle_cutoff": SEMIANGLE},
                probe_tilt=probe_tilt,
            ),
            detector_model=DetectorPixelated(),
            device="cpu",
            verbose=0,
            rng=0,
        )
        ptycho.preprocess(obj_padding_px=(8, 8), plot_rotation=False, plot_com=False)
        with torch.no_grad():
            gen = torch.Generator().manual_seed(0)
            obj = ptycho.obj_model._obj
            obj.data = 0.3 * torch.rand(obj.shape, generator=gen, dtype=obj.dtype)
        return ptycho

    def _intensities(self, ptycho, indices):
        patch_data, _, fractional, descan = ptycho.dset.forward(indices, ptycho.obj_padding_px)
        probes = ptycho.probe_model.forward(fractional)
        patches = ptycho.obj_model.forward(patch_data)
        ptycho.compute_propagator_arrays()
        _, overlap = ptycho.forward_operator(
            patches, probes, descan, ptycho.dset.batch_tilts(indices)
        )
        return ptycho.predict_intensities(overlap, probes).detach()

    def test_uniform_position_tilts_match_probe_tilt(self, counts):
        indices = torch.arange(6)
        global_tilt = self._ptycho(counts, probe_tilt=(3.0, -2.0))
        local_tilt = self._ptycho(counts)
        local_tilt.dset.position_tilts = np.tile([3.0, -2.0], (local_tilt.dset.num_positions, 1))
        np.testing.assert_allclose(
            self._intensities(local_tilt, indices).numpy(),
            self._intensities(global_tilt, indices).numpy(),
            rtol=1e-4,
            atol=1e-6,
        )
        plain = self._intensities(self._ptycho(counts), indices).numpy()
        assert np.abs(plain - self._intensities(global_tilt, indices).numpy()).max() > 1e-4

    def test_raster_shaped_tilts_and_analytic_path(self, counts):
        ptycho = self._ptycho(counts)
        rows, cols = (int(n) for n in ptycho.dset.gpts)
        ptycho.dset.position_tilts = np.zeros((rows, cols, 2))
        assert tuple(ptycho.dset.position_tilts.shape) == (ptycho.dset.num_positions, 2)
        ptycho.reconstruct(
            num_iters=1,
            reset=True,
            optimizer_params={"object": OptimizerParams.Adam(lr=1e-3)},
            batch_size=8,
        )
        with pytest.raises(ValueError, match="per-position tilts"):
            ptycho.reconstruct(num_iters=1, autograd=False, batch_size=8)


class TestEstimateTiltMap:
    def test_recovers_kikuchi_center_offsets(self):
        from quantem.core.utils.diffractive_imaging_utils import fit_probe_circle
        from quantem.diffractive_imaging import estimate_tilt_map

        size, center, radius = 128, 64.0, 20.0
        rows, cols = np.indices((size, size)).astype(float)

        def pattern(dy, dx):
            # bright-field disk plus Kikuchi-like bands through the zone axis (the disk center
            # shifted by dy, dx), under a radial fall-off about the disk center
            bands = np.zeros((size, size))
            for angle, width in (
                (0.0, 3.0),
                (np.pi / 4, 2.0),
                (np.pi / 2, 3.0),
                (3 * np.pi / 4, 2.0),
            ):
                dist = (rows - center - dy) * np.cos(angle) - (cols - center - dx) * np.sin(angle)
                bands += np.exp(-((dist / width) ** 2))
            r = np.hypot(rows - center, cols - center)
            return 100.0 * (r < radius) + (r >= radius) * np.exp(-r / 30) * (1 + bands)

        data = np.zeros((8, 4, size, size))
        data[:4] = pattern(2.0, 1.0)
        data[4:] = pattern(-3.0, 2.0)
        fitted_radius = fit_probe_circle(data[0, 0], show=False)[2]
        # with the semiangle set to the fitted disk radius, tilts come out in detector pixels
        tilts, blocks = estimate_tilt_map(
            data, semiangle_mrad=fitted_radius, block_shape=(4, 4), search_px=4, smooth_blocks=0
        )
        assert tilts.shape == (8, 4, 2)
        np.testing.assert_allclose(blocks[0, 0], [2.0, 1.0], atol=0.26)
        np.testing.assert_allclose(blocks[1, 0], [-3.0, 2.0], atol=0.26)
        np.testing.assert_allclose(tilts[0, 0], blocks[0, 0])
