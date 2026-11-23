# Crystal-field tomography toys

Prototypes for multislice inverse tomography of HRTEM tilt series binned below lattice resolution,
where Bragg flicker (a dark grain plus a displaced bright copy, one or two tilts wide) has to be
modelled rather than masked. Not part of the package. Run scripts from inside their folder; outputs
go to `runs/`.

## 3D (`crystal_field_3d/`)

- `reproduce_tilt_series.py`: the simulated tilt series. FCC particle (twin pair plus an amorphous
  cap), 300 kV, 100 nm defocus, no aperture, tilts -90 to +89 degrees about y, 2 A pixels. Writes an
  npz with the images, angles, optics and the true mean potential. `python reproduce_tilt_series.py`
  (6.4 nm particle, a few minutes) or `--scale 1` (3.2 nm).
- `sim3d.py`: sample, fine multislice, truth maps, no-Bragg oracle, `band_limit` (round smooth taper
  that removes the pixel checkerboard).
- `invert_ms3d.py`: per-voxel sub-block multislice inversion (`SubMS3`); producer `run_ms3d.py`.
- `movie.py`: data and model side by side over the full tilt series, one gray scale.
- `crystal_field_3d_01.ipynb` (3.2 nm particle), `crystal_field_3d_02.ipynb` (6.4 nm, movies).

## Method

- Unknown: a real potential U on a fine grid, a 5^3 or 7^3 block of sub-voxels in every 2 A voxel.
- Forward: rotate U about the tilt axis, sum 2 A slices, multislice, defocus, |psi|^2, bin to the
  2 A pixel. The flicker comes from the same physics as the data.
- Constraint: after each step U >= -0.8 x local density (U low-passed to 2 A), so smooth fringes can
  carry the Bragg strength without inflating the mass. No atoms or lattice are needed.
- Fit: image MSE over the tilts plus a small TV on the 2 A density; every tenth tilt held out.
- Output: the density, U Fourier-cropped to the voxel grid. Crystallinity is not recovered yet.

## 2D (`crystal_field_2d/`)

The earlier 2D toy, notebooks 01 to 06 in order. 06 is the sub-block method; 01 to 05 are the beam
model, lattice, smooth-orientation, cell-field and oracle attempts that led to it.
