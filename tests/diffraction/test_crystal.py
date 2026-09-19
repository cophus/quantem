"""Tests for quantem.diffraction.crystal."""

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk

from quantem.diffraction.crystal import Crystal
from quantem.diffraction.rotations import quat_from_zone_axis


@pytest.fixture
def ti_beta():
    xtl = Crystal.from_ase(bulk("Ti", "bcc", a=3.31, cubic=True), name="Ti beta")
    xtl.calculate_structure_factors(k_max=1.5)
    return xtl


def test_symmetry_detection(ti_beta):
    assert ti_beta.pointgroup == "m-3m"
    assert ti_beta.laue_group == "m-3m"
    assert ti_beta.sym_quats.shape[0] == 24  # proper rotations of m-3m


def test_hcp_symmetry():
    xtl = Crystal.from_ase(bulk("Ti", "hcp", a=2.95, c=4.686))
    assert xtl.pointgroup == "6/mmm"
    assert xtl.sym_quats.shape[0] == 12


def test_bcc_absences(ti_beta):
    # h + k + l odd forbidden in bcc
    parity = ti_beta.hkl.sum(dim=1) % 2
    assert (parity == 0).all()


def test_ring_positions(ti_beta):
    # (110) ring at sqrt(2)/a
    g110 = np.sqrt(2) / 3.31
    assert np.isclose(float(ti_beta.g_len.min()), g110, atol=1e-6)


def test_pseudo_symmetry():
    ortho = Atoms("Au", positions=[[0, 0, 0]], cell=[4.000, 4.001, 4.002], pbc=True)
    exact = Crystal.from_ase(ortho, pseudo_symmetry_tol=None)
    pseudo = Crystal.from_ase(ortho, pseudo_symmetry_tol=0.01)  # 0.04 A on a 4 A cell
    assert exact.pointgroup_matching == "mmm"
    assert pseudo.pointgroup_matching == "m-3m"
    assert pseudo.sym_quats_matching.shape[0] == 24
    # exact group is retained for reporting/refinement
    assert pseudo.pointgroup == "mmm"


def test_zone_axis_wedge_anchored_001(ti_beta):
    wedge = ti_beta.zone_axis_wedge()
    assert torch.allclose(wedge[0], torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))


def test_generate_pattern(ti_beta):
    q = quat_from_zone_axis(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
    p = ti_beta.generate_pattern(q, energy_ev=200e3)
    # [001] zone: peaks on a square grid of 110-type spacings
    assert p["qx"].shape[0] > 4
    qr = torch.hypot(p["qx"], p["qy"])
    assert float(qr.min()) > 0.4  # no direct beam
    # pattern symmetric under 90 degree rotation
    rot = torch.stack((-p["qy"], p["qx"]), dim=1)
    orig = torch.stack((p["qx"], p["qy"]), dim=1)
    d = torch.cdist(rot, orig).min(dim=1).values
    assert float(d.max()) < 1e-6


def _images_in_wedge(xtl, n=3000, tol=1e-9):
    """Count, per random direction, its symmetry images inside the wedge."""
    from quantem.diffraction.rotations import quat_to_matrix

    rng = np.random.default_rng(0)
    d = rng.normal(size=(n, 3))
    d = torch.as_tensor(d / np.linalg.norm(d, axis=1, keepdims=True))
    c = xtl.zone_axis_wedge()
    Rs = quat_to_matrix(xtl.sym_quats_matching)
    imgs = torch.einsum("sij,nj->nsi", Rs, d)
    imgs = torch.cat([imgs, -imgs], dim=1).reshape(-1, 3)
    ok = torch.ones(imgs.shape[0], dtype=torch.bool)
    for i in range(3):
        nrm = torch.cross(c[i], c[(i + 1) % 3], dim=0)
        ok &= (imgs @ nrm) * torch.sign(nrm @ c[(i + 2) % 3]) >= -tol
    return ok.reshape(n, -1).sum(dim=1)


@pytest.mark.parametrize(
    "label,spacegroup,symbols,basis,cellpar,laue",
    [
        ("Si", 227, ["Si"], [(0, 0, 0)], [5.43] * 3 + [90] * 3, "m-3m"),
        (
            "pyrite",
            205,
            ["Fe", "S"],
            [(0, 0, 0), (0.385, 0.385, 0.385)],
            [5.42] * 3 + [90] * 3,
            "m-3",
        ),
        ("Ti", 194, ["Ti"], [(1 / 3, 2 / 3, 0.25)], [2.95, 2.95, 4.68, 90, 90, 120], "6/mmm"),
        (
            "CdI2 -3m1",
            164,
            ["Cd", "I"],
            [(0, 0, 0), (1 / 3, 2 / 3, 0.25)],
            [4.24, 4.24, 6.84, 90, 90, 120],
            "-3m",
        ),
        ("Bi R-3m", 166, ["Bi"], [(0, 0, 0.234)], [4.55, 4.55, 11.86, 90, 90, 120], "-3m"),
        (
            "P-31m",
            162,
            ["Cu", "O"],
            [(1 / 3, 2 / 3, 0), (0.4, 0, 0.3)],
            [5.0, 5.0, 7.0, 90, 90, 120],
            "-3m",
        ),
        (
            "ilmenite",
            148,
            ["Fe", "Ti", "O"],
            [(0, 0, 0.355), (0, 0, 0.146), (0.317, 0.023, 0.245)],
            [5.09, 5.09, 14.09, 90, 90, 120],
            "-3",
        ),
        (
            "rutile",
            136,
            ["Ti", "O"],
            [(0, 0, 0), (0.305, 0.305, 0)],
            [4.59, 4.59, 2.96, 90, 90, 90],
            "4/mmm",
        ),
        (
            "Pnma",
            62,
            ["Fe", "C"],
            [(0.18, 0.25, 0.33), (0.04, 0.25, 0.87)],
            [5.0, 6.7, 4.5, 90, 90, 90],
            "mmm",
        ),
    ],
)
def test_wedge_is_fundamental_domain(label, spacegroup, symbols, basis, cellpar, laue):
    from ase.spacegroup import crystal as ase_crystal

    atoms = ase_crystal(symbols, basis=basis, spacegroup=spacegroup, cellpar=cellpar)
    xtl = Crystal.from_ase(atoms, name=label, verbose=False)
    assert xtl.laue_group_matching == laue
    # exactly one symmetry image of every direction lies in the wedge: the
    # wedge covers all of orientation space once (the -3m1 setting used to
    # get a wedge rotated by 30 degrees, covering half the directions twice)
    hits = _images_in_wedge(xtl)
    assert int(hits.min()) == 1 and int(hits.max()) == 1
    labels = xtl.zone_axis_wedge_labels()
    assert len(labels) == 3 and all(len(t) > 2 for t in labels)


def test_wedge_follows_cell_setting():
    # a rotated Cartesian setting moves the symmetry axes; the wedge follows
    atoms = bulk("Si", "diamond", a=5.431, cubic=True)
    atoms.rotate(37, "z", rotate_cell=True)
    atoms.rotate(20, "x", rotate_cell=True)
    xtl = Crystal.from_ase(atoms, verbose=False)
    hits = _images_in_wedge(xtl)
    assert int(hits.min()) == 1 and int(hits.max()) == 1
    assert xtl.zone_axis_wedge_labels(mathtext=False) == ["[001]", "[011]", "[111]"]


def test_pseudo_symmetry_default_and_warning():
    ortho = Atoms("Au", positions=[[0, 0, 0]], cell=[4.000, 4.001, 4.002], pbc=True)
    xtl = Crystal.from_ase(ortho, verbose=False)  # default tolerance 1% of the cell
    assert xtl.pointgroup == "mmm" and xtl.pointgroup_matching == "m-3m"
    assert xtl.pseudo_symmetry_report["intensity_mismatch"] < 0.05
    msg = xtl.matching_symmetry_warning()
    assert msg is not None and "pseudo_symmetry_tol=None" in msg
    # the pseudo group's operators are exact rotations (orthonormalized), so
    # its wedge is a fundamental domain up to the cell distortion
    hits = _images_in_wedge(xtl, tol=1e-3)
    assert int(hits.min()) >= 1
    exact = Crystal.from_ase(bulk("Ti", "bcc", a=3.31, cubic=True), verbose=False)
    assert exact.matching_symmetry_warning() is None


def test_pseudo_symmetry_dimensionless_and_intensity_check():
    # an almost body-centered cell: the center atom 0.002 A off (0.5, 0.5, 0.5)
    # is body centered at the default tolerance, and its 100/010/001
    # patterns are identical within any measurable intensity
    almost_bcc = Atoms(
        "Fe2", scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5005]], cell=[4.0, 4.0, 4.0], pbc=True
    )
    xtl = Crystal.from_ase(almost_bcc, verbose=False)
    assert xtl.pointgroup_matching == "m-3m"
    assert xtl.pseudo_symmetry_report["intensity_mismatch"] < 1e-3
    # the same cell at an unmeasurably tight distance tolerance keeps its
    # own (lower) symmetry; the tolerance is a fraction of the lattice
    tight = Crystal.from_ase(almost_bcc, pseudo_symmetry_tol=1e-7, verbose=False)
    assert tight.pointgroup_matching == tight.pointgroup
    # a candidate whose intensities do not match within the intensity
    # tolerance is rejected and the cell keeps its own symmetry
    strict = Crystal.from_ase(almost_bcc, pseudo_symmetry_intensity_tol=1e-9, verbose=False)
    assert strict.pseudo_symmetry_report.get("candidate") == "m-3m"
    assert strict.pseudo_symmetry_report.get("rejected") is True
    assert strict.pointgroup_matching == strict.pointgroup
    assert "rejected" in strict.symmetry_summary()


def _l10(other: str, a: float = 3.58) -> Atoms:
    """Two species ordered in alternating (001) layers of an fcc lattice.

    The lattice stays cubic and every atom sits exactly on its site, so no
    relaxation of the positions recovers the cubic parent: only the
    diffracted intensities can say whether the ordering is visible.
    """
    at = Atoms(
        "Ni4",
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
        cell=[a, a, a],
        pbc=True,
    )
    at.symbols = ["Ni", "Ni", other, other]
    return at


def test_pseudo_symmetry_from_weak_ordering():
    """Ordering of species that scatter alike is found through the lattice.

    Transition metals next to each other in the periodic table (the Ni, Co,
    Mn of a cathode) give superlattice reflections far too weak to index, so
    the orientation library must fold the variants together. The relaxed
    position search cannot find this: the atoms are already where they
    belong and only the species differ.
    """
    weak = Crystal.from_ase(_l10("Co"), verbose=False)
    assert weak.pointgroup == "4/mmm"
    assert weak.pointgroup_matching == "m-3m"
    assert weak.sym_quats_matching.shape[0] == 3 * weak.sym_quats.shape[0]
    assert weak.pseudo_symmetry_report["route"] == "lattice"
    assert weak.pseudo_symmetry_report["intensity_mismatch"] < 0.01

    # a light partner makes the same ordering plainly visible, and the
    # candidate is rejected
    strong = Crystal.from_ase(_l10("Li"), verbose=False)
    assert strong.pointgroup_matching == strong.pointgroup
    assert strong.pseudo_symmetry_report["rejected"] is True
    assert strong.pseudo_symmetry_report["intensity_mismatch"] > 0.1

    # the intensity tolerance is the decision, and it is the user's
    borderline = Crystal.from_ase(_l10("Al"), verbose=False)
    assert borderline.pointgroup_matching == borderline.pointgroup
    loose = Crystal.from_ase(_l10("Al"), pseudo_symmetry_intensity_tol=0.1, verbose=False)
    assert loose.pointgroup_matching == "m-3m"


def test_true_symmetry_cells_are_unchanged():
    """The lattice route must not disturb cells that are already at their
    lattice's symmetry, nor accept a lattice symmetry the structure breaks."""
    from ase.build import bulk

    for atoms, pg in (
        (bulk("Si", "diamond", a=5.43), "m-3m"),
        (bulk("Ti", "hcp", a=2.95, c=4.686), "6/mmm"),
        (bulk("Ti", "bcc", a=3.26, cubic=True), "m-3m"),
    ):
        xtl = Crystal.from_ase(atoms, verbose=False)
        assert xtl.pointgroup_matching == pg
        assert xtl.sym_quats_matching.shape[0] == xtl.sym_quats.shape[0]

    # corundum sits on a hexagonal lattice but its structure is only -3m;
    # the lattice route proposes 6/mmm and the intensities reject it
    from ase.spacegroup import crystal as ase_crystal

    al2o3 = ase_crystal(
        ("Al", "O"),
        basis=[(0, 0, 0.3522), (0.3064, 0, 0.25)],
        spacegroup=167,
        cellpar=[4.7607, 4.7607, 12.9947, 90, 90, 120],
    )
    xtl = Crystal.from_ase(al2o3, verbose=False)
    assert xtl.pointgroup_matching == "-3m"
    assert xtl.pseudo_symmetry_report["candidate"] == "6/mmm"
    assert xtl.pseudo_symmetry_report["rejected"] is True


def test_projected_rotation_order():
    """Apparent zero-layer symmetry, which limits in-plane indexing."""
    bcc = Crystal.from_ase(
        bulk("Ti", "bcc", a=3.26, cubic=True), verbose=False
    ).calculate_structure_factors(k_max=1.5)
    hcp = Crystal.from_ase(
        bulk("Ti", "hcp", a=2.95, c=4.686), verbose=False
    ).calculate_structure_factors(k_max=1.5)

    def cartesian(xtl, uvw):
        d = torch.as_tensor(np.asarray(uvw, dtype=float), dtype=torch.float64) @ xtl.lat_real
        return (d / torch.linalg.norm(d)).numpy()

    # the zero-layer net of {110} along <111> is hexagonal, so the pattern
    # repeats every 60 degrees while the crystal repeats every 120
    assert bcc.projected_rotation_order(cartesian(bcc, (1, 1, 1))) == 6
    assert bcc.projected_rotation_order(cartesian(bcc, (0, 0, 1))) == 4
    assert bcc.projected_rotation_order(cartesian(bcc, (0, 1, 1))) == 2
    # a general zone axis keeps the two-fold that Friedel's law provides
    assert bcc.projected_rotation_order(cartesian(bcc, (1, 2, 3))) == 2
    assert hcp.projected_rotation_order(cartesian(hcp, (0, 0, 1))) == 6
    assert hcp.projected_rotation_order(cartesian(hcp, (1, 0, 0))) == 2

    # vectorized over a stack
    axes = np.stack([cartesian(bcc, u) for u in ((1, 1, 1), (0, 0, 1), (0, 1, 1))])
    assert list(bcc.projected_rotation_order(axes)) == [6, 4, 2]


def test_projected_order_matches_pattern_degeneracy():
    """The reported order is the rotation that leaves the pattern unchanged."""
    from quantem.diffraction.rotations import (
        qmult,
        qnormalize,
        quat_from_axis_angle,
        quat_from_zone_axis,
    )

    bcc = Crystal.from_ase(
        bulk("Ti", "bcc", a=3.26, cubic=True), verbose=False
    ).calculate_structure_factors(k_max=1.5)
    d = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64) @ bcc.lat_real
    axis = d / torch.linalg.norm(d)
    n = bcc.projected_rotation_order(axis.numpy())
    q = quat_from_zone_axis(axis)
    beam = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    spun = qnormalize(
        qmult(quat_from_axis_angle(beam, torch.tensor(2 * np.pi / n)), q)
    )

    a = bcc.generate_pattern(q, energy_ev=200e3, sigma_excitation=0.02)
    b = bcc.generate_pattern(spun, energy_ev=200e3, sigma_excitation=0.02)
    pa = torch.stack([a["qx"], a["qy"]], dim=1)
    pb = torch.stack([b["qx"], b["qy"]], dim=1)
    assert pa.shape == pb.shape
    # every peak of one pattern sits on a peak of the other, same intensity
    dist = torch.cdist(pa, pb)
    dmin, j = dist.min(dim=1)
    assert float(dmin.max()) < 1e-6
    rel = (a["intensity"] - b["intensity"][j]).abs().max() / a["intensity"].max()
    assert float(rel) < 1e-6
