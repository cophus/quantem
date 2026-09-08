"""Tests for quantem.diffraction.rotations."""

import numpy as np
import pytest
import torch

from quantem.diffraction.rotations import (
    misorientation_angle_deg,
    qconj,
    qmult,
    qnormalize,
    qrotate,
    quat_from_axis_angle,
    quat_from_euler_zxz,
    quat_from_matrix,
    quat_from_zone_axis,
    quat_to_euler_zxz,
    quat_to_matrix,
    sample_zone_axes,
    zone_axis_from_quat,
)


@pytest.fixture
def random_quats():
    torch.manual_seed(0)
    return qnormalize(torch.randn(100, 4, dtype=torch.float64))


def test_matrix_roundtrip(random_quats):
    R = quat_to_matrix(random_quats)
    assert torch.allclose(quat_from_matrix(R), random_quats, atol=1e-10)


def test_euler_roundtrip(random_quats):
    e = quat_to_euler_zxz(random_quats)
    assert torch.allclose(qnormalize(quat_from_euler_zxz(e)), random_quats, atol=1e-8)


def test_rotate_matches_matrix(random_quats):
    v = torch.randn(100, 3, dtype=torch.float64)
    R = quat_to_matrix(random_quats)
    assert torch.allclose(qrotate(random_quats, v), (R @ v[..., None]).squeeze(-1), atol=1e-10)


def test_mult_conj_identity(random_quats):
    q = random_quats
    ident = qmult(q, qconj(q))
    expect = torch.zeros_like(q)
    expect[:, 0] = 1.0
    assert torch.allclose(ident, expect, atol=1e-10)


def test_zone_axis_roundtrip():
    torch.manual_seed(1)
    v = torch.randn(50, 3, dtype=torch.float64)
    v = v / torch.linalg.norm(v, dim=-1, keepdim=True)
    q = quat_from_zone_axis(v, in_plane_deg=25.0)
    assert torch.allclose(zone_axis_from_quat(q), v, atol=1e-10)


def test_axis_angle():
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    q = quat_from_axis_angle(axis, torch.tensor(np.pi / 2, dtype=torch.float64))
    v = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    assert torch.allclose(
        qrotate(q, v), torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64), atol=1e-9
    )


def test_misorientation_symmetry():
    # 90 degree rotation about z is a cubic symmetry: misorientation 0
    from ase.build import bulk

    from quantem.diffraction.crystal import Crystal

    xtl = Crystal.from_ase(bulk("Au", "fcc", a=4.08, cubic=True))
    qa = torch.tensor([1.0, 0, 0, 0], dtype=torch.float64)
    qb = quat_from_axis_angle(
        torch.tensor([0.0, 0, 1.0], dtype=torch.float64),
        torch.tensor(np.pi / 2, dtype=torch.float64),
    )
    ang = misorientation_angle_deg(qa, qb, xtl.sym_quats)
    assert float(ang) < 1e-4
    ang_nosym = misorientation_angle_deg(qa, qb)
    assert abs(float(ang_nosym) - 90.0) < 1e-6


def test_sample_zone_axes_wedge():
    corners = torch.tensor([[0, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=torch.float64)
    corners = corners / torch.linalg.norm(corners, dim=-1, keepdim=True)
    v, inds = sample_zone_axes(corners, 2.0)
    assert torch.allclose(torch.linalg.norm(v, dim=-1), torch.ones(v.shape[0], dtype=v.dtype))
    # corners present
    for c in corners:
        assert torch.linalg.norm(v - c, dim=-1).min() < 1e-8


def test_sample_zone_axes_isotropic():
    # the requested step holds in every direction, whatever the apex angle
    for apex_deg in (30.0, 120.0):
        a = np.deg2rad(apex_deg)
        corners = torch.tensor(
            [[0, 0, 1], [1, 0, 0], [np.cos(a), np.sin(a), 0]], dtype=torch.float64
        )
        v, _ = sample_zone_axes(corners, 2.0)
        dots = (v @ v.T).clamp(-1, 1)
        ang = torch.rad2deg(torch.acos(dots))
        ang.fill_diagonal_(1e9)
        nn = ang.min(dim=1).values
        # points on the equator row: spacing within 25% of the step
        eq = v[:, 2].abs() < 1e-9
        assert nn[eq].min() > 1.5 and nn[eq].max() < 2.5


def test_symmetry_reduced_zone_angles():
    from ase.build import bulk

    from quantem.diffraction.crystal import Crystal
    from quantem.diffraction.rotations import symmetry_reduced_zone_angles

    xtl = Crystal.from_ase(bulk("Ti", "bcc", a=3.31, cubic=True), verbose=False)
    za = torch.tensor(
        [[0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]],
        dtype=torch.float64,
    )
    za = za / torch.linalg.norm(za, dim=1, keepdim=True)
    ang = symmetry_reduced_zone_angles(za, xtl.sym_quats)
    # cubic axes are symmetry equivalent, as are the two <111> directions
    assert float(ang[0, 1]) < 1e-6 and float(ang[0, 2]) < 1e-6
    assert float(ang[3, 4]) < 1e-6
    assert abs(float(ang[0, 3]) - 54.7356) < 1e-3
