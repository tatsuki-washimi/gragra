"""Tests for reading prescribed trajectories from CSV."""

import io

import numpy as np
import pytest

from gragra.gravity_sources.trajectory import (
    PrescribedPointTrajectory,
    PrescribedRigidTransform,
    read_prescribed_point_trajectory_csv,
    read_prescribed_rigid_transform_csv,
)


def test_read_point_trajectory_csv_non_uniform():
    """Verify loading point trajectory with non-uniform time sampling."""
    csv_data = io.StringIO(
        "time_s,x_m,y_m,z_m\n"
        "0.0,0.0,0.0,1.0\n"
        "0.12,0.5,0.0,1.2\n"
        "0.45,1.0,0.2,1.5\n"
        "1.10,1.5,0.8,2.0\n"
    )
    traj = read_prescribed_point_trajectory_csv(csv_data, mass_kg=1200.0)

    assert isinstance(traj, PrescribedPointTrajectory)
    assert traj.mass_kg == 1200.0
    assert np.allclose(traj.time_s, [0.0, 0.12, 0.45, 1.10])
    assert traj.positions_m.shape == (4, 3)
    assert np.allclose(traj.positions_m[1], [0.5, 0.0, 1.2])


def test_read_point_trajectory_non_monotonic_rejection():
    """Non-monotonic timestamps must be rejected."""
    csv_data = io.StringIO(
        "time_s,x_m,y_m,z_m\n0.0,0.0,0.0,1.0\n0.5,0.5,0.0,1.2\n0.3,1.0,0.2,1.5\n"
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        read_prescribed_point_trajectory_csv(csv_data, mass_kg=100.0)


def test_read_rigid_transform_csv_with_rotations():
    """Verify loading rigid transforms with 3x3 rotation matrices."""
    # Identity rotation rows
    csv_data = io.StringIO(
        "time_s,tx_m,ty_m,tz_m,r00,r01,r02,r10,r11,r12,r20,r21,r22\n"
        "0.0,10.0,0.0,0.0,1,0,0,0,1,0,0,0,1\n"
        "0.35,12.0,1.0,0.5,0,-1,0,1,0,0,0,0,1\n"
    )
    rotation_cols = ("r00", "r01", "r02", "r10", "r11", "r12", "r20", "r21", "r22")
    transform = read_prescribed_rigid_transform_csv(
        csv_data,
        rotation_columns=rotation_cols,
    )

    assert isinstance(transform, PrescribedRigidTransform)
    assert np.allclose(transform.time_s, [0.0, 0.35])
    assert transform.translations_m.shape == (2, 3)
    assert transform.rotation_matrices.shape == (2, 3, 3)
    # Check 90 deg z-rotation
    expected_r1 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    assert np.allclose(transform.rotation_matrices[1], expected_r1)
