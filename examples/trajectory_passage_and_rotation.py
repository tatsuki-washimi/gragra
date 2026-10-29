"""Evaluate a point flyby and an opposed-mass rotor.

The rotor diagnostic checks its actual pi periodicity and reports a Fourier
fit.  A finite point rotor generally contains higher even harmonics at an
off-axis target, so the fit is deliberately accompanied by residual and
higher-even-harmonic diagnostics.
"""

import numpy as np

from gragra.gravity_sources import PointMassCloud
from gragra.gravity_sources.trajectory import (
    PrescribedPointTrajectory,
    PrescribedRigidTransform,
)
from gragra.observables.trajectory import (
    rigid_transform_acceleration,
    trajectory_acceleration,
    trajectory_potential,
)


def _rotation_z(theta: np.ndarray) -> np.ndarray:
    """Build the active z rotations for a one-dimensional angle array."""
    rotations = np.zeros((len(theta), 3, 3), dtype=np.float64)
    rotations[:, 0, 0] = np.cos(theta)
    rotations[:, 0, 1] = -np.sin(theta)
    rotations[:, 1, 0] = np.sin(theta)
    rotations[:, 1, 1] = np.cos(theta)
    rotations[:, 2, 2] = 1.0
    return rotations


def point_flyby() -> None:
    """Print instantaneous potential and acceleration during a flyby."""
    time_s = np.linspace(-2.0, 2.0, 9)
    trajectory = PrescribedPointTrajectory(
        mass_kg=10.0,
        time_s=time_s,
        positions_m=np.column_stack(
            (1.5 * time_s, np.zeros_like(time_s), time_s * 0.0)
        ),
    )
    targets = np.array([[0.0, 4.0, 0.0]])
    potential = trajectory_potential(trajectory, targets)[:, 0]
    acceleration = trajectory_acceleration(trajectory, targets)[:, 0]

    print("point flyby (target=(0, 4, 0) m)")
    for time, phi, vector in zip(time_s, potential, acceleration, strict=True):
        print(f"  t={time:+.2f} s  potential={phi:+.6e}  acceleration={vector}")


def opposed_mass_rotor() -> None:
    """Print pi periodicity and harmonic diagnostics for an opposed rotor."""
    sample_count = 128
    theta = 2.0 * np.pi * np.arange(sample_count) / sample_count
    transform = PrescribedRigidTransform(
        time_s=theta,
        translations_m=np.zeros((sample_count, 3)),
        rotation_matrices=_rotation_z(theta),
    )
    source = PointMassCloud(
        masses_kg=[1.0, 1.0], positions_m=[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    target = np.array([[3.2, 1.1, 0.8]])
    acceleration_x = rigid_transform_acceleration(source, transform, target)[:, 0, 0]

    half_period_error = np.max(
        np.abs(
            acceleration_x[: sample_count // 2] - acceleration_x[sample_count // 2 :]
        )
    )
    design = np.column_stack(
        (np.ones(sample_count), np.cos(2.0 * theta), np.sin(2.0 * theta))
    )
    coefficients, *_ = np.linalg.lstsq(design, acceleration_x, rcond=None)
    residual = acceleration_x - design @ coefficients

    spectrum = np.fft.rfft(acceleration_x - acceleration_x.mean()) / sample_count
    even_modes = np.arange(4, sample_count // 2 + 1, 2)
    high_even_rms = np.sqrt(np.sum(2.0 * np.abs(spectrum[even_modes]) ** 2))
    fundamental_rms = np.hypot(coefficients[1], coefficients[2]) / np.sqrt(2.0)

    print("opposed two-mass rotor (target=(3.2, 1.1, 0.8) m)")
    print(f"  pi-period max error: {half_period_error:.6e}")
    print(
        "  constant/cos(2theta)/sin(2theta) coefficients: "
        f"{coefficients[0]:+.6e}, {coefficients[1]:+.6e}, {coefficients[2]:+.6e}"
    )
    print(f"  2theta-fit residual RMS: {np.sqrt(np.mean(residual**2)):.6e}")
    print(
        f"  high even-harmonic RMS (4theta and above): {high_even_rms:.6e} "
        f"(relative to 2theta RMS {fundamental_rms:.6e})"
    )


if __name__ == "__main__":
    point_flyby()
    opposed_mass_rotor()
