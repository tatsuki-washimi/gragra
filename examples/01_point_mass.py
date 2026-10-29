"""Example 01: Point mass gravitational acceleration.

Demonstrates the simplest use of gragra: compute the gravitational acceleration
produced by a 1 kg point mass at the origin, evaluated at the target (1, 0, 0) m.

Expected result:
- Acceleration points toward the origin: negative x-component.
- Magnitude equals G_SI (since mass = 1 kg, distance = 1 m).
"""

import numpy as np

from gragra import G_SI, PointMass, point_acceleration


def main() -> None:
    # Create a 1 kg point mass at the origin
    source = PointMass(mass_kg=1.0, position_m=[0.0, 0.0, 0.0])

    # Evaluate acceleration at target position (1, 0, 0) m
    # array-like [x, y, z] is promoted to shape (1, 3) internally
    target = [1.0, 0.0, 0.0]
    acc = point_acceleration(source, target)

    print(f"G_SI = {G_SI:.5e}  m^3 kg^-1 s^-2")
    print(f"Acceleration shape : {acc.shape}")  # expected: (1, 3)
    print(f"Acceleration vector: {acc[0]}")  # expected: [-G_SI, 0, 0]
    print(f"Magnitude          : {np.linalg.norm(acc[0]):.5e}  m s^-2")

    # Light sanity checks (should never fail for a correct implementation)
    assert acc.shape == (1, 3), f"Unexpected shape: {acc.shape}"
    assert np.isclose(np.linalg.norm(acc[0]), G_SI), (
        f"Magnitude mismatch: {np.linalg.norm(acc[0])} != {G_SI}"
    )
    # Acceleration must point toward the origin (negative x direction)
    assert acc[0, 0] < 0, "x-component should be negative (pointing toward origin)"

    print("All assertions passed.")


if __name__ == "__main__":
    main()
