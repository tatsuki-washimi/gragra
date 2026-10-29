"""Example 02: WeightedPointSource with signed and complex weights.

Demonstrates that gragra supports generic signed and complex-valued weights:
- Real (float64) weights can be negative, representing mass underdensities.
- Complex (complex128) weights represent linear complex amplitudes or
  transfer-function-like responses (not directly real-valued time-domain fields).

The output dtype follows the weight dtype: real weights -> float64 output,
complex weights -> complex128 output.
"""

import numpy as np

from gragra import WeightedPointSource, point_acceleration, point_potential


def main() -> None:
    # -----------------------------------------------------------------------
    # Case 1: Signed (real) weights
    # Two point masses: +2 kg at (-1, 0, 0) and -1 kg at (+1, 0, 0).
    # Negative weight represents a mass underdensity (delta-mass element).
    # -----------------------------------------------------------------------
    source_signed = WeightedPointSource(
        positions_m=np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        weights_kg=np.array([2.0, -1.0]),  # float64 weights
    )

    target = np.array([[0.0, 0.0, 1.0]])  # observe at (0, 0, 1) m
    acc_signed = point_acceleration(source_signed, target)
    phi_signed = point_potential(source_signed, target)

    print("=== Case 1: Signed (real) weights ===")
    print(f"  weights dtype : {source_signed.weights_kg.dtype}")
    print(f"  potential     : {phi_signed[0]:.5e}  (output dtype: {phi_signed.dtype})")
    print(f"  acceleration  : {acc_signed[0]}  (output dtype: {acc_signed.dtype})")
    assert acc_signed.dtype == np.float64, "Real weights must yield float64 output"
    assert phi_signed.dtype == np.float64, "Real weights must yield float64 output"

    # -----------------------------------------------------------------------
    # Case 2: Complex weights
    # Complex output represents a linear complex amplitude, not a real-time
    # gravitational field.  The output dtype becomes complex128.
    # -----------------------------------------------------------------------
    source_complex = WeightedPointSource(
        positions_m=np.array([[0.0, 0.0, 0.0]]),
        weights_kg=np.array([1.0 + 0.5j]),  # complex128 weights
    )

    acc_complex = point_acceleration(source_complex, target)
    phi_complex = point_potential(source_complex, target)

    print("\n=== Case 2: Complex weights ===")
    print(f"  weights dtype : {source_complex.weights_kg.dtype}")
    print(
        f"  potential     : {phi_complex[0]:.5e}  (output dtype: {phi_complex.dtype})"
    )
    print(f"  acceleration  : {acc_complex[0]}  (output dtype: {acc_complex.dtype})")
    assert acc_complex.dtype == np.complex128, (
        "Complex weights must yield complex128 output"
    )
    assert phi_complex.dtype == np.complex128, (
        "Complex weights must yield complex128 output"
    )

    print("\nAll dtype assertions passed.")


if __name__ == "__main__":
    main()
