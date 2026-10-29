"""Example 05: Gravity gradient tensor – sign convention and tr(T) behaviour.

Demonstrates the canonical sign test and the softening-dependent trace:

* Source at the origin, target at (1, 0, 0) m with softening_m=0.
  The tensor must be proportional to G_SI * diag(2, -1, -1)
  (standard point-source gravity gradient formula for the x-axis).

* tr(T) ≈ 0 holds **only** for softening_m = 0 (Laplace equation in vacuum).
  With softening_m > 0 the Plummer-softened kernel is not strictly harmonic,
  so tr(T) ≠ 0.  This example prints both cases to make that explicit.
"""

import numpy as np

from gragra import G_SI, PointMass, point_gravity_gradient


def main() -> None:
    # Source: 1 kg at the origin
    source = PointMass(mass_kg=1.0, position_m=[0.0, 0.0, 0.0])
    target = [[1.0, 0.0, 0.0]]  # (1, 3) target

    # ------------------------------------------------------------------
    # Case 1: No softening (softening_m = 0)
    # Expected: G_SI * diag(2, -1, -1)
    # ------------------------------------------------------------------
    grad_exact = point_gravity_gradient(source, target, softening_m=0.0)

    expected_diag = G_SI * np.array([2.0, -1.0, -1.0])
    print("=== Case 1: softening_m = 0 ===")
    print(f"  Gradient tensor T[0] =\n{grad_exact[0]}")
    print(f"\n  Diagonal             : {np.diag(grad_exact[0])}")
    print(f"  Expected diagonal    : {expected_diag}")
    print(f"  tr(T)                : {np.trace(grad_exact[0]):.3e}  (should be ~0)")

    # Diagonal sign assertions
    diag = np.diag(grad_exact[0])
    assert diag[0] > 0, f"T[0,0] should be positive; got {diag[0]}"
    assert diag[1] < 0, f"T[1,1] should be negative; got {diag[1]}"
    assert diag[2] < 0, f"T[2,2] should be negative; got {diag[2]}"
    assert np.isclose(np.trace(grad_exact[0]), 0.0, atol=1e-20), (
        f"tr(T) should be ~0 for softening=0; got {np.trace(grad_exact[0])}"
    )
    print("  Diagonal sign check: PASSED")

    # ------------------------------------------------------------------
    # Case 2: Non-zero softening (softening_m = 0.5 m)
    # tr(T) is NOT zero because the softened kernel is approximate.
    # ------------------------------------------------------------------
    eps = 0.5  # softening in metres
    grad_soft = point_gravity_gradient(source, target, softening_m=eps)

    print(f"\n=== Case 2: softening_m = {eps} m (Plummer-softened) ===")
    print(f"  Gradient tensor T[0] =\n{grad_soft[0]}")
    print(f"  tr(T)                : {np.trace(grad_soft[0]):.6e}")
    print(
        "  tr(T) ≠ 0 with softening > 0: the softened kernel is NOT strictly harmonic."
    )

    trace_soft = np.trace(grad_soft[0])
    assert not np.isclose(trace_soft, 0.0, atol=1e-30), (
        "Expected tr(T) != 0 for softened kernel"
    )
    print("  Softened trace check: PASSED (trace is non-zero as expected)")


if __name__ == "__main__":
    main()
