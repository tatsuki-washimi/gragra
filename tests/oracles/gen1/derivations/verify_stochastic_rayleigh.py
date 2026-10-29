"""Independent directional-average check for Gen1 stochastic Rayleigh gravity.

This script does not import gragra production code.  It verifies the
single-target, above-surface acceleration CSD tensor implied by the
one-direction Rayleigh gravity transfer after factoring out the common scalar
coefficient

    A_R = 2*pi*G*rho0*gamma(nu)*exp(-h*k_R).

For propagation azimuth phi, the normalized acceleration transfer from vertical
surface displacement is

    v(phi) = (-1j*cos(phi), -1j*sin(phi), 1).

The isotropic CSD factor is <v v^H>_phi.
"""

from __future__ import annotations

import numpy as np


EXPECTED = np.diag([0.5, 0.5, 1.0]).astype(np.complex128)


def rayleigh_normalized_transfer(azimuth_rad: np.ndarray) -> np.ndarray:
    """Return v(phi) with shape (..., 3)."""
    phi = np.asarray(azimuth_rad, dtype=np.float64)
    return np.stack(
        (
            -1j * np.cos(phi),
            -1j * np.sin(phi),
            np.ones_like(phi, dtype=np.complex128),
        ),
        axis=-1,
    )


def discrete_isotropic_average(count: int = 4096) -> np.ndarray:
    """Uniform periodic quadrature of <v v^H> over 0 <= phi < 2*pi."""
    if count < 4:
        raise ValueError("count must be at least 4")
    phi = 2.0 * np.pi * np.arange(count, dtype=np.float64) / count
    v = rayleigh_normalized_transfer(phi)
    return np.einsum("ni,nj->ij", v, np.conjugate(v)) / count


def main() -> None:
    got = discrete_isotropic_average()
    np.testing.assert_allclose(got, EXPECTED, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(got, np.conjugate(got.T), rtol=0.0, atol=0.0)

    # Explicit symmetry/null checks make accidental phase/sign changes visible.
    np.testing.assert_allclose(got[0, 0], 0.5, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(got[1, 1], 0.5, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(got[2, 2], 1.0, rtol=0.0, atol=2.0e-14)
    offdiag = got - np.diag(np.diag(got))
    np.testing.assert_allclose(offdiag, 0.0, rtol=0.0, atol=2.0e-14)

    print("stochastic Rayleigh directional average: PASS")
    print(got)


if __name__ == "__main__":
    main()
