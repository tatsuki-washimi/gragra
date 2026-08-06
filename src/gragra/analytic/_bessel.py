"""Internal high-precision spherical Bessel functions and cavity wall geometric factors.

Physics conventions and formulation notes:
- G-free convention: the spherical Bessel functions and geometric factors
  in this module are dimensionless. The caller does not multiply them by
  G_SI (6.6743e-11).
- Complex-output disclaimer: this module assumes real-valued input, so its
  output is real (float64).
- Formulation distinction: Newtonian gravity for a spherical cavity admits
  two formulations -- a "pure bulk" formulation (which keeps the density
  perturbation and the surface sheet as separate terms) and a
  "dipole/total" formulation (which obtains the total directly from a
  dipole expansion). These geometric factors exist so that the two can be
  evaluated separately.
"""

import numpy as np

# Threshold _X0 = 0.1.
# Rationale: in the direct expression for j1, cancellation gives a relative
# error of ~3*eps/x^2 (~7e-14 at x = 0.1), while the series (through the
# x^7 term) has a truncation error of ~x^8/1330560 (~8e-15 at x = 0.1). At
# X0 = 0.1 both branches meet the target accuracy of < 1e-13, so they cross
# over continuously.
_X0 = 0.1


def spherical_j0(x: float | list | tuple | np.ndarray) -> np.ndarray:
    """Calculate the spherical Bessel function of the first kind of order 0, j0(x).

    This function uses a high-precision series expansion for small arguments |x| < _X0
    to avoid division-by-zero and loss of precision (catastrophic cancellation).

    G-free convention:
        Input and output are dimensionless.
    Complex-output disclaimer:
        The output is a real-valued array.
    Formulation distinction:
        Used, among other things, in computing the pure-bulk term.

    Parameters
    ----------
    x : float or array-like
        The input coordinates.

    Returns
    -------
    np.ndarray
        The values of spherical Bessel function j0(x).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    original_shape = x_arr.shape
    x_flat = x_arr.ravel()

    abs_x = np.abs(x_flat)
    mask_series = abs_x < _X0
    out_flat = np.empty_like(x_flat)

    # Series expansion for j0(x): 1 - x^2/6 + x^4/120 - x^6/5040
    if np.any(mask_series):
        xs = x_flat[mask_series]
        x2 = xs**2
        out_flat[mask_series] = 1.0 - x2 / 6.0 + (x2**2) / 120.0 - (x2**3) / 5040.0

    # Direct calculation: sin(x) / x
    mask_direct = ~mask_series
    if np.any(mask_direct):
        xd = x_flat[mask_direct]
        out_flat[mask_direct] = np.sin(xd) / xd

    return out_flat.reshape(original_shape)


def spherical_j1(x: float | list | tuple | np.ndarray) -> np.ndarray:
    """Calculate the spherical Bessel function of the first kind of order 1, j1(x).

    This function uses a high-precision series expansion for small arguments |x| < _X0
    to avoid division-by-zero and loss of precision.

    G-free convention:
        Input and output are dimensionless.
    Complex-output disclaimer:
        The output is a real-valued array.
    Formulation distinction:
        Used in computing the total term and the spherical-shell dipole
        term.

    Parameters
    ----------
    x : float or array-like
        The input coordinates.

    Returns
    -------
    np.ndarray
        The values of spherical Bessel function j1(x).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    original_shape = x_arr.shape
    x_flat = x_arr.ravel()

    abs_x = np.abs(x_flat)
    mask_series = abs_x < _X0
    out_flat = np.empty_like(x_flat)

    # Series expansion for j1(x): x/3 - x^3/30 + x^5/840 - x^7/45360
    if np.any(mask_series):
        xs = x_flat[mask_series]
        x2 = xs**2
        out_flat[mask_series] = xs * (
            1.0 / 3.0 - x2 / 30.0 + (x2**2) / 840.0 - (x2**3) / 45360.0
        )

    # Direct calculation: sin(x)/x^2 - cos(x)/x
    mask_direct = ~mask_series
    if np.any(mask_direct):
        xd = x_flat[mask_direct]
        out_flat[mask_direct] = np.sin(xd) / (xd**2) - np.cos(xd) / xd

    return out_flat.reshape(original_shape)


def cavity_wall_geometric(x: float | list | tuple | np.ndarray) -> np.ndarray:
    """Calculate the geometric factor for the cavity wall Newtonian gravity.

    Geometric factor is defined as j0(x) - 2 * j1(x) / x.
    This function uses a high-precision series expansion for small arguments |x| < _X0
    to avoid catastrophic cancellation.

    G-free convention:
        Input and output are dimensionless.
    Complex-output disclaimer:
        The output is a real-valued array.
    Formulation distinction:
        Used as the geometric factor of the wall term in the "pure bulk"
        formulation.

    Parameters
    ----------
    x : float or array-like
        The dimensionless argument (wavenumber * radius).

    Returns
    -------
    np.ndarray
        The geometric factor j0(x) - 2 * j1(x) / x.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    original_shape = x_arr.shape
    x_flat = x_arr.ravel()

    abs_x = np.abs(x_flat)
    mask_series = abs_x < _X0
    out_flat = np.empty_like(x_flat)

    # Series expansion for cavity wall geometric factor:
    # 1/3 - x^2/10 + x^4/168 - x^6/6480 + x^8/443520
    if np.any(mask_series):
        xs = x_flat[mask_series]
        x2 = xs**2
        out_flat[mask_series] = (
            1.0 / 3.0
            - x2 / 10.0
            + (x2**2) / 168.0
            - (x2**3) / 6480.0
            + (x2**4) / 443520.0
        )

    # Direct calculation: j0(x) - 2 * j1(x) / x
    mask_direct = ~mask_series
    if np.any(mask_direct):
        xd = x_flat[mask_direct]
        j0_d = np.sin(xd) / xd
        j1_d = np.sin(xd) / (xd**2) - np.cos(xd) / xd
        out_flat[mask_direct] = j0_d - 2.0 * j1_d / xd

    return out_flat.reshape(original_shape)


def _j1_over_x(x: float | list | tuple | np.ndarray) -> np.ndarray:
    """Calculate high-precision j1(x)/x.

    G-free convention:
        Input and output are dimensionless.
    Complex-output disclaimer:
        The output is a real-valued array.
    Formulation distinction:
        Used in computing the P-wave total term of the dipole/total
        formulation and the spherical-shell dipole factor.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    original_shape = x_arr.shape
    x_flat = x_arr.ravel()

    abs_x = np.abs(x_flat)
    mask_series = abs_x < _X0
    out_flat = np.empty_like(x_flat)

    # Series expansion for j1(x)/x: 1/3 - x^2/30 + x^4/840 - x^6/45360
    if np.any(mask_series):
        xs = x_flat[mask_series]
        x2 = xs**2
        out_flat[mask_series] = (
            1.0 / 3.0 - x2 / 30.0 + (x2**2) / 840.0 - (x2**3) / 45360.0
        )

    # Direct calculation: sin(x)/x^3 - cos(x)/x^2
    mask_direct = ~mask_series
    if np.any(mask_direct):
        xd = x_flat[mask_direct]
        out_flat[mask_direct] = np.sin(xd) / (xd**3) - np.cos(xd) / (xd**2)

    return out_flat.reshape(original_shape)
