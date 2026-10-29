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


def _j1_over_x_diff(
    x: float | list | tuple | np.ndarray,
    y: float | list | tuple | np.ndarray,
    x_minus_y: float | list | tuple | np.ndarray | None = None,
) -> np.ndarray:
    """Calculate ``_j1_over_x(x) - _j1_over_x(y)`` without catastrophic cancellation.

    G-free convention:
        Input and output are dimensionless.
    Complex-output disclaimer:
        The output is a real-valued array.
    Formulation distinction:
        Used for the finite-spherical-shell S-wave total term, where the
        naive ``_j1_over_x(x) - _j1_over_x(y)`` loses significant digits in
        two distinct regimes (small wavenumber and/or a thin shell R ~ a,
        i.e. ``x = k*a`` and ``y = k*R``):

        1. Both ``x`` and ``y`` small: both values round to
           ``1/3 - O(x^2)`` in float64, and the ``O(x^2)`` difference of
           interest is far below float64's resolution around the shared
           ``1/3`` leading term.
        2. ``x`` and ``y`` close to each other (a thin shell) but *not*
           both small: ``_j1_over_x``'s own direct-evaluation branch
           (``sin(x)/x^3 - cos(x)/x^2``) internally forms and cancels
           O(1/x^2) terms to produce an O(1) result, so it carries
           ~1e-13 relative error near the small-argument threshold; that
           error is negligible for ``_j1_over_x(x)`` alone, but once ``x``
           and ``y`` are close enough that their true difference is itself
           much smaller than either value, subtracting two independently
           ~1e-13-accurate evaluations amplifies that error relative to
           the (small) true difference.

    Case 1 (both small) is handled by evaluating the series difference
    directly in terms of ``x^2 - y^2 = (x-y)(x+y)`` (computed once, then
    reused via multiplication for the higher-order terms), which never
    forms the shared ``1/3`` term and so never cancels it away:

        f(x) - f(y) = -(x^2-y^2)/30 + (x^4-y^4)/840 - (x^6-y^6)/45360

    Case 2 (close but not both small) rewrites
    ``f(x) - f(y) = [sin(x)/x^3 - sin(y)/y^3] - [cos(x)/x^2 - cos(y)/y^2]``
    by anchoring each bracket at ``x`` and isolating the ``y``-only
    remainder, e.g. ``sin(x)/x^3 - sin(y)/y^3
    = sin(x)*(1/x^3 - 1/y^3) + [sin(x) - sin(y)]/y^3`` (and similarly for
    the cosine bracket), using the exact reciprocal-power differences
    ``1/x^2 - 1/y^2 = -(x-y)(x+y)/(x^2*y^2)`` and
    ``1/x^3 - 1/y^3 = -(x-y)(x^2+xy+y^2)/(x^3*y^3)``, and the sum-to-product
    identities ``sin(x)-sin(y) = 2*cos(m)*sin(h)``,
    ``cos(x)-cos(y) = -2*sin(m)*sin(h)`` (with midpoint ``m = (x+y)/2`` and
    half-difference ``h = (x-y)/2``) for the remainder terms. This never
    evaluates ``_j1_over_x(x)`` and ``_j1_over_x(y)`` independently and
    subtracts them; it only ever forms the (exact, Sterbenz-safe)
    difference ``x-y`` once and multiplies it through, and evaluates
    ``sin(h)`` directly at the (possibly tiny) half-difference, which is
    not subject to cancellation.

    Otherwise (``x``, ``y`` neither both small nor close to each other),
    this falls back to the direct difference of ``_j1_over_x``, which is
    not subject to either cancellation above.

    Both cases 1 and 2 use ``x - y`` (or the caller-supplied
    ``x_minus_y``, see below) rather than re-deriving it from ``x`` and
    ``y`` inside each branch, so a caller with access to a more accurately
    computed difference should pass it via ``x_minus_y``.

    Numerical note:
        Case 2's formula still evaluates trigonometric functions at ``x``,
        ``y``, and ``m`` directly and combines O(1/x^2)-O(1/x^3)-scale
        terms, so it is not exact to machine precision the way case 1 is;
        empirically (cross-checked against an 80-120 digit mpmath
        reference, across a battery spanning the "close"/"both small"
        boundary, moderate x, and x up to O(1000)), its worst-case relative
        error is a few times 1e-10 when combined with an accurate
        ``x_minus_y`` (worse, up to ~1e-9, if ``x_minus_y`` is omitted and
        the caller's own ``x``, ``y`` already carry upstream rounding
        error), versus up to 100% relative error (or an exact but wrong
        zero) for the naive difference in the same regime.

    Parameters
    ----------
    x : float or array-like
        First argument (dimensionless).
    y : float or array-like
        Second argument (dimensionless), broadcastable against ``x``.
    x_minus_y : float or array-like, optional
        A pre-computed, more accurate value of ``x - y`` (broadcastable
        against ``x``/``y``), used in place of forming ``x - y`` from the
        (already-rounded) ``x`` and ``y`` arrays themselves. This matters
        when ``x`` and ``y`` are each computed upstream as a product (e.g.
        ``x = k*a``, ``y = k*R``): rounding each product to float64
        independently can lose relative precision in their difference even
        though ``x - y`` looks "exact" by Sterbenz's lemma, because
        Sterbenz only guarantees the *subtraction itself* introduces no
        further error -- it says nothing about error already present in
        the rounded ``x``/``y`` inputs. Passing an accurately-computed
        difference directly (e.g. ``-(k*(R-a))``, i.e. ``k*(a-R)``,
        computed via a single subtraction ``R-a`` or ``a-R`` before
        multiplying by ``k``, rather than ``k*a - k*R``) avoids that
        upstream loss. If omitted, ``x - y`` is
        computed from ``x`` and ``y`` as before.

    Returns
    -------
    np.ndarray
        The values of ``j1(x)/x - j1(y)/y``, broadcast over ``x`` and ``y``.
    """
    x_arr, y_arr = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    )
    original_shape = x_arr.shape
    x_flat = x_arr.ravel()
    y_flat = y_arr.ravel()

    if x_minus_y is None:
        dx_flat = x_flat - y_flat
    else:
        dx_flat = np.broadcast_to(
            np.asarray(x_minus_y, dtype=np.float64), original_shape
        ).ravel()

    out_flat = np.empty_like(x_flat)

    mask_series = (np.abs(x_flat) < _X0) & (np.abs(y_flat) < _X0)
    if np.any(mask_series):
        xs = x_flat[mask_series]
        ys = y_flat[mask_series]
        dxs = dx_flat[mask_series]
        x2 = xs * xs
        y2 = ys * ys
        d2 = dxs * (xs + ys)
        d4 = d2 * (x2 + y2)
        d6 = d2 * (x2 * x2 + x2 * y2 + y2 * y2)
        out_flat[mask_series] = -d2 / 30.0 + d4 / 840.0 - d6 / 45360.0

    remaining = ~mask_series
    denom = np.maximum(np.abs(x_flat), np.abs(y_flat))
    denom = np.where(denom == 0.0, 1.0, denom)
    mask_close = remaining & (np.abs(dx_flat) < 0.25 * denom)
    if np.any(mask_close):
        xc = x_flat[mask_close]
        yc = y_flat[mask_close]
        dx = dx_flat[mask_close]
        m = 0.5 * (xc + yc)
        h = 0.5 * dx
        sm, cm = np.sin(m), np.cos(m)
        sh = np.sin(h)
        sx, cx = np.sin(xc), np.cos(xc)
        x2c, y2c = xc * xc, yc * yc
        term_dx = cx * (xc + yc) / (x2c * y2c) - sx * (x2c + xc * yc + y2c) / (
            x2c * xc * y2c * yc
        )
        term_h = 2.0 * sh * (cm / (y2c * yc) + sm / y2c)
        out_flat[mask_close] = dx * term_dx + term_h

    mask_direct = remaining & ~mask_close
    if np.any(mask_direct):
        out_flat[mask_direct] = _j1_over_x(x_flat[mask_direct]) - _j1_over_x(
            y_flat[mask_direct]
        )

    return out_flat.reshape(original_shape)
