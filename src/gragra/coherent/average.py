"""Directional power average and cross average utilities."""

import numpy as np


def _validate_direction_weights(
    weights, n_directions: int, *, allow_negative: bool
) -> np.ndarray:
    """Validate direction weights for power and cross average functions."""
    if isinstance(weights, bool):
        raise TypeError("direction_weights must not be bool")
    w = np.asarray(weights)
    if w.dtype.kind == "b":
        raise TypeError("direction_weights must not be bool")

    if np.iscomplexobj(w) or w.dtype.kind in ("c", "C"):
        raise TypeError("direction_weights must be real-valued")

    try:
        w = np.asarray(weights, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise TypeError(f"direction_weights must be numeric: {e}") from e

    if w.ndim != 1:
        raise ValueError(f"direction_weights must be a 1D array, got ndim {w.ndim}")

    if len(w) != n_directions:
        raise ValueError(
            "Length of direction_weights must match response shape along "
            f"direction_axis: {len(w)} != {n_directions}"
        )

    if not np.isfinite(w).all():
        raise ValueError("direction_weights must contain only finite numbers")

    if not allow_negative and (w < 0.0).any():
        raise ValueError("direction_weights must be non-negative")

    return w


def directional_power_average(
    response: list | tuple | np.ndarray,
    direction_weights: list | tuple | np.ndarray,
    direction_axis: int = 1,
) -> np.ndarray:
    """Calculate the directional power average of a response.

    Formula: sum(w_d * |response|^2) along direction_axis.
    """
    resp = np.asarray(response)
    # Promote to at least float64 so integer inputs (e.g. int64 with values
    # around 1e10) do not silently overflow when squared -- np.abs(int_arr)
    # preserves the integer dtype, and int64**2 wraps rather than raising.
    resp = resp.astype(np.result_type(resp.dtype, np.float64), copy=False)

    if direction_axis < 0 or direction_axis >= resp.ndim:
        raise ValueError(
            f"direction_axis {direction_axis} is out of bounds for "
            f"response with ndim {resp.ndim}"
        )

    w = _validate_direction_weights(
        direction_weights, resp.shape[direction_axis], allow_negative=False
    )

    # Expand w to match response ndim
    w_shape = [1] * resp.ndim
    w_shape[direction_axis] = len(w)
    w_expanded = w.reshape(w_shape)

    power = np.abs(resp) ** 2
    res = np.sum(power * w_expanded, axis=direction_axis)

    res = np.asarray(res, dtype=np.float64)
    res = res.copy()
    res.setflags(write=False)
    return res


def directional_cross_average(
    response_a,
    response_b,
    direction_weights,
    direction_axis: int = 1,
) -> np.ndarray:
    """Calculate the directional cross average between two responses.

    Formula: sum(w_d * A_d * conj(B_d)) along direction_axis,
    where the multiplication is element-wise, not a dot product.

    This implements the second-order cross statistics convention.
    Reference:
    - coherent_statistics_conventions.md §1/§2/§4

    Parameters
    ----------
    response_a : array_like
        First response array (A).
    response_b : array_like
        Second response array (B). Must have the exact same shape as response_a.
    direction_weights : array_like
        Direction weights (w). Must be real-valued. Negative weights are allowed.
        It is the user's responsibility to normalize the weights; this function
        performs a raw weighted sum without dividing by the sum of weights.
    direction_axis : int, optional
        Axis along which the direction average is computed. Default is 1.

    Returns
    -------
    numpy.ndarray
        Cross average array of dtype complex128. It is read-only.
        The axis corresponding to `direction_axis` is removed.

    Notes
    -----
    - Conjugation is applied to the second argument (response_b).
    - Under a directional ensemble model where fields from different directions
      are uncorrelated, this average corresponds to the ensemble expectation ⟨A·B*⟩.
    - Since negative weights are allowed, cross(A, A, w) may yield a negative value,
      and the Hermiticity/positive semi-definiteness (PSD) of the cross tensor
      is not guaranteed.
    - Complex outputs are second-order cross statistics (cross-spectral quantities
      such as ⟨A·B*⟩), not time-domain fields; the phase encodes the relative
      propagation delay or coherence between the two inputs, not an absolute
      field phase.
    """
    resp_a = np.asarray(response_a)
    resp_b = np.asarray(response_b)
    # Promote to at least float64 (preserving complex128 if either input is
    # complex) so integer inputs do not silently overflow in resp_a * conj(resp_b)
    # -- e.g. int64 values around 1e10 wrap instead of raising before the
    # final complex128 cast below.
    common_dtype = np.result_type(resp_a.dtype, resp_b.dtype, np.float64)
    resp_a = resp_a.astype(common_dtype, copy=False)
    resp_b = resp_b.astype(common_dtype, copy=False)

    if resp_a.shape != resp_b.shape:
        raise ValueError(
            "Shapes of response_a and response_b must match: "
            f"{resp_a.shape} != {resp_b.shape}"
        )

    if direction_axis < 0 or direction_axis >= resp_a.ndim:
        raise ValueError(
            f"direction_axis {direction_axis} is out of bounds for "
            f"response with ndim {resp_a.ndim}"
        )

    w = _validate_direction_weights(
        direction_weights, resp_a.shape[direction_axis], allow_negative=True
    )

    # Expand w to match response ndim
    w_shape = [1] * resp_a.ndim
    w_shape[direction_axis] = len(w)
    w_expanded = w.reshape(w_shape)

    cross = resp_a * np.conj(resp_b)
    res = np.sum(cross * w_expanded, axis=direction_axis)

    res = np.asarray(res, dtype=np.complex128)
    res = res.copy()
    res.setflags(write=False)
    return res
