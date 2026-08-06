import numpy as np

from gragra._arrays import _check_no_bool
from gragra.coherent.average import directional_cross_average


def ensemble_outer_cross_average(
    realizations_a,
    realizations_b,
) -> np.ndarray:
    """Calculate the ensemble outer cross average over realizations.

    Formula: c_ij(x) = (1/R) sum_{r=1}^R A_i(x, r) * conj(B_j(x, r))
    where the component axes form an outer product (3, 3) tensor.

    Parameters
    ----------
    realizations_a : array_like
        First realization field array (A) of shape (N, R, 3) and numeric type.
    realizations_b : array_like
        Second realization field array (B) of shape (N, R, 3) and numeric type.
        N and R axes must be identical in shape to realizations_a.

    Returns
    -------
    numpy.ndarray
        Outer cross average array of shape (N, 3, 3) and dtype complex128.
        It is read-only.

    Raises
    ------
    TypeError
        If inputs cannot be converted to complex128 or contain booleans.
    ValueError
        If shapes are invalid, non-finite values are encountered, R = 0,
        or shape of N and R axes mismatch.

    Notes
    -----
    - Conjugation is applied to the second argument (realizations_b).
    - The output represents second-order cross statistics (cross-spectral quantities
      such as ⟨A·B*⟩), not time-domain fields; the phase encodes the relative
      propagation delay or coherence between the two inputs, not an absolute
      field phase.
    - This function computes the equal-weight average over realizations (w = 1/R),
      which has a different statistical meaning from the direction-uncorrelated
      models (which allow arbitrary real and possibly negative weights).
    - Although the underlying `directional_cross_average` performs a raw weighted
      sum without dividing by the sum of weights, w = 1/R makes the result an
      ensemble average.
    - The component axes are treated as an outer product (yielding a (3, 3) tensor),
      unlike the element-wise component product used in directional_cross_average.
    - This function is G-agnostic: if the inputs are G-free (like witness
      displacements), the output is G⁰. If the inputs include G, the output
      preserves the product of G orders.
      No additional gravitational constant G_SI is multiplied internally.
    """
    _check_no_bool(realizations_a, "realizations_a")
    _check_no_bool(realizations_b, "realizations_b")

    try:
        a = np.asarray(realizations_a)
        b = np.asarray(realizations_b)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Inputs must be convertible to NumPy arrays: {e}") from e

    if a.dtype.kind == "b" or b.dtype.kind == "b":
        raise TypeError("Inputs must not contain bool")

    if a.ndim != 3:
        raise ValueError(f"realizations_a must have rank 3, got shape {a.shape}")
    if b.ndim != 3:
        raise ValueError(f"realizations_b must have rank 3, got shape {b.shape}")

    if a.shape[2] != 3:
        raise ValueError(f"realizations_a component axis must be 3, got {a.shape[2]}")
    if b.shape[2] != 3:
        raise ValueError(f"realizations_b component axis must be 3, got {b.shape[2]}")

    if a.shape[0] != b.shape[0]:
        raise ValueError(f"N axis mismatch: {a.shape[0]} != {b.shape[0]}")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"realization axis mismatch: {a.shape[1]} != {b.shape[1]}")

    n_realizations = a.shape[1]
    if n_realizations == 0:
        raise ValueError("n_realizations (R) must be positive, got 0")

    try:
        a_comp = a.astype(np.complex128, copy=False)
        b_comp = b.astype(np.complex128, copy=False)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Inputs must be convertible to complex128: {e}") from e

    if not np.isfinite(a_comp).all() or not np.isfinite(b_comp).all():
        raise ValueError("Inputs must contain only finite numbers")

    # Insert singleton axis to form outer product
    a2 = a_comp[:, :, :, None]  # (N, R, 3, 1)
    b2 = b_comp[:, :, None, :]  # (N, R, 1, 3)

    a3, b3 = np.broadcast_arrays(a2, b2)  # (N, R, 3, 3)
    weights = np.full(n_realizations, 1.0 / n_realizations, dtype=np.float64)

    res = directional_cross_average(a3, b3, weights, direction_axis=1)

    # Ensure output is complex128 and read-only
    res = np.asarray(res, dtype=np.complex128).copy()
    res.setflags(write=False)
    return res
