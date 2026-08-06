import numpy as np

from gragra.array_types import FloatArray
from gragra.waves.directions import normalize_directions


def lebedev_grid(order: int) -> tuple[FloatArray, FloatArray]:
    """Generate a Lebedev spherical-quadrature grid (directions + normalized weights).

    Thin wrapper over ``scipy.integrate.lebedev_rule`` (scipy >= 1.15). Returns
    directions and weights in exactly the form ``DirectionalComplexKernel`` /
    ``volume_transfer_kernel`` expect, so the result is drop-in for the next-8
    coherent-combination adapter. The Lebedev *integrators* (ObservationGeometry,
    VolumeIntegrator, surface accumulation) remain a nn-sus-side concern (Stays).

    Some supported orders (13/25/27) carry negative quadrature weights; the
    returned wnorm can contain negatives (sum is still 1).
    """
    if isinstance(order, bool):
        raise ValueError("order must be a positive integer, got bool")
    if not isinstance(order, (int, np.integer)):
        raise ValueError(
            f"order must be a positive integer, got type {type(order).__name__}"
        )
    if order <= 0:
        raise ValueError(f"order must be positive, got {order}")

    try:
        from scipy.integrate import lebedev_rule
    except (ImportError, AttributeError) as e:
        raise ImportError(
            "Lebedev grid generation requires scipy>=1.15; install gragra[lebedev]"
        ) from e

    try:
        dirs_raw, w_raw = lebedev_rule(order)
    except (ValueError, NotImplementedError) as e:
        valid_orders = (
            "[3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35, "
            "41, 47, 53, 59, 65, 71, 77, 83, 89, 95, 101, 107, 113, 119, 125, 131]"
        )
        raise ValueError(
            f"Unsupported Lebedev order: {order}. "
            f"Valid orders include: {valid_orders}. "
            f"Underlying error: {e}"
        ) from e

    khat = normalize_directions(dirs_raw.T)

    w_scaled = (w_raw / (4.0 * np.pi)).astype(np.float64)
    wnorm = w_scaled / w_scaled.sum()

    if not (np.isfinite(khat).all() and np.isfinite(wnorm).all()):
        raise ValueError("Generated grid contains non-finite values")

    khat = khat.copy()
    wnorm = wnorm.copy()
    khat.setflags(write=False)
    wnorm.setflags(write=False)

    return khat, wnorm
