"""Backend selection and its error contract.

Runs the same point-mass acceleration on the reference NumPy path and then on
each optional accelerated path. The point of the example is what happens when
an accelerated path is *not* available: gragra raises, and never quietly
returns a NumPy answer under the name of a backend you did not get.

Tolerances are explicit throughout. Every quantity here is scaled by
G_SI ~ 6.7e-11, which is below NumPy's default absolute tolerance of 1e-8, so
a comparison left at defaults would accept a sign flip and an array of zeros
alike. The fixed point is therefore checked exactly, the non-zero component of
a backend comparison relatively (atol=0.0), and the whole vector through a
residual normalized by the largest component rather than against zero.
"""

import numpy as np

from gragra import G_SI, PointMass, point_acceleration

source = PointMass(1.0, [0.0, 0.0, 0.0])
targets = [[1.0, 0.0, 0.0]]

# The reference implementation. One unit mass at the origin, one target at
# 1 m along +x: the acceleration points back at the source and its magnitude
# is exactly G_SI, with no rounding involved (r = r_eff = 1.0 exactly).
reference = point_acceleration(source, targets)
assert reference.shape == (1, 3)
assert reference[0, 0] == -G_SI
assert reference[0, 1] == 0.0 and reference[0, 2] == 0.0
print(f"numpy (reference): a_x = {reference[0, 0]:.6e} m/s^2")

# An unrecognised backend name is rejected before any work is done, and the
# message enumerates the four names that are accepted.
try:
    point_acceleration(source, targets, backend="gpu")
except ValueError as exc:
    assert "must be 'numpy', 'numba', 'cpp', or 'cupy'" in str(exc)
    print("unknown name: ValueError, as documented")
else:
    raise AssertionError("an unrecognised backend name must raise ValueError")

# Each optional path either produces the reference answer or refuses. There is
# no third outcome: nothing here silently degrades to NumPy.
#
# The exception classes below are the documented contract. `cupy` is the
# awkward one: its missing-dependency error is an instance of both ImportError
# and ValueError, and the class itself is defined inside the function that
# raises it, so it cannot be imported from gragra and must be caught through
# one of its base classes. `RuntimeError` covers cupy being installed with no
# usable CUDA device, which is only discovered at call time.
unavailable = (ImportError, RuntimeError)

for backend in ("numba", "cpp", "cupy"):
    try:
        result = point_acceleration(source, targets, backend=backend)
    except unavailable as exc:
        print(f"{backend}: unavailable -> {type(exc).__name__}, no fallback")
        continue

    # Non-zero component: relative comparison only, absolute tolerance pinned
    # to zero so that the default 1e-8 cannot swallow the whole quantity.
    assert np.allclose(result[:, 0], reference[:, 0], rtol=1e-12, atol=0.0)

    # Whole vector, including the two structurally zero components: compare a
    # residual normalized by the largest reference component. A bare rtol/atol
    # judgement against a zero target is ill-posed and is not used here.
    scale = np.max(np.abs(reference))
    residual = np.max(np.abs(result - reference)) / scale
    assert residual < 1e-12, (backend, residual)
    print(f"{backend}: agrees with numpy (normalized residual {residual:.1e})")
