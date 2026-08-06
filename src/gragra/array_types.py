"""Type definitions and array conventions for gragra.

This module establishes the core numpy array types and axis-ordering
conventions used throughout the package to ensure physical and numerical
consistency, particularly when interfacing with other tools like gwexpy.
"""

import numpy as np
import numpy.typing as npt

# Float array type for positions, coordinates, and physical parameters
FloatArray = npt.NDArray[np.float64]

# Complex array type for complex amplitudes and phase-weighted quantities
ComplexArray = npt.NDArray[np.complex128]

"""
Axis-Ordering Conventions:
==========================

In gragra, we enforce a strict axis-ordering convention compatible with GWexpy:
1. Component / Tensor axes are ALWAYS placed at the very end of the array.
   - Vector fields end with a single axis of size 3. Shape: (..., 3)
     e.g., Target points positions (M, 3) or accelerations (M, 3).
   - Tensor fields end with two axes of size 3x3. Shape: (..., 3, 3)
     e.g., Gravity gradients (M, 3, 3).
2. Physical parameters (like frequency, directions) are inserted
   BEFORE the component axes.
   - e.g., (M, Nf, 3) for M target points, Nf frequencies, and 3 spatial components.
   - e.g., (M, Ndir, 3) for M target points, Ndir directions, and 3 spatial components.
   - e.g., (M, Nf, 3, 3) for M target points, Nf frequencies, and 3x3 tensor components.

This convention aligns directly with GWexpy's ScalarField (4D), VectorField, and
TensorField representations, allowing simple mapping via adapters.
"""
