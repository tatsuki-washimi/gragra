"""Optional high precision scalar reference calculations.

The public symbols are importable without the optional numerical dependency;
``mpmath`` is resolved only when a calculation is requested.
"""

from ._cavern import centered_cavern_volume_kernel
from ._types import (
    ReferenceConvergenceError,
    ReferenceErrorComponents,
    ReferenceEvaluation,
    ReferenceResourceError,
    ReferenceScalar,
)

__all__ = [
    "ReferenceConvergenceError",
    "ReferenceErrorComponents",
    "ReferenceEvaluation",
    "ReferenceResourceError",
    "ReferenceScalar",
    "centered_cavern_volume_kernel",
]
