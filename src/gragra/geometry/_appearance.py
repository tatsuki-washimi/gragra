"""Display-only metadata shared by pure geometry values."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real


@dataclass(frozen=True, eq=False)
class Appearance:
    """Optional color, opacity, and display label metadata.

    Values are retained for a drawing backend to interpret. They do not
    participate in shape dimensions, placement, or physics.
    """

    color: str | None = None
    alpha: float | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        if self.color is not None and not isinstance(self.color, str):
            raise TypeError("color must be a string or None")
        if self.label is not None and not isinstance(self.label, str):
            raise TypeError("label must be a string or None")
        if self.alpha is None:
            return
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, Real):
            raise TypeError("alpha must be a real number or None")
        alpha = float(self.alpha)
        if not isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be finite and between 0 and 1")
        object.__setattr__(self, "alpha", alpha)
