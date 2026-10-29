"""Gravitational observation targets."""

from typing import TYPE_CHECKING

__all__ = [
    "TargetPoints",
]

if TYPE_CHECKING:
    from gragra.targets._high_points import TargetPoints


def __getattr__(name: str):
    """Load the unit-aware target container on explicit access."""
    if name == "TargetPoints":
        from gragra.targets._high_points import TargetPoints

        return TargetPoints
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
