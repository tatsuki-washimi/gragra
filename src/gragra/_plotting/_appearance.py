"""Resolve optional geometry appearance fields for existing plot artists."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from gragra.geometry import Appearance


def appearance_fields(
    appearance: Appearance | None,
    geometry: object,
    *,
    color: str,
    alpha: float,
    label: str,
) -> tuple[str, float, str, bool]:
    """Return call-over-geometry-over-default fields and whether color was set."""
    from gragra.geometry import Appearance as _Appearance

    if appearance is not None and not isinstance(appearance, _Appearance):
        raise TypeError("appearance must be an Appearance or None")
    geometry_appearance = getattr(geometry, "appearance", None)

    def field(name: str, default: object) -> object:
        value = None if appearance is None else getattr(appearance, name)
        if value is None and geometry_appearance is not None:
            value = getattr(geometry_appearance, name)
        return default if value is None else value

    return (
        cast(str, field("color", color)),
        cast(float, field("alpha", alpha)),
        cast(str, field("label", label)),
        (appearance is not None and appearance.color is not None)
        or (geometry_appearance is not None and geometry_appearance.color is not None),
    )
