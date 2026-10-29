"""Analytic gravity models for atmospheric perturbations."""

from typing import TYPE_CHECKING

__all__ = [
    "acoustic_halfspace",
    "isobaric_temperature",
    "acoustic_cavity_mode",
    "SphereAcousticMode",
    "CuboidAcousticMode",
    "CylinderAcousticMode",
    "AcousticCavityNumerics",
]

if TYPE_CHECKING:
    from gragra.analytic.atmospheric._acoustic import acoustic_halfspace
    from gragra.analytic.atmospheric._cavity._model import (
        AcousticCavityNumerics,
        CuboidAcousticMode,
        CylinderAcousticMode,
        SphereAcousticMode,
        acoustic_cavity_mode,
    )
    from gragra.analytic.atmospheric._temperature import isobaric_temperature


def __getattr__(name: str):
    """Load the high-level atmosphere factories on explicit access."""
    if name == "acoustic_halfspace":
        from gragra.analytic.atmospheric._acoustic import acoustic_halfspace

        return acoustic_halfspace
    if name == "isobaric_temperature":
        from gragra.analytic.atmospheric._temperature import isobaric_temperature

        return isobaric_temperature
    if name in {
        "acoustic_cavity_mode",
        "SphereAcousticMode",
        "CuboidAcousticMode",
        "CylinderAcousticMode",
        "AcousticCavityNumerics",
    }:
        from gragra.analytic.atmospheric._cavity._model import (
            AcousticCavityNumerics,
            CuboidAcousticMode,
            CylinderAcousticMode,
            SphereAcousticMode,
            acoustic_cavity_mode,
        )

        return {
            "acoustic_cavity_mode": acoustic_cavity_mode,
            "SphereAcousticMode": SphereAcousticMode,
            "CuboidAcousticMode": CuboidAcousticMode,
            "CylinderAcousticMode": CylinderAcousticMode,
            "AcousticCavityNumerics": AcousticCavityNumerics,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
