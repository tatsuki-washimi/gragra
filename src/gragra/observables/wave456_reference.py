"""Physical observable wrappers for the independent Wave 4--6 references."""

from __future__ import annotations

from typing import Any

from gragra.analytic import wave456_reference as _reference
from gragra.analytic.wave456_reference import GFreeObservables
from gragra.constants import G_SI


def _scale(value: GFreeObservables) -> GFreeObservables:
    return GFreeObservables(
        G_SI * value.potential,
        G_SI * value.acceleration,
        G_SI * value.gradient,
    )


def point_mass_observables(*args: Any, **kwargs: Any) -> GFreeObservables:
    return _scale(_reference.point_mass_observables_g_free(*args, **kwargs))


def multipole_observables(*args: Any, **kwargs: Any) -> GFreeObservables:
    return _scale(_reference.multipole_observables_g_free(*args, **kwargs))


def solid_cylinder_observables(*args: Any, **kwargs: Any) -> GFreeObservables:
    return _scale(_reference.solid_cylinder_observables_g_free(*args, **kwargs))


def opposed_rotor_observables(*args: Any, **kwargs: Any) -> GFreeObservables:
    return _scale(_reference.opposed_rotor_observables_g_free(*args, **kwargs))


def opposed_rotor_leading_phasor(*args: Any, **kwargs: Any) -> GFreeObservables:
    return _scale(_reference.opposed_rotor_leading_phasor_g_free(*args, **kwargs))


def passing_cuboid_observables(*args: Any, **kwargs: Any) -> GFreeObservables:
    return _scale(_reference.passing_cuboid_observables_g_free(*args, **kwargs))


def integrate_box_observables(*args: Any, **kwargs: Any) -> GFreeObservables:
    return _scale(_reference.integrate_box_observables_g_free(*args, **kwargs))


def integrate_displacement_box_observables(
    *args: Any, **kwargs: Any
) -> GFreeObservables:
    return _scale(
        _reference.integrate_displacement_box_observables_g_free(*args, **kwargs)
    )


def adaptive_box_observables(*args: Any, **kwargs: Any):
    value, evidence = _reference.adaptive_box_observables_g_free(*args, **kwargs)
    return _scale(value), evidence


def adaptive_displacement_box_observables(*args: Any, **kwargs: Any):
    value, evidence = _reference.adaptive_displacement_box_observables_g_free(
        *args, **kwargs
    )
    return _scale(value), evidence


def host_inclusion_observables(*args: Any, **kwargs: Any):
    value, evidence = _reference.host_inclusion_observables_g_free(*args, **kwargs)
    return _scale(value), evidence


def acoustic_finite_box_observables(*args: Any, **kwargs: Any) -> GFreeObservables:
    return _scale(_reference.acoustic_finite_box_observables_g_free(*args, **kwargs))


def acoustic_infinite_halfspace_observables(
    *args: Any, **kwargs: Any
) -> GFreeObservables:
    return _scale(
        _reference.acoustic_infinite_halfspace_observables_g_free(*args, **kwargs)
    )


def rayleigh_halfspace_observables(*args: Any, **kwargs: Any) -> GFreeObservables:
    return _scale(_reference.rayleigh_halfspace_observables_g_free(*args, **kwargs))
