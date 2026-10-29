"""Amplitude-free Rayleigh half-space response model."""

from __future__ import annotations

from dataclasses import dataclass

from astropy import units as u

from gragra._rayleigh_contract import RayleighResponseEvaluation, RayleighResponseSpec
from gragra.geometry._frame import Frame
from gragra.geometry._halfspace import HalfSpace
from gragra.targets._high_points import TargetPoints


@dataclass(frozen=True, eq=False)
class _RayleighResponseModel:
    """Rayleigh medium and phase geometry without an input phasor."""

    spec: RayleighResponseSpec

    @property
    def geometry(self) -> HalfSpace:
        return self.spec.geometry

    def at(
        self,
        targets: TargetPoints,
        *,
        output_frame: Frame | None = None,
    ):
        """Resolve target positions and output components for this response."""
        if not isinstance(targets, TargetPoints):
            raise TypeError("targets must be TargetPoints")
        from gragra._rayleigh_response_setup import RayleighResponseSetup

        resolved = RayleighResponseEvaluation(
            self,
            targets,
            targets.frame if output_frame is None else output_frame,
        )
        return RayleighResponseSetup(resolved)


def rayleigh_response(
    geometry: HalfSpace,
    *,
    density: u.Quantity,
    vp: u.Quantity,
    vs: u.Quantity,
    frequencies: u.Quantity,
    direction: object,
    phase_origin: u.Quantity,
) -> _RayleighResponseModel:
    """Create a setup for reusable Rayleigh-to-gravity response coefficients.

    Parameters
    ----------
    geometry
        The elastic half-space and its surface frame.
    density
        Half-space density, with units convertible to kg/m³.
    vp
        Compressional-wave speed, with units convertible to m/s.
    vs
        Shear-wave speed, with units convertible to m/s.
    frequencies
        Positive, strictly increasing frequency bins, with units convertible
        to Hz.
    direction
        Nonzero horizontal propagation direction in the surface frame; it is
        normalized internally.
    phase_origin
        Surface location where the complex displacement phasor is defined,
        with units convertible to m.

    Returns
    -------
    Rayleigh response setup
        A deferred model description. Call ``at(targets).run()`` to calculate
        a ``GravityResponse`` with coefficients for potential, acceleration,
        and gradient.

    Notes
    -----
    This setup contains no displacement amplitude or reference time. The
    response uses a one-metre basis displacement to calculate coefficients.
    Applying a vertical-displacement frequency series reuses those
    coefficients. The resulting values are complex phasor quantities, not
    directly real-valued time-domain gravitational fields.
    """
    return _RayleighResponseModel(
        RayleighResponseSpec(
            geometry=geometry,
            density=density,
            vp=vp,
            vs=vs,
            frequencies=frequencies,
            phase_origin=phase_origin,
            direction=direction,
        )
    )


__all__ = ["rayleigh_response"]
