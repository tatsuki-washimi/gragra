"""High-level Rayleigh half-space phasor model."""

from __future__ import annotations

from dataclasses import dataclass

from astropy import units as u
from gwexpy.frequencyseries import FrequencySeries

from gragra._rayleigh_contract import RayleighModelSpec, RayleighResolvedEvaluation
from gragra.geometry._frame import Frame
from gragra.geometry._halfspace import HalfSpace
from gragra.results._spectral import PhasorRepresentation
from gragra.targets._high_points import TargetPoints


@dataclass(frozen=True, eq=False)
class _RayleighModel:
    """Rayleigh half-space and its owned phasor inputs."""

    spec: RayleighModelSpec

    @property
    def geometry(self) -> HalfSpace:
        return self.spec.geometry

    def at(
        self,
        targets: TargetPoints,
        *,
        output_frame: Frame | None = None,
    ):
        """Resolve target positions and output component frame for evaluation."""
        if not isinstance(targets, TargetPoints):
            raise TypeError("targets must be TargetPoints")
        from gragra._rayleigh_evaluation_setup import RayleighEvaluationSetup

        resolved = RayleighResolvedEvaluation(
            self,
            targets,
            targets.frame if output_frame is None else output_frame,
        )
        return RayleighEvaluationSetup(resolved)


def rayleigh_halfspace(
    geometry: HalfSpace,
    density: u.Quantity,
    vp: u.Quantity,
    vs: u.Quantity,
    vertical_displacement: FrequencySeries,
    representation: PhasorRepresentation,
    reference_time: object,
    phase_origin: u.Quantity,
    direction: object,
) -> _RayleighModel:
    """Create a Rayleigh half-space model from an explicit phasor contract."""
    return _RayleighModel(
        RayleighModelSpec(
            geometry=geometry,
            density=density,
            vp=vp,
            vs=vs,
            vertical_displacement=vertical_displacement,
            representation=representation,
            reference_time=reference_time,
            phase_origin=phase_origin,
            direction=direction,
        )
    )


__all__ = ["rayleigh_halfspace"]
