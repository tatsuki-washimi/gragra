"""High-level static uniform-sphere analytic model."""

from __future__ import annotations

from dataclasses import dataclass

from astropy import units as u

from gragra._evaluation_setup import (
    OBSERVABLES,
    ResolvedEvaluation,
    UniformSphereModelSpec,
)
from gragra.geometry._frame import Frame
from gragra.geometry._sphere import Sphere
from gragra.targets._high_points import TargetPoints


@dataclass(frozen=True, eq=False)
class UniformSphereModel:
    """Uniform sphere and its immutable physical assumptions."""

    spec: UniformSphereModelSpec

    @property
    def geometry(self) -> Sphere:
        return self.spec.geometry

    @property
    def density(self) -> u.Quantity:
        return self.spec.density

    def at(
        self,
        targets: TargetPoints,
        *,
        observables: tuple[str, ...] = OBSERVABLES,
        output_frame: Frame | None = None,
    ):
        """Resolve target positions, requested fields, and output components."""
        from gragra.evaluation import EvaluationSetup

        resolved = ResolvedEvaluation(
            self,
            targets,
            self.geometry.frame,
            targets.frame if output_frame is None else output_frame,
            observables,
            None,
        )
        return EvaluationSetup(resolved)


def uniform_sphere(geometry: Sphere, density: u.Quantity) -> UniformSphereModel:
    """Create an analytic static model of a real, finite-density sphere."""
    return UniformSphereModel(UniformSphereModelSpec(geometry, density))


__all__ = ["UniformSphereModel", "uniform_sphere"]
