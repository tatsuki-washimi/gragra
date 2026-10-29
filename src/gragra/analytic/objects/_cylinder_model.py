"""High-level static uniform-cylinder analytic model."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from platform import python_version
from types import MappingProxyType

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import QTable

from gragra._evaluation_setup import density_kg_m3
from gragra._interactions import Newtonian
from gragra.analytic.cylinder_potential import cylinder_analytic_observables_g_free
from gragra.constants import G_SI
from gragra.geometry._cylinder import Cylinder
from gragra.geometry._frame import Frame
from gragra.results import (
    Contribution,
    EvaluationContext,
    EvaluationRecord,
    EvaluationResult,
    GravityFieldDefinition,
    GravityResult,
    PotentialReference,
    StaticRepresentation,
)
from gragra.results._record import OBSERVABLES
from gragra.targets._high_points import TargetPoints


@dataclass(frozen=True, eq=False, init=False)
class _UniformCylinderModelSpec:
    """Canonical immutable physical assumptions of a uniform cylinder."""

    geometry: Cylinder
    density_kg_m3: float
    field_definition: GravityFieldDefinition
    interaction: Newtonian

    def __init__(self, geometry: Cylinder, density: u.Quantity) -> None:
        if not isinstance(geometry, Cylinder):
            raise TypeError("geometry must be a high-level Cylinder")
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "density_kg_m3", density_kg_m3(density))
        object.__setattr__(
            self,
            "field_definition",
            GravityFieldDefinition.source_field(
                "uniform_cylinder",
                Contribution("total"),
                PotentialReference.ZERO_AT_INFINITY,
            ),
        )
        object.__setattr__(self, "interaction", Newtonian())

    @property
    def density(self) -> u.Quantity:
        return self.density_kg_m3 * u.kg / u.m**3

    @property
    def continuous_mass(self) -> u.Quantity:
        return (
            np.pi
            * self.geometry.radius_m**2
            * self.geometry.height_m
            * self.density_kg_m3
            * u.kg
        )


@dataclass(frozen=True)
class _CylinderValidationReport:
    valid: bool
    observables: tuple[str, ...]
    target_count: int


@dataclass(frozen=True, eq=False)
class _UniformCylinderEvaluation:
    """One fixed request for an analytic cylinder field."""

    model: UniformCylinderModel
    targets: TargetPoints
    output_frame: Frame
    observables: tuple[str, ...]
    context: EvaluationContext
    _target_positions_bytes: bytes

    def __init__(
        self,
        model: UniformCylinderModel,
        targets: TargetPoints,
        output_frame: Frame,
        observables: tuple[str, ...],
    ) -> None:
        if not isinstance(targets, TargetPoints):
            raise TypeError("targets must be high-level TargetPoints")
        if not isinstance(output_frame, Frame):
            raise TypeError("output_frame must be a Frame")
        names = tuple(observables)
        if (
            not names
            or len(set(names)) != len(names)
            or any(name not in OBSERVABLES for name in names)
        ):
            raise ValueError(
                "observables must be a nonempty unique subset of "
                "potential, acceleration, gradient"
            )

        geometry = model.geometry
        target_positions = targets.frame.transform_points_m(
            targets.positions_m, geometry.frame
        )
        geometry.frame._path_to(output_frame)
        context = EvaluationContext(
            model_spec=model.spec,
            targets=targets,
            frame=output_frame,
            observables=names,
            representation=StaticRepresentation(),
        )
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "output_frame", output_frame)
        object.__setattr__(self, "observables", names)
        object.__setattr__(self, "context", context)
        object.__setattr__(self, "_target_positions_bytes", target_positions.tobytes())

    @property
    def target_positions_m(self) -> np.ndarray:
        return np.frombuffer(self._target_positions_bytes, dtype=np.float64).reshape(
            len(self.targets), 3
        )

    @property
    def parameters(self) -> QTable:
        geometry = self.model.geometry
        center = geometry.center
        return QTable(
            {
                "center_x": [center[0]],
                "center_y": [center[1]],
                "center_z": [center[2]],
                "radius": [geometry.radius],
                "height": [geometry.height],
                "density": [self.model.density],
                "continuous_mass": [self.model.spec.continuous_mass],
                "target_count": [len(self.targets)],
                "geometry_frame": [geometry.frame.name],
                "target_frame": [self.targets.frame.name],
                "output_frame": [self.output_frame.name],
            }
        )

    @property
    def formulas(self) -> pd.DataFrame:
        expressions = {
            "potential": "finite uniform cylinder; potential zero at infinity",
            "acceleration": "negative gradient of finite-cylinder potential",
            "gradient": "spatial derivative of acceleration",
        }
        return pd.DataFrame(
            [
                {
                    "observable": name,
                    "expression": expressions[name],
                    "convention": "G-free analytic cylinder value; G applied once",
                }
                for name in self.observables
            ]
        )

    @property
    def references(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "source": (
                        "gragra.analytic.cylinder_potential."
                        "cylinder_analytic_observables_g_free"
                    ),
                    "detail": "elliptic and convergent analytic-series evaluation",
                },
                {
                    "source": "gragra.constants.G_SI",
                    "detail": "G_SI is applied once per requested result",
                },
                {"source": "CODATA 2018", "detail": "gravitational constant G_SI"},
            ]
        )

    def validate(self) -> _CylinderValidationReport:
        """Validate resolved frame paths before analytic evaluation."""
        self.context.validate()
        if not np.isfinite(self.target_positions_m).all():
            raise ValueError("transformed target positions must be finite")
        return _CylinderValidationReport(
            valid=True,
            observables=self.observables,
            target_count=len(self.targets),
        )

    def run(self) -> EvaluationResult:
        """Evaluate all analytic observables, then retain the requested fields."""
        self.validate()
        geometry = self.model.geometry
        values = cylinder_analytic_observables_g_free(
            geometry.center_m,
            geometry.radius_m,
            geometry.height_m,
            self.model.spec.density_kg_m3,
            self.target_positions_m,
        )
        output: dict[str, u.Quantity | None] = {
            "potential": None,
            "acceleration": None,
            "gradient": None,
        }
        for name in self.observables:
            raw = G_SI * getattr(values, name)
            if name == "acceleration":
                raw = geometry.frame.transform_vectors_m(raw, self.output_frame)
                output[name] = raw * (u.m / u.s**2)
            elif name == "gradient":
                raw = geometry.frame.transform_tensors(raw, self.output_frame)
                output[name] = raw / u.s**2
            else:
                output[name] = raw * (u.m**2 / u.s**2)

        result_values = GravityResult(**output)
        record = EvaluationRecord(
            backend="analytic",
            versions=MappingProxyType(
                {
                    "gragra": version("gragra"),
                    "python": python_version(),
                    "numpy": np.__version__,
                    "astropy": version("astropy"),
                }
            ),
            gravitational_constant_si=G_SI,
            g_applications=1,
        )
        return EvaluationResult(values=result_values, evaluation=self, record=record)


@dataclass(frozen=True, eq=False)
class UniformCylinderModel:
    """Uniform cylinder and its immutable physical assumptions."""

    spec: _UniformCylinderModelSpec

    @property
    def geometry(self) -> Cylinder:
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
    ) -> _UniformCylinderEvaluation:
        """Resolve targets, requested fields, and output components."""
        return _UniformCylinderEvaluation(
            self,
            targets,
            targets.frame if output_frame is None else output_frame,
            observables,
        )


def uniform_cylinder(geometry: Cylinder, density: u.Quantity) -> UniformCylinderModel:
    """Create an analytic static model of a real, finite-density cylinder."""
    return UniformCylinderModel(_UniformCylinderModelSpec(geometry, density))


__all__ = ["UniformCylinderModel", "uniform_cylinder"]
