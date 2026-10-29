"""High-level static uniform-cuboid analytic model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version
from platform import python_version
from types import MappingProxyType

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import QTable

from gragra._evaluation_setup import OBSERVABLES, EvaluationRecord, density_kg_m3
from gragra._interactions import Newtonian
from gragra.constants import G_SI
from gragra.geometry._cuboid import Cuboid
from gragra.geometry._frame import Frame, PointTrace
from gragra.results import (
    Contribution,
    GravityFieldDefinition,
    PotentialReference,
    StaticRepresentation,
)
from gragra.targets._high_points import TargetPoints


@dataclass(frozen=True, eq=False, init=False)
class CuboidModelSpec:
    """Immutable physical inputs and field meaning for a uniform cuboid."""

    geometry: Cuboid
    density_kg_m3: float
    field_definition: GravityFieldDefinition
    interaction: Newtonian

    def __init__(self, geometry: Cuboid, density: u.Quantity) -> None:
        if not isinstance(geometry, Cuboid):
            raise TypeError("geometry must be a high-level Cuboid")
        density_value = density_kg_m3(density)
        if density_value < 0:
            raise ValueError("density must be non-negative")
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "density_kg_m3", density_value)
        object.__setattr__(
            self,
            "field_definition",
            GravityFieldDefinition.source_field(
                "uniform_cuboid",
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
        return float(np.prod(self.geometry.lengths_m) * self.density_kg_m3) * u.kg


@dataclass(frozen=True, eq=False)
class UniformCuboidModel:
    """Uniform cuboid and its immutable physical assumptions."""

    spec: CuboidModelSpec

    @property
    def geometry(self) -> Cuboid:
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
    ) -> CuboidEvaluationSetup:
        """Resolve target coordinates and output components for evaluation."""
        resolved = CuboidResolvedEvaluation(
            self,
            targets,
            self.geometry.frame,
            targets.frame if output_frame is None else output_frame,
            observables,
        )
        return CuboidEvaluationSetup(resolved)


def uniform_cuboid(geometry: Cuboid, density: u.Quantity) -> UniformCuboidModel:
    """Create an analytic static model of a finite-density uniform cuboid."""
    return UniformCuboidModel(CuboidModelSpec(geometry, density))


@dataclass(frozen=True, eq=False, init=False)
class CuboidResolvedEvaluation:
    """Fixed cuboid request with targets mapped into the geometry frame."""

    model: UniformCuboidModel
    model_spec: CuboidModelSpec
    targets: TargetPoints
    computation_frame: Frame
    output_frame: Frame
    observables: tuple[str, ...]
    target_traces: tuple[PointTrace, ...]
    _target_positions_bytes: bytes

    def __init__(
        self,
        model: UniformCuboidModel,
        targets: TargetPoints,
        computation_frame: Frame,
        output_frame: Frame,
        observables: tuple[str, ...] = OBSERVABLES,
    ) -> None:
        if not isinstance(model, UniformCuboidModel):
            raise TypeError("model must be a UniformCuboidModel")
        if not isinstance(targets, TargetPoints):
            raise TypeError("targets must be high-level TargetPoints")
        if computation_frame is not model.geometry.frame:
            raise ValueError("computation_frame must be model geometry.frame")
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
        computation_frame._path_to(output_frame)
        traces = tuple(
            targets.frame.trace_point_m(point, computation_frame)
            for point in targets.positions_m
        )
        positions = np.array([trace.point_m for trace in traces], dtype=np.float64)
        if not traces:
            positions = positions.reshape(0, 3)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "model_spec", model.spec)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "computation_frame", computation_frame)
        object.__setattr__(self, "output_frame", output_frame)
        object.__setattr__(self, "observables", names)
        object.__setattr__(self, "target_traces", traces)
        object.__setattr__(self, "_target_positions_bytes", positions.tobytes())

    @property
    def target_positions_m(self) -> np.ndarray:
        return np.frombuffer(self._target_positions_bytes, dtype=np.float64).reshape(
            len(self.targets), 3
        )

    @property
    def parameters(self) -> QTable:
        geometry = self.model_spec.geometry
        center = geometry.center_m
        lengths = geometry.lengths_m
        return QTable(
            {
                "center_x": [center[0] * u.m],
                "center_y": [center[1] * u.m],
                "center_z": [center[2] * u.m],
                "length_x": [lengths[0] * u.m],
                "length_y": [lengths[1] * u.m],
                "length_z": [lengths[2] * u.m],
                "density": [self.model_spec.density],
                "continuous_mass": [self.model_spec.continuous_mass],
                "target_count": [len(self.targets)],
                "geometry_frame": [geometry.frame.name],
                "target_frame": [self.targets.frame.name],
                "output_frame": [self.output_frame.name],
            }
        )

    @property
    def formulas(self) -> pd.DataFrame:
        expressions = {
            "potential": "G * cuboid_observables_g_free(...).potential",
            "acceleration": "G * cuboid_observables_g_free(...).acceleration",
            "gradient": "G * cuboid_observables_g_free(...).gradient",
        }
        return pd.DataFrame(
            [
                {
                    "observable": name,
                    "expression": expressions[name],
                    "convention": "uniform density; potential zero at infinity; "
                    "G applied once",
                }
                for name in self.observables
            ]
        )

    @property
    def references(self) -> pd.DataFrame:
        methods = [
            {
                "source": "gragra.analytic.cavity_cuboid.cuboid_observables_g_free",
                "detail": "G-free analytical cuboid calculation",
            },
            {
                "source": "gragra.constants.G_SI",
                "detail": "G_SI is applied once per requested result",
            },
            {"source": "CODATA 2018", "detail": "gravitational constant G_SI"},
        ]
        return pd.DataFrame(methods)


@dataclass(frozen=True)
class CuboidValidationReport:
    """Geometry-region summary for one fixed cuboid target set."""

    valid: bool
    observables: tuple[str, ...]
    target_count: int
    target_regions: Mapping[str, str]


class CuboidEvaluationSetup:
    """One cuboid request shared by inspection, validation, and execution."""

    __slots__ = ("_resolved", "_context")

    def __init__(self, resolved: CuboidResolvedEvaluation) -> None:
        if not isinstance(resolved, CuboidResolvedEvaluation):
            raise TypeError("resolved must be a CuboidResolvedEvaluation")
        self._resolved = resolved
        from gragra.results._context import EvaluationContext

        self._context = EvaluationContext(
            model_spec=resolved.model_spec,
            targets=resolved.targets,
            frame=resolved.output_frame,
            observables=resolved.observables,
            representation=StaticRepresentation(),
        )

    @property
    def context(self):
        return self._context

    @property
    def resolved(self) -> CuboidResolvedEvaluation:
        return self._resolved

    @property
    def evaluation(self) -> CuboidResolvedEvaluation:
        return self._resolved

    @property
    def parameters(self) -> QTable:
        return self._resolved.parameters

    @property
    def formulas(self) -> pd.DataFrame:
        return self._resolved.formulas

    @property
    def references(self) -> pd.DataFrame:
        return self._resolved.references

    def validate(self) -> CuboidValidationReport:
        geometry = self._resolved.model_spec.geometry
        half_lengths = geometry.lengths_m / 2.0
        regions: dict[str, str] = {}
        from gragra.analytic.cavity_cuboid import _is_on_cuboid_boundary

        boundary = _is_on_cuboid_boundary(
            self._resolved.target_positions_m - geometry.center_m,
            half_lengths,
        )
        for target_id, point, on_boundary in zip(
            self._resolved.targets.ids,
            self._resolved.target_positions_m,
            boundary,
            strict=True,
        ):
            offset = np.abs(point - geometry.center_m)
            if on_boundary:
                region = "boundary"
            elif np.all(offset < half_lengths):
                region = "interior"
            else:
                region = "exterior"
            regions[target_id] = region
        return CuboidValidationReport(
            valid=not bool(np.any(boundary)),
            observables=self._resolved.observables,
            target_count=len(self._resolved.targets),
            target_regions=MappingProxyType(regions),
        )

    def run(self):
        """Evaluate the native G-free kernel and create an SI gravity result."""
        self.validate()
        resolved = self._resolved
        geometry = resolved.model_spec.geometry
        from gragra.analytic.cavity_cuboid import cuboid_observables_g_free

        raw = cuboid_observables_g_free(
            geometry.center_m,
            geometry.lengths_m,
            resolved.model_spec.density_kg_m3,
            resolved.target_positions_m,
        )
        arrays = {
            "potential": G_SI * raw.potential,
            "acceleration": G_SI * raw.acceleration,
            "gradient": G_SI * raw.gradient,
        }
        arrays["acceleration"] = resolved.computation_frame.transform_vectors_m(
            arrays["acceleration"], resolved.output_frame
        )
        arrays["gradient"] = resolved.computation_frame.transform_tensors(
            arrays["gradient"], resolved.output_frame
        )
        units = {
            "potential": u.m**2 / u.s**2,
            "acceleration": u.m / u.s**2,
            "gradient": u.s**-2,
        }
        from gragra.results import GravityResult
        from gragra.results._evaluation import EvaluationResult

        values = GravityResult(
            **{
                name: arrays[name] * units[name]
                if name in resolved.observables
                else None
                for name in OBSERVABLES
            }
        )
        record = EvaluationRecord(
            backend="numpy",
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
        return EvaluationResult(values=values, evaluation=self, record=record)


__all__ = [
    "CuboidEvaluationSetup",
    "CuboidModelSpec",
    "CuboidResolvedEvaluation",
    "CuboidValidationReport",
    "UniformCuboidModel",
    "uniform_cuboid",
]
