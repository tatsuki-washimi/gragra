"""Shared immutable C5 evaluation inputs, metadata, and precision diagnostics.

This module has no dependency on the analytic, plotting, or result workers.
Numerical kernels stay in their existing modules; no source is generated here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import QTable

from gragra._interactions import Newtonian
from gragra.geometry._frame import Frame, PointTrace, _xyz_m
from gragra.geometry._sphere import Sphere
from gragra.results import (
    Contribution,
    GravityFieldDefinition,
    PotentialReference,
)
from gragra.results._record import OBSERVABLES as OBSERVABLES
from gragra.results._record import EvaluationRecord as EvaluationRecord
from gragra.targets._high_points import TargetPoints


def density_kg_m3(value: u.Quantity) -> float:
    """Validate a finite, scalar, real density and return its SI value."""
    if not isinstance(value, u.Quantity):
        raise TypeError("density must be an astropy Quantity")
    raw = np.asarray(value.to_value(u.kg / u.m**3))
    if raw.shape != ():
        raise ValueError("density must be scalar")
    if np.iscomplexobj(raw):
        raise ValueError("density must be real")
    density = float(raw)
    if not np.isfinite(density):
        raise ValueError("density must be finite")
    return density


@dataclass(frozen=True, eq=False, init=False)
class UniformSphereModelSpec:
    """Canonical immutable physical assumptions of a static uniform sphere.

    The analytic factory may hold this as ``model.spec``. The density here is
    physical mass density, independent of any midpoint discretization.
    """

    geometry: Sphere
    density_kg_m3: float
    field_definition: GravityFieldDefinition
    interaction: Newtonian

    def __init__(self, geometry: Sphere, density: u.Quantity) -> None:
        if not isinstance(geometry, Sphere):
            raise TypeError("geometry must be a high-level Sphere")
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "density_kg_m3", density_kg_m3(density))
        object.__setattr__(
            self,
            "field_definition",
            GravityFieldDefinition.source_field(
                "uniform_sphere",
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
        return self.geometry.volume * self.density


def _plain_metadata(name: str, source: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(source, Mapping):
        raise TypeError(f"{name} must be a mapping")
    entries: dict[str, object] = {}
    for key, raw in source.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be nonempty strings")
        if raw is not None and not isinstance(raw, (str, int, float, bool, np.number)):
            raise TypeError(f"{name}[{key!r}] must be a scalar or None")
        value = raw.item() if isinstance(raw, np.generic) else raw
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f"{name}[{key!r}] must be finite")
        entries[key] = value
    return MappingProxyType(entries)


def _validate_field_definition(value: object) -> GravityFieldDefinition:
    if not isinstance(value, GravityFieldDefinition):
        raise TypeError("field_definition must be a GravityFieldDefinition")
    if value.potential_reference is not PotentialReference.ZERO_AT_INFINITY:
        raise ValueError(
            "static point kernels require potential_reference ZERO_AT_INFINITY"
        )
    return value


def _normalize_point_source_provenance(
    value: Mapping[str, object],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("provenance must be a mapping")
    normalized: dict[str, object] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("provenance keys must be nonempty strings")
        if isinstance(raw, np.generic):
            raw = raw.item()
        if raw is None or isinstance(raw, (str, bool, int)):
            normalized[key] = raw
            continue
        if isinstance(raw, (float, np.floating)):
            item = float(raw)
            if not np.isfinite(item):
                raise ValueError(f"provenance[{key!r}] must be finite")
            normalized[key] = item
            continue
        if isinstance(raw, np.ndarray):
            if raw.ndim != 1:
                raise ValueError(f"provenance[{key!r}] must be a flat value")
            raw = tuple(raw.tolist())
        elif isinstance(raw, list):
            raw = tuple(raw)
        if isinstance(raw, tuple):
            items = []
            for item in raw:
                if isinstance(item, np.generic):
                    item = item.item()
                if isinstance(item, bool) or not isinstance(item, (int, float)):
                    raise TypeError(
                        f"provenance[{key!r}] tuple values must be real numbers"
                    )
                number = float(item)
                if not np.isfinite(number):
                    raise ValueError(f"provenance[{key!r}] tuple values must be finite")
                items.append(number)
            normalized[key] = tuple(items)
            continue
        raise TypeError(
            f"provenance[{key!r}] must be a scalar, string, or flat numeric tuple"
        )
    if "origin_m" in normalized:
        origin = normalized["origin_m"]
        if not isinstance(origin, tuple) or len(origin) != 3:
            raise ValueError("provenance['origin_m'] must be a finite 3-vector")
    return MappingProxyType(normalized)


@dataclass(frozen=True, eq=False, init=False)
class _StaticPointSourceBinding:
    """Owned real SI source arrays plus explicit frame and caller provenance."""

    frame: Frame
    field_definition: GravityFieldDefinition
    source_type: str
    provenance: Mapping[str, object]
    _positions_bytes: bytes
    _weights_bytes: bytes
    _count: int
    _weight_sum_kg: float

    def __init__(
        self,
        positions_m: object,
        weights_kg: object,
        frame: Frame,
        field_definition: GravityFieldDefinition,
        source_type: str,
        provenance: Mapping[str, object],
    ) -> None:
        if not isinstance(frame, Frame):
            raise TypeError("frame must be a Frame")
        if not isinstance(source_type, str) or not source_type:
            raise ValueError("source_type must be a nonempty string")
        definition = _validate_field_definition(field_definition)
        positions = _xyz_m("positions_m", positions_m)
        if positions.ndim != 2 or len(positions) == 0:
            raise ValueError("positions_m must have nonempty shape (N, 3)")
        raw_weights = np.asarray(weights_kg)
        if np.iscomplexobj(raw_weights):
            raise ValueError("Static point sums reject complex weights")
        try:
            weights = np.asarray(raw_weights, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError("weights_kg must contain real numbers") from exc
        if weights.shape != (len(positions),):
            raise ValueError("weights_kg must have shape (N,)")
        if not np.isfinite(weights).all():
            raise ValueError("weights_kg must contain only finite values")
        with np.errstate(over="ignore", invalid="ignore"):
            weight_sum = float(np.sum(weights, dtype=np.float64))
        if not np.isfinite(weight_sum):
            raise ValueError("sum(weights_kg) must remain finite in float64")

        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "field_definition", definition)
        object.__setattr__(self, "source_type", source_type)
        object.__setattr__(
            self,
            "provenance",
            _normalize_point_source_provenance(provenance),
        )
        object.__setattr__(self, "_positions_bytes", positions.copy().tobytes())
        object.__setattr__(self, "_weights_bytes", weights.copy().tobytes())
        object.__setattr__(self, "_count", len(positions))
        object.__setattr__(self, "_weight_sum_kg", weight_sum)

    @property
    def positions_m(self) -> np.ndarray:
        return np.frombuffer(self._positions_bytes, dtype=np.float64).reshape(
            self._count, 3
        )

    @property
    def weights_kg(self) -> np.ndarray:
        return np.frombuffer(self._weights_bytes, dtype=np.float64)

    @property
    def weight_sum_kg(self) -> float:
        return self._weight_sum_kg

    def __len__(self) -> int:
        return self._count


@dataclass(frozen=True, eq=False, init=False)
class _StaticPointSumSpec:
    """Private static interpretation for a bound point source without geometry."""

    frame: Frame
    field_definition: GravityFieldDefinition
    interaction: Newtonian

    def __init__(self, frame: Frame, field_definition: GravityFieldDefinition) -> None:
        if not isinstance(frame, Frame):
            raise TypeError("frame must be a Frame")
        definition = _validate_field_definition(field_definition)
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "field_definition", definition)
        object.__setattr__(self, "interaction", Newtonian())


@dataclass(frozen=True, eq=False, init=False)
class DiscreteSource:
    """One actual, G-free midpoint source shared by plot, report, and run.

    ``positions_m`` are source-frame SI coordinates. ``weights_kg`` are
    element masses, not density or G-weighted strengths. ``max_points`` in
    discretization metadata limits the *bounding-box grid* count before
    sphere inclusion, not the number of retained sources.
    """

    geometry: Sphere
    density_kg_m3: float
    frame: Frame
    discretization: Mapping[str, object]
    _positions_bytes: bytes
    _weights_bytes: bytes
    _count: int

    def __init__(
        self,
        geometry: Sphere,
        density: u.Quantity,
        frame: Frame,
        positions_m: object,
        weights_kg: object,
        discretization: Mapping[str, object],
    ) -> None:
        if not isinstance(geometry, Sphere):
            raise TypeError("geometry must be a high-level Sphere")
        if frame is not geometry.frame:
            raise ValueError("source frame must be geometry.frame")
        positions = _xyz_m("positions_m", positions_m)
        if positions.ndim != 2 or len(positions) == 0:
            raise ValueError("positions_m must have nonempty shape (N, 3)")
        raw_weights = np.asarray(weights_kg)
        if np.iscomplexobj(raw_weights):
            raise ValueError("weights_kg must be real for static C5 sources")
        weights = np.asarray(raw_weights, dtype=np.float64)
        if weights.shape != (len(positions),) or not np.isfinite(weights).all():
            raise ValueError("weights_kg must be finite and have shape (N,)")
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "density_kg_m3", density_kg_m3(density))
        object.__setattr__(self, "frame", frame)
        object.__setattr__(
            self, "discretization", _plain_metadata("discretization", discretization)
        )
        object.__setattr__(self, "_positions_bytes", positions.copy().tobytes())
        object.__setattr__(self, "_weights_bytes", weights.copy().tobytes())
        object.__setattr__(self, "_count", len(positions))

    @property
    def density(self) -> u.Quantity:
        return self.density_kg_m3 * u.kg / u.m**3

    @property
    def positions_m(self) -> np.ndarray:
        return np.frombuffer(self._positions_bytes, dtype=np.float64).reshape(
            self._count, 3
        )

    @property
    def weights_kg(self) -> np.ndarray:
        return np.frombuffer(self._weights_bytes, dtype=np.float64)

    @property
    def mass(self) -> u.Quantity:
        return float(np.sum(self.weights_kg)) * u.kg

    def __len__(self) -> int:
        return self._count


@dataclass(frozen=True, eq=False, init=False)
class ResolvedEvaluation:
    """A fixed request; all consumers use these transformed targets and source.

    ``model`` is either a Sphere spec, the private static point-sum spec, or an
    analytic model exposing a Sphere spec. ``source`` is None for analytic
    evaluation. No discretization or numeric execution occurs here.
    """

    model: object
    model_spec: UniformSphereModelSpec | _StaticPointSumSpec
    targets: TargetPoints
    computation_frame: Frame
    output_frame: Frame
    observables: tuple[str, ...]
    source: DiscreteSource | _StaticPointSourceBinding | None
    target_traces: tuple[PointTrace, ...]
    _target_positions_bytes: bytes

    def __init__(
        self,
        model: object,
        targets: TargetPoints,
        computation_frame: Frame,
        output_frame: Frame,
        observables: tuple[str, ...] = OBSERVABLES,
        source: DiscreteSource | _StaticPointSourceBinding | None = None,
    ) -> None:
        spec = (
            model
            if isinstance(model, (UniformSphereModelSpec, _StaticPointSumSpec))
            else getattr(model, "spec", None)
        )
        if not isinstance(spec, (UniformSphereModelSpec, _StaticPointSumSpec)):
            raise TypeError(
                "model must be or expose a supported static model spec as .spec"
            )
        if not isinstance(targets, TargetPoints):
            raise TypeError("targets must be high-level TargetPoints")
        model_frame = (
            spec.geometry.frame
            if isinstance(spec, UniformSphereModelSpec)
            else spec.frame
        )
        if computation_frame is not model_frame:
            raise ValueError("computation_frame must be the model source frame")
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
        if source is not None:
            if isinstance(spec, UniformSphereModelSpec):
                if not isinstance(source, DiscreteSource):
                    raise TypeError("Sphere spec source must be a DiscreteSource")
                if (
                    source.geometry is not spec.geometry
                    or source.frame is not computation_frame
                ):
                    raise ValueError(
                        "source must refer to this geometry and computation_frame"
                    )
                if not np.isclose(
                    source.density_kg_m3, spec.density_kg_m3, rtol=1e-12, atol=0
                ):
                    raise ValueError("source density must match model density")
            else:
                if not isinstance(source, _StaticPointSourceBinding):
                    raise TypeError("static point-sum spec requires a bound source")
                if (
                    source.frame is not computation_frame
                    or source.frame is not spec.frame
                ):
                    raise ValueError("bound source must retain the model source frame")
                if source.field_definition is not spec.field_definition:
                    raise ValueError(
                        "bound source field definition must match model spec"
                    )
        elif isinstance(spec, _StaticPointSumSpec):
            raise ValueError("static point-sum spec requires a bound source")
        # Check both roots before storing any resolved state.
        computation_frame._path_to(output_frame)
        traces = tuple(
            targets.frame.trace_point_m(point, computation_frame)
            for point in targets.positions_m
        )
        target_positions = np.array(
            [trace.point_m for trace in traces], dtype=np.float64
        )
        if len(traces) == 0:
            target_positions = target_positions.reshape(0, 3)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "model_spec", spec)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "computation_frame", computation_frame)
        object.__setattr__(self, "output_frame", output_frame)
        object.__setattr__(self, "observables", names)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target_traces", traces)
        object.__setattr__(self, "_target_positions_bytes", target_positions.tobytes())

    @property
    def target_positions_m(self) -> np.ndarray:
        return np.frombuffer(self._target_positions_bytes, dtype=np.float64).reshape(
            len(self.targets), 3
        )

    @property
    def parameters(self) -> QTable:
        """One-row, dimensionally typed pre-run physical and source parameters."""
        if isinstance(self.model_spec, _StaticPointSumSpec):
            assert isinstance(self.source, _StaticPointSourceBinding)
            table = QTable(
                {
                    "target_count": [len(self.targets)],
                    "source_frame": [self.source.frame.name],
                    "target_frame": [self.targets.frame.name],
                    "output_frame": [self.output_frame.name],
                }
            )
            table.meta = {
                "source_type": self.source.source_type,
                "source_count": len(self.source),
                "weights_sum_kg": self.source.weight_sum_kg,
                "source_provenance": dict(self.source.provenance),
            }
            return table

        sphere = self.model_spec.geometry
        values: dict[str, object] = {
            "center_x": [sphere.center[0]],
            "center_y": [sphere.center[1]],
            "center_z": [sphere.center[2]],
            "radius": [sphere.radius],
            "density": [self.model_spec.density],
            "continuous_mass": [self.model_spec.continuous_mass],
            "target_count": [len(self.targets)],
            "geometry_frame": [sphere.frame.name],
            "target_frame": [self.targets.frame.name],
            "output_frame": [self.output_frame.name],
        }
        if self.source is not None:
            values["discrete_mass"] = [self.source.mass]
            values["mass_difference"] = [
                self.source.mass - self.model_spec.continuous_mass
            ]
            values["source_count"] = [len(self.source)]
            for key in ("spacing_m", "max_points", "grid_points", "cell_volume_m3"):
                if key in self.source.discretization:
                    value = self.source.discretization[key]
                    if key == "spacing_m" and value is not None:
                        value = float(value) * u.m
                    elif key == "cell_volume_m3" and value is not None:
                        value = float(value) * u.m**3
                    values[key] = [value]
        return QTable(values)

    @property
    def formulas(self) -> pd.DataFrame:
        """Formulas tied to the immutable model assumptions, before or after run."""
        if self.source is None:
            expressions = {
                "potential": "exterior -GM/r; interior -GM(3R^2-r^2)/(2R^3)",
                "acceleration": "exterior -GM r_vec/r^3; interior -GM r_vec/R^3",
                "gradient": "exterior GM(3 rr^T/r^5-I/r^3); interior -GM I/R^3",
            }
        elif isinstance(self.source, DiscreteSource):
            expressions = {
                "potential": "-G sum_j m_j/|r_j|",
                "acceleration": "-G sum_j m_j r_j/|r_j|^3",
                "gradient": "G sum_j m_j (3 r_j r_j^T/|r_j|^5-I/|r_j|^3)",
            }
            convention = (
                "r_j=target-source_j; source mass; "
                "potential zero at infinity; G applied once"
            )
        else:
            expressions = {
                "potential": "-G sum_j w_j/|r_j|",
                "acceleration": "-G sum_j w_j r_j/|r_j|^3",
                "gradient": "G sum_j w_j (3 r_j r_j^T/|r_j|^5-I/|r_j|^3)",
            }
            convention = (
                "r_j=target-source_j; declared static source weights; "
                "potential zero at infinity; G applied once"
            )
        return pd.DataFrame(
            [
                {
                    "observable": name,
                    "expression": expressions[name],
                    "convention": (
                        convention
                        if isinstance(self.source, _StaticPointSourceBinding)
                        else "r_j=target-source_j; source mass; "
                        "potential zero at infinity; G applied once"
                    ),
                }
                for name in self.observables
            ]
        )

    @property
    def references(self) -> pd.DataFrame:
        if self.source is None:
            if set(self.observables) == set(OBSERVABLES):
                method = [
                    {
                        "source": (
                            "gragra.observables.analytic.uniform_sphere_observables"
                        ),
                        "detail": "SI wrapper over G-free uniform-sphere formulas",
                    }
                ]
            else:
                helpers = {
                    "potential": "uniform_sphere_potential",
                    "acceleration": "uniform_sphere_acceleration",
                    "gradient": "uniform_sphere_gradient",
                }
                method = [
                    {
                        "source": f"gragra.analytic.uniform_sphere.{helpers[name]}",
                        "detail": "G-free analytical uniform-sphere calculation",
                    }
                    for name in self.observables
                ]
                method.append(
                    {
                        "source": "gragra.constants.G_SI",
                        "detail": ("G_SI is applied once per requested result"),
                    }
                )
        elif isinstance(self.source, DiscreteSource):
            method = [
                {
                    "source": f"gragra.observables.{name}.{function}",
                    "detail": "SI point-mass sum with actual midpoint weights",
                }
                for name, function in (
                    ("potential", "point_potential"),
                    ("acceleration", "point_acceleration"),
                    ("gradient", "point_gravity_gradient"),
                )
                if name in self.observables
            ]
            method.append(
                {
                    "source": "gragra.geometry.primitives.Sphere.to_weighted_points",
                    "detail": "existing Cartesian midpoint inclusion convention",
                }
            )
        else:
            method = [
                {
                    "source": f"gragra.observables.{name}.{function}",
                    "detail": "SI point sum over an immutable real-weight binding",
                }
                for name, function in (
                    ("potential", "point_potential"),
                    ("acceleration", "point_acceleration"),
                    ("gradient", "point_gravity_gradient"),
                )
                if name in self.observables
            ]
        method.append(
            {"source": "CODATA 2018", "detail": "gravitational constant G_SI"}
        )
        return pd.DataFrame(method)


class CoordinatePrecisionError(ValueError):
    """Structured collision or conservative cross-frame proximity failure."""

    def __init__(
        self,
        *,
        target_id: str,
        source_index: int,
        distance_m: float,
        threshold_m: float,
        edge_count: int,
        scale_m: float,
        observables: tuple[str, ...],
        reason: str,
    ) -> None:
        self.target_id = target_id
        self.source_index = source_index
        self.distance_m = distance_m
        self.threshold_m = threshold_m
        self.edge_count = edge_count
        self.scale_m = scale_m
        self.unit = "m"
        self.observables = observables
        self.reason = reason
        super().__init__(
            f"{reason}: target ID={target_id!r}, source index={source_index}, "
            f"d={distance_m:g} m, tau={threshold_m:g} m, K={edge_count}, "
            f"L={scale_m:g} m, observables={observables}"
        )


def check_pair_precision(
    source_position_m: object,
    target_trace: PointTrace,
    *,
    target_id: str,
    source_index: int,
    observables: tuple[str, ...],
) -> None:
    """Stop for exact collision or pair-specific cross-frame near coincidence.

    Both points are in the source frame. L includes that source coordinate and
    only target coordinates/origins on the actual common-ancestor path. Same
    frame, distinct near points keep the existing numeric kernel's behavior.
    This test is a conservative diagnostic, not an error bound or softening.
    """
    source = _xyz_m("source_position_m", source_position_m)
    if source.shape != (3,):
        raise ValueError("source_position_m must have shape (3,)")
    if not isinstance(target_trace, PointTrace):
        raise TypeError("target_trace must be a PointTrace")
    with np.errstate(over="ignore", invalid="ignore"):
        difference = source - target_trace.point_m
    if not np.isfinite(difference).all():
        raise ValueError("pair distance is not finite in source coordinates")
    distance_m = float(np.max(np.abs(difference)))
    scale_m = max(target_trace.scale_m, float(np.max(np.abs(source))))
    threshold_m = 64 * target_trace.edge_count * np.finfo(np.float64).eps * scale_m
    reason = (
        "collision"
        if distance_m == 0
        else "coordinate_precision"
        if target_trace.edge_count > 0 and distance_m <= threshold_m
        else None
    )
    if reason is not None:
        raise CoordinatePrecisionError(
            target_id=target_id,
            source_index=source_index,
            distance_m=distance_m,
            threshold_m=threshold_m,
            edge_count=target_trace.edge_count,
            scale_m=scale_m,
            observables=tuple(observables),
            reason=reason,
        )
