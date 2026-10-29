"""Gen1 uniform-displacement cavity wall gravity model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import QTable

from gragra._arrays import _check_no_bool
from gragra._interactions import Newtonian
from gragra.analytic.cavity_cylinder import _reject_cylinder_boundary_targets
from gragra.analytic.sca_tensor import (
    _reject_cuboid_boundary,
    _reject_sphere_boundary,
)
from gragra.analytic.u9_sca_wall_observables import (
    cuboid_sca_wall_observables_g_free,
    cylinder_sca_wall_observables_g_free,
    sphere_sca_wall_observables_g_free,
)
from gragra.geometry._cuboid import Cuboid
from gragra.geometry._cylinder import Cylinder
from gragra.geometry._frame import Frame
from gragra.geometry._sphere import Sphere
from gragra.results._context import EvaluationContext
from gragra.results._coords import normalize_gps
from gragra.results._field_definition import GravityFieldDefinition
from gragra.results._gen1_definitions import gen1_field_definition
from gragra.results._record import OBSERVABLES
from gragra.results._representation import StaticRepresentation
from gragra.results._spectral import PhasorRepresentation
from gragra.targets._high_points import TargetPoints

_OBSERVABLES = ("potential", "acceleration", "gradient")
_GEOMETRIES = (Sphere, Cuboid, Cylinder)


def _density_kg_m3(value: object) -> float:
    if not isinstance(value, u.Quantity):
        raise TypeError("density must be an Astropy Quantity")
    _check_no_bool(value, "density")
    raw = np.asarray(value.to_value(u.kg / u.m**3))
    if raw.shape != () or np.iscomplexobj(raw):
        raise ValueError("density must be a real scalar")
    density = float(raw)
    if not np.isfinite(density) or density < 0.0:
        raise ValueError("density must be finite and nonnegative")
    return density


def _displacement_m(value: object) -> np.ndarray:
    if not isinstance(value, u.Quantity):
        raise TypeError("displacement must be an Astropy length Quantity")
    _check_no_bool(value, "displacement")
    raw = np.asarray(value.to_value(u.m))
    if raw.shape != (3,) or raw.dtype.kind not in "iufc":
        raise ValueError("displacement must be a real or complex three-vector")
    result = np.asarray(
        raw, dtype=np.complex128 if np.iscomplexobj(raw) else np.float64
    )
    if not np.isfinite(result).all():
        raise ValueError("displacement must be finite")
    return np.array(result, copy=True)


def _frequency_hz(value: object) -> float:
    if not isinstance(value, u.Quantity):
        raise TypeError("frequency must be an Astropy Quantity")
    _check_no_bool(value, "frequency")
    raw = np.asarray(value.to_value(u.Hz))
    if raw.shape != () or np.iscomplexobj(raw):
        raise ValueError("frequency must be a real scalar")
    frequency = float(raw)
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("frequency must be positive and finite")
    return frequency


@dataclass(frozen=True, eq=False, init=False)
class SCAWallModelSpec:
    """Validated immutable physical and representation inputs for SCA wall."""

    geometry: Sphere | Cuboid | Cylinder
    density_kg_m3: float
    _displacement_tuple_m: tuple[float | complex, float | complex, float | complex]
    representation: StaticRepresentation | PhasorRepresentation
    frequencies_hz: tuple[float, ...] | None
    reference_time_gps: tuple[int, int] | None
    field_definition: GravityFieldDefinition
    interaction: Newtonian
    g_convention: str
    _parameters: QTable
    _formulas: pd.DataFrame
    _references: pd.DataFrame

    def __init__(
        self,
        geometry: Sphere | Cuboid | Cylinder,
        displacement: u.Quantity,
        density: u.Quantity,
        frequency: u.Quantity | None,
        reference_time: object | None,
    ) -> None:
        if not isinstance(geometry, _GEOMETRIES):
            raise TypeError("geometry must be a high-level Sphere, Cuboid, or Cylinder")
        density_si = _density_kg_m3(density)
        displacement_m = _displacement_m(displacement)
        if frequency is None:
            if reference_time is not None:
                raise ValueError("reference_time requires a phasor frequency")
            if np.iscomplexobj(displacement_m):
                raise ValueError("static displacement must be real")
            representation = StaticRepresentation()
            frequencies_hz = None
            reference_time_gps = None
        else:
            frequency_si = _frequency_hz(frequency)
            if reference_time is None:
                raise ValueError("reference_time is required with frequency")
            representation = PhasorRepresentation()
            frequencies_hz = (frequency_si,)
            reference_time_gps = normalize_gps(reference_time)

        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "density_kg_m3", density_si)
        object.__setattr__(
            self,
            "_displacement_tuple_m",
            tuple(
                value.item() if isinstance(value, np.generic) else value
                for value in displacement_m
            ),
        )
        object.__setattr__(self, "representation", representation)
        object.__setattr__(self, "frequencies_hz", frequencies_hz)
        object.__setattr__(self, "reference_time_gps", reference_time_gps)
        object.__setattr__(self, "field_definition", gen1_field_definition("sca_wall"))
        object.__setattr__(self, "interaction", Newtonian())
        object.__setattr__(self, "g_convention", "SI_G_applied_once")
        object.__setattr__(self, "_parameters", self._make_parameters())
        object.__setattr__(self, "_formulas", self._make_formulas())
        object.__setattr__(self, "_references", self._make_references())

    @property
    def displacement_m(self) -> np.ndarray:
        return np.asarray(self._displacement_tuple_m)

    @property
    def parameters(self) -> QTable:
        return self._parameters

    def _make_parameters(self) -> QTable:
        geometry = self.geometry
        values: dict[str, list[object]] = {
            "geometry_type": [type(geometry).__name__],
            "density": [self.density_kg_m3 * u.kg / u.m**3],
            "center_x": [geometry.center_m[0] * u.m],
            "center_y": [geometry.center_m[1] * u.m],
            "center_z": [geometry.center_m[2] * u.m],
            "displacement_x": [self.displacement_m[0] * u.m],
            "displacement_y": [self.displacement_m[1] * u.m],
            "displacement_z": [self.displacement_m[2] * u.m],
            "geometry_frame": [geometry.frame.name],
        }
        if isinstance(geometry, Sphere):
            values["radius"] = [geometry.radius_m * u.m]
        elif isinstance(geometry, Cuboid):
            values.update(
                {
                    "length_x": [geometry.lengths_m[0] * u.m],
                    "length_y": [geometry.lengths_m[1] * u.m],
                    "length_z": [geometry.lengths_m[2] * u.m],
                }
            )
        else:
            values["radius"] = [geometry.radius_m * u.m]
            values["height"] = [geometry.height_m * u.m]
        if self.frequencies_hz is not None:
            values["frequency"] = [self.frequencies_hz[0] * u.Hz]
            assert self.reference_time_gps is not None
            values["reference_gps_seconds"] = [self.reference_time_gps[0]]
            values["reference_gps_nanoseconds"] = [self.reference_time_gps[1]]
        return QTable(values)

    @property
    def formulas(self) -> pd.DataFrame:
        return self._formulas

    def _make_formulas(self) -> pd.DataFrame:
        expressions = {
            "potential": "wall_Phi = -u dot a_body",
            "acceleration": "wall_a = Gamma_body dot u",
            "gradient": "wall_Gamma_ij = partial_k(Gamma_body_ij) u_k",
        }
        return pd.DataFrame(
            [
                {
                    "observable": name,
                    "expression": expressions[name],
                    "convention": (
                        "selected wall contribution; G-free native values; "
                        "G applied once"
                    ),
                }
                for name in OBSERVABLES
            ]
        )

    @property
    def references(self) -> pd.DataFrame:
        return self._references

    def _make_references(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "source": "gragra.analytic.u9_sca_wall_observables",
                    "detail": (
                        "closed-form sphere, cuboid, and cylinder wall observables"
                    ),
                }
            ]
        )


@dataclass(frozen=True, eq=False)
class SCAWallModel:
    """Model for the selected wall contribution of a uniformly displaced cavity."""

    spec: SCAWallModelSpec

    def at(
        self,
        targets: TargetPoints,
        *,
        observables: tuple[str, ...] = OBSERVABLES,
        output_frame: Frame | None = None,
    ):
        """Fix target coordinates and return the shared analytic evaluator."""
        if not isinstance(targets, TargetPoints):
            raise TypeError("targets must be TargetPoints")
        if not len(targets):
            raise ValueError("targets must be nonempty")
        computation_frame = self.spec.geometry.frame
        if output_frame is None:
            output_frame = targets.frame
        if not isinstance(output_frame, Frame):
            raise TypeError("output_frame must be a Frame")
        targets.frame._path_to(computation_frame)
        computation_frame._path_to(output_frame)
        positions = np.stack(
            [
                targets.frame.trace_point_m(point, computation_frame).point_m
                for point in targets.positions_m
            ]
        )
        context = EvaluationContext(
            model_spec=self.spec,
            targets=targets,
            frame=output_frame,
            observables=observables,
            representation=self.spec.representation,
            frequencies=(
                None
                if self.spec.frequencies_hz is None
                else np.asarray(self.spec.frequencies_hz) * u.Hz
            ),
            reference_time=self.spec.reference_time_gps,
            g_convention=self.spec.g_convention,
        )
        from gragra._analytic_evaluation import AnalyticEvaluation, NativeEvaluation

        def evaluate_native(points: np.ndarray, names: tuple[str, ...]):
            return NativeEvaluation(_evaluate_sca_wall_native(self.spec, points, names))

        return AnalyticEvaluation(
            context=context,
            computation_frame=computation_frame,
            target_positions_m=positions,
            evaluate_native=evaluate_native,
            validate_targets=lambda points: _validate_sca_wall_targets(
                self.spec, points
            ),
        )


def _requested_observables(observables: object) -> tuple[str, ...]:
    if isinstance(observables, (str, bytes)):
        raise TypeError("observables must be an iterable of observable names")
    try:
        names = tuple(observables)
    except TypeError as error:
        raise TypeError(
            "observables must be an iterable of observable names"
        ) from error
    if (
        not names
        or len(set(names)) != len(names)
        or any(name not in _OBSERVABLES for name in names)
    ):
        raise ValueError("observables must be a nonempty unique known subset")
    return names


def _evaluate_sca_wall_native(
    spec: SCAWallModelSpec,
    target_positions_m: object,
    observables: object,
) -> dict[str, np.ndarray]:
    """Evaluate requested G-free observables in the geometry's local frame."""
    if not isinstance(spec, SCAWallModelSpec):
        raise TypeError("spec must be an SCAWallModelSpec")
    names = _requested_observables(observables)
    points = np.asarray(target_positions_m)
    if points.dtype.kind not in "iuf" or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("target_positions_m must be a real (N, 3) table")
    points = np.asarray(points, dtype=np.float64)
    if not len(points):
        raise ValueError("target_positions_m must be nonempty")
    if not np.isfinite(points).all():
        raise ValueError("target_positions_m must be finite")

    geometry = spec.geometry
    options = {
        "center_m": geometry.center_m,
        "density_kg_m3": spec.density_kg_m3,
        "displacement_m": spec.displacement_m,
    }
    if isinstance(geometry, Sphere):
        native = sphere_sca_wall_observables_g_free(
            points, radius_m=geometry.radius_m, **options
        )
    elif isinstance(geometry, Cuboid):
        native = cuboid_sca_wall_observables_g_free(
            points, half_lengths_m=geometry.lengths_m / 2.0, **options
        )
    else:
        native = cylinder_sca_wall_observables_g_free(
            points,
            radius_m=geometry.radius_m,
            height_m=geometry.height_m,
            **options,
        )

    values = {name: np.asarray(getattr(native, name)) for name in names}
    if isinstance(spec.representation, PhasorRepresentation):
        values = {
            "potential": values["potential"][:, None, None]
            if "potential" in values
            else None,
            "acceleration": values["acceleration"][:, :, None]
            if "acceleration" in values
            else None,
            "gradient": values["gradient"].reshape(len(points), 9)[:, :, None]
            if "gradient" in values
            else None,
        }
        values = {name: value for name, value in values.items() if value is not None}
        values = {
            name: np.asarray(value, dtype=np.complex128)
            for name, value in values.items()
        }
    return values


def _validate_sca_wall_targets(spec: SCAWallModelSpec, points: np.ndarray) -> None:
    with np.errstate(over="ignore", invalid="ignore"):
        relative = np.asarray(points, dtype=np.float64) - spec.geometry.center_m
    if not np.isfinite(relative).all():
        raise ValueError("target-relative geometry coordinates must be finite")
    geometry = spec.geometry
    if isinstance(geometry, Sphere):
        _reject_sphere_boundary(relative, geometry.radius_m)
    elif isinstance(geometry, Cuboid):
        _reject_cuboid_boundary(relative, geometry.lengths_m / 2.0)
    else:
        _reject_cylinder_boundary_targets(
            relative, geometry.radius_m, geometry.height_m
        )


def small_cavity_wall(
    geometry: Sphere | Cuboid | Cylinder,
    displacement: u.Quantity,
    *,
    density: u.Quantity,
    frequency: u.Quantity | None = None,
    reference_time: object | None = None,
) -> SCAWallModel:
    """Create a static or single-frequency SCA wall field-change model.

    Displacement is a three-vector in ``geometry.frame``. Without ``frequency``
    it is a real static displacement. Supplying ``frequency`` and
    ``reference_time`` interprets its real or complex vector as a peak phasor.
    """
    return SCAWallModel(
        SCAWallModelSpec(geometry, displacement, density, frequency, reference_time)
    )


__all__ = ["small_cavity_wall"]
