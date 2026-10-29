"""Owned, frame-aware inputs for isobaric temperature perturbations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy import units as u

from gragra._arrays import _check_no_bool
from gragra.fields.thermodynamic import isobaric_temperature_density_perturbation
from gragra.geometry._frame import Frame
from gragra.gravity_sources.volume import VolumeElementSource
from gragra.results import GravityFieldDefinition
from gragra.results._gen1_definitions import gen1_field_definition


def _quantity_array(
    name: str,
    value: object,
    unit: u.UnitBase,
    *,
    shape: tuple[int, ...] | None = None,
    equivalencies: list | tuple = (),
) -> np.ndarray:
    if not isinstance(value, u.Quantity):
        raise TypeError(f"{name} must be an Astropy Quantity")
    _check_no_bool(value, name)
    try:
        raw = np.asarray(value.to_value(unit, equivalencies=equivalencies))
    except (TypeError, ValueError, u.UnitConversionError) as exc:
        raise u.UnitConversionError(f"{name} must convert to {unit}") from exc
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain only real values")
    try:
        array = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must contain real numeric values") from exc
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _scalar_quantity(
    name: str,
    value: object,
    unit: u.UnitBase,
    *,
    positive: bool = False,
) -> float:
    array = _quantity_array(name, value, unit)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar Quantity")
    scalar = float(array)
    if positive and scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _readonly_quantity(value: np.ndarray | tuple[float, ...], unit: u.UnitBase):
    result = np.asarray(value, dtype=np.float64).copy() * unit
    result.setflags(write=False)
    return result


@dataclass(frozen=True, eq=False, slots=True, init=False)
class TemperatureFourierMode:
    """Spatial Fourier-mode temperature profile in a fixed Cartesian frame."""

    frame: Frame
    _wavevector_tuple_rad_m: tuple[float, float, float]
    _phase_origin_tuple_m: tuple[float, float, float]

    def __init__(
        self,
        *,
        wavevector: u.Quantity,
        phase_origin: u.Quantity,
        frame: Frame,
    ) -> None:
        if not isinstance(frame, Frame):
            raise TypeError("frame must be a Frame")
        vector = _quantity_array(
            "wavevector",
            wavevector,
            u.rad / u.m,
            shape=(3,),
            equivalencies=u.dimensionless_angles(),
        )
        if not np.any(vector != 0.0):
            raise ValueError("wavevector must be nonzero")
        origin = _quantity_array("phase_origin", phase_origin, u.m, shape=(3,))
        object.__setattr__(self, "frame", frame)
        object.__setattr__(
            self, "_wavevector_tuple_rad_m", tuple(float(x) for x in vector)
        )
        object.__setattr__(
            self, "_phase_origin_tuple_m", tuple(float(x) for x in origin)
        )

    @property
    def wavevector(self) -> u.Quantity:
        return _readonly_quantity(self._wavevector_tuple_rad_m, u.rad / u.m)

    @property
    def phase_origin(self) -> u.Quantity:
        return _readonly_quantity(self._phase_origin_tuple_m, u.m)

    @property
    def wavevector_rad_m(self) -> np.ndarray:
        return np.asarray(self._wavevector_tuple_rad_m, dtype=np.float64)

    @property
    def phase_origin_m(self) -> np.ndarray:
        return np.asarray(self._phase_origin_tuple_m, dtype=np.float64)


@dataclass(frozen=True, eq=False, slots=True, init=False)
class TemperatureGaussian:
    """All-space isotropic Gaussian temperature profile."""

    frame: Frame
    _center_tuple_m: tuple[float, float, float]
    sigma_m: float

    def __init__(
        self,
        *,
        center: u.Quantity,
        sigma: u.Quantity,
        frame: Frame,
    ) -> None:
        if not isinstance(frame, Frame):
            raise TypeError("frame must be a Frame")
        center_m = _quantity_array("center", center, u.m, shape=(3,))
        sigma_m = _scalar_quantity("sigma", sigma, u.m, positive=True)
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "_center_tuple_m", tuple(float(x) for x in center_m))
        object.__setattr__(self, "sigma_m", sigma_m)

    @property
    def center(self) -> u.Quantity:
        return _readonly_quantity(self._center_tuple_m, u.m)

    @property
    def center_m(self) -> np.ndarray:
        return np.asarray(self._center_tuple_m, dtype=np.float64)

    @property
    def sigma(self) -> u.Quantity:
        return self.sigma_m * u.m


@dataclass(frozen=True, eq=False, slots=True, init=False)
class TemperaturePerturbationGrid:
    """Rectilinear real Kelvin anomaly grid with explicit fixed support.

    The generic static-source route can be composed without importing private
    model helpers::

        source = grid.as_volume_element_source(rho0=rho0, T0=T0)
        model = point_sum(
            source, frame=grid.frame, field_definition=grid.field_definition,
            provenance=grid.point_sum_provenance(rho0=rho0, T0=T0),
        )
    """

    frame: Frame
    _x_edges_tuple_m: tuple[float, ...]
    _y_edges_tuple_m: tuple[float, ...]
    _z_edges_tuple_m: tuple[float, ...]
    _delta_temperature_bytes: bytes
    shape: tuple[int, int, int]
    _field_definition: GravityFieldDefinition

    def __init__(
        self,
        *,
        x_edges: u.Quantity,
        y_edges: u.Quantity,
        z_edges: u.Quantity,
        delta_temperature: u.Quantity,
        frame: Frame,
    ) -> None:
        if not isinstance(frame, Frame):
            raise TypeError("frame must be a Frame")
        temperatures = _quantity_array("delta_temperature", delta_temperature, u.K)
        if temperatures.ndim != 3:
            raise ValueError("delta_temperature must have shape (Nx, Ny, Nz)")
        if any(length <= 0 for length in temperatures.shape):
            raise ValueError("delta_temperature grid must contain cells on every axis")
        edges = tuple(
            _quantity_array(name, value, u.m)
            for name, value in (
                ("x_edges", x_edges),
                ("y_edges", y_edges),
                ("z_edges", z_edges),
            )
        )
        if any(axis.ndim != 1 for axis in edges):
            raise ValueError("each grid edge array must be one-dimensional")
        expected_edges = tuple(length + 1 for length in temperatures.shape)
        for name, axis, expected in zip(
            ("x_edges", "y_edges", "z_edges"), edges, expected_edges, strict=True
        ):
            if len(axis) != expected:
                raise ValueError(f"{name} must have length {expected}, got {len(axis)}")
            if not np.all(np.diff(axis) > 0.0):
                raise ValueError(f"{name} must be strictly increasing")
        temperatures = temperatures.copy()
        temperatures.setflags(write=False)
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "_x_edges_tuple_m", tuple(float(x) for x in edges[0]))
        object.__setattr__(self, "_y_edges_tuple_m", tuple(float(x) for x in edges[1]))
        object.__setattr__(self, "_z_edges_tuple_m", tuple(float(x) for x in edges[2]))
        object.__setattr__(self, "_delta_temperature_bytes", temperatures.tobytes())
        object.__setattr__(self, "shape", tuple(int(x) for x in temperatures.shape))
        object.__setattr__(
            self, "_field_definition", gen1_field_definition("temperature")
        )

    @property
    def x_edges(self) -> u.Quantity:
        return _readonly_quantity(self._x_edges_tuple_m, u.m)

    @property
    def y_edges(self) -> u.Quantity:
        return _readonly_quantity(self._y_edges_tuple_m, u.m)

    @property
    def z_edges(self) -> u.Quantity:
        return _readonly_quantity(self._z_edges_tuple_m, u.m)

    @property
    def delta_temperature(self) -> u.Quantity:
        values = np.frombuffer(self._delta_temperature_bytes, dtype=np.float64).reshape(
            self.shape
        )
        return _readonly_quantity(values, u.K)

    @property
    def field_definition(self) -> GravityFieldDefinition:
        return self._field_definition

    def point_sum_provenance(
        self,
        *,
        rho0: u.Quantity,
        T0: u.Quantity,  # noqa: N803 - public Gen1 spelling
    ) -> dict[str, object]:
        """Return finite provenance accepted by the existing point-sum route."""
        density = _scalar_quantity("rho0", rho0, u.kg / u.m**3, positive=True)
        temperature0 = _scalar_quantity("T0", T0, u.K, positive=True)
        return {
            "temperature_model": "isobaric_temperature_perturbation",
            "rho0_kg_m3": density,
            "T0_K": temperature0,
            "grid_shape": tuple(float(x) for x in self.shape),
            "x_edges_m": self._x_edges_tuple_m,
            "y_edges_m": self._y_edges_tuple_m,
            "z_edges_m": self._z_edges_tuple_m,
            "source_frame": self.frame.name,
            "support": "fixed_grid_edges",
            "discretization": "cell_center_volume_elements",
        }

    def as_volume_element_source(
        self,
        *,
        rho0: u.Quantity,
        T0: u.Quantity,  # noqa: N803 - public Gen1 spelling
    ) -> VolumeElementSource:
        """Convert cell centers to signed density-times-volume point masses."""
        density0 = _scalar_quantity("rho0", rho0, u.kg / u.m**3, positive=True)
        temperature0 = _scalar_quantity("T0", T0, u.K, positive=True)
        x_edges = np.asarray(self._x_edges_tuple_m, dtype=np.float64)
        y_edges = np.asarray(self._y_edges_tuple_m, dtype=np.float64)
        z_edges = np.asarray(self._z_edges_tuple_m, dtype=np.float64)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
        z_centers = (z_edges[:-1] + z_edges[1:]) / 2.0
        x_grid, y_grid, z_grid = np.meshgrid(
            x_centers, y_centers, z_centers, indexing="ij"
        )
        positions = np.column_stack(
            (x_grid.ravel(order="C"), y_grid.ravel(order="C"), z_grid.ravel(order="C"))
        )
        volumes = (
            np.diff(x_edges)[:, None, None]
            * np.diff(y_edges)[None, :, None]
            * np.diff(z_edges)[None, None, :]
        ).ravel(order="C")
        density = isobaric_temperature_density_perturbation(
            density0,
            np.frombuffer(self._delta_temperature_bytes, dtype=np.float64).reshape(
                self.shape
            ),
            temperature0,
        )
        return VolumeElementSource(
            positions_m=positions,
            volumes_m3=volumes,
            density_kg_m3=np.asarray(density).ravel(order="C"),
        )


__all__ = [
    "TemperatureFourierMode",
    "TemperatureGaussian",
    "TemperaturePerturbationGrid",
]
