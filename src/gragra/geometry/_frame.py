"""Owned, static Cartesian frames for the C5 high-level gravity API."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass

import numpy as np
from astropy import units as u

from gragra._arrays import _check_no_bool


def _lengths_m(name: str, value: u.Quantity, shape: tuple[int, ...]) -> np.ndarray:
    if not isinstance(value, u.Quantity):
        raise TypeError(f"{name} must be an astropy Quantity with length units")
    _check_no_bool(value, name)
    raw = np.asarray(value.to_value(u.m))
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain only real coordinates")
    array = np.asarray(raw, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite coordinates")
    return array.copy()


def _xyz_m(name: str, value: object) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain only real coordinates")
    array = np.asarray(raw, dtype=np.float64)
    if array.shape[-1:] != (3,) or array.ndim not in (1, 2):
        raise ValueError(f"{name} must have shape (3,) or (N, 3)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite coordinates")
    return array


@dataclass(frozen=True)
class PointTrace:
    """One target's actual path into a destination frame, in SI meters.

    Coordinates include both endpoints and every intermediate coordinate.
    Origins are the parent-coordinate origin used on each traversed edge.
    Tuples own scalar values, so no caller can mutate the trace.
    """

    coordinates_m: tuple[tuple[float, float, float], ...]
    origins_m: tuple[tuple[float, float, float], ...]
    edge_count: int

    @property
    def point_m(self) -> np.ndarray:
        return np.asarray(self.coordinates_m[-1], dtype=np.float64)

    @property
    def scale_m(self) -> float:
        values = (*self.coordinates_m, *self.origins_m)
        return max((max(abs(x) for x in vector) for vector in values), default=0.0)


@dataclass(frozen=True, eq=False, init=False)
class Frame:
    """Static right-handed Cartesian frame, relative to its explicit parent.

    ``origin`` is a length Quantity in the parent's coordinates. Columns of
    ``basis`` are this frame's unit axes expressed in the parent. A frame with
    no parent is a distinct root and has identity origin/basis. Equal names do
    not connect independent roots.
    """

    name: str
    parent: Frame | None
    _origin_tuple_m: tuple[float, float, float]
    _basis_tuple: tuple[tuple[float, float, float], ...]

    def __init__(
        self,
        name: str,
        origin: u.Quantity | None = None,
        basis: np.ndarray | u.Quantity | None = None,
        parent: Frame | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Frame name must be a nonempty string")
        if parent is not None and not isinstance(parent, Frame):
            raise TypeError("parent must be a Frame or None")
        origin_m = (
            np.zeros(3, dtype=np.float64)
            if origin is None
            else _lengths_m("origin", origin, (3,))
        )
        if basis is None:
            raw_axes = np.eye(3)
        else:
            # Preserve NumPy's existing failure for recursive containers.
            with suppress(RecursionError):
                _check_no_bool(basis, "basis")
            if isinstance(basis, u.Quantity):
                raw_axes = np.asarray(basis.to_value(u.dimensionless_unscaled))
            else:
                raw_axes = np.asarray(basis)
        if np.iscomplexobj(raw_axes):
            raise ValueError("basis must contain only real values")
        axes = np.asarray(raw_axes, dtype=np.float64)
        if axes.shape != (3, 3):
            raise ValueError(f"basis must have shape (3, 3), got {axes.shape}")
        if not np.isfinite(axes).all():
            raise ValueError("basis must contain only finite values")
        if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-12, rtol=0):
            raise ValueError("basis must be orthogonal within atol=1e-12")
        if np.linalg.det(axes) <= 0:
            raise ValueError("basis must be right-handed (determinant > 0)")
        if parent is None and (
            np.any(origin_m != 0) or not np.array_equal(axes, np.eye(3))
        ):
            raise ValueError("a root Frame must have zero origin and identity basis")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "parent", parent)
        object.__setattr__(self, "_origin_tuple_m", tuple(float(x) for x in origin_m))
        object.__setattr__(
            self, "_basis_tuple", tuple(tuple(float(x) for x in row) for row in axes)
        )

    @property
    def origin_m(self) -> np.ndarray:
        return np.asarray(self._origin_tuple_m, dtype=np.float64)

    @property
    def origin(self) -> u.Quantity:
        return self.origin_m * u.m

    @property
    def basis(self) -> np.ndarray:
        return np.asarray(self._basis_tuple, dtype=np.float64)

    @property
    def root(self) -> Frame:
        current = self
        while current.parent is not None:
            current = current.parent
        return current

    def _path_to(self, destination: Frame) -> tuple[list[Frame], list[Frame]]:
        if not isinstance(destination, Frame):
            raise TypeError("destination must be a Frame")
        own_chain: list[Frame] = []
        current: Frame | None = self
        while current is not None:
            own_chain.append(current)
            current = current.parent
        destination_chain: list[Frame] = []
        current = destination
        while current is not None:
            destination_chain.append(current)
            current = current.parent
        own_ids = {id(frame): index for index, frame in enumerate(own_chain)}
        match = next(
            (j for j, frame in enumerate(destination_chain) if id(frame) in own_ids),
            None,
        )
        if match is None:
            raise ValueError("Frames have disconnected roots")
        ancestor = destination_chain[match]
        return own_chain[: own_ids[id(ancestor)]], list(
            reversed(destination_chain[:match])
        )

    def trace_point_m(self, point_m: object, destination: Frame) -> PointTrace:
        """Convert one SI point through the shortest common-ancestor path."""
        point = _xyz_m("point_m", point_m)
        if point.shape != (3,):
            raise ValueError("point_m must have shape (3,)")
        up, down = self._path_to(destination)
        current = point.copy()
        coordinates = [tuple(float(x) for x in current)]
        origins: list[tuple[float, float, float]] = []
        for frame in up:
            origin = frame.origin_m
            with np.errstate(over="ignore", invalid="ignore"):
                current = origin + frame.basis @ current
            if not np.isfinite(current).all():
                raise ValueError("Frame transform produced non-finite coordinates")
            origins.append(tuple(float(x) for x in origin))
            coordinates.append(tuple(float(x) for x in current))
        for frame in down:
            origin = frame.origin_m
            with np.errstate(over="ignore", invalid="ignore"):
                current = np.linalg.solve(frame.basis, current - origin)
            if not np.isfinite(current).all():
                raise ValueError("Frame transform produced non-finite coordinates")
            origins.append(tuple(float(x) for x in origin))
            coordinates.append(tuple(float(x) for x in current))
        return PointTrace(tuple(coordinates), tuple(origins), len(up) + len(down))

    def transform_points_m(self, points_m: object, destination: Frame) -> np.ndarray:
        """Convert one or N SI points to ``destination`` without using root detours."""
        points = _xyz_m("points_m", points_m)
        if points.ndim == 1:
            return self.trace_point_m(points, destination).point_m
        if len(points) == 0:
            self._path_to(destination)
            return np.empty((0, 3), dtype=np.float64)
        return np.stack(
            [self.trace_point_m(point, destination).point_m for point in points]
        )

    def rotation_to(self, destination: Frame) -> np.ndarray:
        """Return the component rotation from this basis to ``destination``."""
        up, down = self._path_to(destination)
        rotation = np.eye(3)
        for frame in up:
            rotation = frame.basis @ rotation
        for frame in down:
            rotation = frame.basis.T @ rotation
        return rotation

    def transform_vectors_m(self, vectors_m: object, destination: Frame) -> np.ndarray:
        vectors = _xyz_m("vectors_m", vectors_m)
        with np.errstate(over="ignore", invalid="ignore"):
            transformed = vectors @ self.rotation_to(destination).T
        if not np.isfinite(transformed).all():
            raise ValueError("Frame vector transform produced non-finite components")
        return transformed

    def transform_tensors(self, tensors: object, destination: Frame) -> np.ndarray:
        raw = np.asarray(tensors)
        if np.iscomplexobj(raw):
            raise ValueError("tensors must contain only real values")
        array = np.asarray(raw, dtype=np.float64)
        if array.shape[-2:] != (3, 3) or array.ndim not in (2, 3):
            raise ValueError("tensors must have shape (3, 3) or (N, 3, 3)")
        if not np.isfinite(array).all():
            raise ValueError("tensors must contain only finite values")
        rotation = self.rotation_to(destination)
        with np.errstate(over="ignore", invalid="ignore"):
            transformed = np.einsum("ij,...jk,lk->...il", rotation, array, rotation)
        if not np.isfinite(transformed).all():
            raise ValueError("Frame tensor transform produced non-finite components")
        return transformed
