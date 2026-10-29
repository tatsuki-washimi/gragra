"""High-level rigid-motion evaluation (M3-B H1)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.metadata import version
from platform import python_version
from types import MappingProxyType
from typing import TYPE_CHECKING, overload

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import QTable

from gragra._interactions import Newtonian
from gragra._time_matrix import make_time_matrix
from gragra.constants import G_SI
from gragra.geometry._frame import Frame
from gragra.gravity_sources.trajectory import PrescribedRigidTransform
from gragra.gravity_sources.weighted_points import PointMassCloud
from gragra.observables.rigid_motion import rigid_motion_observable
from gragra.results import (
    Contribution,
    GravityFieldDefinition,
    GravityResult,
    PotentialReference,
)
from gragra.results._context import EvaluationContext
from gragra.results._coords import normalize_gps
from gragra.results._evaluation import EvaluationResult
from gragra.results._record import OBSERVABLES, EvaluationRecord
from gragra.results._representation import TimeSeriesRepresentation
from gragra.results._time import TimeSampling
from gragra.results._time_grid import (
    DEFAULT_TIME_GRID_POLICY,
    UniformTimeGridPolicy,
    validate_representable_uniform_grid,
    validate_time_s_against_sampling,
)
from gragra.targets._high_points import TargetPoints

if TYPE_CHECKING:
    from gragra._continuous_contract import PoseFunction
    from gragra._continuous_fourier import ContinuousFourierModel

_COMPONENTS = {
    "potential": ("potential",),
    "acceleration": ("x", "y", "z"),
    "gradient": ("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"),
}
_UNITS = {
    "potential": u.m**2 / u.s**2,
    "acceleration": u.m / u.s**2,
    "gradient": u.s**-2,
}
_M3A_SHAPES = {
    "potential": (),
    "acceleration": (3,),
    "gradient": (3, 3),
}
_EPOCH_FORM_MESSAGE = (
    "epoch must be a LIGOTimeGPS, astropy Time, or (seconds, nanoseconds) "
    "tuple of integers"
)


def _observables(names: tuple[str, ...] | str) -> tuple[str, ...]:
    if isinstance(names, str):
        names = (names,)
    ordered = tuple(names)
    if (
        not ordered
        or any(name not in OBSERVABLES for name in ordered)
        or len(set(ordered)) != len(ordered)
    ):
        raise ValueError("observables must be a nonempty unique subset of known names")
    return ordered


def _default_field_definition(source: object) -> GravityFieldDefinition:
    return GravityFieldDefinition.source_field(
        type(source).__name__,
        Contribution("total"),
        PotentialReference.ZERO_AT_INFINITY,
    )


@dataclass(frozen=True, eq=False)
class RigidMotionModelSpec:
    """Physical interpretation for finite rigid-motion time series."""

    frame: Frame
    sampling: TimeSampling
    field_definition: GravityFieldDefinition
    interaction: Newtonian = field(default_factory=Newtonian)

    @property
    def parameters(self) -> QTable:
        table = QTable()
        table["motion_frame"] = [repr(self.frame)]
        table["sampling_epoch_gps_sec"] = [self.sampling.epoch_gps[0]]
        table["sampling_epoch_gps_nsec"] = [self.sampling.epoch_gps[1]]
        table["dt_s"] = [self.sampling.dt_seconds]
        table["sample_count"] = [self.sampling.count]
        return table

    @property
    def formulas(self) -> pd.DataFrame:
        return pd.DataFrame()

    @property
    def references(self) -> pd.DataFrame:
        return pd.DataFrame()


@dataclass(frozen=True, eq=False, init=False)
class RigidMotionModel:
    """Evaluate a static point source under prescribed rigid motion."""

    source: PointMassCloud
    transform: PrescribedRigidTransform
    spec: RigidMotionModelSpec
    _grid_policy: UniformTimeGridPolicy

    def __init__(
        self,
        source: PointMassCloud,
        transform: PrescribedRigidTransform,
        *,
        frame: Frame,
        sampling: TimeSampling,
        field_definition: GravityFieldDefinition | None = None,
        grid_policy: UniformTimeGridPolicy = DEFAULT_TIME_GRID_POLICY,
    ) -> None:
        if not isinstance(source, PointMassCloud):
            raise TypeError("source must be a PointMassCloud")
        if not isinstance(transform, PrescribedRigidTransform):
            raise TypeError("transform must be a PrescribedRigidTransform")
        if not isinstance(frame, Frame):
            raise TypeError("frame must be a Frame")
        if not isinstance(sampling, TimeSampling):
            raise TypeError("sampling must be a TimeSampling")
        if not isinstance(grid_policy, UniformTimeGridPolicy):
            raise TypeError("grid_policy must be a UniformTimeGridPolicy")
        definition = (
            _default_field_definition(source)
            if field_definition is None
            else field_definition
        )
        if not isinstance(definition, GravityFieldDefinition):
            raise TypeError("field_definition must be a GravityFieldDefinition")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "transform", transform)
        object.__setattr__(
            self,
            "spec",
            RigidMotionModelSpec(
                frame=frame,
                sampling=sampling,
                field_definition=definition,
            ),
        )
        object.__setattr__(self, "_grid_policy", grid_policy)

    def at(
        self,
        targets: TargetPoints,
        *,
        observables: tuple[str, ...] = OBSERVABLES,
        output_frame: Frame | None = None,
    ) -> RigidMotionEvaluation:
        return RigidMotionEvaluation(
            model=self,
            targets=targets,
            observables=_observables(observables),
            output_frame=output_frame,
        )


@dataclass(frozen=True, eq=False, init=False)
class RigidMotionEvaluation:
    """One fixed H1 request backed by the M3-A SI executor."""

    context: EvaluationContext
    computation_frame: Frame
    model: RigidMotionModel
    _target_positions_bytes: bytes
    _target_count: int

    def __init__(
        self,
        *,
        model: RigidMotionModel,
        targets: TargetPoints,
        observables: tuple[str, ...],
        output_frame: Frame | None = None,
    ) -> None:
        if not isinstance(model, RigidMotionModel):
            raise TypeError("model must be a RigidMotionModel")
        if not isinstance(targets, TargetPoints):
            raise TypeError("targets must be TargetPoints")
        if output_frame is None:
            output_frame = targets.frame
        if not isinstance(output_frame, Frame):
            raise TypeError("output_frame must be a Frame")
        sampling = model.spec.sampling
        if len(model.transform.time_s) != sampling.count:
            raise ValueError(
                "PrescribedRigidTransform frame count must match TimeSampling count"
            )
        validate_representable_uniform_grid(
            sampling.epoch_gps, sampling.dt_seconds, sampling.count
        )
        validate_time_s_against_sampling(
            model.transform.time_s,
            epoch_gps=sampling.epoch_gps,
            dt_seconds=sampling.dt_seconds,
            count=sampling.count,
            policy=model._grid_policy,
        )
        targets.frame._path_to(model.spec.frame)
        model.spec.frame._path_to(output_frame)
        traces = tuple(
            targets.frame.trace_point_m(point, model.spec.frame)
            for point in targets.positions_m
        )
        positions = np.stack([trace.point_m for trace in traces])
        context = EvaluationContext(
            model_spec=model.spec,
            targets=targets,
            frame=output_frame,
            observables=observables,
            representation=TimeSeriesRepresentation(),
            time_sampling=sampling,
        )
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "context", context)
        object.__setattr__(self, "computation_frame", model.spec.frame)
        object.__setattr__(self, "_target_positions_bytes", positions.copy().tobytes())
        object.__setattr__(self, "_target_count", len(positions))

    @property
    def target_positions_m(self) -> np.ndarray:
        return np.frombuffer(self._target_positions_bytes, dtype=np.float64).reshape(
            self._target_count, 3
        )

    @property
    def parameters(self) -> QTable:
        return self.model.spec.parameters

    @property
    def formulas(self) -> pd.DataFrame:
        return self.model.spec.formulas

    @property
    def references(self) -> pd.DataFrame:
        return self.model.spec.references

    def validate(self) -> None:
        self.context.validate()
        positions = self.target_positions_m
        if positions.shape != (len(self.context.targets), 3):
            raise ValueError("target coordinates are inconsistent with context")
        if len(positions) and not np.isfinite(positions).all():
            raise ValueError("target coordinates must be finite")

    def run(self) -> EvaluationResult:
        self.validate()
        sampling = self.context.time_sampling
        assert sampling is not None
        payloads: dict[str, object] = {}
        rotation = self.computation_frame.rotation_to(self.context.frame)
        for name in self.context.observables:
            dense = rigid_motion_observable(
                self.model.source,
                self.model.transform,
                self.target_positions_m,
                observable=name,
            )
            nct = _motion_target_to_nct(dense, name)
            if name == "potential":
                rotated = nct
            elif name == "acceleration":
                rotated = _rotate_vector_series(nct, rotation)
            else:
                rotated = _rotate_gradient_series(nct, rotation)
            payloads[name] = make_time_matrix(
                rotated,
                sampling=sampling,
                unit=_UNITS[name],
                rows=self.context.targets.ids,
                cols=_COMPONENTS[name],
                name=f"rigid_motion {name}",
            )
        record = EvaluationRecord(
            backend="m3a_rigid_motion_h1",
            versions=MappingProxyType(
                {
                    "gragra": version("gragra"),
                    "python": python_version(),
                    "numpy": np.__version__,
                    "astropy": version("astropy"),
                    "gwexpy": version("gwexpy"),
                }
            ),
            gravitational_constant_si=G_SI,
            g_applications=1,
        )
        return EvaluationResult(
            values=GravityResult(**payloads),
            evaluation=self,
            record=record,
        )


def _motion_target_to_nct(values: np.ndarray, observable: str) -> np.ndarray:
    """Map M3-A (motion, target, …) to (target, component, motion)."""
    tail = _M3A_SHAPES[observable]
    expected = (values.shape[0], values.shape[1]) + tail
    if values.shape != expected:
        raise ValueError(
            f"M3-A {observable} result has unexpected shape {values.shape}"
        )
    if observable == "potential":
        return np.transpose(values, (1, 0))[:, np.newaxis, :]
    if observable == "acceleration":
        return np.transpose(values, (1, 2, 0))
    n_target, n_motion = values.shape[1], values.shape[0]
    return np.transpose(values, (1, 2, 3, 0)).reshape(n_target, 9, n_motion)


def _rotate_vector_series(values: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Rotate ``(N, 3, T)`` acceleration components."""
    return np.einsum("ij,njt->nit", rotation, values)


def _rotate_gradient_series(values: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Rotate ``(N, 9, T)`` row-major gradient components."""
    count, _, samples = values.shape
    tensors = values.reshape(count, 3, 3, samples)
    rotated = np.einsum("ia,nabt,jb->nijt", rotation, tensors, rotation)
    return rotated.reshape(count, 9, samples)


@overload
def rigid_motion(
    source: PointMassCloud,
    transform: PrescribedRigidTransform,
    *,
    frame: Frame,
    sampling: TimeSampling,
    field_definition: GravityFieldDefinition | None = None,
    provenance: Mapping[str, object] | None = None,
) -> RigidMotionModel: ...


@overload
def rigid_motion(
    source: PointMassCloud,
    transform: PoseFunction,
    *,
    frame: Frame,
    epoch: object,
    field_definition: GravityFieldDefinition | None = None,
    provenance: Mapping[str, object] | None = None,
) -> ContinuousFourierModel: ...


def rigid_motion(
    source: PointMassCloud,
    transform: PrescribedRigidTransform | PoseFunction,
    *,
    frame: Frame,
    sampling: TimeSampling | None = None,
    epoch: object | None = None,
    field_definition: GravityFieldDefinition | None = None,
    provenance: Mapping[str, object] | None = None,
) -> RigidMotionModel | ContinuousFourierModel:
    """Create a rigid-motion model of a point-mass cloud.

    The second argument selects one of two forms.

    **Sampled form.** ``transform`` is a ``PrescribedRigidTransform``. Pass
    ``sampling`` (a ``TimeSampling``) and do not pass ``epoch``. The model is
    evaluated with ``.at(targets).run()``, which returns a time-series
    ``EvaluationResult``.

    **Continuous form.** ``transform`` is a callable pose. It takes a
    ``Quantity`` of times in seconds, relative to ``epoch``, and returns
    ``(Rotation, Quantity[m] of shape (N, 3))``, with one rotation per time.
    Pass ``epoch`` (a ``LIGOTimeGPS``, an astropy ``Time``, or a
    ``(seconds, nanoseconds)`` tuple of integers, giving the absolute GPS
    time of relative time zero) and do not pass
    ``sampling``. The model is evaluated with ``.at(targets)`` and then
    ``.fourier_series(...)`` or ``.fourier_transform(...)``.

    Any other combination of arguments raises ``TypeError``.

    Parameters
    ----------
    source : PointMassCloud
        Positive point masses. Their positions are body coordinates in metres.
    transform : PrescribedRigidTransform or callable pose
        Rigid motion of the body. The body-to-frame map is
        ``x = R x_body + c``.
    frame : Frame
        Computation frame. The output is expressed in this frame, and the
        pose maps body coordinates into it.
    sampling : TimeSampling, optional
        Required for the sampled form only.
    epoch : LIGOTimeGPS, astropy Time, or (int, int), optional
        Required for the continuous form only. It is the absolute GPS time of
        pose relative time zero.
    field_definition : GravityFieldDefinition, optional
        Field metadata. The default is the total field of the source.
    provenance : Mapping, optional
        Not supported. Passing a value raises ``TypeError``.

    Notes
    -----
    **Units and phase.** Times are in seconds, positions in metres, and
    accelerations in m/s**2. Fourier coefficients use the convention
    ``exp[-2*pi*i*f*(t - tref)]``. ``tref`` is an explicit argument of the
    Fourier methods and has no default, so the epoch is never used as an
    implicit phase reference.

    **Continuous-form Fourier methods.** ``fourier_series`` covers periodic
    motion only (periodic integrator). Its amplitudes satisfy ``A_0 = C_0`` and
    ``A_n = 2 C_n``. ``fourier_transform`` integrates a finite window
    (finite-window integrator). Its coefficients are one-sided and not doubled.
    The result is limited by ``tau_min`` and by the finite sampling, which
    together control aliasing. For ``fourier_transform`` the phase range must
    satisfy ``f_max * |epoch - tref| <= 1e6`` cycles. Place ``tref`` near the
    window.

    **Stability.** The return type of the continuous form is provisional, and
    its signature may change in future.

    **Gravitational constant.** G is applied exactly once, in the observable
    layer. The kernels are G-free.

    **Complex outputs.** Complex Fourier coefficients are linear complex
    amplitudes or transfer-function-like responses. They are not real-valued
    time-domain gravitational fields.

    **Memory.** ``memory_budget_bytes`` is a per-lane budget for the arrays
    managed by the library. It is not a process RSS limit, and it is not
    summed across lanes.
    """
    if provenance is not None:
        raise TypeError("provenance is not supported for rigid_motion")
    if isinstance(transform, PrescribedRigidTransform):
        if epoch is not None:
            raise TypeError(
                "epoch applies only to a callable pose; a PrescribedRigidTransform "
                "uses sampling"
            )
        if sampling is None:
            raise TypeError("sampling is required for a PrescribedRigidTransform")
        return RigidMotionModel(
            source,
            transform,
            frame=frame,
            sampling=sampling,
            field_definition=field_definition,
        )
    if callable(transform):
        if sampling is not None:
            raise TypeError(
                "sampling applies only to a PrescribedRigidTransform; a callable "
                "pose uses epoch"
            )
        if epoch is None:
            raise TypeError("epoch is required for a callable pose")
        # Validate the GPS form early, so a bad epoch fails at construction.
        # The original error names reference_time, so re-raise it under the
        # public argument name and keep the original exception type.
        try:
            normalize_gps(epoch)
        except TypeError as exc:
            raise TypeError(_EPOCH_FORM_MESSAGE) from exc
        except ValueError as exc:
            raise ValueError(_EPOCH_FORM_MESSAGE) from exc
        # Imported lazily so the sampled form does not load the continuous path.
        from gragra._continuous_fourier import ContinuousFourierModel

        return ContinuousFourierModel(
            source,
            transform,
            frame=frame,
            field_definition=field_definition,
            epoch=epoch,
        )
    raise TypeError("transform must be a PrescribedRigidTransform or a callable pose")


__all__ = [
    "RigidMotionEvaluation",
    "RigidMotionModel",
    "RigidMotionModelSpec",
    "rigid_motion",
]
