"""Connect continuous rigid-motion responses to spectral ``EvaluationResult`` objects.

This private adapter wires the P3 pose response into the P4 periodic and P5
finite-window Fourier integrators and packages the coefficients as
``EvaluationResult`` payloads with the C3 spectral descriptors.

**Reproducibility.** ``parameters`` records quadrature settings (interval,
period, orders or frequencies, ``epoch``, ``tref``, ``tau_min``, tolerances,
``max_nodes`` or ``max_evaluations``, and the managed-array budget). It does
**not** record the pose callable, the ``PointMassCloud`` definition, or the
diagnostic dict returned alongside the result. Those are still required to
reproduce the physics in general.

**Diagnostics.** :meth:`ContinuousPeriodicFourierEvaluation.run_with_diagnostics`
and :meth:`ContinuousWindowFourierEvaluation.run_with_diagnostics` return
``(EvaluationResult, dict[str, dict])`` mapping each observable name to the
P4/P5 diagnostic dict (keys in ``DIAGNOSTIC_KEYS``).

**Memory budget.** ``memory_budget_bytes`` is applied per lane. The same value
is passed to the P3 pose response, to the P4 or P5 integrator, and each lane
checks its own managed-array estimate against it. The budgets are not summed.
The budget counts the arrays that the library manages, not the process RSS.
Temporary arrays created inside the user's pose callable are outside the P4
and P5 estimates. During one P5 callback, the P3 call-time arrays exist
alongside the P5 arrays, so the lane budgets do not bound their total.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.metadata import version
from platform import python_version
from types import MappingProxyType

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import QTable
from gwexpy.frequencyseries import FrequencySeriesMatrix

from gragra._continuous_contract import (
    PoseFunction,
    component_shape,
    validate_observable,
)
from gragra._continuous_rigid_motion import make_rigid_pose_response
from gragra._finite_window_fourier import _tau_min_seconds, finite_window_fourier
from gragra._interactions import Newtonian
from gragra._periodic_fourier import periodic_fourier
from gragra.constants import G_SI
from gragra.geometry._frame import Frame
from gragra.gravity_sources.weighted_points import PointMassCloud
from gragra.results import (
    Contribution,
    EvaluationContext,
    EvaluationRecord,
    EvaluationResult,
    GravityFieldDefinition,
    GravityResult,
    PotentialReference,
)
from gragra.results._coords import normalize_gps
from gragra.results._record import OBSERVABLES
from gragra.results._spectral import (
    AmplitudeSpectrumRepresentation,
    FiniteRecordInterpretation,
    FourierTransformRepresentation,
    Sidedness,
)
from gragra.targets._high_points import TargetPoints

DEFAULT_MAX_EVALUATIONS = 100_000
"""Default callback budget for one P5 integral (integration lane, Step 4b)."""

DEFAULT_MEMORY_BUDGET_BYTES = 64 * 1024**2
"""Default managed-array budget (matches ``continuous_acceptance.json``)."""

DEFAULT_P5_CACHE_SIZE_BYTES = 8 * 1024**2

_COMPONENTS = {
    "potential": ("potential",),
    "acceleration": ("x", "y", "z"),
    "gradient": ("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"),
}


def _default_field_definition(source: object) -> GravityFieldDefinition:
    return GravityFieldDefinition.source_field(
        type(source).__name__,
        Contribution("total"),
        PotentialReference.ZERO_AT_INFINITY,
    )


@dataclass(frozen=True, eq=False)
class ContinuousFourierModelSpec:
    """Physical interpretation for continuous Fourier evaluation."""

    frame: Frame
    field_definition: GravityFieldDefinition
    interaction: Newtonian = field(default_factory=Newtonian)

    @property
    def parameters(self) -> QTable:
        table = QTable()
        table["computation_frame"] = [repr(self.frame)]
        return table

    @property
    def formulas(self) -> pd.DataFrame:
        return pd.DataFrame()

    @property
    def references(self) -> pd.DataFrame:
        return pd.DataFrame()


@dataclass(frozen=True, eq=False, init=False)
class ContinuousFourierModel:
    """One rigid source with a continuous body pose.

    ``epoch`` is the absolute GPS time of the pose's relative time zero. It is
    optional here and is required only by :meth:`at`, which the public
    ``rigid_motion`` factory uses for callable poses.

    ``softening_m`` is the Plummer softening length in metres. It must be finite
    and nonnegative. The public ``rigid_motion`` factory does not pass it, so the
    public API fixes the softening at 0.
    """

    source: PointMassCloud
    pose: PoseFunction
    spec: ContinuousFourierModelSpec
    softening_m: float
    epoch: object | None

    def __init__(
        self,
        source: PointMassCloud,
        pose: PoseFunction,
        *,
        frame: Frame,
        field_definition: GravityFieldDefinition | None = None,
        softening_m: float = 0.0,
        epoch: object | None = None,
    ) -> None:
        if not isinstance(source, PointMassCloud):
            raise TypeError("source must be a PointMassCloud")
        if not isinstance(frame, Frame):
            raise TypeError("frame must be a Frame")
        definition = (
            _default_field_definition(source)
            if field_definition is None
            else field_definition
        )
        if not isinstance(definition, GravityFieldDefinition):
            raise TypeError("field_definition must be a GravityFieldDefinition")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "pose", pose)
        object.__setattr__(
            self,
            "spec",
            ContinuousFourierModelSpec(frame=frame, field_definition=definition),
        )
        softening = float(softening_m)
        if not (math.isfinite(softening) and softening >= 0.0):
            raise ValueError(
                f"softening_m must be finite and nonnegative, got {softening_m!r}"
            )
        object.__setattr__(self, "softening_m", softening)
        object.__setattr__(self, "epoch", epoch)

    def at(
        self,
        targets: TargetPoints,
        *,
        observables: tuple[str, ...] = ("acceleration",),
        output_frame: Frame | None = None,
    ) -> ContinuousRigidMotionEvaluation:
        """Return the short evaluation view used by ``rigid_motion`` results.

        **Only the computation frame is supported in the continuous form.**
        ``output_frame`` must be None. Passing any other value raises
        ``ValueError``, and the returned view does not change the output frame.
        """
        return ContinuousRigidMotionEvaluation(
            model=self,
            targets=targets,
            observables=observables,
            output_frame=output_frame,
        )

    def periodic_at(
        self,
        targets: TargetPoints,
        *,
        epoch: object,
        tref: object,
        period: u.Quantity,
        orders: object,
        tau_min: u.Quantity,
        atol: u.Quantity,
        rtol: float,
        max_nodes: int,
        memory_budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES,
        observables: tuple[str, ...] = ("acceleration",),
    ) -> ContinuousPeriodicFourierEvaluation:
        """Return a periodic Fourier evaluation of the rigid body.

        ``memory_budget_bytes`` is a per-lane budget for managed arrays. The same
        value goes to the P3 pose response and to the P4 integrator. Each lane
        checks it separately, and the budgets are not summed. It is not a process
        RSS limit, and temporary arrays inside the pose callable are not counted.
        See the module docstring for details.
        """
        return ContinuousPeriodicFourierEvaluation(
            model=self,
            targets=targets,
            observables=_observables(observables),
            epoch=epoch,
            tref=tref,
            period=period,
            orders=orders,
            tau_min=tau_min,
            atol=atol,
            rtol=rtol,
            max_nodes=max_nodes,
            memory_budget_bytes=memory_budget_bytes,
        )

    def window_at(
        self,
        targets: TargetPoints,
        *,
        epoch: object,
        tref: object,
        t_start: u.Quantity,
        t_stop: u.Quantity,
        frequencies: u.Quantity,
        tau_min: u.Quantity,
        atol: u.Quantity,
        rtol: float,
        limit: int | None = None,
        cache_size: int = DEFAULT_P5_CACHE_SIZE_BYTES,
        max_evaluations: int = DEFAULT_MAX_EVALUATIONS,
        memory_budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES,
        observables: tuple[str, ...] = ("acceleration",),
    ) -> ContinuousWindowFourierEvaluation:
        """Return a finite-window Fourier evaluation of the rigid body.

        ``memory_budget_bytes`` is a per-lane budget for managed arrays. The same
        value goes to the P3 pose response and to the P5 integrator. Each lane
        checks it separately, and the budgets are not summed. It is not a process
        RSS limit, and temporary arrays inside the pose callable are not counted.
        See the module docstring for details.
        """
        return ContinuousWindowFourierEvaluation(
            model=self,
            targets=targets,
            observables=_observables(observables),
            epoch=epoch,
            tref=tref,
            t_start=t_start,
            t_stop=t_stop,
            frequencies=frequencies,
            tau_min=tau_min,
            atol=atol,
            rtol=rtol,
            limit=limit,
            cache_size=cache_size,
            max_evaluations=max_evaluations,
            memory_budget_bytes=memory_budget_bytes,
        )


def _observables(names: tuple[str, ...]) -> tuple[str, ...]:
    ordered = tuple(names)
    if (
        not ordered
        or any(name not in OBSERVABLES for name in ordered)
        or len(set(ordered)) != len(ordered)
    ):
        raise ValueError("observables must be a nonempty unique subset of known names")
    return ordered


def _orders_array(orders: object) -> np.ndarray:
    arr = np.asarray(orders, dtype=np.int64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("orders must be a one-dimensional nonempty array")
    if not np.all(arr >= 0):
        raise ValueError("orders must be nonnegative")
    return arr


def _frequency_hz_from_orders(orders: np.ndarray, period_s: float) -> np.ndarray:
    return orders.astype(np.float64) / period_s


def _coefficients_to_matrix(
    coefficients: u.Quantity,
    *,
    orders: np.ndarray | None,
    frequencies_hz: np.ndarray,
    target_ids: tuple[str, ...],
    observable: str,
    amplitude: bool,
) -> FrequencySeriesMatrix:
    """Map ``(Nf, Ntarget, *component)`` to GWexpy ``(target, component, f)``."""
    name = validate_observable(observable)
    values = np.asarray(coefficients.value, dtype=np.complex128)
    comp = component_shape(name)
    expected = (len(frequencies_hz), len(target_ids)) + comp
    if values.shape != expected:
        raise ValueError(
            f"{name} coefficient shape {values.shape} != expected {expected}"
        )
    if amplitude:
        if orders is None:
            raise ValueError("orders is required when amplitude=True")
        # One-sided amplitudes: A_n = 2 C_n for n > 0, A_0 = C_0.
        scaled = values.copy()
        for row, order in enumerate(orders):
            if int(order) > 0:
                scaled[row] *= 2.0
        values = scaled
    unit = coefficients.unit
    # Flatten the trailing component axes row-major (gradient: xx, xy, ..., zz).
    flat = values.reshape(values.shape[0], values.shape[1], -1)
    matrix = np.transpose(flat, (1, 2, 0))
    n_target, n_comp, _ = matrix.shape
    out_units = np.empty((n_target, n_comp), dtype=object)
    out_units[...] = unit
    return FrequencySeriesMatrix(
        matrix,
        frequencies=frequencies_hz * u.Hz,
        units=out_units,
        rows=list(target_ids),
        cols=list(_COMPONENTS[name]),
        name=f"continuous_{'amplitude' if amplitude else 'fourier'}_{name}",
    )


def _evaluation_record() -> EvaluationRecord:
    return EvaluationRecord(
        backend="continuous_fourier",
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


@dataclass(frozen=True, eq=False, init=False)
class ContinuousPeriodicFourierEvaluation:
    """One periodic Fourier request backed by P3 and P4."""

    context: EvaluationContext
    model: ContinuousFourierModel
    _request: Mapping[str, object]

    def __init__(
        self,
        *,
        model: ContinuousFourierModel,
        targets: TargetPoints,
        observables: tuple[str, ...],
        epoch: object,
        tref: object,
        period: u.Quantity,
        orders: object,
        tau_min: u.Quantity,
        atol: u.Quantity,
        rtol: float,
        max_nodes: int,
        memory_budget_bytes: int,
    ) -> None:
        if not isinstance(model, ContinuousFourierModel):
            raise TypeError("model must be a ContinuousFourierModel")
        if not isinstance(targets, TargetPoints):
            raise TypeError("targets must be TargetPoints")
        order_arr = _orders_array(orders)
        period_s = float(period.to_value(u.s))
        freqs = _frequency_hz_from_orders(order_arr, period_s)
        representation = AmplitudeSpectrumRepresentation(
            record_interpretation=FiniteRecordInterpretation.PERIODIC
        )
        context = EvaluationContext(
            model_spec=model.spec,
            targets=targets,
            frame=model.spec.frame,
            observables=observables,
            representation=representation,
            frequencies=freqs * u.Hz,
            reference_time=tref,
        )
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "context", context)
        object.__setattr__(
            self,
            "_request",
            MappingProxyType(
                {
                    "epoch": epoch,
                    "tref": tref,
                    "period": period,
                    "orders": order_arr,
                    "tau_min": tau_min,
                    "atol": atol,
                    "rtol": float(rtol),
                    "max_nodes": int(max_nodes),
                    "memory_budget_bytes": int(memory_budget_bytes),
                }
            ),
        )

    @property
    def parameters(self) -> QTable:
        req = self._request
        table = QTable()
        epoch = normalize_gps(req["epoch"])
        tref = normalize_gps(req["tref"])
        table["epoch_gps_sec"] = [epoch[0]]
        table["epoch_gps_nsec"] = [epoch[1]]
        table["tref_gps_sec"] = [tref[0]]
        table["tref_gps_nsec"] = [tref[1]]
        table["period_s"] = [req["period"].to_value(u.s)]
        table["orders"] = [",".join(str(int(n)) for n in req["orders"])]
        table["tau_min_s"] = [req["tau_min"].to_value(u.s)]
        table["rtol"] = [req["rtol"]]
        table["max_nodes"] = [req["max_nodes"]]
        table["memory_budget_bytes"] = [req["memory_budget_bytes"]]
        table["mode"] = ["periodic"]
        # Appended last, so the existing columns keep their names and order.
        # Shape (1, *atol.shape) keeps a per-target, per-component tolerance.
        table["atol"] = req["atol"][np.newaxis, ...]
        return table

    @property
    def formulas(self) -> pd.DataFrame:
        return pd.DataFrame()

    @property
    def references(self) -> pd.DataFrame:
        return pd.DataFrame()

    def validate(self) -> None:
        self.context.validate()

    def run(self) -> EvaluationResult:
        result, _ = self.run_with_diagnostics()
        return result

    def run_with_diagnostics(self) -> tuple[EvaluationResult, dict[str, dict]]:
        self.validate()
        req = self._request
        response = make_rigid_pose_response(
            self.model.source,
            self.model.pose,
            self.context.targets,
            frame=self.model.spec.frame,
            memory_budget_bytes=req["memory_budget_bytes"],
            softening_m=self.model.softening_m,
        )
        payloads: dict[str, FrequencySeriesMatrix] = {}
        diagnostics: dict[str, dict] = {}
        target_ids = tuple(self.context.targets.ids)
        orders = req["orders"]
        period_s = float(req["period"].to_value(u.s))
        freqs = _frequency_hz_from_orders(orders, period_s)
        for name in self.context.observables:
            coeff, diag = periodic_fourier(
                response,
                name,
                epoch=req["epoch"],
                tref=req["tref"],
                period=req["period"],
                orders=orders,
                tau_min=req["tau_min"],
                atol=req["atol"],
                rtol=req["rtol"],
                max_nodes=req["max_nodes"],
                memory_budget_bytes=req["memory_budget_bytes"],
            )
            diagnostics[name] = diag
            payloads[name] = _coefficients_to_matrix(
                coeff,
                orders=orders,
                frequencies_hz=freqs,
                target_ids=target_ids,
                observable=name,
                amplitude=True,
            )
        result = EvaluationResult(
            GravityResult(name="continuous periodic fourier", **payloads),
            evaluation=self,
            record=_evaluation_record(),
            context=self.context,
        )
        return result, diagnostics


@dataclass(frozen=True, eq=False, init=False)
class ContinuousWindowFourierEvaluation:
    """One finite-window Fourier request backed by P3 and P5."""

    context: EvaluationContext
    model: ContinuousFourierModel
    _request: Mapping[str, object]

    def __init__(
        self,
        *,
        model: ContinuousFourierModel,
        targets: TargetPoints,
        observables: tuple[str, ...],
        epoch: object,
        tref: object,
        t_start: u.Quantity,
        t_stop: u.Quantity,
        frequencies: u.Quantity,
        tau_min: u.Quantity,
        atol: u.Quantity,
        rtol: float,
        limit: int | None,
        cache_size: int,
        max_evaluations: int,
        memory_budget_bytes: int,
    ) -> None:
        if not isinstance(model, ContinuousFourierModel):
            raise TypeError("model must be a ContinuousFourierModel")
        if not isinstance(targets, TargetPoints):
            raise TypeError("targets must be TargetPoints")
        representation = FourierTransformRepresentation(sidedness=Sidedness.ONE_SIDED)
        context = EvaluationContext(
            model_spec=model.spec,
            targets=targets,
            frame=model.spec.frame,
            observables=observables,
            representation=representation,
            frequencies=frequencies,
            reference_time=tref,
        )
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "context", context)
        object.__setattr__(
            self,
            "_request",
            MappingProxyType(
                {
                    "epoch": epoch,
                    "tref": tref,
                    "t_start": t_start,
                    "t_stop": t_stop,
                    "frequencies": frequencies,
                    "tau_min": tau_min,
                    "atol": atol,
                    "rtol": float(rtol),
                    "limit": limit,
                    "cache_size": int(cache_size),
                    "max_evaluations": int(max_evaluations),
                    "memory_budget_bytes": int(memory_budget_bytes),
                }
            ),
        )

    @property
    def parameters(self) -> QTable:
        req = self._request
        epoch = normalize_gps(req["epoch"])
        tref = normalize_gps(req["tref"])
        table = QTable()
        table["epoch_gps_sec"] = [epoch[0]]
        table["epoch_gps_nsec"] = [epoch[1]]
        table["tref_gps_sec"] = [tref[0]]
        table["tref_gps_nsec"] = [tref[1]]
        table["t_start_s"] = [req["t_start"].to_value(u.s)]
        table["t_stop_s"] = [req["t_stop"].to_value(u.s)]
        table["frequencies_hz"] = [
            ",".join(str(float(f)) for f in req["frequencies"].to_value(u.Hz))
        ]
        table["tau_min_s"] = [req["tau_min"].to_value(u.s)]
        table["rtol"] = [req["rtol"]]
        table["max_evaluations"] = [req["max_evaluations"]]
        table["memory_budget_bytes"] = [req["memory_budget_bytes"]]
        table["mode"] = ["finite_window"]
        if req["limit"] is not None:
            table["limit"] = [req["limit"]]
        # Appended last, so the existing columns keep their names and order.
        # Shape (1, *atol.shape) keeps a per-target, per-component tolerance.
        table["atol"] = req["atol"][np.newaxis, ...]
        return table

    @property
    def formulas(self) -> pd.DataFrame:
        return pd.DataFrame()

    @property
    def references(self) -> pd.DataFrame:
        return pd.DataFrame()

    def validate(self) -> None:
        self.context.validate()
        # _default_limit divides by tau_min before finite_window_fourier runs,
        # so an invalid tau_min must be rejected here (blind v2 N-04, R3).
        _tau_min_seconds(self._request["tau_min"])

    def run(self) -> EvaluationResult:
        result, _ = self.run_with_diagnostics()
        return result

    def run_with_diagnostics(self) -> tuple[EvaluationResult, dict[str, dict]]:
        self.validate()
        req = self._request
        response = make_rigid_pose_response(
            self.model.source,
            self.model.pose,
            self.context.targets,
            frame=self.model.spec.frame,
            memory_budget_bytes=req["memory_budget_bytes"],
            softening_m=self.model.softening_m,
        )
        limit = req["limit"]
        payloads: dict[str, FrequencySeriesMatrix] = {}
        diagnostics: dict[str, dict] = {}
        target_ids = tuple(self.context.targets.ids)
        freqs = np.asarray(req["frequencies"].to_value(u.Hz), dtype=np.float64)
        for name in self.context.observables:
            coeff, diag = finite_window_fourier(
                response,
                name,
                epoch=req["epoch"],
                tref=req["tref"],
                t_start=req["t_start"],
                t_stop=req["t_stop"],
                frequencies=req["frequencies"],
                tau_min=req["tau_min"],
                atol=req["atol"],
                rtol=req["rtol"],
                limit=limit if limit is not None else _default_limit(req),
                cache_size=req["cache_size"],
                max_evaluations=req["max_evaluations"],
                memory_budget_bytes=req["memory_budget_bytes"],
            )
            diagnostics[name] = diag
            payloads[name] = _coefficients_to_matrix(
                coeff,
                orders=None,
                frequencies_hz=freqs,
                target_ids=target_ids,
                observable=name,
                amplitude=False,
            )
        result = EvaluationResult(
            GravityResult(name="continuous window fourier", **payloads),
            evaluation=self,
            record=_evaluation_record(),
            context=self.context,
        )
        return result, diagnostics


@dataclass(frozen=True, eq=False, init=False)
class ContinuousRigidMotionEvaluation:
    """Fixed-target view over a continuous rigid-motion model.

    The methods delegate to :meth:`ContinuousFourierModel.periodic_at` and
    :meth:`ContinuousFourierModel.window_at`, so they use the same periodic and
    finite-window integrators, diagnostics, and budgets. Each method returns an
    ``EvaluationResult``.
    """

    model: ContinuousFourierModel
    targets: TargetPoints
    observables: tuple[str, ...]

    def __init__(
        self,
        *,
        model: ContinuousFourierModel,
        targets: TargetPoints,
        observables: tuple[str, ...],
        output_frame: Frame | None,
    ) -> None:
        if not isinstance(model, ContinuousFourierModel):
            raise TypeError("model must be a ContinuousFourierModel")
        if not isinstance(targets, TargetPoints):
            raise TypeError("targets must be TargetPoints")
        if output_frame is not None:
            raise ValueError(
                "continuous rigid-motion evaluation supports only the computation "
                "frame; output_frame must be None"
            )
        if model.epoch is None:
            raise ValueError("continuous rigid-motion evaluation requires an epoch")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "observables", _observables(observables))

    def fourier_series(
        self,
        *,
        period: u.Quantity,
        orders: object,
        tref: object,
        tau_min: u.Quantity,
        atol: u.Quantity,
        rtol: float = 1e-10,
        max_nodes: int = 65536,
        memory_budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES,
    ) -> EvaluationResult:
        """Return Fourier-series amplitudes of a periodic rigid motion.

        ``A_0 = C_0`` and ``A_n = 2 C_n`` for ``n > 0``. ``tref`` is the phase
        reference of the convention ``exp[-2*pi*i*f*(t - tref)]``.

        Complex outputs represent linear complex amplitudes or transfer-function-like
        responses, not directly real-valued time-domain gravitational fields.

        Only the computation frame is supported. ``atol`` applies to the
        coefficients ``C_n`` before doubling, so each returned ``A_n = 2 C_n``
        can differ from its exact value by up to twice ``atol``.

        Parameters
        ----------
        period : astropy Quantity
            Period of the motion, as a time Quantity. Harmonic ``n`` has
            frequency ``n / period``.
        orders : array-like of int
            One-dimensional, nonempty, nonnegative harmonic orders. The output
            keeps this order.
        tref : LIGOTimeGPS or (int, int)
            Absolute GPS time of the phase reference. It is not the epoch of the
            model, which ``rigid_motion`` fixes. Pass the same ``tref`` to
            every method call that is compared.
        tau_min : astropy Quantity
            Smallest time scale of the response, as a time Quantity. It is
            required, and it sets the initial quadrature grid. A feature narrower
            than ``tau_min`` can be missed. Pass a value no larger than the
            narrowest structure of the response. If ``tau_min`` is too large, the
            quadrature can still return a successful but wrong value, because the
            coarse grid and the initial partition miss the narrow structure.
        atol : astropy Quantity
            Absolute tolerance in the unit of the coefficients ``C_n``, which is
            the unit of the requested observable (for example ``m / s**2`` for
            acceleration). Because the returned amplitudes are ``A_n = 2 C_n``,
            the error of an ``A_n`` can reach twice ``atol``. It is a scalar or
            any shape that broadcasts to ``(Ntarget, *component)``.
        rtol : float, optional
            Dimensionless nonnegative relative tolerance. The per-component
            tolerance is ``atol + rtol * max|q_i|`` on the initial grid.
        max_nodes : int, optional
            Upper bound on the number of nodes of one quadrature rule. It is not
            a cumulative count over all levels.
        memory_budget_bytes : int, optional
            Budget in bytes for the managed arrays of the periodic integrator,
            checked before the computation. It is applied per lane and is not a
            process RSS limit. Temporary arrays inside the pose callable are not
            counted.
        """
        # The epoch is the stored model epoch; tref only sets the phase reference.
        evaluation = self.model.periodic_at(
            self.targets,
            epoch=self.model.epoch,
            tref=tref,
            period=period,
            orders=orders,
            tau_min=tau_min,
            atol=atol,
            rtol=rtol,
            max_nodes=max_nodes,
            memory_budget_bytes=memory_budget_bytes,
            observables=self.observables,
        )
        return evaluation.run()

    def fourier_transform(
        self,
        *,
        t_start: u.Quantity,
        t_stop: u.Quantity,
        frequencies: u.Quantity,
        tref: object,
        tau_min: u.Quantity,
        atol: u.Quantity,
        rtol: float = 1e-10,
        max_evaluations: int = DEFAULT_MAX_EVALUATIONS,
        memory_budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES,
    ) -> EvaluationResult:
        """Return one-sided Fourier-transform coefficients over a finite window.

        The coefficients are not doubled. ``tref`` is the phase reference of the
        convention ``exp[-2*pi*i*f*(t - tref)]``.

        Complex outputs represent linear complex amplitudes or transfer-function-like
        responses, not directly real-valued time-domain gravitational fields.

        Only the computation frame is supported. ``output_frame`` is not a
        parameter here, and the continuous form cannot change the output frame.

        Parameters
        ----------
        t_start : astropy Quantity
            Window start, a scalar time Quantity. It is measured in seconds
            relative to the epoch of the rigid motion, not as an absolute GPS
            time.
        t_stop : astropy Quantity
            Window end, a scalar time Quantity greater than ``t_start``. It uses
            the same epoch-relative scale as ``t_start``.
        frequencies : astropy Quantity
            Finite, nonnegative, strictly increasing frequencies as a frequency
            Quantity, given as a scalar or a one-dimensional array. The result
            frequency axis requires this order, so an unsorted or repeated
            sequence raises ``ValueError``; sort the frequencies first.
        tref : LIGOTimeGPS or (int, int)
            Absolute GPS time of the phase reference. It is not the epoch of the
            model. The phase range must satisfy
            ``f_max * |epoch - tref| <= 1e6`` cycles, so place ``tref`` near the
            window.
        tau_min : astropy Quantity
            Smallest time scale of the response, as a time Quantity. It is
            required, and it sets the coarse-grid spacing and the initial panel
            width. A feature narrower than ``tau_min`` can be missed. Pass a value
            no larger than the narrowest structure of the response. If ``tau_min``
            is too large, the quadrature can still return a successful but wrong
            value, because the coarse grid and the initial partition miss the
            narrow structure.
        atol : astropy Quantity
            Absolute tolerance whose unit is the response unit times seconds, for
            example ``m / s**2 * s`` for acceleration. It is a scalar or any
            shape that broadcasts to ``(Ntarget, *component)``.
        rtol : float, optional
            Dimensionless nonnegative relative tolerance.
        max_evaluations : int, optional
            Budget for the total number of callback evaluations, counting the
            coarse grid, the main integration, and the shifted-partition check.
        memory_budget_bytes : int, optional
            Budget in bytes for the managed arrays of the finite-window
            integrator, checked before the computation. It is applied per lane
            and is not a process RSS limit. Temporary arrays inside the pose
            callable are not counted.
        """
        evaluation = self.model.window_at(
            self.targets,
            epoch=self.model.epoch,
            tref=tref,
            t_start=t_start,
            t_stop=t_stop,
            frequencies=frequencies,
            tau_min=tau_min,
            atol=atol,
            rtol=rtol,
            max_evaluations=max_evaluations,
            memory_budget_bytes=memory_budget_bytes,
            observables=self.observables,
        )
        return evaluation.run()


def _default_limit(request: Mapping[str, object]) -> int:
    """Apply the acceptance rule ``limit = max(10000, 4·n_init)`` when unset."""
    width = float(request["t_stop"].to_value(u.s) - request["t_start"].to_value(u.s))
    tau_s = float(request["tau_min"].to_value(u.s))
    freqs = np.asarray(request["frequencies"].to_value(u.Hz), dtype=np.float64)
    f_max = float(np.max(freqs)) if freqs.size else 0.0
    panel = 8.0 * tau_s
    if f_max > 0.0:
        panel = min(panel, 1.0 / (4.0 * f_max))
    n_init = max(1, int(math.ceil(width / panel)))
    return max(10000, 4 * n_init)


__all__ = [
    "ContinuousFourierModel",
    "ContinuousFourierModelSpec",
    "ContinuousPeriodicFourierEvaluation",
    "ContinuousRigidMotionEvaluation",
    "ContinuousWindowFourierEvaluation",
    "DEFAULT_MAX_EVALUATIONS",
    "DEFAULT_MEMORY_BUDGET_BYTES",
]
