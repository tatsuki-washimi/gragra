"""Resolved C5 evaluations for static analytic and point-source gravity."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version
from platform import python_version
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u

from gragra._evaluation_setup import (
    OBSERVABLES,
    CoordinatePrecisionError,
    EvaluationRecord,
    ResolvedEvaluation,
    UniformSphereModelSpec,
    check_pair_precision,
)
from gragra.constants import G_SI
from gragra.results._representation import StaticRepresentation

if TYPE_CHECKING:
    from gragra.geometry import Appearance


@dataclass(frozen=True)
class ValidationReport:
    """Mechanical pre-run facts for one fixed evaluation request."""

    valid: bool
    observables: tuple[str, ...]
    target_count: int
    source_count: int | None
    checked_pairs: int
    target_regions: Mapping[str, str]


class EvaluationError(ValueError):
    """A failed requested calculation with target and observable context."""

    def __init__(
        self,
        *,
        target_id: str | None,
        source_index: int | None,
        observables: tuple[str, ...],
        observable: str,
        cause: Exception,
    ) -> None:
        self.target_id = target_id
        self.source_index = source_index
        self.observables = observables
        self.observable = observable
        self.cause = cause
        context = (
            "target ID unavailable" if target_id is None else f"target ID={target_id!r}"
        )
        if source_index is not None:
            context += f", source index={source_index}"
        super().__init__(
            f"evaluation failed for {context}, observable={observable!r}, "
            f"requested={observables}: {cause}"
        )


def _evaluation_error(
    evaluation: ResolvedEvaluation,
    cause: Exception,
    *,
    target_index: int | None,
    source_index: int | None = None,
    observable: str | None = None,
) -> EvaluationError:
    requested = evaluation.observables
    failed = observable or requested[0]
    target_id = (
        None
        if target_index is None or not len(evaluation.targets)
        else evaluation.targets.ids[target_index]
    )
    return EvaluationError(
        target_id=target_id,
        source_index=source_index,
        observables=requested,
        observable=failed,
        cause=cause,
    )


def _failed_field(cause: Exception, fallback: str) -> str:
    message = str(cause).lower()
    return next((name for name in OBSERVABLES if name in message), fallback)


def _check_discrete_pairs(evaluation: ResolvedEvaluation) -> int:
    source = evaluation.source
    if source is None:
        return 0

    positions = source.positions_m
    pair_count = 0
    eps64 = np.finfo(np.float64).eps
    for target_index, trace in enumerate(evaluation.target_traces):
        pair_count += len(source)
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                difference = positions - trace.point_m
                distances = np.max(np.abs(difference), axis=1)
                source_scales = np.max(np.abs(positions), axis=1)
        except Exception as cause:
            raise _evaluation_error(
                evaluation, cause, target_index=target_index
            ) from cause
        nonfinite = np.flatnonzero(~np.isfinite(difference).all(axis=1))
        if len(nonfinite):
            source_index = int(nonfinite[0])
            try:
                check_pair_precision(
                    positions[source_index],
                    trace,
                    target_id=evaluation.targets.ids[target_index],
                    source_index=source_index,
                    observables=evaluation.observables,
                )
            except CoordinatePrecisionError:
                raise
            except Exception as cause:
                raise _evaluation_error(
                    evaluation,
                    cause,
                    target_index=target_index,
                    source_index=source_index,
                ) from cause

        # The point kernels form squared distances before taking a root.
        # Check that same intermediate here so validate() cannot label a
        # finite but unusable pair as an ordinary request. Keep the common
        # path vectorized; isolate the first failing pair only on overflow.
        try:
            with np.errstate(over="raise", invalid="raise"):
                np.sum(difference**2, axis=1)
        except Exception as cause:
            for source_index, offset in enumerate(difference):
                try:
                    with np.errstate(over="raise", invalid="raise"):
                        np.sum(offset**2)
                except Exception as pair_cause:
                    raise _evaluation_error(
                        evaluation,
                        pair_cause,
                        target_index=target_index,
                        source_index=source_index,
                    ) from pair_cause
            raise _evaluation_error(
                evaluation, cause, target_index=target_index
            ) from cause

        # Vectorize the ordinary no-error path. Call the shared diagnostic for
        # each colliding or cross-frame near pair so its structured exception
        # carries the precise target and source indices.
        scales = np.maximum(trace.scale_m, source_scales)
        thresholds = 64 * trace.edge_count * eps64 * scales
        candidates = distances == 0
        if trace.edge_count:
            candidates |= distances <= thresholds
        for source_index in np.flatnonzero(candidates):
            index = int(source_index)
            try:
                check_pair_precision(
                    positions[index],
                    trace,
                    target_id=evaluation.targets.ids[target_index],
                    source_index=index,
                    observables=evaluation.observables,
                )
            except CoordinatePrecisionError:
                raise
            except Exception as cause:
                raise _evaluation_error(
                    evaluation,
                    cause,
                    target_index=target_index,
                    source_index=index,
                ) from cause
    return pair_count


def _compute_values(
    evaluation: ResolvedEvaluation, *, check_pairs: bool = True
) -> dict[str, np.ndarray]:
    """Evaluate requested SI observables as plain arrays in output components."""
    if not isinstance(evaluation, ResolvedEvaluation):
        raise TypeError("evaluation must be a ResolvedEvaluation")
    if check_pairs:
        _check_discrete_pairs(evaluation)

    if evaluation.source is None:
        sphere = evaluation.model_spec.geometry
        if set(evaluation.observables) == set(OBSERVABLES):
            from gragra.observables.analytic import uniform_sphere_observables

            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    all_values = uniform_sphere_observables(
                        sphere.center_m,
                        sphere.radius_m,
                        evaluation.model_spec.density_kg_m3,
                        evaluation.target_positions_m,
                    )
            except Exception as cause:
                from gragra.analytic.uniform_sphere import (
                    uniform_sphere_acceleration,
                    uniform_sphere_gradient,
                    uniform_sphere_potential,
                )

                helpers = {
                    "potential": uniform_sphere_potential,
                    "acceleration": uniform_sphere_acceleration,
                    "gradient": uniform_sphere_gradient,
                }
                for target_index, point in enumerate(evaluation.target_positions_m):
                    for name in evaluation.observables:
                        try:
                            with np.errstate(
                                over="raise", invalid="raise", divide="raise"
                            ):
                                value = G_SI * helpers[name](
                                    sphere.center_m,
                                    sphere.radius_m,
                                    evaluation.model_spec.density_kg_m3,
                                    point.reshape(1, 3),
                                )
                            if not np.isfinite(value).all():
                                raise ValueError(
                                    f"{name} returned non-finite SI values"
                                )
                        except Exception as target_cause:
                            raise _evaluation_error(
                                evaluation,
                                target_cause,
                                target_index=target_index,
                                observable=name,
                            ) from target_cause
                raise _evaluation_error(
                    evaluation,
                    cause,
                    target_index=0 if len(evaluation.targets) else None,
                    observable=_failed_field(cause, evaluation.observables[0]),
                ) from cause
            values = {
                name: np.asarray(getattr(all_values, name))
                for name in evaluation.observables
            }
        else:
            from gragra.analytic.uniform_sphere import (
                uniform_sphere_acceleration,
                uniform_sphere_gradient,
                uniform_sphere_potential,
            )

            helpers = {
                "potential": uniform_sphere_potential,
                "acceleration": uniform_sphere_acceleration,
                "gradient": uniform_sphere_gradient,
            }
            values = {}
            for name in evaluation.observables:
                helper = helpers[name]

                def calculate(
                    points: np.ndarray, *, selected_helper=helper
                ) -> np.ndarray:
                    with np.errstate(over="raise", invalid="raise", divide="raise"):
                        return np.asarray(
                            G_SI
                            * selected_helper(
                                sphere.center_m,
                                sphere.radius_m,
                                evaluation.model_spec.density_kg_m3,
                                points,
                            )
                        )

                try:
                    result = calculate(evaluation.target_positions_m)
                except Exception as cause:
                    for target_index, point in enumerate(evaluation.target_positions_m):
                        try:
                            calculate(point.reshape(1, 3))
                        except Exception as target_cause:
                            raise _evaluation_error(
                                evaluation,
                                target_cause,
                                target_index=target_index,
                                observable=name,
                            ) from target_cause
                    raise _evaluation_error(
                        evaluation,
                        cause,
                        target_index=0 if len(evaluation.targets) else None,
                        observable=name,
                    ) from cause

                if len(evaluation.targets):
                    bad_rows = np.flatnonzero(
                        ~np.isfinite(result.reshape(len(evaluation.targets), -1)).all(
                            axis=1
                        )
                    )
                else:
                    bad_rows = np.empty(0, dtype=np.intp)
                if len(bad_rows):
                    target_index = int(bad_rows[0])
                    cause = ValueError(f"{name} returned non-finite SI values")
                    raise _evaluation_error(
                        evaluation,
                        cause,
                        target_index=target_index,
                        observable=name,
                    ) from cause
                values[name] = result
    else:
        from gragra.gravity_sources.weighted_points import WeightedPointSource
        from gragra.observables.acceleration import point_acceleration
        from gragra.observables.gradient import point_gravity_gradient
        from gragra.observables.potential import point_potential

        source = evaluation.source
        numeric_source = WeightedPointSource(
            positions_m=source.positions_m,
            weights_kg=source.weights_kg,
        )
        kernels = {
            "potential": point_potential,
            "acceleration": point_acceleration,
            "gradient": point_gravity_gradient,
        }
        values = {}
        for name in evaluation.observables:
            kernel = kernels[name]
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    result = np.asarray(
                        kernel(
                            numeric_source,
                            evaluation.target_positions_m,
                            backend="numpy",
                        )
                    )
            except Exception as cause:
                for target_index, point in enumerate(evaluation.target_positions_m):
                    try:
                        with np.errstate(over="raise", invalid="raise", divide="raise"):
                            kernel(
                                numeric_source,
                                point.reshape(1, 3),
                                backend="numpy",
                            )
                    except Exception as target_cause:
                        raise _evaluation_error(
                            evaluation,
                            target_cause,
                            target_index=target_index,
                            observable=name,
                        ) from target_cause
                raise _evaluation_error(
                    evaluation,
                    cause,
                    target_index=0 if len(evaluation.targets) else None,
                    observable=name,
                ) from cause
            if len(evaluation.targets):
                bad_rows = np.flatnonzero(
                    ~np.isfinite(result.reshape(len(evaluation.targets), -1)).all(
                        axis=1
                    )
                )
            else:
                bad_rows = np.empty(0, dtype=np.intp)
            if len(bad_rows):
                target_index = int(bad_rows[0])
                cause = ValueError(f"{name} returned non-finite SI values")
                raise _evaluation_error(
                    evaluation,
                    cause,
                    target_index=target_index,
                    observable=name,
                ) from cause
            values[name] = result

    return _transform_values(evaluation, values)


def _transform_values(
    evaluation: ResolvedEvaluation, values: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Transform each result row and add request context to numeric failures."""
    transformed = dict(values)
    for name in ("acceleration", "gradient"):
        if name not in transformed or not len(evaluation.targets):
            continue
        rows = []
        for target_index, row in enumerate(transformed[name]):
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    if name == "acceleration":
                        result = evaluation.computation_frame.transform_vectors_m(
                            row, evaluation.output_frame
                        )
                    else:
                        result = evaluation.computation_frame.transform_tensors(
                            row, evaluation.output_frame
                        )
                if not np.isfinite(result).all():
                    raise ValueError(
                        f"{name} frame transform returned non-finite values"
                    )
            except Exception as cause:
                raise _evaluation_error(
                    evaluation,
                    cause,
                    target_index=target_index,
                    observable=name,
                ) from cause
            rows.append(result)
        transformed[name] = np.stack(rows)
    return transformed


class EvaluationSetup:
    """One fixed request shared by validation, plotting, and numeric execution."""

    __slots__ = ("_resolved", "_context")

    def __init__(self, evaluation: ResolvedEvaluation) -> None:
        if not isinstance(evaluation, ResolvedEvaluation):
            raise TypeError("evaluation must be a ResolvedEvaluation")
        self._resolved = evaluation
        from gragra.results._context import EvaluationContext

        self._context = EvaluationContext(
            model_spec=evaluation.model_spec,
            targets=evaluation.targets,
            frame=evaluation.output_frame,
            observables=evaluation.observables,
            representation=StaticRepresentation(),
        )

    @property
    def context(self):
        return self._context

    @property
    def resolved(self) -> ResolvedEvaluation:
        return self._resolved

    @property
    def evaluation(self) -> ResolvedEvaluation:
        return self._resolved

    @property
    def parameters(self):
        return self._resolved.parameters

    @property
    def formulas(self):
        return self._resolved.formulas

    @property
    def references(self):
        return self._resolved.references

    def validate(self) -> ValidationReport:
        """Check pairwise source-target validity and report resolved conditions.

        Collision and cross-frame precision failures raise a structured
        ``CoordinatePrecisionError`` before evaluation. Same-frame, nonzero
        close pairs retain the existing unsoftened point-kernel behavior.
        """
        checked_pairs = _check_discrete_pairs(self._resolved)
        regions = {}
        if isinstance(self._resolved.model_spec, UniformSphereModelSpec):
            geometry = self._resolved.model_spec.geometry
            with np.errstate(over="ignore", invalid="ignore"):
                relative = self._resolved.target_positions_m - geometry.center_m
            for target_index, offset in enumerate(relative):
                if not np.isfinite(offset).all():
                    cause = ValueError("target-to-sphere offset is non-finite")
                    raise _evaluation_error(
                        self._resolved,
                        cause,
                        target_index=target_index,
                    ) from cause
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    radii = np.linalg.norm(relative, axis=1)
            except Exception as cause:
                for target_index, offset in enumerate(relative):
                    try:
                        with np.errstate(over="raise", invalid="raise", divide="raise"):
                            np.linalg.norm(offset)
                    except Exception as target_cause:
                        raise _evaluation_error(
                            self._resolved,
                            target_cause,
                            target_index=target_index,
                        ) from target_cause
                raise _evaluation_error(
                    self._resolved,
                    cause,
                    target_index=0 if len(self._resolved.targets) else None,
                ) from cause
            if len(radii) and not np.isfinite(radii).all():
                target_index = int(np.flatnonzero(~np.isfinite(radii))[0])
                cause = ValueError("target-to-sphere distance is non-finite")
                raise _evaluation_error(
                    self._resolved, cause, target_index=target_index
                ) from cause
            regions = {
                target_id: (
                    "interior"
                    if radius < geometry.radius_m
                    else "boundary"
                    if radius == geometry.radius_m
                    else "exterior"
                )
                for target_id, radius in zip(
                    self._resolved.targets.ids, radii, strict=True
                )
            }
        return ValidationReport(
            valid=True,
            observables=self._resolved.observables,
            target_count=len(self._resolved.targets),
            source_count=(
                None if self._resolved.source is None else len(self._resolved.source)
            ),
            checked_pairs=checked_pairs,
            target_regions=MappingProxyType(regions),
        )

    def plot_setup(
        self,
        *,
        view: str = "3d",
        unit: str = "m",
        ax=None,
        max_sources: int | None = 5000,
        appearance: Appearance | None = None,
    ):
        """Plot Sphere setups; generic point sums do not define a shape to draw."""
        if not isinstance(self._resolved.model_spec, UniformSphereModelSpec):
            raise NotImplementedError(
                "plot_setup currently supports Sphere-backed evaluations only"
            )
        from gragra._plotting._setup import plot_setup

        return plot_setup(
            self,
            view=view,
            unit=unit,
            ax=ax,
            max_sources=max_sources,
            appearance=appearance,
        )

    def run(self):
        """Validate and return computed SI quantities with their fixed context."""
        self.validate()
        arrays = _compute_values(self._resolved, check_pairs=False)

        from gragra.results import GravityResult
        from gragra.results._evaluation import EvaluationResult

        units = {
            "potential": u.m**2 / u.s**2,
            "acceleration": u.m / u.s**2,
            "gradient": u.s**-2,
        }
        values = GravityResult(
            **{
                name: arrays[name] * units[name] if name in arrays else None
                for name in ("potential", "acceleration", "gradient")
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
        return EvaluationResult(
            values=values,
            evaluation=self,
            record=record,
        )


__all__ = ["EvaluationError", "EvaluationSetup", "ValidationReport"]
