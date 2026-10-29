"""Two small independent extension models for the common result contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy import units as u
from astropy.table import QTable
from gwexpy.frequencyseries import FrequencySeriesMatrix
from pandas import DataFrame

from gragra import TargetPoints
from gragra.interactions import Newtonian
from gragra.results import (
    Contribution,
    EvaluationContext,
    EvaluationRecord,
    EvaluationResult,
    GravityFieldDefinition,
    GravityResult,
    PhasorRepresentation,
    PotentialReference,
    StaticRepresentation,
)


@dataclass(frozen=True)
class ExternalSpec:
    field_definition: GravityFieldDefinition = GravityFieldDefinition.source_field(
        "external_fixture",
        Contribution("total"),
        PotentialReference.ZERO_AT_INFINITY,
    )
    interaction: Newtonian = Newtonian()


class ExternalStaticModel:
    """Return direct SI quantities with no gragra model implementation."""

    def __init__(self, offset: float = 0.0) -> None:
        self.spec = ExternalSpec()
        self.offset = offset

    def at(self, targets: TargetPoints) -> ExternalStaticEvaluation:
        return ExternalStaticEvaluation(self, targets)


class ExternalPhasorModel:
    """Return direct complex matrices at one positive frequency."""

    def __init__(self, offset: float = 0.0) -> None:
        self.spec = ExternalSpec()
        self.offset = offset

    def at(self, targets: TargetPoints) -> ExternalPhasorEvaluation:
        return ExternalPhasorEvaluation(self, targets)


class _BaseEvaluation:
    def __init__(self, model: object, targets: TargetPoints, **context_kwargs: object):
        self.model = model
        self._context = EvaluationContext(
            model_spec=model.spec,
            targets=targets,
            frame=None,
            observables=("potential", "acceleration", "gradient"),
            **context_kwargs,
        )
        self.run_count = 0

    @property
    def context(self) -> EvaluationContext:
        return self._context

    @property
    def parameters(self) -> QTable:
        return QTable({"offset": [self.model.offset]})

    @property
    def formulas(self) -> DataFrame:
        return DataFrame()

    @property
    def references(self) -> DataFrame:
        return DataFrame()

    def validate(self) -> None:
        self.context.validate()

    def _result(self, values: GravityResult) -> EvaluationResult:
        return EvaluationResult(
            values,
            evaluation=self,
            record=EvaluationRecord(
                backend="external_fixture", versions={"fixture": "1"}
            ),
        )


class ExternalStaticEvaluation(_BaseEvaluation):
    def __init__(self, model: ExternalStaticModel, targets: TargetPoints) -> None:
        super().__init__(model, targets, representation=StaticRepresentation())

    def run(self) -> EvaluationResult:
        self.validate()
        self.run_count += 1
        count = len(self.context.targets)
        offset = self.model.offset
        potential = np.full(count, 1.0 + offset) * u.m**2 / u.s**2
        acceleration = (np.tile([2.0, 3.0, 4.0], (count, 1)) + offset) * u.m / u.s**2
        gradient = (
            np.tile(np.arange(5.0, 14.0).reshape(1, 3, 3), (count, 1, 1)) + offset
        ) / u.s**2
        return self._result(
            GravityResult(
                name="external static",
                potential=potential,
                acceleration=acceleration,
                gradient=gradient,
            )
        )


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


class ExternalPhasorEvaluation(_BaseEvaluation):
    def __init__(self, model: ExternalPhasorModel, targets: TargetPoints) -> None:
        super().__init__(
            model,
            targets,
            representation=PhasorRepresentation(),
            frequencies=[0.25] * u.Hz,
            reference_time=(1_000_000_000, 123),
        )

    def run(self) -> EvaluationResult:
        self.validate()
        self.run_count += 1
        count = len(self.context.targets)
        matrices = {}
        for name, components in _COMPONENTS.items():
            values = np.empty((count, len(components), 1), dtype=np.complex128)
            values[:, :, 0] = (
                np.arange(1.0, len(components) + 1.0) + self.model.offset + 0.5j
            )
            matrices[name] = FrequencySeriesMatrix(
                values,
                frequencies=[0.25] * u.Hz,
                units=np.full((count, len(components)), _UNITS[name], dtype=object),
                rows=list(self.context.targets.ids),
                cols=list(components),
                name=f"external {name}",
            )
        return self._result(GravityResult(name="external phasor", **matrices))
