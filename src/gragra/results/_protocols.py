"""Small structural contracts for models and their evaluations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from gragra._interactions import Interaction
from gragra.results._field_definition import GravityFieldDefinition

if TYPE_CHECKING:
    from astropy.table import QTable
    from pandas import DataFrame

    from gragra.results._context import EvaluationContext
    from gragra.results._evaluation import EvaluationResult


class ModelSpec(Protocol):
    """Physical interpretation supplied by a concrete model."""

    @property
    def field_definition(self) -> GravityFieldDefinition: ...

    @property
    def interaction(self) -> Interaction: ...


class FieldEvaluation(Protocol):
    """An evaluation that exposes one stable context and result."""

    @property
    def context(self) -> EvaluationContext: ...

    @property
    def parameters(self) -> QTable: ...

    @property
    def formulas(self) -> DataFrame: ...

    @property
    def references(self) -> DataFrame: ...

    def validate(self) -> object: ...

    def run(self) -> EvaluationResult: ...
