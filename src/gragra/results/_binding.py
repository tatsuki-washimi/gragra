"""Per-observable views of a shared evaluation context."""

from __future__ import annotations

from dataclasses import dataclass

from gragra.results._context import EvaluationContext
from gragra.results._time import TimeSampling


@dataclass(frozen=True, slots=True, eq=False)
class ObservableBinding:
    """One observable name and a reference to its owning context."""

    observable: str
    context: EvaluationContext

    def __post_init__(self) -> None:
        if not isinstance(self.context, EvaluationContext):
            raise TypeError("context must be an EvaluationContext")
        if self.observable not in self.context.observables:
            raise ValueError("observable must be requested in context.observables")

    @property
    def targets(self):
        return self.context.targets

    @property
    def target_ids(self) -> tuple[str, ...]:
        return self.context.targets.ids

    @property
    def frame(self):
        return None if self.observable == "potential" else self.context.frame

    @property
    def component_axes(self) -> tuple[tuple[str, ...], ...]:
        axes = ("x", "y", "z")
        return {"potential": (), "acceleration": (axes,), "gradient": (axes, axes)}[
            self.observable
        ]

    @property
    def representation(self):
        return self.context.representation

    @property
    def descriptor(self):
        return self.context.representation

    @property
    def frequencies_hz(self) -> tuple[float, ...] | None:
        return self.context.frequencies_hz

    @property
    def time_sampling(self) -> TimeSampling | None:
        """The shared expected time axis owned by this binding's context."""
        return self.context.time_sampling

    @property
    def reference_time_gps(self) -> tuple[int, int] | None:
        return self.context.reference_time_gps

    @property
    def field_definition(self):
        return self.context.field_definition

    @property
    def g_convention(self) -> str:
        return self.context.g_convention
