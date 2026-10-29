"""Model-independent interpretation of one resolved field evaluation."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u
from gwpy.time import LIGOTimeGPS

from gragra.geometry._frame import Frame
from gragra.results._coords import (
    _frequencies_hz,
    normalize_gps,
    one_sided_frequencies_from_quantity,
)
from gragra.results._field_definition import GravityFieldDefinition
from gragra.results._protocols import ModelSpec
from gragra.results._record import OBSERVABLES
from gragra.results._representation import (
    StaticRepresentation,
    TimeSeriesRepresentation,
)
from gragra.results._spectral import (
    AmplitudeSpectrumRepresentation,
    FourierTransformRepresentation,
    PhasorRepresentation,
)
from gragra.results._time import TimeSampling
from gragra.targets._high_points import TargetPoints

if TYPE_CHECKING:
    from gragra._interactions import Interaction
    from gragra.results._binding import ObservableBinding


def _interaction_identity(
    interaction: object,
) -> tuple[str, tuple[tuple[str, type, object], ...]]:
    """Copy the valid scalar identity without retaining a caller-owned mapping."""
    name = getattr(interaction, "name", None)
    if not isinstance(name, str) or not name:
        raise TypeError("interaction.name must be a nonempty string")
    parameters = getattr(interaction, "parameters", None)
    if not isinstance(parameters, Mapping):
        raise TypeError("interaction.parameters must be a mapping")
    if isinstance(parameters, MutableMapping) or any(
        hasattr(parameters, method) for method in ("__setitem__", "__delitem__")
    ):
        raise TypeError("interaction.parameters must be an immutable mapping")
    snapshot = []
    for key, value in parameters.items():
        if not isinstance(key, str) or not key:
            raise ValueError("interaction.parameters keys must be nonempty strings")
        if isinstance(value, np.generic):
            value = value.item()
        if value is not None and not isinstance(
            value, (str, bool, int, float, complex)
        ):
            raise TypeError("interaction.parameters values must be immutable scalars")
        if isinstance(value, (float, complex)) and not np.isfinite(value):
            raise ValueError("interaction.parameters values must be finite")
        snapshot.append((key, type(value), value))
    return name, tuple(sorted(snapshot, key=lambda item: item[0]))


@dataclass(frozen=True, eq=False, init=False)
class EvaluationContext:
    """Owned coordinates and shared interpretation for all requested observables."""

    model_spec: ModelSpec
    targets: TargetPoints
    frame: Frame
    observables: tuple[str, ...]
    representation: (
        StaticRepresentation
        | PhasorRepresentation
        | TimeSeriesRepresentation
        | FourierTransformRepresentation
        | AmplitudeSpectrumRepresentation
    )
    time_sampling: TimeSampling | None
    frequencies_hz: tuple[float, ...] | None
    reference_time_gps: tuple[int, int] | None
    g_convention: str
    _field_definition: GravityFieldDefinition = field(repr=False)
    _interaction: Interaction = field(repr=False)
    _interaction_name: str = field(repr=False)
    _interaction_parameters: tuple[tuple[str, type, object], ...] = field(repr=False)
    _binding_cache: dict[str, ObservableBinding] = field(repr=False)

    def __init__(
        self,
        *,
        model_spec: ModelSpec,
        targets: TargetPoints,
        frame: Frame | None,
        observables: tuple[str, ...],
        representation: (
            StaticRepresentation
            | PhasorRepresentation
            | TimeSeriesRepresentation
            | FourierTransformRepresentation
            | AmplitudeSpectrumRepresentation
        ),
        time_sampling: TimeSampling | None = None,
        frequencies: u.Quantity | None = None,
        reference_time: object | None = None,
        g_convention: str = "SI_G_applied_once",
    ) -> None:
        definition = getattr(model_spec, "field_definition", None)
        if not isinstance(definition, GravityFieldDefinition):
            raise TypeError(
                "model_spec.field_definition must be a GravityFieldDefinition"
            )
        interaction = getattr(model_spec, "interaction", None)
        interaction_name, interaction_parameters = _interaction_identity(interaction)
        if not isinstance(targets, TargetPoints):
            raise TypeError("targets must be high-level TargetPoints")
        if frame is None:
            frame = targets.frame
        if not isinstance(frame, Frame):
            raise TypeError("frame must be a Frame or None")
        targets.frame._path_to(frame)
        if isinstance(observables, (str, bytes)):
            raise ValueError(
                "observables must be a nonempty unique subset of known names"
            )
        try:
            names = tuple(observables)
        except TypeError as exc:
            raise TypeError("observables must be an iterable") from exc
        if (
            not names
            or any(name not in OBSERVABLES for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError(
                "observables must be a nonempty unique subset of known names"
            )
        if isinstance(representation, StaticRepresentation):
            if (
                time_sampling is not None
                or frequencies is not None
                or reference_time is not None
            ):
                raise ValueError(
                    "static requires time_sampling, frequencies, and "
                    "reference_time to be None"
                )
            axis = None
            gps = None
        elif isinstance(representation, PhasorRepresentation):
            if time_sampling is not None:
                raise ValueError("phasor requires time_sampling to be None")
            axis = tuple(float(value) for value in _frequencies_hz(frequencies))
            if reference_time is None:
                raise ValueError("reference_time is required for phasor representation")
            gps = normalize_gps(reference_time)
        elif isinstance(representation, TimeSeriesRepresentation):
            if time_sampling is None:
                raise ValueError("time_series requires time_sampling")
            if not isinstance(time_sampling, TimeSampling):
                raise TypeError("time_series representation requires TimeSampling")
            if frequencies is not None or reference_time is not None:
                raise ValueError(
                    "time_series requires frequencies and reference_time to be None"
                )
            axis = None
            gps = None
        elif isinstance(representation, FourierTransformRepresentation):
            if time_sampling is not None:
                raise ValueError("fourier_transform requires time_sampling to be None")
            if frequencies is None or reference_time is None:
                raise ValueError(
                    "fourier_transform requires frequencies and reference_time"
                )
            axis = one_sided_frequencies_from_quantity(frequencies)
            gps = normalize_gps(reference_time)
        elif isinstance(representation, AmplitudeSpectrumRepresentation):
            if time_sampling is not None:
                raise ValueError("amplitude_spectrum requires time_sampling to be None")
            if frequencies is None or reference_time is None:
                raise ValueError(
                    "amplitude_spectrum requires frequencies and reference_time"
                )
            axis = one_sided_frequencies_from_quantity(frequencies)
            gps = normalize_gps(reference_time)
        else:
            raise TypeError(
                "representation must be StaticRepresentation, "
                "PhasorRepresentation, TimeSeriesRepresentation, "
                "FourierTransformRepresentation, or AmplitudeSpectrumRepresentation"
            )
        if not isinstance(g_convention, str) or not g_convention:
            raise ValueError("g_convention must be a nonempty string")
        for name, value in (
            ("model_spec", model_spec),
            ("targets", targets),
            ("frame", frame),
            ("observables", names),
            ("representation", representation),
            ("time_sampling", time_sampling),
            ("frequencies_hz", axis),
            ("reference_time_gps", gps),
            ("g_convention", g_convention),
            ("_field_definition", definition),
            ("_interaction", interaction),
            ("_interaction_name", interaction_name),
            ("_interaction_parameters", interaction_parameters),
            ("_binding_cache", {}),
        ):
            object.__setattr__(self, name, value)

    @property
    def field_definition(self) -> GravityFieldDefinition:
        return self.model_spec.field_definition

    @property
    def interaction(self) -> Interaction:
        self.validate()
        return self._interaction

    @property
    def frequencies(self) -> u.Quantity | None:
        if self.frequencies_hz is None:
            return None
        return np.asarray(self.frequencies_hz) * u.Hz

    @property
    def reference_time(self) -> LIGOTimeGPS | None:
        if self.reference_time_gps is None:
            return None
        return LIGOTimeGPS(*self.reference_time_gps)

    def binding(self, observable: str) -> ObservableBinding:
        """Return the same binding for repeated requests on this context."""
        if observable not in self.observables:
            raise ValueError("observable must be requested in context.observables")
        binding = self._binding_cache.get(observable)
        if binding is None:
            from gragra.results._binding import ObservableBinding

            binding = ObservableBinding(observable, self)
            self._binding_cache[observable] = binding
        return binding

    def validate(self) -> None:
        """Detect replacement of the model's shared physical interpretation."""
        representation = self.representation
        if isinstance(representation, StaticRepresentation):
            if (
                self.time_sampling is not None
                or self.frequencies_hz is not None
                or self.reference_time_gps is not None
            ):
                raise ValueError("static context has incompatible time coordinates")
        elif isinstance(representation, PhasorRepresentation):
            if (
                self.time_sampling is not None
                or self.frequencies_hz is None
                or self.reference_time_gps is None
            ):
                raise ValueError("phasor context has incompatible time coordinates")
        elif isinstance(representation, TimeSeriesRepresentation):
            if (
                not isinstance(self.time_sampling, TimeSampling)
                or self.frequencies_hz is not None
                or self.reference_time_gps is not None
            ):
                raise ValueError(
                    "time_series context has incompatible time coordinates"
                )
        elif isinstance(representation, (FourierTransformRepresentation, AmplitudeSpectrumRepresentation)):
            if (
                self.time_sampling is not None
                or self.frequencies_hz is None
                or self.reference_time_gps is None
            ):
                raise ValueError(
                    "spectral context has incompatible frequency coordinates"
                )
        else:
            raise TypeError("context representation is unsupported")
        if self.model_spec.field_definition is not self._field_definition:
            raise ValueError(
                "model_spec.field_definition changed after context construction"
            )
        if self.model_spec.interaction is not self._interaction:
            raise ValueError(
                "model_spec.interaction changed after context construction"
            )
        name, parameters = _interaction_identity(self._interaction)
        if name != self._interaction_name:
            raise ValueError("model_spec.interaction name changed after construction")
        if parameters != self._interaction_parameters:
            raise ValueError(
                "model_spec.interaction parameters changed after construction"
            )
