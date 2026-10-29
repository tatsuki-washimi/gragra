"""Shared interpretation for static and phasor evaluations."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
from astropy import units as u
from gwexpy.frequencyseries import FrequencySeriesMatrix
from gwpy.time import LIGOTimeGPS

from gragra._interactions import Newtonian
from gragra.geometry._frame import Frame
from gragra.results import Contribution, GravityFieldDefinition, PotentialReference
from gragra.results._binding import ObservableBinding
from gragra.results._context import EvaluationContext
from gragra.results._record import OBSERVABLES, EvaluationRecord
from gragra.results._representation import StaticRepresentation
from gragra.results._spectral import PhasorRepresentation, PSDRepresentation
from gragra.targets._high_points import TargetPoints


@dataclass(frozen=True)
class _Spec:
    field_definition: GravityFieldDefinition = GravityFieldDefinition.source_field(
        "external", Contribution("total"), PotentialReference.ZERO_AT_INFINITY
    )
    interaction: Newtonian = Newtonian()


def _targets(frame: Frame) -> TargetPoints:
    return TargetPoints(np.zeros((1, 3)) * u.m, ("only",), frame)


def _context(**overrides: object) -> EvaluationContext:
    frame = Frame("world")
    values = {
        "model_spec": _Spec(),
        "targets": _targets(frame),
        "frame": frame,
        "observables": ("gradient", "potential", "acceleration"),
        "representation": StaticRepresentation(),
    }
    values.update(overrides)
    return EvaluationContext(**values)


def test_static_context_and_binding_share_interpretation_by_identity() -> None:
    context = _context()
    assert context.observables == ("gradient", "potential", "acceleration")
    assert context.field_definition is context.model_spec.field_definition
    assert context.interaction is context.model_spec.interaction
    assert context.frequencies is None
    assert context.reference_time is None
    assert context.binding("gradient") is context.binding("gradient")
    assert context.binding("gradient").context is context
    assert context.binding("gradient").component_axes == (("x", "y", "z"),) * 2
    assert context.binding("gradient").frame is context.frame
    assert context.binding("potential").frame is None
    assert context.binding("potential").component_axes == ()
    assert context.binding("acceleration").component_axes == (("x", "y", "z"),)
    assert context.binding("potential").descriptor is context.representation
    assert context.binding("potential").representation is context.representation
    assert context.binding("potential").targets is context.targets
    assert context.binding("potential").field_definition is context.field_definition
    assert context.binding("potential").g_convention == "SI_G_applied_once"
    assert isinstance(context.binding("potential"), ObservableBinding)
    assert context.binding("potential").frequencies_hz is None
    assert context.binding("potential").reference_time_gps is None


def test_phasor_context_owns_real_matrix_frequency_axis_and_gps() -> None:
    matrix = FrequencySeriesMatrix(
        np.ones((1, 1, 1), dtype=np.complex128),
        frequencies=[250] * u.mHz,
        rows=["only"],
        cols=["potential"],
    )
    frequencies = matrix.frequencies.copy()
    context = _context(
        representation=PhasorRepresentation(),
        frequencies=frequencies,
        reference_time=LIGOTimeGPS(1_000_000_000, 123),
    )
    frequencies[0] = 900 * u.mHz
    assert context.frequencies_hz == (0.25,)
    np.testing.assert_array_equal(context.frequencies.to_value(u.Hz), [0.25])
    assert context.reference_time_gps == (1_000_000_000, 123)
    assert context.reference_time == LIGOTimeGPS(1_000_000_000, 123)
    assert context.binding("potential").frequencies_hz is context.frequencies_hz
    assert context.binding("potential").reference_time_gps is context.reference_time_gps


@pytest.mark.parametrize(
    ("frequencies", "message"),
    [
        ([1, 1] * u.Hz, "strictly increasing"),
        ([2, 1] * u.Hz, "strictly increasing"),
        ([1, np.nan] * u.Hz, "finite"),
        ([-1] * u.Hz, "positive"),
        (np.ones((1, 1)) * u.Hz, "nonempty"),
        ([] * u.Hz, "nonempty"),
    ],
)
def test_phasor_rejects_malformed_frequency_axis(frequencies, message) -> None:
    with pytest.raises(ValueError, match=message):
        _context(
            representation=PhasorRepresentation(),
            frequencies=frequencies,
            reference_time=(1, 0),
        )


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"frequencies": [1] * u.Hz}, ValueError, "static"),
        ({"reference_time": (1, 0)}, ValueError, "static"),
        (
            {"representation": PhasorRepresentation(), "frequencies": [1] * u.Hz},
            ValueError,
            "reference_time",
        ),
        ({"representation": PSDRepresentation()}, TypeError, "representation"),
        ({"observables": ("unknown",)}, ValueError, "observables"),
        ({"observables": ("potential", "potential")}, ValueError, "observables"),
        ({"observables": ()}, ValueError, "observables"),
        ({"model_spec": object()}, TypeError, "field_definition"),
        ({"targets": object()}, TypeError, "targets"),
        ({"frame": object()}, TypeError, "frame"),
    ],
)
def test_context_rejects_invalid_common_contract(overrides, error, message) -> None:
    with pytest.raises(error, match=message):
        _context(**overrides)


def test_disconnected_frame_is_rejected_and_none_resolves_target_frame() -> None:
    frame = Frame("targets")
    targets = _targets(frame)
    assert _context(targets=targets, frame=None).frame is frame
    with pytest.raises(ValueError, match="disconnected roots"):
        _context(targets=targets, frame=Frame("other"))


def test_record_and_observables_have_single_source() -> None:
    from gragra._evaluation_setup import OBSERVABLES as LEGACY_OBSERVABLES
    from gragra._evaluation_setup import EvaluationRecord as LegacyRecord

    assert LegacyRecord is EvaluationRecord
    assert LEGACY_OBSERVABLES is OBSERVABLES
    assert OBSERVABLES == ("potential", "acceleration", "gradient")


def test_interaction_requires_immutable_finite_scalar_parameters() -> None:
    class CustomInteraction:
        name = "custom"

        def __init__(self, parameters):
            self.parameters = parameters

    @dataclass(frozen=True)
    class CustomSpec:
        field_definition: GravityFieldDefinition
        interaction: CustomInteraction

    definition = _Spec().field_definition
    for parameters, error, message in (
        ({"scale": 1.0}, TypeError, "immutable"),
        (MappingProxyType({"scale": np.inf}), ValueError, "finite"),
        (MappingProxyType({"scale": []}), TypeError, "scalar"),
        (MappingProxyType({"": 1}), ValueError, "nonempty"),
    ):
        spec = CustomSpec(definition, CustomInteraction(parameters))
        with pytest.raises(error, match=message):
            _context(model_spec=spec)
    spec = CustomSpec(definition, CustomInteraction(MappingProxyType({"scale": 2.0})))
    context = _context(model_spec=spec)
    assert context.interaction.parameters["scale"] == 2.0


def test_interaction_snapshot_rejects_backing_mapping_and_name_drift() -> None:
    backing = {"scale": 1.0}
    interaction = SimpleNamespace(name="law-a", parameters=MappingProxyType(backing))
    spec = SimpleNamespace(
        field_definition=_Spec().field_definition, interaction=interaction
    )
    context = _context(model_spec=spec)
    assert context.interaction is interaction
    context.validate()

    backing["scale"] = 2.0
    with pytest.raises(ValueError, match="interaction.*parameters"):
        context.validate()
    with pytest.raises(ValueError, match="interaction.*parameters"):
        _ = context.interaction

    backing["scale"] = 1.0
    interaction.name = "law-b"
    with pytest.raises(ValueError, match="interaction.*name"):
        context.validate()


def test_interaction_accepts_other_immutable_mapping_and_checks_scalar_type() -> None:
    class ReadOnlyMapping(Mapping):
        def __init__(self, values):
            self._values = dict(values)

        def __getitem__(self, key):
            return self._values[key]

        def __iter__(self):
            return iter(self._values)

        def __len__(self):
            return len(self._values)

    values = ReadOnlyMapping({"scale": 1})
    interaction = SimpleNamespace(name="law", parameters=values)
    spec = SimpleNamespace(
        field_definition=_Spec().field_definition, interaction=interaction
    )
    context = _context(model_spec=spec)
    assert context.interaction is interaction
    context.validate()
    values._values["scale"] = True
    with pytest.raises(ValueError, match="interaction.*parameters"):
        context.validate()


def test_millihertz_normalization_does_not_reject_rounding_difference() -> None:
    from gragra._rayleigh_contract import _frequencies_hz as rayleigh_frequencies
    from gragra._rayleigh_contract import gps_offset_seconds as rayleigh_offset
    from gragra._rayleigh_contract import normalize_gps as rayleigh_gps
    from gragra.results._coords import (
        _frequencies_hz,
        gps_offset_seconds,
        normalize_gps,
    )

    assert rayleigh_frequencies is _frequencies_hz
    assert rayleigh_offset is gps_offset_seconds
    assert rayleigh_gps is normalize_gps
    axis = _frequencies_hz([100, 300, 700] * u.mHz)
    assert axis.shape == (3,)
    assert np.isfinite(axis).all()
