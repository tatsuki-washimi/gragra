"""Contracts for spectral representation descriptors."""

from dataclasses import FrozenInstanceError, fields
from enum import StrEnum
from typing import get_args

import pytest


def test_public_names_are_available_from_results_only() -> None:
    import gragra
    import gragra.results as results

    expected = {
        "Sidedness",
        "FiniteRecordInterpretation",
        "FourierTransformRepresentation",
        "AmplitudeSpectrumRepresentation",
        "PhasorRepresentation",
        "PSDRepresentation",
        "ASDRepresentation",
        "CSDRepresentation",
        "SpectralRepresentation",
    }
    c2_names = {
        "FieldKind",
        "ReferenceStatus",
        "ReferenceTemporalKind",
        "ChangeForm",
        "PotentialReference",
        "ReferenceDefinition",
        "ChangeDefinition",
        "Contribution",
        "GravityFieldDefinition",
    }
    expected_inventory = {
        "GravityResult",
        *expected,
        *c2_names,
        "EvaluationResult",
        "compare",
        "GravityResponse",
        "ResponseContract",
        "EvaluationContext",
        "StaticRepresentation",
        "TimeSeriesRepresentation",
        "TimeSampling",
        "ModelSpec",
        "FieldEvaluation",
        "EvaluationRecord",
        "NumericalDiagnostic",
    }
    assert expected <= set(results.__all__)
    assert set(results.__all__) == expected_inventory
    assert len(results.__all__) == len(expected_inventory)
    context_names = {
        "EvaluationContext",
        "StaticRepresentation",
        "ModelSpec",
        "FieldEvaluation",
        "EvaluationRecord",
    }
    results_only_names = expected | c2_names | context_names
    assert not results_only_names.intersection(gragra.__all__)
    assert all(not hasattr(gragra, name) for name in results_only_names)

    import gragra.interactions as interactions

    assert interactions.__all__ == ["Interaction", "Newtonian"]
    assert not set(interactions.__all__).intersection(gragra.__all__)


def test_enums_are_strict_string_enums() -> None:
    from gragra.results import FiniteRecordInterpretation, Sidedness

    assert issubclass(Sidedness, StrEnum)
    assert [(item.name, item.value) for item in Sidedness] == [
        ("ONE_SIDED", "one_sided"),
        ("TWO_SIDED", "two_sided"),
    ]
    assert [(item.name, item.value) for item in FiniteRecordInterpretation] == [
        ("PERIODIC", "periodic"),
        ("TRANSIENT", "transient"),
    ]


def test_descriptors_have_only_the_declared_configuration_and_kinds() -> None:
    from gragra.results import (
        AmplitudeSpectrumRepresentation,
        ASDRepresentation,
        CSDRepresentation,
        FiniteRecordInterpretation,
        FourierTransformRepresentation,
        PhasorRepresentation,
        PSDRepresentation,
        Sidedness,
    )

    configs = [
        FourierTransformRepresentation(),
        AmplitudeSpectrumRepresentation(
            record_interpretation=FiniteRecordInterpretation.PERIODIC
        ),
        PhasorRepresentation(),
        PSDRepresentation(),
        ASDRepresentation(),
        CSDRepresentation(),
    ]
    assert [item.kind for item in configs] == [
        "fourier_transform",
        "amplitude_spectrum",
        "phasor",
        "psd",
        "asd",
        "csd",
    ]
    assert configs[0].sidedness is Sidedness.TWO_SIDED
    assert configs[1].record_interpretation is FiniteRecordInterpretation.PERIODIC
    assert [tuple(f.name for f in fields(type(item))) for item in configs] == [
        ("sidedness",),
        ("record_interpretation",),
        (),
        (),
        (),
        (),
    ]
    for item in configs:
        assert not hasattr(item, "__dict__")
        with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
            item.extra = 1
        if item is configs[0]:
            with pytest.raises(TypeError):
                type(item)(kind="unexpected")
        elif item is configs[1]:
            with pytest.raises(TypeError):
                type(item)(
                    record_interpretation=FiniteRecordInterpretation.PERIODIC,
                    kind="unexpected",
                )
        else:
            with pytest.raises(TypeError):
                type(item)(kind="unexpected")
    with pytest.raises(FrozenInstanceError):
        configs[0].sidedness = Sidedness.ONE_SIDED
    with pytest.raises(FrozenInstanceError):
        del configs[0].sidedness
    with pytest.raises(FrozenInstanceError):
        configs[1].record_interpretation = FiniteRecordInterpretation.TRANSIENT
    with pytest.raises(FrozenInstanceError):
        del configs[1].record_interpretation


def test_enum_fields_reject_values_outside_their_enum() -> None:
    from gragra.results import (
        AmplitudeSpectrumRepresentation,
        FiniteRecordInterpretation,
        FourierTransformRepresentation,
        Sidedness,
    )

    for value in ("one_sided", FiniteRecordInterpretation.PERIODIC, None, 1):
        with pytest.raises(TypeError):
            FourierTransformRepresentation(sidedness=value)
    for value in ("periodic", Sidedness.ONE_SIDED, None, 1):
        with pytest.raises(TypeError):
            AmplitudeSpectrumRepresentation(record_interpretation=value)
    with pytest.raises(TypeError):
        AmplitudeSpectrumRepresentation()
    with pytest.raises(TypeError):
        FourierTransformRepresentation(Sidedness.ONE_SIDED)
    with pytest.raises(TypeError):
        AmplitudeSpectrumRepresentation(FiniteRecordInterpretation.PERIODIC)


def test_descriptors_are_value_objects_with_distinct_types() -> None:
    from gragra.results import (
        AmplitudeSpectrumRepresentation,
        FiniteRecordInterpretation,
        FourierTransformRepresentation,
        PhasorRepresentation,
        Sidedness,
    )

    a = FourierTransformRepresentation(sidedness=Sidedness.ONE_SIDED)
    b = FourierTransformRepresentation(sidedness=Sidedness.ONE_SIDED)
    assert a == b and hash(a) == hash(b)
    assert a != FourierTransformRepresentation()
    assert PhasorRepresentation() == PhasorRepresentation()
    assert PhasorRepresentation() != FourierTransformRepresentation()
    assert AmplitudeSpectrumRepresentation(
        record_interpretation=FiniteRecordInterpretation.PERIODIC
    ) != AmplitudeSpectrumRepresentation(
        record_interpretation=FiniteRecordInterpretation.TRANSIENT
    )


def test_union_is_a_six_descriptor_type_alias() -> None:
    from gragra.results import (
        AmplitudeSpectrumRepresentation,
        ASDRepresentation,
        CSDRepresentation,
        FourierTransformRepresentation,
        PhasorRepresentation,
        PSDRepresentation,
        SpectralRepresentation,
    )

    assert set(get_args(SpectralRepresentation)) == {
        FourierTransformRepresentation,
        AmplitudeSpectrumRepresentation,
        PhasorRepresentation,
        PSDRepresentation,
        ASDRepresentation,
        CSDRepresentation,
    }


def test_private_module_uses_only_python_311_standard_library_syntax() -> None:
    import ast
    from pathlib import Path

    repository_root = Path(__file__).resolve().parents[1]
    source = (repository_root / "src/gragra/results/_spectral.py").read_text()
    tree = ast.parse(source, feature_version=(3, 11))
    allowed = {"dataclasses", "enum", "typing"}
    imports = {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert imports <= allowed
