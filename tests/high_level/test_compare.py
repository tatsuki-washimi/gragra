"""Contract tests for alignment and compatibility of C5 result comparisons."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gragra._evaluation_setup import DiscreteSource
from gragra.evaluation import EvaluationSetup
from gragra.geometry._frame import Frame
from gragra.geometry._sphere import Sphere
from gragra.results import PotentialReference
from gragra.results._comparison import compare
from gragra.results._evaluation import EvaluationResult
from gragra.targets._high_points import TargetPoints


def _make_result(
    *,
    root: Frame | None = None,
    ids: tuple[str, ...] = ("A", "B"),
    positions: u.Quantity | None = None,
    target_frame: Frame | None = None,
    output_frame: Frame | None = None,
    observables: tuple[str, ...] = ("potential", "acceleration", "gradient"),
    values: dict[str, u.Quantity] | None = None,
    potential_reference: PotentialReference | None = None,
    source: bool = False,
    backend: str = "numpy",
) -> EvaluationResult:
    from gragra._evaluation_setup import (
        EvaluationRecord,
        ResolvedEvaluation,
        UniformSphereModelSpec,
    )
    from gragra.results import GravityResult

    root = root or Frame("world")
    target_frame = target_frame or root
    output_frame = output_frame or root
    if positions is None:
        positions = (
            np.arange(1, len(ids) + 1, dtype=float)[:, None]
            * np.array([[0.0, 0.0, 1.0]])
            * u.m
        )
    sphere = Sphere(np.zeros(3) * u.m, 1 * u.m, root)
    model = UniformSphereModelSpec(sphere, 1 * u.kg / u.m**3)
    if potential_reference is not None:
        from gragra.results import Contribution, GravityFieldDefinition

        object.__setattr__(
            model,
            "field_definition",
            GravityFieldDefinition.source_field(
                "uniform_sphere", Contribution("total"), potential_reference
            ),
        )
    targets = TargetPoints(positions, ids, target_frame)
    discrete_source = None
    if source:
        discrete_source = DiscreteSource(
            sphere,
            model.density,
            root,
            [[0.0, 0.0, 0.0]],
            [1.0],
            {"spacing_m": 0.5, "max_points": 1000, "grid_points": 8},
        )
    evaluation = ResolvedEvaluation(
        model,
        targets,
        root,
        output_frame,
        observables,
        discrete_source,
    )
    if values is None:
        defaults = {
            "potential": np.ones(len(ids)) * u.m**2 / u.s**2,
            "acceleration": np.ones((len(ids), 3)) * u.m / u.s**2,
            "gradient": np.ones((len(ids), 3, 3)) / u.s**2,
        }
        values = {name: defaults[name] for name in observables}
    return EvaluationResult(
        values=GravityResult(
            potential=values.get("potential"),
            acceleration=values.get("acceleration"),
            gradient=values.get("gradient"),
        ),
        evaluation=EvaluationSetup(evaluation),
        record=EvaluationRecord(backend=backend, versions={"solver": "1"}),
    )


def test_compare_reorders_ids_converts_units_and_summarizes_each_norm() -> None:
    root = Frame("shared world")
    reference = _make_result(
        root=root,
        ids=("A", "B"),
        positions=np.array([[0, 0, 10], [0, 0, 20]]) * u.m,
        values={
            "potential": np.array([2.0, 4.0]) * u.m**2 / u.s**2,
            "acceleration": np.array([[3, 4, 0], [0, 0, 2]]) * u.m / u.s**2,
            "gradient": np.array([[[1, 0, 0], [0, 2, 0], [0, 0, 2]], np.zeros((3, 3))])
            / u.s**2,
        },
    )
    candidate = _make_result(
        root=root,
        ids=("B", "A"),
        positions=np.array([[0, 0, 2000], [0, 0, 1000]]) * u.cm,
        values={
            "potential": np.array([42000, 21000]) * u.cm**2 / u.s**2,
            "acceleration": np.array([[0, 0, 240], [330, 400, 0]]) * u.cm / u.s**2,
            "gradient": np.array(
                [
                    [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[1.1, 0, 0], [0, 2.2, 0], [0, 0, 2.2]],
                ]
            )
            * u.Hz**2,
        },
        source=True,
        backend="different-backend",
    )

    compared = compare(reference=reference, candidate=candidate)
    assert compared.reference is reference
    assert compared.candidate is candidate
    assert compared.summary["target_id"].tolist() == ["A", "B"]
    np.testing.assert_allclose(
        compared.differences.potential.to_value(u.m**2 / u.s**2), [0.1, 0.2]
    )
    np.testing.assert_allclose(
        compared.differences.acceleration.to_value(u.m / u.s**2),
        [[0.3, 0, 0], [0, 0, 0.4]],
    )
    np.testing.assert_allclose(
        compared.summary["potential_absolute_difference"].to_value(u.m**2 / u.s**2),
        [0.1, 0.2],
    )
    np.testing.assert_allclose(
        compared.summary["potential_relative_difference"].value, [0.05, 0.05]
    )
    np.testing.assert_allclose(
        compared.summary["acceleration_absolute_difference"].to_value(u.m / u.s**2),
        [0.3, 0.4],
    )
    np.testing.assert_allclose(
        compared.summary["acceleration_relative_difference"].value, [0.06, 0.2]
    )
    np.testing.assert_allclose(
        compared.summary["gradient_absolute_difference"].to_value(1 / u.s**2),
        [0.3, 1.0],
    )
    gradient_relative = compared.summary["gradient_relative_difference"].value
    assert gradient_relative[0] == pytest.approx(0.1)
    assert np.isnan(gradient_relative[1])
    assert compared.summary["gradient_relative_defined"].tolist() == [True, False]
    assert "NaN" in compared.summary.meta["relative_difference_zero_reference"]


def test_compare_requires_exact_observable_set_and_selects_explicitly() -> None:
    root = Frame("world")
    reference = _make_result(root=root, observables=("potential",))
    candidate = _make_result(root=root)
    with pytest.raises(ValueError, match="sets must match exactly"):
        compare(reference=reference, candidate=candidate)
    compared = compare(reference=reference, candidate=candidate.select("potential"))
    assert compared.differences.acceleration is None
    assert compared.differences.gradient is None


def test_compare_accepts_positions_within_explicit_si_tolerance() -> None:
    root = Frame("world")
    reference = _make_result(
        root=root,
        observables=("potential",),
        ids=("A",),
        positions=np.array([[0, 0, 20]]) * u.m,
    )
    candidate = _make_result(
        root=root,
        observables=("potential",),
        ids=("A",),
        positions=np.array([[0, 0, 2000 + 0.5e-10]]) * u.cm,
    )
    compare(reference=reference, candidate=candidate)


def test_compare_rejects_disconnected_roots_even_when_names_match() -> None:
    reference = _make_result(root=Frame("world"), observables=("potential",))
    candidate = _make_result(root=Frame("world"), observables=("potential",))
    with pytest.raises(ValueError, match="connected root"):
        compare(reference=reference, candidate=candidate)


def test_compare_rejects_potential_reference_mismatch() -> None:
    root = Frame("world")
    reference = _make_result(
        root=root,
        observables=("potential",),
        potential_reference=PotentialReference.SPHERICAL_MEAN_ZERO_AT_INFINITY,
    )
    candidate = _make_result(
        root=root,
        observables=("potential",),
        potential_reference=PotentialReference.HALFSPACE_FOURIER_GREEN_POSITIVE_Q,
    )

    with pytest.raises(ValueError, match="potential_reference"):
        compare(reference=reference, candidate=candidate)


def test_compare_rejects_target_id_or_position_mismatch() -> None:
    root = Frame("world")
    reference = _make_result(root=root, observables=("potential",))
    wrong_ids = _make_result(root=root, observables=("potential",), ids=("A", "C"))
    with pytest.raises(ValueError, match="IDs must match exactly"):
        compare(reference=reference, candidate=wrong_ids)

    shifted = _make_result(
        root=root,
        observables=("potential",),
        positions=np.array([[0, 0, 1], [0, 0, 2 + 1e-9]]) * u.m,
    )
    with pytest.raises(ValueError, match="target positions differ"):
        compare(reference=reference, candidate=shifted)


def test_compare_requires_matching_vector_output_basis_but_scalar_does_not() -> None:
    root = Frame("world")
    angle = np.pi / 2
    rotated = Frame(
        "rotated output",
        basis=np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        ),
        parent=root,
    )
    reference_scalar = _make_result(
        root=root, output_frame=root, observables=("potential",)
    )
    candidate_scalar = _make_result(
        root=root, output_frame=rotated, observables=("potential",)
    )
    compare(reference=reference_scalar, candidate=candidate_scalar)

    reference_vector = _make_result(
        root=root, output_frame=root, observables=("acceleration",)
    )
    candidate_vector = _make_result(
        root=root, output_frame=rotated, observables=("acceleration",)
    )
    with pytest.raises(ValueError, match="common output_frame"):
        compare(reference=reference_vector, candidate=candidate_vector)


def test_compare_uses_stable_norms_for_large_and_small_finite_vectors() -> None:
    root = Frame("world")
    reference = _make_result(
        root=root,
        ids=("large", "small"),
        observables=("acceleration",),
        positions=np.array([[0, 0, 1], [0, 0, 2]]) * u.m,
        values={
            "acceleration": np.array(
                [[1e200, 1e200, 0], [1e-200, 1e-200, 0]], dtype=float
            )
            * u.m
            / u.s**2
        },
    )
    candidate = _make_result(
        root=root,
        ids=("large", "small"),
        observables=("acceleration",),
        positions=np.array([[0, 0, 1], [0, 0, 2]]) * u.m,
        values={
            "acceleration": np.array(
                [[1.1e200, 1e200, 0], [2e-200, 1e-200, 0]], dtype=float
            )
            * u.m
            / u.s**2
        },
    )
    compared = compare(reference=reference, candidate=candidate)
    absolute = compared.summary["acceleration_absolute_difference"].to_value(
        u.m / u.s**2
    )
    assert np.isfinite(absolute).all()
    np.testing.assert_allclose(absolute, [1e199, 1e-200], rtol=1e-14)
    assert compared.summary["acceleration_relative_defined"].tolist() == [True, True]
    np.testing.assert_allclose(
        compared.summary["acceleration_relative_difference"].value,
        [0.1 / np.sqrt(2), 1 / np.sqrt(2)],
        rtol=1e-14,
    )


def test_compare_promotes_integer_payloads_before_subtraction() -> None:
    root = Frame("world")
    largest = np.iinfo(np.int64).max
    reference = _make_result(
        root=root,
        ids=("A",),
        observables=("potential",),
        positions=np.array([[0, 0, 1]]) * u.m,
        values={"potential": np.array([-largest], dtype=np.int64) * u.m**2 / u.s**2},
    )
    candidate = _make_result(
        root=root,
        ids=("A",),
        observables=("potential",),
        positions=np.array([[0, 0, 1]]) * u.m,
        values={"potential": np.array([largest], dtype=np.int64) * u.m**2 / u.s**2},
    )
    compared = compare(reference=reference, candidate=candidate)
    expected = float(largest) - float(-largest)
    assert compared.differences.potential.value[0] == pytest.approx(expected)


def test_empty_results_compare_without_float_indexing_failure() -> None:
    root = Frame("world")
    reference = _make_result(
        root=root,
        ids=(),
        positions=np.empty((0, 3)) * u.m,
        observables=("potential", "acceleration", "gradient"),
    )
    candidate = _make_result(
        root=root,
        ids=(),
        positions=np.empty((0, 3)) * u.cm,
        observables=("potential", "acceleration", "gradient"),
    )
    compared = compare(reference=reference, candidate=candidate)
    assert len(compared.summary) == 0
    assert compared.differences.potential.shape == (0,)
