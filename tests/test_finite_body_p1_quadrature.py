"""Selectable higher/local-order tetra/face quadrature for P1 finite bodies.

Covers ``gragra.geometry.tetra_quadrature`` in isolation and the
``quadrature_order`` option on ``piecewise_tetra_p1_density_sources``. The
single most important property under test is that the default (no
``quadrature_order`` argument, or an explicit ``None``) call path stays
bit-identical to the fixed classical degree-2 rule that existed before this
option was added -- see ``test_default_quadrature_order_is_bit_identical_*``.
"""

import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from gragra.fields.finite_body_p1 import (
    _TETRA_BARYCENTRIC,
    _TRIANGLE_BARYCENTRIC,
    compare_tetra_p1_density_representations,
    piecewise_tetra_p1_density_sources,
)
from gragra.fields.unstructured import UnstructuredDisplacementField
from gragra.geometry.tetra_quadrature import (
    tetra_fixed_degree2,
    tetra_gauss_quadrature,
    triangle_fixed_degree2,
    triangle_gauss_quadrature,
)
from scripts.research_wave4_finite_body import build_cartesian_six_tetra_mesh
from tests._finite_body_p1_reference import displacement
from tests._provenance_contract import verify_content_addressed_provenance
from tests.oracles import generate_finite_body_affine_oracle as _affine_oracle

_AFFINE_ORACLE_JSON = (
    Path(__file__).resolve().parent / "oracles" / "data" / "finite_body_affine_v1.json"
)

# ---------------------------------------------------------------------------
# gragra.geometry.tetra_quadrature
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", [2, 3, 4, 6, 8])
def test_gauss_quadrature_weights_and_barycentric_rows_sum_to_one(order):
    for barycentric, weights in (
        tetra_gauss_quadrature(order),
        triangle_gauss_quadrature(order),
    ):
        assert np.sum(weights) == pytest.approx(1.0, rel=1e-12, abs=1e-13)
        np.testing.assert_allclose(np.sum(barycentric, axis=1), 1.0, rtol=1e-12)
        assert np.all(weights >= 0.0)


@pytest.mark.parametrize("order", [2, 3, 5])
def test_gauss_quadrature_point_counts(order):
    bary, weights = tetra_gauss_quadrature(order)
    assert bary.shape == (order**3, 4)
    assert weights.shape == (order**3,)
    bary2, weights2 = triangle_gauss_quadrature(order)
    assert bary2.shape == (order**2, 3)
    assert weights2.shape == (order**2,)


@pytest.mark.parametrize("order", [0, 1, -1])
def test_gauss_quadrature_rejects_orders_below_two(order):
    with pytest.raises(ValueError):
        tetra_gauss_quadrature(order)
    with pytest.raises(ValueError):
        triangle_gauss_quadrature(order)


def test_fixed_degree2_matches_finite_body_p1_constants():
    bary, weights = tetra_fixed_degree2()
    np.testing.assert_array_equal(bary, _TETRA_BARYCENTRIC)
    np.testing.assert_array_equal(weights, np.full(4, 0.25))

    bary, weights = triangle_fixed_degree2()
    np.testing.assert_array_equal(bary, _TRIANGLE_BARYCENTRIC)
    np.testing.assert_array_equal(weights, np.full(3, 1.0 / 3.0))


def test_quadrature_rule_outputs_are_readonly():
    for barycentric, weights in (
        tetra_fixed_degree2(),
        triangle_fixed_degree2(),
        tetra_gauss_quadrature(4),
        triangle_gauss_quadrature(4),
    ):
        assert not barycentric.flags.writeable
        assert not weights.flags.writeable


# ---------------------------------------------------------------------------
# piecewise_tetra_p1_density_sources(..., quadrature_order=...)
# ---------------------------------------------------------------------------


def _field(cube_vertices, cube_tetra):
    return UnstructuredDisplacementField(
        cube_vertices, cube_tetra, displacement(cube_vertices)
    )


def test_default_quadrature_order_is_bit_identical_to_no_argument(
    cube_vertices, cube_tetra
):
    field = _field(cube_vertices, cube_tetra)
    rho = np.full((6, 4), 2.0)
    baseline = piecewise_tetra_p1_density_sources(field, rho)
    explicit_none = piecewise_tetra_p1_density_sources(
        field, rho, quadrature_order=None
    )
    for name in (
        "mass_positions_m",
        "mass_kg",
        "mass_displacement_m",
        "mass_cell_index",
        "interface_triangles",
        "interface_density_jump_kg_m3",
        "interface_quadrature_face_index",
        "negative_side_cell",
        "positive_side_cell",
    ):
        np.testing.assert_array_equal(
            getattr(baseline, name), getattr(explicit_none, name)
        )
    np.testing.assert_array_equal(
        baseline.bulk_source.density_kg_m3, explicit_none.bulk_source.density_kg_m3
    )
    np.testing.assert_array_equal(
        baseline.interface_source.weights_kg, explicit_none.interface_source.weights_kg
    )


def test_default_path_uses_the_exact_fixed_degree2_rule(cube_vertices, cube_tetra):
    """Locks the default (no argument) path to the classical rule bit-for-bit."""
    field = _field(cube_vertices, cube_tetra)
    rho = np.full((6, 4), 2.0)
    result = piecewise_tetra_p1_density_sources(field, rho)

    bary, _weights = tetra_fixed_degree2()
    nodes = cube_vertices[cube_tetra]
    expected_positions = np.einsum("ql,nli->nqi", bary, nodes).reshape(-1, 3)
    np.testing.assert_array_equal(result.mass_positions_m, expected_positions)
    assert result.mass_positions_m.shape == (6 * 4, 3)
    assert result.mass_kg.shape == (6 * 4,)
    assert np.all(result.mass_kg > 0.0)


@pytest.mark.parametrize("bad", [0, 1, -3])
def test_quadrature_order_requires_at_least_two(cube_vertices, cube_tetra, bad):
    field = _field(cube_vertices, cube_tetra)
    rho = np.full((6, 4), 2.0)
    with pytest.raises(ValueError):
        piecewise_tetra_p1_density_sources(field, rho, quadrature_order=bad)


@pytest.mark.parametrize(
    "bad",
    [
        True,
        np.array([True, False, True, False, True, False]),
        2.5,
        np.array([2.0, 3.0, 4.0, 2.0, 3.0, 4.0]),
    ],
)
def test_quadrature_order_rejects_bool_and_non_integer(cube_vertices, cube_tetra, bad):
    field = _field(cube_vertices, cube_tetra)
    rho = np.full((6, 4), 2.0)
    with pytest.raises(TypeError):
        piecewise_tetra_p1_density_sources(field, rho, quadrature_order=bad)


def test_quadrature_order_rejects_wrong_shape(cube_vertices, cube_tetra):
    field = _field(cube_vertices, cube_tetra)
    rho = np.full((6, 4), 2.0)
    with pytest.raises(ValueError):
        piecewise_tetra_p1_density_sources(
            field, rho, quadrature_order=np.array([4, 4, 4])
        )


def test_uniform_higher_order_point_counts(cube_vertices, cube_tetra):
    field = _field(cube_vertices, cube_tetra)
    rho = np.full((6, 4), 2.0)
    order = 3
    result = piecewise_tetra_p1_density_sources(field, rho, quadrature_order=order)
    points_per_cell = order**3
    assert result.mass_positions_m.shape == (6 * points_per_cell, 3)
    np.testing.assert_array_equal(
        result.mass_cell_index, np.repeat(np.arange(6), points_per_cell)
    )
    nf = len(result.interface_triangles)
    points_per_face = order**2
    np.testing.assert_array_equal(
        result.interface_quadrature_face_index,
        np.repeat(np.arange(nf), points_per_face),
    )
    assert result.interface_source.positions_m.shape == (nf * points_per_face, 3)


def test_local_per_cell_order_matches_manual_per_cell_slice(cube_vertices, cube_tetra):
    field = _field(cube_vertices, cube_tetra)
    rho = np.full((6, 4), 2.0)
    orders = np.array([2, 2, 2, 2, 6, 6])
    result = piecewise_tetra_p1_density_sources(field, rho, quadrature_order=orders)

    points_per_cell = np.where(orders == 2, 2**3, 6**3)
    np.testing.assert_array_equal(
        result.mass_cell_index, np.repeat(np.arange(6), points_per_cell)
    )
    assert result.mass_positions_m.shape[0] == int(np.sum(points_per_cell))

    uniform_high = piecewise_tetra_p1_density_sources(field, rho, quadrature_order=6)
    start = int(np.sum(points_per_cell[:4]))
    stop = start + int(points_per_cell[4])
    uniform_start = 4 * (6**3)
    uniform_stop = uniform_start + 6**3
    np.testing.assert_allclose(
        result.mass_positions_m[start:stop],
        uniform_high.mass_positions_m[uniform_start:uniform_stop],
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        result.mass_kg[start:stop],
        uniform_high.mass_kg[uniform_start:uniform_stop],
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        result.mass_displacement_m[start:stop],
        uniform_high.mass_displacement_m[uniform_start:uniform_stop],
        rtol=1e-13,
    )


def test_face_order_uses_max_of_owning_cells(cube_vertices, cube_tetra):
    field = _field(cube_vertices, cube_tetra)
    rho = np.full((6, 4), 2.0)
    high_cell = 5
    orders = np.full(6, 2)
    orders[high_cell] = 6
    result = piecewise_tetra_p1_density_sources(field, rho, quadrature_order=orders)

    for face_index, (neg, pos) in enumerate(
        zip(result.negative_side_cell, result.positive_side_cell, strict=True)
    ):
        touches_high = neg == high_cell or pos == high_cell
        expected_points = (6 if touches_high else 2) ** 2
        actual_points = int(
            np.count_nonzero(result.interface_quadrature_face_index == face_index)
        )
        assert actual_points == expected_points


def test_uniform_order_sweep_self_converges_near_field(cube_vertices, cube_tetra):
    """Mirrors docs/research/wave4-6-F.md stage 4/5: raising Gauss order (not
    subcells) shrinks the residual against a much finer self-reference."""
    field = _field(cube_vertices, cube_tetra)
    rho = np.full((6, 4), 2.0)
    near_target = np.array([[1.35, 0.2, -0.15]])
    orders = (2, 4, 8, 16)
    responses = {}
    for order in orders:
        sources = piecewise_tetra_p1_density_sources(field, rho, quadrature_order=order)
        report = compare_tetra_p1_density_representations(
            sources, near_target, observable="gradient"
        )
        responses[order] = report.mass_displacement_response[0]

    finest = responses[orders[-1]]
    errors = [np.linalg.norm(responses[order] - finest) for order in orders[:-1]]
    assert errors[0] > errors[1] > errors[2]
    assert errors[0] / errors[2] > 5.0


def test_empty_mesh_with_quadrature_order():
    empty = UnstructuredDisplacementField(
        np.empty((0, 3)), np.empty((0, 4), dtype=np.int64), np.empty((0, 3))
    )
    result = piecewise_tetra_p1_density_sources(
        empty, np.empty((0, 4)), quadrature_order=4
    )
    assert result.mass_positions_m.shape == (0, 3)
    assert result.interface_triangles.shape == (0, 3)


def test_complex_displacement_higher_order_dtype(cube_vertices, cube_tetra):
    field = UnstructuredDisplacementField(
        cube_vertices, cube_tetra, (1 + 2j) * displacement(cube_vertices)
    )
    rho = np.full((6, 4), 2.0)
    result = piecewise_tetra_p1_density_sources(field, rho, quadrature_order=4)
    assert result.mass_displacement_m.dtype == np.complex128
    assert result.interface_source.weights_kg.dtype == np.complex128
    assert result.mass_kg.dtype == np.float64


def test_higher_order_outputs_are_readonly(cube_vertices, cube_tetra):
    field = _field(cube_vertices, cube_tetra)
    rho = np.full((6, 4), 2.0)
    result = piecewise_tetra_p1_density_sources(field, rho, quadrature_order=3)
    for item in fields(result):
        value = getattr(result, item.name)
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable
    for array in (
        result.bulk_source.positions_m,
        result.bulk_source.volumes_m3,
        result.interface_source.positions_m,
        result.interface_source.weights_kg,
    ):
        assert not array.flags.writeable


# ---------------------------------------------------------------------------
# Independent affine oracle (tests/oracles/generate_finite_body_affine_oracle.py)
# ---------------------------------------------------------------------------


def test_finite_body_affine_oracle_provenance_is_verified():
    """Fail-closed: catches a generator edit or hand-edited JSON that wasn't
    followed by regeneration (same contract as the halfspace oracle, see
    ``tests/_provenance_contract.py``)."""
    data = json.loads(_AFFINE_ORACLE_JSON.read_text())
    verify_content_addressed_provenance(
        data,
        oracle_name="finite_body_affine_v1",
        generator_path=Path(_affine_oracle.__file__),
        derivation_input_sha256=_affine_oracle.derivation_input_sha256(),
    )


def _affine_oracle_field_and_density():
    mesh = build_cartesian_six_tetra_mesh(
        np.array(_affine_oracle.BOX_BOUNDS_M),
        1.0,
        host_density_kg_m3=_affine_oracle.DENSITY_RHO0_KG_M3,
    )
    field = UnstructuredDisplacementField(
        mesh.vertices_m, mesh.tetra, _affine_oracle.displacement_m(mesh.vertices_m)
    )
    density_nodes = _affine_oracle.density_kg_m3(mesh.vertices_m[mesh.tetra])
    return field, density_nodes


@pytest.mark.parametrize("observable", ["potential", "acceleration", "gradient"])
def test_high_order_quadrature_converges_to_the_affine_oracle(observable):
    """Item (a) of the ⑥-B task: a globally-affine density/displacement box
    has zero P1 interpolation error on any mesh, so at sufficiently high
    quadrature order both production representations must agree with the
    independent, from-scratch oracle to near machine precision -- not just
    approximately."""
    oracle = json.loads(_AFFINE_ORACLE_JSON.read_text())
    field, density_nodes = _affine_oracle_field_and_density()
    targets = np.array(_affine_oracle.TARGETS_M)

    high_sources = piecewise_tetra_p1_density_sources(
        field, density_nodes, quadrature_order=8
    )
    high_report = compare_tetra_p1_density_representations(
        high_sources, targets, observable=observable, chunk_size=8192
    )
    default_sources = piecewise_tetra_p1_density_sources(field, density_nodes)
    default_report = compare_tetra_p1_density_representations(
        default_sources, targets, observable=observable, chunk_size=8192
    )

    accepted = np.stack(
        [
            np.asarray(point["observables"][observable]["accepted_value"]["values"])
            for point in oracle["points"]
        ]
    )

    high_mass_error = np.abs(high_report.mass_displacement_response - accepted)
    high_combined_error = np.abs(high_report.bulk_interface_response - accepted)
    default_mass_error = np.abs(default_report.mass_displacement_response - accepted)

    scale = max(float(np.max(np.abs(accepted))), np.finfo(float).tiny)
    assert np.max(high_mass_error) / scale < 1e-10
    assert np.max(high_combined_error) / scale < 1e-10
    # order=8 must be dramatically closer to the independent oracle than the
    # existing fixed degree-2 rule -- the whole point of the new option.
    assert np.max(high_mass_error) < np.max(default_mass_error) * 1e-3
