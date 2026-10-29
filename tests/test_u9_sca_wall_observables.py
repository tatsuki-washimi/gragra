"""U9 physics, API connection, and strict input regression tests."""

import importlib
import subprocess
import sys

import numpy as np
import pytest

from tests.oracles.u9_sca_wall_observables_reference import (
    acceptance,
    fixtures,
    review_points,
    surface,
    wall_reference,
)

SHAPES = ("sphere", "cuboid", "cylinder")
GEOMETRY = {
    "sphere": {"radius_m": 2.0},
    "cuboid": {"half_lengths_m": [2.0, 3.0, 5.0]},
    "cylinder": {"radius_m": 2.0, "height_m": 6.0},
}
U = 1e-9 * np.array([1 + 0.4j, 2 - 0.7j, -1 + 0.2j])


def evaluate(shape, targets, displacement=U, **kwargs):
    module = importlib.import_module("gragra.analytic.u9_sca_wall_observables")
    function = getattr(module, shape + "_sca_wall_observables_g_free")
    settings = dict(
        center_m=[0.0, 0.0, 0.0],
        density_kg_m3=2300.0,
        displacement_m=displacement,
        **GEOMETRY[shape],
    )
    settings.update(kwargs)
    return function(targets, **settings)


def test_new_import():
    importlib.import_module("gragra.analytic.u9_sca_wall_observables")


def test_sphere_internal():
    points = fixtures("sphere")[:3]
    actual = evaluate("sphere", points)
    k = 4 * np.pi * 2300 / 3
    np.testing.assert_allclose(actual.potential, k * (points @ U), atol=1e-20)
    np.testing.assert_allclose(actual.acceleration, np.tile(-k * U, (3, 1)))
    np.testing.assert_array_equal(actual.gradient, 0)


@pytest.mark.parametrize("shape", SHAPES)
def test_surface_area_and_normals(shape):
    p, n, w = surface(shape, order=12, angular_order=40)
    expected = {"sphere": 16 * np.pi, "cuboid": 248, "cylinder": 32 * np.pi}[shape]
    np.testing.assert_allclose(w.sum(), expected)
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1)
    np.testing.assert_allclose(w @ n, 0, atol=5e-13)
    np.testing.assert_allclose(
        np.sum(w * np.einsum("ni,ni->n", p, n)),
        {"sphere": 32 * np.pi, "cuboid": 720, "cylinder": 72 * np.pi}[shape],
    )


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.slow
def test_independent_wall_integral(shape):
    points = fixtures(shape)
    samples = np.vstack((np.eye(3) * 1e-9, U))
    actual = evaluate(shape, points, samples)
    coarse = wall_reference(shape, points, samples, order=96, angular_order=384)
    fine = wall_reference(shape, points, samples, order=128, angular_order=512)
    for block, (value, ref, previous) in enumerate(
        zip(actual, fine, coarse, strict=True)
    ):
        tolerance = acceptance(ref, shape, block)
        assert np.all(np.abs(ref - previous) <= tolerance / 10), (
            "REFERENCE_NOT_CONVERGED"
        )
        assert np.all(np.abs(value - ref) <= tolerance), (
            shape,
            block,
            np.max(np.abs(value - ref) / tolerance),
        )


@pytest.mark.parametrize("shape", SHAPES)
def test_samples_complex_linearity_and_storage(shape):
    points = fixtures(shape)[:3].copy()
    samples = np.array([U, 2j * U, np.zeros(3), -U])
    before = points.copy(), samples.copy()
    output = evaluate(shape, points, samples)
    single = evaluate(shape, points)
    for value, one, tail in zip(output, single, ((), (3,), (3, 3)), strict=True):
        assert value.shape == (4, 3, *tail)
        assert value.dtype == np.complex128
        assert not value.flags.writeable
        np.testing.assert_allclose(value[0], one)
        np.testing.assert_allclose(value[1], 2j * one)
        np.testing.assert_array_equal(value[2], 0)
        np.testing.assert_allclose(value[3], -one)
    np.testing.assert_array_equal(points, before[0])
    np.testing.assert_array_equal(samples, before[1])
    assert points.flags.writeable and samples.flags.writeable
    for value in evaluate(shape, points, [1.0, 2.0, -1.0]):
        assert value.dtype == np.float64


@pytest.mark.parametrize("shape", SHAPES)
def test_existing_sca_connection(shape):
    from gragra.analytic.cylinder_sca import cylinder_sca_wall_coupling_tensor_g_free
    from gragra.analytic.sca_tensor import (
        cuboid_sca_wall_coupling_tensor_g_free,
        sphere_sca_wall_coupling_tensor_g_free,
    )

    targets = fixtures(shape)
    if shape == "sphere":
        tensor = sphere_sca_wall_coupling_tensor_g_free([0, 0, 0], 2, targets)
    elif shape == "cuboid":
        tensor = cuboid_sca_wall_coupling_tensor_g_free([0, 0, 0], [4, 6, 10], targets)
    else:
        tensor = cylinder_sca_wall_coupling_tensor_g_free([0, 0, 0], 2, 6, targets)
    np.testing.assert_allclose(
        evaluate(shape, targets).acceleration,
        -2300 * np.einsum("mij,j->mi", tensor, U),
        rtol=1e-10,
        atol=3e-17,
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_target_only_derivatives(shape):
    point = np.array([[0.4, -0.3, 0.7], [4.1, -0.8, 6.3]])
    value = evaluate(shape, point)
    step = 2e-4
    for j in range(3):
        offset = np.eye(3)[j] * step
        plus, minus = evaluate(shape, point + offset), evaluate(shape, point - offset)
        plus2, minus2 = (
            evaluate(shape, point + 2 * offset),
            evaluate(shape, point - 2 * offset),
        )
        for block in (0, 1):
            deriv = (
                -tuple(plus2)[block]
                + 8 * tuple(plus)[block]
                - 8 * tuple(minus)[block]
                + tuple(minus2)[block]
            ) / (12 * step)
            expected = (
                -value.acceleration[:, j] if block == 0 else value.gradient[:, :, j]
            )
            np.testing.assert_allclose(deriv, expected, rtol=2e-7, atol=1e-15)
    np.testing.assert_allclose(
        value.gradient, value.gradient.swapaxes(-1, -2), atol=1e-20
    )
    np.testing.assert_allclose(
        np.trace(value.gradient, axis1=-2, axis2=-1), 0, atol=1e-20
    )


def test_cylinder_axis_transverse_mixed():
    value = evaluate("cylinder", [[0, 0, 0.8]], [1e-9, 0, 0])
    assert abs(value.gradient[0, 0, 2]) > 1e-8
    np.testing.assert_allclose(value.gradient[0, 0, 2], value.gradient[0, 2, 0])


def rotation():
    a, b, c = 0.31, -0.47, 0.23
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return rz @ ry @ rx


@pytest.mark.parametrize("shape", SHAPES)
def test_rotation_translation_and_si(shape):
    q, center = rotation(), np.array([13.0, -7.0, 5.0])
    points = fixtures(shape)[:4]
    reference = evaluate(shape, points)
    options = {} if shape == "sphere" else {"rotation": q}
    actual = evaluate(shape, points @ q.T + center, U @ q.T, center_m=center, **options)
    np.testing.assert_allclose(actual.potential, reference.potential, atol=1e-19)
    np.testing.assert_allclose(
        actual.acceleration, reference.acceleration @ q.T, atol=1e-19
    )
    np.testing.assert_allclose(
        actual.gradient,
        np.einsum("ij,mjk,lk->mil", q, reference.gradient, q),
        atol=1e-19,
    )
    module = importlib.import_module("gragra.observables.u9_sca_wall_observables")
    si = getattr(module, shape + "_sca_wall_observables")(
        points,
        center_m=[0, 0, 0],
        density_kg_m3=2300,
        displacement_m=U,
        **GEOMETRY[shape],
    )
    from gragra.constants import G_SI

    for value, ref in zip(si, reference, strict=True):
        np.testing.assert_array_equal(value, G_SI * ref)
        assert not value.flags.writeable


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize(
    "bad", [True, np.bool_(False), "1", 1j, np.nan, np.inf, np.array(1, dtype=object)]
)
def test_bad_geometry_scalars(shape, bad):
    key = "half_lengths_m" if shape == "cuboid" else "radius_m"
    value = [2, bad, 5] if shape == "cuboid" else bad
    with pytest.raises((TypeError, ValueError)):
        evaluate(shape, [[0, 0, 0]], **{key: value})
    with pytest.raises((TypeError, ValueError)):
        evaluate(shape, [[0, 0, 0]], density_kg_m3=bad)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize(
    "bad",
    [
        [[True, 0, 0]],
        [["1", 0, 0]],
        [[1j, 0, 0]],
        [[np.nan, 0, 0]],
        [[np.inf, 0, 0]],
        np.zeros((1, 3), dtype=object),
        [],
        np.empty((0, 3)),
        [0, 0, 0],
        [[[0, 0, 0]]],
    ],
)
def test_bad_targets(shape, bad):
    with pytest.raises((TypeError, ValueError)):
        evaluate(shape, bad)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize(
    "bad",
    [
        [True, 0, 0],
        ["1", 0, 0],
        [np.inf, 0, 0],
        np.zeros(3, dtype=object),
        [],
        np.empty((0, 3)),
        [[1, 2]],
        [[[1, 2, 3]]],
    ],
)
def test_bad_displacements(shape, bad):
    with pytest.raises((TypeError, ValueError)):
        evaluate(shape, [[0, 0, 0]], bad)


@pytest.mark.parametrize("shape", SHAPES)
def test_zero_still_checks_boundary(shape):
    point = {"sphere": [[2, 0, 0]], "cuboid": [[2, 0, 0]], "cylinder": [[2, 0, 0]]}[
        shape
    ]
    for settings in ({"density_kg_m3": 0}, {"displacement": [0, 0, 0]}):
        with pytest.raises(ValueError, match="boundary"):
            evaluate(shape, point, **settings)
    for value in evaluate(shape, [[0, 0, 0]], density_kg_m3=0):
        np.testing.assert_array_equal(value, 0)


@pytest.mark.parametrize("shape", ("cuboid", "cylinder"))
@pytest.mark.parametrize(
    "bad",
    [
        np.diag([1, 1, -1]),
        np.eye(3) * 2,
        [[True, 0, 0], [0, 1, 0], [0, 0, 1]],
        np.eye(3).astype(str),
        np.eye(3).astype(complex),
    ],
)
def test_invalid_rotation_even_zero(shape, bad):
    with pytest.raises((TypeError, ValueError)):
        evaluate(shape, [[0, 0, 0]], [0, 0, 0], rotation=bad)


def test_numpy_only_production_import():
    script = """
import sys
class Block:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'scipy','sympy','mpmath'}:
            raise ImportError(fullname)
sys.meta_path.insert(0, Block())
from gragra.analytic.u9_sca_wall_observables import cylinder_sca_wall_observables_g_free
cylinder_sca_wall_observables_g_free([[.4,.2,.7]], center_m=[0,0,0],
    radius_m=2, height_m=6, density_kg_m3=2300, displacement_m=[1e-9,0,0])
"""
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize(
    "bad",
    [
        [True, 0, 0],
        ["1", 0, 0],
        [1j, 0, 0],
        [np.nan, 0, 0],
        [np.inf, 0, 0],
        np.zeros(3, dtype=object),
        [0, 0],
        [[0, 0, 0]],
    ],
)
def test_bad_centers(shape, bad):
    with pytest.raises((TypeError, ValueError)):
        evaluate(shape, [[0, 0, 0]], center_m=bad)


@pytest.mark.parametrize(
    "bad", [True, "6", 6j, np.nan, np.inf, 0, -1, np.array(6, dtype=object)]
)
def test_bad_cylinder_height(bad):
    with pytest.raises((TypeError, ValueError)):
        evaluate("cylinder", [[0, 0, 0]], [0, 0, 0], height_m=bad)


@pytest.mark.parametrize("shape", SHAPES)
def test_h_is_target_derivative_of_static_gamma(shape):
    from gragra.analytic.cavity_cuboid import cuboid_gradient_g_free
    from gragra.analytic.cavity_cylinder import cylinder_acceleration_gradient_g_free
    from gragra.analytic.uniform_sphere import uniform_sphere_gradient

    def static(points):
        if shape == "sphere":
            return uniform_sphere_gradient([0, 0, 0], 2, 1, points)
        if shape == "cuboid":
            return cuboid_gradient_g_free([0, 0, 0], [4, 6, 10], 1, points)
        return cylinder_acceleration_gradient_g_free([0, 0, 0], 2, 6, 1, points)[1]

    points = fixtures(shape)
    step = 4e-4
    for axis in range(3):
        shift = np.eye(3)[axis] * step
        derived = (
            -static(points + 2 * shift)
            + 8 * static(points + shift)
            - 8 * static(points - shift)
            + static(points - 2 * shift)
        ) / (12 * step)
        actual = evaluate(shape, points, np.eye(3)[axis], density_kg_m3=1).gradient
        np.testing.assert_allclose(actual, derived, rtol=2e-6, atol=2e-9)


@pytest.mark.slow
@pytest.mark.parametrize("shape", SHAPES)
def test_rotated_translated_independent_reference(shape):
    q, center = rotation(), np.array([13.0, -7.0, 5.0])
    points = fixtures(shape)[1:4] @ q.T + center
    displacement = U @ q.T
    options = {} if shape == "sphere" else {"rotation": q}
    actual = evaluate(shape, points, displacement, center_m=center, **options)
    first = wall_reference(
        shape,
        points,
        displacement,
        center=center,
        rotation=q,
        order=96,
        angular_order=384,
    )
    second = wall_reference(
        shape,
        points,
        displacement,
        center=center,
        rotation=q,
        order=128,
        angular_order=512,
    )
    for block, (value, ref, previous) in enumerate(
        zip(actual, second, first, strict=True)
    ):
        tolerance = acceptance(ref, shape, block)
        assert np.all(np.abs(previous - ref) <= tolerance / 10), (
            "REFERENCE_NOT_CONVERGED"
        )
        assert np.all(np.abs(value - ref) <= tolerance)


def test_cylinder_failure_diagnostics_are_call_local(monkeypatch):
    from gragra.analytic import u9_sca_wall_observables as module

    function = module.cylinder_third
    points = [[0, 0, 0], [0.4, -0.3, 0.7], [0, 0, 0.8]]
    evaluate("cylinder", points)

    def fail_second(radius, height, point, gamma):
        if point[0] > 0:
            raise ArithmeticError("injected cel nonconvergence")
        return function(radius, height, point, gamma)

    monkeypatch.setattr(module, "cylinder_third", fail_second)
    with pytest.raises(module.U9EvaluationError) as caught:
        evaluate("cylinder", points)
    diagnostics = caught.value.diagnostics
    assert diagnostics["failed_index"] == 1
    assert diagnostics["evaluated_indices"] == [0]
    assert diagnostics["unevaluated_indices"] == [2]
    assert diagnostics["status"] == "ANALYTIC_EVALUATION_FAILED"
    monkeypatch.setattr(module, "cylinder_third", function)
    evaluate("cylinder", points)
    assert diagnostics["status"] == "ANALYTIC_EVALUATION_FAILED"


@pytest.mark.parametrize(
    "shape,displacement",
    [
        (
            "cuboid",
            [1e-9, 2e-9, -1e-9],
        ),
        (
            "cylinder",
            [1e-9, 0, 0],
        ),
    ],
)
@pytest.mark.slow
def test_review01_converged_counterexamples(shape, displacement):
    points = review_points(shape)
    references = [
        wall_reference(shape, points, displacement, order=order, angular_order=angular)
        for order, angular in ((192, 768), (256, 768), (256, 1024))
    ]
    value = evaluate(shape, points, displacement)
    for block, (actual, ref) in enumerate(zip(value, references[-1], strict=True)):
        width = acceptance(ref, shape, block)
        for previous in (references[0][block], references[1][block]):
            difference = abs(previous - ref)
            assert np.all(difference <= width / 10), "REFERENCE_NOT_CONVERGED"
            if block:
                axes = tuple(range(-block, 0))
                norm_ref = np.linalg.norm(ref, axis=axes)
                assert np.all(
                    np.linalg.norm(difference, axis=axes)
                    <= acceptance(norm_ref, shape, block) / 10
                ), "REFERENCE_NOT_CONVERGED"
        assert np.all(abs(actual - ref) <= width), (
            shape,
            block,
            np.max(abs(actual - ref) / width),
        )
        if block:
            assert np.all(
                np.linalg.norm(actual - ref, axis=axes)
                <= acceptance(norm_ref, shape, block)
            )


@pytest.mark.parametrize(
    "shape,point",
    [
        ("cuboid", [2 + 1e-7, 3, 8]),
        ("cuboid", [2 - 1e-7, 3, 8]),
        ("cylinder", [0.002001, 0, 200]),
    ],
)
@pytest.mark.parametrize("zero", ["density", "displacement"])
def test_review01_valid_geometry_zero(shape, point, zero):
    options = (
        {"density_kg_m3": 0} if zero == "density" else {"displacement_m": [0j, 0, 0]}
    )
    for array in evaluate(shape, [point], **options):
        np.testing.assert_array_equal(array, 0)


@pytest.mark.slow
def test_review02_complex_displacement_below_far_switch():
    radial = 0.100001
    distance = 16 * np.sqrt(13) - 1e-6
    point = [[radial, 0, np.sqrt(distance**2 - radial**2)]]
    references = [
        wall_reference("cylinder", point, U, order=order, angular_order=angular)
        for order, angular in ((192, 768), (256, 768), (256, 1024))
    ]
    actual = evaluate("cylinder", point, U).gradient
    reference = references[-1][2]
    tolerance = acceptance(reference, "cylinder", 2)
    norm_tolerance = acceptance(np.linalg.norm(reference), "cylinder", 2)
    for previous in references[:-1]:
        difference = abs(previous[2] - reference)
        assert np.all(difference <= tolerance / 10), "REFERENCE_NOT_CONVERGED"
        assert np.linalg.norm(difference) <= norm_tolerance / 10
    assert np.all(abs(actual - reference) <= tolerance)
    assert np.linalg.norm(actual - reference) <= norm_tolerance
