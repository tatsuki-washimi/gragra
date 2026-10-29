"""Tests for background density advection in unstructured displacement fields."""

import numpy as np
import pytest

from gragra.fields.unstructured import UnstructuredDisplacementField


class AffineRho0Model:
    """Helper background density model with spatial affine variation."""

    def __init__(self, rho_c, gx=0.0, gy=0.0, gz=0.0):
        self.rho_c = rho_c
        self.gx = gx
        self.gy = gy
        self.gz = gz

    def __call__(self, pts):
        return (
            self.rho_c + self.gx * pts[:, 0] + self.gy * pts[:, 1] + self.gz * pts[:, 2]
        )

    def gradient(self, pts):
        g = np.array([self.gx, self.gy, self.gz])
        return np.broadcast_to(g, pts.shape).copy()


@pytest.fixture
def simple_tetra_field():
    # Single tetrahedron mesh
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    tetra = np.array([[0, 1, 2, 3]], dtype=np.int32)
    # Displacement u: for linear tetra, nodes can have specific values
    # Let's initialize with zeros
    displacement = np.zeros((4, 3), dtype=np.float64)
    return vertices, tetra, displacement


def test_advection_affine_rho0_constant_u(simple_tetra_field):
    # ρ₀ = ρ_c + g·x, u = u_0 (constant vector)
    # div(u) = 0
    # δρ = -u₀ · ∇ρ₀ = -u_0_x * g
    vertices, tetra, _ = simple_tetra_field
    u_0 = np.array([2.0, 3.0, 4.0])
    displacement = np.broadcast_to(u_0, (4, 3)).copy()

    field = UnstructuredDisplacementField(
        vertices_m=vertices, tetra=tetra, displacement_m=displacement
    )

    rho_c = 1000.0
    gx = 50.0
    model = AffineRho0Model(rho_c, gx=gx)

    # With include_rho0_gradient_term=True
    src = field.as_density_perturbation_source(model, include_rho0_gradient_term=True)
    # Expected density perturbation: - u_0 · [gx, 0, 0] = - 2.0 * 50.0 = -100.0
    assert np.allclose(src.density_kg_m3, -100.0)


def test_advection_affine_rho0_affine_u(simple_tetra_field):
    # ρ₀ = ρ_c + g·x, u = α·r
    # div(u) = 3α
    # Centroid r_c of the tetra [0,0,0], [1,0,0], [0,1,0], [0,0,1] is [0.25, 0.25, 0.25]
    # u_centroid = α * r_c = α * [0.25, 0.25, 0.25]
    # ρ₀(r_c) = ρ_c + g * 0.25
    # δρ = -ρ₀(r_c) · 3α - α · r_c_x · g
    vertices, tetra, _ = simple_tetra_field
    alpha = 2.0
    displacement = alpha * vertices  # u = alpha * r at each vertex

    field = UnstructuredDisplacementField(
        vertices_m=vertices, tetra=tetra, displacement_m=displacement
    )

    rho_c = 1000.0
    gx = 80.0
    model = AffineRho0Model(rho_c, gx=gx)

    src = field.as_density_perturbation_source(model, include_rho0_gradient_term=True)

    r_c = np.array([0.25, 0.25, 0.25])
    rho0_rc = rho_c + gx * r_c[0]
    div_u = 3.0 * alpha
    u_c = alpha * r_c
    grad_rho0 = np.array([gx, 0.0, 0.0])

    expected_bulk = -rho0_rc * div_u
    expected_adv = -np.dot(u_c, grad_rho0)
    expected_total = expected_bulk + expected_adv

    assert np.allclose(src.density_kg_m3, expected_total)


def test_advection_matches_no_advection_when_rho0_uniform(simple_tetra_field):
    # Uniform ρ₀, i.e., gradient = 0.
    # Results should be identical to bulk only.
    vertices, tetra, _ = simple_tetra_field
    displacement = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2],
        ]
    )
    field = UnstructuredDisplacementField(
        vertices_m=vertices, tetra=tetra, displacement_m=displacement
    )

    model = AffineRho0Model(1000.0, gx=0.0)

    src_with = field.as_density_perturbation_source(
        model, include_rho0_gradient_term=True
    )
    src_without = field.as_density_perturbation_source(
        model, include_rho0_gradient_term=False
    )

    assert np.allclose(src_with.density_kg_m3, src_without.density_kg_m3)


def test_complex_displacement_advection_preserves_dtype(simple_tetra_field):
    # Displacement complex, advection term preserves complex dtype.
    vertices, tetra, _ = simple_tetra_field
    u_0 = np.array([2.0 + 1j, 3.0 - 2j, 4.0 + 3j])
    displacement = np.broadcast_to(u_0, (4, 3)).copy()

    field = UnstructuredDisplacementField(
        vertices_m=vertices, tetra=tetra, displacement_m=displacement
    )

    model = AffineRho0Model(1000.0, gx=50.0)
    src = field.as_density_perturbation_source(model, include_rho0_gradient_term=True)

    assert src.density_kg_m3.dtype == np.complex128
    # u_centroid = [2+1j, 3-2j, 4+3j]
    # grad_rho0 = [50, 0, 0]
    # δρ = - (2+1j)*50 = -100 - 50j
    assert np.allclose(src.density_kg_m3, -100.0 - 50.0j)


def test_gradient_term_without_gradient_method_raises(simple_tetra_field):
    # model is callable but doesn't have a gradient method
    vertices, tetra, displacement = simple_tetra_field
    field = UnstructuredDisplacementField(
        vertices_m=vertices, tetra=tetra, displacement_m=displacement
    )

    # Standard python function has no gradient() method
    def simple_model(pts):
        return np.ones(pts.shape[0]) * 1000.0

    with pytest.raises(TypeError, match="gradient"):
        field.as_density_perturbation_source(
            simple_model, include_rho0_gradient_term=True
        )


def test_gradient_attr_not_callable_raises(simple_tetra_field):
    # model has a gradient attribute, but it is not callable
    vertices, tetra, displacement = simple_tetra_field
    field = UnstructuredDisplacementField(
        vertices_m=vertices, tetra=tetra, displacement_m=displacement
    )

    class BadModel:
        def __call__(self, pts):
            return np.ones(pts.shape[0]) * 1000.0

        gradient = 42

    with pytest.raises(TypeError, match="gradient"):
        field.as_density_perturbation_source(
            BadModel(), include_rho0_gradient_term=True
        )


def test_scalar_rho0_with_gradient_term_true_raises(simple_tetra_field):
    # scalar + include_rho0_gradient_term=True -> TypeError
    vertices, tetra, displacement = simple_tetra_field
    field = UnstructuredDisplacementField(
        vertices_m=vertices, tetra=tetra, displacement_m=displacement
    )

    with pytest.raises(TypeError, match="gradient"):
        field.as_density_perturbation_source(1000.0, include_rho0_gradient_term=True)


def test_gradient_wrong_shape_raises(simple_tetra_field):
    # gradient() returns wrong shape (Nc,) instead of (Nc, 3)
    vertices, tetra, displacement = simple_tetra_field
    field = UnstructuredDisplacementField(
        vertices_m=vertices, tetra=tetra, displacement_m=displacement
    )

    class WrongShapeGradientModel:
        def __call__(self, pts):
            return np.ones(pts.shape[0]) * 1000.0

        def gradient(self, pts):
            return np.ones(pts.shape[0])

    with pytest.raises(ValueError, match="shape"):
        field.as_density_perturbation_source(
            WrongShapeGradientModel(), include_rho0_gradient_term=True
        )


def test_gradient_nonfinite_raises(simple_tetra_field):
    # gradient() returns infinite values
    vertices, tetra, displacement = simple_tetra_field
    field = UnstructuredDisplacementField(
        vertices_m=vertices, tetra=tetra, displacement_m=displacement
    )

    class NonFiniteGradientModel:
        def __call__(self, pts):
            return np.ones(pts.shape[0]) * 1000.0

        def gradient(self, pts):
            out = np.ones((pts.shape[0], 3))
            out[0, 0] = np.inf
            return out

    with pytest.raises(ValueError, match="finite"):
        field.as_density_perturbation_source(
            NonFiniteGradientModel(), include_rho0_gradient_term=True
        )


def test_gradient_complex_raises(simple_tetra_field):
    # gradient() returns complex values
    vertices, tetra, displacement = simple_tetra_field
    field = UnstructuredDisplacementField(
        vertices_m=vertices, tetra=tetra, displacement_m=displacement
    )

    class ComplexGradientModel:
        def __call__(self, pts):
            return np.ones(pts.shape[0]) * 1000.0

        def gradient(self, pts):
            return np.ones((pts.shape[0], 3), dtype=np.complex128) * (1.0 + 1j)

    with pytest.raises(TypeError, match="real-valued"):
        field.as_density_perturbation_source(
            ComplexGradientModel(), include_rho0_gradient_term=True
        )


def test_include_gradient_false_does_not_call_gradient(simple_tetra_field):
    # When include_rho0_gradient_term=False, gradient() should not be called
    vertices, tetra, displacement = simple_tetra_field
    field = UnstructuredDisplacementField(
        vertices_m=vertices, tetra=tetra, displacement_m=displacement
    )

    class ExplodingGradientModel:
        def __call__(self, pts):
            return np.ones(pts.shape[0]) * 1000.0

        def gradient(self, pts):
            del pts
            raise RuntimeError("Should not be called")

    # This should not raise RuntimeError because include_rho0_gradient_term=False
    field.as_density_perturbation_source(
        ExplodingGradientModel(), include_rho0_gradient_term=False
    )


def test_non_bool_include_gradient_raises(simple_tetra_field):
    # non-bool include_rho0_gradient_term -> TypeError
    vertices, tetra, displacement = simple_tetra_field
    field = UnstructuredDisplacementField(
        vertices_m=vertices, tetra=tetra, displacement_m=displacement
    )
    model = AffineRho0Model(1000.0)

    with pytest.raises(TypeError, match="bool"):
        field.as_density_perturbation_source(model, include_rho0_gradient_term=1)

    with pytest.raises(TypeError, match="bool"):
        field.as_density_perturbation_source(model, include_rho0_gradient_term="True")


def test_non_x_advection_raw_dot(simple_tetra_field):
    # g = [0, β, 0] (y component), u = [0, u_y, 0] -> δρ = -u_y * β
    vertices, tetra, _ = simple_tetra_field
    u_0 = np.array([0.0, 3.0, 0.0])
    displacement = np.broadcast_to(u_0, (4, 3)).copy()

    field = UnstructuredDisplacementField(
        vertices_m=vertices, tetra=tetra, displacement_m=displacement
    )

    rho_c = 1000.0
    gy = 40.0
    model = AffineRho0Model(rho_c, gy=gy)

    src = field.as_density_perturbation_source(model, include_rho0_gradient_term=True)
    # Expected density perturbation: - u_y * gy = - 3.0 * 40.0 = -120.0
    assert np.allclose(src.density_kg_m3, -120.0)


def test_scalar_rho0_unchanged_with_new_api(simple_tetra_field):
    # scalar float + include_rho0_gradient_term=False works and matches old API behavior
    vertices, tetra, _ = simple_tetra_field
    displacement = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2],
        ]
    )
    field = UnstructuredDisplacementField(
        vertices_m=vertices, tetra=tetra, displacement_m=displacement
    )

    src = field.as_density_perturbation_source(1000.0, include_rho0_gradient_term=False)

    # Compute expected: -rho0 * div_u
    # We can just check that it runs and returns finite results matching
    # the default call.
    src_default = field.as_density_perturbation_source(1000.0)
    assert np.allclose(src.density_kg_m3, src_default.density_kg_m3)


def test_callable_bulk_only_unchanged(simple_tetra_field):
    # callable + include_rho0_gradient_term=False matches default
    vertices, tetra, _ = simple_tetra_field
    displacement = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2],
        ]
    )
    field = UnstructuredDisplacementField(
        vertices_m=vertices, tetra=tetra, displacement_m=displacement
    )

    model = AffineRho0Model(1000.0, gx=50.0)
    src_default = field.as_density_perturbation_source(model)
    src_false = field.as_density_perturbation_source(
        model, include_rho0_gradient_term=False
    )
    assert np.allclose(src_default.density_kg_m3, src_false.density_kg_m3)


def test_empty_mesh_without_gradient_method_raises():
    # Empty mesh + include_rho0_gradient_term=True + callable without
    # gradient method -> TypeError
    vertices = np.zeros((0, 3))
    tetra = np.zeros((0, 4), dtype=np.int32)
    displacement = np.zeros((0, 3))
    field = UnstructuredDisplacementField(
        vertices_m=vertices, tetra=tetra, displacement_m=displacement
    )

    def simple_model(pts):
        return np.ones(pts.shape[0]) * 1000.0

    with pytest.raises(TypeError, match="gradient"):
        field.as_density_perturbation_source(
            simple_model, include_rho0_gradient_term=True
        )


def test_empty_mesh_scalar_raises():
    # Empty mesh + include_rho0_gradient_term=True + scalar -> TypeError
    vertices = np.zeros((0, 3))
    tetra = np.zeros((0, 4), dtype=np.int32)
    displacement = np.zeros((0, 3))
    field = UnstructuredDisplacementField(
        vertices_m=vertices, tetra=tetra, displacement_m=displacement
    )

    with pytest.raises(TypeError, match="gradient"):
        field.as_density_perturbation_source(1000.0, include_rho0_gradient_term=True)
