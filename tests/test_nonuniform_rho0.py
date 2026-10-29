import numpy as np
import pytest

from gragra.fields.regular_grid import DisplacementGrid
from gragra.fields.unstructured import UnstructuredDisplacementField


def test_grid_nonuniform_rho0_analytic_linear_profile():
    # Grid: non-uniform spacing to test np.gradient with actual axes
    x = np.array([0.0, 1.0, 3.0, 6.0, 10.0])
    y = np.array([0.0, 2.0, 5.0, 9.0, 14.0])
    z = np.array([0.0, 1.5, 3.5, 6.0, 9.0])
    nx, ny, nz = len(x), len(y), len(z)

    # Displacement u = [alpha * x, 0, 0]
    # u_x = alpha * x
    alpha = 0.02
    disp = np.zeros((nx, ny, nz, 3))
    for i in range(nx):
        disp[i, :, :, 0] = alpha * x[i]

    # rho0(x, y, z) = rho_0 + beta * x
    # grad(rho0) = [beta, 0, 0]
    rho_0 = 1000.0
    beta = 50.0
    rho0_arr = np.zeros((nx, ny, nz))
    for i in range(nx):
        rho0_arr[i, :, :] = rho_0 + beta * x[i]

    grid = DisplacementGrid(x_m=x, y_m=y, z_m=z, displacement_m=disp)
    density_grid = grid.as_density_perturbation_grid(rho0_arr)

    # Analytic delta_rho = -div(rho0 * u) = -rho0 * div(u) - u . grad(rho0)
    # div(u) = alpha
    # u . grad(rho0) = u_x * beta = alpha * x * beta
    # delta_rho = - (rho_0 + beta * x) * alpha - (alpha * x) * beta
    #           = - alpha * rho_0 - 2 * alpha * beta * x

    # We will verify using the grid's delta_density_kg_m3
    expected = np.zeros((nx, ny, nz))
    for i in range(nx):
        expected[i, :, :] = -alpha * rho_0 - 2.0 * alpha * beta * x[i]

    # Note that np.gradient with edge_order=2 uses finite differences.
    # Because u_x is linear in x, its gradient (div_u) is exactly alpha.
    # Because rho0 is linear in x, its gradient is exactly beta.
    # So the numerical result should match the analytic result exactly
    # (to machine precision).
    np.testing.assert_allclose(
        density_grid.delta_density_kg_m3, expected, rtol=1e-12, atol=1e-12
    )


def test_grid_uniform_rho0_array_matches_scalar():
    x = np.linspace(0.0, 4.0, 5)
    y = np.linspace(0.0, 4.0, 5)
    z = np.linspace(0.0, 4.0, 5)
    nx, ny, nz = len(x), len(y), len(z)

    # Converging/diverging displacement to make it non-zero
    disp = np.zeros((nx, ny, nz, 3))
    disp[..., 0] = np.sin(x)[:, None, None]

    grid = DisplacementGrid(x_m=x, y_m=y, z_m=z, displacement_m=disp)

    rho0_scalar = 1200.0
    rho0_arr = np.ones((nx, ny, nz)) * rho0_scalar

    res_scalar = grid.as_density_perturbation_grid(rho0_scalar)
    res_array = grid.as_density_perturbation_grid(rho0_arr)

    np.testing.assert_allclose(
        res_array.delta_density_kg_m3,
        res_scalar.delta_density_kg_m3,
        rtol=1e-12,
        atol=1e-12,
    )


def test_grid_complex_displacement_real_rho0_array_preserves_complex_dtype():
    x = np.linspace(0.0, 4.0, 5)
    y = np.linspace(0.0, 4.0, 5)
    z = np.linspace(0.0, 4.0, 5)
    disp = np.zeros((5, 5, 5, 3), dtype=np.complex128)
    disp[..., 0] = np.sin(x)[:, None, None] * (1.0 + 2.0j)

    grid = DisplacementGrid(x_m=x, y_m=y, z_m=z, displacement_m=disp)
    rho0_arr = np.ones((5, 5, 5)) * 1000.0

    res = grid.as_density_perturbation_grid(rho0_arr)
    assert res.delta_density_kg_m3.dtype == np.complex128


def test_grid_rho0_array_shape_mismatch_raises():
    x = np.linspace(0.0, 2.0, 3)
    y = np.linspace(0.0, 2.0, 3)
    z = np.linspace(0.0, 2.0, 3)
    disp = np.zeros((3, 3, 3, 3))
    grid = DisplacementGrid(x_m=x, y_m=y, z_m=z, displacement_m=disp)

    # 1D array
    with pytest.raises(ValueError, match="must be 3-D"):
        grid.as_density_perturbation_grid(np.array([1000.0, 1000.0, 1000.0]))

    # Wrong 3D shape
    with pytest.raises(ValueError, match="must match grid dimensions"):
        grid.as_density_perturbation_grid(np.zeros((3, 3, 4)))


def test_grid_rho0_array_complex_raises():
    x = np.linspace(0.0, 2.0, 3)
    y = np.linspace(0.0, 2.0, 3)
    z = np.linspace(0.0, 2.0, 3)
    disp = np.zeros((3, 3, 3, 3))
    grid = DisplacementGrid(x_m=x, y_m=y, z_m=z, displacement_m=disp)

    rho0_complex = np.ones((3, 3, 3), dtype=np.complex128)
    with pytest.raises(TypeError, match="must be real-valued"):
        grid.as_density_perturbation_grid(rho0_complex)


def test_grid_rho0_array_bool_raises():
    x = np.linspace(0.0, 2.0, 3)
    y = np.linspace(0.0, 2.0, 3)
    z = np.linspace(0.0, 2.0, 3)
    disp = np.zeros((3, 3, 3, 3))
    grid = DisplacementGrid(x_m=x, y_m=y, z_m=z, displacement_m=disp)

    rho0_bool = np.ones((3, 3, 3), dtype=bool)
    with pytest.raises(TypeError, match="must not contain bool"):
        grid.as_density_perturbation_grid(rho0_bool)


def test_grid_rho0_array_nonfinite_raises():
    x = np.linspace(0.0, 2.0, 3)
    y = np.linspace(0.0, 2.0, 3)
    z = np.linspace(0.0, 2.0, 3)
    disp = np.zeros((3, 3, 3, 3))
    grid = DisplacementGrid(x_m=x, y_m=y, z_m=z, displacement_m=disp)

    rho0_inf = np.ones((3, 3, 3))
    rho0_inf[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="must be finite"):
        grid.as_density_perturbation_grid(rho0_inf)


def test_grid_rho0_array_negative_raises():
    x = np.linspace(0.0, 2.0, 3)
    y = np.linspace(0.0, 2.0, 3)
    z = np.linspace(0.0, 2.0, 3)
    disp = np.zeros((3, 3, 3, 3))
    grid = DisplacementGrid(x_m=x, y_m=y, z_m=z, displacement_m=disp)

    rho0_neg = np.ones((3, 3, 3))
    rho0_neg[0, 0, 0] = -1.0
    with pytest.raises(ValueError, match="must be non-negative"):
        grid.as_density_perturbation_grid(rho0_neg)


def test_grid_rho0_scalar_zero_dim_ndarray_accepted():
    x = np.linspace(0.0, 2.0, 3)
    y = np.linspace(0.0, 2.0, 3)
    z = np.linspace(0.0, 2.0, 3)
    disp = np.zeros((3, 3, 3, 3))
    grid = DisplacementGrid(x_m=x, y_m=y, z_m=z, displacement_m=disp)

    res = grid.as_density_perturbation_grid(np.array(1000.0))
    # Should work and produce uniform scalar path result
    assert np.allclose(res.delta_density_kg_m3, 0.0)


def test_unstructured_callable_rho0_bulk_scales_linearly():
    # Simple tetrahedron
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    # Displacement
    disp = np.zeros((4, 3))
    disp[1, 0] = 0.01  # expand x

    tetra = np.array([[0, 1, 2, 3]])

    field = UnstructuredDisplacementField(
        vertices_m=vertices, displacement_m=disp, tetra=tetra
    )

    rho0_val = 1500.0
    res_scalar = field.as_density_perturbation_source(rho0_val)

    # Callable that returns double density
    def rho0_callable(centroids):
        # centroids shape (Nc, 3)
        assert centroids.ndim == 2
        assert centroids.shape[1] == 3
        return np.ones(centroids.shape[0]) * (rho0_val * 2.0)

    res_callable = field.as_density_perturbation_source(rho0_callable)

    np.testing.assert_allclose(
        res_callable.density_kg_m3,
        res_scalar.density_kg_m3 * 2.0,
        rtol=1e-12,
        atol=1e-12,
    )


def test_unstructured_callable_not_called_for_empty_tetra():
    vertices = np.zeros((0, 3))
    disp = np.zeros((0, 3))
    tetra = np.zeros((0, 4), dtype=int)

    field = UnstructuredDisplacementField(
        vertices_m=vertices, displacement_m=disp, tetra=tetra
    )

    called = False

    def rho0_callable(_centroids):
        nonlocal called
        called = True
        return np.array([], dtype=np.float64)

    res = field.as_density_perturbation_source(rho0_callable)
    assert not called
    assert len(res.density_kg_m3) == 0


def test_unstructured_callable_wrong_length_raises():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    disp = np.zeros((4, 3))
    tetra = np.array([[0, 1, 2, 3]])

    field = UnstructuredDisplacementField(
        vertices_m=vertices, displacement_m=disp, tetra=tetra
    )

    # returns 2 elements instead of 1
    def rho0_callable(_centroids):
        return np.array([1000.0, 1000.0])

    with pytest.raises(ValueError, match="must return shape"):
        field.as_density_perturbation_source(rho0_callable)


def test_unstructured_callable_returns_scalar_raises():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    disp = np.zeros((4, 3))
    tetra = np.array([[0, 1, 2, 3]])

    field = UnstructuredDisplacementField(
        vertices_m=vertices, displacement_m=disp, tetra=tetra
    )

    # returns scalar instead of array (shape mismatch (1,))
    def rho0_callable(_centroids):
        return 1000.0

    with pytest.raises(ValueError, match="must return shape"):
        field.as_density_perturbation_source(rho0_callable)


def test_unstructured_callable_complex_output_raises():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    disp = np.zeros((4, 3))
    tetra = np.array([[0, 1, 2, 3]])

    field = UnstructuredDisplacementField(
        vertices_m=vertices, displacement_m=disp, tetra=tetra
    )

    def rho0_callable(_centroids):
        return np.array([1000.0 + 1j])

    with pytest.raises(TypeError, match="must be real-valued"):
        field.as_density_perturbation_source(rho0_callable)


def test_unstructured_callable_nonfinite_output_raises():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    disp = np.zeros((4, 3))
    tetra = np.array([[0, 1, 2, 3]])

    field = UnstructuredDisplacementField(
        vertices_m=vertices, displacement_m=disp, tetra=tetra
    )

    def rho0_callable(_centroids):
        return np.array([np.inf])

    with pytest.raises(ValueError, match="must be finite"):
        field.as_density_perturbation_source(rho0_callable)


def test_unstructured_callable_negative_output_raises():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    disp = np.zeros((4, 3))
    tetra = np.array([[0, 1, 2, 3]])

    field = UnstructuredDisplacementField(
        vertices_m=vertices, displacement_m=disp, tetra=tetra
    )

    def rho0_callable(_centroids):
        return np.array([-1000.0])

    with pytest.raises(ValueError, match="must be non-negative"):
        field.as_density_perturbation_source(rho0_callable)


def test_unstructured_empty_tetra_invalid_rho0_raises():
    vertices = np.zeros((0, 3))
    disp = np.zeros((0, 3))
    tetra = np.zeros((0, 4), dtype=int)

    field = UnstructuredDisplacementField(
        vertices_m=vertices, displacement_m=disp, tetra=tetra
    )

    # invalid scalar
    with pytest.raises(ValueError, match="must be non-negative"):
        field.as_density_perturbation_source(-1000.0)

    # bool
    with pytest.raises(TypeError, match="must not contain bool"):
        field.as_density_perturbation_source(True)

    # complex
    with pytest.raises(TypeError, match="must be a scalar|must be real-valued"):
        field.as_density_perturbation_source(1000.0 + 1j)

    # invalid non-scalar/object
    with pytest.raises(TypeError, match="must be a scalar"):
        field.as_density_perturbation_source(object())
