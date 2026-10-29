"""Tests for the native HDF5 field importer."""

import subprocess
import sys

import numpy as np
import pytest

from gragra.adapters.hdf5_field import (
    DifferentiableDiscreteBackgroundDensity,
    DiscreteBackgroundDensity,
    from_hdf5_batched_weights,
    from_hdf5_displacement_grid,
    from_hdf5_field,
    from_hdf5_unstructured_field,
    iter_hdf5_field_snapshots,
    read_hdf5_background_density,
)
from gragra.fields import DisplacementGrid, UnstructuredDisplacementField
from gragra.fields.background import DifferentiableBackgroundDensityModel
from gragra.fields.batch import (
    batched_unstructured_weights,
)
from gragra.observables.acceleration import point_acceleration
from gragra.targets.points import TargetPoints


def _h5py():
    """Helper to check if h5py is installed and skip otherwise."""
    return pytest.importorskip("h5py")


@pytest.fixture
def unstructured_data():
    """Simple linear tetra mesh and displacement data."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    connectivity = np.array([[0, 1, 2, 3]], dtype=np.int64)
    displacement = np.array(
        [
            [0.01, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [0.01, 0.01, 0.0],
            [0.01, 0.0, 0.01],
        ],
        dtype=np.float64,
    )
    return vertices, connectivity, displacement


@pytest.fixture
def structured_data():
    """Grid axes and displacement data (Nx=Ny=Nz=3)."""
    x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    y = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    z = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    displacement = np.zeros((3, 3, 3, 3), dtype=np.float64)
    displacement[1, 1, 1] = [0.05, 0.02, 0.01]
    return x, y, z, displacement


def test_from_hdf5_unstructured_roundtrip(tmp_path, unstructured_data):
    """Test reading and writing unstructured mesh data."""
    h5py = _h5py()
    vertices, connectivity, displacement = unstructured_data
    file_path = tmp_path / "unstructured.h5"

    # Write synthetic HDF5 file
    with h5py.File(file_path, "w") as f:
        f.attrs["mesh_type"] = "unstructured"
        f.create_dataset("vertices_m", data=vertices)
        f.create_dataset("connectivity", data=connectivity)
        f.create_dataset("displacement_m", data=displacement)

    # Read back and verify
    field = from_hdf5_unstructured_field(str(file_path))
    assert np.allclose(field.vertices_m, vertices)
    assert np.allclose(field.tetra, connectivity)
    assert np.allclose(field.displacement_m, displacement)


def test_from_hdf5_displacement_grid_roundtrip(tmp_path, structured_data):
    """Test reading and writing structured grid data."""
    h5py = _h5py()
    x, y, z, displacement = structured_data
    file_path = tmp_path / "structured.h5"

    # Write synthetic HDF5 file
    with h5py.File(file_path, "w") as f:
        f.attrs["mesh_type"] = "structured"
        f.create_dataset("x_m", data=x)
        f.create_dataset("y_m", data=y)
        f.create_dataset("z_m", data=z)
        f.create_dataset("displacement_m", data=displacement)

    # Read back and verify
    grid = from_hdf5_displacement_grid(str(file_path))
    assert np.allclose(grid.x_m, x)
    assert np.allclose(grid.y_m, y)
    assert np.allclose(grid.z_m, z)
    assert np.allclose(grid.displacement_m, displacement)


def test_from_hdf5_field_dispatch(tmp_path, unstructured_data, structured_data):
    """Test automatic dispatching in from_hdf5_field."""
    h5py = _h5py()
    vertices, connectivity, displacement = unstructured_data
    file_path_unstruct = tmp_path / "unstruct_dispatch.h5"

    with h5py.File(file_path_unstruct, "w") as f:
        f.attrs["mesh_type"] = "unstructured"
        f.create_dataset("vertices_m", data=vertices)
        f.create_dataset("connectivity", data=connectivity)
        f.create_dataset("displacement_m", data=displacement)

    field1 = from_hdf5_field(str(file_path_unstruct))
    assert isinstance(field1, UnstructuredDisplacementField)

    x, y, z, displacement_grid = structured_data
    file_path_struct = tmp_path / "struct_dispatch.h5"

    with h5py.File(file_path_struct, "w") as f:
        f.attrs["mesh_type"] = "structured"
        f.create_dataset("x_m", data=x)
        f.create_dataset("y_m", data=y)
        f.create_dataset("z_m", data=z)
        f.create_dataset("displacement_m", data=displacement_grid)

    field2 = from_hdf5_field(str(file_path_struct))
    assert isinstance(field2, DisplacementGrid)


def test_complex_displacement_dtype_preserved(tmp_path, unstructured_data):
    """Test that complex displacement keeps complex128 type."""
    h5py = _h5py()
    vertices, connectivity, displacement = unstructured_data
    displacement_complex = displacement + 1j * 0.05 * displacement
    file_path = tmp_path / "complex.h5"

    with h5py.File(file_path, "w") as f:
        f.attrs["mesh_type"] = "unstructured"
        f.create_dataset("vertices_m", data=vertices)
        f.create_dataset("connectivity", data=connectivity)
        f.create_dataset("displacement_m", data=displacement_complex)

    field = from_hdf5_unstructured_field(str(file_path))
    assert field.displacement_m.dtype == np.complex128
    assert np.allclose(field.displacement_m, displacement_complex)


def test_schema_violations(tmp_path, unstructured_data):
    """Test that schema violations raise appropriate exceptions."""
    h5py = _h5py()
    vertices, connectivity, displacement = unstructured_data

    # Missing mesh_type
    file_path_missing_type = tmp_path / "missing_type.h5"
    with h5py.File(file_path_missing_type, "w") as f:
        f.create_dataset("vertices_m", data=vertices)
    with pytest.raises(ValueError, match="mesh_type"):
        from_hdf5_field(str(file_path_missing_type))

    # Invalid mesh_type value
    file_path_invalid_type = tmp_path / "invalid_type.h5"
    with h5py.File(file_path_invalid_type, "w") as f:
        f.attrs["mesh_type"] = "invalid"
    with pytest.raises(ValueError, match="mesh_type"):
        from_hdf5_field(str(file_path_invalid_type))

    # Missing dataset
    file_path_missing_ds = tmp_path / "missing_ds.h5"
    with h5py.File(file_path_missing_ds, "w") as f:
        f.attrs["mesh_type"] = "unstructured"
        f.create_dataset("vertices_m", data=vertices)
    with pytest.raises(ValueError, match="connectivity"):
        from_hdf5_unstructured_field(str(file_path_missing_ds))


def test_background_density_protocol_and_classes(
    tmp_path, unstructured_data, structured_data
):
    """Test reading background density and verifying protocol matching."""
    h5py = _h5py()
    vertices, connectivity, displacement = unstructured_data
    file_path = tmp_path / "density_test.h5"

    rho0_scalar = 1200.0
    rho0_array = np.array([1250.0], dtype=np.float64)  # Nc = 1
    grad_array = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)  # (Nc, 3)

    # 1. No density dataset
    with h5py.File(file_path, "w") as f:
        f.attrs["mesh_type"] = "unstructured"
    assert read_hdf5_background_density(str(file_path)) is None

    # 2. Scalar density
    with h5py.File(file_path, "w") as f:
        f.attrs["mesh_type"] = "unstructured"
        f.create_dataset("rho0_kg_m3", data=rho0_scalar)
    assert read_hdf5_background_density(str(file_path)) == rho0_scalar

    # 3. Array density without gradient
    with h5py.File(file_path, "w") as f:
        f.attrs["mesh_type"] = "unstructured"
        f.create_dataset("vertices_m", data=vertices)
        f.create_dataset("connectivity", data=connectivity)
        f.create_dataset("rho0_kg_m3", data=rho0_array)

    model = read_hdf5_background_density(str(file_path))
    assert isinstance(model, DiscreteBackgroundDensity)
    assert not isinstance(model, DifferentiableBackgroundDensityModel)

    # Validate __call__ with matching points
    conn = connectivity
    centroids = (
        vertices[conn[:, 0]]
        + vertices[conn[:, 1]]
        + vertices[conn[:, 2]]
        + vertices[conn[:, 3]]
    ) / 4.0
    assert np.allclose(model(centroids), rho0_array)

    # ValueError on centroid mismatch
    with pytest.raises(ValueError, match="tolerance"):
        model(centroids + 1.0)
    with pytest.raises(ValueError, match="count"):
        model(np.zeros((5, 3)))

    # 4. Array density with gradient
    with h5py.File(file_path, "w") as f:
        f.attrs["mesh_type"] = "unstructured"
        f.create_dataset("vertices_m", data=vertices)
        f.create_dataset("connectivity", data=connectivity)
        f.create_dataset("rho0_kg_m3", data=rho0_array)
        f.create_dataset("grad_rho0_kg_m4", data=grad_array)

    diff_model = read_hdf5_background_density(str(file_path))
    assert isinstance(diff_model, DifferentiableDiscreteBackgroundDensity)
    assert isinstance(diff_model, DifferentiableBackgroundDensityModel)
    assert np.allclose(diff_model(centroids), rho0_array)
    assert np.allclose(diff_model.gradient(centroids), grad_array)

    with pytest.raises(ValueError, match="tolerance"):
        diff_model.gradient(centroids + 1.0)

    # 5. Scalar density + gradient coexist -> ValueError
    with h5py.File(file_path, "w") as f:
        f.attrs["mesh_type"] = "unstructured"
        f.create_dataset("rho0_kg_m3", data=rho0_scalar)
        f.create_dataset("grad_rho0_kg_m4", data=grad_array)

    with pytest.raises(ValueError, match="scalar"):
        read_hdf5_background_density(str(file_path))

    # 6. Structured background density returns raw np.ndarray
    x, y, z, displacement_grid = structured_data
    rho0_grid = np.full((3, 3, 3), 1500.0, dtype=np.float64)
    with h5py.File(file_path, "w") as f:
        f.attrs["mesh_type"] = "structured"
        f.create_dataset("displacement_m", data=displacement_grid)
        f.create_dataset("rho0_kg_m3", data=rho0_grid)

    struct_rho = read_hdf5_background_density(str(file_path))
    assert isinstance(struct_rho, np.ndarray)
    assert struct_rho.shape == (3, 3, 3)
    assert np.allclose(struct_rho, rho0_grid)


def test_end_to_end_observable_equivalence(tmp_path, unstructured_data):
    """Test HDF5 import produces equivalent results to direct setup."""
    h5py = _h5py()
    vertices, connectivity, displacement = unstructured_data
    file_path = tmp_path / "e2e.h5"

    rho0_val = 1800.0

    # Write unstructured field to HDF5
    with h5py.File(file_path, "w") as f:
        f.attrs["mesh_type"] = "unstructured"
        f.create_dataset("vertices_m", data=vertices)
        f.create_dataset("connectivity", data=connectivity)
        f.create_dataset("displacement_m", data=displacement)

    # Direct setup
    direct_field = UnstructuredDisplacementField(
        vertices_m=vertices, tetra=connectivity, displacement_m=displacement
    )
    direct_source = direct_field.as_density_perturbation_source(rho0_val)

    # Importer setup
    imported_field = from_hdf5_field(str(file_path))
    imported_source = imported_field.as_density_perturbation_source(rho0_val)

    # Verify sources match
    assert np.allclose(imported_source.positions_m, direct_source.positions_m)
    assert np.allclose(imported_source.volumes_m3, direct_source.volumes_m3)
    assert np.allclose(imported_source.density_kg_m3, direct_source.density_kg_m3)

    # Verify observable acceleration match
    targets = TargetPoints(np.array([[2.0, 2.0, 2.0]]))
    direct_acc = point_acceleration(direct_source.as_weighted_source(), targets)
    imported_acc = point_acceleration(imported_source.as_weighted_source(), targets)
    assert np.allclose(imported_acc, direct_acc)


def test_end_to_end_advection_equivalence(tmp_path, unstructured_data):
    """Test unstructured advection equivalence between importer and direct setup."""
    h5py = _h5py()
    vertices, connectivity, displacement = unstructured_data
    file_path = tmp_path / "e2e_advection.h5"

    rho0_array = np.array([1250.0], dtype=np.float64)  # Nc = 1
    grad_array = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)  # (Nc, 3)

    # Write unstructured field + density with gradient to HDF5
    with h5py.File(file_path, "w") as f:
        f.attrs["mesh_type"] = "unstructured"
        f.create_dataset("vertices_m", data=vertices)
        f.create_dataset("connectivity", data=connectivity)
        f.create_dataset("displacement_m", data=displacement)
        f.create_dataset("rho0_kg_m3", data=rho0_array)
        f.create_dataset("grad_rho0_kg_m4", data=grad_array)

    # Direct setup
    conn = connectivity
    centroids = (
        vertices[conn[:, 0]]
        + vertices[conn[:, 1]]
        + vertices[conn[:, 2]]
        + vertices[conn[:, 3]]
    ) / 4.0
    direct_model = DifferentiableDiscreteBackgroundDensity(
        centroids_m=centroids,
        rho0_kg_m3=rho0_array,
        grad_rho0_kg_m4=grad_array,
    )
    direct_field = UnstructuredDisplacementField(
        vertices_m=vertices, tetra=connectivity, displacement_m=displacement
    )
    direct_source = direct_field.as_density_perturbation_source(
        direct_model, include_rho0_gradient_term=True
    )

    # Importer setup
    imported_field = from_hdf5_field(str(file_path))
    imported_model = read_hdf5_background_density(str(file_path))
    imported_source = imported_field.as_density_perturbation_source(
        imported_model, include_rho0_gradient_term=True
    )

    # Verify sources match
    assert np.allclose(imported_source.positions_m, direct_source.positions_m)
    assert np.allclose(imported_source.volumes_m3, direct_source.volumes_m3)
    assert np.allclose(imported_source.density_kg_m3, direct_source.density_kg_m3)

    # Verify observable acceleration match
    targets = TargetPoints(np.array([[2.0, 2.0, 2.0]]))
    direct_acc = point_acceleration(direct_source.as_weighted_source(), targets)
    imported_acc = point_acceleration(imported_source.as_weighted_source(), targets)
    assert np.allclose(imported_acc, direct_acc)


def test_immutability(tmp_path, unstructured_data):
    """Test that imported field and background density arrays are read-only."""
    h5py = _h5py()
    vertices, connectivity, displacement = unstructured_data
    file_path = tmp_path / "immutable.h5"

    rho0_array = np.array([1250.0], dtype=np.float64)

    with h5py.File(file_path, "w") as f:
        f.attrs["mesh_type"] = "unstructured"
        f.create_dataset("vertices_m", data=vertices)
        f.create_dataset("connectivity", data=connectivity)
        f.create_dataset("displacement_m", data=displacement)
        f.create_dataset("rho0_kg_m3", data=rho0_array)

    field = from_hdf5_field(str(file_path))
    assert not field.vertices_m.flags.writeable
    assert not field.tetra.flags.writeable
    assert not field.displacement_m.flags.writeable

    density = read_hdf5_background_density(str(file_path))
    assert not density.centroids_m.flags.writeable
    assert not density.rho0_kg_m3.flags.writeable


def test_length_scale_scaling(tmp_path, unstructured_data):
    """Test length_scale_m scaling on coords/displacement, not density."""
    h5py = _h5py()
    vertices, connectivity, displacement = unstructured_data
    file_path = tmp_path / "scale.h5"

    rho0_array = np.array([1000.0], dtype=np.float64)

    # mm coordinates/displacement in HDF5 (1000x smaller unit value)
    # e.g., coordinate 1m is written as 1000mm.
    # To convert mm to meters, length_scale_m = 0.001.
    vertices_mm = vertices * 1000.0
    displacement_mm = displacement * 1000.0

    with h5py.File(file_path, "w") as f:
        f.attrs["mesh_type"] = "unstructured"
        f.create_dataset("vertices_m", data=vertices_mm)
        f.create_dataset("connectivity", data=connectivity)
        f.create_dataset("displacement_m", data=displacement_mm)
        f.create_dataset("rho0_kg_m3", data=rho0_array)

    # Import with scale factor
    field = from_hdf5_unstructured_field(str(file_path), length_scale_m=0.001)
    assert np.allclose(field.vertices_m, vertices)
    assert np.allclose(field.displacement_m, displacement)

    # Read density with same scale factor
    density = read_hdf5_background_density(str(file_path), length_scale_m=0.001)
    # Centroids should be correctly scaled to match field centroids
    # but density value rho0 itself is not scaled.
    expected_centroids = (
        vertices[connectivity[:, 0]]
        + vertices[connectivity[:, 1]]
        + vertices[connectivity[:, 2]]
        + vertices[connectivity[:, 3]]
    ) / 4.0
    assert np.allclose(density.centroids_m, expected_centroids)
    assert np.allclose(density.rho0_kg_m3, rho0_array)


def test_structured_gradient_raises_value_error(tmp_path, structured_data):
    """Test structured density with grad_rho0_kg_m4 raises ValueError."""
    h5py = _h5py()
    x, y, z, displacement_grid = structured_data
    rho0_grid = np.full((3, 3, 3), 1500.0, dtype=np.float64)
    grad_array = np.zeros((3, 3, 3, 3), dtype=np.float64)
    file_path = tmp_path / "struct_grad_error.h5"

    with h5py.File(file_path, "w") as f:
        f.attrs["mesh_type"] = "structured"
        f.create_dataset("displacement_m", data=displacement_grid)
        f.create_dataset("rho0_kg_m3", data=rho0_grid)
        f.create_dataset("grad_rho0_kg_m4", data=grad_array)

    msg = (
        "grad_rho0_kg_m4 is not supported for structured meshes; "
        "the gradient is computed internally via np.gradient"
    )
    with pytest.raises(ValueError, match=msg):
        read_hdf5_background_density(str(file_path))


def test_import_core_does_not_load_h5py():
    """Importing core gragra should not load h5py."""
    code = (
        "import sys\n"
        "import gragra\n"
        "import gragra.fields.unstructured\n"
        "import gragra.adapters.hdf5_field\n"
        "assert 'h5py' not in sys.modules, f'h5py leaked: {sys.modules.keys()}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_hdf5_importer_raises_importerror_when_h5py_missing(monkeypatch):
    """Verify ImportError with pip install tip is raised if h5py is missing."""
    monkeypatch.setitem(sys.modules, "h5py", None)
    with pytest.raises(ImportError, match='pip install "gragra\\[field-io\\]"'):
        from_hdf5_field("dummy.h5")


# ---------------------------------------------------------------------------
# Phase 10-d batched HDF5 tests (10-d-2, spec: docs/design/phase10d_batch_import.md)
# ---------------------------------------------------------------------------

_TETRA_VERTS = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)
_TETRA_CONN = np.array([[0, 1, 2, 3]])
_GRID_AXIS = np.array([0.0, 1.0, 2.0])


def _write_batched_unstructured_h5(path, beta_list, extra_datasets=None):
    h5py = _h5py()
    with h5py.File(path, "w") as f:
        f.attrs["mesh_type"] = "unstructured"
        f.create_dataset("vertices_m", data=_TETRA_VERTS)
        f.create_dataset("connectivity", data=_TETRA_CONN)
        disp = np.zeros((len(beta_list), 4, 3))
        for p, beta in enumerate(beta_list):
            disp[p, :, 0] = beta * _TETRA_VERTS[:, 0]
        f.create_dataset("displacement_m", data=disp)
        for name, data in (extra_datasets or {}).items():
            f.create_dataset(name, data=data)
    return h5py


def _write_batched_structured_h5(
    path, alpha_list, rho0_dataset=None, extra_datasets=None
):
    h5py = _h5py()
    x, y, z = np.meshgrid(_GRID_AXIS, _GRID_AXIS, _GRID_AXIS, indexing="ij")
    with h5py.File(path, "w") as f:
        f.attrs["mesh_type"] = "structured"
        f.create_dataset("x_m", data=_GRID_AXIS)
        f.create_dataset("y_m", data=_GRID_AXIS)
        f.create_dataset("z_m", data=_GRID_AXIS)
        disp = np.stack(
            [
                np.stack([alpha * x, alpha * y, alpha * z], axis=-1)
                for alpha in alpha_list
            ],
            axis=0,
        )
        f.create_dataset("displacement_m", data=disp)
        if rho0_dataset is not None:
            f.create_dataset("rho0_kg_m3", data=rho0_dataset)
        for name, data in (extra_datasets or {}).items():
            f.create_dataset(name, data=data)
    return h5py


def test_h1_batched_unstructured_roundtrip(tmp_path):
    """T3 golden via HDF5 roundtrip."""
    path = tmp_path / "batched_unstructured.h5"
    _write_batched_unstructured_h5(path, [0.5, -1.0])
    positions, weights = from_hdf5_batched_weights(str(path), rho0_kg_m3=2.0)
    np.testing.assert_allclose(weights[0], [-1.0 / 6.0, 1.0 / 3.0], rtol=1e-12)
    np.testing.assert_allclose(positions[0], [0.25, 0.25, 0.25])


def test_h2_batched_structured_roundtrip(tmp_path):
    """T5 golden via HDF5 roundtrip."""
    path = tmp_path / "batched_structured.h5"
    _write_batched_structured_h5(path, [0.01, 0.02])
    positions, weights = from_hdf5_batched_weights(str(path), rho0_kg_m3=1.0)
    np.testing.assert_allclose(weights[:, 0], np.full(27, -0.03), rtol=1e-12)
    np.testing.assert_allclose(weights[:, 1], np.full(27, -0.06), rtol=1e-12)


def test_h3_in_memory_path_matches_batched_hdf5(tmp_path):
    """File layout -> in-memory conversion must be bit-identical to the
    fields.batch in-memory helper given the same data."""
    path = tmp_path / "batched_unstructured.h5"
    beta_list = [0.5, -1.0]
    _write_batched_unstructured_h5(path, beta_list)
    positions_h5, weights_h5 = from_hdf5_batched_weights(str(path), rho0_kg_m3=2.0)

    disp = np.zeros((2, 4, 3))
    for p, beta in enumerate(beta_list):
        disp[p, :, 0] = beta * _TETRA_VERTS[:, 0]
    positions_mem, weights_mem = batched_unstructured_weights(
        _TETRA_VERTS, _TETRA_CONN, disp, 2.0
    )
    np.testing.assert_array_equal(positions_h5, positions_mem)
    np.testing.assert_array_equal(weights_h5, weights_mem)


def test_h4_snapshots_partial_selection(tmp_path):
    path = tmp_path / "batched_unstructured.h5"
    _write_batched_unstructured_h5(path, [0.5, -1.0])
    positions, weights = from_hdf5_batched_weights(
        str(path), rho0_kg_m3=2.0, snapshots=[1]
    )
    assert weights.shape == (1, 1)
    np.testing.assert_allclose(weights[0, 0], 1.0 / 3.0, rtol=1e-12)


def test_h4_snapshots_out_of_range_raises(tmp_path):
    path = tmp_path / "batched_unstructured.h5"
    _write_batched_unstructured_h5(path, [0.5, -1.0])
    with pytest.raises(ValueError):
        from_hdf5_batched_weights(str(path), rho0_kg_m3=2.0, snapshots=[5])


def test_h4_non_integer_snapshot_index_raises(tmp_path):
    """Regression (Codex review #40): a float snapshot index (e.g. 1.9) must
    not be silently truncated to a valid integer index; bool must also be
    rejected (bool is an int subclass and would otherwise pass silently)."""
    path = tmp_path / "batched_unstructured.h5"
    _write_batched_unstructured_h5(path, [0.5, -1.0])
    with pytest.raises(ValueError):
        from_hdf5_batched_weights(str(path), rho0_kg_m3=2.0, snapshots=[1.9])
    with pytest.raises(ValueError):
        from_hdf5_batched_weights(str(path), rho0_kg_m3=2.0, snapshots=[True])
    with pytest.raises(ValueError):
        list(iter_hdf5_field_snapshots(str(path), snapshots=[1.9]))


def test_h5a_explicit_argument_takes_precedence_over_dataset(tmp_path):
    path = tmp_path / "batched_structured.h5"
    # Dataset says rho0=999.0 (would give very different weights); explicit
    # argument rho0=1.0 must be the one actually used.
    _write_batched_structured_h5(path, [0.01, 0.02], rho0_dataset=999.0)
    _, weights = from_hdf5_batched_weights(str(path), rho0_kg_m3=1.0)
    np.testing.assert_allclose(weights[:, 0], np.full(27, -0.03), rtol=1e-12)


def test_h5b_none_falls_back_to_file_dataset(tmp_path):
    path = tmp_path / "batched_structured.h5"
    _write_batched_structured_h5(path, [0.01, 0.02], rho0_dataset=1.0)
    _, weights = from_hdf5_batched_weights(str(path), rho0_kg_m3=None)
    np.testing.assert_allclose(weights[:, 0], np.full(27, -0.03), rtol=1e-12)


def test_h5b_none_without_dataset_raises(tmp_path):
    path = tmp_path / "batched_structured.h5"
    _write_batched_structured_h5(path, [0.01, 0.02], rho0_dataset=None)
    with pytest.raises(ValueError):
        from_hdf5_batched_weights(str(path), rho0_kg_m3=None)


def test_h5c_batch_dependent_rho0_argument_raises(tmp_path):
    path = tmp_path / "batched_structured.h5"
    _write_batched_structured_h5(path, [0.01, 0.02])
    bad_rho0 = np.ones((2, 3, 3, 3))  # (P, Nx, Ny, Nz) -- unsupported
    with pytest.raises(ValueError):
        from_hdf5_batched_weights(str(path), rho0_kg_m3=bad_rho0)


def test_h5d_grid_shape_uses_axes_not_displacement_shape(tmp_path):
    """Regression (Codex review #39): read_hdf5_background_density must not
    derive structured grid shape from displacement_m.shape[:3], which for a
    batched (P,Nx,Ny,Nz,3) array yields (P,Nx,Ny) and would wrongly reject a
    valid (Nx,Ny,Nz) rho0 dataset."""
    path = tmp_path / "batched_structured.h5"
    _write_batched_structured_h5(
        path, [0.01, 0.02], rho0_dataset=np.full((3, 3, 3), 1.0)
    )
    _, weights = from_hdf5_batched_weights(str(path), rho0_kg_m3=None)
    np.testing.assert_allclose(weights[:, 0], np.full(27, -0.03), rtol=1e-12)


def test_h5e_scalar_ndarray_rho0_is_accepted(tmp_path):
    """Regression (Codex review #40): a 0-D ndarray scalar (e.g.
    np.array(1.0)) must be accepted the same as a plain float, not rejected
    by the (Nx,Ny,Nz) shape check meant for spatially-varying rho0 arrays."""
    path = tmp_path / "batched_structured.h5"
    _write_batched_structured_h5(path, [0.01, 0.02])
    _, weights = from_hdf5_batched_weights(str(path), rho0_kg_m3=np.array(1.0))
    np.testing.assert_allclose(weights[:, 0], np.full(27, -0.03), rtol=1e-12)


def test_h9_empty_batch_raises(tmp_path):
    """Regression (Codex review #40): a leading snapshot axis of size 0 must
    raise ValueError instead of silently yielding zero snapshots."""
    path = tmp_path / "batched_unstructured.h5"
    _write_batched_unstructured_h5(path, [])
    with pytest.raises(ValueError, match="P >= 1"):
        from_hdf5_batched_weights(str(path), rho0_kg_m3=1.0)
    with pytest.raises(ValueError, match="P >= 1"):
        list(iter_hdf5_field_snapshots(str(path)))


def test_h6_coordinate_metadata_ignored(tmp_path):
    path_plain = tmp_path / "plain.h5"
    path_with_meta = tmp_path / "with_meta.h5"
    _write_batched_unstructured_h5(path_plain, [0.5, -1.0])
    _write_batched_unstructured_h5(
        path_with_meta,
        [0.5, -1.0],
        extra_datasets={
            "time_s": np.zeros(999),
            "frequency_hz": np.array([np.nan, np.inf]),
        },
    )
    pos_plain, w_plain = from_hdf5_batched_weights(str(path_plain), rho0_kg_m3=2.0)
    pos_meta, w_meta = from_hdf5_batched_weights(str(path_with_meta), rho0_kg_m3=2.0)
    np.testing.assert_array_equal(pos_plain, pos_meta)
    np.testing.assert_array_equal(w_plain, w_meta)


def test_h7_single_snapshot_unaffected(tmp_path, unstructured_data):
    """Existing single-snapshot import must be bit-for-bit unaffected by the
    batched-import extension (no shared-logic regressions)."""
    h5py = _h5py()
    vertices, connectivity, displacement = unstructured_data
    path = tmp_path / "single.h5"
    with h5py.File(path, "w") as f:
        f.attrs["mesh_type"] = "unstructured"
        f.create_dataset("vertices_m", data=vertices)
        f.create_dataset("connectivity", data=connectivity)
        f.create_dataset("displacement_m", data=displacement)
    field = from_hdf5_unstructured_field(str(path))
    assert np.allclose(field.displacement_m, displacement)


def test_h8_complex_batched_roundtrip(tmp_path):
    h5py = _h5py()
    path = tmp_path / "batched_complex.h5"
    with h5py.File(path, "w") as f:
        f.attrs["mesh_type"] = "unstructured"
        f.create_dataset("vertices_m", data=_TETRA_VERTS)
        f.create_dataset("connectivity", data=_TETRA_CONN)
        disp = np.zeros((2, 4, 3), dtype=np.complex128)
        disp[0, :, 0] = (0.5 + 0.1j) * _TETRA_VERTS[:, 0]
        disp[1, :, 0] = (-1.0 + 0.2j) * _TETRA_VERTS[:, 0]
        f.create_dataset("displacement_m", data=disp)
    _, weights = from_hdf5_batched_weights(str(path), rho0_kg_m3=2.0)
    assert weights.dtype == np.complex128


def test_h9_read_only(tmp_path):
    path = tmp_path / "batched_unstructured.h5"
    _write_batched_unstructured_h5(path, [0.5, -1.0])
    positions, weights = from_hdf5_batched_weights(str(path), rho0_kg_m3=2.0)
    assert positions.flags.writeable is False
    assert weights.flags.writeable is False


def test_iter_hdf5_field_snapshots_rejects_single_snapshot_file(
    tmp_path, unstructured_data
):
    h5py = _h5py()
    vertices, connectivity, displacement = unstructured_data
    path = tmp_path / "single.h5"
    with h5py.File(path, "w") as f:
        f.attrs["mesh_type"] = "unstructured"
        f.create_dataset("vertices_m", data=vertices)
        f.create_dataset("connectivity", data=connectivity)
        f.create_dataset("displacement_m", data=displacement)
    with pytest.raises(ValueError):
        list(iter_hdf5_field_snapshots(str(path)))
