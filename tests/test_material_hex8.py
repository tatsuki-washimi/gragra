"""Material identity and Q1 geometry/patch tests with independent expectations."""

import importlib

import numpy as np
import pytest


def api():
    return importlib.import_module("gragra.fields.material_hex8")


def cube():
    return np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )


def material(ids=("cell",), regions=(10,), mapping=None, **kwargs):
    return api().MaterialRegions(
        ids,
        np.array(regions),
        {10: 2300.0} if mapping is None else mapping,
        exterior_density_kg_m3=0.0,
        **kwargs,
    )


def test_q1_mass_centroid_and_affine_patch():
    points = cube() * [2, 3, 4] + [7, -2, 1]
    a = np.array([[0.1, 0.2, -0.3], [0.4, -0.2, 0.1], [0.2, 0.3, 0.4]])
    field = api().Hex8DisplacementField(points, np.arange(8)[None], points @ a.T)
    sources = field.sources(material(), quadrature_order=3)
    assert len(sources.mass_kg) == 27
    np.testing.assert_allclose(sources.volumes_m3.sum(), 24, rtol=2e-15)
    np.testing.assert_allclose(sources.mass_kg.sum(), 24 * 2300, rtol=2e-15)
    np.testing.assert_allclose(
        np.average(sources.positions_m, weights=sources.volumes_m3, axis=0),
        [8, -0.5, 3],
        atol=2e-15,
    )
    np.testing.assert_allclose(
        sources.displacement_m, sources.positions_m @ a.T, atol=3e-15
    )
    np.testing.assert_allclose(
        sources.bulk_source.density_kg_m3, -2300 * np.trace(a), rtol=3e-15
    )
    assert not sources.positions_m.flags.writeable
    assert not sources.mass_kg.flags.writeable


@pytest.mark.parametrize("change", ["negative", "zero", "ordering"])
def test_q1_rejects_bad_jacobian(change):
    points = cube()
    if change == "negative":
        points[:, 0] *= -1
    elif change == "zero":
        points[:, 2] = 0
    else:
        points[[0, 1]] = points[[1, 0]]
    with pytest.raises(ValueError, match="Jacobian"):
        api().Hex8DisplacementField(points, np.arange(8)[None], np.zeros((8, 3)))


@pytest.mark.parametrize(
    "ids,regions,mapping",
    [
        (("x", "x"), (1, 2), {1: 1.0, 2: 2.0}),
        (("x",), (1, 2), {1: 1.0, 2: 2.0}),
        (("x",), (1,), {2: 2.0}),
        (("x",), (1,), {1: 1.0, 2: 2.0}),
        (("x",), (1,), {1: -1.0}),
        (("x",), (True,), {1: 1.0}),
    ],
)
def test_material_rejects_ambiguous_ids_density(ids, regions, mapping):
    with pytest.raises((ValueError, TypeError)):
        material(ids, regions, mapping)


def test_material_tetra_reuses_p1_and_interface_orientation():
    from gragra.fields.finite_body_p1 import piecewise_tetra_p1_density_sources
    from gragra.fields.unstructured import UnstructuredDisplacementField

    # Three tetrahedra sharing two faces; regions 2 and 3 have equal density.
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1], [0, -1, 0]],
        dtype=float,
    )
    cells = np.array([[0, 1, 2, 3], [0, 2, 1, 4], [0, 1, 4, 5]])
    field = UnstructuredDisplacementField(
        points, cells, np.tile([0.1, 0.2, 0.3], (6, 1))
    )
    regions = material(("a", "b", "c"), (1, 2, 3), {1: 0.0, 2: 2.0, 3: 2.0})
    got = regions.tetra_sources(field, cell_ids=("a", "b", "c"), quadrature_order=3)
    expected = piecewise_tetra_p1_density_sources(
        field, np.repeat([[0.0], [2.0], [2.0]], 4, axis=1), quadrature_order=3
    )
    np.testing.assert_array_equal(
        got.interface_source.weights_kg, expected.interface_source.weights_kg
    )
    assert len(got.interface_triangles) == 10  # 12 faces minus 2 shared copies
    internal = got.positive_side_cell >= 0
    jumps = got.interface_density_jump_kg_m3[internal]
    assert np.any(np.all(jumps == 0.0, axis=1))
    face_points = points[got.interface_triangles[internal]]
    normals = np.cross(
        face_points[:, 1] - face_points[:, 0], face_points[:, 2] - face_points[:, 0]
    )
    centers = points[cells].mean(axis=1)
    directions = (
        centers[got.positive_side_cell[internal]]
        - centers[got.negative_side_cell[internal]]
    )
    assert np.all(np.einsum("fi,fi->f", normals, directions) > 0)
    rho = np.array([0.0, 2.0, 2.0])
    expected_jump = (
        rho[got.negative_side_cell[internal]] - rho[got.positive_side_cell[internal]]
    )
    # P1 traces evaluate a weighted sum even for constant density; allow its
    # three-term floating-point summation bound, not a physical error budget.
    np.testing.assert_allclose(
        jumps,
        np.repeat(expected_jump[:, None], 3, axis=1),
        rtol=0,
        atol=4 * np.finfo(float).eps * np.max(rho),
    )
    # Use the symmetric degree-2 face rule for a point-for-point permutation
    # check. A finite Duffy rule rotates its nonsymmetric nodes when owners
    # change and need not give identical gravity before quadrature convergence.
    from gragra.integrators import direct_point_sum

    original_symmetric = regions.tetra_sources(field, cell_ids=("a", "b", "c"))
    flipped_field = UnstructuredDisplacementField(
        points, cells[::-1], field.displacement_m
    )
    flipped = material(
        ("c", "b", "a"), (3, 2, 1), {1: 0.0, 2: 2.0, 3: 2.0}
    ).tetra_sources(flipped_field, cell_ids=("c", "b", "a"))
    for observable in ("potential", "acceleration", "gradient"):
        before = direct_point_sum(
            original_symmetric.interface_source,
            [[3.0, 2.0, 4.0]],
            observable=observable,
        )
        after = direct_point_sum(
            flipped.interface_source, [[3.0, 2.0, 4.0]], observable=observable
        )
        np.testing.assert_allclose(before, after, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(got.bulk_source.density_kg_m3, 0, atol=1e-15)
    with pytest.raises(ValueError, match="cell_ids"):
        regions.tetra_sources(field, cell_ids=("b", "a", "c"))


def test_material_rejects_implicit_exterior_and_frame_mix():
    with pytest.raises(ValueError, match="exterior"):
        api().MaterialRegions(
            ("a",), np.array([1]), {1: 1.0}, exterior_density_kg_m3=2.0
        )
    field = api().Hex8DisplacementField(cube(), np.arange(8)[None], cube())
    with pytest.raises(ValueError, match="frame"):
        field.sources(material(coordinate_frame="local"))
    with pytest.raises(ValueError, match="SI"):
        material(unit_system="CGS")


def test_region_ids_cannot_overflow_the_stored_int64_contract():
    with pytest.raises(ValueError, match="int64"):
        api().MaterialRegions(
            ("cell",),
            np.array([2**63], dtype=np.uint64),
            {2**63: 1.0},
            exterior_density_kg_m3=0.0,
        )


def test_distorted_hex_volume_centroid_and_rigid_patch():
    # Analytic map from [0,1]^3: (r,s,t*(1+.2*r+.1*s)).
    points = cube()
    points[:, 2] *= 1 + 0.2 * points[:, 0] + 0.1 * points[:, 1]
    disp = np.tile([0.2, -0.1, 0.3], (8, 1))
    field = api().Hex8DisplacementField(points, np.arange(8)[None], disp)
    got = field.sources(material(), quadrature_order=3)
    volume = 1.15
    centroid = (
        np.array(
            [
                0.5 + 0.2 / 3 + 0.1 / 4,
                0.5 + 0.2 / 4 + 0.1 / 3,
                (1 + 0.2 + 0.1 + 0.04 / 3 + 0.01 / 3 + 0.02 / 2) / 2,
            ]
        )
        / volume
    )
    np.testing.assert_allclose(got.volumes_m3.sum(), volume, atol=1e-15)
    np.testing.assert_allclose(
        np.sum(got.positions_m * got.volumes_m3[:, None], axis=0) / volume,
        centroid,
        atol=3e-15,
    )
    np.testing.assert_allclose(
        got.displacement_m, np.broadcast_to(disp[0], (27, 3)), atol=2e-16
    )
    np.testing.assert_allclose(got.bulk_source.density_kg_m3, 0, atol=5e-13)
    a = np.array([[0.1, 0.2, 0.3], [-0.2, 0.3, 0.1], [0.4, -0.5, 0.2]])
    affine = api().Hex8DisplacementField(points, np.arange(8)[None], points @ a.T)
    affine_sources = affine.sources(material())
    np.testing.assert_allclose(
        affine_sources.bulk_source.density_kg_m3, -2300 * np.trace(a), rtol=2e-15
    )


def _independent_box_integral(targets, order, observable):
    # Integrate a physical unit cube directly; no Q1 node generator or kernel.
    z, w = np.polynomial.legendre.leggauss(order)
    z, w = (z + 1) / 2, w / 2
    x = np.array([[a, b, c] for a in z for b in z for c in z])
    mass = 2300 * np.array([a * b * c for a in w for b in w for c in w])
    u = np.zeros_like(x, dtype=complex)
    # Q1 interpolation of prescribed x-directed P-wave samples at x=0,1.
    u[:, 0] = (1 - x[:, 0]) + x[:, 0] * np.exp(0.8j)
    d = targets[:, None, :] - x
    r = np.linalg.norm(d, axis=-1)
    dot = np.einsum("mni,ni->mn", d, u)
    if observable == "potential":
        return np.sum(-mass * dot / r**3, axis=1)
    if observable == "acceleration":
        return np.sum(
            mass[None, :, None]
            * (
                u[None] / r[..., None] ** 3 - 3 * d * dot[..., None] / r[..., None] ** 5
            ),
            axis=1,
        )
    value = (
        -3
        * (
            u[None, :, :, None] * d[:, :, None, :]
            + d[:, :, :, None] * u[None, :, None, :]
            + dot[..., None, None] * np.eye(3)
        )
        / r[..., None, None] ** 5
    )
    value += (
        15
        * dot[..., None, None]
        * d[:, :, :, None]
        * d[:, :, None, :]
        / r[..., None, None] ** 7
    )
    return np.sum(mass[None, :, None, None] * value, axis=1)


@pytest.mark.parametrize("observable", ["potential", "acceleration", "gradient"])
def test_prescribed_pwave_external_observables_independent_integral(observable):
    from gragra.integrators.point_direct import direct_point_displacement_sum

    points = cube()
    u = np.zeros((8, 3), dtype=complex)
    u[:, 0] = np.exp(0.8j * points[:, 0])
    sources = (
        api()
        .Hex8DisplacementField(points, np.arange(8)[None], u)
        .sources(material(), quadrature_order=8)
    )
    targets = np.array([[4.0, -3.0, 5.0], [-3.0, 2.0, 4.0]])
    got = direct_point_displacement_sum(
        sources.mass_kg,
        sources.displacement_m,
        sources.positions_m,
        targets,
        observable=observable,
    )
    ref = _independent_box_integral(targets, 12, observable)
    ref_high = _independent_box_integral(targets, 16, observable)
    np.testing.assert_allclose(ref, ref_high, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(got, ref, rtol=1e-11, atol=1e-12)

    def norms(x):
        return np.linalg.norm(x.reshape(len(targets), -1), axis=1)

    assert np.all(norms(got - ref) <= 1e-12 + 1e-11 * norms(ref))
    assert np.all(norms(ref - ref_high) <= 0.1 * (1e-12 + 1e-11 * norms(ref)))


def test_multicell_separate_materials_and_signed_bulk():
    points = np.concatenate([cube(), cube() + [1.0, 0, 0]])
    field = api().Hex8DisplacementField(
        points, np.arange(16).reshape(2, 8), points * 0.1
    )
    regions = material(("left", "right"), (7, 8), {7: 2.0, 8: 3.0})
    sources = field.sources(regions, quadrature_order=2)
    np.testing.assert_allclose(sources.mass_kg.sum(), 5.0, atol=2e-15)
    np.testing.assert_allclose(
        sources.bulk_source.density_kg_m3, np.repeat([-0.6, -0.9], 8), atol=5e-16
    )
    with pytest.raises(ValueError, match="cell_ids"):
        field.sources(regions, cell_ids=("right", "left"))


@pytest.mark.parametrize("samples", [1, 3])
def test_hex_exchange_series_roundtrip_and_chunk_boundaries(tmp_path, samples):
    pytest.importorskip("h5py")
    exchange = importlib.import_module("gragra.adapters.hex8_series")
    from gragra.adapters.field_series_hdf5 import write_fixed_mesh_series_hdf5

    points = cube()
    field = api().Hex8DisplacementField(points, np.arange(8)[None], points)
    values = np.stack([(j + 1j) * points for j in range(samples)])
    series = exchange.hex8_displacement_series(
        field,
        material(),
        values,
        np.arange(samples, dtype=float) + 1.0,
        "frequency",
        coordinate_origin_m=np.array([1.0, 2.0, 3.0]),
        t_ref_s=2.0,
        phase_origin_m=np.array([0.1, 0.2, 0.3]),
        quadrature_order=2,
    )
    path = write_fixed_mesh_series_hdf5(
        tmp_path / "hex.h5", series, sample_block_size=2, source_block_size=3
    )
    result = exchange.read_hex8_series_hdf5(path)
    np.testing.assert_array_equal(result.geometry.vertices_m, points)
    np.testing.assert_array_equal(result.geometry.hex8, np.arange(8)[None])
    assert result.materials.cell_ids == ("cell",)
    np.testing.assert_array_equal(result.materials.region_ids, [10])
    assert result.provider.metadata.t_ref_s == 2.0
    np.testing.assert_array_equal(
        result.provider.metadata.phase_origin_m, [0.1, 0.2, 0.3]
    )
    blocks = list(result.provider.blocks(slice(0, samples), 3))
    assert [len(b.positions_m) for b in blocks] == [3, 3, 2]
    np.testing.assert_array_equal(
        np.concatenate([b.displacement_m for b in blocks], axis=1),
        series.displacement_m,
    )


def test_hex_exchange_rejects_metadata_corruption(tmp_path):
    h5py = pytest.importorskip("h5py")
    exchange = importlib.import_module("gragra.adapters.hex8_series")
    from gragra.adapters.field_series_hdf5 import write_fixed_mesh_series_hdf5

    field = api().Hex8DisplacementField(cube(), np.arange(8)[None], cube())
    series = exchange.hex8_displacement_series(
        field, material(), cube()[None], np.array([0.0]), "time"
    )
    path = write_fixed_mesh_series_hdf5(tmp_path / "bad.h5", series)
    with h5py.File(path, "r+") as h5:
        h5["metadata"].attrs["source_metadata_json"] = "{}"
    with pytest.raises(ValueError, match="hex8"):
        exchange.read_hex8_series_hdf5(path)


def test_hex_hdf5_bounded_values_and_legacy_v1_reader(tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    exchange = importlib.import_module("gragra.adapters.hex8_series")
    from gragra.adapters.field_series_hdf5 import (
        read_fixed_mesh_series_hdf5,
        write_fixed_mesh_series_hdf5,
    )

    field = api().Hex8DisplacementField(cube(), np.arange(8)[None], cube())
    series = exchange.hex8_displacement_series(
        field,
        material(),
        np.repeat(cube()[None], 5, axis=0),
        np.arange(5, dtype=float),
        "time",
    )
    original = h5py.Dataset.__getitem__
    requests = []

    def bounded(dataset, key):
        if dataset.name == "/values":
            assert isinstance(key, tuple) and len(key) == 3
            assert all(
                isinstance(k, slice) and k.start is not None and k.stop is not None
                for k in key
            )
            assert key[0].stop - key[0].start <= 2
            assert key[1].stop - key[1].start <= 5
            requests.append(key)
        return original(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", bounded)
    path = write_fixed_mesh_series_hdf5(
        tmp_path / "stream.h5", series, sample_block_size=2, source_block_size=5
    )
    # Existing v1 reader accepts new optional metadata, without schema changes.
    provider = read_fixed_mesh_series_hdf5(path)
    assert provider.metadata.source_ids == series.source_ids
    for start in range(0, 5, 2):
        list(provider.blocks(slice(start, min(start + 2, 5)), 5))
    assert requests
    with pytest.raises(ValueError):
        list(provider.blocks(slice(0, 1), 0))


@pytest.mark.parametrize(
    "key,value",
    [
        ("version", 2),
        ("version", True),
        ("node_order", "unknown"),
        ("unit_system", "CGS"),
        ("cell_ids", ["wrong"]),
        ("region_density", [[10, 2300.0], [10, 2300.0]]),
        ("region_density", [[11, 2300.0]]),
        ("exterior_density_kg_m3", 1.0),
        ("density_convention", "delta_rho"),
        ("quadrature_order", 1),
    ],
)
def test_hex_exchange_rejects_inconsistent_metadata(tmp_path, key, value):
    import json

    h5py = pytest.importorskip("h5py")
    exchange = importlib.import_module("gragra.adapters.hex8_series")
    from gragra.adapters.field_series_hdf5 import write_fixed_mesh_series_hdf5

    field = api().Hex8DisplacementField(cube(), np.arange(8)[None], cube())
    series = exchange.hex8_displacement_series(
        field, material(), cube()[None], np.array([0.0]), "time"
    )
    path = write_fixed_mesh_series_hdf5(tmp_path / "bad.h5", series)
    with h5py.File(path, "r+") as h5:
        data = json.loads(h5["metadata"].attrs["source_metadata_json"])
        data[key] = value
        h5["metadata"].attrs["source_metadata_json"] = json.dumps(data)
    with pytest.raises((ValueError, TypeError)):
        exchange.read_hex8_series_hdf5(path)


@pytest.mark.parametrize("observable", ["potential", "acceleration", "gradient"])
def test_hex_exchange_observable_e2e(tmp_path, observable):
    pytest.importorskip("h5py")
    exchange = importlib.import_module("gragra.adapters.hex8_series")
    from gragra.adapters.field_series_hdf5 import write_fixed_mesh_series_hdf5
    from gragra.constants import G_SI
    from gragra.observables.streaming import (
        collect_observable_blocks,
        stream_observable,
    )

    points = cube()
    u = np.zeros((8, 3), dtype=complex)
    u[:, 0] = np.exp(0.8j * points[:, 0])
    field = api().Hex8DisplacementField(points, np.arange(8)[None], u)
    series = exchange.hex8_displacement_series(
        field,
        material(),
        u[None],
        np.array([2.0]),
        "frequency",
        t_ref_s=0.0,
        phase_origin_m=np.zeros(3),
        quadrature_order=8,
    )
    path = write_fixed_mesh_series_hdf5(
        tmp_path / "e2e.h5", series, source_block_size=73
    )
    provider = exchange.read_hex8_series_hdf5(path).provider
    targets = np.array([[4.0, -3.0, 5.0], [-3.0, 2.0, 4.0]])
    values = collect_observable_blocks(
        stream_observable(
            provider,
            targets,
            observable=observable,
            source_block_size=73,
            target_block_size=1,
        ),
        sample_count=1,
        target_count=2,
        axis_kind="frequency",
    )
    reference = G_SI * _independent_box_integral(targets, 16, observable)
    np.testing.assert_allclose(values[:, 0], reference, rtol=1e-11, atol=G_SI * 1e-12)
    assert not values.flags.writeable
