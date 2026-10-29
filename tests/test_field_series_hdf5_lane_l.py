"""Fixed-mesh series HDF5 profile and bounded-provider tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py")

import gragra
from gragra.adapters.field_series_hdf5 import (
    FOURIER_CONVENTION,
    PROFILE,
    VERSION,
    read_fixed_mesh_series_hdf5,
    write_fixed_mesh_series_hdf5,
    write_observable_series_hdf5,
)
from gragra.fields.density_series import DensityPhasors, DensityTimeSeries
from gragra.fields.displacement_series import FixedMeshDisplacementSeries
from gragra.fields.surface_density_series import FixedMeshSurfaceDensitySeries
from gragra.integrators.streaming import (
    DisplacementSourceSeriesBlock,
    ScalarSourceSeriesBlock,
    as_series_provider,
)
from gragra.observables.streaming import (
    ObservableSeriesBlock,
    collect_observable_blocks,
    stream_observable,
)

WORKTREE = Path(__file__).resolve().parents[1]


def _common(axis_kind="time"):
    return {
        "positions_m": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "coordinate_frame": "ENU",
        "coordinate_origin_m": np.array([10.0, 20.0, 30.0]),
        "source_ids": ("left", "right"),
        "axis_values": np.array([2.0, 4.0])
        if axis_kind == "frequency"
        else np.array([0.0, 0.3]),
        "axis_kind": axis_kind,
        "t_ref_s": 0.125 if axis_kind == "frequency" else None,
        "phase_origin_m": np.array([7.0, 8.0, 9.0])
        if axis_kind == "frequency"
        else None,
    }


def _series_cases():
    time = _common()
    frequency = _common("frequency")
    return [
        DensityTimeSeries(
            time["positions_m"],
            np.array([2.0, 3.0]),
            np.array([[1.0, -2.0], [3.0, 4.0]]),
            time["axis_values"],
            "delta_rho",
            time["coordinate_frame"],
            time["coordinate_origin_m"],
            time["source_ids"],
        ),
        DensityPhasors(
            frequency["positions_m"],
            np.array([2.0, 3.0]),
            np.array([[1.0 + 2j, -2.0j], [3.0 - 1j, 4.0 + 5j]]),
            frequency["axis_values"],
            frequency["t_ref_s"],
            frequency["phase_origin_m"],
            frequency["coordinate_frame"],
            frequency["coordinate_origin_m"],
            frequency["source_ids"],
        ),
        FixedMeshDisplacementSeries(
            time["positions_m"],
            np.array([4.0, 5.0]),
            np.arange(12, dtype=np.float64).reshape(2, 2, 3) / 10.0,
            time["axis_values"],
            "time",
            time["coordinate_frame"],
            time["coordinate_origin_m"],
            time["source_ids"],
        ),
        FixedMeshSurfaceDensitySeries(
            frequency["positions_m"],
            np.array([7.0, 11.0]),
            np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
            np.array([[2.0 + 1j, 3.0 - 2j], [4.0 + 3j, 5.0 + 4j]]),
            frequency["axis_values"],
            "frequency",
            {"quadrature": "faces"},
            frequency["coordinate_frame"],
            frequency["coordinate_origin_m"],
            frequency["source_ids"],
            t_ref_s=frequency["t_ref_s"],
            phase_origin_m=frequency["phase_origin_m"],
        ),
    ]


def _mutate_on_first_reopen(monkeypatch, mutation):
    h5py = pytest.importorskip("h5py")
    original_file = h5py.File
    mutated = False

    def intercepted_file(name, mode="r", *args, **kwargs):
        nonlocal mutated
        if mode == "r" and str(name).endswith(".partial") and not mutated:
            mutated = True
            with original_file(name, "r+") as h5:
                mutation(h5)
        return original_file(name, mode, *args, **kwargs)

    monkeypatch.setattr(h5py, "File", intercepted_file)


@pytest.mark.parametrize("series", _series_cases())
def test_roundtrip_all_quantities_axes_geometry_ids_units_and_readonly(
    tmp_path, series
):
    assert Path(gragra.__file__).resolve().is_relative_to(WORKTREE / "src")
    direct = as_series_provider(series)
    assert direct.sample_count == len(series.axis_values)
    assert direct.source_count == len(series.source_ids)
    assert sum(len(block.source_ids) for block in direct.blocks(slice(0, 2), 1)) == len(
        series.source_ids
    )
    integrated = collect_observable_blocks(
        stream_observable(
            direct,
            [[20.0, 0.0, 0.0]],
            observable="acceleration",
            sample_block_size=1,
            source_block_size=1,
            target_block_size=1,
            kernel_chunk_size=1,
        ),
        sample_count=series.sample_count,
        target_count=1,
        axis_kind=series.axis_kind,
    )
    expected_integrated_shape = (
        (1, series.sample_count, 3)
        if series.axis_kind == "frequency"
        else (series.sample_count, 1, 3)
    )
    assert integrated.shape == expected_integrated_shape
    assert integrated.dtype == (
        np.complex128 if series.axis_kind == "frequency" else np.float64
    )
    assert np.isfinite(integrated).all()
    assert not integrated.flags.writeable
    path = tmp_path / f"{type(series).__name__}.h5"
    written = write_fixed_mesh_series_hdf5(
        path, series, sample_block_size=1, source_block_size=1
    )
    assert written == path
    assert path.exists()
    assert not Path(f"{path}.partial").exists()

    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "r") as h5:
        assert h5.attrs["profile"] == PROFILE
        assert h5.attrs["version"] == VERSION
        assert bool(h5.attrs["complete"])
        assert h5["values"].attrs["unit"] in {"kg/m3", "m", "kg/m2"}
        assert h5["metadata/positions_m"].attrs["unit"] == "m"
        assert h5["metadata/axis_values"].attrs["unit"] in {"s", "Hz"}

    provider = read_fixed_mesh_series_hdf5(path)
    assert provider.source_ids == series.source_ids
    assert np.array_equal(provider.positions_m, series.positions_m)
    assert np.array_equal(provider.axis_values, series.axis_values)
    assert provider.coordinate_frame == series.coordinate_frame
    assert np.array_equal(provider.coordinate_origin_m, series.coordinate_origin_m)
    assert provider.axis_kind == series.axis_kind
    assert provider.t_ref_s == getattr(series, "t_ref_s", None)
    if provider.phase_origin_m is not None:
        assert np.array_equal(provider.phase_origin_m, series.phase_origin_m)
    blocks = list(provider.blocks(slice(0, 2), 1))
    assert len(blocks) == 2
    assert all(not block.positions_m.flags.writeable for block in blocks)
    if hasattr(series, "displacement_m"):
        assert all(isinstance(block, DisplacementSourceSeriesBlock) for block in blocks)
        actual = np.concatenate([block.displacement_m for block in blocks], axis=1)
        assert np.array_equal(actual, series.displacement_m)
        assert np.array_equal(
            np.concatenate([b.mass_kg for b in blocks]), series.mass_kg
        )
    else:
        assert all(isinstance(block, ScalarSourceSeriesBlock) for block in blocks)
        actual = np.concatenate([block.weights_kg for block in blocks], axis=1)
        scale = series.volumes_m3 if hasattr(series, "volumes_m3") else series.areas_m2
        values = (
            series.density_kg_m3
            if hasattr(series, "density_kg_m3")
            else series.surface_density_kg_m2
        )
        assert np.array_equal(actual, values * scale)


def test_provider_reads_values_only_with_bounded_sample_source_slices(
    tmp_path, monkeypatch
):
    series = _series_cases()[2]
    path = tmp_path / "bounded.h5"
    write_fixed_mesh_series_hdf5(path, series)
    provider = read_fixed_mesh_series_hdf5(path)
    h5py = pytest.importorskip("h5py")
    original = h5py.Dataset.__getitem__
    keys = []

    def spy(dataset, key):
        if dataset.name == "/values":
            keys.append(key)
            assert key != ()
            assert key is not Ellipsis
            assert isinstance(key, tuple)
            assert isinstance(key[0], slice) and key[0].start is not None
            assert isinstance(key[1], slice) and key[1].start is not None
        return original(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", spy)
    blocks = list(provider.blocks(slice(1, 2), 1))
    assert len(blocks) == 2
    assert len(keys) == 2


def test_profile_validation_reads_geometry_and_ids_in_explicit_slabs(
    tmp_path, monkeypatch
):
    path = tmp_path / "metadata-slabs.h5"
    write_fixed_mesh_series_hdf5(path, _series_cases()[3])
    h5py = pytest.importorskip("h5py")
    original = h5py.Dataset.__getitem__
    keys = []

    def spy(dataset, key):
        keys.append((dataset.name, key))
        assert key != ()
        assert key is not Ellipsis
        selectors = key if isinstance(key, tuple) else (key,)
        assert all(
            not (isinstance(item, slice) and item == slice(None)) for item in selectors
        )
        return original(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", spy)
    provider = read_fixed_mesh_series_hdf5(
        path, metadata_sample_block_size=1, metadata_source_block_size=1
    )
    list(provider.blocks(slice(0, 1), 1))
    touched = {name for name, _ in keys}
    assert {
        "/metadata/positions_m",
        "/metadata/source_ids",
        "/metadata/areas_m2",
        "/metadata/normals",
    } <= touched


def test_fixed_writer_reopened_readback_uses_only_bounded_slabs(tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    original = h5py.Dataset.__getitem__
    seen = []

    def spy(dataset, key):
        if dataset.name == "/values":
            seen.append(dataset.name)
            assert isinstance(key, tuple)
            assert key[0].stop - key[0].start <= 1
            assert key[1].stop - key[1].start <= 1
        elif dataset.name.startswith("/metadata/"):
            seen.append(dataset.name)
            assert key != () and key is not Ellipsis
            selectors = key if isinstance(key, tuple) else (key,)
            assert all(item != slice(None) for item in selectors)
        return original(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", spy)
    write_fixed_mesh_series_hdf5(
        tmp_path / "bounded-fixed-writer.h5",
        _series_cases()[3],
        sample_block_size=1,
        source_block_size=1,
    )
    assert "/values" in seen
    assert {
        "/metadata/positions_m",
        "/metadata/source_ids",
        "/metadata/axis_values",
        "/metadata/areas_m2",
        "/metadata/normals",
    } <= set(seen)


def test_writer_never_overwrites_final_or_existing_partial(tmp_path):
    series = _series_cases()[0]
    final = tmp_path / "series.h5"
    final.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        write_fixed_mesh_series_hdf5(final, series)
    final.unlink()
    partial = Path(f"{final}.partial")
    partial.write_bytes(b"incomplete")
    with pytest.raises(FileExistsError):
        write_fixed_mesh_series_hdf5(final, series)


def test_late_write_failure_retains_incomplete_partial(tmp_path):
    good = _series_cases()[0]
    bad = type("Bad", (), {})()
    for name in (
        "positions_m",
        "volumes_m3",
        "density_kg_m3",
        "density_kind",
        "axis_kind",
        "axis_values",
        "sample_count",
        "source_count",
        "coordinate_frame",
        "coordinate_origin_m",
        "source_ids",
    ):
        setattr(bad, name, getattr(good, name))
    bad.density_kg_m3 = np.array(good.density_kg_m3, copy=True)
    bad.density_kg_m3[-1, -1] = np.nan
    bad.t_ref_s = None
    bad.phase_origin_m = None
    final = tmp_path / "bad.h5"
    with pytest.raises(ValueError, match="finite"):
        write_fixed_mesh_series_hdf5(
            final, bad, sample_block_size=1, source_block_size=1
        )
    partial = Path(f"{final}.partial")
    assert partial.exists()
    assert not final.exists()
    h5py = pytest.importorskip("h5py")
    with h5py.File(partial, "r") as h5:
        assert not bool(h5.attrs["complete"])


@pytest.mark.parametrize(
    ("case_index", "dataset", "replacement"),
    [
        (0, "volumes_m3", np.array([3.0, 3.0])),
        (2, "mass_kg", np.array([6.0, 5.0])),
        (3, "areas_m2", np.array([7.0, 13.0])),
        (3, "normals", np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])),
    ],
)
def test_fixed_writer_rejects_valid_but_changed_geometry_before_publish(
    tmp_path, monkeypatch, case_index, dataset, replacement
):
    _mutate_on_first_reopen(
        monkeypatch,
        lambda h5: h5[f"metadata/{dataset}"].__setitem__(
            tuple(slice(0, width) for width in replacement.shape), replacement
        ),
    )
    path = tmp_path / f"changed-{dataset}.h5"
    with pytest.raises(OSError, match="metadata differs"):
        write_fixed_mesh_series_hdf5(
            path, _series_cases()[case_index], sample_block_size=1, source_block_size=1
        )
    assert Path(f"{path}.partial").exists()
    assert not path.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda h5: h5["metadata/axis_values"].__setitem__(0, 1.0),
        lambda h5: h5["metadata/positions_m"].__setitem__((0, 0), 9.0),
        lambda h5: h5["metadata/source_ids"].__setitem__(0, "changed"),
        lambda h5: h5["metadata"].attrs.__setitem__("coordinate_frame", "NED"),
        lambda h5: h5["metadata/coordinate_origin_m"].__setitem__(0, 11.0),
        lambda h5: h5["metadata/phase_origin_m"].__setitem__(0, 6.0),
        lambda h5: h5["metadata"].attrs.__setitem__(
            "source_metadata_json", '{"changed":true}'
        ),
    ],
    ids=[
        "axis",
        "position",
        "source-id",
        "frame",
        "origin",
        "phase",
        "source-metadata",
    ],
)
def test_fixed_writer_rejects_valid_but_changed_common_metadata_before_publish(
    tmp_path, monkeypatch, mutation
):
    _mutate_on_first_reopen(monkeypatch, mutation)
    path = tmp_path / "changed-common.h5"
    with pytest.raises(OSError, match="metadata differs"):
        write_fixed_mesh_series_hdf5(
            path, _series_cases()[3], sample_block_size=1, source_block_size=1
        )
    assert Path(f"{path}.partial").exists()
    assert not path.exists()


def test_reader_rejects_incomplete_wrong_units_and_shape(tmp_path):
    h5py = pytest.importorskip("h5py")
    for mutation, match in [
        (lambda h5: h5.attrs.__setitem__("complete", False), "incomplete"),
        (lambda h5: h5["values"].attrs.__setitem__("unit", "cm"), "unit"),
    ]:
        path = tmp_path / f"bad-{match}.h5"
        write_fixed_mesh_series_hdf5(path, _series_cases()[0])
        with h5py.File(path, "r+") as h5:
            mutation(h5)
        with pytest.raises(ValueError, match=match):
            read_fixed_mesh_series_hdf5(path)


@pytest.mark.parametrize(
    ("attribute", "value", "match"),
    [
        ("version", np.float64(1.9), "version"),
        ("version", np.float64(1.0), "version"),
        ("version", True, "version"),
        ("version", "1", "version"),
        ("complete", "false", "Boolean"),
        ("complete", 1, "Boolean"),
        ("complete", np.float64(1.0), "Boolean"),
    ],
)
def test_reader_rejects_coercible_but_nonliteral_profile_attributes(
    tmp_path, attribute, value, match
):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / f"bad-{attribute}-{type(value).__name__}.h5"
    write_fixed_mesh_series_hdf5(path, _series_cases()[0])
    with h5py.File(path, "r+") as h5:
        del h5.attrs[attribute]
        h5.attrs[attribute] = value
    with pytest.raises(ValueError, match=match):
        read_fixed_mesh_series_hdf5(path)


def test_reader_rejects_missing_or_contradictory_fourier_metadata(tmp_path):
    h5py = pytest.importorskip("h5py")
    frequency_path = tmp_path / "missing-convention.h5"
    write_fixed_mesh_series_hdf5(frequency_path, _series_cases()[1])
    with h5py.File(frequency_path, "r+") as h5:
        del h5["metadata"].attrs["fourier_convention"]
    with pytest.raises(ValueError, match="Fourier convention"):
        read_fixed_mesh_series_hdf5(frequency_path)

    time_path = tmp_path / "contradictory-convention.h5"
    write_fixed_mesh_series_hdf5(time_path, _series_cases()[0])
    with h5py.File(time_path, "r+") as h5:
        h5["metadata"].attrs["fourier_convention"] = FOURIER_CONVENTION
    with pytest.raises(ValueError, match="must not"):
        read_fixed_mesh_series_hdf5(time_path)


@pytest.mark.parametrize(
    ("case_index", "mutation", "match"),
    [
        (1, "rho", "delta_rho"),
        (1, "missing", "requires density_kind"),
        (2, "delta_rho", "only for density"),
    ],
)
def test_reader_enforces_density_kind_by_quantity_and_axis(
    tmp_path, case_index, mutation, match
):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / f"bad-density-kind-{case_index}-{mutation}.h5"
    write_fixed_mesh_series_hdf5(path, _series_cases()[case_index])
    with h5py.File(path, "r+") as h5:
        metadata = h5["metadata"]
        if mutation == "missing":
            del metadata.attrs["density_kind"]
        else:
            metadata.attrs["density_kind"] = mutation
    with pytest.raises(ValueError, match=match):
        read_fixed_mesh_series_hdf5(path)


def test_corrupted_late_hdf5_value_is_rejected_before_first_output(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "late-nan.h5"
    write_fixed_mesh_series_hdf5(path, _series_cases()[0])
    with h5py.File(path, "r+") as h5:
        h5["values"][-1, -1] = np.nan
    provider = read_fixed_mesh_series_hdf5(path)
    from gragra.observables.streaming import stream_observable

    with pytest.raises(ValueError, match="finite"):
        next(
            stream_observable(
                provider,
                [[100.0, 0.0, 0.0]],
                observable="potential",
                sample_block_size=1,
            )
        )


@pytest.mark.parametrize("axis_kind", ["time", "frequency"])
def test_observable_writer_uses_public_axis_order_and_complete_readback(
    tmp_path, axis_kind
):
    path = tmp_path / f"observable-{axis_kind}.h5"
    dtype = np.complex128 if axis_kind == "frequency" else np.float64
    blocks = [
        ObservableSeriesBlock(
            slice(0, 2), slice(1, 2), np.full((2, 1, 3), 2.0, dtype=dtype)
        ),
        ObservableSeriesBlock(
            slice(0, 2), slice(0, 1), np.full((2, 1, 3), 1.0, dtype=dtype)
        ),
    ]
    write_observable_series_hdf5(
        path,
        blocks,
        sample_count=2,
        target_count=2,
        axis_kind=axis_kind,
        axis_values=np.array([1.0, 2.0])
        if axis_kind == "frequency"
        else np.array([0.0, 1.0]),
        targets_m=np.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
        observable="acceleration",
        coordinate_frame="ENU",
        coordinate_origin_m=np.zeros(3),
        **(
            {
                "t_ref_s": 0.25,
                "phase_origin_m": np.array([1.0, 2.0, 3.0]),
                "fourier_convention": FOURIER_CONVENTION,
            }
            if axis_kind == "frequency"
            else {}
        ),
    )
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "r") as h5:
        assert bool(h5.attrs["complete"])
        assert len(h5.attrs["readback_sha256"]) == 64
        expected_shape = (2, 2, 3)
        assert h5["values"].shape == expected_shape
        assert h5["values"].attrs["unit"] == "m/s2"
        if axis_kind == "frequency":
            assert h5.attrs["t_ref_s"] == pytest.approx(0.25)
            assert h5.attrs["fourier_convention"] == FOURIER_CONVENTION
            assert np.array_equal(h5["phase_origin_m"][:], [1.0, 2.0, 3.0])
        values = h5["values"][:]
        if axis_kind == "frequency":
            assert np.array_equal(values[0], np.ones((2, 3)))
            assert np.array_equal(values[1], np.full((2, 3), 2.0))
        else:
            assert np.array_equal(values[:, 0], np.ones((2, 3)))
            assert np.array_equal(values[:, 1], np.full((2, 3), 2.0))


def test_observable_writer_requires_frequency_phase_and_rejects_time_phase(tmp_path):
    complex_block = [
        ObservableSeriesBlock(
            slice(0, 1), slice(0, 1), np.ones((1, 1), dtype=np.complex128)
        )
    ]
    common = {
        "sample_count": 1,
        "target_count": 1,
        "axis_values": np.array([1.0]),
        "targets_m": np.array([[2.0, 0.0, 0.0]]),
        "observable": "potential",
        "coordinate_frame": "ENU",
        "coordinate_origin_m": np.zeros(3),
    }
    with pytest.raises(ValueError, match="t_ref_s"):
        write_observable_series_hdf5(
            tmp_path / "missing-tref.h5",
            complex_block,
            axis_kind="frequency",
            **common,
        )
    with pytest.raises(ValueError, match="phase_origin_m"):
        write_observable_series_hdf5(
            tmp_path / "missing-phase.h5",
            complex_block,
            axis_kind="frequency",
            t_ref_s=0.0,
            **common,
        )
    with pytest.raises(ValueError, match="Fourier convention"):
        write_observable_series_hdf5(
            tmp_path / "wrong-convention.h5",
            complex_block,
            axis_kind="frequency",
            t_ref_s=0.0,
            phase_origin_m=np.zeros(3),
            fourier_convention="exp(+i omega t)",
            **common,
        )
    with pytest.raises(ValueError, match="must not"):
        write_observable_series_hdf5(
            tmp_path / "time-phase.h5",
            [ObservableSeriesBlock(slice(0, 1), slice(0, 1), np.ones((1, 1)))],
            axis_kind="time",
            t_ref_s=0.0,
            phase_origin_m=np.zeros(3),
            fourier_convention=FOURIER_CONVENTION,
            **{**common, "axis_values": np.array([0.0])},
        )


@pytest.mark.parametrize("axis_kind", ["time", "frequency"])
def test_observable_reopened_readback_is_bounded_on_sample_and_target_axes(
    tmp_path, monkeypatch, axis_kind
):
    blocks = []
    dtype = np.complex128 if axis_kind == "frequency" else np.float64
    for sample_start in range(0, 4, 2):
        for target_start in range(0, 5, 2):
            sample_stop = min(sample_start + 2, 4)
            target_stop = min(target_start + 2, 5)
            blocks.append(
                ObservableSeriesBlock(
                    slice(sample_start, sample_stop),
                    slice(target_start, target_stop),
                    np.ones(
                        (sample_stop - sample_start, target_stop - target_start),
                        dtype=dtype,
                    ),
                )
            )
    h5py = pytest.importorskip("h5py")
    original = h5py.Dataset.__getitem__
    seen = []

    def spy(dataset, key):
        if dataset.name in {"/values", "/coverage"}:
            seen.append((dataset.name, key))
            assert isinstance(key, tuple) and len(key) >= 2
            assert key[0].stop - key[0].start <= 2
            assert key[1].stop - key[1].start <= 2
        elif dataset.name == "/axis_values":
            seen.append((dataset.name, key))
            assert isinstance(key, tuple) and len(key) == 1
            assert key[0].stop - key[0].start <= 2
        elif dataset.name == "/targets_m":
            seen.append((dataset.name, key))
            assert isinstance(key, tuple) and key == (key[0], slice(0, 3))
            assert key[0].stop - key[0].start <= 2
        elif dataset.name in {"/coordinate_origin_m", "/phase_origin_m"}:
            seen.append((dataset.name, key))
            assert key == (slice(0, 3),)
        return original(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", spy)
    write_observable_series_hdf5(
        tmp_path / f"bounded-output-{axis_kind}.h5",
        blocks,
        sample_count=4,
        target_count=5,
        axis_kind=axis_kind,
        axis_values=(np.arange(4.0) + 1.0)
        if axis_kind == "frequency"
        else np.arange(4.0),
        targets_m=np.column_stack((np.arange(5.0) + 10.0, np.zeros((5, 2)))),
        observable="potential",
        coordinate_frame="ENU",
        coordinate_origin_m=np.zeros(3),
        sample_block_size=2,
        target_block_size=2,
        **(
            {
                "t_ref_s": 0.0,
                "phase_origin_m": np.zeros(3),
                "fourier_convention": FOURIER_CONVENTION,
            }
            if axis_kind == "frequency"
            else {}
        ),
    )
    assert {name for name, _ in seen} == {
        "/values",
        "/coverage",
        "/axis_values",
        "/targets_m",
        "/coordinate_origin_m",
        *(("/phase_origin_m",) if axis_kind == "frequency" else ()),
    }


@pytest.mark.parametrize(
    ("attribute", "value"),
    [("version", np.float64(1.9)), ("complete", "false")],
)
def test_observable_reopened_readback_rejects_nonliteral_header_attributes(
    tmp_path, monkeypatch, attribute, value
):
    h5py = pytest.importorskip("h5py")
    original_file = h5py.File
    corrupted = False

    def intercepted_file(name, mode="r", *args, **kwargs):
        nonlocal corrupted
        if mode == "r" and str(name).endswith(".partial") and not corrupted:
            corrupted = True
            with original_file(name, "r+") as h5:
                del h5.attrs[attribute]
                h5.attrs[attribute] = value
        return original_file(name, mode, *args, **kwargs)

    monkeypatch.setattr(h5py, "File", intercepted_file)
    path = tmp_path / f"bad-output-{attribute}.h5"
    with pytest.raises(OSError, match="metadata failed readback"):
        write_observable_series_hdf5(
            path,
            [ObservableSeriesBlock(slice(0, 1), slice(0, 1), np.ones((1, 1)))],
            sample_count=1,
            target_count=1,
            axis_kind="time",
            axis_values=np.array([0.0]),
            targets_m=np.array([[2.0, 0.0, 0.0]]),
            observable="potential",
            coordinate_frame="ENU",
            coordinate_origin_m=np.zeros(3),
        )
    assert Path(f"{path}.partial").exists()
    assert not path.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda h5: h5["axis_values"].__setitem__(0, 0.5),
        lambda h5: h5["targets_m"].__setitem__((0, 0), 3.0),
        lambda h5: h5["coordinate_origin_m"].__setitem__(0, 1.0),
        lambda h5: h5.attrs.__setitem__("coordinate_frame", "NED"),
        lambda h5: h5["axis_values"].attrs.__setitem__("unit", "s"),
        lambda h5: h5["targets_m"].attrs.__setitem__("unit", "km"),
        lambda h5: h5["coordinate_origin_m"].attrs.__setitem__("unit", "km"),
        lambda h5: h5["phase_origin_m"].__setitem__(0, 1.0),
        lambda h5: h5.attrs.__setitem__("t_ref_s", 0.5),
    ],
    ids=[
        "axis",
        "target",
        "origin",
        "frame",
        "axis-unit",
        "target-unit",
        "origin-unit",
        "phase",
        "t-ref",
    ],
)
def test_observable_writer_rejects_changed_metadata_before_publish(
    tmp_path, monkeypatch, mutation
):
    _mutate_on_first_reopen(monkeypatch, mutation)
    path = tmp_path / "changed-observable.h5"
    with pytest.raises(OSError, match="metadata differs"):
        write_observable_series_hdf5(
            path,
            [
                ObservableSeriesBlock(
                    slice(0, 2),
                    slice(0, 2),
                    np.ones((2, 2), dtype=np.complex128),
                )
            ],
            sample_count=2,
            target_count=2,
            axis_kind="frequency",
            axis_values=np.array([1.0, 2.0]),
            targets_m=np.array([[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
            observable="potential",
            coordinate_frame="ENU",
            coordinate_origin_m=np.zeros(3),
            t_ref_s=0.0,
            phase_origin_m=np.zeros(3),
            fourier_convention=FOURIER_CONVENTION,
            sample_block_size=1,
            target_block_size=1,
        )
    assert Path(f"{path}.partial").exists()
    assert not path.exists()


def test_observable_writer_retains_incomplete_partial_on_gap_overlap_or_exception(
    tmp_path,
):
    cases = {
        "gap": [ObservableSeriesBlock(slice(0, 1), slice(0, 1), np.ones((1, 1)))],
        "overlap": [
            ObservableSeriesBlock(slice(0, 2), slice(0, 1), np.ones((2, 1))),
            ObservableSeriesBlock(slice(0, 1), slice(0, 1), np.ones((1, 1))),
        ],
    }
    for name, blocks in cases.items():
        path = tmp_path / f"{name}.h5"
        with pytest.raises(ValueError, match="coverage"):
            write_observable_series_hdf5(
                path,
                blocks,
                sample_count=2,
                target_count=1,
                axis_kind="time",
                axis_values=np.array([0.0, 1.0]),
                targets_m=np.array([[2.0, 0.0, 0.0]]),
                observable="potential",
                coordinate_frame="ENU",
                coordinate_origin_m=np.zeros(3),
            )
        assert Path(f"{path}.partial").exists()
        assert not path.exists()

    def exploding():
        yield ObservableSeriesBlock(slice(0, 1), slice(0, 1), np.ones((1, 1)))
        raise RuntimeError("late output failure")

    path = tmp_path / "exception.h5"
    with pytest.raises(RuntimeError, match="late output"):
        write_observable_series_hdf5(
            path,
            exploding(),
            sample_count=2,
            target_count=1,
            axis_kind="time",
            axis_values=np.array([0.0, 1.0]),
            targets_m=np.array([[2.0, 0.0, 0.0]]),
            observable="potential",
            coordinate_frame="ENU",
            coordinate_origin_m=np.zeros(3),
        )
    assert Path(f"{path}.partial").exists()
