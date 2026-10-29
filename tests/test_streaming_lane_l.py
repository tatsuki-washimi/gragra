"""Contract and numerical tests for lane-L core streaming."""

from __future__ import annotations

import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import gragra
from gragra.constants import G_SI
from gragra.gravity_sources import WeightedPointSource
from gragra.integrators.point_direct import (
    direct_point_displacement_sum,
    direct_point_sum,
)
from gragra.integrators.streaming import (
    DisplacementSourceSeriesBlock,
    ScalarSourceSeriesBlock,
    SeriesMetadata,
    as_series_provider,
    estimate_streaming_working_bytes,
)
from gragra.observables.streaming import (
    ObservableSeriesBlock,
    collect_observable_blocks,
    stream_observable,
)

WORKTREE = Path(__file__).resolve().parents[1]


def _metadata(*, axis_kind="time", positions=None, ids=None):
    positions = np.asarray(
        positions
        if positions is not None
        else [[1.0, 0.2, -0.3], [2.0, -0.1, 0.4], [3.0, 0.5, 0.2]],
        dtype=np.float64,
    )
    ids = tuple(ids if ids is not None else ("a", "b", "c"))
    return SeriesMetadata(
        positions_m=positions,
        source_ids=ids,
        coordinate_frame="ENU",
        coordinate_origin_m=np.zeros(3),
        axis_kind=axis_kind,
        axis_values=np.array([2.0, 5.0] if axis_kind == "frequency" else [0.0, 0.7]),
        sample_count=2,
        source_count=len(positions),
        t_ref_s=0.25 if axis_kind == "frequency" else None,
        phase_origin_m=np.array([1.0, 2.0, 3.0]) if axis_kind == "frequency" else None,
    )


def _scalar_factory(meta, values, *, source_partition=(1, 2)):
    def factory(samples, requested_size):
        del requested_size
        start = 0
        for width in source_partition:
            stop = start + width
            yield ScalarSourceSeriesBlock(
                meta.positions_m[start:stop],
                samples,
                start,
                values[samples, start:stop],
                meta.source_ids[start:stop],
            )
            start = stop

    return factory


def _displacement_factory(meta, mass, displacement, *, source_partition=(2, 1)):
    def factory(samples, requested_size):
        del requested_size
        start = 0
        for width in source_partition:
            stop = start + width
            yield DisplacementSourceSeriesBlock(
                meta.positions_m[start:stop],
                samples,
                start,
                mass[start:stop],
                displacement[samples, start:stop],
                meta.source_ids[start:stop],
            )
            start = stop

    return factory


def _collect(provider, targets, observable, **kwargs):
    return collect_observable_blocks(
        stream_observable(provider, targets, observable=observable, **kwargs),
        sample_count=provider.sample_count,
        target_count=len(targets),
        axis_kind=provider.axis_kind,
    )


def _normalized(actual, expected):
    scale = max(float(np.linalg.norm(expected)), np.finfo(np.float64).tiny)
    return float(np.linalg.norm(actual - expected) / scale)


def test_imports_this_worktree_source():
    assert Path(gragra.__file__).resolve().is_relative_to(WORKTREE / "src")


def test_metadata_and_blocks_are_defensive_readonly_canonical_copies():
    positions = np.array([[1.0, 0.0, 0.0]])
    axis = np.array([0.0])
    origin = np.zeros(3)
    meta = SeriesMetadata(
        positions_m=positions,
        source_ids=("s0",),
        coordinate_frame="ENU",
        coordinate_origin_m=origin,
        axis_kind="time",
        axis_values=axis,
        sample_count=1,
        source_count=1,
    )
    weights = np.array([[2]], dtype=np.int16)
    block = ScalarSourceSeriesBlock(positions, slice(0, 1), 0, weights, ("s0",))
    positions[:] = 9
    axis[:] = 9
    origin[:] = 9
    weights[:] = 9
    assert meta.positions_m.dtype == np.float64
    assert block.weights_kg.dtype == np.float64
    assert np.array_equal(meta.positions_m, [[1.0, 0.0, 0.0]])
    assert np.array_equal(block.weights_kg, [[2.0]])
    for value in (
        meta.positions_m,
        meta.axis_values,
        meta.coordinate_origin_m,
        block.positions_m,
        block.weights_kg,
    ):
        assert not value.flags.writeable
    provider = as_series_provider(
        meta, _scalar_factory(meta, np.ones((1, 1)), source_partition=(1,))
    )
    with pytest.raises(AttributeError):
        provider.metadata = meta


@pytest.mark.parametrize(
    "field,value", [("sample_count", True), ("source_count", np.bool_(True))]
)
def test_metadata_rejects_boolean_counts(field, value):
    kwargs = {
        "positions_m": np.array([[1.0, 0.0, 0.0]]),
        "source_ids": ("a",),
        "coordinate_frame": "ENU",
        "coordinate_origin_m": np.zeros(3),
        "axis_kind": "time",
        "axis_values": np.array([0.0]),
        "sample_count": 1,
        "source_count": 1,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        SeriesMetadata(**kwargs)


def test_frequency_and_time_axis_contracts_are_explicit():
    base = {
        "positions_m": np.array([[1.0, 0.0, 0.0]]),
        "source_ids": ("a",),
        "coordinate_frame": "ENU",
        "coordinate_origin_m": np.zeros(3),
        "axis_values": np.array([1.0]),
        "sample_count": 1,
        "source_count": 1,
    }
    with pytest.raises(ValueError, match="t_ref_s"):
        SeriesMetadata(axis_kind="frequency", **base)
    base["axis_values"] = np.array([0.0])
    with pytest.raises(ValueError, match="must not"):
        SeriesMetadata(axis_kind="time", t_ref_s=0.0, **base)


def test_density_kind_semantics_cannot_be_weakened_in_generic_metadata():
    common = {
        "positions_m": np.array([[1.0, 0.0, 0.0]]),
        "source_ids": ("a",),
        "coordinate_frame": "ENU",
        "coordinate_origin_m": np.zeros(3),
        "axis_values": np.array([1.0]),
        "sample_count": 1,
        "source_count": 1,
    }
    with pytest.raises(ValueError, match="requires density_kind"):
        SeriesMetadata(axis_kind="time", quantity="density", **common)
    with pytest.raises(ValueError, match="delta_rho"):
        SeriesMetadata(
            axis_kind="frequency",
            quantity="density",
            density_kind="rho",
            t_ref_s=0.0,
            phase_origin_m=np.zeros(3),
            **common,
        )
    with pytest.raises(ValueError, match="only for density"):
        SeriesMetadata(
            axis_kind="time",
            quantity="displacement",
            density_kind="delta_rho",
            **common,
        )
    accepted = SeriesMetadata(
        axis_kind="time", quantity="density", density_kind="rho", **common
    )
    assert accepted.density_kind == "rho"


def test_block_rejects_bool_start_complex_mass_and_wrong_shapes():
    with pytest.raises(ValueError, match="source_start"):
        ScalarSourceSeriesBlock([[1.0, 0.0, 0.0]], slice(0, 1), True, [[1.0]], ("a",))
    with pytest.raises(TypeError, match="real-valued"):
        DisplacementSourceSeriesBlock(
            [[1.0, 0.0, 0.0]],
            slice(0, 1),
            0,
            np.array([1.0 + 0.0j]),
            np.zeros((1, 1, 3)),
            ("a",),
        )
    with pytest.raises(ValueError, match="Q,B,3"):
        DisplacementSourceSeriesBlock(
            [[1.0, 0.0, 0.0]],
            slice(0, 1),
            0,
            np.array([1.0]),
            np.zeros((1, 1, 2)),
            ("a",),
        )


def test_provider_rejects_bare_and_reused_one_shot_generators():
    meta = _metadata()
    generator = _scalar_factory(meta, np.ones((2, 3)))(slice(0, 1), 3)
    with pytest.raises(TypeError, match="callable"):
        as_series_provider(meta, generator)

    def same_generator(_samples, _size):
        return generator

    provider = as_series_provider(meta, same_generator)
    with pytest.raises(TypeError, match="fresh iterator"):
        next(stream_observable(provider, [[0.0, 0.0, 0.0]], observable="potential"))


@pytest.mark.parametrize(
    "mode,match",
    [
        ("missing_tail", "missing tail"),
        ("gap", "gap"),
        ("overlap", "overlap"),
        ("reverse", "reverse"),
        ("id", "ID"),
    ],
)
def test_every_source_scan_requires_exact_ordered_coverage(mode, match):
    meta = _metadata()

    def broken(samples, _size):
        specs = {
            "missing_tail": [(0, 1, ("a",))],
            "gap": [(0, 1, ("a",)), (2, 1, ("c",))],
            "overlap": [(0, 2, ("a", "b")), (1, 2, ("b", "c"))],
            "reverse": [(0, 2, ("a", "b")), (1, 1, ("b",))],
            "id": [(0, 1, ("x",)), (1, 2, ("b", "c"))],
        }[mode]
        for start, width, ids in specs:
            yield ScalarSourceSeriesBlock(
                meta.positions_m[start : start + width],
                samples,
                start,
                np.ones((samples.stop - samples.start, width)),
                ids,
            )

    with pytest.raises(ValueError, match=match):
        next(
            stream_observable(
                as_series_provider(meta, broken),
                [[0.0, 0.0, 0.0]],
                observable="potential",
            )
        )


def test_late_last_sample_nan_is_rejected_before_first_output():
    meta = _metadata()
    values = np.ones((2, 3))
    values[-1, -1] = np.nan
    blocks = stream_observable(
        as_series_provider(meta, _scalar_factory(meta, values)),
        [[0.0, 0.0, 0.0]],
        observable="potential",
        sample_block_size=1,
    )
    with pytest.raises(ValueError, match="finite"):
        next(blocks)


def test_computation_uses_one_validated_scan_and_rejects_replay_changes():
    meta = _metadata(positions=[[1.0, 0.0, 0.0]], ids=("a",))
    calls = 0

    def changing(samples, _size):
        nonlocal calls
        calls += 1
        value = 1.0 if calls <= 2 else 2.0
        yield ScalarSourceSeriesBlock(
            meta.positions_m,
            samples,
            0,
            np.full((samples.stop - samples.start, 1), value),
            meta.source_ids,
        )

    blocks = stream_observable(
        as_series_provider(meta, changing),
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        observable="potential",
        sample_block_size=2,
        target_block_size=1,
    )
    first = next(blocks)
    assert calls == 2
    assert first.values[0, 0] == pytest.approx(-G_SI)
    with pytest.raises(ValueError, match="replay"):
        next(blocks)


def test_global_collision_precheck_reports_global_indices_and_id():
    meta = _metadata()
    provider = as_series_provider(meta, _scalar_factory(meta, np.ones((2, 3))))
    with pytest.raises(ValueError, match=r"sample 0.*target 1.*source 2.*'c'"):
        next(
            stream_observable(
                provider,
                [[50.0, 0.0, 0.0], meta.positions_m[2]],
                observable="gradient",
                target_block_size=1,
                source_block_size=1,
                softening_m=0.0,
            )
        )


def test_numeric_settings_reject_bool_nonfinite_and_nonpositive():
    meta = _metadata()
    provider = as_series_provider(meta, _scalar_factory(meta, np.ones((2, 3))))
    for kwargs, match in [
        ({"sample_block_size": True}, "sample_block_size"),
        ({"source_block_size": 0}, "source_block_size"),
        ({"target_block_size": -1}, "target_block_size"),
        ({"kernel_chunk_size": False}, "kernel_chunk_size"),
        ({"memory_budget_bytes": np.inf}, "memory_budget_bytes"),
        ({"softening_m": np.nan}, "softening_m"),
    ]:
        with pytest.raises(ValueError, match=match):
            next(
                stream_observable(
                    provider, [[0.0, 0.0, 0.0]], observable="potential", **kwargs
                )
            )


def test_small_memory_budget_adapts_blocks_and_impossible_budget_fails():
    positions = np.column_stack((np.arange(1.0, 9.0), np.zeros((8, 2))))
    meta = _metadata(positions=positions, ids=tuple(f"s{i}" for i in range(8)))
    provider = as_series_provider(
        meta, _scalar_factory(meta, np.ones((2, 8)), source_partition=(8,))
    )
    targets = np.column_stack((np.arange(20.0, 28.0), np.ones((8, 2))))
    output = list(
        stream_observable(
            provider,
            targets,
            observable="gradient",
            sample_block_size=2,
            source_block_size=8,
            target_block_size=8,
            kernel_chunk_size=8,
            memory_budget_bytes=20_000,
        )
    )
    assert len(output) > 1
    with pytest.raises(MemoryError, match="minimum working set"):
        next(
            stream_observable(
                provider, targets, observable="gradient", memory_budget_bytes=1
            )
        )


def test_memory_estimate_includes_zero_softening_collision_equality_plane():
    common = {
        "sample_block_size": 1,
        "target_block_size": 128,
        "kernel_chunk_size": 1,
        "observable": "potential",
        "complex_values": False,
    }
    small = estimate_streaming_working_bytes(source_block_size=1, **common)
    large = estimate_streaming_working_bytes(source_block_size=100_001, **common)
    # The three-coordinate comparison and its reduced equality plane require
    # at least four bool bytes per target/source pair.
    assert large - small >= 4 * 128 * 100_000


def _guard_numba_installed() -> None:
    try:
        import numba  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != "numba":
            raise
        pytest.skip("numba is not installed")


@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize("observable", ["potential", "acceleration", "gradient"])
@pytest.mark.parametrize("axis_kind", ["time", "frequency"])
@pytest.mark.parametrize("source_kind", ["scalar", "displacement"])
def test_streaming_matches_eager_all_observables_and_dtypes(
    backend, observable, axis_kind, source_kind
):
    if backend == "numba":
        _guard_numba_installed()
    meta = _metadata(axis_kind=axis_kind)
    targets = np.array([[0.2, -0.5, 0.1], [4.2, 0.3, -0.7]])
    factor = 1.0 if axis_kind == "time" else (1.0 + 0.25j)
    if source_kind == "scalar":
        values = factor * np.array([[1.0, -2.0, 0.5], [0.25, 3.0, -1.5]])
        provider = as_series_provider(meta, _scalar_factory(meta, values))
        eager = np.stack(
            [
                G_SI
                * direct_point_sum(
                    WeightedPointSource(meta.positions_m, values[q]),
                    targets,
                    observable=observable,
                    chunk_size=2,
                    backend=backend,
                )
                for q in range(2)
            ]
        )
    else:
        mass = np.array([2.0, 0.0, 3.0])
        displacement = factor * np.array(
            [
                [[0.1, 0.2, -0.1], [0.2, -0.3, 0.4], [0.0, 0.5, 0.1]],
                [[-0.2, 0.1, 0.3], [0.4, 0.0, -0.1], [0.3, -0.2, 0.2]],
            ]
        )
        provider = as_series_provider(
            meta, _displacement_factory(meta, mass, displacement)
        )
        eager = np.stack(
            [
                G_SI
                * direct_point_displacement_sum(
                    mass,
                    displacement[q],
                    meta.positions_m,
                    targets,
                    observable=observable,
                    chunk_size=2,
                    backend=backend,
                )
                for q in range(2)
            ]
        )
    actual = _collect(
        provider,
        targets,
        observable,
        sample_block_size=1,
        source_block_size=2,
        target_block_size=1,
        kernel_chunk_size=2,
        backend=backend,
    )
    expected = eager if axis_kind == "time" else np.swapaxes(eager, 0, 1)
    assert _normalized(actual, expected) <= 1e-11
    assert actual.dtype == (np.float64 if axis_kind == "time" else np.complex128)
    assert not actual.flags.writeable


def test_as_series_provider_adapts_density_displacement_and_surface_without_g_imports():
    common = {
        "positions_m": np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        "source_ids": ("a", "b"),
        "coordinate_frame": "ENU",
        "coordinate_origin_m": np.zeros(3),
        "axis_kind": "time",
        "axis_values": np.array([0.0, 1.0]),
        "sample_count": 2,
        "source_count": 2,
        "t_ref_s": None,
        "phase_origin_m": None,
    }
    density = SimpleNamespace(
        **common,
        volumes_m3=np.array([2.0, 3.0]),
        density_kg_m3=np.array([[1.0, -1.0], [0.5, 2.0]]),
        density_kind="delta_rho",
    )
    block = next(as_series_provider(density).blocks(slice(0, 1), 2))
    assert np.array_equal(block.weights_kg, [[2.0, -3.0]])
    displacement = SimpleNamespace(
        **common,
        mass_kg=np.array([4.0, 5.0]),
        displacement_m=np.ones((2, 2, 3)),
    )
    assert isinstance(
        next(as_series_provider(displacement).blocks(slice(0, 1), 2)),
        DisplacementSourceSeriesBlock,
    )
    surface = SimpleNamespace(
        **common,
        areas_m2=np.array([7.0, 11.0]),
        normals=np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        surface_density_kg_m2=np.array([[2.0, 3.0], [4.0, 5.0]]),
    )
    block = next(as_series_provider(surface).blocks(slice(0, 1), 2))
    assert np.array_equal(block.weights_kg, [[14.0, 33.0]])


def test_observable_collector_validates_exact_coverage_axes_and_readonly():
    with pytest.raises(ValueError, match="axis_kind"):
        collect_observable_blocks([], sample_count=1, target_count=1, axis_kind="x")
    one = ObservableSeriesBlock(slice(0, 1), slice(0, 1), np.ones((1, 1)))
    with pytest.raises(ValueError, match="missing"):
        collect_observable_blocks(
            [one], sample_count=2, target_count=1, axis_kind="time"
        )
    with pytest.raises(ValueError, match="overlap"):
        collect_observable_blocks(
            [one, one], sample_count=1, target_count=1, axis_kind="time"
        )
    bad = ObservableSeriesBlock(slice(0, 1), slice(1, 2), np.ones((1, 1)))
    with pytest.raises(ValueError, match="range"):
        collect_observable_blocks(
            [bad], sample_count=1, target_count=1, axis_kind="time"
        )
    collected = collect_observable_blocks(
        [ObservableSeriesBlock(slice(0, 2), slice(0, 1), np.ones((2, 1)))],
        sample_count=2,
        target_count=1,
        axis_kind="frequency",
    )
    assert collected.shape == (1, 2)
    assert not collected.flags.writeable


def test_output_block_is_a_defensive_readonly_copy():
    values = np.ones((1, 1, 3))
    block = ObservableSeriesBlock(slice(0, 1), slice(0, 1), values)
    values[:] = 9
    assert np.array_equal(block.values, np.ones((1, 1, 3)))
    assert not block.values.flags.writeable


def test_source_scan_does_not_retain_all_block_objects():
    positions = np.column_stack((np.arange(1.0, 51.0), np.zeros((50, 2))))
    meta = _metadata(positions=positions, ids=tuple(f"s{i}" for i in range(50)))
    live = []
    maximum = 0

    def factory(samples, _size):
        nonlocal maximum, live
        for start in range(50):
            block = ScalarSourceSeriesBlock(
                meta.positions_m[start : start + 1],
                samples,
                start,
                np.ones((samples.stop - samples.start, 1)),
                meta.source_ids[start : start + 1],
            )
            live = [reference for reference in live if reference() is not None]
            live.append(weakref.ref(block))
            maximum = max(maximum, len(live))
            yield block

    list(
        stream_observable(
            as_series_provider(meta, factory),
            [[0.0, 0.0, 0.0]],
            observable="potential",
            sample_block_size=2,
            source_block_size=1,
        )
    )
    assert maximum <= 2


def test_guard_numba_installed_distinguishes_missing_numba_from_internal_errors(
    monkeypatch,
):
    """Ensure _guard_numba_installed only skips when numba itself is missing.

    Internal dependency failures (e.g. llvmlite missing) and generic ImportErrors
    must be propagated as failures rather than silently skipped.
    """
    import builtins

    real_import = builtins.__import__

    # Case 1: numba itself is missing -> pytest.skip
    def mock_missing_numba(name, *args, **kwargs):
        if name == "numba":
            raise ModuleNotFoundError("No module named 'numba'", name="numba")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_missing_numba)
    with pytest.raises(pytest.skip.Exception, match="numba is not installed"):
        _guard_numba_installed()

    # Case 2: numba internal dependency is missing -> ModuleNotFoundError raised
    def mock_missing_internal(name, *args, **kwargs):
        if name == "numba":
            raise ModuleNotFoundError("No module named 'llvmlite'", name="llvmlite")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_missing_internal)
    with pytest.raises(ModuleNotFoundError, match="llvmlite"):
        _guard_numba_installed()

    # Case 3: generic ImportError -> ImportError raised
    def mock_import_error(name, *args, **kwargs):
        if name == "numba":
            raise ImportError("DLL load failed or symbol missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import_error)
    with pytest.raises(ImportError, match="DLL load failed"):
        _guard_numba_installed()
