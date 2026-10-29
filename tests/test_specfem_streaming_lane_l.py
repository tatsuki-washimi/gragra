"""Bounded SPECFEM context/provider tests for lane L."""

from __future__ import annotations

import weakref
from pathlib import Path

import numpy as np
import pytest
from test_specfem_reader import _write_custom_time_sampled_gll

import gragra
from gragra.adapters.specfem import schema
from gragra.adapters.specfem.reader import read_specfem_gll_hdf5
from gragra.adapters.specfem.streaming import SpecfemGLLStreamReader
from gragra.constants import G_SI
from gragra.integrators.point_direct import direct_point_displacement_sum
from gragra.observables.streaming import collect_observable_blocks, stream_observable

WORKTREE = Path(__file__).resolve().parents[1]


def _fixture(path, *, elements=2, samples=3):
    rng = np.random.default_rng(20260912)
    positions = rng.normal(size=(elements, 125, 3))
    positions[..., 0] += np.arange(elements)[:, None] * 10.0 + 5.0
    density = np.full((elements, 125), 2200.0)
    volume = np.linspace(0.5, 1.5, elements * 125).reshape(elements, 125)
    displacement = rng.normal(scale=1e-4, size=(samples, elements, 125, 3))
    _write_custom_time_sampled_gll(
        path,
        ngll_shape=(5, 5, 5),
        coordinates=positions,
        density=density,
        quadrature_volume=volume,
        displacement=displacement,
    )
    return positions, density, volume, displacement


def test_context_provider_preserves_gll_order_ids_mass_and_closes(tmp_path):
    assert Path(gragra.__file__).resolve().is_relative_to(WORKTREE / "src")
    path = tmp_path / "gll.h5"
    positions, density, volume, displacement = _fixture(path)
    reader = SpecfemGLLStreamReader(
        path, profile_density_range=(2000.0, 2300.0), slab_elements=1
    )
    with reader as opened:
        provider = opened.provider
        assert provider.source_ids[0] == "0:0"
        assert provider.source_ids[124] == "0:124"
        assert provider.source_ids[125] == "1:0"
        assert provider.source_ids[-1] == "1:124"
        assert np.array_equal(provider.positions_m, positions.reshape(-1, 3))
        blocks = list(provider.blocks(slice(0, 3), 73))
        assert [len(block.source_ids) for block in blocks] == [73, 73, 73, 31]
        assert np.array_equal(
            np.concatenate([block.mass_kg for block in blocks]),
            (density * volume).reshape(-1),
        )
        assert np.array_equal(
            np.concatenate([block.displacement_m for block in blocks], axis=1),
            displacement.reshape(3, -1, 3),
        )
        assert opened.chunk_cache_bytes >= 0
        assert opened.estimated_resident_bytes <= 2 * 1024**3
    assert reader.closed
    with pytest.raises(RuntimeError, match="closed"):
        _ = reader.provider
    with pytest.raises(RuntimeError, match="closed"):
        next(provider.blocks(slice(0, 1), 10))


def test_context_can_expose_one_contiguous_element_slab_with_global_ids(tmp_path):
    path = tmp_path / "element-slab.h5"
    positions, density, volume, _ = _fixture(path)
    with SpecfemGLLStreamReader(
        path,
        profile_density_range=None,
        element_slice=slice(1, 2),
        sample_slice=slice(0, 1),
        slab_elements=1,
    ) as reader:
        provider = reader.provider
        assert provider.source_count == 125
        assert provider.source_ids[0] == "1:0"
        assert provider.source_ids[-1] == "1:124"
        assert np.array_equal(provider.positions_m, positions[1])
        block = next(provider.blocks(slice(0, 1), 125))
        assert np.array_equal(block.mass_kg, (density[1] * volume[1]).reshape(-1))


def test_collision_error_uses_global_element_local_gll_id(tmp_path):
    path = tmp_path / "collision.h5"
    positions, _, _, _ = _fixture(path)
    with SpecfemGLLStreamReader(
        path,
        profile_density_range=None,
        sample_slice=slice(0, 1),
        slab_elements=1,
    ) as reader:
        target = positions[1, 5][None, :]
        blocks = stream_observable(
            reader.provider,
            target,
            observable="acceleration",
            sample_block_size=1,
            source_block_size=37,
            target_block_size=1,
            kernel_chunk_size=16,
        )
        with pytest.raises(ValueError, match=r"source_id '1:5'"):
            next(blocks)


def test_context_closes_when_body_or_enter_validation_raises(tmp_path):
    path = tmp_path / "body.h5"
    _fixture(path)
    reader = SpecfemGLLStreamReader(path, profile_density_range=None)
    with pytest.raises(RuntimeError, match="body"), reader:
        raise RuntimeError("body")
    assert reader.closed

    bad = tmp_path / "bad.h5"
    _fixture(bad)
    h5py = pytest.importorskip("h5py")
    with h5py.File(bad, "r+") as h5:
        h5[schema.PATH_FIELD_DISPLACEMENT][-1, -1, -1, -1] = np.nan
    failed = SpecfemGLLStreamReader(bad, profile_density_range=None, slab_elements=1)
    with pytest.raises(ValueError, match="finite"), failed:
        pass
    assert failed.closed


def test_stream_validation_preserves_eager_full_acceptance_contract(tmp_path):
    path = tmp_path / "invalid-provenance.h5"
    _fixture(path)
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "r+") as h5:
        h5["metadata"].attrs["coordinate_convention"] = "NED"
    with pytest.raises(ValueError, match="coordinate_convention"):
        read_specfem_gll_hdf5(path, profile_density_range=None)
    with (
        pytest.raises(ValueError, match="coordinate_convention"),
        SpecfemGLLStreamReader(path, profile_density_range=None),
    ):
        pass


def test_stream_validator_never_calls_eager_reader_or_full_numeric_dataset(
    tmp_path, monkeypatch
):
    path = tmp_path / "spy.h5"
    _fixture(path)
    import gragra.adapters.specfem.reader as eager_module

    def eager_must_not_run(*_args, **_kwargs):
        raise AssertionError("eager called")

    monkeypatch.setattr(eager_module, "read_specfem_gll_hdf5", eager_must_not_run)
    h5py = pytest.importorskip("h5py")
    original = h5py.Dataset.__getitem__
    seen = []

    def spy(dataset, key):
        if not dataset.name.startswith("/metadata/"):
            seen.append((dataset.name, key))
            assert key != ()
            assert key is not Ellipsis
        return original(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", spy)
    with SpecfemGLLStreamReader(
        path, profile_density_range=None, sample_slice=slice(1, 3), slab_elements=1
    ) as reader:
        blocks = list(reader.provider.blocks(slice(0, 1), 60))
        assert blocks[0].sample_slice == slice(0, 1)
    assert seen


def test_provider_reads_each_time_chunk_into_one_bounded_displacement_buffer(
    tmp_path, monkeypatch
):
    path = tmp_path / "reused-provider-field-buffer.h5"
    _fixture(path, elements=2, samples=3)
    h5py = pytest.importorskip("h5py")
    original = h5py.Dataset.read_direct
    destinations = []

    def spy(dataset, destination, source_sel=None, dest_sel=None):
        if dataset.name == schema.PATH_FIELD_DISPLACEMENT:
            destinations.append(destination)
            assert destination.shape == (3, 1, 125, 3)
            assert isinstance(source_sel, tuple)
            assert source_sel[0].stop - source_sel[0].start == 1
            assert source_sel[1].stop - source_sel[1].start == 1
        return original(dataset, destination, source_sel, dest_sel)

    with SpecfemGLLStreamReader(
        path, profile_density_range=None, slab_elements=1
    ) as reader:
        monkeypatch.setattr(h5py.Dataset, "read_direct", spy)
        block = next(reader.provider.blocks(slice(0, 3), 125))
    assert block.displacement_m.shape == (3, 125, 3)
    assert len(destinations) == 3
    assert len({id(value) for value in destinations}) == 1


def test_stream_reader_does_not_cache_full_width_displacement_chunks(tmp_path):
    path = tmp_path / "no-full-width-cache.h5"
    _fixture(path, elements=2, samples=3)
    with SpecfemGLLStreamReader(
        path, profile_density_range=None, slab_elements=1
    ) as reader:
        dataset = reader._h5[schema.PATH_FIELD_DISPLACEMENT]
        chunk_bytes = int(np.prod(dataset.chunks) * dataset.dtype.itemsize)
        _, cache_bytes, _ = dataset.id.get_access_plist().get_chunk_cache()
        assert cache_bytes < chunk_bytes


@pytest.mark.parametrize("backend", ["numpy"])
@pytest.mark.parametrize("observable", ["potential", "acceleration", "gradient"])
def test_specfem_streaming_matches_eager_for_three_observables(
    tmp_path, backend, observable
):
    path = tmp_path / "parity.h5"
    positions, density, volume, displacement = _fixture(path, elements=1, samples=2)
    targets = np.array([[20.0, 2.0, -1.0], [25.0, -3.0, 4.0]])
    with SpecfemGLLStreamReader(
        path, profile_density_range=None, slab_elements=1
    ) as reader:
        provider = reader.provider
        actual = collect_observable_blocks(
            stream_observable(
                provider,
                targets,
                observable=observable,
                backend=backend,
                sample_block_size=1,
                source_block_size=47,
                target_block_size=1,
                kernel_chunk_size=31,
            ),
            sample_count=2,
            target_count=2,
            axis_kind="time",
        )
    mass = (density * volume).reshape(-1)
    eager = np.stack(
        [
            G_SI
            * direct_point_displacement_sum(
                mass,
                displacement[q].reshape(-1, 3),
                positions.reshape(-1, 3),
                targets,
                observable=observable,
                backend=backend,
                chunk_size=31,
            )
            for q in range(2)
        ]
    )
    scale = max(np.linalg.norm(eager), np.finfo(np.float64).tiny)
    assert np.linalg.norm(actual - eager) / scale <= 1e-11


def test_stream_reader_rejects_bool_sizes_and_budget_below_chunk(tmp_path):
    path = tmp_path / "limits.h5"
    _fixture(path)
    with pytest.raises(ValueError, match="slab_elements"):
        SpecfemGLLStreamReader(path, profile_density_range=None, slab_elements=True)
    with pytest.raises(ValueError, match="memory_budget_bytes"):
        SpecfemGLLStreamReader(
            path, profile_density_range=None, memory_budget_bytes=True
        )
    with (
        pytest.raises(MemoryError, match="budget"),
        SpecfemGLLStreamReader(
            path, profile_density_range=None, memory_budget_bytes=1024
        ),
    ):
        pass


def test_stream_reader_rejects_budget_below_decoded_chunk_workspace(tmp_path):
    path = tmp_path / "single-large-field-chunk.h5"
    _fixture(path, elements=8, samples=16)
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "r+") as h5:
        displacement = np.asarray(h5[schema.PATH_FIELD_DISPLACEMENT])
        del h5[schema.PATH_FIELD_DISPLACEMENT]
        h5.create_dataset(
            schema.PATH_FIELD_DISPLACEMENT,
            data=displacement,
            chunks=displacement.shape,
            compression="gzip",
        )

    reader = SpecfemGLLStreamReader(
        path,
        profile_density_range=None,
        element_slice=slice(0, 1),
        slab_elements=1,
        memory_budget_bytes=100_000,
    )
    with pytest.raises(MemoryError, match="chunk workspace"), reader:
        pass
    assert reader.closed

    with SpecfemGLLStreamReader(
        path,
        profile_density_range=None,
        element_slice=slice(0, 1),
        slab_elements=1,
        memory_budget_bytes=1_000_000,
    ) as opened:
        assert opened.decoded_chunk_workspace_bytes == displacement.nbytes
        assert opened.validation_field_buffer_bytes == displacement.nbytes
        assert opened.estimated_resident_bytes <= opened.memory_budget_bytes


def test_provider_bounds_multisample_displacement_and_defensive_copy(tmp_path):
    path = tmp_path / "multisample-provider-budget.h5"
    _, _, _, displacement = _fixture(path, elements=8, samples=64)
    expected = displacement.reshape(64, 1000, 3)

    with SpecfemGLLStreamReader(
        path,
        profile_density_range=None,
        slab_elements=8,
        memory_budget_bytes=500_000,
    ) as reader:
        source_start = 0
        widths = []
        for block in reader.provider.blocks(slice(0, 64), 1000):
            source_stop = source_start + len(block.source_ids)
            assert block.sample_slice == slice(0, 64)
            assert np.array_equal(
                block.displacement_m, expected[:, source_start:source_stop]
            )
            assert reader.last_provider_block_estimated_bytes <= 500_000
            widths.append(source_stop - source_start)
            source_start = source_stop
        assert source_start == 1000
        assert max(widths) < 1000

    with SpecfemGLLStreamReader(
        path,
        profile_density_range=None,
        slab_elements=8,
        memory_budget_bytes=4_000_000,
    ) as reader:
        block = next(reader.provider.blocks(slice(0, 64), 1000))
        assert block.displacement_m.shape == (64, 1000, 3)
        assert reader.last_provider_block_estimated_bytes <= 4_000_000


def test_provider_releases_raw_destination_before_yield_and_next_allocation(
    tmp_path, monkeypatch
):
    path = tmp_path / "provider-raw-lifetime.h5"
    _fixture(path, elements=8, samples=64)
    import gragra.adapters.specfem.streaming as streaming_module

    destinations = []
    original_empty = np.empty

    def tracked_empty(shape, *args, **kwargs):
        if tuple(shape)[0] == 64 and tuple(shape)[-2:] == (125, 3):
            assert not destinations or destinations[-1]() is None
        result = original_empty(shape, *args, **kwargs)
        if tuple(shape)[0] == 64 and tuple(shape)[-2:] == (125, 3):
            destinations.append(weakref.ref(result))
        return result

    with SpecfemGLLStreamReader(
        path,
        profile_density_range=None,
        slab_elements=8,
        memory_budget_bytes=500_000,
    ) as reader:
        monkeypatch.setattr(streaming_module.np, "empty", tracked_empty)
        blocks = reader.provider.blocks(slice(0, 64), 1000)
        first = next(blocks)
        assert first.displacement_m.shape == (64, 99, 3)
        assert destinations[0]() is None
        second = next(blocks)
        assert second.displacement_m.shape == (64, 26, 3)
        assert destinations[1]() is None
