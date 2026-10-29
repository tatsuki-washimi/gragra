"""SPECFEM streaming parity tests for the Numba backend."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numba")
pytest.importorskip("h5py")

from gragra.adapters.specfem.streaming import SpecfemGLLStreamReader
from gragra.constants import G_SI
from gragra.integrators.point_direct import direct_point_displacement_sum
from gragra.observables.streaming import collect_observable_blocks, stream_observable
from tests.test_specfem_streaming_lane_l import _fixture


@pytest.mark.parametrize("observable", ["potential", "acceleration", "gradient"])
def test_specfem_streaming_matches_eager_for_three_observables_numba(
    tmp_path, observable
):
    path = tmp_path / "parity_numba.h5"
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
                backend="numba",
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
                backend="numba",
                chunk_size=31,
            )
            for q in range(2)
        ]
    )
    scale = max(np.linalg.norm(eager), np.finfo(np.float64).tiny)
    assert np.linalg.norm(actual - eager) / scale <= 1e-11
