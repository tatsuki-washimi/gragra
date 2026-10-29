# ruff: noqa: E402, I001
"""Saved-Rayleigh A/B/C cross-path regression tests."""

from __future__ import annotations

import hashlib
import json
import gc
import multiprocessing
import sys
from pathlib import Path

import numpy as np
import pytest

WORKTREE = Path(__file__).resolve().parents[1]
sys.meta_path[:] = [
    item
    for item in sys.meta_path
    if type(item).__name__ != "ScikitBuildRedirectingFinder"
]
for entry in (WORKTREE / "src", WORKTREE):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import gragra
from gragra.observables.dipole import (
    displacement_transfer_acceleration,
    displacement_transfer_gravity_gradient,
    displacement_transfer_potential,
)
from gragra.research.rayleigh_timefit import snapshot_physical_times
from gragra.waves.rayleigh import rayleigh_displacement
from scripts import research_wave5_rayleigh_crosspaths as study


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _decode_complex(record: dict[str, object]) -> np.ndarray:
    assert record["encoding"] == "complex128_real_imag_last_axis"
    values = np.asarray(record["values"], dtype=np.float64)
    result = values[..., 0] + 1j * values[..., 1]
    assert list(result.shape) == record["shape"]
    return result


@pytest.fixture
def canonical_rayleigh_hdf5(tmp_path: Path):
    pytest.importorskip("h5py")
    from gragra.adapters.specfem import schema, writer
    from test_specfem_reader import _time_sampled_provenance

    assert Path(gragra.__file__).resolve().is_relative_to(WORKTREE / "src")
    axes = (
        np.linspace(-180.0, 180.0, 5),
        np.linspace(-140.0, 140.0, 5),
        np.linspace(-300.0, -20.0, 5),
    )
    base = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(125, 3)
    positions = np.stack((base, base + np.array([360.0, 0.0, 0.0])))
    # One shared coordinate remains two element-local GLL sources with distinct
    # quadrature weights and IDs. The cross-path must never deduplicate it.
    positions[1, 0] = positions[0, -1]
    density = np.full((2, 125), study.RHO_KG_M3, dtype=np.float64)
    volume = np.linspace(0.75, 1.75, 250, dtype=np.float64).reshape(2, 125)
    phasor = rayleigh_displacement(
        positions.reshape(-1, 3),
        study.DIRECTION,
        study.WAVENUMBER_RAD_M,
        study.VP_M_S,
        study.VS_M_S,
        amplitude_m=study.SURFACE_VERTICAL_AMPLITUDE_M,
    ).reshape(2, 125, 3)
    steps = np.arange(study.SAVED_SAMPLE_COUNT, dtype=np.int64)
    physical_times = snapshot_physical_times(
        steps, t_init_s=study.T_REF_S, dt_s=study.DT_S
    )
    phase = study.OMEGA_RAD_S * (physical_times - study.T_REF_S)
    displacement = (
        phasor.real[None, ...] * np.cos(phase)[:, None, None, None]
        + phasor.imag[None, ...] * np.sin(phase)[:, None, None, None]
    )
    provenance = _time_sampled_provenance(np.zeros(2, dtype=np.bool_))
    provenance.update(
        {
            "dt_s": study.DT_S,
            "nstep": study.SAVED_SAMPLE_COUNT,
            "t0": -study.T_REF_S,
            "output_interval_steps": 1,
            "f_max_analysis": 4.0,
            "f_min_analysis": study.FREQUENCY_HZ,
            "minimum_cycles": 2.0,
        }
    )
    path = tmp_path / "canonical-rayleigh.h5"
    writer.write_gll_hdf5(
        path,
        output_sampling_kind="time_sampled",
        ngll_shape=(5, 5, 5),
        coordinates=positions,
        quadrature_volume=volume,
        element_id=np.arange(2, dtype=np.int64),
        partition_id=np.zeros(2, dtype=np.int32),
        domain_id=np.full(2, schema.IDOMAIN_ELASTIC, dtype=np.int32),
        material_id=np.ones(2, dtype=np.int32),
        is_pml=np.zeros(2, dtype=np.bool_),
        density=density,
        provenance=provenance,
        displacement=displacement,
        time_step=steps,
        chunk_element_size=1,
    )
    return path, positions, density * volume, steps


def test_rayleigh_windows_preserve_steps_and_use_post_predictor_physical_times():
    steps = np.arange(study.SAVED_SAMPLE_COUNT, dtype=np.int64)
    raw = study.T_REF_S + steps.astype(np.float64) * study.DT_S
    windows = study.rayleigh_windows(raw, steps)
    assert windows["full"]["input_sample_indices"] == list(range(240))
    assert windows["front"]["input_sample_indices"][-1] == 159
    assert windows["full"]["time_step_indices"][0] == 0
    assert windows["full"]["physical_times_s"][0] == 10.005
    assert windows["front"]["physical_times_s"][-1] <= 10.8 + 1e-12
    assert windows["full"]["raw_times_s"] != windows["full"]["physical_times_s"]


@pytest.mark.parametrize(
    ("raw_transform", "step_transform", "match"),
    [
        (lambda raw: raw - study.T_REF_S, lambda steps: steps, "raw times"),
        (lambda raw: raw + study.DT_S, lambda steps: steps, "raw times"),
        (lambda raw: raw, lambda steps: steps + 1, "time_step"),
        (lambda raw: raw[:-1], lambda steps: steps[:-1], "exactly 240"),
    ],
)
def test_rayleigh_windows_reject_noncanonical_phase_axes(
    raw_transform, step_transform, match
):
    steps = np.arange(study.SAVED_SAMPLE_COUNT, dtype=np.int64)
    raw = study.T_REF_S + steps.astype(np.float64) * study.DT_S
    with pytest.raises(ValueError, match=match):
        study.rayleigh_windows(raw_transform(raw), step_transform(steps))


def test_resource_plan_allows_blocks_larger_than_old_fixed_ceiling():
    plan = study._resource_plan(
        reader_resident_bytes=0,
        requested_block_elements=512,
        selected_elements=512,
        memory_budget_bytes=study.DEFAULT_MEMORY_BUDGET_BYTES,
    )
    assert plan["effective_source_block_elements"] == 512
    assert "np.linalg.lstsq RHS copy" in plan["estimate_scope"]


def test_memory_preflight_shrinks_until_combined_rss_is_safe(monkeypatch):
    measurements = iter(
        [
            {
                "status": "error",
                "type": "WorkerExit",
                "message": "memory preflight worker exited with code -9",
            },
            {"status": "ok", "peak_rss_bytes": 1_000_000_000},
        ]
    )
    monkeypatch.setattr(
        study,
        "_measure_real_memory_preflight",
        lambda *_args, **_kwargs: next(measurements),
    )
    probe = {
        "selected_start": 0,
        "selected_stop": 512,
        "reader_estimated_resident_bytes": 10_000_000,
        "selected_elements": 512,
    }
    result = study._choose_source_block(
        probe,
        hdf5_path="unused.h5",
        requested_block_elements=512,
        memory_budget_bytes=study.DEFAULT_MEMORY_BUDGET_BYTES,
    )
    assert [item["source_block_elements"] for item in result["attempts"]] == [512, 256]
    assert result["attempts"][0]["type"] == "WorkerExit"
    assert result["effective_source_block_elements"] == 256
    assert result["passed"] is True


def test_memory_preflight_worker_fits_nonzero_touched_buffers(monkeypatch):
    observed = {}

    def fit(values, physical_times, omega_rad_s, *, t_ref_s):
        observed["values"] = np.array(values, copy=True)
        observed["physical_times"] = np.array(physical_times, copy=True)
        observed["omega_rad_s"] = omega_rad_s
        observed["t_ref_s"] = t_ref_s
        return np.zeros((study.GLL_POINTS_PER_ELEMENT, 3), dtype=np.complex128)

    class Connection:
        def __init__(self):
            self.messages = []
            self.closed = False

        def send(self, value):
            self.messages.append(value)

        def close(self):
            self.closed = True

    connection = Connection()
    monkeypatch.setattr(study, "fit_complex_amplitude", fit)
    study._memory_preflight_worker(connection, 1)

    values = observed["values"]
    assert values.shape == (study.SAVED_SAMPLE_COUNT, study.GLL_POINTS_PER_ELEMENT, 3)
    assert np.isfinite(values).all()
    assert np.count_nonzero(values) == values.size
    assert values.flags.owndata
    assert observed["omega_rad_s"] == study.OMEGA_RAD_S
    assert observed["t_ref_s"] == study.T_REF_S
    assert connection.messages[0]["status"] == "ok"
    assert connection.closed is True


def _run_released_parent_high_water_probe(connection):
    """Run the parent-history probe in a fresh process for full-suite safety."""
    before_current = study._current_rss_bytes()
    parent_load = np.ones(96 * 1024 * 1024, dtype=np.uint8)
    parent_load[::4096] = 7
    parent_high_water = study._rss_bytes()
    del parent_load
    gc.collect()
    result = study._measure_memory_preflight(1)
    connection.send(
        {
            "before_current": before_current,
            "parent_high_water": parent_high_water,
            "result": result,
        }
    )
    connection.close()


def test_spawn_preflight_ignores_released_parent_high_water_mark():
    if sys.platform != "linux":
        pytest.skip("spawn/procfs high-water probe is Linux-specific")

    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_run_released_parent_high_water_probe,
        args=(child,),
    )
    process.start()
    child.close()
    record = parent.recv()
    process.join()

    assert process.exitcode == 0
    assert record["parent_high_water"] >= record["before_current"] + 64 * 1024 * 1024
    result = record["result"]

    assert result["status"] == "ok"
    assert result["measurement"] == "isolated worker /proc/self/status VmHWM"
    assert result["peak_rss_bytes"] < record["parent_high_water"] - 16 * 1024 * 1024


def test_memory_preflight_reports_interval_peak_after_transient_release(monkeypatch):
    def fit(values, physical_times, omega_rad_s, *, t_ref_s):
        del physical_times, omega_rad_s, t_ref_s
        temporary = np.ones((32 * 1024 * 1024,), dtype=np.uint8)
        temporary[::4096] = 5
        del temporary
        return np.zeros((values.shape[1], values.shape[2]), dtype=np.complex128)

    class Connection:
        def __init__(self):
            self.messages = []

        def send(self, value):
            self.messages.append(value)

        def close(self):
            pass

    monkeypatch.setattr(study, "fit_complex_amplitude", fit)
    if sys.platform != "linux":
        pytest.skip("VmHWM interval probe is Linux-specific")
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=study._memory_preflight_worker, args=(child, 1))
    process.start()
    child.close()
    result = parent.recv()
    process.join()
    parent.close()

    assert result["status"] == "ok"
    assert result["peak_rss_bytes"] > result["current_rss_bytes"] + 16 * 1024 * 1024


def test_memory_preflight_normalizes_closed_worker_pipe(monkeypatch):
    class Parent:
        def poll(self):
            return True

        def recv(self):
            raise EOFError

        def close(self):
            pass

    class Child:
        def close(self):
            pass

    class Process:
        exitcode = -9

        def start(self):
            pass

        def join(self):
            pass

        def close(self):
            pass

    class Context:
        def Pipe(self, *, duplex):  # noqa: N802 - mirrors multiprocessing API
            assert duplex is False
            return Parent(), Child()

        def Process(self, *, target, args):  # noqa: N802 - mirrors multiprocessing API
            assert target is study._memory_preflight_worker
            assert args[1] == 8
            return Process()

    monkeypatch.setattr(study.multiprocessing, "get_context", lambda _kind: Context())
    result = study._measure_memory_preflight(8)
    assert result == {
        "status": "error",
        "type": "WorkerExit",
        "message": "memory preflight worker exited with code -9 without a result",
    }


def test_memory_preflight_preserves_failed_attempt_evidence(monkeypatch):
    monkeypatch.setattr(
        study,
        "_measure_real_memory_preflight",
        lambda *_args, **_kwargs: {
            "status": "error",
            "type": "MemoryError",
            "message": "oom",
        },
    )
    probe = {
        "selected_start": 0,
        "selected_stop": 1,
        "reader_estimated_resident_bytes": 0,
        "selected_elements": 1,
    }
    result = study._choose_source_block(
        probe,
        hdf5_path="unused.h5",
        requested_block_elements=1,
        memory_budget_bytes=study.DEFAULT_MEMORY_BUDGET_BYTES,
    )
    assert result["passed"] is False
    assert result["effective_source_block_elements"] is None
    assert result["attempts"] == [
        {
            "source_block_elements": 1,
            "status": "error",
            "type": "MemoryError",
            "message": "oom",
        }
    ]


def test_real_memory_preflight_runs_two_actual_provider_blocks(
    canonical_rayleigh_hdf5,
):
    path, _, _, _ = canonical_rayleigh_hdf5

    class Connection:
        def __init__(self):
            self.messages = []

        def send(self, value):
            self.messages.append(value)

        def close(self):
            pass

    connection = Connection()
    study._real_memory_preflight_worker(
        connection,
        str(path),
        0,
        2,
        1,
        study.DEFAULT_MEMORY_BUDGET_BYTES,
    )
    result = connection.messages[0]
    assert result["status"] == "ok"
    assert result["measurement_scope"] == (
        "actual HDF reader plus two 240-sample provider blocks and harmonic fits"
    )
    assert result["selected_elements"] == 2
    assert result["blocks_processed"] == 2
    assert result["current_rss_bytes"] > 0
    assert result["peak_rss_bytes"] > 0
    assert result["peak_rss_bytes"] >= result["current_rss_bytes"]


def test_real_memory_preflight_uses_procfs_peak_not_process_lifetime_peak(
    canonical_rayleigh_hdf5, monkeypatch
):
    path, _, _, _ = canonical_rayleigh_hdf5

    class Connection:
        def __init__(self):
            self.messages = []

        def send(self, value):
            self.messages.append(value)

        def close(self):
            pass

    monkeypatch.setattr(
        study, "_rss_bytes", lambda: study.DEFAULT_MEMORY_BUDGET_BYTES + 1
    )
    monkeypatch.setattr(study, "_current_rss_bytes", lambda: 654_321)
    monkeypatch.setattr(study, "_peak_rss_bytes", lambda: 123_456)
    connection = Connection()
    study._real_memory_preflight_worker(
        connection,
        str(path),
        0,
        2,
        1,
        study.DEFAULT_MEMORY_BUDGET_BYTES,
    )

    result = connection.messages[0]
    assert result["status"] == "ok"
    assert result["current_rss_bytes"] == 654_321
    assert result["peak_rss_bytes"] == 123_456


def test_synthetic_solver_matches_analytic_gll_for_both_windows_and_observables(
    canonical_rayleigh_hdf5,
):
    path, positions, mass, _ = canonical_rayleigh_hdf5
    result = study.evaluate_saved_rayleigh(path, source_block_elements=1)

    assert result["case_scope"]["label"] == "FULL_SAVED_CASE"
    assert result["case_scope"]["is_full_saved_case"] is True
    assert result["targets_m"] == study.canonical_targets_m().tolist()
    sources = result["discrete_sources"]
    assert sources["source_count"] == 250
    assert sources["gll_points_per_element"] == 125
    assert sources["first_source_id"] == "0:0"
    assert sources["last_source_id"] == "1:124"
    assert sources["positions_float64_sha256"] == _array_sha256(
        positions.reshape(-1, 3)
    )
    assert sources["mass_float64_sha256"] == _array_sha256(mass.reshape(-1))
    resources = result["resource_estimate"]
    logical_evaluation_bytes = (160 + 240) * 2 * 125 * 3 * 8
    assert resources["estimated_evaluation_logical_field_bytes"] == (
        logical_evaluation_bytes
    )
    assert resources["estimated_evaluation_decoded_chunk_bytes"] == (
        logical_evaluation_bytes
    )
    assert resources["field_chunk_shape"] == [1, 1, 125, 3]

    analytic = rayleigh_displacement(
        positions.reshape(-1, 3),
        study.DIRECTION,
        study.WAVENUMBER_RAD_M,
        study.VP_M_S,
        study.VS_M_S,
        amplitude_m=study.SURFACE_VERTICAL_AMPLITUDE_M,
    )
    expected_b = {
        "potential": displacement_transfer_potential(
            positions.reshape(-1, 3),
            mass.reshape(-1),
            analytic,
            study.canonical_targets_m(),
        ),
        "acceleration": displacement_transfer_acceleration(
            positions.reshape(-1, 3),
            mass.reshape(-1),
            analytic,
            study.canonical_targets_m(),
        ),
        "gradient": displacement_transfer_gravity_gradient(
            positions.reshape(-1, 3),
            mass.reshape(-1),
            analytic,
            study.canonical_targets_m(),
        ),
    }

    for window_name, expected_count in (("front", 160), ("full", 240)):
        window = result["windows"][window_name]
        assert len(window["time_step_indices"]) == expected_count
        for observable in ("potential", "acceleration", "gradient"):
            a = _decode_complex(window["paths"]["A_solver_fitted"][observable])
            b = _decode_complex(window["paths"]["B_analytic_same_gll"][observable])
            c = _decode_complex(window["paths"]["C_infinite_halfspace"][observable])
            np.testing.assert_allclose(a, b, rtol=2e-13, atol=0.0)
            np.testing.assert_allclose(b, expected_b[observable], rtol=2e-14, atol=0.0)
            assert np.all(np.isfinite(c))
            metric = window["metrics_by_target"][observable]
            assert metric["norm"] in {
                "complex_scalar_absolute",
                "complex_vector_2_norm",
                "complex_tensor_frobenius_norm",
            }
            assert max(metric["A_minus_B_normalized_by_C"]) < 1e-12
            assert "acceptance_gate" not in metric
            for comparison in ("A_minus_B", "B_minus_C", "A_minus_C"):
                normalized = np.asarray(
                    metric[f"{comparison}_normalized_by_C"], dtype=np.float64
                )
                statistics = metric[f"{comparison}_normalized_by_C_statistics"]
                assert statistics == pytest.approx(
                    {
                        "mean": float(np.mean(normalized)),
                        "median": float(np.median(normalized)),
                        "standard_deviation": float(np.std(normalized)),
                    }
                )
                decoded = _decode_complex(window["differences"][comparison][observable])
                assert decoded.shape == c.shape


def test_source_block_size_is_numerically_transparent(canonical_rayleigh_hdf5):
    path, _, _, _ = canonical_rayleigh_hdf5
    one = study.evaluate_saved_rayleigh(path, source_block_elements=1)
    two = study.evaluate_saved_rayleigh(path, source_block_elements=2)
    for window_name in ("front", "full"):
        for path_name in (
            "A_solver_fitted",
            "B_analytic_same_gll",
            "C_infinite_halfspace",
        ):
            for observable in ("potential", "acceleration", "gradient"):
                np.testing.assert_allclose(
                    _decode_complex(
                        one["windows"][window_name]["paths"][path_name][observable]
                    ),
                    _decode_complex(
                        two["windows"][window_name]["paths"][path_name][observable]
                    ),
                    rtol=2e-14,
                    atol=0.0,
                )


def test_element_subset_is_loudly_labeled(canonical_rayleigh_hdf5):
    path, _, _, _ = canonical_rayleigh_hdf5
    result = study.evaluate_saved_rayleigh(
        path, element_start=1, element_stop=2, source_block_elements=1
    )
    scope = result["case_scope"]
    assert scope["label"] == "SUBSET_ONLY_NOT_WHOLE_SAVED_CASE"
    assert scope["is_full_saved_case"] is False
    assert scope["selected_element_range"] == [1, 2]
    assert scope["selected_element_count"] == 1
    assert scope["total_element_count"] == 2


def test_run_serializes_complete_complex_results_and_refuses_overwrite(
    canonical_rayleigh_hdf5, tmp_path
):
    path, _, _, _ = canonical_rayleigh_hdf5
    output = tmp_path / "result.json"
    returned = study.run(
        path,
        output,
        source_block_elements=1,
        argv=["research_wave5_rayleigh_crosspaths.py", "fixture"],
    )
    assert returned == output
    payload = json.loads(output.read_text())
    assert payload["status"] == "complete"
    assert payload["exit"] == 0
    assert payload["argv"][-1] == "fixture"
    assert payload["peak_rss_bytes"] > 0
    assert payload["peak_rss_bytes"] <= study.DEFAULT_MEMORY_BUDGET_BYTES
    assert payload["resource_estimate"]["measured_memory_preflight"]["attempts"]
    assert (
        payload["input_file"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert payload["source_files"]
    _decode_complex(payload["windows"]["full"]["paths"]["A_solver_fitted"]["gradient"])
    assert not Path(f"{output}.partial").exists()
    with pytest.raises(FileExistsError, match="overwrite"):
        study.run(path, output, source_block_elements=1)


def test_run_does_not_publish_changed_reopened_json(
    canonical_rayleigh_hdf5, tmp_path, monkeypatch
):
    path, _, _, _ = canonical_rayleigh_hdf5
    output = tmp_path / "changed-readback.json"
    original_loads = study.json.loads

    def changed_loads(value, *args, **kwargs):
        payload = original_loads(value, *args, **kwargs)
        if isinstance(payload, dict) and payload.get("status") == "complete":
            payload["targets_m"][0][0] += 1.0
        return payload

    monkeypatch.setattr(study.json, "loads", changed_loads)
    with pytest.raises(RuntimeError, match="serialized result differs"):
        study.run(path, output, source_block_elements=1)
    partial = Path(f"{output}.partial")
    assert partial.exists()
    assert not output.exists()
    retained = original_loads(partial.read_text())
    assert retained["status"] == "incomplete"
    assert retained["failure"]["type"] == "RuntimeError"


def test_run_retains_incomplete_partial_when_evaluation_fails(
    canonical_rayleigh_hdf5, tmp_path, monkeypatch
):
    path, _, _, _ = canonical_rayleigh_hdf5
    output = tmp_path / "failed.json"

    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic evaluation failure")

    monkeypatch.setattr(study, "evaluate_saved_rayleigh", fail)
    with pytest.raises(RuntimeError, match="synthetic evaluation failure"):
        study.run(path, output, argv=["rayleigh", "failure"])
    partial = Path(f"{output}.partial")
    assert not output.exists()
    retained = json.loads(partial.read_text())
    assert retained["status"] == "incomplete"
    assert retained["exit"] == 1
    assert retained["failure"]["type"] == "RuntimeError"
    assert retained["failure"]["message"] == "synthetic evaluation failure"
    assert (
        retained["input_file"]["sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert retained["source_files"]


def test_run_does_not_mark_an_over_budget_result_complete(
    canonical_rayleigh_hdf5, tmp_path, monkeypatch
):
    path, _, _, _ = canonical_rayleigh_hdf5
    output = tmp_path / "over-budget.json"
    monkeypatch.setattr(
        study, "_rss_bytes", lambda: study.DEFAULT_MEMORY_BUDGET_BYTES + 1
    )
    with pytest.raises(MemoryError, match="actual peak RSS exceeded"):
        study.run(path, output, source_block_elements=1)
    retained = json.loads(Path(f"{output}.partial").read_text())
    assert retained["status"] == "incomplete"
    assert retained["exit"] == 1
    assert retained["failure"]["type"] == "MemoryError"


def test_run_refuses_an_existing_partial_without_changing_it(
    canonical_rayleigh_hdf5, tmp_path
):
    path, _, _, _ = canonical_rayleigh_hdf5
    output = tmp_path / "occupied.json"
    partial = Path(f"{output}.partial")
    partial.write_text("prior incomplete evidence\n")
    with pytest.raises(FileExistsError, match="overwrite"):
        study.run(path, output)
    assert partial.read_text() == "prior incomplete evidence\n"
