#!/usr/bin/env python3
"""Bounded M3-A timing/memory diagnostic, with no performance thresholds.

Run in the verified private editable environment:
    python scripts/research_m3a_rigid_motion.py --output /path/to/evidence.json

The private helper wrappers time actual calls with perf_counter; no tracing or
public timing API is installed. Timings are diagnostic and include wrapper and
clock overhead. Newton tile time includes its required provider replay checks.
RSS is sampled separately from the planner's library-owned byte arithmetic.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import tracemalloc
from pathlib import Path

import numpy as np

import gragra
import gragra.integrators.rigid_motion as core
from gragra.gravity_sources import PointMassCloud
from gragra.gravity_sources.trajectory import PrescribedRigidTransform
from gragra.observables import stream_rigid_motion_observable

ROOT = Path(__file__).resolve().parents[1]
PHASES = ("_snapshot_inputs", "_scan_source", "_preflight_collisions", "_compute_tile")


def _json_scalar_tree(value):
    """Serialize arithmetic/plan diagnostics without dumping scientific arrays."""
    if dataclasses.is_dataclass(value):
        return {
            field.name: _json_scalar_tree(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _json_scalar_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_scalar_tree(item) for item in value]
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "nbytes": value.nbytes,
        }
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, float, int, bool)):
        return value
    return {"type": type(value).__qualname__}


class PhaseTimers:
    """Scoped controlled wrappers at the stable private stage boundaries."""

    def __init__(self):
        self.originals = {}
        self.samples = {name: [] for name in PHASES}
        self.arithmetic = {"_working_bytes": [], "_plan_tiles": []}
        self.current_gfree_tile_bytes = []

    def __enter__(self):
        for name in (*PHASES, *self.arithmetic):
            original = getattr(core, name)
            self.originals[name] = original
            setattr(core, name, self._wrap(name, original))
        return self

    def _wrap(self, name, original):
        @functools.wraps(original)
        def timed(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = original(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                if name in self.samples:
                    self.samples[name].append(elapsed)
            if name in self.arithmetic:
                self.arithmetic[name].append(_json_scalar_tree(result))
            if name == "_compute_tile":
                payload = result if isinstance(result, np.ndarray) else result.values
                self.current_gfree_tile_bytes.append(payload.nbytes)
            return result

        return timed

    def __exit__(self, *_exception):
        for name, original in self.originals.items():
            setattr(core, name, original)


def _rss_bytes():
    """Linux resident pages, sampled baseline and peak in the same process."""
    pages = int(Path("/proc/self/statm").read_text().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE")


class RssSampler:
    def __enter__(self):
        self.baseline = _rss_bytes()
        self.peak = self.baseline
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        while not self.stop.wait(0.002):
            self.peak = max(self.peak, _rss_bytes())

    def __exit__(self, *_exception):
        self.peak = max(self.peak, _rss_bytes())
        self.stop.set()
        self.thread.join()


def fixture():
    """Frozen noncolliding F=16,T=8,S=4096, binary64, NumPy-only fixture."""
    rng = np.random.default_rng(20261017)
    source = PointMassCloud(
        rng.uniform(0.5, 1.5, 4096), rng.uniform(-1.0, 1.0, (4096, 3))
    )
    times = np.arange(16, dtype=np.float64) / 16
    translations = np.column_stack((times, times / 4, -times / 8))
    transform = PrescribedRigidTransform(
        times, translations, np.repeat(np.eye(3)[None], 16, axis=0)
    )
    targets = rng.uniform(4.0, 6.0, (8, 3))
    return source, transform, targets


def benchmark(softening_m):
    source, transform, targets = fixture()
    options = {
        "observable": "gradient",
        "softening_m": softening_m,
        "motion_block_size": 2,
        "target_block_size": 4,
        "source_block_size": 512,
        "kernel_chunk_size": 128,
        "memory_budget_bytes": 64 * 1024**2,
    }
    maximum_block_bytes = 0
    block_count = 0
    checksum = hashlib.sha256()
    tracemalloc.start()
    with RssSampler() as rss, PhaseTimers() as stages:
        started = time.perf_counter()
        stream = stream_rigid_motion_observable(source, transform, targets, **options)
        call_completed = time.perf_counter()
        first = next(stream)
        first_yield = time.perf_counter()
        maximum_block_bytes = max(maximum_block_bytes, first.values.nbytes)
        checksum.update(first.values.tobytes())
        block_count += 1
        del first
        for block in stream:
            maximum_block_bytes = max(maximum_block_bytes, block.values.nbytes)
            checksum.update(block.values.tobytes())
            block_count += 1
        completed = time.perf_counter()
    python_current, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    scan_samples = stages.samples["_scan_source"]
    maximum_gfree_bytes = max(stages.current_gfree_tile_bytes, default=0)
    return {
        "fixture": {
            "frames": 16,
            "targets": 8,
            "sources": 4096,
            "dtype": "float64",
            "seed": 20261017,
        },
        "options": options,
        "timing_seconds_observed": {
            "call_input_validation_snapshot_planning": call_completed - started,
            "snapshot_helper": sum(stages.samples["_snapshot_inputs"]),
            "source_baseline_first_scan": scan_samples[0] if scan_samples else None,
            "source_later_scan_total": sum(scan_samples[1:]),
            "collision_preflight": sum(stages.samples["_preflight_collisions"]),
            "newton_tiles_including_replay_checks": sum(
                stages.samples["_compute_tile"]
            ),
            "call_to_first_yield": first_yield - started,
            "completion": completed - started,
        },
        "phase_call_counts": {key: len(value) for key, value in stages.samples.items()},
        "memory_bytes": {
            "planner_arithmetic_observed": stages.arithmetic,
            "max_published_block": maximum_block_bytes,
            "maximum_current_gfree_tile_payload": maximum_gfree_bytes,
            "maximum_current_si_tile_payload": maximum_block_bytes,
            "current_gfree_plus_si_payload_arithmetic": (
                maximum_gfree_bytes + maximum_block_bytes
            ),
            "previous_generator_held_public_block_payload_bytes_inferred": 0,
            "python_allocator_current_including_runtime": python_current,
            "python_allocator_peak_including_runtime": python_peak,
            "rss_baseline": rss.baseline,
            "rss_sampled_peak": rss.peak,
            "rss_baseline_adjusted_peak": max(0, rss.peak - rss.baseline),
            "caller_input_arrays_excluded_from_owned_limit": sum(
                array.nbytes
                for array in (
                    source.positions_m,
                    source.masses_kg,
                    transform.time_s,
                    transform.translations_m,
                    transform.rotation_matrices,
                    targets,
                )
            ),
        },
        "result": {"block_count": block_count, "sha256": checksum.hexdigest()},
        "generator_retention_contract": {
            "previous_public_block_count_inferred": 0,
            "runtime_measured": False,
            "basis": (
                "API and executor release the previous block "
                "before requesting the next tile"
            ),
        },
        "interpretation": {
            "timing_thresholds": None,
            "rss_is_owned_buffer_bound": False,
            "rss_sampling_interval_seconds": 0.002,
            "previous_public_block_count_source": (
                "code-derived inference, not a runtime observation"
            ),
            "current_gfree_plus_si_payload_is_arithmetic_not_rss": True,
            "stages_are_inclusive_and_must_not_be_added": True,
            "unaccounted_by_owned_limit": [
                "caller-owned input arrays",
                "consumer-only retained blocks",
                "Python/runtime overhead",
                "unreported provider internal memory",
            ],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    origin = Path(gragra.__file__).resolve()
    core_origin = Path(core.__file__).resolve()
    if not origin.is_relative_to(ROOT / "src") or not core_origin.is_relative_to(
        ROOT / "src"
    ):
        raise RuntimeError(
            f"wrong editable import: gragra={origin}, core={core_origin}"
        )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "classification": (
            "Observed only when this script is executed; no imputed timings"
        ),
        "provenance": {
            "python": sys.executable,
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "gragra_origin": str(origin),
            "core_origin": str(core_origin),
            "revision": revision,
        },
        "runs": [benchmark(0.0), benchmark(0.125)],
    }
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
