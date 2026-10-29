"""Single-request, non-spawning leaves for the A2 diagnostic epoch."""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import json
import math
import os
import re
import signal
import socket
import statistics
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any


_RUNTIME_PACKAGE = "_gragra_a2_v2_runtime"
_RUNTIME_ROOT = Path(__file__).resolve(strict=True).parent
_package = sys.modules.get(_RUNTIME_PACKAGE)
if _package is None:
    _package = ModuleType(_RUNTIME_PACKAGE)
    _package.__path__ = [str(_RUNTIME_ROOT)]
    _package.__package__ = _RUNTIME_PACKAGE
    sys.modules[_RUNTIME_PACKAGE] = _package
elif getattr(_package, "__path__", None) != [str(_RUNTIME_ROOT)]:
    raise RuntimeError("A2 runtime package identity collides with another import root")
binding = importlib.import_module(f"{_RUNTIME_PACKAGE}.binding")
coverage = importlib.import_module(f"{_RUNTIME_PACKAGE}.coverage")
manifest_module = importlib.import_module(f"{_RUNTIME_PACKAGE}.manifest")


REQUEST_SCHEMA = "gragra.a2.diagnostic.worker-request.v1"
RESULT_SCHEMA = "gragra.a2.diagnostic.worker-result.v1"
REQUEST_FIELDS = {
    "schema",
    "model",
    "phase_id",
    "role",
    "manifest_path",
    "manifest_sha256",
    "output_root",
    "result_path",
}
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
MAX_REQUEST_BYTES = 1_048_576


class WorkerError(RuntimeError):
    """A frozen A2 worker request or diagnostic operation is invalid."""


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WorkerError(f"duplicate request field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise WorkerError(f"nonfinite JSON constant: {value}")


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as exc:
        raise WorkerError("worker result is not finite canonical JSON") from exc


def _request_line() -> dict[str, Any]:
    raw = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES or not raw.endswith(b"\n"):
        raise WorkerError("worker requires one bounded newline-terminated request")
    if sys.stdin.buffer.read(1):
        raise WorkerError("worker received more than one request line")
    try:
        value = json.loads(raw[:-1].decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerError("worker request JSON is malformed") from exc
    if not isinstance(value, dict) or set(value) != REQUEST_FIELDS:
        raise WorkerError("worker request fields differ from the fixed A2 schema")
    if (
        value.get("schema") != REQUEST_SCHEMA
        or value.get("model") != "A2"
        or not isinstance(value.get("phase_id"), str)
        or not isinstance(value.get("role"), str)
        or any(not isinstance(value.get(name), str) for name in ("manifest_path", "output_root", "result_path"))
        or not isinstance(value.get("manifest_sha256"), str)
        or not HEX64.fullmatch(value["manifest_sha256"])
    ):
        raise WorkerError("worker request identity or pin is malformed")
    for name in ("manifest_path", "output_root", "result_path"):
        if not Path(value[name]).is_absolute():
            raise WorkerError(f"worker request {name} must be absolute")
    return value


def _load_context() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    request = _request_line()
    model = os.environ.get("GRAGRA_DIAGNOSTIC_MODEL")
    phase = os.environ.get("GRAGRA_DIAGNOSTIC_PHASE")
    ledger_id = os.environ.get("GRAGRA_DIAGNOSTIC_LEDGER_ID")
    token = os.environ.get("GRAGRA_DIAGNOSTIC_LAUNCH_TOKEN")
    if (
        model != "A2"
        or phase != request["phase_id"]
        or not isinstance(ledger_id, str)
        or not ledger_id.startswith("A2:")
        or not isinstance(token, str)
        or not re.fullmatch(r"[0-9a-f]{64}", token)
    ):
        raise WorkerError("worker inherited launch context differs from its frozen request")
    manifest_raw = binding.read_pinned_bytes(
        request["manifest_path"], request["manifest_sha256"], label="worker manifest"
    )
    manifest = binding.strict_json(manifest_raw)
    if not isinstance(manifest, dict):
        raise WorkerError("worker manifest is not an object")
    if (
        manifest.get("schema") != "gragra-diagnostic-manifest-v1"
        or manifest.get("model") != "A2"
        or manifest.get("diagnostic_only") is not True
        or type(manifest.get("science_credit")) is not int
        or manifest.get("science_credit") != 0
        or type(manifest.get("old_credit_seconds")) is not int
        or manifest.get("old_credit_seconds") != 0
    ):
        raise WorkerError("worker manifest is not the fixed zero-credit A2 diagnostic")
    paths = manifest.get("paths")
    if not isinstance(paths, dict):
        raise WorkerError("worker manifest paths are missing")
    output_root = Path(request["output_root"]).resolve(strict=True)
    manifest_output = Path(paths.get("output_root", "")).resolve(strict=True)
    if output_root != manifest_output:
        raise WorkerError("worker output root differs from the immutable manifest")
    result_path = Path(request["result_path"])
    try:
        result_path.resolve(strict=False).relative_to(output_root)
    except ValueError as exc:
        raise WorkerError("worker result path escapes the external output root") from exc
    if ledger_id != f"A2:{manifest.get('epoch_id')}":
        raise WorkerError("worker ledger identity differs from the immutable manifest epoch")
    manifest_module.validate_retained_reference_object(manifest)
    return request, manifest, binding.load_source_catalog(manifest), token


def _write_result(request: dict[str, Any], result: dict[str, Any]) -> None:
    raw = _canonical(result)
    path = Path(request["result_path"])
    if path.exists() or path.is_symlink():
        raise WorkerError("worker result path already exists")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(raw)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                raise WorkerError("worker result short write")
            view = view[count:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    view = memoryview(raw)
    while view:
        count = os.write(1, view)
        if count <= 0:
            raise WorkerError("worker stdout short write")
        view = view[count:]


def _base_result(request: dict[str, Any], status: str, **fields: Any) -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "model": "A2",
        "phase_id": request["phase_id"],
        "role": request["role"],
        "status": status,
        "diagnostic_only": True,
        "science_credit": 0,
        "old_credit_seconds": 0,
        "formal_comparison": "REJECTED",
        **fields,
    }


def _source_evidence(manifest: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any]:
    root = Path(manifest["paths"]["worktree_root"]).resolve(strict=True)
    return binding.verify_source_catalog(root, catalog)


def _dependency_imports(manifest: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any]:
    """Import only the pinned environment then the named A2 public APIs."""
    binding.reject_production_modules()
    source_before = _source_evidence(manifest, catalog)
    environment = binding.environment_report(manifest)
    analytic = importlib.import_module("gragra.analytic.stochastic_acoustic")
    observable = importlib.import_module("gragra.observables.stochastic_acoustic")
    constants = importlib.import_module("gragra.constants")
    dependencies = manifest["dependency_inventory"]
    expected_origins = {
        pin["module"]: {"path": pin["path"], "sha256": pin["sha256"]}
        for pin in dependencies
    }
    expected_versions = {pin["module"]: pin["version"] for pin in dependencies}
    import_state = binding.verify_import_state(
        expected_origins,
        expected_versions=expected_versions,
        six_pin=binding.SIX_IMPORTER_PIN,
        manifest=manifest,
    )
    loaded_candidate = binding.verify_loaded_candidate_origins(
        manifest["paths"]["worktree_root"], catalog
    )
    source_after = _source_evidence(manifest, catalog)
    if source_before["catalog_sha256"] != source_after["catalog_sha256"]:
        raise WorkerError("candidate source catalog changed during controlled imports")
    return {
        "environment": environment,
        "import_state": import_state,
        "loaded_candidate": loaded_candidate,
        "source_catalog": source_after,
        "analytic_module": analytic,
        "observable_module": observable,
        "constants_module": constants,
    }


class _TimerValue(ctypes.Union):
    _fields_ = [("sival_int", ctypes.c_int), ("sival_ptr", ctypes.c_void_p)]


class _TimerEvent(ctypes.Structure):
    _fields_ = [
        ("sigev_value", _TimerValue),
        ("sigev_signo", ctypes.c_int),
        ("sigev_notify", ctypes.c_int),
        ("pad", ctypes.c_int * 12),
    ]


class _Timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


class _Itimerspec(ctypes.Structure):
    _fields_ = [("it_interval", _Timespec), ("it_value", _Timespec)]


def _install_child_sigkill_timer() -> None:
    if os.name != "posix" or os.uname().sysname != "Linux" or os.uname().machine != "x86_64":
        raise WorkerError("contained timer fixture requires reviewed Linux x86_64")
    if ctypes.sizeof(ctypes.c_void_p) != 8 or ctypes.sizeof(ctypes.c_long) != 8:
        raise WorkerError("contained timer fixture requires LP64")
    timer_layout = {
        "sigval_bytes": ctypes.sizeof(_TimerValue),
        "sigevent_bytes": ctypes.sizeof(_TimerEvent),
        "sigevent_offsets": {
            "value": _TimerEvent.sigev_value.offset,
            "signo": _TimerEvent.sigev_signo.offset,
            "notify": _TimerEvent.sigev_notify.offset,
            "pad": _TimerEvent.pad.offset,
        },
        "timespec_bytes": ctypes.sizeof(_Timespec),
        "itimerspec_bytes": ctypes.sizeof(_Itimerspec),
        "itimerspec_value_offset": _Itimerspec.it_value.offset,
    }
    if timer_layout != {
        "sigval_bytes": 8,
        "sigevent_bytes": 64,
        "sigevent_offsets": {"value": 0, "signo": 8, "notify": 12, "pad": 16},
        "timespec_bytes": 16,
        "itimerspec_bytes": 32,
        "itimerspec_value_offset": 16,
    }:
        raise WorkerError("contained timer fixture ABI layout differs from reviewed glibc x86_64")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.timer_create.restype = ctypes.c_int
    libc.timer_create.argtypes = [ctypes.c_int, ctypes.POINTER(_TimerEvent), ctypes.POINTER(ctypes.c_void_p)]
    libc.timer_settime.restype = ctypes.c_int
    libc.timer_settime.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(_Itimerspec), ctypes.POINTER(_Itimerspec)]
    event = _TimerEvent()
    event.sigev_value.sival_int = 0
    event.sigev_signo = 9
    event.sigev_notify = 0
    timer = ctypes.c_void_p()
    ctypes.set_errno(0)
    if libc.timer_create(1, ctypes.byref(event), ctypes.byref(timer)) != 0:
        raise WorkerError(f"child timer_create failed with errno {ctypes.get_errno()}")
    spec = _Itimerspec(_Timespec(0, 0), _Timespec(2, 0))
    ctypes.set_errno(0)
    if libc.timer_settime(timer, 0, ctypes.byref(spec), None) != 0:
        raise WorkerError(f"child timer_settime failed with errno {ctypes.get_errno()}")
    while True:
        signal.pause()


def _attempt_pdeath_clear() -> dict[str, int]:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.restype = ctypes.c_int
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    ctypes.set_errno(0)
    result = libc.prctl(1, 0, 0, 0, 0)
    error = ctypes.get_errno()
    if result != -1 or error != 1:
        raise WorkerError("seccomp did not refuse clearing PDEATHSIG with EPERM")
    return {"prctl_result": result, "errno": error, "expected_errno": 1}


def _process_identity() -> dict[str, Any]:
    stat_raw = Path("/proc/self/stat").read_bytes()
    prefix, separator, tail = stat_raw.rpartition(b")")
    fields = tail.strip().split()
    if not separator or not prefix.split(maxsplit=1)[0].isdigit() or len(fields) < 20:
        raise WorkerError("worker /proc stat identity is malformed")
    argv = Path("/proc/self/cmdline").read_bytes().split(b"\0")
    if argv and argv[-1] == b"":
        argv.pop()
    return {
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "starttime_ticks": int(fields[19]),
        "argv": [item.decode("utf-8", "strict") for item in argv],
    }


def _capture_retained(manifest: dict[str, Any]) -> dict[str, Any]:
    binding.reject_production_modules()
    retained_module = importlib.import_module(f"{_RUNTIME_PACKAGE}.retained_reference")

    result = retained_module.inspect_retained(manifest)
    binding.reject_production_modules()
    return result


def _map_and_pilot(manifest: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    root = Path(manifest["paths"]["worktree_root"]).resolve(strict=True)
    fixture_path = root / "docs/design/gen1/stochastic/A2-fixtures-v1.json"
    fixture_raw = fixture_path.read_bytes()
    fixture_map = coverage.build_map(fixture_raw)
    pins = manifest.get("external_pins")
    if not isinstance(pins, dict):
        raise WorkerError("external model artifact pins are missing")
    map_pin = pins.get("coverage_map")
    pilot_pin = pins.get("pilot_selection")
    if not isinstance(map_pin, dict) or not isinstance(pilot_pin, dict):
        raise WorkerError("frozen coverage-map/pilot-selection pins are missing")
    map_raw = binding.read_pinned_bytes(map_pin.get("path", ""), map_pin.get("sha256", ""), label="coverage map")
    map_evidence = coverage.verify_map_artifact(fixture_map, map_raw, map_pin["sha256"])
    pilot_raw = binding.read_pinned_bytes(
        pilot_pin.get("path", ""), pilot_pin.get("sha256", ""), label="pilot selection"
    )
    pilot, pilot_adapter = coverage.adapt_frozen_pilot_selection(
        pilot_raw,
        fixture_map,
        coverage_map_sha256=map_evidence["sha256"],
        expected_sha256=pilot_pin["sha256"],
    )
    return fixture_map, pilot, {
        "coverage_map": map_evidence,
        "coverage_map_path": map_pin["path"],
        "pilot_selection_path": pilot_pin["path"],
        "pilot_selection_sha256": pilot_pin["sha256"],
        "pilot_adapter": pilot_adapter,
    }


def _write_timing_record(descriptor: int, event: dict[str, Any]) -> None:
    raw = _canonical(event)
    view = memoryview(raw)
    while view:
        count = os.write(descriptor, view)
        if count <= 0:
            raise WorkerError("timing JSONL short write")
        view = view[count:]
    os.fsync(descriptor)


def _timing(manifest: dict[str, Any], catalog: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    fixture_map, pilot, input_evidence = _map_and_pilot(manifest)
    imported = _dependency_imports(manifest, catalog)
    analytic = imported["analytic_module"]
    observable = imported["observable_module"]
    constants = imported["constants_module"]
    g_si = float(getattr(constants, "G_SI"))
    output_path = Path(request["output_root"]) / "phases" / "timing" / "timing-calls.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(
        output_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    calls: list[dict[str, Any]] = []
    pid, start_ticks = os.getpid(), _process_identity()["starttime_ticks"]
    try:
        for entry in pilot:
            key = entry["native_key"]
            for iteration in entry["iterations"]:
                identity = {
                    "canonical_key_json": entry["canonical_key_json"],
                    "fixture_id": key["fixture_id"],
                    "record_kind": entry["record_kind"],
                    "representation": key["representation"],
                    "input_convention": (
                        "nominal_exact_x_scalar"
                        if entry["record_kind"] == "nominal_scalar"
                        else "frozen_physical_tuple"
                    ),
                    "iteration": iteration,
                    "pid": pid,
                    "starttime_ticks": start_ticks,
                    "same_process": True,
                }
                start_record = {
                    "event": "call_start",
                    **identity,
                    "recorded_monotonic_ns": time.monotonic_ns(),
                }
                _write_timing_record(descriptor, start_record)
                call_start = time.monotonic_ns()
                try:
                    if entry["record_kind"] == "nominal_scalar":
                        decimal_inputs = {
                            "frequency_hz": key["frequency_hz"],
                            "nominal_exact_x": entry["nominal_exact_x"],
                        }
                        value = float(entry["nominal_exact_x"])
                        output = analytic.isotropic_acoustic_depth_factor(value)
                        api = "gragra.analytic.stochastic_acoustic.isotropic_acoustic_depth_factor"
                        input_binary64 = {"x": value.hex()}
                        actual_x = value
                    else:
                        decimal_inputs = dict(entry["inputs"])
                        physical = {name: float(value) for name, value in decimal_inputs.items()}
                        if physical["g_si"].hex() != g_si.hex():
                            raise WorkerError("frozen g_si differs from the pinned production constant")
                        if key["representation"] == "Gfree":
                            function = analytic.underground_isotropic_acoustic_acceleration_csd_g_free
                            api = "gragra.analytic.stochastic_acoustic.underground_isotropic_acoustic_acceleration_csd_g_free"
                        elif key["representation"] == "SI":
                            function = observable.underground_isotropic_acoustic_acceleration_csd
                            api = "gragra.observables.stochastic_acoustic.underground_isotropic_acoustic_acceleration_csd"
                        else:
                            raise WorkerError("pilot contains an unreviewed physical representation")
                        output = function(
                            depth_m=physical["depth_m"],
                            sound_speed_m_s=physical["sound_speed_m_s"],
                            frequency_hz=physical["frequency_hz"],
                            surface_pressure_psd_pa2_hz=physical["pressure_psd_pa2_hz"],
                        )
                        input_binary64 = {name: value.hex() for name, value in physical.items()}
                        actual_x = 4.0 * math.pi * physical["frequency_hz"] * physical["depth_m"] / physical["sound_speed_m_s"]
                    ended = time.monotonic_ns()
                    import numpy as np

                    array = np.asarray(output, dtype=np.complex128)
                    if entry["record_kind"] == "nominal_scalar":
                        if array.ndim != 0 or not np.isfinite(array).all():
                            raise WorkerError("scalar production API returned nonfinite/non-scalar output")
                        shape = []
                        scalar = complex(array.item())
                        values = [[scalar.real.hex(), scalar.imag.hex()]]
                    else:
                        if array.shape != (3, 3) or not np.isfinite(array).all():
                            raise WorkerError("physical production API returned a nonfinite/non-3x3 tensor")
                        shape = list(array.shape)
                        values = [
                            [complex(item).real.hex(), complex(item).imag.hex()]
                            for item in array.reshape(-1)
                        ]
                    completion = {
                        "event": "call_complete",
                        **identity,
                        "api": api,
                        "decimal_inputs": decimal_inputs,
                        "binary64_inputs_hex": input_binary64,
                        "actual_physical_x_binary64_hex": actual_x.hex(),
                        "elapsed_ns": ended - call_start,
                        "ended_monotonic_ns": ended,
                        "output_shape": shape,
                        "output_complex128_hex_layout": "flat_c_order_[real_hex,imag_hex]_pairs",
                        "output_complex128_hex": values,
                        "output_sha256": hashlib.sha256(_canonical(values)).hexdigest(),
                    }
                    _write_timing_record(descriptor, completion)
                    calls.append(completion)
                except BaseException as exc:
                    ended = time.monotonic_ns()
                    _write_timing_record(
                        descriptor,
                        {
                            "event": "call_error",
                            **identity,
                            "api": locals().get("api"),
                            "elapsed_ns": ended - call_start,
                            "ended_monotonic_ns": ended,
                            "exception_type": type(exc).__name__,
                            "exception_text": str(exc)[:2048],
                        },
                    )
                    raise
    finally:
        os.fsync(descriptor)
        os.close(descriptor)
    if len(calls) != 48:
        raise WorkerError(f"timing worker completed {len(calls)} calls instead of 48")
    elapsed_seconds = [call["elapsed_ns"] / 1_000_000_000 for call in calls]
    strata: dict[str, list[float]] = {"initial": [], "warm": []}
    by_key: dict[str, list[float]] = {}
    by_kind: dict[str, list[float]] = {"nominal_scalar": [], "physical_csd": []}
    for call in calls:
        seconds = call["elapsed_ns"] / 1_000_000_000
        strata["initial" if call["iteration"] == 0 else "warm"].append(seconds)
        by_key.setdefault(call["canonical_key_json"], []).append(seconds)
        by_kind[call["record_kind"]].append(seconds)

    def summarize(values: list[float]) -> dict[str, float | int]:
        if not values:
            return {"count": 0}
        return {
            "count": len(values),
            "mean_seconds": statistics.fmean(values),
            "median_seconds": statistics.median(values),
            "population_stdev_seconds": statistics.pstdev(values),
            "max_seconds": max(values),
        }

    return {
        "scope": "fixed_12_key_48_call_diagnostic_timing",
        "coverage_map_sha256": input_evidence["coverage_map"]["sha256"],
        "pilot_selection_sha256": input_evidence["pilot_selection_sha256"],
        "record_file": str(output_path),
        "record_file_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "key_count": len(pilot),
        "completed_call_count": len(calls),
        "completed_call_summary": {
            "all": summarize(elapsed_seconds),
            "iteration": {name: summarize(values) for name, values in strata.items()},
            "record_kind": {name: summarize(values) for name, values in by_kind.items()},
            "native_key": {name: summarize(values) for name, values in sorted(by_key.items())},
        },
        "same_process_pid": pid,
        "same_process_starttime_ticks": start_ticks,
        "call_elapsed_seconds": elapsed_seconds,
        "source_module_origins": imported["loaded_candidate"],
        "summary_scope": "completed_calls_only; 12-key sample does not predict full 84-key runtime",
        "science_credit": 0,
        "formal_comparison": "REJECTED",
    }


def execute_leaf(
    request: dict[str, Any], manifest: dict[str, Any], catalog: dict[str, Any]
) -> dict[str, Any]:
    phase, role = request["phase_id"], request["role"]
    if (phase, role) == ("environment", "environment_leaf"):
        report = binding.environment_report(manifest)
        return _base_result(request, "PASS", environment=report)
    if (phase, role) == ("accounting", "accounting_timer_fixture"):
        _install_child_sigkill_timer()
        raise WorkerError("contained timer unexpectedly returned")
    if (phase, role) == ("accounting", "accounting_pdeath_refusal"):
        return _base_result(request, "PASS", pdeath_clear_refusal=_attempt_pdeath_clear())
    if (phase, role) == ("accounting", "accounting_handshake_leaf"):
        identity = _process_identity()
        if identity["ppid"] <= 1 or identity["starttime_ticks"] <= 0:
            raise WorkerError("positive handshake child identity is incomplete")
        return _base_result(request, "PASS", managed_child_identity=identity)
    if (phase, role) == ("accounting", "accounting_cleanup_leaf"):
        identity = _process_identity()
        children = Path(f"/proc/{identity['pid']}/task/{identity['pid']}/children").read_text().strip()
        if children:
            raise WorkerError("no-fork leaf unexpectedly has task children")
        return _base_result(request, "PASS", managed_child_identity=identity, task_children=[])
    if (phase, role) == ("retained", "retained_leaf"):
        retained = _capture_retained(manifest)
        return _base_result(request, "PASS", retained_reference=retained)
    if (phase, role) == ("metadata", "metadata_worker"):
        imported = _dependency_imports(manifest, catalog)
        fixture_map, pilot, input_evidence = _map_and_pilot(manifest)
        functions = {
            "isotropic_acoustic_depth_factor": imported["analytic_module"].isotropic_acoustic_depth_factor,
            "underground_isotropic_acoustic_acceleration_csd_g_free": imported["analytic_module"].underground_isotropic_acoustic_acceleration_csd_g_free,
            "underground_isotropic_acoustic_acceleration_csd": imported["observable_module"].underground_isotropic_acoustic_acceleration_csd,
        }
        import inspect

        metadata = {
            name: {
                "signature": str(inspect.signature(function)),
                "module": function.__module__,
                "qualname": function.__qualname__,
            }
            for name, function in functions.items()
        }
        return _base_result(
            request,
            "PASS",
            coverage={"entry_count": len(fixture_map["entries"]), "pilot_count": len(pilot), **input_evidence},
            import_state=imported["import_state"],
            loaded_candidate=imported["loaded_candidate"],
            public_api_metadata=metadata,
            diagnostic_only=True,
        )
    if (phase, role) == ("timing", "timing_worker"):
        timing = _timing(manifest, catalog, request)
        return _base_result(request, "PASS", timing=timing)
    if (phase, role) == ("terminal", "record_leaf"):
        summaries = {}
        for prior_phase in ("environment", "accounting", "retained", "metadata", "timing"):
            path = Path(request["output_root"]) / "phases" / prior_phase / "result.json"
            value = binding.strict_json(path.read_bytes())
            if not isinstance(value, dict) or value.get("status") != "success":
                raise WorkerError(f"prior phase is not successful: {prior_phase}")
            summaries[prior_phase] = value
        return _base_result(
            request,
            "PASS",
            diagnostic_summary={
                "phase_summaries": summaries,
                "reuse_identity": "retained_reference_bytes_verified_for_diagnostic_only",
                "formal_comparison": "REJECTED",
                "science_acceptance": False,
            },
        )
    raise WorkerError(f"no fixed leaf implementation for {phase}/{role}")


def main() -> int:
    request: dict[str, Any] | None = None
    try:
        request, manifest, catalog, _token = _load_context()
        result = execute_leaf(request, manifest, catalog)
        if result.get("status") != "PASS":
            raise WorkerError("leaf operation did not produce PASS")
        _write_result(request, result)
        return 0
    except BaseException as exc:
        if request is None or request["role"] == "accounting_timer_fixture":
            raise
        failure = _base_result(
            request,
            "FAIL",
            error_type=type(exc).__name__,
            error_text=str(exc)[:4096],
        )
        _write_result(request, failure)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
