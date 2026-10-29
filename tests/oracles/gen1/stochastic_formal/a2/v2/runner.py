"""Non-spawning A2 controllers that request only root-reserved child roles."""

from __future__ import annotations

import contextlib
import io
import json
import importlib
import os
import select
import socket
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
worker = importlib.import_module("_gragra_a2_v2_runtime.worker")


CONTROLLERS = {
    "accounting": (
        "accounting_test_controller",
        ("accounting_timer_fixture", "accounting_pdeath_refusal", "accounting_handshake_leaf", "accounting_cleanup_leaf"),
    ),
    "metadata": ("metadata_controller", ("metadata_worker", "metadata_helper")),
    "timing": ("timing_controller", ("timing_worker", "timing_helper")),
}


class RunnerError(RuntimeError):
    """A controller request or root-broker reply contradicts its frozen role."""


def _read_reply(sock: socket.socket, token: str, deadline_ns: int) -> dict[str, Any]:
    data = bytearray()
    while time.monotonic_ns() < deadline_ns:
        ready, _, _ = select.select([sock], [], [], 0.25)
        if not ready:
            continue
        chunk = sock.recv(65536)
        if not chunk:
            raise RunnerError("root closed the role broker before ROLE_RESULT")
        data.extend(chunk)
        if len(data) > worker.MAX_REQUEST_BYTES:
            raise RunnerError("root broker reply exceeds its fixed size bound")
        newline = data.find(b"\n")
        if newline >= 0:
            if data[newline + 1 :]:
                raise RunnerError("root sent multiple role replies for one request")
            try:
                value = json.loads(data[:newline].decode("utf-8"), object_pairs_hook=worker._pairs, parse_constant=worker._reject_constant)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RunnerError("root ROLE_RESULT is malformed JSON") from exc
            if not isinstance(value, dict):
                raise RunnerError("root ROLE_RESULT must be an object")
            return value
    raise RunnerError("root role-result wait exceeded controller bound")


def _broker(
    sock: socket.socket, request: dict[str, Any], token: str, roles: tuple[str, ...]
) -> tuple[list[dict[str, Any]], float]:
    start_ns = time.monotonic_ns()
    outcomes: dict[str, dict[str, Any]] = {}
    for index, role in enumerate(roles):
        request_id = f"{token[:16]}-{index:02d}"
        message = {
            "type": "REQUEST_ROLE",
            "launch_token": token,
            "phase_id": request["phase_id"],
            "role": role,
            "request_id": request_id,
        }
        raw = worker._canonical(message)
        sock.sendall(raw)
        reply = _read_reply(sock, token, time.monotonic_ns() + 210_000_000_000)
        expected_fields = {
            "type",
            "launch_token",
            "phase_id",
            "role",
            "request_id",
            "status",
            "terminal_pin",
            "result_pin",
            "result_status",
        }
        if (
            set(reply) != expected_fields
            or reply.get("type") != "ROLE_RESULT"
            or reply.get("launch_token") != token
            or reply.get("phase_id") != request["phase_id"]
            or reply.get("role") != role
            or reply.get("request_id") != request_id
            or reply.get("status") not in {"success", "failed", "refused", "not_started"}
        ):
            raise RunnerError(f"root ROLE_RESULT identity/status differs for {role}")
        terminal_pin = reply.get("terminal_pin")
        result_pin = reply.get("result_pin")
        if reply["status"] == "not_started":
            if terminal_pin is not None or result_pin is not None:
                raise RunnerError("not_started role unexpectedly has child evidence pins")
        elif (
            not isinstance(terminal_pin, dict)
            or set(terminal_pin) != {"path", "sha256"}
            or not isinstance(result_pin, dict)
            or set(result_pin) != {"path", "sha256"}
        ):
            raise RunnerError(f"reaped child lacks terminal/result pins: {role}")
        outcomes[role] = {
            "role": role,
            "status": "success" if reply["status"] == "success" else "not_started" if reply["status"] == "not_started" else "failure",
            "reaped": True,
            "terminal_pin": terminal_pin,
            "result_pin": result_pin,
            "result_status": reply.get("result_status"),
        }
    elapsed = max(1, time.monotonic_ns() - start_ns) / 1_000_000_000
    return [outcomes.get(role, {"role": role, "status": "not_started", "reaped": True}) for role in roles], elapsed


def _run_accounting_tests() -> dict[str, Any]:
    # Pytest collection/execution stays in this already-reserved controller.
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    import pytest

    stdout_capture, stderr_capture = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
        exit_code = int(
            pytest.main(
                [
                    "-q",
                    "--noconftest",
                    "-p",
                    "no:cacheprovider",
                    "tests/test_diagnostic_runtime_v1.py",
                    "tests/test_stochastic_a2_v2_diagnostic.py",
                ],
                plugins=[],
            )
        )
    return {
        "exit_code": exit_code,
        "stdout_tail": stdout_capture.getvalue()[-8192:],
        "stderr_tail": stderr_capture.getvalue()[-8192:],
        "in_process": True,
        "subprocesses_allowed": False,
    }


def main() -> int:
    request: dict[str, Any] | None = None
    try:
        request, _manifest, _catalog, token = worker._load_context()
        controller, roles = CONTROLLERS.get(request["phase_id"], (None, ()))
        if controller is None or request["role"] != controller:
            raise RunnerError("runner entry was assigned to a noncontroller A2 role")
        fd_text = os.environ.get("GRAGRA_DIAGNOSTIC_CONTROL_FD")
        if not isinstance(fd_text, str) or not fd_text.isdecimal():
            raise RunnerError("root control socket descriptor is missing")
        sock = socket.socket(fileno=int(fd_text))
        sock.setblocking(True)
        controller_start_ns = time.monotonic_ns()
        test_report = _run_accounting_tests() if request["phase_id"] == "accounting" else None
        try:
            child_results, broker_wait = _broker(sock, request, token, roles)
        finally:
            sock.close()
        mandatory = set(roles)
        if request["phase_id"] == "metadata":
            mandatory = {"metadata_worker"}
        elif request["phase_id"] == "timing":
            mandatory = {"timing_worker"}
        children_ok = all(
            child["status"] == "success"
            for child in child_results
            if child["role"] in mandatory
        )
        tests_ok = test_report is None or test_report["exit_code"] == 0
        controller_status = "success" if children_ok and tests_ok else "failure"
        wait_seconds = max(1, time.monotonic_ns() - controller_start_ns) / 1_000_000_000
        controller_terminal = {
            "role": controller,
            "status": controller_status,
            "waited_for_roles": list(roles),
            "wait_seconds": wait_seconds,
            "broker_wait_seconds": broker_wait,
            "children": child_results,
            "accounting_tests": test_report,
        }
        result = worker._base_result(
            request,
            "PASS" if controller_status == "success" else "FAIL",
            controller_terminal=controller_terminal,
            controller_wait_seconds=wait_seconds,
        )
        worker._write_result(request, result)
        return 0 if controller_status == "success" else 1
    except BaseException as exc:
        if request is None:
            raise
        failure = worker._base_result(
            request,
            "FAIL",
            error_type=type(exc).__name__,
            error_text=str(exc)[:4096],
            controller_terminal=None,
        )
        worker._write_result(request, failure)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
