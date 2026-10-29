"""Low-level supervisor for fresh Gen1 stochastic reference attempts.

The scientific-policy adapter is intentionally not exposed yet. The frozen
common policy must provide its actual key fields and expected keys before a
formal CLI can be wired; callers cannot supply counts in place of those keys.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stochastic_attempt import (
    AttemptPendingError,
    _artifact_digests,
    _make_readonly,
    canonical_json_bytes,
    seal_attempt,
    validate_key_boundary,
)

REQUIRED_PIN_GROUPS = ("policy", "fixture", "generator", "source", "infrastructure")
REQUIRED_INFRASTRUCTURE_FILES = {
    "tests/oracles/gen1/stochastic_attempt.py",
    "tests/oracles/gen1/run_stochastic_local.py",
}
ATTEMPT_LEDGER_FILENAME = "attempt.json"
LAUNCH_STATE_FILENAME = "launch.json"
REAP_GRACE_SECONDS = 2.0
REAP_FINAL_SECONDS = 2.0
PROCESS_GROUP_POLL_SECONDS = 0.05
ATTEMPT_LEDGER_FORMAT = "gen1-stochastic-attempt-ledger-v1"
LAUNCH_STATE_FORMAT = "gen1-stochastic-launch-state-v1"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_once_json(path: Path, value: Mapping[str, Any]) -> bytes:
    """Publish canonical JSON without replacement and sync file plus directory."""
    contents = canonical_json_bytes(value) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        with suppress(FileNotFoundError):
            temporary.unlink()
        raise
    return contents


def _create_attempt_directory(root: Path) -> None:
    root.mkdir(parents=False, exist_ok=False)
    _fsync_directory(root.parent)


def _create_capture_files(root: Path) -> None:
    for name in ("stdout.log", "stderr.log"):
        with (root / name).open("xb") as stream:
            stream.flush()
            os.fsync(stream.fileno())
    _fsync_directory(root)


def _attempt_ledger(
    root: Path,
    *,
    command: Sequence[str],
    key_fields: Sequence[str],
    expected_keys: Sequence[Any],
    record_key_path: Sequence[str],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    fields, expected = validate_key_boundary(
        key_fields, expected_keys, record_key_path
    )
    expected.sort(
        key=lambda item: canonical_json_bytes({name: item[name] for name in fields})
    )
    return {
        "format": ATTEMPT_LEDGER_FORMAT,
        "attempt_id": root.name,
        "prepared_at_utc": _utc_now(),
        "launch_protocol": "stdin-gated-process-group-v1",
        "command": list(command),
        "key_boundary": {
            "key_fields": list(fields),
            "record_key_path": list(record_key_path),
            "expected_keys": expected,
        },
        "provenance": dict(provenance),
    }


def _prepare_attempt(
    root: Path,
    *,
    command: Sequence[str],
    key_fields: Sequence[str],
    expected_keys: Sequence[Any],
    record_key_path: Sequence[str],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    _create_attempt_directory(root)
    _create_capture_files(root)
    ledger = _attempt_ledger(
        root,
        command=command,
        key_fields=key_fields,
        expected_keys=expected_keys,
        record_key_path=record_key_path,
        provenance=provenance,
    )
    ledger["launcher"] = {
        "protocol": "stdin-gated-process-group-v1",
        "executable": "/bin/sh",
        "script_sha256": hashlib.sha256(
            b'IFS= read -r _gragra_attempt_gate || exit 125\n'
            b'[ "$_gragra_attempt_gate" = go ] || exit 125\n'
            b'exec "$@" <&<inherited-stdin-fd>'
        ).hexdigest(),
    }
    _write_once_json(root / ATTEMPT_LEDGER_FILENAME, ledger)
    return ledger


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source_pin(
    repository_root: str | Path,
    *,
    expected_head: str,
    expected_tree: str,
    file_hashes_by_group: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Verify a clean committed checkout and categorized file digests.

    The caller supplies paths from the frozen policy boundary. This helper does
    not parse policy schemas or decide which scientific files are relevant.
    """
    root = Path(repository_root).resolve()
    if not root.is_dir():
        raise ValueError(f"repository root does not exist: {root}")
    if set(file_hashes_by_group) != set(REQUIRED_PIN_GROUPS):
        raise ValueError(f"source pin groups must be exactly {REQUIRED_PIN_GROUPS}")

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def git_bytes(*args: str) -> bytes:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
        )
        return result.stdout

    actual_head = git("rev-parse", "HEAD")
    actual_tree = git("rev-parse", "HEAD^{tree}")
    dirty = git("status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise ValueError("formal source checkout is not clean")
    if actual_head != expected_head:
        raise ValueError("Git HEAD differs from the frozen source pin")
    if actual_tree != expected_tree:
        raise ValueError("Git tree differs from the frozen source pin")

    actual_hashes: dict[str, dict[str, str]] = {}
    for group in REQUIRED_PIN_GROUPS:
        expected_group = file_hashes_by_group[group]
        if not isinstance(expected_group, Mapping):
            raise TypeError(
                f"source pin group {group!r} must be a path-to-hash mapping"
            )
        if not expected_group:
            raise ValueError(f"source pin group {group!r} must not be empty")
        actual_hashes[group] = {}
        for relative_name, expected_digest in sorted(expected_group.items()):
            relative = Path(relative_name)
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError(
                    f"source pin path must be repository-relative: {relative_name!r}"
                )
            if (
                not isinstance(expected_digest, str)
                or len(expected_digest) != 64
                or any(char not in "0123456789abcdef" for char in expected_digest)
            ):
                raise ValueError(
                    f"source pin is not a lowercase SHA256: {relative_name!r}"
                )
            path = root / relative
            parent = root
            for part in relative.parts[:-1]:
                parent = parent / part
                if parent.is_symlink():
                    raise ValueError(
                        f"source pin path traverses a symlink: {relative_name}"
                    )
            if path.is_symlink() or not path.is_file():
                raise ValueError(
                    f"source pin file is missing or is a symlink: {relative_name}"
                )
            tree_entries = git_bytes(
                "ls-tree",
                "-z",
                "--full-tree",
                actual_head,
                "--",
                relative.as_posix(),
            ).split(b"\0")
            matching_entries: list[tuple[bytes, bytes]] = []
            requested_path = relative.as_posix().encode("utf-8")
            for entry in tree_entries:
                if not entry:
                    continue
                metadata, separator, entry_path = entry.partition(b"\t")
                if separator and entry_path == requested_path:
                    matching_entries.append((metadata, entry_path))
            if len(matching_entries) != 1:
                raise ValueError(
                    "source pin file is not a regular Git blob in the pinned tree: "
                    f"{relative_name}"
                )
            metadata, _entry_path = matching_entries[0]
            mode, object_type, object_id = metadata.split()
            if mode not in {b"100644", b"100755"} or object_type != b"blob":
                raise ValueError(
                    "source pin file is not a regular Git blob in the pinned tree: "
                    f"{relative_name}"
                )
            tree_contents = git_bytes("cat-file", "blob", object_id.decode("ascii"))
            working_contents = path.read_bytes()
            actual_digest = hashlib.sha256(working_contents).hexdigest()
            if working_contents != tree_contents:
                raise ValueError(
                    f"source pin file content differs from pinned Git blob: {relative_name}"
                )
            if actual_digest != expected_digest:
                raise ValueError(f"source pin digest mismatch: {relative_name}")
            actual_hashes[group][relative.as_posix()] = actual_digest

    return {"head": actual_head, "tree": actual_tree, "files_sha256": actual_hashes}


def verify_environment_inventory(
    inventory_path: str | Path, expected_sha256: str
) -> dict[str, str]:
    """Verify the exact externally retained environment inventory bytes."""
    candidate = Path(inventory_path)
    if candidate.is_symlink():
        raise ValueError(f"environment inventory is a symlink: {candidate}")
    path = candidate.resolve()
    if not path.is_file():
        raise ValueError(f"environment inventory is missing or is a symlink: {path}")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(char not in "0123456789abcdef" for char in expected_sha256)
    ):
        raise ValueError("environment inventory pin must be a lowercase SHA256")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError("environment inventory digest mismatch")
    return {"path": str(path), "sha256": actual}


def _process_group_exists(process_group_id: int) -> bool:
    if os.name != "posix":
        return True
    proc_root = Path("/proc")
    if proc_root.is_dir():
        try:
            for stat_path in proc_root.glob("[0-9]*/stat"):
                try:
                    fields = stat_path.read_text().rsplit(")", 1)[1].split()
                    state = fields[0]
                    group_id = int(fields[2])
                except (OSError, IndexError, ValueError):
                    continue
                if group_id == process_group_id and state not in {"Z", "X"}:
                    return True
            return False
        except OSError:
            pass
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def _signal_process_group(process_group_id: int, sig: int) -> bool:
    if os.name != "posix":
        return False
    try:
        os.killpg(process_group_id, sig)
    except ProcessLookupError:
        return False
    except OSError:
        return False
    return True


def _wait_for_process_group_exit(
    process_group_id: int,
    timeout_seconds: float,
    poll_interval_seconds: float = PROCESS_GROUP_POLL_SECONDS,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while _process_group_exists(process_group_id):
        if time.monotonic() >= deadline:
            return False
        time.sleep(min(poll_interval_seconds, max(0.0, deadline - time.monotonic())))
    return True


def _call_process_signal(process: Any, sig: int) -> None:
    method = process.terminate if sig == signal.SIGTERM else process.kill
    with suppress(BaseException):
        method()


def _reap(
    process: Any,
    *,
    terminate: bool,
    grace_seconds: float = REAP_GRACE_SECONDS,
    final_reap_seconds: float = REAP_FINAL_SECONDS,
) -> dict[str, Any]:
    """Bound every wait and confirm both leader reaping and group shutdown."""
    process_id = getattr(process, "pid", None)
    process_group_id = process_id if isinstance(process_id, int) else None
    first_signal = signal.SIGTERM if terminate else signal.SIGKILL
    signals_sent: list[str] = []
    wait_errors: list[str] = []
    leader_reaped = getattr(process, "returncode", None) is not None

    def send(sig: int) -> None:
        sent = False
        if process_group_id is not None and os.name == "posix":
            sent = _signal_process_group(process_group_id, sig)
        if sent:
            signals_sent.append(signal.Signals(sig).name)
        else:
            _call_process_signal(process, sig)
            signals_sent.append(signal.Signals(sig).name)

    if process_group_id is not None and _process_group_exists(process_group_id):
        send(first_signal)
    elif not leader_reaped:
        _call_process_signal(process, first_signal)
        signals_sent.append(signal.Signals(first_signal).name)
    try:
        returned = process.wait(timeout=grace_seconds)
        leader_reaped = isinstance(returned, int) or getattr(process, "returncode", None) is not None
    except subprocess.TimeoutExpired as exc:
        wait_errors.append(f"{type(exc).__name__}: {exc}")
    except BaseException as exc:
        wait_errors.append(f"{type(exc).__name__}: {exc}")
        leader_reaped = getattr(process, "returncode", None) is not None

    group_empty = (
        process_group_id is None or not _process_group_exists(process_group_id)
    )
    if not group_empty:
        send(signal.SIGKILL)
        try:
            returned = process.wait(timeout=final_reap_seconds)
            leader_reaped = isinstance(returned, int) or getattr(process, "returncode", None) is not None
        except subprocess.TimeoutExpired as exc:
            wait_errors.append(f"{type(exc).__name__}: {exc}")
        except BaseException as exc:
            wait_errors.append(f"{type(exc).__name__}: {exc}")
            leader_reaped = getattr(process, "returncode", None) is not None
        group_empty = _wait_for_process_group_exit(
            process_group_id, final_reap_seconds
        )
    elif not leader_reaped:
        send(signal.SIGKILL)
        try:
            returned = process.wait(timeout=final_reap_seconds)
            leader_reaped = isinstance(returned, int) or getattr(process, "returncode", None) is not None
        except subprocess.TimeoutExpired as exc:
            wait_errors.append(f"{type(exc).__name__}: {exc}")
        except BaseException as exc:
            wait_errors.append(f"{type(exc).__name__}: {exc}")
            leader_reaped = getattr(process, "returncode", None) is not None

    return {
        "status": "confirmed_stopped" if leader_reaped and group_empty else "unresolved",
        "pid": process_id,
        "process_group_id": process_group_id,
        "process_start_ticks": (
            _process_start_ticks(process_id) if process_id is not None else None
        ),
        "leader_reaped": leader_reaped,
        "process_group_empty": group_empty,
        "signals_sent": signals_sent,
        "wait_errors": wait_errors,
    }


def _cleanup_recorded_group(
    process_group_id: int,
    *,
    process_start_ticks: str | None,
    grace_seconds: float,
    final_reap_seconds: float,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    if os.name != "posix":
        return {
            "status": "unresolved",
            "process_group_id": process_group_id,
            "reason": "process-group recovery is unsupported on this platform",
        }

    group_exists = _process_group_exists(process_group_id)
    if not group_exists:
        return {
            "status": "confirmed_stopped",
            "process_group_id": process_group_id,
            "process_group_empty": True,
            "signals_sent": [],
        }

    # When the group leader still exists, refuse to signal a PID reused since
    # the durable launch record was written.
    leader_stat = Path(f"/proc/{process_group_id}/stat")
    if leader_stat.exists():
        try:
            stat_fields = leader_stat.read_text().rsplit(")", 1)[1].split()
            leader_pgid = int(stat_fields[2])
            leader_start_ticks = stat_fields[19]
        except (OSError, IndexError, ValueError):
            return {
                "status": "unresolved",
                "process_group_id": process_group_id,
                "process_group_empty": False,
                "reason": "could not verify recorded process identity",
            }
        if (
            leader_pgid != process_group_id
            or process_start_ticks is None
            or leader_start_ticks != process_start_ticks
        ):
            return {
                "status": "unresolved",
                "process_group_id": process_group_id,
                "process_group_empty": False,
                "reason": "recorded process identity no longer matches",
            }

    signals_sent: list[str] = []
    if _signal_process_group(process_group_id, signal.SIGTERM):
        signals_sent.append("SIGTERM")
    if not _wait_for_process_group_exit(
        process_group_id, grace_seconds, poll_interval_seconds
    ):
        if _signal_process_group(process_group_id, signal.SIGKILL):
            signals_sent.append("SIGKILL")
    group_empty = _wait_for_process_group_exit(
        process_group_id, final_reap_seconds, poll_interval_seconds
    )
    return {
        "status": "confirmed_stopped" if group_empty else "unresolved",
        "process_group_id": process_group_id,
        "process_group_empty": group_empty,
        "signals_sent": signals_sent,
    }


def _process_start_ticks(process_id: int) -> str | None:
    stat_path = Path(f"/proc/{process_id}/stat")
    try:
        fields = stat_path.read_text().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return None
    return fields[19] if len(fields) > 19 else None


def _gate_command(command: Sequence[str], inherited_stdin_fd: int) -> list[str]:
    script = (
        "IFS= read -r _gragra_attempt_gate || exit 125\n"
        '[ "$_gragra_attempt_gate" = go ] || exit 125\n'
        f'exec "$@" <&{inherited_stdin_fd}'
    )
    return ["/bin/sh", "-c", script, "gen1-attempt-gate", *command]


def _record_recovery_event(root: Path, event: Mapping[str, Any]) -> Path:
    sequence = 1
    while True:
        destination = root / f"recovery-{sequence:06d}.json"
        try:
            _write_once_json(destination, event)
            return destination
        except FileExistsError:
            sequence += 1


def _read_canonical_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or canonical_json_bytes(value) + b"\n" != raw:
        raise ValueError(f"attempt metadata is not canonical JSON: {path.name}")
    return value


def _fsync_attempt_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            continue
        if path.is_file():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        elif path.is_dir():
            _fsync_directory(path)


def recover_attempt(
    attempt_dir: str | Path,
    *,
    cleanup_grace_seconds: float = REAP_GRACE_SECONDS,
    final_reap_seconds: float = REAP_FINAL_SECONDS,
    poll_interval_seconds: float = PROCESS_GROUP_POLL_SECONDS,
) -> dict[str, Any]:
    """Seal an interrupted attempt after cleanup; this function never launches.

    An attempt with a durable launch record may have computed, so recovery stops
    and verifies its process group before sealing its preserved partial output.
    An attempt without a launch record cannot have passed the stdin launch gate.
    """
    timing = (cleanup_grace_seconds, final_reap_seconds, poll_interval_seconds)
    if any(not math.isfinite(value) or value < 0 for value in timing):
        raise ValueError("recovery intervals must be finite and nonnegative")
    if poll_interval_seconds == 0:
        raise ValueError("poll_interval_seconds must be positive")
    root = Path(attempt_dir).resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("attempt directory must already exist and be a real directory")
    seal_path = root / "seal.json"
    if seal_path.exists() or seal_path.is_symlink():
        manifest = _read_canonical_json(seal_path)
        if manifest.get("artifacts_sha256") != _artifact_digests(root):
            raise ValueError("existing seal artifact inventory does not match attempt")
        _make_readonly(root)
        return manifest

    ledger = _read_canonical_json(root / ATTEMPT_LEDGER_FILENAME)
    if (
        ledger.get("format") != ATTEMPT_LEDGER_FORMAT
        or ledger.get("attempt_id") != root.name
    ):
        raise ValueError("attempt ledger format or attempt ID is invalid")
    key_boundary = ledger.get("key_boundary")
    if not isinstance(key_boundary, Mapping):
        raise ValueError("attempt ledger has no frozen key boundary")
    key_fields = key_boundary.get("key_fields")
    expected_keys = key_boundary.get("expected_keys")
    record_key_path = key_boundary.get("record_key_path")
    validate_key_boundary(key_fields, expected_keys, record_key_path)
    command = ledger.get("command")
    provenance = ledger.get("provenance")
    if (
        not isinstance(command, list)
        or any(not isinstance(part, str) or not part for part in command)
        or not isinstance(provenance, Mapping)
    ):
        raise ValueError("attempt ledger command or provenance is malformed")

    launch_path = root / LAUNCH_STATE_FILENAME
    launch_recorded = launch_path.exists()
    cleanup: dict[str, Any]
    launch_state = "not_started"
    execution_status = "not_started"
    if launch_recorded:
        launch = _read_canonical_json(launch_path)
        if (
            launch.get("format") != LAUNCH_STATE_FORMAT
            or launch.get("attempt_id") != root.name
            or launch.get("state") != "gated"
        ):
            raise ValueError("attempt launch record is malformed")
        process_id = launch.get("pid")
        process_group_id = launch.get("process_group_id")
        start_ticks = launch.get("process_start_ticks")
        if (
            isinstance(process_id, bool)
            or not isinstance(process_id, int)
            or process_id <= 0
            or process_group_id != process_id
            or (start_ticks is not None and not isinstance(start_ticks, str))
        ):
            raise ValueError("attempt launch record has an invalid process identity")
        cleanup = _cleanup_recorded_group(
            process_group_id,
            process_start_ticks=start_ticks,
            grace_seconds=cleanup_grace_seconds,
            final_reap_seconds=final_reap_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
        launch_state = "started"
        execution_status = "interruption"
    else:
        pending_events = sorted(root.glob("recovery-*.json"))
        pending_cleanup: Mapping[str, Any] | None = None
        for event_path in reversed(pending_events):
            event = _read_canonical_json(event_path)
            candidate = event.get("cleanup")
            if event.get("status") == "cleanup_unresolved" and isinstance(
                candidate, Mapping
            ):
                pending_cleanup = candidate
                break
        if pending_cleanup is None:
            cleanup = {"status": "not_needed", "process_group_empty": True}
        else:
            process_id = pending_cleanup.get("pid")
            process_group_id = pending_cleanup.get("process_group_id")
            start_ticks = pending_cleanup.get("process_start_ticks")
            if (
                isinstance(process_id, bool)
                or not isinstance(process_id, int)
                or process_id <= 0
                or process_group_id != process_id
            ):
                raise ValueError("pending cleanup record has no valid process identity")
            cleanup = _cleanup_recorded_group(
                process_group_id,
                process_start_ticks=(
                    start_ticks if isinstance(start_ticks, str) else None
                ),
                grace_seconds=cleanup_grace_seconds,
                final_reap_seconds=final_reap_seconds,
                poll_interval_seconds=poll_interval_seconds,
            )
            execution_status = "exception"

    event_status = (
        "cleanup_confirmed"
        if cleanup.get("status") == "confirmed_stopped"
        else "cleanup_unresolved"
    )
    _record_recovery_event(
        root,
        {
            "format": "gen1-stochastic-recovery-event-v1",
            "attempt_id": root.name,
            "recorded_at_utc": _utc_now(),
            "status": event_status,
            "cleanup": cleanup,
            "launch_recorded": launch_recorded,
        },
    )
    if cleanup.get("status") not in {"confirmed_stopped", "not_needed"}:
        raise AttemptPendingError(root, "recovery could not confirm all writers stopped")

    _fsync_attempt_tree(root)
    return seal_attempt(
        root,
        key_fields=key_fields,
        expected_keys=expected_keys,
        record_key_path=record_key_path,
        records_bytes=None,
        execution={
            "status": execution_status,
            "exit_code": None,
            "command": command,
            "launch_state": launch_state,
            "cleanup": cleanup,
            "recovered_after_supervisor_loss": True,
        },
        artifacts={},
        provenance=provenance,
    )


def supervise_attempt(
    attempt_dir: str | Path,
    *,
    repository_root: str | Path,
    command: Sequence[str],
    key_fields: Sequence[str],
    expected_keys: Sequence[Any],
    record_key_path: Sequence[str],
    provenance: Mapping[str, Any],
    timeout_seconds: float,
    environment: Mapping[str, str] | None = None,
    process_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run one gated child in a fresh directory and seal only after cleanup.

    The generator writes JSONL to ``records.raw.jsonl`` in its working
    directory. This helper is only the process boundary; formal callers must
    verify frozen policy/source/environment pins before invoking it.
    """
    repo = Path(repository_root).resolve()
    root = Path(attempt_dir).resolve()
    try:
        root.relative_to(repo)
    except ValueError:
        pass
    else:
        raise ValueError("attempt directory must be outside the Git working tree")
    if (
        not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be a finite positive number")
    argv = list(command)
    if not argv or any(not isinstance(part, str) or not part for part in argv):
        raise ValueError("command must be a nonempty argument vector")
    validate_key_boundary(key_fields, expected_keys, record_key_path)
    if os.name != "posix":
        raise OSError("the durable process-group launch gate currently requires POSIX")

    _prepare_attempt(
        root,
        command=argv,
        key_fields=key_fields,
        expected_keys=expected_keys,
        record_key_path=record_key_path,
        provenance=provenance,
    )
    try:
        inherited_stdin_fd = os.dup(0)
    except OSError as exc:
        raise OSError("could not preserve inherited stdin for gated launch") from exc
    process_factory = process_factory or subprocess.Popen
    process: Any | None = None
    status = "not_started"
    exit_code: int | None = None
    supervisor_error: str | None = None
    started = time.monotonic_ns()
    cleanup: dict[str, Any] = {"status": "not_needed"}
    launch_state = "not_started"
    launch_recorded = False
    try:
        with (root / "stdout.log").open("ab") as stdout, (root / "stderr.log").open(
            "ab"
        ) as stderr:
            try:
                gated_argv = _gate_command(argv, inherited_stdin_fd)
                process = process_factory(
                    gated_argv,
                    cwd=str(root),
                    env=dict(environment) if environment is not None else None,
                    stdin=subprocess.PIPE,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                    pass_fds=(inherited_stdin_fd,),
                )
                process_id = getattr(process, "pid", None)
                if (
                    isinstance(process_id, bool)
                    or not isinstance(process_id, int)
                    or process_id <= 0
                ):
                    raise RuntimeError("child launcher did not report a valid PID")
                launch_record = {
                    "format": LAUNCH_STATE_FORMAT,
                    "attempt_id": root.name,
                    "state": "gated",
                    "pid": process_id,
                    "process_group_id": process_id,
                    "process_start_ticks": _process_start_ticks(process_id),
                    "recorded_at_utc": _utc_now(),
                }
                _write_once_json(root / LAUNCH_STATE_FILENAME, launch_record)
                launch_recorded = True
                launch_state = "started"
                gate = getattr(process, "stdin", None)
                if gate is None:
                    raise RuntimeError("gated child launcher has no control pipe")
                if gate.write(b"go\n") != 3:
                    raise RuntimeError("could not release gated child launcher")
                gate.flush()
                gate.close()
                try:
                    returned = process.wait(timeout=timeout_seconds)
                    reported_exit = (
                        process.returncode
                        if process.returncode is not None
                        else returned
                    )
                    if isinstance(reported_exit, int) and not isinstance(
                        reported_exit, bool
                    ):
                        exit_code = reported_exit
                        status = "success" if exit_code == 0 else "error"
                    else:
                        status = "exception"
                        supervisor_error = (
                            "child wait did not report an integer exit code"
                        )
                except subprocess.TimeoutExpired as exc:
                    status = "timeout"
                    supervisor_error = f"{type(exc).__name__}: {exc}"
                except KeyboardInterrupt as exc:
                    status = "interruption"
                    supervisor_error = (
                        f"{type(exc).__name__}: interrupted by supervisor"
                    )
                except BaseException as exc:
                    status = "exception"
                    supervisor_error = f"{type(exc).__name__}: {exc}"
            except KeyboardInterrupt as exc:
                status = "interruption"
                supervisor_error = f"{type(exc).__name__}: interrupted by supervisor"
            except BaseException as exc:
                status = "exception"
                supervisor_error = f"{type(exc).__name__}: {exc}"
            finally:
                if process is not None:
                    gate = getattr(process, "stdin", None)
                    if gate is not None:
                        with suppress(BaseException):
                            gate.close()
    finally:
        os.close(inherited_stdin_fd)

    if process is not None:
        cleanup = _reap(
            process,
            terminate=(status in {"success", "interruption"} or not launch_recorded),
        )
        raw_exit_code = getattr(process, "returncode", None)
        if isinstance(raw_exit_code, int) and not isinstance(raw_exit_code, bool):
            exit_code = raw_exit_code
        if cleanup["status"] != "confirmed_stopped":
            pending_event = {
                "format": "gen1-stochastic-recovery-event-v1",
                "attempt_id": root.name,
                "recorded_at_utc": _utc_now(),
                "status": "cleanup_unresolved",
                "cleanup": cleanup,
                "launch_recorded": launch_recorded,
            }
            _record_recovery_event(root, pending_event)
            raise AttemptPendingError(root, "child or writer process remains active")
        if launch_recorded and cleanup["signals_sent"] and status == "success":
            status = "exception"
            supervisor_error = (
                "generator leader exited while process-group writers remained"
            )
    _fsync_attempt_tree(root)

    execution: dict[str, Any] = {
        "status": status,
        "exit_code": exit_code,
        "duration_ns": time.monotonic_ns() - started,
        "command": argv,
        "launch_state": launch_state,
        "cleanup": cleanup,
    }
    if supervisor_error is not None:
        execution["supervisor_error"] = supervisor_error
    return seal_attempt(
        root,
        key_fields=key_fields,
        expected_keys=expected_keys,
        record_key_path=record_key_path,
        records_bytes=None,
        execution=execution,
        artifacts={},
        provenance=provenance,
    )


def run_frozen_attempt(
    attempt_dir: str | Path,
    *,
    repository_root: str | Path,
    command: Sequence[str],
    key_fields: Sequence[str],
    expected_keys: Sequence[Any],
    record_key_path: Sequence[str],
    expected_head: str,
    expected_tree: str,
    file_hashes_by_group: Mapping[str, Mapping[str, str]],
    environment_inventory_path: str | Path,
    environment_inventory_sha256: str,
    timeout_seconds: float,
    environment: Mapping[str, str] | None = None,
    process_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Verify E and its external environment inventory before starting a child.

    ``key_fields`` and ``expected_keys`` must be extracted from the frozen
    policy by the policy adapter. The runner checks the full observed output
    against those exact keys; it never accepts a caller-supplied case count.
    """
    root = Path(attempt_dir).resolve()
    repo = Path(repository_root).resolve()
    try:
        root.relative_to(repo)
    except ValueError:
        pass
    else:
        raise ValueError("attempt directory must be outside the Git working tree")
    validate_key_boundary(key_fields, expected_keys, record_key_path)

    try:
        infrastructure_paths = set(file_hashes_by_group["infrastructure"])
        if not REQUIRED_INFRASTRUCTURE_FILES.issubset(infrastructure_paths):
            missing = sorted(REQUIRED_INFRASTRUCTURE_FILES - infrastructure_paths)
            raise ValueError(
                f"source pin omits required infrastructure files: {missing}"
            )
        inventory_candidate = Path(environment_inventory_path).resolve()
        try:
            inventory_candidate.relative_to(repo)
        except ValueError:
            pass
        else:
            raise ValueError(
                "environment inventory must remain outside the Git working tree"
            )
        source_pin = verify_source_pin(
            repo,
            expected_head=expected_head,
            expected_tree=expected_tree,
            file_hashes_by_group=file_hashes_by_group,
        )
        environment_pin = verify_environment_inventory(
            environment_inventory_path, environment_inventory_sha256
        )
    except Exception as exc:
        preflight_provenance = {
            "expected_head": expected_head,
            "expected_tree": expected_tree,
            "expected_file_hashes_by_group": file_hashes_by_group,
            "environment_inventory_path": str(environment_inventory_path),
            "environment_inventory_sha256": environment_inventory_sha256,
        }
        _prepare_attempt(
            root,
            command=command,
            key_fields=key_fields,
            expected_keys=expected_keys,
            record_key_path=record_key_path,
            provenance=preflight_provenance,
        )
        _fsync_attempt_tree(root)
        return seal_attempt(
            root,
            key_fields=key_fields,
            expected_keys=expected_keys,
            record_key_path=record_key_path,
            records_bytes=b"",
            execution={
                "status": "exception",
                "exit_code": None,
                "command": list(command),
                "launch_state": "not_started",
                "cleanup": {"status": "not_needed"},
                "supervisor_error": f"preflight {type(exc).__name__}: {exc}",
            },
            artifacts={},
            provenance=preflight_provenance,
        )

    return supervise_attempt(
        root,
        repository_root=repo,
        command=command,
        key_fields=key_fields,
        expected_keys=expected_keys,
        record_key_path=record_key_path,
        provenance={
            "source_pin": source_pin,
            "environment_inventory": environment_pin,
        },
        timeout_seconds=timeout_seconds,
        environment=environment,
        process_factory=process_factory,
    )


def main() -> int:
    raise SystemExit(
        "formal CLI is held until the frozen common-policy adapter is ready; "
        "no attempt was started"
    )


if __name__ == "__main__":
    main()
