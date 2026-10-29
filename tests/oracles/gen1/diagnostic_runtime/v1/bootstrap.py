"""No-fork child bootstrap for the shared diagnostic governor."""

from __future__ import annotations

import base64
import ctypes
import errno
import hashlib
import importlib.machinery
import io
import json
import os
import platform
import re
import runpy
import socket
import stat
import sys
from pathlib import Path
from typing import Any


_SIGKILL = 9
_PR_SET_PDEATHSIG = 1
_PR_GET_PDEATHSIG = 2
_PR_SET_NO_NEW_PRIVS = 38
_PR_SET_SECCOMP = 22
_SECCOMP_MODE_FILTER = 2
_AUDIT_ARCH_X86_64 = 0xC000003E
_SECCOMP_RET_KILL_PROCESS = 0x80000000
_SECCOMP_RET_ERRNO = 0x00050000
_SECCOMP_RET_ALLOW = 0x7FFF0000
_EPERM = errno.EPERM
_MAX_CONTROL_BYTES = 1_048_576


class BootstrapError(RuntimeError):
    pass


class SockFilter(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_ushort),
        ("jt", ctypes.c_ubyte),
        ("jf", ctypes.c_ubyte),
        ("k", ctypes.c_uint),
    ]


class SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(SockFilter))]


def _strict_json(raw: bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise BootstrapError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise BootstrapError(f"nonfinite JSON constant: {value}")

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapError("control or launch JSON is malformed") from exc


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
        raise BootstrapError("launch packet is not finite JSON") from exc


def _read_pinned_file(path: str | os.PathLike[str], expected_sha256: Any) -> bytes:
    try:
        path_text = os.fspath(path)
    except TypeError as exc:
        raise BootstrapError("source catalog path/hash pin is malformed") from exc
    if (
        not isinstance(path_text, str)
        or not Path(path_text).is_absolute()
        or not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise BootstrapError("source catalog path/hash pin is malformed")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path_text, flags)
    except OSError as exc:
        raise BootstrapError("pinned source catalog cannot be opened") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise BootstrapError("pinned source catalog is not a regular file")
        chunks = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after:
            raise BootstrapError("pinned source catalog changed during capture")
    finally:
        os.close(descriptor)
    captured = b"".join(chunks)
    if len(captured) != after.st_size:
        raise BootstrapError("pinned source catalog read length differs from fstat")
    if hashlib.sha256(captured).hexdigest() != expected_sha256:
        raise BootstrapError("captured source catalog bytes differ from the immutable pin")
    return captured


def _normalize_source_catalog(raw: Any, *, manifest: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize the original A2 map or C file list without rewriting its manifest."""
    candidate = manifest.get("candidate")
    configured_roots = (
        candidate.get("source_scan_roots") if isinstance(candidate, dict) else None
    )
    if raw is None and isinstance(candidate, dict):
        raw = candidate.get("source_catalog")
    if not isinstance(raw, (dict, list)):
        raise BootstrapError("source catalog is not an A2 map or Cylinder list")
    if isinstance(raw, dict) and set(raw) == {"path", "sha256"}:
        raise BootstrapError("pinned source catalog must be decoded before ADMIT")

    supplied_inventory = None
    if isinstance(raw, list):
        rows = raw
    else:
        if not set(raw).issubset({"files", "inventory", "scan_roots"}) or "files" not in raw:
            raise BootstrapError("source catalog fields differ from the canonical adapter")
        configured_roots = raw.get("scan_roots", configured_roots)
        supplied_inventory = raw.get("inventory")
        values = raw.get("files")
        if isinstance(values, list):
            rows = values
        elif isinstance(values, dict):
            rows = [{"path": name, "sha256": digest} for name, digest in values.items()]
        else:
            raise BootstrapError("source catalog files must be a map or original C row list")

    files: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise BootstrapError("source catalog row is malformed")
        name, digest = row["path"], row["sha256"]
        if (
            not isinstance(name, str)
            or not name
            or name.startswith("/")
            or "\\" in name
            or any(part in {"", ".", ".."} for part in name.split("/"))
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or name in files
        ):
            raise BootstrapError("source catalog path/hash is malformed or duplicated")
        files[name] = digest
    inventory = sorted(files)
    if not inventory:
        raise BootstrapError("source catalog is empty")
    if supplied_inventory is not None and supplied_inventory != inventory:
        raise BootstrapError("source catalog inventory contradicts its file pins")
    if configured_roots is None:
        if any("/" not in name for name in inventory):
            scan_roots = ["."]
        else:
            # Scan each top-level project source root, not only the deepest
            # directory represented by a listed file. This keeps unlisted
            # importable siblings under e.g. src/ inside the completeness gate.
            scan_roots = sorted({name.split("/", 1)[0] for name in inventory})
    else:
        scan_roots = configured_roots
    if (
        not isinstance(scan_roots, list)
        or not scan_roots
        or any(
            not isinstance(path, str)
            or path == ""
            or path.startswith("/")
            or "\\" in path
            or (path != "." and any(part in {"", ".", ".."} for part in path.split("/")))
            for path in scan_roots
        )
        or scan_roots != sorted(set(scan_roots))
    ):
        raise BootstrapError("source catalog scan roots are malformed or nondeterministic")
    return {"files": files, "inventory": inventory, "scan_roots": scan_roots}


def _read_line(sock: socket.socket) -> dict[str, Any]:
    data = bytearray()
    while len(data) <= _MAX_CONTROL_BYTES:
        chunk = sock.recv(1)
        if not chunk:
            raise BootstrapError("root control socket closed before a complete record")
        if chunk == b"\n":
            value = _strict_json(bytes(data))
            if not isinstance(value, dict):
                raise BootstrapError("control record must be a JSON object")
            return value
        data.extend(chunk)
    raise BootstrapError("control record exceeds the fixed size limit")


def _send(sock: socket.socket, value: dict[str, Any]) -> None:
    raw = _canonical(value)
    if len(raw) > _MAX_CONTROL_BYTES:
        raise BootstrapError("control record exceeds the fixed size limit")
    sock.sendall(raw)


def _check_meta_path() -> list[str]:
    known = {
        "BuiltinImporter": importlib.machinery.BuiltinImporter,
        "FrozenImporter": importlib.machinery.FrozenImporter,
        "PathFinder": importlib.machinery.PathFinder,
    }
    observed = []
    for finder in sys.meta_path:
        name = next((key for key, value in known.items() if finder is value), None)
        if name is None:
            raise BootstrapError("bootstrap found an untrusted meta_path finder")
        observed.append(name)
    if len(observed) != 3 or set(observed) != set(known):
        raise BootstrapError("bootstrap standard finder inventory is incomplete")
    return observed


def _proc_stat(pid: int) -> dict[str, int]:
    try:
        raw = Path(f"/proc/{pid}/stat").read_bytes()
    except OSError as exc:
        raise BootstrapError(f"proc stat unavailable for PID {pid}: errno={exc.errno}") from exc
    prefix, delimiter, tail = raw.rpartition(b")")
    if not delimiter:
        raise BootstrapError("proc stat lacks its final comm delimiter")
    first = prefix.split(maxsplit=1)[0]
    fields = tail.strip().split()
    if not first.isdigit() or int(first) != pid or len(fields) < 22:
        raise BootstrapError("proc stat PID or field count differs")
    try:
        return {
            "pid": pid,
            "ppid": int(fields[1]),
            "pgrp": int(fields[2]),
            "session": int(fields[3]),
            "starttime_ticks": int(fields[19]),
        }
    except ValueError as exc:
        raise BootstrapError("proc stat identity fields are malformed") from exc


def _proc_argv(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError as exc:
        raise BootstrapError(f"proc cmdline unavailable for PID {pid}: errno={exc.errno}") from exc
    if not raw.endswith(b"\0"):
        raise BootstrapError("proc cmdline lacks a final NUL")
    return [os.fsdecode(item) for item in raw[:-1].split(b"\0")]


def _boot_id() -> str:
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except OSError as exc:
        raise BootstrapError("kernel boot ID is unavailable") from exc
    if len(value) != 36 or value.count("-") != 4:
        raise BootstrapError("kernel boot ID is malformed")
    return value


def _libc() -> ctypes.CDLL:
    if sys.platform != "linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise BootstrapError("child containment supports only Linux x86_64")
    if ctypes.sizeof(ctypes.c_void_p) != 8 or ctypes.sizeof(ctypes.c_long) != 8:
        raise BootstrapError("child containment requires LP64")
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.restype = ctypes.c_int
    prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    return libc


def _pdeath_evidence(expected_parent_pid: int) -> dict[str, int]:
    libc = _libc()
    before = int(libc.getppid())
    if before != expected_parent_pid:
        raise BootstrapError("parent PID changed before PDEATHSIG readback")
    value = ctypes.c_int(0)
    ctypes.set_errno(0)
    result = libc.prctl(
        _PR_GET_PDEATHSIG,
        ctypes.c_ulong(ctypes.addressof(value)),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
    )
    if result != 0:
        raise BootstrapError(f"PR_GET_PDEATHSIG failed with errno {ctypes.get_errno()}")
    after = int(libc.getppid())
    if after != expected_parent_pid or value.value != _SIGKILL:
        raise BootstrapError("PDEATHSIG or parent PID did not survive both race checks")
    return {"parent_pid_before": before, "parent_pid_after": after, "pdeath_signal": value.value}


def _seccomp_filter() -> dict[str, Any]:
    """Deny process/thread creation, exec, credentials, and PDEATHSIG clearing."""
    arch = platform.machine().lower()
    if sys.platform != "linux" or arch not in {"x86_64", "amd64"}:
        raise BootstrapError("seccomp containment supports only Linux x86_64")
    if ctypes.sizeof(ctypes.c_void_p) != 8 or ctypes.sizeof(ctypes.c_long) != 8:
        raise BootstrapError("seccomp containment requires LP64")

    # cBPF instruction encodings from linux/filter.h.
    instructions: list[tuple[int, int, int, int]] = [
        (0x20, 0, 0, 4),  # LD W ABS offsetof(seccomp_data.arch)
        (0x15, 1, 0, _AUDIT_ARCH_X86_64),  # JEQ arch: skip KILL on match
        (0x06, 0, 0, _SECCOMP_RET_KILL_PROCESS),
        (0x20, 0, 0, 0),  # LD W ABS offsetof(seccomp_data.nr)
        (0x45, 0, 1, 0x40000000),  # JSET x32 ABI bit -> KILL
        (0x06, 0, 0, _SECCOMP_RET_KILL_PROCESS),
    ]
    denied = {
        "clone": 56,
        "fork": 57,
        "vfork": 58,
        "execve": 59,
        "setuid": 105,
        "setgid": 106,
        "setreuid": 113,
        "setregid": 114,
        "setgroups": 116,
        "setresuid": 117,
        "setresgid": 119,
        "setfsuid": 122,
        "setfsgid": 123,
        "capset": 126,
        "prctl": 157,
        "unshare": 272,
        "setns": 308,
        "execveat": 322,
        "clone3": 435,
    }
    for number in denied.values():
        instructions.extend(
            [
                (0x15, 0, 1, number),  # JEQ nr: next instruction denies exact syscall
                (0x06, 0, 0, _SECCOMP_RET_ERRNO | _EPERM),
            ]
        )
    instructions.append((0x06, 0, 0, _SECCOMP_RET_ALLOW))
    array_type = SockFilter * len(instructions)
    filters = array_type(*(SockFilter(*row) for row in instructions))
    program = SockFprog(len(instructions), filters)

    libc = _libc()
    ctypes.set_errno(0)
    if libc.prctl(
        _PR_SET_NO_NEW_PRIVS,
        ctypes.c_ulong(1),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
    ) != 0:
        raise BootstrapError(f"PR_SET_NO_NEW_PRIVS failed with errno {ctypes.get_errno()}")
    ctypes.set_errno(0)
    if libc.prctl(
        _PR_SET_SECCOMP,
        ctypes.c_ulong(_SECCOMP_MODE_FILTER),
        ctypes.c_ulong(ctypes.addressof(program)),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
    ) != 0:
        raise BootstrapError(f"PR_SET_SECCOMP failed with errno {ctypes.get_errno()}")
    try:
        status_raw = Path("/proc/self/status").read_bytes()
    except OSError as exc:
        raise BootstrapError("seccomp readback from proc status failed") from exc
    fields: dict[str, list[bytes]] = {"NoNewPrivs": [], "Seccomp": []}
    for line in status_raw.splitlines():
        name, separator, value = line.partition(b":")
        if separator and name.decode("ascii", "ignore") in fields:
            fields[name.decode("ascii")].append(value.strip())
    if (
        len(fields["NoNewPrivs"]) != 1
        or fields["NoNewPrivs"][0] != b"1"
        or len(fields["Seccomp"]) != 1
        or fields["Seccomp"][0] != b"2"
    ):
        raise BootstrapError("NoNewPrivs/Seccomp readback differs")
    return {
        "architecture": "x86_64",
        "audit_architecture": _AUDIT_ARCH_X86_64,
        "denied_syscalls": denied,
        "no_new_privs": 1,
        "seccomp_mode": 2,
        "status_sha256": hashlib.sha256(status_raw).hexdigest(),
        "status_base64": base64.b64encode(status_raw).decode("ascii"),
    }


def _read_payload_line() -> bytes:
    chunks = bytearray()
    while len(chunks) <= _MAX_CONTROL_BYTES:
        part = os.read(0, 65536)
        if not part:
            break
        newline = part.find(b"\n")
        if newline >= 0:
            chunks.extend(part[: newline + 1])
            if part[newline + 1 :]:
                raise BootstrapError("worker stdin contains bytes after its sole request line")
            break
        chunks.extend(part)
    if len(chunks) > _MAX_CONTROL_BYTES or not chunks.endswith(b"\n"):
        raise BootstrapError("worker stdin does not contain exactly one bounded JSON line")
    # EOF is part of the API; reject a second payload or a pipe that remains open.
    if os.read(0, 1):
        raise BootstrapError("worker stdin did not close after its single request")
    value = _strict_json(bytes(chunks[:-1]))
    if not isinstance(value, dict):
        raise BootstrapError("worker request must be a JSON object")
    return bytes(chunks)


def _verify_launch_packet(packet: dict[str, Any], *, token: str, binding: dict[str, Any]) -> dict[str, Any]:
    required = {
        "model",
        "epoch_id",
        "ledger_id",
        "phase",
        "role",
        "launch_token",
        "manifest_path",
        "manifest_sha256",
        "root_binding",
        "child_binding",
        "max_seconds",
        "effective_deadline_mono_ns",
        "entry_path",
        "entry_sha256",
        "worktree_root",
        "site_packages",
        "result_path",
        "source_catalog",
    }
    if set(packet) != required or packet.get("launch_token") != token:
        raise BootstrapError("launch packet fields or token differ from the fixed contract")
    if packet.get("child_binding") != binding:
        raise BootstrapError("launch packet child binding differs from the current process")
    if not isinstance(packet.get("model"), str) or packet["model"] not in {"A2", "Cylinder"}:
        raise BootstrapError("launch packet model is invalid")
    if not isinstance(packet.get("phase"), str) or not isinstance(packet.get("role"), str):
        raise BootstrapError("launch packet role context is invalid")
    if not isinstance(packet.get("epoch_id"), str) or not packet["epoch_id"]:
        raise BootstrapError("launch packet epoch is invalid")
    if packet["ledger_id"] != f"{packet['model']}:{packet['epoch_id']}":
        raise BootstrapError("launch packet ledger identity is contradictory")
    if not isinstance(packet.get("entry_path"), str) or not Path(packet["entry_path"]).is_absolute():
        raise BootstrapError("launch packet entry path is invalid")
    entry_hash = packet.get("entry_sha256")
    if not isinstance(entry_hash, str) or len(entry_hash) != 64:
        raise BootstrapError("launch packet entry hash is invalid")
    _read_pinned_file(packet["entry_path"], entry_hash)
    manifest_path = packet.get("manifest_path")
    if not isinstance(manifest_path, str) or not Path(manifest_path).is_absolute():
        raise BootstrapError("launch packet manifest path is invalid")
    manifest_raw = _read_pinned_file(manifest_path, packet.get("manifest_sha256"))
    manifest = _strict_json(manifest_raw)
    catalog_pin = manifest.get("source_catalog")
    if catalog_pin is None and isinstance(manifest.get("candidate"), dict):
        catalog_pin = manifest["candidate"].get("source_catalog")
    packet_catalog = packet.get("source_catalog")
    if isinstance(catalog_pin, dict) and set(catalog_pin) == {"path", "sha256"}:
        catalog_raw = _read_pinned_file(catalog_pin["path"], catalog_pin["sha256"])
        catalog = _normalize_source_catalog(_strict_json(catalog_raw), manifest=manifest)
    elif isinstance(catalog_pin, (dict, list)):
        catalog = _normalize_source_catalog(catalog_pin, manifest=manifest)
    else:
        raise BootstrapError("source catalog pin is malformed")
    if packet_catalog != catalog:
        raise BootstrapError("ADMIT source catalog differs from immutable manifest")

    worktree = Path(packet["worktree_root"]).resolve(strict=True)
    if not worktree.is_dir():
        raise BootstrapError("candidate worktree root is absent")
    files = catalog.get("files")
    inventory = catalog.get("inventory")
    scan_roots = catalog.get("scan_roots", ["."])
    if (
        not isinstance(files, dict)
        or not isinstance(inventory, list)
        or inventory != sorted(inventory)
        or len(inventory) != len(set(inventory))
        or set(inventory) != set(files)
        or not isinstance(scan_roots, list)
        or not scan_roots
    ):
        raise BootstrapError("source catalog inventory is malformed")
    suffixes = {".py", ".pyi", ".pth", ".so"}

    def normalized_scan_root(raw: Any) -> str:
        if raw == ".":
            return ""
        if (
            not isinstance(raw, str)
            or not raw
            or "\\" in raw
            or Path(raw).is_absolute()
            or any(part in {"", ".", ".."} for part in Path(raw).parts)
        ):
            raise BootstrapError("source scan root is unsafe")
        return Path(raw).as_posix()

    normalized_roots = [normalized_scan_root(raw) for raw in scan_roots]
    if len(normalized_roots) != len(set(normalized_roots)):
        raise BootstrapError("source scan roots contain duplicate aliases")

    def in_scan_root(name: str, normalized_root: str) -> bool:
        return not normalized_root or name == normalized_root or name.startswith(normalized_root + "/")

    observed_sources: set[str] = set()

    def walk_error(exc: OSError) -> None:
        raise BootstrapError(f"candidate source walk failed: errno={exc.errno}") from exc

    for normalized_root in normalized_roots:
        scan_base = worktree if not normalized_root else worktree / normalized_root
        if scan_base.is_symlink() or not scan_base.is_dir():
            raise BootstrapError("source scan root is absent or a symlink")
        for directory, child_dirs, child_files in os.walk(
            scan_base, followlinks=False, onerror=walk_error
        ):
            base = Path(directory)
            for child in list(child_dirs):
                child_path = base / child
                if child_path.is_symlink():
                    raise BootstrapError("symlink appears in candidate source scan")
                if child == "__pycache__":
                    child_dirs.remove(child)
            for child in child_files:
                path = base / child
                if path.is_symlink():
                    raise BootstrapError("symlink appears in candidate source scan")
                if path.suffix not in suffixes:
                    continue
                relative = path.relative_to(worktree).as_posix()
                if in_scan_root(relative, normalized_root):
                    observed_sources.add(relative)
    expected_sources = {
        name
        for name in inventory
        if Path(name).suffix in suffixes and "__pycache__" not in Path(name).parts
    }
    expected_scanned = {
        name
        for name in expected_sources
        if any(in_scan_root(name, root) for root in normalized_roots)
    }
    if observed_sources != expected_scanned:
        raise BootstrapError("candidate importable source inventory changed after root binding")
    for raw_name, expected_hash in files.items():
        if (
            not isinstance(raw_name, str)
            or not raw_name
            or "\\" in raw_name
            or Path(raw_name).is_absolute()
            or any(part in {"", ".", ".."} for part in Path(raw_name).parts)
            or not isinstance(expected_hash, str)
            or len(expected_hash) != 64
        ):
            raise BootstrapError("catalogued source name/hash is malformed")
        source = worktree.joinpath(*Path(raw_name).parts)
        resolved = source.resolve(strict=True)
        resolved.relative_to(worktree)
        if source.is_symlink() or not source.is_file():
            raise BootstrapError("catalogued candidate source is not a regular file")
        _read_pinned_file(source, expected_hash)
    site_packages = packet.get("site_packages")
    if not isinstance(site_packages, list) or any(
        not isinstance(path, str) or not Path(path).is_absolute() for path in site_packages
    ):
        raise BootstrapError("site-packages inventory is malformed")
    return {"manifest": manifest, "catalog": catalog, "worktree_root": str(worktree)}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2 or argv[0] != "--token" or not argv[1]:
        raise BootstrapError("bootstrap argv must be exactly --token TOKEN")
    token = argv[1]
    fd_text = os.environ.get("GRAGRA_DIAGNOSTIC_CONTROL_FD")
    if not fd_text or not fd_text.isdecimal():
        raise BootstrapError("control socket descriptor is missing")
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"):
        if os.environ.get(name):
            raise BootstrapError(f"ambient Python variable is forbidden: {name}")
    model = os.environ.get("GRAGRA_DIAGNOSTIC_MODEL")
    phase = os.environ.get("GRAGRA_DIAGNOSTIC_PHASE")
    ledger_id = os.environ.get("GRAGRA_DIAGNOSTIC_LEDGER_ID")
    if model not in {"A2", "Cylinder"} or not phase or not ledger_id:
        raise BootstrapError("fixed launch context environment is incomplete")

    pid = os.getpid()
    _check_meta_path()
    parent_before = os.getppid()
    if parent_before <= 1:
        raise BootstrapError("bootstrap has no live direct governor parent")
    pdeath = _pdeath_evidence(parent_before)
    self_before = _proc_stat(pid)
    parent_identity = _proc_stat(parent_before)
    boot = _boot_id()
    argv_bytes = _proc_argv(pid)
    expected_argv = [
        str(Path(sys.executable).resolve(strict=True)),
        "-I",
        "-S",
        str(Path(__file__).resolve(strict=True)),
        "--token",
        token,
    ]
    if argv_bytes != expected_argv:
        raise BootstrapError("actual /proc argv differs from fixed child command")
    containment = _seccomp_filter()
    self_after = _proc_stat(pid)
    parent_after = _proc_stat(parent_before)
    if self_before != self_after or parent_identity != parent_after or os.getppid() != parent_before:
        raise BootstrapError("root/child process identity changed during containment setup")

    binding = {
        "pid": pid,
        "parent_pid": parent_before,
        "boot_id": boot,
        "starttime_ticks": self_after["starttime_ticks"],
        "parent_boot_id": boot,
        "parent_starttime_ticks": parent_after["starttime_ticks"],
        "pdeath_signal": pdeath["pdeath_signal"],
        "argv": argv_bytes,
        "launch_token": token,
        "phase": phase,
        "model": model,
        "ledger_id": ledger_id,
    }
    control = socket.socket(fileno=int(fd_text))
    try:
        _send(
            control,
            {
                "type": "HELLO",
                "launch_token": token,
                "binding": binding,
                "parent_identity": parent_after,
                "containment": containment,
            },
        )
        response = _read_line(control)
        if response.get("type") != "ADMIT" or response.get("launch_token") != token:
            reason = response.get("reason", "root refused launch")
            raise BootstrapError(f"root did not admit child: {reason}")
        packet = response.get("launch_packet")
        if not isinstance(packet, dict):
            raise BootstrapError("ADMIT lacks launch packet")
        verified = _verify_launch_packet(packet, token=token, binding=binding)
        expected_root = packet.get("root_binding")
        if not isinstance(expected_root, dict):
            raise BootstrapError("launch packet root binding is missing")
        parent_after_admit = _proc_stat(parent_before)
        if (
            os.getppid() != parent_before
            or parent_after_admit != expected_root.get("stat_identity")
            or expected_root.get("pid") != parent_before
            or expected_root.get("boot_id") != boot
        ):
            raise BootstrapError("root identity changed between HELLO and ADMIT")
        control_fd = control.detach()
    except BaseException:
        control.close()
        raise

    payload = _read_payload_line()
    worktree = verified["worktree_root"]
    site_packages = packet["site_packages"]
    candidate_roots = [str(Path(worktree)), str(Path(worktree) / "src")]
    candidate_roots.extend(str(Path(item).resolve(strict=True)) for item in site_packages)
    if len(candidate_roots) != len(set(candidate_roots)):
        os.close(control_fd)
        raise BootstrapError("candidate import roots are duplicated")
    candidate_path_set = {Path(root).resolve(strict=True) for root in candidate_roots}
    stdlib_paths = [
        item
        for item in sys.path
        if item and Path(item).resolve() not in candidate_path_set
    ]
    sys.path[:] = [*stdlib_paths, *candidate_roots]
    if sys.flags.isolated != 1 or sys.flags.no_site != 1:
        os.close(control_fd)
        raise BootstrapError("worker interpreter did not start with -I -S")

    # Preserve the one-line request for the family entry without a second child
    # pipe or any bootstrap messages on stdout.
    sys.stdin = io.TextIOWrapper(io.BytesIO(payload), encoding="utf-8")
    os.environ["GRAGRA_DIAGNOSTIC_CONTROL_FD"] = str(control_fd)
    os.environ["GRAGRA_DIAGNOSTIC_LAUNCH_TOKEN"] = token
    runpy.run_path(packet["entry_path"], run_name="__main__")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except SystemExit as exc:
        code = exc.code if type(exc.code) is int else (0 if exc.code is None else 1)
    except BaseException as exc:
        # Keep stdout reserved for the one family result. Refusals are sent on
        # the control channel when it is still connected; stderr is diagnostic.
        try:
            fd_text = os.environ.get("GRAGRA_DIAGNOSTIC_CONTROL_FD")
            token = sys.argv[-1] if sys.argv else "unknown"
            if fd_text and fd_text.isdecimal():
                _send(
                    socket.socket(fileno=int(fd_text)),
                    {"type": "REFUSED", "launch_token": token, "reason": str(exc)},
                )
        except BaseException:
            pass
        os.write(2, (f"diagnostic bootstrap refused: {exc}\n").encode("utf-8", "replace"))
        code = 126
    os._exit(code)
