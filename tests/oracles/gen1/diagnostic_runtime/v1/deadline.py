"""Checked glibc POSIX timer and parent-death primitives for the root governor.

This module deliberately supports only Linux x86_64 LP64 glibc.  The hard
deadline is installed as one absolute CLOCK_MONOTONIC SIGKILL timer and is
never disarmed by this API.
"""

from __future__ import annotations

import ctypes
import os
import re
import signal
from pathlib import Path
from typing import Any


CLOCK_MONOTONIC = 1
CLOCK_BOOTTIME = 7
SIGEV_SIGNAL = 0
TIMER_ABSTIME = 1
SIGKILL = 9
PR_SET_PDEATHSIG = 1
PR_GET_PDEATHSIG = 2


class DeadlineError(RuntimeError):
    """The required root-kernel-deadline or parent-death primitive refused."""


class Sigval(ctypes.Union):
    _fields_ = [("sival_int", ctypes.c_int), ("sival_ptr", ctypes.c_void_p)]


class Sigevent(ctypes.Structure):
    _fields_ = [
        ("sigev_value", Sigval),
        ("sigev_signo", ctypes.c_int),
        ("sigev_notify", ctypes.c_int),
        ("pad", ctypes.c_int * 12),
    ]


class Timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


class Itimerspec(ctypes.Structure):
    _fields_ = [("it_interval", Timespec), ("it_value", Timespec)]


TimerT = ctypes.c_void_p
_ROOT_TIMER: TimerT | None = None
_LIBC: Any = None


def timer_abi_layout() -> dict[str, Any]:
    """Return ctypes sizes/offsets; callers compare this with the reviewed ABI."""
    return {
        "pointer_bytes": ctypes.sizeof(ctypes.c_void_p),
        "long_bytes": ctypes.sizeof(ctypes.c_long),
        "int_bytes": ctypes.sizeof(ctypes.c_int),
        "sigval_bytes": ctypes.sizeof(Sigval),
        "sigevent_bytes": ctypes.sizeof(Sigevent),
        "sigevent_offsets": {
            "value": Sigevent.sigev_value.offset,
            "signo": Sigevent.sigev_signo.offset,
            "notify": Sigevent.sigev_notify.offset,
            "pad": Sigevent.pad.offset,
        },
        "timespec_bytes": ctypes.sizeof(Timespec),
        "timespec_nsec_offset": Timespec.tv_nsec.offset,
        "itimerspec_bytes": ctypes.sizeof(Itimerspec),
        "itimerspec_value_offset": Itimerspec.it_value.offset,
    }


def require_supported_abi() -> dict[str, Any]:
    """Refuse every platform except the locally reviewed ABI."""
    if os.name != "posix" or os.uname().sysname != "Linux" or os.uname().machine != "x86_64":
        raise DeadlineError("only Linux x86_64 LP64 glibc is reviewed")
    layout = timer_abi_layout()
    expected = {
        "pointer_bytes": 8,
        "long_bytes": 8,
        "int_bytes": 4,
        "sigval_bytes": 8,
        "sigevent_bytes": 64,
        "sigevent_offsets": {"value": 0, "signo": 8, "notify": 12, "pad": 16},
        "timespec_bytes": 16,
        "timespec_nsec_offset": 8,
        "itimerspec_bytes": 32,
        "itimerspec_value_offset": 16,
    }
    if layout != expected:
        raise DeadlineError("ctypes timer ABI layout differs from reviewed glibc x86_64")
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        version_function = libc.gnu_get_libc_version
    except AttributeError as exc:
        raise DeadlineError("the process libc is not glibc") from exc
    version_function.restype = ctypes.c_char_p
    version_function.argtypes = []
    if not version_function():
        raise DeadlineError("glibc is required for the reviewed timer ABI")
    return {"abi": "linux-x86_64-lp64-glibc", "layout": layout}


def _libc() -> Any:
    global _LIBC
    if _LIBC is not None:
        return _LIBC
    require_supported_abi()
    libc = ctypes.CDLL(None, use_errno=True)
    signatures = {
        "timer_create": (
            ctypes.c_int,
            [ctypes.c_int, ctypes.POINTER(Sigevent), ctypes.POINTER(TimerT)],
        ),
        "timer_settime": (
            ctypes.c_int,
            [TimerT, ctypes.c_int, ctypes.POINTER(Itimerspec), ctypes.POINTER(Itimerspec)],
        ),
        "timer_gettime": (ctypes.c_int, [TimerT, ctypes.POINTER(Itimerspec)]),
        "clock_gettime": (ctypes.c_int, [ctypes.c_int, ctypes.POINTER(Timespec)]),
        "getppid": (ctypes.c_int, []),
        "prctl": (
            ctypes.c_int,
            [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong],
        ),
    }
    for name, (restype, argtypes) in signatures.items():
        try:
            function = getattr(libc, name)
        except AttributeError as exc:
            raise DeadlineError(f"libc.{name} is unavailable") from exc
        function.restype = restype
        function.argtypes = argtypes
    _LIBC = libc
    return libc


def _checked_ns(clock_id: int) -> int:
    ts = Timespec()
    ctypes.set_errno(0)
    if _libc().clock_gettime(clock_id, ctypes.byref(ts)) != 0:
        code = ctypes.get_errno()
        raise DeadlineError(f"clock_gettime({clock_id}) failed with errno {code}")
    if ts.tv_sec < 0 or not 0 <= ts.tv_nsec < 1_000_000_000:
        raise DeadlineError("clock_gettime returned an invalid timespec")
    return int(ts.tv_sec) * 1_000_000_000 + int(ts.tv_nsec)


def _proc_start_ticks(raw_stat: bytes) -> int:
    """Read stat field22 after the final ')' so spaces/parentheses in comm are safe."""
    try:
        _prefix, close, tail = raw_stat.rpartition(b")")
        pid_field = _prefix.split(maxsplit=1)[0]
        if not close or not pid_field.isdigit():
            raise ValueError
        fields = tail.strip().split()
        value = fields[19]  # tail begins at field 3 (state); field22 is index19.
        ticks = int(value)
    except (IndexError, ValueError) as exc:
        raise DeadlineError("/proc/self/stat start-time field22 is malformed") from exc
    if ticks <= 0:
        raise DeadlineError("/proc/self/stat start-time must be positive")
    return ticks


def _time_namespace_evidence() -> dict[str, Any]:
    try:
        own = os.readlink("/proc/self/ns/time")
        children = os.readlink("/proc/self/ns/time_for_children")
    except OSError as exc:
        raise DeadlineError("self time namespace identity is unavailable") from exc
    if own != children:
        raise DeadlineError("self/time_for_children namespaces differ")
    offsets_path = Path("/proc/self/timens_offsets")
    try:
        offsets_raw = offsets_path.read_bytes()
        offsets = offsets_raw.decode("ascii")
    except OSError as exc:
        raise DeadlineError("time namespace offset evidence is unavailable") from exc
    except UnicodeDecodeError as exc:
        raise DeadlineError("time namespace offset evidence is not ASCII") from exc
    seen = set()
    for line in offsets.splitlines():
        pieces = line.split()
        if len(pieces) != 3 or pieces[0] not in {"monotonic", "boottime"}:
            raise DeadlineError("time namespace offset evidence is malformed")
        try:
            seconds, nanoseconds = int(pieces[1]), int(pieces[2])
        except ValueError as exc:
            raise DeadlineError("time namespace offset evidence is malformed") from exc
        if seconds != 0 or nanoseconds != 0:
            raise DeadlineError("nonzero time namespace offsets are not reviewed")
        seen.add(pieces[0])
    if len(offsets.splitlines()) != 2 or seen != {"monotonic", "boottime"}:
        raise DeadlineError("monotonic and boottime namespace offsets are not both present")
    import base64

    return {
        "self_ns_time": own,
        "self_ns_time_for_children": children,
        "timens_offsets": offsets,
        "timens_offsets_raw_base64": base64.b64encode(offsets_raw).decode("ascii"),
    }


def conservative_deadlines(
    *,
    start_ticks: int,
    ticks_per_second: int,
    mono_before_ns: int,
    boot_sample_ns: int,
    mono_after_ns: int,
    root_seconds: int,
    cleanup_seconds: int,
) -> dict[str, int]:
    """Compute deadlines from an earlier bound on the kernel process start."""
    values = (
        start_ticks,
        ticks_per_second,
        mono_before_ns,
        boot_sample_ns,
        mono_after_ns,
        root_seconds,
        cleanup_seconds,
    )
    if any(type(value) is not int or value < 0 for value in values):
        raise DeadlineError("deadline inputs must be finite nonnegative integers")
    if (
        start_ticks <= 0
        or ticks_per_second <= 0
        or mono_before_ns <= 0
        or mono_after_ns < mono_before_ns
        or boot_sample_ns < mono_before_ns
        or root_seconds <= 0
        or cleanup_seconds <= 0
        or cleanup_seconds >= root_seconds
    ):
        raise DeadlineError("clock samples or deadline allocation are inconsistent")
    start_boot_lower_ns = start_ticks * 1_000_000_000 // ticks_per_second
    boot_minus_mono_upper_ns = boot_sample_ns - mono_before_ns
    root_start_mono_lower_ns = start_boot_lower_ns - boot_minus_mono_upper_ns
    if root_start_mono_lower_ns <= 0:
        raise DeadlineError("derived root start lower bound is not positive")
    hard = root_start_mono_lower_ns + root_seconds * 1_000_000_000
    soft = hard - cleanup_seconds * 1_000_000_000
    return {
        "start_ticks": start_ticks,
        "ticks_per_second": ticks_per_second,
        "start_boot_lower_ns": start_boot_lower_ns,
        "mono_before_ns": mono_before_ns,
        "boot_sample_ns": boot_sample_ns,
        "mono_after_ns": mono_after_ns,
        "boot_minus_mono_upper_ns": boot_minus_mono_upper_ns,
        "root_start_mono_lower_ns": root_start_mono_lower_ns,
        "hard_deadline_mono_ns": hard,
        "soft_deadline_mono_ns": soft,
    }


def capture_root_deadline_bounds(root_seconds: int, cleanup_seconds: int) -> dict[str, Any]:
    """Capture procfs/clock evidence in the specified M1, B, M2 order."""
    namespace_before = _time_namespace_evidence()
    stat_bytes = Path("/proc/self/stat").read_bytes()
    start_ticks = _proc_start_ticks(stat_bytes)
    try:
        hz = int(os.sysconf("SC_CLK_TCK"))
    except (OSError, ValueError) as exc:
        raise DeadlineError("SC_CLK_TCK is unavailable") from exc
    m1 = _checked_ns(CLOCK_MONOTONIC)
    boot = _checked_ns(CLOCK_BOOTTIME)
    m2 = _checked_ns(CLOCK_MONOTONIC)
    namespace_after = _time_namespace_evidence()
    if namespace_before != namespace_after:
        raise DeadlineError("time namespace evidence changed across clock samples")
    bounds = conservative_deadlines(
        start_ticks=start_ticks,
        ticks_per_second=hz,
        mono_before_ns=m1,
        boot_sample_ns=boot,
        mono_after_ns=m2,
        root_seconds=root_seconds,
        cleanup_seconds=cleanup_seconds,
    )
    import base64
    import hashlib

    bounds["namespace_before"] = namespace_before
    bounds["namespace_after_clock_samples"] = namespace_after
    bounds["proc_stat_sha256"] = hashlib.sha256(stat_bytes).hexdigest()
    bounds["proc_stat_raw_base64"] = base64.b64encode(stat_bytes).decode("ascii")
    return bounds


def _timespec(ns: int) -> Timespec:
    if ns < 0:
        raise DeadlineError("negative absolute timer value")
    return Timespec(ns // 1_000_000_000, ns % 1_000_000_000)


def _remaining_ns(spec: Itimerspec) -> int:
    sec = int(spec.it_value.tv_sec)
    nsec = int(spec.it_value.tv_nsec)
    if sec < 0 or not 0 <= nsec < 1_000_000_000:
        raise DeadlineError("timer_gettime returned an invalid timespec")
    return sec * 1_000_000_000 + nsec


def parse_proc_timers(raw: bytes) -> dict[int, dict[str, Any]]:
    """Parse complete ID-bound four-line /proc/<pid>/timers records."""
    try:
        lines = [line.strip() for line in raw.decode("ascii").splitlines() if line.strip()]
    except UnicodeDecodeError as exc:
        raise DeadlineError("proc timer inventory is not ASCII") from exc
    if len(lines) % 4:
        raise DeadlineError("proc timer inventory has a partial four-line record")
    records: dict[int, dict[str, Any]] = {}
    for index in range(0, len(lines), 4):
        id_match = re.fullmatch(r"ID:\s+([0-9]+)", lines[index])
        signal_match = re.fullmatch(r"signal:\s+([0-9]+)/([0-9a-fA-F]+)", lines[index + 1])
        notify_match = re.fullmatch(r"notify:\s+signal/pid\.([0-9]+)", lines[index + 2])
        clock_match = re.fullmatch(r"ClockID:\s+(-?[0-9]+)", lines[index + 3])
        if not all((id_match, signal_match, notify_match, clock_match)):
            raise DeadlineError("proc timer inventory contains a malformed record")
        timer_id = int(id_match.group(1))
        if timer_id in records:
            raise DeadlineError("proc timer inventory contains a duplicate kernel timer ID")
        records[timer_id] = {
            "kernel_timer_id": timer_id,
            "signal": int(signal_match.group(1)),
            "sigev_value_hex": signal_match.group(2).lower(),
            "sigev_value_zero": int(signal_match.group(2), 16) == 0,
            "notify": "signal",
            "notify_pid": int(notify_match.group(1)),
            "clock_id": int(clock_match.group(1)),
            "raw_lines": lines[index : index + 4],
        }
    return records


def verify_new_proc_timer(before: bytes, after: bytes, *, root_pid: int) -> dict[str, Any]:
    """Require exactly one new SIGKILL/signal-to-root/monotonic timer record."""
    old = parse_proc_timers(before)
    new = parse_proc_timers(after)
    if any(timer_id not in new or new[timer_id] != record for timer_id, record in old.items()):
        raise DeadlineError("preexisting proc timer records changed during root timer creation")
    added_ids = set(new) - set(old)
    if len(added_ids) != 1:
        raise DeadlineError("proc timer inventory did not gain exactly one record")
    record = new[next(iter(added_ids))]
    if (
        record["signal"] != SIGKILL
        or record["notify"] != "signal"
        or record["notify_pid"] != root_pid
        or record["clock_id"] != CLOCK_MONOTONIC
        or not record["sigev_value_zero"]
    ):
        raise DeadlineError("new proc timer record differs from SIGKILL/CLOCK_MONOTONIC/root contract")
    return record


def arm_root_deadline(root_seconds: int = 900, cleanup_seconds: int = 60) -> dict[str, Any]:
    """Arm one absolute SIGKILL timer; successful return leaves it permanently armed."""
    global _ROOT_TIMER
    if _ROOT_TIMER is not None:
        raise DeadlineError("root deadline timer has already been created")
    evidence = capture_root_deadline_bounds(root_seconds, cleanup_seconds)
    hard_deadline = evidence["hard_deadline_mono_ns"]
    libc = _libc()

    before = Path("/proc/self/timers").read_bytes()
    event = Sigevent()
    event.sigev_value.sival_int = 0
    event.sigev_signo = SIGKILL
    event.sigev_notify = SIGEV_SIGNAL
    timer = TimerT()
    ctypes.set_errno(0)
    if libc.timer_create(CLOCK_MONOTONIC, ctypes.byref(event), ctypes.byref(timer)) != 0:
        code = ctypes.get_errno()
        raise DeadlineError(f"timer_create failed with errno {code}")

    setting = Itimerspec()
    setting.it_interval = Timespec(0, 0)
    setting.it_value = _timespec(hard_deadline)
    ctypes.set_errno(0)
    if libc.timer_settime(timer, TIMER_ABSTIME, ctypes.byref(setting), None) != 0:
        code = ctypes.get_errno()
        raise DeadlineError(f"absolute timer_settime failed with errno {code}")

    # Retain the opaque libc handle. It is never passed to timer_delete and is
    # never treated as a kernel timer id.
    _ROOT_TIMER = timer
    capture_before = _checked_ns(CLOCK_MONOTONIC)
    remaining = Itimerspec()
    ctypes.set_errno(0)
    if libc.timer_gettime(timer, ctypes.byref(remaining)) != 0:
        code = ctypes.get_errno()
        raise DeadlineError(f"timer_gettime readback failed with errno {code}")
    capture_after = _checked_ns(CLOCK_MONOTONIC)
    remaining_ns = _remaining_ns(remaining)
    lower = hard_deadline - capture_after
    upper = hard_deadline - capture_before
    if lower <= 0 or remaining_ns <= 0:
        os.kill(os.getpid(), signal.SIGKILL)
        raise DeadlineError("absolute root deadline was already reached while arming")
    if remaining_ns < lower - 5_000_000 or remaining_ns > upper + 5_000_000:
        raise DeadlineError("timer remaining-time readback is inconsistent with absolute deadline")
    if int(remaining.it_interval.tv_sec) != 0 or int(remaining.it_interval.tv_nsec) != 0:
        raise DeadlineError("root timer is not one-shot")

    after = Path("/proc/self/timers").read_bytes()
    namespace_after_arm = _time_namespace_evidence()
    if namespace_after_arm != evidence["namespace_before"]:
        raise DeadlineError("time namespace evidence changed after timer arming")
    timer_record = verify_new_proc_timer(before, after, root_pid=os.getpid())
    import base64

    evidence.update(
        {
            "timer_clock_id": CLOCK_MONOTONIC,
            "timer_signal": SIGKILL,
            "timer_notify": SIGEV_SIGNAL,
            "timer_flags": TIMER_ABSTIME,
            "timer_interval_ns": 0,
            "timer_absolute_value": {
                "seconds": hard_deadline // 1_000_000_000,
                "nanoseconds": hard_deadline % 1_000_000_000,
            },
            "timer_remaining_ns": remaining_ns,
            "timer_remaining_capture_before_ns": capture_before,
            "timer_remaining_capture_after_ns": capture_after,
            "proc_self_timers_before_base64": base64.b64encode(before).decode("ascii"),
            "proc_self_timers_after_base64": base64.b64encode(after).decode("ascii"),
            "new_timer_proc_record": timer_record,
            "namespace_after_timer_arm": namespace_after_arm,
            "timer_handle_is_opaque": True,
        }
    )
    return evidence


def _getppid() -> int:
    return int(_libc().getppid())


def _prctl(option: int, arg2: int = 0, arg3: int = 0, arg4: int = 0, arg5: int = 0) -> int:
    ctypes.set_errno(0)
    result = _libc().prctl(
        option,
        ctypes.c_ulong(arg2),
        ctypes.c_ulong(arg3),
        ctypes.c_ulong(arg4),
        ctypes.c_ulong(arg5),
    )
    if result != 0:
        code = ctypes.get_errno()
        raise DeadlineError(f"prctl({option}) failed with errno {code}")
    return result


def verify_parent_death(expected_parent_pid: int, *, install: bool = False) -> dict[str, int]:
    """Set/read back PDEATHSIG with expected-parent checks on both sides."""
    if type(expected_parent_pid) is not int or expected_parent_pid <= 0:
        raise DeadlineError("expected parent PID is invalid")
    before = _getppid()
    if before != expected_parent_pid:
        raise DeadlineError("creating parent PID changed before PDEATHSIG operation")
    readback = ctypes.c_int(0)
    if install:
        _prctl(PR_SET_PDEATHSIG, SIGKILL)
    address = ctypes.addressof(readback)
    _prctl(PR_GET_PDEATHSIG, address)
    after = _getppid()
    if after != expected_parent_pid or readback.value != SIGKILL:
        raise DeadlineError("PDEATHSIG readback or parent race check failed")
    return {
        "expected_parent_pid": expected_parent_pid,
        "parent_pid_before": before,
        "parent_pid_after": after,
        "pdeath_signal": int(readback.value),
    }


def make_pdeath_preexec(expected_parent_pid: int):
    """Build the minimal, preallocated child-side PDEATHSIG closure for Popen."""
    require_supported_abi()
    libc = _libc()
    getppid_fn = libc.getppid
    prctl_fn = libc.prctl
    readback = ctypes.c_int(0)
    address_arg = ctypes.c_ulong(ctypes.addressof(readback))
    zero = ctypes.c_ulong(0)
    signal_arg = ctypes.c_ulong(SIGKILL)

    def install_in_child() -> None:
        if getppid_fn() != expected_parent_pid:
            os._exit(125)
        if prctl_fn(PR_SET_PDEATHSIG, signal_arg, zero, zero, zero) != 0:
            os._exit(126)
        if getppid_fn() != expected_parent_pid:
            os._exit(127)
        if prctl_fn(PR_GET_PDEATHSIG, address_arg, zero, zero, zero) != 0:
            os._exit(128)
        if getppid_fn() != expected_parent_pid or readback.value != SIGKILL:
            os._exit(129)

    return install_in_child
