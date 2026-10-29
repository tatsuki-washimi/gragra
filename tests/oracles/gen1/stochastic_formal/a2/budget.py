"""Persistent A2 subprocess accounting; fail closed on abandoned reservations."""

import contextlib
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import time
import uuid
from pathlib import Path


class BudgetError(RuntimeError):
    pass


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def transaction(self, operation):
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = (
                json.loads(self.path.read_text())
                if self.path.exists()
                else {
                    "schema_version": 1,
                    "entries": [],
                    "production_fix_revisions": [],
                }
            )
            result = operation(data)
            temp = self.path.with_suffix(".tmp")
            with temp.open("w") as stream:
                json.dump(data, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, self.path)
            return result

    @staticmethod
    def elapsed(data):
        now = time.time()
        children = sum(
            e.get("elapsed_seconds", max(0, now - e["started"]))
            for e in data["entries"]
        )
        # A controller is a separate role, never an available child slot. Its
        # whole-wall bound covers nonwait conservatively; no pure wait is claimed.
        controllers = sum(
            c.get(
                "external_span_upper_seconds",
                max(c["wall_bound_seconds"], time.monotonic() - c["started_monotonic"]),
            )
            for c in data.get("controllers", [])
        )
        return children + controllers

    def reserve(self, kind, command):
        if kind not in (
            "formal",
            "calibration",
            "repair_validation",
            "tests",
            "preflight",
        ):
            raise BudgetError("unknown campaign subprocess kind")

        def operation(data):
            running = [e for e in data["entries"] if "elapsed_seconds" not in e]
            if (
                any(e["pid"] is None for e in running)
                or len(running) >= 2
                or self.elapsed(data) >= 21600
            ):
                raise BudgetError(
                    "pending reservation/concurrency/aggregate resource exhausted"
                )
            if (
                kind == "formal"
                and sum(effective_kind(e, data) == "formal" for e in data["entries"])
                >= 4
            ):
                raise BudgetError("four formal attempts exhausted")
            ident = len(data["entries"])
            data["entries"].append(
                {
                    "id": ident,
                    "kind": kind,
                    "command": command,
                    "started": time.time(),
                    "pid": None,
                    "parent_pid": os.getpid(),
                    "status": "running",
                }
            )
            return ident

        return self.transaction(operation)

    def update(self, ident, **fields):
        def operation(data):
            entry = data["entries"][ident]
            if "elapsed_seconds" in entry:
                raise BudgetError("completed entry immutable")
            if any(
                key in fields and fields[key] != entry[key]
                for key in ("id", "kind", "command", "started", "parent_pid")
                if key in entry
            ):
                raise BudgetError("reserved identity/classification fields immutable")
            entry.update(fields)

        return self.transaction(operation)

    def run(self, command, *, kind, stdout, stderr, timeout=3600, reservation=None):
        if timeout > 3600 or timeout <= 0:
            raise BudgetError("timeout must be positive and <=3600 seconds")
        ident = self.reserve(kind, command) if reservation is None else reservation
        if reservation is not None:

            def verify(data):
                entry = data["entries"][ident]
                if (
                    entry["kind"] != kind
                    or entry["command"] != command
                    or entry["pid"] is not None
                    or "elapsed_seconds" in entry
                ):
                    raise BudgetError("exact unused preallocated reservation required")

            self.transaction(verify)
        start = time.time()
        process = None
        status = "exception"
        code = None
        try:
            with open(stdout, "xb") as out, open(stderr, "xb") as err:
                # Never deliver the controller alarm in the Popen-return/PID
                # assignment gap: finally must know the actual group to stop.
                previous_handlers = {
                    s: signal.getsignal(s) for s in (signal.SIGALRM, signal.SIGINT)
                }
                deferred = []

                def defer_interrupt(received, _frame):
                    deferred.append(received)

                for s in previous_handlers:
                    signal.signal(s, defer_interrupt)
                try:
                    process = subprocess.Popen(
                        command, stdout=out, stderr=err, start_new_session=True
                    )
                    self.update(ident, pid=process.pid)
                finally:
                    for s, previous in previous_handlers.items():
                        signal.signal(s, previous)
                    for s in deferred:
                        previous = previous_handlers[s]
                        if callable(previous):
                            previous(s, None)
                        elif previous == signal.SIG_DFL:
                            raise KeyboardInterrupt(
                                "interrupt deferred until actual child PID recorded"
                            )
                while process.poll() is None:
                    rss = 0
                    try:
                        for line in (
                            Path(f"/proc/{process.pid}/status").read_text().splitlines()
                        ):
                            if line.startswith("VmRSS:"):
                                rss = int(line.split()[1]) * 1024
                    except FileNotFoundError:
                        pass
                    aggregate = self.transaction(self.elapsed)
                    if (
                        rss > 4 * 1024**3
                        or time.time() - start >= timeout
                        or aggregate >= 21600
                    ):
                        status = "timeout" if rss <= 4 * 1024**3 else "rss_limit"
                        os.killpg(process.pid, signal.SIGKILL)
                        break
                    time.sleep(0.05)
                code = process.wait()
                if status == "exception":
                    status = "success" if code == 0 else "error"
        except BaseException as exc:
            status = (
                "interruption" if isinstance(exc, KeyboardInterrupt) else "exception"
            )
        finally:
            cleanup = True
            if process is not None:
                try:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    code = process.wait(timeout=10)
                    deadline = time.time() + 10
                    while live_group(process.pid) and time.time() < deadline:
                        time.sleep(0.05)
                    cleanup = not live_group(process.pid)
                except BaseException:
                    cleanup = False
            fields = {"status": status, "exit_code": code, "cleanup_confirmed": cleanup}
            if cleanup:
                fields["elapsed_seconds"] = time.time() - start
            else:
                status = fields["status"] = "pending_cleanup"
            self.update(ident, **fields)
        return {
            "ledger_id": ident,
            "status": status,
            "exit_code": code,
            "cleanup_confirmed": cleanup,
            "launch_state": "started" if process is not None else "not_started",
        }

    def record_repair_revision(self, commit):
        def operation(data):
            revisions = data["production_fix_revisions"]
            if commit in revisions:
                return
            if len(revisions) >= 2:
                raise BudgetError("two production repair revisions exhausted")
            revisions.append(commit)

        return self.transaction(operation)

    def recover(self, ident):
        """Preserve interrupted reservation; never claim an extant child stopped."""

        def operation(data):
            entry = data["entries"][ident]
            if "elapsed_seconds" in entry:
                raise BudgetError("completed entry immutable")
            if entry["pid"] is not None and live_group(entry["pid"]):
                raise BudgetError("cleanup unresolved: process remains present")
            entry.update(
                status="interruption",
                elapsed_seconds=max(0, time.time() - entry["started"]),
                exit_code=None,
                recovery=(
                    "confirmed_absent; elapsed conservatively charged through recovery"
                ),
            )

        return self.transaction(operation)


def process_identity(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def same_process(pid, identity):
    return identity is not None and process_identity(pid) == identity


class ControllerAccount:
    """Root reserves BEFORE direct spawn; full wall is an upper-bound charge.

    The active deadline defaults to 3600 seconds and may be shortened. Its
    reservation adds 100 seconds for stopped-child cleanup and final I/O.
    Only root's external observed span, after process absence, may replace
    that conservative reservation. An overrun keeps charging through recovery.
    """

    def __init__(self, ledger):
        self.ledger = ledger

    def reserve(
        self,
        command,
        *,
        started,
        started_monotonic=None,
        active_timeout_seconds=3600,
    ):
        if (
            not isinstance(active_timeout_seconds, int)
            or isinstance(active_timeout_seconds, bool)
            or not 1 <= active_timeout_seconds <= 3600
        ):
            raise BudgetError("active timeout must be an integer in 1..3600 seconds")
        wall_bound_seconds = active_timeout_seconds + 100
        now = time.time()
        monotonic_now = time.monotonic()
        if not math.isfinite(started) or not 0 <= now - started <= 10:
            raise BudgetError("fresh external controller start required")
        explicit_monotonic = started_monotonic is not None
        if started_monotonic is None:
            started_monotonic = monotonic_now - (now - started)
        if (
            not math.isfinite(started_monotonic)
            or not 0 <= monotonic_now - started_monotonic <= 10
        ):
            raise BudgetError("fresh external monotonic start required")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(v, str) for v in command)
        ):
            raise BudgetError("exact controller argv required")

        def operation(data):
            controllers = data.setdefault("controllers", [])
            if any(c["status"] != "completed" for c in controllers):
                raise BudgetError("pending controller blocks restart")
            if self.ledger.elapsed(data) + wall_bound_seconds >= 21600:
                raise BudgetError("controller whole-wall aggregate exhausted")
            token = len(controllers)
            controllers.append(
                {
                    "id": token,
                    "role": "controller",
                    "command": command,
                    "started": started,
                    "wall_bound_seconds": wall_bound_seconds,
                    "started_monotonic": started_monotonic,
                    "explicit_monotonic_origin": explicit_monotonic,
                    "deadline_seconds": active_timeout_seconds,
                    "pid": None,
                    "status": "reserved",
                    "waiting_seconds": None,
                    "waiting_interpretation": (
                        "unmeasured; nonwait <= externally bounded whole wall"
                    ),
                }
            )
            return token

        return self.ledger.transaction(operation)

    def claim(self, token, command, *, require_explicit_origin=False):
        def operation(data):
            c = data["controllers"][token]
            if require_explicit_origin and not c["explicit_monotonic_origin"]:
                raise BudgetError(
                    "formal requires root explicit pre-spawn monotonic origin"
                )
            if (
                c["status"] != "reserved"
                or c["pid"] is not None
                or c["command"] != command
            ):
                raise BudgetError("exact unclaimed controller reservation required")
            if any(
                e.get("pid") == os.getpid() and "elapsed_seconds" not in e
                for e in data["entries"]
            ):
                raise BudgetError("controller must not have an outer child reservation")
            if time.monotonic() >= c["started_monotonic"] + c["deadline_seconds"]:
                raise BudgetError("controller deadline exhausted")
            if self.ledger.elapsed(data) >= 21600:
                raise BudgetError("controller aggregate exhausted")
            c.update(
                status="running",
                pid=os.getpid(),
                process_identity=process_identity(os.getpid()),
            )
            return c["started_monotonic"] + c["deadline_seconds"]

        return self.ledger.transaction(operation)

    def outcome(self, token, status):
        def operation(data):
            c = data["controllers"][token]
            if c["status"] == "completed":
                raise BudgetError("completed controller immutable")
            if c["status"] != "running" or c["pid"] != os.getpid():
                raise BudgetError("only claimed controller may attest outcome")
            c.update(
                status="awaiting_external_span",
                outcome=status,
                outcome_recorded=time.time(),
                outcome_monotonic=time.monotonic(),
            )
            return dict(c)

        return self.ledger.transaction(operation)

    def finalize(self, token, pin):
        raw = Path(pin["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != pin["sha256"]:
            raise BudgetError("external controller receipt bytes mismatch")
        report = json.loads(raw)

        def operation(data):
            c = data["controllers"][token]
            if c["status"] == "completed":
                raise BudgetError("completed controller immutable")
            if (
                c["status"] != "awaiting_external_span"
                or c.get("outcome") == "pending_recovery"
            ):
                raise BudgetError(
                    "controller outcome/recovery unresolved; bound retained"
                )
            if c["pid"] is None or same_process(c["pid"], c["process_identity"]):
                raise BudgetError("controller remains alive or never claimed")
            descendants = {c["pid"]}
            owned = []
            for _ in range(len(data["entries"]) + 1):
                new = [
                    e
                    for e in data["entries"]
                    if e.get("parent_pid") in descendants and e not in owned
                ]
                if not new:
                    break
                owned.extend(new)
                descendants.update(e["pid"] for e in new if e["pid"] is not None)
            if any(
                "elapsed_seconds" not in e
                or (e["pid"] is not None and live_group(e["pid"]))
                for e in owned
            ):
                raise BudgetError("controller child cleanup unresolved; bound retained")
            finish = report.get("finished", float("nan"))
            finish_monotonic = report.get("finished_monotonic", float("nan"))
            span = finish_monotonic - c["started_monotonic"]
            if (
                report.get("schema_version") != 1
                or report.get("issuer") != "root"
                or report.get("decision") != "CONTROLLER_EXTERNAL_SPAN_ATTESTED"
                or report.get("controller_id") != token
                or report.get("ledger_path") != str(self.ledger.path.resolve())
                or report.get("command") != c["command"]
                or report.get("started") != c["started"]
                or report.get("started_monotonic") != c["started_monotonic"]
                or not math.isfinite(finish)
                or finish > time.time()
                or finish < c.get("outcome_recorded", c["started"])
                or not math.isfinite(finish_monotonic)
                or finish_monotonic > time.monotonic()
                or finish_monotonic < c.get("outcome_monotonic", c["started_monotonic"])
                or not 0 <= span <= c["wall_bound_seconds"]
            ):
                raise BudgetError("external span invalid/overrun; reservation retained")
            c.update(
                status="completed",
                external_span_upper_seconds=span,
                external_span_pin=pin,
            )

        return self.ledger.transaction(operation)


class BoundedGit:
    """Charge every validation Git child before launch, including refusals."""

    def __init__(self, ledger):
        self.ledger = Ledger(ledger)

    def execute(self, command, cwd=None):
        token = uuid.uuid4().hex
        stdout = self.ledger.path.parent / ("preflight-" + token + ".stdout")
        stderr = self.ledger.path.parent / ("preflight-" + token + ".stderr")
        if cwd is not None:
            command = [command[0], "-C", str(cwd), *command[1:]]
        result = self.ledger.run(
            command, kind="preflight", stdout=stdout, stderr=stderr
        )
        stdout.chmod(0o444)
        stderr.chmod(0o444)
        return result, stdout.read_bytes(), stderr.read_bytes()

    def check_output(self, command, cwd=None, text=False):
        result, output, error = self.execute(command, cwd)
        if result["exit_code"] != 0 or result["status"] != "success":
            raise subprocess.CalledProcessError(
                result["exit_code"], command, output=output, stderr=error
            )
        return output.decode() if text else output

    def run(self, command, cwd=None, check=False, capture_output=False):
        del capture_output
        result, output, error = self.execute(command, cwd)
        if check and (result["exit_code"] != 0 or result["status"] != "success"):
            raise subprocess.CalledProcessError(
                result["exit_code"], command, output=output, stderr=error
            )
        return subprocess.CompletedProcess(command, result["exit_code"], output, error)


def live_group(pgid):
    """Zombies are stopped; any runnable group member keeps cleanup pending."""
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid and fields[0] != "Z":
                return True
        except (FileNotFoundError, ProcessLookupError):
            continue
    return False


def effective_kind(entry, data):
    """Append-only corrections have no effect before independent/root admission."""
    corrections = [
        c
        for c in data.get("classification_corrections", [])
        if c["entry_id"] == entry["id"]
    ]
    if len(corrections) > 1:
        raise BudgetError("duplicate classification correction")
    if not corrections or "admission_pin" not in corrections[0]:
        return entry["kind"]
    correction = corrections[0]

    def actual(pin):
        raw = Path(pin["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != pin["sha256"]:
            raise BudgetError("classification review actual bytes mismatch")
        return json.loads(raw)

    proof = actual(correction["proof_pin"])
    original = json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()
    if (
        proof["original_entry_hashes"].get(str(entry["id"]))
        != hashlib.sha256(original).hexdigest()
    ):
        raise BudgetError("classification original entry mismatch")
    admission = actual(correction["admission_pin"])
    if (
        admission.get("decision") != "PASS_FOR_A2_RESOURCE_CLASSIFICATION_CORRECTION"
        or admission.get("family") != "A2"
        or admission.get("proof_sha256") != correction["proof_pin"]["sha256"]
        or set(admission.get("reviews", {})) != {"numerical", "provenance"}
    ):
        raise BudgetError("classification correction not independently admitted")
    identities = set()
    for role, pin in admission["reviews"].items():
        review = actual(pin)
        if (
            review.get("decision") != "PASS"
            or review.get("role") != role
            or review.get("family") != "A2"
            or review.get("scope") != "resource classification correction"
            or review.get("proof_sha256") != correction["proof_pin"]["sha256"]
            or not review.get("reviewer_id")
            or review["reviewer_id"] in identities
        ):
            raise BudgetError("classification independent review mismatch")
        identities.add(review["reviewer_id"])
    if correction["effective_kind"] != "tests" or entry["command"][1:] != [
        "-c",
        "import time; time.sleep(30)",
    ]:
        raise BudgetError("only pinned synthetic sleep tests may be corrected")
    return "tests"
