"""Regression tests for fail-closed Git checks in pin reconciliation."""

import hashlib
import json
import subprocess
import sys

from scripts import check_pin_reconciliation as checker


def _git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        check=True,
    )


def _repository(root, files):
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    _git(root, "config", "user.name", "Pin checker test")
    _git(root, "config", "user.email", "pin-checker@example.invalid")
    for path, content in files.items():
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    _git(root, "add", "--all")
    _git(root, "commit", "--quiet", "-m", "fixture")
    return _git(root, "rev-parse", "HEAD").stdout.decode().strip()


def _record(base_commit, history=()):
    return {
        "base_commit": base_commit,
        "summary": {
            "repository_path_status": {
                "resolved_git_history": len(history),
                "resolved_local_run_snapshot": 0,
                "unresolved": 0,
            }
        },
        "resolved_by_git_history": list(history),
        "resolved_by_local_run_snapshot": [],
        "unresolved": [],
    }


def _run(root, monkeypatch, record):
    (root / "record.json").write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(checker, "REPO", root)
    monkeypatch.setattr(
        sys, "argv", ["check_pin_reconciliation", "--record", "record.json"]
    )
    return checker.main()


def test_missing_base_commit_fails_closed(tmp_path, monkeypatch, capsys):
    head = _repository(tmp_path, {"manifest.json": b"{}\n"})
    missing = "f" * 40
    assert missing != head

    status = _run(tmp_path, monkeypatch, _record(missing))

    assert status == 1
    assert "manifest diff command failed" in capsys.readouterr().out


def test_unavailable_git_blob_fails_closed(tmp_path, monkeypatch, capsys):
    head = _repository(tmp_path, {"manifest.json": b"pinned bytes\n"})
    history = {
        "matching_commit": head,
        "path": "manifest.json",
        "pinned_sha256": hashlib.sha256(b"").hexdigest(),
        "manifest": "manifest.json",
    }
    real_git = checker.git

    def git_without_blob(*args, **kwargs):
        if args[:2] == ("cat-file", "blob"):
            return subprocess.CompletedProcess(args, 128, b"", b"missing blob")
        return real_git(*args, **kwargs)

    monkeypatch.setattr(checker, "git", git_without_blob)
    status = _run(tmp_path, monkeypatch, _record(head, [history]))

    assert status == 1
    assert "historical blob unavailable" in capsys.readouterr().out


def test_git_diff_failure_fails_closed(tmp_path, monkeypatch, capsys):
    head = _repository(tmp_path, {"manifest.json": b"{}\n"})
    real_git = checker.git

    def git_with_diff_failure(*args, **kwargs):
        if args and args[0] == "diff":
            return subprocess.CompletedProcess(args, 1, b"", b"diff failed")
        return real_git(*args, **kwargs)

    monkeypatch.setattr(checker, "git", git_with_diff_failure)
    status = _run(tmp_path, monkeypatch, _record(head))

    assert status == 1
    assert "manifest diff command failed" in capsys.readouterr().out


def test_wrong_historical_path_fails_closed(tmp_path, monkeypatch, capsys):
    head = _repository(tmp_path, {"manifest.json": b"{}\n"})
    history = {
        "matching_commit": head,
        "path": "missing.json",
        "pinned_sha256": "0" * 64,
        "manifest": "manifest.json",
    }

    status = _run(tmp_path, monkeypatch, _record(head, [history]))

    assert status == 1
    assert "historical blob unavailable" in capsys.readouterr().out


def test_valid_history_and_manifest_base_pass(tmp_path, monkeypatch, capsys):
    head = _repository(tmp_path, {"manifest.json": b"{}\n"})
    history = {
        "matching_commit": head,
        "path": "manifest.json",
        "pinned_sha256": hashlib.sha256(b"{}\n").hexdigest(),
        "manifest": "manifest.json",
    }

    status = _run(tmp_path, monkeypatch, _record(head, [history]))

    assert status == 0
    assert "PASS" in capsys.readouterr().out
