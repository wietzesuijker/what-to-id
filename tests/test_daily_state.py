"""State release failures must not reset the historical served log."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def daily(tmp_path):
    binary = tmp_path / "gh"
    binary.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['CALL_LOG'], 'a') as out:\n"
        "    out.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1] == 'api':\n"
        "    status = os.environ['HTTP_STATUS']\n"
        "    print('HTTP/2.0 ' + status)\n"
        "    sys.exit(0 if status == '200' else 1)\n"
        "if sys.argv[1:3] == ['release', 'view']:\n"
        "    print(os.environ.get('ASSETS', ''))\n"
    )
    binary.chmod(0o755)
    state = tmp_path / "state"
    (state / "days").mkdir(parents=True)
    for filename in (
        "pool.parquet",
        "served.parquet",
        "days/pool-2026-09-28.parquet",
        "days/build-2026-09-28.json",
    ):
        (state / filename).write_text("existing state")
    log = tmp_path / "calls.jsonl"

    def invoke(command, status, assets="", repo=None):
        env = {
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "STATE_DIR": str(state),
            "TODAY": "2026-09-28",
            "STATE_TAG": "pool-state",
            "CALL_LOG": str(log),
            "HTTP_STATUS": status,
            "ASSETS": assets,
        }
        env.pop("STATE_REPO", None)
        env.pop("GITHUB_REPOSITORY", None)
        if repo:
            env["STATE_REPO"] = repo
        result = subprocess.run(
            ["bash", "scripts/daily.sh", command],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            text=True,
            capture_output=True,
            timeout=10,
        )
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        return result, calls

    return invoke, state


@pytest.mark.parametrize("command", ["fetch", "save"])
@pytest.mark.parametrize("status", ["401", "503", "connection-failed"])
def test_release_errors_fail_closed(daily, command, status):
    invoke, state = daily
    result, calls = invoke(command, status)
    assert result.returncode != 0
    assert "refusing to assume" in result.stderr
    assert len(calls) == 1 and calls[0][0] == "api"
    assert (state / "served.parquet").read_text() == "existing state"


def test_confirmed_missing_release_allows_first_fetch(daily):
    invoke, _ = daily
    result, calls = invoke("fetch", "404")
    assert result.returncode == 0
    assert "no pool-state release yet" in result.stdout
    assert calls[0][-1] == "repos/PollockLab/what-to-id/releases/tags/pool-state"


def test_missing_release_created_in_explicit_state_repository(daily):
    invoke, _ = daily
    result, calls = invoke("save", "404", repo="owner/state")
    assert result.returncode == 0, result.stderr
    assert calls[0][-1] == "repos/owner/state/releases/tags/pool-state"
    assert [call[1] for call in calls[1:]] == ["create", "upload"]
    assert all(call[call.index("--repo") + 1] == "owner/state" for call in calls[1:])


def test_existing_history_without_served_log_fails(daily):
    invoke, state = daily
    (state / "served.parquet").unlink()
    result, calls = invoke("fetch", "200", assets="build-2026-09-27.json")
    assert result.returncode != 0
    assert "earlier builds but no served log" in result.stderr
    assert calls[1][calls[1].index("--repo") + 1] == "PollockLab/what-to-id"


def test_bundle_eligibility_is_fetched_and_saved(daily):
    invoke, state = daily
    eligibility = state / "bundle-eligibility.json"
    eligibility.write_text('{"closed_ids": [8, 10]}')
    result, calls = invoke("save", "200")
    assert result.returncode == 0, result.stderr
    upload = next(call for call in calls if call[:2] == ["release", "upload"])
    assert str(eligibility) in upload
    result, calls = invoke("fetch", "200", assets="bundle-eligibility.json")
    assert result.returncode == 0, result.stderr
    downloads = [call for call in calls if call[:2] == ["release", "download"]]
    assert any(call[call.index("-p") + 1] == eligibility.name for call in downloads)


def test_build_stops_on_edited_cell_scores(tmp_path):
    from what_to_id.manifest import WHERE_TO_BLITZ_REF

    wtb = tmp_path / "wtb"
    wtb.mkdir()
    (wtb / "provenance.json").write_text("{}")
    git = tmp_path / "bin" / "git"
    git.parent.mkdir()
    # HEAD matches the pin, but the score files differ from it.
    sha = WHERE_TO_BLITZ_REF.split("@")[1]
    git.write_text(f'#!/bin/sh\ncase "$*" in *diff*) exit 1;; esac\nprintf "%s\\n" {sha}\n')
    git.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(git.parent) + os.pathsep + os.environ["PATH"],
        "PYTHON": sys.executable,
        "WHAT_TO_ID_KEY": "00" * 32,
        "BLITZ_D1": "2026-11-01",
        "WTB_DIR": str(wtb),
        "STATE_DIR": str(tmp_path / "state"),
        "OUT_DIR": str(tmp_path / "out"),
    }
    root = Path(__file__).resolve().parents[1]
    run = subprocess.run(
        ["bash", "scripts/daily.sh", "build"], cwd=root, env=env, text=True, capture_output=True
    )
    assert run.returncode == 1
    assert "local changes to the cell scores" in run.stderr
