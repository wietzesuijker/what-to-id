"""Installed LabelFirst provenance without importing its optional numerical stack."""

from __future__ import annotations

import json
import subprocess
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlparse


def labelfirst_provenance() -> dict[str, object] | None:
    """Actual installed version and source revision; unknown fields remain null.

    Git installations record the resolved commit in PEP 610 direct_url.json.
    Editable local checkouts additionally report whether their tree is dirty.
    A wheel without source metadata has a version but no verifiable revision.
    """
    try:
        dist = metadata.distribution("labelfirst")
    except metadata.PackageNotFoundError:
        return None
    result: dict[str, object] = {"version": dist.version, "commit": None, "dirty": None}
    raw = dist.read_text("direct_url.json")
    if not raw:
        return result
    try:
        source = json.loads(raw)
    except (ValueError, TypeError):
        return result
    vcs = source.get("vcs_info", {})
    if vcs.get("vcs") == "git":
        result["commit"] = vcs.get("commit_id")
    elif source.get("dir_info", {}).get("editable"):
        url = urlparse(source.get("url", ""))
        if url.scheme == "file" and url.netloc in ("", "localhost"):
            path = Path(unquote(url.path))
            try:
                commit = (
                    subprocess.check_output(
                        ["git", "-C", str(path), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                    )
                    .decode()
                    .strip()
                )
                dirty = subprocess.check_output(
                    ["git", "-C", str(path), "status", "--porcelain"], stderr=subprocess.DEVNULL
                )
            except (OSError, subprocess.CalledProcessError):
                return result
            result.update(commit=commit, dirty=bool(dirty))
    return result
