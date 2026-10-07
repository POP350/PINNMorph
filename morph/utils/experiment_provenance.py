"""Portable source identity and guards against mixing experiment versions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def source_identity(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    files = {}
    for folder in ("morph", "scripts", "third_party"):
        for path in sorted((root / folder).rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".sh"}:
                continue
            # Hash normalized newlines so transferring Windows sources to Linux
            # does not manufacture a different experiment version.
            content = path.read_bytes().replace(b"\r\n", b"\n")
            files[path.relative_to(root).as_posix()] = hashlib.sha256(content).hexdigest()
    req = root / "requirements.txt"
    if req.is_file():
        files["requirements.txt"] = hashlib.sha256(req.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "files": files}


def assert_run_compatible(previous: Mapping[str, Any], config_digest: str, source_digest: str) -> None:
    if not previous:
        return
    saved = previous.get("hashes") or {}
    if saved.get("experiment_config") != config_digest or saved.get("source_tree") != source_digest:
        raise ValueError(
            "Existing run has different or missing config/source fingerprints. "
            "Use a fresh results root; do not resume or silently reuse this run."
        )


def assert_manifest_compatible(previous: Mapping[str, Any], identity: Mapping[str, Any]) -> None:
    if previous and previous.get("experiment_identity") != identity:
        raise ValueError(
            "This results root belongs to a different or legacy experiment. "
            "Choose a new --results-root; its manifest and results were preserved."
        )
