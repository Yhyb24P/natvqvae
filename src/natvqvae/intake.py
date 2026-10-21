"""Hash-verified, read-only access to input data.

No file in the source repository is ever opened for writing.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "data" / "manifest.json"


@dataclass(frozen=True)
class VerifyResult:
    name: str
    ok: bool
    detail: str
    path: Path | None = None


def load_manifest(path: str | Path | None = None) -> dict:
    p = Path(path) if path else MANIFEST_PATH
    with p.open(encoding="utf-8") as fh:
        manifest = json.load(fh)
    if "entries" not in manifest or not manifest["entries"]:
        raise ValueError(f"{p}: manifest has no entries")
    for entry in manifest["entries"]:
        for key in ("name", "path", "sha256", "role", "link"):
            if key not in entry:
                raise ValueError(f"{p}: entry {entry.get('name')!r} missing {key!r}")
    return manifest


def source_root(manifest: dict) -> Path:
    env_key = manifest.get("source_root_env_override")
    if env_key and os.environ.get(env_key):
        return Path(os.environ[env_key]).expanduser()
    return Path(manifest["source_root_default"]).expanduser()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_entry(entry: dict, root: Path) -> VerifyResult:
    path = root / entry["path"]
    if not path.exists():
        return VerifyResult(entry["name"], False, f"missing: {path}", path)
    actual = sha256_file(path)
    if actual != entry["sha256"]:
        return VerifyResult(entry["name"], False,
                            f"hash mismatch: expected {entry['sha256'][:12]}..., got {actual[:12]}...", path)
    size = path.stat().st_size
    return VerifyResult(entry["name"], True, f"{size} bytes, sha256 ok", path)


def verify_all(manifest: dict, root: Path) -> list[VerifyResult]:
    return [verify_entry(e, root) for e in manifest["entries"]]


def resolve(name: str, root: Path | None = None) -> Path:
    """Return the verified absolute path of a manifest entry."""
    manifest = load_manifest()
    entry = next((e for e in manifest["entries"] if e["name"] == name), None)
    if entry is None:
        raise KeyError(f"unknown input {name!r}")
    result = verify_entry(entry, root or source_root(manifest))
    if not result.ok:
        raise IOError(result.detail)
    return result.path
