from pathlib import Path

import pytest

from natvqvae.intake import load_manifest, sha256_file, source_root, verify_all, verify_entry


def test_manifest_structure():
    manifest = load_manifest()
    names = [e["name"] for e in manifest["entries"]]
    links = [e["link"] for e in manifest["entries"]]
    assert len(names) == len(set(names))
    assert len(links) == len(set(links))
    assert len(names) >= 8
    for entry in manifest["entries"]:
        assert len(entry["sha256"]) == 64


def test_all_inputs_verify_against_source():
    manifest = load_manifest()
    root = source_root(manifest)
    if not root.exists():
        pytest.skip("source repository not available")
    results = verify_all(manifest, root)
    assert all(r.ok for r in results), [r.detail for r in results if not r.ok]


def test_family_fasta_record_count():
    manifest = load_manifest()
    root = source_root(manifest)
    if not root.exists():
        pytest.skip("source repository not available")
    entry = next(e for e in manifest["entries"] if e["name"] == "family_population")
    path = root / entry["path"]
    records = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.startswith(">"))
    assert records == entry["expected_records"] == 2430


def test_hash_mismatch_fails_closed(tmp_path: Path):
    target = tmp_path / "x.txt"
    target.write_text("not the expected content")
    entry = {"name": "bogus", "path": "x.txt", "sha256": "0" * 64, "role": "test", "link": "x.txt"}
    result = verify_entry(entry, tmp_path)
    assert not result.ok
    assert "hash mismatch" in result.detail
    assert sha256_file(target) != entry["sha256"]
