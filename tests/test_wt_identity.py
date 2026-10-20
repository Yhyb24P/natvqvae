from natvqvae.sequence import WT_SHA256, WT_SEQUENCE, canonical_hash, wt_sequence


def test_length_and_alphabet():
    seq = wt_sequence()
    assert len(seq) == 341
    assert set(seq) <= set("ACDEFGHIKLMNPQRSTVWY")


def test_frozen_hash():
    assert canonical_hash(WT_SEQUENCE) == WT_SHA256
    assert WT_SHA256.startswith("8804bca1")


def test_matches_source_contract_if_present():
    """The canonical constant must agree with the source target contract."""
    import pytest

    from natvqvae.intake import load_manifest, resolve, source_root

    root = source_root(load_manifest())
    if not root.exists():
        pytest.skip("source repository not available")
    path = resolve("wt_target_contract")
    text = path.read_text(encoding="utf-8")
    marker = "amino_acids:"
    line = next((ln for ln in text.splitlines() if ln.strip().startswith(marker)), None)
    assert line is not None, "amino_acids not found in target contract"
    source_seq = line.split(marker, 1)[1].strip().strip('"').strip("'")
    assert source_seq == WT_SEQUENCE
    assert canonical_hash(source_seq) == WT_SHA256
