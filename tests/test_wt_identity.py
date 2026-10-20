from natvqvae.sequence import WT_SHA256, WT_SEQUENCE, canonical_hash, wt_sequence


def test_length_and_alphabet():
    seq = wt_sequence()
    assert len(seq) == 341
    assert set(seq) <= set("ACDEFGHIKLMNPQRSTVWY")


def test_frozen_hash():
    assert canonical_hash(WT_SEQUENCE) == WT_SHA256
    assert WT_SHA256.startswith("8804bca1")
