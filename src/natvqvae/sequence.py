"""Canonical NatV sequence and identity checks."""
from __future__ import annotations

import hashlib

WT_SEQUENCE = (
    "MSHAIFIGRFRPFHNGHLSAITQAFDALDLDKMTILIGSSNRHRSVKNPFVFEEVRDMMGVALPDHIRSKVRFVPLGDYAKDDVWQ"
    "SNVRSRARGATHIVGYDKDESSYYLKLFPELKLFQPEPVKMYNKVISATDFRELYFSEILLNHPVMSGLIPKETMMFLDNWSKTEFF"
    "TEMKAEYDSSVREIEKFKDYPYQGHLNIACADNVVTCAGHVLLVERKFNPGKGCLALPGGHKHEKETFLDAAIRELQEETNIKVPE"
    "KVLRGSLVGEKMFDNPNRSYPHTRITMAYHLKVHPNPDNTFPKVKPADDAVSAKWYPLSEVRDMQERLYDDHYQIIQYFTGI"
)
WT_SHA256 = "8804bca153eceb5063f1d8a879d4daeff327964d5d1c9cd693d217fb4f86a7e8"
AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def canonical_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.strip().upper().replace("-", "").encode()).hexdigest()


def wt_sequence() -> str:
    if len(WT_SEQUENCE) != 341:
        raise ValueError(f"canonical WT has length {len(WT_SEQUENCE)}, expected 341")
    if canonical_hash(WT_SEQUENCE) != WT_SHA256:
        raise ValueError("canonical WT sequence does not match the frozen sha256")
    if not set(WT_SEQUENCE) <= AMINO_ACIDS:
        raise ValueError("canonical WT contains non-standard residues")
    return WT_SEQUENCE
