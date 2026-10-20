"""Canonical NatV sequence and substitution-only variant handling."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from . import regions

WT_SEQUENCE = (
    "MSHAIFIGRFRPFHNGHLSAITQAFDALDLDKMTILIGSSNRHRSVKNPFVFEEVRDMMGVALPDHIRSKVRFVPLGDYAKDDVWQ"
    "SNVRSRARGATHIVGYDKDESSYYLKLFPELKLFQPEPVKMYNKVISATDFRELYFSEILLNHPVMSGLIPKETMMFLDNWSKTEFF"
    "TEMKAEYDSSVREIEKFKDYPYQGHLNIACADNVVTCAGHVLLVERKFNPGKGCLALPGGHKHEKETFLDAAIRELQEETNIKVPE"
    "KVLRGSLVGEKMFDNPNRSYPHTRITMAYHLKVHPNPDNTFPKVKPADDAVSAKWYPLSEVRDMQERLYDDHYQIIQYFTGI"
)
WT_SHA256 = "8804bca153eceb5063f1d8a879d4daeff327964d5d1c9cd693d217fb4f86a7e8"
AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


class VariantError(ValueError):
    """Raised when a variant violates the sequence or design-class contract."""


def canonical_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.strip().upper().replace("-", "").encode()).hexdigest()


def wt_sequence() -> str:
    if len(WT_SEQUENCE) != 341:
        raise VariantError(f"canonical WT has length {len(WT_SEQUENCE)}, expected 341")
    if canonical_hash(WT_SEQUENCE) != WT_SHA256:
        raise VariantError("canonical WT sequence does not match the frozen sha256")
    if not set(WT_SEQUENCE) <= AMINO_ACIDS:
        raise VariantError("canonical WT contains non-standard residues")
    return WT_SEQUENCE


@dataclass(frozen=True)
class Variant:
    """A substitution-only NatV variant with a declared design class."""

    substitutions: tuple[tuple[int, str, str], ...]
    design_class: str

    def __post_init__(self) -> None:
        if self.design_class not in regions.DESIGN_CLASSES:
            raise VariantError(f"unknown design class {self.design_class!r}")
        wt = wt_sequence()
        editable = regions.editable_positions(self.design_class)
        positions = [s[0] for s in self.substitutions]
        if len(set(positions)) != len(positions):
            raise VariantError(f"duplicate positions in {positions}")
        for position, wt_aa, alt_aa in self.substitutions:
            if not 1 <= position <= 341:
                raise VariantError(f"position {position} outside 1..341")
            if wt_aa != wt[position - 1]:
                raise VariantError(f"position {position}: WT is {wt[position - 1]}, got {wt_aa}")
            if alt_aa not in AMINO_ACIDS:
                raise VariantError(f"position {position}: non-standard residue {alt_aa!r}")
            if alt_aa == wt_aa:
                raise VariantError(f"position {position}: substitution is a no-op")
            if position not in editable:
                raise VariantError(
                    f"position {position} ({regions.region_of(position)}) is not editable in class {self.design_class}"
                )
        if self.design_class == "A" and set(positions) & regions.catalytic_anchors():
            raise VariantError("class A must not edit catalytic anchors")

    def edits(self) -> str:
        return ";".join(f"{w}{p}{a}" for p, w, a in self.substitutions)

    def sequence(self) -> str:
        seq = list(wt_sequence())
        for position, _, alt_aa in self.substitutions:
            seq[position - 1] = alt_aa
        return "".join(seq)

    def positions(self) -> tuple[int, ...]:
        return tuple(sorted(p for p, _, _ in self.substitutions))

    def region_summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for p in self.positions():
            name = regions.region_of(p)
            counts[name] = counts.get(name, 0) + 1
        return counts
