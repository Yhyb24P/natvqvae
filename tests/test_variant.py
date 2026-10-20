import pytest

from natvqvae.sequence import Variant, VariantError, WT_SEQUENCE, wt_sequence


def test_valid_class_a():
    v = Variant((((24, "A", "E"),)), "A")
    assert v.positions() == (24,)
    assert v.sequence()[23] == "E"
    assert v.sequence()[:23] + v.sequence()[24:] == WT_SEQUENCE[:23] + WT_SEQUENCE[24:]
    assert v.edits() == "A24E"
    assert v.region_summary() == {"NMNAT": 1}


def test_valid_class_b_uses_nudix_anchors():
    v = Variant(((232, "G", "A"), (307, "D", "N")), "B")
    assert v.region_summary() == {"NUDIX": 2}
    assert v.edits() == "G232A;D307N"


def test_valid_class_c_joins_both_domains():
    v = Variant(((24, "A", "E"), (232, "G", "A")), "C")
    assert v.region_summary() == {"NMNAT": 1, "NUDIX": 1}


@pytest.mark.parametrize("subs,cls", [
    (((24, "A", "A"),), "A"),          # no-op
    (((24, "V", "E"),), "A"),          # wrong WT residue
    (((24, "A", "E"), (24, "A", "D")), "A"),  # duplicate position
    (((250, "Q", "R"),), "A"),         # outside class A editable set
    (((24, "A", "E"),), "B"),          # outside class B editable set
    (((182, "S", "A"),), "C"),         # linker is never editable
    (((0, "M", "A"),), "A"),           # out of range
    (((342, "I", "A"),), "A"),
    (((24, "A", "B"),), "A"),          # non-standard residue
])
def test_invalid_variants_rejected(subs, cls):
    with pytest.raises(VariantError):
        Variant(subs, cls)


def test_class_a_cannot_touch_nudix_anchor():
    with pytest.raises(VariantError):
        Variant(((232, "G", "A"),), "A")


def test_unknown_class_rejected():
    with pytest.raises(VariantError):
        Variant(((24, "A", "E"),), "X")


def test_empty_variant_is_identity():
    v = Variant(tuple(), "C")
    assert v.sequence() == wt_sequence()
