import pytest

from natvqvae import regions


def test_partition_is_clean():
    regions.validate_partition()
    covered = set()
    for reg in regions.regions().values():
        covered |= reg.positions
    assert covered == set(range(1, 342))


def test_region_of_boundaries():
    assert regions.region_of(1) == "NTERM"
    assert regions.region_of(4) == "NMNAT"
    assert regions.region_of(181) == "NMNAT"
    assert regions.region_of(182) == "LINKER"
    assert regions.region_of(203) == "LINKER"
    assert regions.region_of(204) == "NUDIX"
    assert regions.region_of(338) == "NUDIX"
    assert regions.region_of(341) == "CTERM"
    with pytest.raises(ValueError):
        regions.region_of(0)
    with pytest.raises(ValueError):
        regions.region_of(342)


def test_anchors_lie_in_their_domains():
    anchors = regions.anchors()
    assert anchors["NMNAT"] == (14, 17, 135, 138)
    assert anchors["NUDIX"] == (232, 236, 247, 248, 251, 252, 307, 308)
    for domain, positions in anchors.items():
        for p in positions:
            assert regions.region_of(p) == domain


def test_class_sets():
    a = regions.editable_positions("A")
    b = regions.editable_positions("B")
    c = regions.editable_positions("C")
    assert a == frozenset(range(4, 182))
    assert b == frozenset(range(204, 339))
    assert c == a | b
    assert not a & b
    # linker and termini are never editable
    for cls in regions.DESIGN_CLASSES:
        editable = regions.editable_positions(cls)
        assert not editable & frozenset(range(1, 4))
        assert not editable & frozenset(range(182, 204))
        assert not editable & frozenset(range(339, 342))


def test_forbidden_and_editable_are_disjoint():
    for cls in regions.DESIGN_CLASSES:
        assert not regions.editable_positions(cls) & regions.forbidden_positions(cls)


def test_unknown_class_rejected():
    with pytest.raises(ValueError):
        regions.editable_positions("D")
