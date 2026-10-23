from natvqvae.nmnat_nmn import (_atom_function, _compatibility, _ligand_group,
                               _protein_group)


def test_nmn_atom_groups_are_not_merged():
    assert _ligand_group('O1P') == 'phosphate'
    assert _ligand_group('O3R') == 'ribose'
    assert _ligand_group('N7') == 'nicotinamide'


def test_target_gly_cannot_supply_a_sidechain_contact():
    match, source_group, target_group = _compatibility(
        'VAL', 'CG2', 'GLY', ['N', 'CA', 'C', 'O'])
    assert match == 'target_lacks_sidechain'
    assert source_group == 'aliphatic'
    assert target_group == 'backbone'


def test_aromatic_ring_is_not_reduced_to_exact_atom_name():
    match, source_group, _ = _compatibility(
        'TRP', 'NE1', 'TYR', ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2',
                              'CE1', 'CE2', 'CZ', 'OH'])
    assert match == 'aromatic_ring_role'
    assert source_group == 'aromatic_ring'


def test_hydroxyl_sidechain_uses_role_equivalence():
    match, source_group, _ = _compatibility('THR', 'OG1', 'SER', ['OG'])
    assert match == 'role_equivalent'
    assert source_group == 'hydroxyl'


def test_protein_group_separates_backbone_and_carboxyl():
    assert _protein_group('ASP', 'CA') == 'backbone'
    assert _protein_group('ASP', 'OD1') == 'carboxyl'


def test_mse_selenium_and_parent_met_are_distinguished():
    assert _protein_group('MSE', 'SE') == 'selenium'
    assert _protein_group('MSE', 'SD') == 'sulfur'
    assert _protein_group('MSE', 'CA') == 'backbone'
    assert _atom_function('MSE', 'SE') == 'selenium_not_equivalent'


def test_proline_backbone_n_remains_backbone():
    assert _protein_group('PRO', 'N') == 'backbone'
    match, source_group, _ = _compatibility('PRO', 'N', 'PRO', ['N', 'CA', 'C', 'O'])
    assert match == 'backbone_same_name'
    assert source_group == 'backbone'
