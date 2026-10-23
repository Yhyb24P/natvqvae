from natvqvae.nmnat_mapping import (
    _audit_status,
    _distance_flag,
    _summary_status,
    _target_atom_candidates,
)


def test_target_atom_candidates_use_source_atom_role():
    names, match, chemistry = _target_atom_candidates('THR', 'OG1', 'O', 'THR')
    assert names == ['OG1']
    assert 'N' not in names
    assert match == 'same_name'
    assert chemistry == 'identical'

    names, match, chemistry = _target_atom_candidates('HIS', 'NE2', 'N', 'HIS')
    assert names == ['NE2']
    assert 'CD2' not in names

    names, match, chemistry = _target_atom_candidates('GLY', 'N', 'N', 'ALA')
    assert names == ['N']
    assert match == 'backbone_same_name'
    assert chemistry == 'backbone_compatible'

    names, match, chemistry = _target_atom_candidates('GLY', 'N', 'N', 'PRO')
    assert names == []
    assert match == 'backbone_proline_mismatch'
    assert chemistry == 'role_mismatch'


def test_symmetric_and_group_equivalents():
    names, match, chemistry = _target_atom_candidates('ASP', 'OD1', 'O', 'ASP')
    assert set(names) == {'OD1', 'OD2'}
    assert match == 'symmetric_equivalent'
    assert chemistry == 'symmetric_equivalent'

    names, match, chemistry = _target_atom_candidates('ARG', 'NH1', 'N', 'ARG')
    assert set(names) == {'NH1', 'NH2'}

    names, match, chemistry = _target_atom_candidates('GLU', 'OE1', 'O', 'ASP')
    assert set(names) == {'OD1', 'OD2'}
    assert match == 'role_equivalent'
    assert chemistry == 'charge_conservative'

    names, match, chemistry = _target_atom_candidates('THR', 'OG1', 'O', 'SER')
    assert names == ['OG']
    assert chemistry == 'polar_role_equivalent'


def test_audit_status_and_summary_counts():
    assert _distance_flag(1.9, 2.0, 2.5) == 'severe'
    assert _distance_flag(2.2, 2.0, 2.5) == 'close'
    assert _distance_flag(3.0, 2.0, 2.5) == 'none'
    assert _audit_status('high', 3.0, 4.0, 'none', 'none', 'identical') == 'supported_candidate'
    assert _audit_status('high', 3.0, 4.0, 'none', 'none', 'role_mismatch') == 'geometric_only'
    assert _audit_status('low', 3.0, 4.0, 'none', 'none', 'identical') == 'uncertain_mapping'
    assert _audit_status('high', 4.5, 4.0, 'none', 'none', 'identical') == 'not_recovered'

    keys = ('supported_candidate', 'geometric_only', 'clash', 'not_recovered',
            'uncertain_mapping', 'unmapped', 'role_mismatch')
    counts = {key: 0 for key in keys}
    counts['supported_candidate'] = 3
    assert _summary_status(counts, 3) == 'supported_consistent'
    counts['supported_candidate'] = 2
    assert _summary_status(counts, 3) == 'supported_majority'
    counts['supported_candidate'] = 1
    assert _summary_status(counts, 3) == 'model_dependent_supported'
    counts['supported_candidate'] = 2
    counts['clash'] = 1
    assert _summary_status(counts, 3) == 'supported_majority_with_model_clash'
