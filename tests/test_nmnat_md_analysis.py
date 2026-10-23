import numpy as np

from natvqvae.nmnat_md_analysis import (
    ATP_PHOSPHATE_O, NMN_PHOSPHATE_O, _ccd_index, _kabsch, _mg_ligand_table,
    _mic, _na_phosphate_summary, _rmsd)


def test_mic_wraps_across_boundary():
    delta = np.array([9.5, 0.0, 0.0])
    assert np.allclose(_mic(delta, np.array([10.0, 10.0, 10.0])),
                       [-0.5, 0.0, 0.0])


def test_ccd_index_uses_topology_number_minus_one():
    mapping = {'models': {'m': {'ligands': {'ATP': {'atoms': [
        {'ccd_name': 'PA', 'topology_index': 5511},
        {'ccd_name': 'O3A', 'topology_index': 5519},
        {'ccd_name': 'O1A', 'topology_index': 5510},
    ]}}}}}
    found = _ccd_index(mapping, 'm', 'ATP', ('PA', 'O3A'))
    assert found == {'PA': 5510, 'O3A': 5518}


def test_rmsd_is_zero_after_rigid_transform():
    reference = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0]])
    rotation, _, _ = _kabsch(reference, reference)
    moved = reference @ rotation + np.array([4.0, -2.0, 7.0])
    assert _rmsd(reference, moved) < 1e-9


def test_analysis_counts_separate_trajectories_frames_and_candidates():
    from natvqvae.nmnat_md_analysis import _analysis_counts
    frames = [
        {'model': 'm1', 'velocity_seed': 1, 'time_ps': 0.0, 'candidate': 'O1P'},
        {'model': 'm1', 'velocity_seed': 1, 'time_ps': 0.0, 'candidate': 'O2P'},
        {'model': 'm1', 'velocity_seed': 1, 'time_ps': 0.0, 'candidate': 'O3P'},
        {'model': 'm1', 'velocity_seed': 1, 'time_ps': 10.0, 'candidate': 'O1P'},
        {'model': 'm1', 'velocity_seed': 1, 'time_ps': 10.0, 'candidate': 'O2P'},
        {'model': 'm1', 'velocity_seed': 1, 'time_ps': 10.0, 'candidate': 'O3P'},
        {'model': 'm2', 'velocity_seed': 2, 'time_ps': 0.0, 'candidate': 'O1P'},
    ]
    assert _analysis_counts(frames) == (2, 3, 7)


def test_phosphate_atom_sets_are_complete():
    assert len(ATP_PHOSPHATE_O) == 9
    assert len(set(ATP_PHOSPHATE_O)) == 9
    assert set(NMN_PHOSPHATE_O) == {'O1P', 'O2P', 'O3P'}
    names = list(ATP_PHOSPHATE_O) + list(NMN_PHOSPHATE_O)
    atoms = [{'ccd_name': name, 'topology_index': index + 1}
             for index, name in enumerate(names)]
    mapping = {'models': {'m': {'ligands': {
        'ATP': {'atoms': atoms[:9]}, 'NMN': {'atoms': atoms[9:]}}}}}
    assert len(_ccd_index(mapping, 'm', 'ATP', ATP_PHOSPHATE_O)) == 9
    assert len(_ccd_index(mapping, 'm', 'NMN', NMN_PHOSPHATE_O)) == 3


def test_na_phosphate_summary_counts_each_ion_once():
    na_atp = np.array([[3.5, 5.0], [5.0, 5.0]])
    na_nmn = np.array([[5.0, 5.0], [3.9, 5.0]])
    minimum, within = _na_phosphate_summary(na_atp, na_nmn)
    assert abs(minimum - 3.5) < 1e-9
    assert within == 2
    both = np.array([[3.0, 3.0]])
    minimum, within = _na_phosphate_summary(both, both)
    assert abs(minimum - 3.0) < 1e-9
    assert within == 1


def test_mg_ligand_table_groups_and_counts_missing_as_zero():
    rows = [
        {'group': 'A', 'model': 'm1', 'velocity_seed': 1,
         'ligand_atom': 'X', 'fraction': 1.0},
        {'group': 'A', 'model': 'm2', 'velocity_seed': 1,
         'ligand_atom': 'X', 'fraction': 0.2},
        {'group': 'A', 'model': 'm2', 'velocity_seed': 1,
         'ligand_atom': 'Y', 'fraction': 0.9},
        {'group': 'B', 'model': 'm3', 'velocity_seed': 1,
         'ligand_atom': 'X', 'fraction': 0.1},
    ]
    table = _mg_ligand_table(rows).set_index(['group', 'ligand_atom'])
    assert table.loc[('A', 'X'), 'trajectories'] == 2
    assert table.loc[('A', 'X'), 'persistent'] == 1
    assert abs(table.loc[('A', 'X'), 'median_fraction'] - 0.6) < 1e-9
    assert table.loc[('A', 'Y'), 'trajectories'] == 2
    assert abs(table.loc[('A', 'Y'), 'median_fraction'] - 0.45) < 1e-9
    assert ('B', 'X') not in table.index
    assert _mg_ligand_table([]).empty


def test_mg_ligand_table_keeps_trajectories_without_rows():
    rows = [{'group': 'A', 'model': 'm1', 'velocity_seed': 1,
             'ligand_atom': 'X', 'fraction': 1.0}]
    trajectories = [
        {'group': 'A', 'model': 'm1', 'velocity_seed': 1},
        {'group': 'A', 'model': 'm2', 'velocity_seed': 1},
    ]
    table = _mg_ligand_table(rows, trajectories).set_index(
        ['group', 'ligand_atom'])
    assert table.loc[('A', 'X'), 'trajectories'] == 2
    assert table.loc[('A', 'X'), 'persistent'] == 1
    assert abs(table.loc[('A', 'X'), 'median_fraction'] - 0.5) < 1e-9
    assert _mg_ligand_table([], trajectories).empty
