"""Regression checks for substrate-state inputs and full-length geometry."""
import gzip
import hashlib
from pathlib import Path

import numpy as np
import yaml
import pytest
from Bio.PDB import Atom, Chain, MMCIFIO, Model, Residue, Structure

from natvqvae.nmnat_binding import (_ccd_reaction_bonds, _contacts, _cross_from_atoms,
                                   _pair_metrics, _pose_rmsd_from_coordinates,
                                   _protein_pose_metrics,
                                   _reaction_from_atoms, _stage_checkpoint,
                                   _structure, _yaml_text)


def _terminal_complex():
    structure = Structure.Structure('terminal')
    model = Model.Model(0)
    structure.add(model)
    chain = Chain.Chain('A')
    model.add(chain)
    residue = Residue.Residue((' ', 341, ' '), 'GLY', '')
    chain.add(residue)
    residue.add(Atom.Atom('CA', np.array([1.0, 0.0, 0.0]), 10.0, 1.0,
                         ' ', ' CA ', 1, element='C'))
    ligand_chain = Chain.Chain('B')
    model.add(ligand_chain)
    ligand = Residue.Residue(('H_ATP', 1, ' '), 'ATP', '')
    ligand_chain.add(ligand)
    ligand.add(Atom.Atom('O1A', np.zeros(3), 10.0, 1.0,
                        ' ', ' O1A', 2, element='O'))
    return structure


def test_reference_mmcif_gzip_is_readable(tmp_path):
    path = tmp_path / 'reference.cif'
    writer = MMCIFIO()
    writer.set_structure(_terminal_complex())
    writer.save(str(path))
    compressed = tmp_path / 'reference.cif.gz'
    with gzip.open(compressed, 'wt') as handle:
        handle.write(path.read_text())
    model = _structure(compressed)
    assert model['A'][341]['CA'].element == 'C'


def test_full_length_terminal_overlap_is_not_omitted():
    rows = _contacts(_terminal_complex()[0], 'ATP', 4.0, 2.0)
    assert len(rows) == 1
    assert rows[0]['protein_resnum'] == 341
    assert rows[0]['severe']


def test_joint_input_assigns_distinct_ligand_ids(tmp_path):
    config = {
        'states': {'JOINT': {'ligands': ['ATP', 'NMN'], 'ions': ['MG']}},
        'constraints': {'JOINT': []},
        'pocket': {'max_distance': 6.0, 'force': False},
    }
    result = yaml.safe_load(_yaml_text(config, 'JOINT', 'ACD',
                                      tmp_path / 'local.a3m', 'unconstrained'))
    ligands = [(row['ligand']['id'], row['ligand']['ccd'])
               for row in result['sequences'] if 'ligand' in row]
    assert ligands == [('B', 'ATP'), ('C', 'NMN'), ('D', 'MG')]


def _reaction_atoms():
    atp = [('PA', 'P', np.array([0.0, 0.0, 0.0])),
           ('O3A', 'O', np.array([1.6, 0.0, 0.0])),
           ('PB', 'P', np.array([3.2, 0.0, 0.0])),
           ("O5'", 'O', np.array([0.0, 1.5, 0.0]))]
    nmn = [('P', 'P', np.array([0.0, 3.0, 0.0])),
           ('O1P', 'O', np.array([0.0, 4.0, 0.0])),
           ('O2P', 'O', np.array([0.0, 2.0, 0.0])),
           ('O3P', 'O', np.array([1.0, 3.0, 0.0])),
           ('O5R', 'O', np.array([0.0, 5.0, 0.0]))]
    return atp, nmn


def test_reaction_geometry_reports_all_nmn_phosphate_oxygens():
    atp, nmn = _reaction_atoms()
    result, candidates = _reaction_from_atoms(atp, nmn)
    assert result['atp_PA_O3A_A'] == 1.6
    assert result['atp_O3A_PB_A'] == 1.6
    assert result['nmn_candidate_atoms'] == 'O1P;O2P;O3P'
    assert [item['candidate_atom'] for item in candidates] == ['O1P', 'O2P', 'O3P']
    assert result['nmn_P_O5R_A'] is not None
    assert all(item['candidate_atom'] != 'O5R' for item in candidates)
    assert result['atp_PA_O3A_PB_deg'] is not None


def test_far_apart_ligands_keep_global_minimum_without_contact_flags():
    atp = [('PA', 'P', np.array([0.0, 0.0, 0.0]))]
    nmn = [('O1P', 'O', np.array([10.0, 0.0, 0.0]))]
    summary, rows = _cross_from_atoms(atp, nmn, 2.0)
    assert summary['cross_min_A'] == 10.0
    assert summary['cross_pairs_4A'] == 0
    assert summary['cross_severe'] == 0
    assert rows == []


def test_far_apart_protein_keeps_global_minimum():
    class Residue:
        resname = 'GLY'
        id = (' ', 1, ' ')

    class Atom:
        name = 'CA'
        coord = np.array([0.0, 0.0, 0.0])

    metrics = _protein_pose_metrics([('O1P', 'O', np.array([10.0, 0.0, 0.0]))],
                                    [(Residue(), Atom())], 2.0)
    assert metrics['protein_min_A'] == 10.0
    assert metrics['protein_pairs_4A'] == 0
    assert metrics['protein_severe'] == 0


def test_ccd_reaction_bonds_distinguish_atp_and_nmn_phosphates():
    cache = Path('/mnt/e/models/boltz2')
    if not cache.exists():
        pytest.skip('local Boltz CCD cache is unavailable')
    bonds = _ccd_reaction_bonds(cache)
    assert bonds['ATP']['PA_O3A'] is True
    assert bonds['ATP']['PA_O3B'] is False
    assert bonds['ATP']['PB_O3A'] is True
    assert bonds['ATP']['PB_O3B'] is True
    assert bonds['NMN']['P_O1P'] is True
    assert bonds['NMN']['P_O2P'] is True
    assert bonds['NMN']['P_O3P'] is True
    assert bonds['NMN']['P_O5R'] is True


def test_cross_ligand_clash_is_detected_independently_of_protein_contacts():
    atp, nmn = _reaction_atoms()
    summary, rows = _cross_from_atoms(atp, nmn, 2.0)
    assert summary['cross_severe'] >= 1
    assert summary['cross_min_A'] is not None
    assert any(row['atom_a'] == 'O3A' and row['atom_b'] == 'O2P' for row in rows)


def test_guided_input_retains_explicit_chemical_state(tmp_path):
    config = {
        'states': {'ATP_MG': {'ligand': 'ATP', 'ions': ['MG']}},
        'constraints': {'ATP_MG': [14, 17, 135, 138]},
        'pocket': {'max_distance': 6.0, 'force': False},
    }
    result = yaml.safe_load(_yaml_text(config, 'ATP_MG', 'ACD',
                                      tmp_path / 'local.a3m', 'anchor'))
    assert [row['ligand']['ccd'] for row in result['sequences']
            if 'ligand' in row] == ['ATP', 'MG']
    pocket = result['constraints'][0]['pocket']
    assert pocket['contacts'] == [['A', 14], ['A', 17], ['A', 135], ['A', 138]]
    assert pocket['force'] is False
    assert pocket['max_distance'] == 6.0
    assert not any('bond' in row for row in result['constraints'])


def test_unconstrained_input_has_no_pocket_features(tmp_path):
    config = {
        'states': {'NMN': {'ligand': 'NMN', 'ions': []}},
        'constraints': {'NMN': []},
    }
    result = yaml.safe_load(_yaml_text(config, 'NMN', 'ACD',
                                      tmp_path / 'local.a3m', 'unconstrained'))
    assert 'constraints' not in result
    assert len(result['sequences']) == 2


def test_staging_rejects_corrupt_existing_weights(tmp_path):
    source = tmp_path / 'source.ckpt'
    source.write_bytes(b'complete model weights')
    cache = tmp_path / 'cache'
    cache.mkdir()
    (cache / source.name).write_bytes(b'partial')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='wrong SHA'):
        _stage_checkpoint(source, cache, digest)


def test_staged_bytes_retain_model_identity(tmp_path):
    source = tmp_path / 'source.ckpt'
    content = bytes(range(256)) * 129
    source.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    staged = _stage_checkpoint(source, tmp_path / 'cache', digest)
    assert staged.read_bytes() == content
    assert source.read_bytes() == content


def _rigid(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0.0],
                     [np.sin(angle), np.cos(angle), 0.0],
                     [0.0, 0.0, 1.0]])


def _synthetic_geometry(rotation=None, translation=None):
    ca = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0],
                   [3.0, 4.0, 0.0], [0.0, 4.0, 1.0]])
    loop = np.array([[1.0, 1.0, 0.5], [2.0, 1.0, 0.8],
                     [2.0, 2.0, 1.1], [1.0, 2.0, 1.4]])
    ligand = {'P': np.array([2.0, 2.0, 3.0]), 'O1A': np.array([2.5, 2.0, 3.0]),
              'O2A': np.array([1.5, 2.0, 3.0]), 'O3A': np.array([2.0, 2.5, 3.0])}
    if rotation is None:
        rotation = np.eye(3)
    if translation is None:
        translation = np.zeros(3)
    transform = lambda points: points @ rotation.T + translation
    return ({index + 4: transform(point) for index, point in enumerate(ca)},
            {index + 102: transform(point) for index, point in enumerate(loop)},
            {name: transform(point) for name, point in ligand.items()})


def _move_geometry(geometry, rotation, translation):
    return tuple({key: point @ rotation.T + translation for key, point in part.items()}
                 for part in geometry)


def test_pair_metrics_invariant_to_common_rigid_transform():
    first = _synthetic_geometry()
    second = _synthetic_geometry(_rigid(0.9), np.array([4.0, -3.0, 2.0]))
    before = _pair_metrics(first, second)
    common_rotation = _rigid(-0.4)
    common_translation = np.array([-7.0, 5.0, 1.5])
    after = _pair_metrics(_move_geometry(first, common_rotation, common_translation),
                          _move_geometry(second, common_rotation, common_translation))
    assert np.allclose(before[:2], after[:2])


def test_pair_metrics_is_not_distance_to_third_reference():
    first = _synthetic_geometry()
    second = _synthetic_geometry(_rigid(np.pi / 2), np.array([5.0, -2.0, 1.0]))
    atp_rmsd, loop_rmsd, n_atp, n_loop = _pair_metrics(first, second)
    assert (n_atp, n_loop) == (4, 4)
    assert atp_rmsd < 1e-9
    assert loop_rmsd < 1e-9
    reference = np.array([100.0, 0.0, 0.0])
    third_reference_difference = abs(np.linalg.norm(first[2]['P'] - reference) -
                                     np.linalg.norm(second[2]['P'] - reference))
    assert third_reference_difference > 0.5


def test_pose_rmsd_removes_common_rigid_motion():
    rng = np.random.default_rng(7)
    ca = {position: rng.normal(size=3) for position in range(4, 24)}
    ligand = {'A': np.array([0.0, 0.0, 0.0]), 'B': np.array([1.0, 0.0, 0.0])}
    rotation = _rigid(0.7)
    translation = np.array([2.0, -1.0, 0.5])
    joint_ca = {position: coord @ rotation.T + translation for position, coord in ca.items()}
    joint_ligand = {name: coord @ rotation.T + translation for name, coord in ligand.items()}
    moved = _pose_rmsd_from_coordinates(joint_ca, joint_ligand, ca, ligand)
    assert moved['ligand_atom_count'] == 2
    assert moved['ligand_rmsd_A'] == 0.0
    displaced = dict(joint_ligand)
    displaced['B'] = displaced['B'] + np.array([1.0, 0.0, 0.0])
    changed = _pose_rmsd_from_coordinates(joint_ca, displaced, ca, ligand)
    assert abs(changed['ligand_rmsd_A'] - np.sqrt(0.5)) < 0.01
