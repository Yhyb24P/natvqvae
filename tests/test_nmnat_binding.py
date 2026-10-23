"""Regression checks for substrate-state inputs and full-length geometry."""
import gzip
import hashlib

import numpy as np
import yaml
import pytest
from Bio.PDB import Atom, Chain, MMCIFIO, Model, Residue, Structure

from natvqvae.nmnat_binding import (_contacts, _pair_metrics, _stage_checkpoint,
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
