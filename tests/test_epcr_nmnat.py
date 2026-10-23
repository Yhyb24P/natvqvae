import numpy as np

from natvqvae.nmnat_mapping import _apply_transform, _kabsch, _target_atom_candidates


def _pairwise(points):
    return np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)


def test_common_rigid_transform_preserves_distances():
    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0],
                       [0.3, -0.4, 1.2]], dtype=float)
    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                         [np.sin(angle), np.cos(angle), 0.0],
                         [0.0, 0.0, 1.0]])
    translation = np.array([3.0, -2.0, 0.5])
    target = source @ rotation.T + translation
    rmsd, fitted_rotation, fitted_translation = _kabsch(source, target)
    aligned = _apply_transform(source, fitted_rotation, fitted_translation)
    assert rmsd < 1e-6
    assert np.allclose(_pairwise(aligned), _pairwise(target))


def test_source_ligand_distance_invariance():
    ligand = np.array([[10.0, 2.0, -1.0], [11.2, 2.4, -0.5]], dtype=float)
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.array([-4.0, 7.0, 2.0])
    moved = _apply_transform(ligand, rotation, translation)
    assert np.isclose(np.linalg.norm(ligand[1] - ligand[0]),
                      np.linalg.norm(moved[1] - moved[0]))


def test_target_atom_correspondence_does_not_use_nearest_arbitrary_atom():
    names, match, chemistry = _target_atom_candidates('THR', 'OG1', 'O', 'THR')
    assert names == ['OG1']
    assert 'N' not in names
    names, match, chemistry = _target_atom_candidates('HIS', 'NE2', 'N', 'HIS')
    assert names == ['NE2']
    assert 'CD2' not in names
    names, match, chemistry = _target_atom_candidates('GLY', 'N', 'N', 'PRO')
    assert names == []
    assert match == 'backbone_proline_mismatch'
    assert chemistry == 'role_mismatch'
