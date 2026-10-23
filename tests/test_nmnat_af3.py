import json
from pathlib import Path

import numpy as np
import pytest

from natvqvae.nmnat_af3 import (_af3_confidence, _af3_models,
                                _ligand_symmetry_rmsd, _ligand_entries,
                                _model_metrics, _scaffold_atoms,
                                _symmetry_rmsd, build_af3_inputs)
from natvqvae.nmnat_binding import _structure


def _swap_ligand_atoms(structure, ligand, first, second):
    clone = structure.copy()
    residue = next(residue for residue in clone.get_residues()
                   if residue.id[0].startswith('H_')
                   and residue.resname == ligand)
    left = residue[first].coord.copy()
    right = residue[second].coord.copy()
    residue[first].set_coord(right)
    residue[second].set_coord(left)
    return clone


def test_ligand_entries_follow_protein_chain_a():
    entries = _ligand_entries(['ATP', 'NMN', 'MG'])
    assert [entry['ligand']['id'][0] for entry in entries] == ['B', 'C', 'D']


def test_symmetry_rmsd_permutes_equivalent_oxygens():
    center = np.zeros(3)
    first = (center, [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
                      np.array([0.0, 0.0, 1.0])])
    second = (center, [np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]),
                       np.array([1.0, 0.0, 0.0])])
    assert _symmetry_rmsd(first, second) == pytest.approx(0.0, abs=1e-9)


def test_ligand_symmetry_rmsd_identity_is_zero():
    structure = _structure('data/raw/structures/nmnat/1F9A.cif')
    assert _ligand_symmetry_rmsd(
        structure, structure, 'ATP', np.eye(3), np.zeros(3)
    ) == pytest.approx(0.0, abs=1e-6)
    assert _ligand_symmetry_rmsd(
        structure, structure, 'MG', np.eye(3), np.zeros(3)
    ) is None or _ligand_symmetry_rmsd(
        structure, structure, 'MG', np.eye(3), np.zeros(3)
    ) == pytest.approx(0.0, abs=1e-6)


def test_cn_scaffold_excludes_phosphate_and_sugar_oxygen():
    structure = _structure('data/raw/structures/nmnat/1F9A.cif')
    scaffold = _scaffold_atoms(structure, 'ATP')
    assert scaffold
    assert not any(name.startswith('O') or name.startswith('P')
                   for name in scaffold)


def test_build_declares_reference_query_only_msa(tmp_path):
    output = tmp_path / 'af3'
    result = build_af3_inputs('config/nmnat_af3.yaml', output)
    assert result['jobs'] == 3
    reference = json.load(open(output / 'inputs' / 'ref_1f9a_atp_mg.json'))
    protein = reference['sequences'][0]['protein']
    assert 'unpairedMsaPath' not in protein
    assert protein['unpairedMsa'].startswith('>query\n')
    assert protein['pairedMsa'] == ''
    assert protein['templates'] == []
    build = json.load(open(output / 'build.json'))
    assert build['reference_sequence_length'] == 164
    assert build['reference_canonical_length'] == 168
    assert build['reference_omitted_terminal_aa'] == 'QTDK'
    assert build['reference_boundary'].startswith(
        'single chain A; observed residues 1-164')
    row = next(row for row in
               __import__('csv').DictReader(open(output / 'inputs' / 'manifest.csv'))
               if row['kind'] == 'reference')
    assert row['run_data_pipeline'] == 'false'


def test_build_uses_supplied_wt_msa_and_three_ligand_chains(tmp_path):
    output = tmp_path / 'af3'
    build_af3_inputs('config/nmnat_af3.yaml', output)
    payload = json.load(open(output / 'inputs' / 'wt_atp_nmn_mg.json'))
    protein = payload['sequences'][0]['protein']
    assert protein['unpairedMsaPath'].endswith('WT.a3m')
    assert [entry['ligand']['id'][0]
            for entry in payload['sequences'][1:]] == ['B', 'C', 'D']
    assert [entry['ligand']['ccdCodes'][0]
            for entry in payload['sequences'][1:]] == ['ATP', 'NMN', 'MG']


def test_build_refuses_nonempty_output(tmp_path):
    output = tmp_path / 'af3'
    output.mkdir()
    (output / 'stale.txt').write_text('x')
    with pytest.raises(FileExistsError):
        build_af3_inputs('config/nmnat_af3.yaml', output)


def test_reference_metal_without_nmn_keeps_atp_and_nulls_nmn_fields():
    row, _, candidates = _model_metrics(
        {'path': Path('data/raw/structures/nmnat/1F9A.cif')})
    assert row['atp_chain_intact'] is True
    assert row['both_in_nmnat'] is None
    assert row['nmn_domain'] is None
    assert row['cross_min_A'] is None
    assert row['metal_NMN_O_min_A'] is None
    assert row['metal_nearby_NMN_O'] is None
    assert row['metal_shared_near_neighbor'] is None
    assert candidates == []


def test_atp_symmetry_permits_terminal_oxygen_swap_only():
    structure = _structure('data/raw/structures/nmnat/1F9A.cif')
    identity = np.eye(3)
    zero = np.zeros(3)
    terminal = _swap_ligand_atoms(structure, 'ATP', 'O1A', 'O2A')
    assert _ligand_symmetry_rmsd(
        terminal, structure, 'ATP', identity, zero
    ) == pytest.approx(0.0, abs=1e-6)
    bridge = _swap_ligand_atoms(structure, 'ATP', 'O3A', 'O1A')
    assert _ligand_symmetry_rmsd(
        bridge, structure, 'ATP', identity, zero) > 0.2
    sugar = _swap_ligand_atoms(structure, 'ATP', "C1'", "C2'")
    assert _ligand_symmetry_rmsd(
        sugar, structure, 'ATP', identity, zero) > 0.2


def test_nmn_symmetry_permits_phosphate_oxygen_swap_only():
    structure = _structure('data/raw/structures/nmnat/1NUS.cif')
    identity = np.eye(3)
    zero = np.zeros(3)
    terminal = _swap_ligand_atoms(structure, 'NMN', 'O1P', 'O2P')
    assert _ligand_symmetry_rmsd(
        terminal, structure, 'NMN', identity, zero
    ) == pytest.approx(0.0, abs=1e-6)
    phosphorus = _swap_ligand_atoms(structure, 'NMN', 'P', 'O1P')
    assert _ligand_symmetry_rmsd(
        phosphorus, structure, 'NMN', identity, zero) > 0.2


def test_ligand_symmetry_rmsd_does_not_refit_the_ligand():
    structure = _structure('data/raw/structures/nmnat/1F9A.cif')
    angle = 0.4
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                         [np.sin(angle), np.cos(angle), 0.0],
                         [0.0, 0.0, 1.0]])
    translation = np.array([3.0, -2.0, 1.0])
    transformed = _ligand_symmetry_rmsd(
        structure, structure, 'ATP', rotation, translation)
    assert transformed > 1.0
    shifted = _ligand_symmetry_rmsd(
        structure, structure, 'ATP', np.eye(3), translation)
    assert shifted == pytest.approx(float(np.linalg.norm(translation)),
                                    abs=1e-3)


def test_archive_models_are_discovered_with_per_sample_confidence(tmp_path):
    archive = tmp_path / 'cifs' / 'wt_atp_nmn'
    archive.mkdir(parents=True)
    model = archive / 'wt_atp_nmn_seed-2_sample-1_model.cif.gz'
    model.write_bytes(b'not parsed here')
    rows = _af3_models(tmp_path)
    assert [(row['condition'], row['seed'], row['sample']) for row in rows] == [
        ('wt_atp_nmn', 2, 1)]
    json.dump({'ptm': 0.11, 'iptm': 0.22}, open(
        archive / 'wt_atp_nmn_seed-2_sample-1_summary_confidences.json', 'w'))
    json.dump({'ptm': 0.99, 'iptm': 0.99}, open(
        archive / 'wt_atp_nmn_seed-2_sample-2_summary_confidences.json', 'w'))
    assert _af3_confidence(model)['ptm'] == 0.11
    assert _af3_confidence(model)['iptm'] == 0.22
    model.with_name('missing_seed-2_sample-1_model.cif.gz').write_bytes(b'x')
    assert _af3_confidence(
        model.with_name('missing_seed-2_sample-1_model.cif.gz')) == {}
