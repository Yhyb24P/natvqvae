import csv
from pathlib import Path

import numpy as np
import pytest
from Bio.PDB import PDBParser

from natvqvae.delivery_coordinates import convert_coordinate
from natvqvae.nmnat_mapping import _open_structure

ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / 'data/results/evaluation'


@pytest.mark.parametrize('candidate,index,batch', [
    ('WT', 0, 'design'), ('A1', 1, 'design'), ('B2', 2, 'design'),
    ('A3B1', 0, 'design'), ('NatV89', 2, 'epcr'), ('WT', 1, 'epcr'),
])
def test_existing_prediction_retains_all_atoms_and_frame(tmp_path, candidate, index, batch):
    if batch == 'design':
        archive = EVALUATION / 'natv_dual_domain_design/coordinates'
        table = ROOT / 'data/results/generation/natv_dual_domain_design/candidates.csv'
    else:
        archive = EVALUATION / 'kvp40_epcr_full_length_structures/coordinates'
        table = archive.parent / 'variants.csv'
    rows = {row['name']: row for row in csv.DictReader(table.open())}
    source = archive / f'{candidate}_{candidate}_model_{index}.cif.gz'
    directory = 'references' if candidate == 'WT' and batch == 'epcr' else 'coordinates'
    record = convert_coordinate(candidate, index, source, batch, tmp_path,
                                rows[candidate]['sequence'], directory)
    path = tmp_path / record['package']
    original = list(_open_structure(source).get_atoms())
    converted = list(PDBParser(QUIET=True).get_structure('converted', path).get_atoms())
    assert len(original) == len(converted) == record['atom_count']
    assert [atom.get_full_id()[2:] for atom in original] == [
        atom.get_full_id()[2:] for atom in converted]
    assert np.max(np.linalg.norm(
        np.array([atom.coord for atom in original])
        - np.array([atom.coord for atom in converted]), axis=1)) < 0.00087
    assert record['sequence_sha256'] == rows[candidate]['sequence_sha256']
    assert record['model_index'] == index


def test_wrong_sequence_and_existing_output_are_not_silently_accepted(tmp_path):
    source = EVALUATION / 'natv_dual_domain_design/coordinates/WT_WT_model_0.cif.gz'
    table = ROOT / 'data/results/generation/natv_dual_domain_design/candidates.csv'
    rows = {row['name']: row for row in csv.DictReader(table.open())}
    with pytest.raises(ValueError, match='source sequence'):
        convert_coordinate('WT', 0, source, 'design', tmp_path,
                           rows['A1']['sequence'], 'coordinates')
    assert not (tmp_path / 'coordinates/WT_model_0.pdb').exists()
    record = convert_coordinate('WT', 0, source, 'design', tmp_path,
                                rows['WT']['sequence'], 'coordinates')
    path = tmp_path / record['package']
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        convert_coordinate('WT', 0, source, 'design', tmp_path,
                           rows['WT']['sequence'], 'coordinates')
    assert path.read_bytes() == before
