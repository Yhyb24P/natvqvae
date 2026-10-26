import csv
import json
import re
from pathlib import Path

import pytest
import yaml

from natvqvae.delivery import build_candidate_delivery

ROOT = Path(__file__).resolve().parents[1]


def _rows(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def _pdb_sequence(path):
    import gemmi

    structure = gemmi.read_structure(str(path))
    chains = [chain for model in structure for chain in model]
    assert len(chains) == 1
    residues = list(chains[0])
    assert len(residues) == 341
    return gemmi.one_letter_code([residue.name for residue in residues])


def test_candidate_delivery_package(tmp_path):
    output = tmp_path / 'delivery'
    result = build_candidate_delivery(ROOT / 'config/natv_candidate_delivery.yaml', output)
    assert result['rows'] == 14
    rows = _rows(output / 'candidates.csv')
    assert len(rows) == 14
    assert len({row['sequence'] for row in rows}) == 14
    assert len({row['sequence_sha256'] for row in rows}) == 14
    assert [row['candidate_id'] for row in rows].count('WT') == 1
    categories = [row['source_category'] for row in rows]
    assert categories.count('shared_reference') == 1
    assert categories.count('computational_design') == 11
    assert categories.count('epcr_clone') == 2

    by_id = {row['candidate_id']: row for row in rows}
    assert by_id['A1']['parent_a'] == 'WT'
    assert by_id['B1']['parent_a'] == 'WT'
    assert by_id['A1B1']['parent_a'] == 'A1'
    assert by_id['A1B1']['parent_b'] == 'B1'
    assert by_id['A1B1']['source_run'] == by_id['A1']['source_run']
    assert by_id['A1']['generation_candidate_id'] == 'QVAE_s8301_NMNAT_m08_r11_d14'
    assert by_id['A3']['generation_candidate_id'] == 'QVAE_s8303_NMNAT_m08_r12_d30'
    assert by_id['A1B1']['generation_candidate_id'] == by_id['A1']['generation_candidate_id']
    assert by_id['B1']['generation_candidate_id'] == ''
    assert by_id['A1']['source_table_record'] == 'candidates.csv:A1'
    assert by_id['NatV10']['source_table_record'] == 'variants.csv:NatV10'
    assert by_id['NatV10']['source_category'] == 'epcr_clone'
    assert '561c>t' in by_id['NatV89']['dna_edits']
    assert 'synonymous' in by_id['NatV89']['notes']

    fasta_names = [line[1:].split('|')[0] for line in
                   (output / 'candidates.fasta').read_text().splitlines() if line.startswith('>')]
    assert len(fasta_names) == 14
    assert set(fasta_names) == set(by_id)

    manifest = json.load(open(output / 'manifest.json'))
    assert manifest['counts'] == {
        'unique_proteins': 14, 'shared_reference': 1,
        'computational_design': 11, 'epcr_clone': 2,
        'coordinates': 42, 'reference_coordinates': 3, 'total_coordinate_files': 45,
    }
    assert manifest['packaging']['module'].endswith('delivery.py')
    assert manifest['packaging']['module_sha256']
    assert manifest['source_data_commit']

    coordinates = manifest['coordinates']
    assert len(coordinates) == 42
    assert len({(item['candidate'], item['model_index']) for item in coordinates}) == 42
    references = manifest['reference_coordinates']
    assert len(references) == 3
    assert all(item['candidate'] == 'epcr_WT' for item in references)

    sequences = {row['candidate_id']: row['sequence'] for row in rows}
    for item in coordinates + references:
        path = output / item['package']
        assert path.is_file()
        assert item['sha256'] == manifest['files'][item['package']]
        assert item['sequence_matches_unified_csv'] is True
        expected = (sequences['WT'] if item['candidate'] == 'epcr_WT'
                    else sequences[item['candidate']])
        assert _pdb_sequence(path) == expected

    for item in manifest['attachments'] + manifest['evidence']:
        path = output / item['package']
        assert path.is_file()
        assert item['sha256'] == manifest['files'][item['package']]
    evidence_names = {Path(item['package']).name for item in manifest['evidence']}
    assert {'design_candidates_original.csv', 'clone_variants.csv',
            'clone_sanger_verification.json', 'clone_build.json'} <= evidence_names
    assert not list(output.rglob('*.ab1'))

    readme = (output / 'README.md').read_text()
    assert not re.search(r'\]\(\.\./', readme), 'README must not link outside the package'
    assert 'github.com' in readme
    assert '没有任何经验证的活性' in readme
    for line in readme.splitlines():
        if '](https://github.com/' in line:
            assert re.search(r'\]\(https://github\.com/[^\s)]+\)', line), line


def _tamper_config(tmp_path, candidate, field, value):
    config = yaml.safe_load((ROOT / 'config/natv_candidate_delivery.yaml').read_text())
    source = ROOT / config['sources']['design_candidates']
    rows = _rows(source)
    for row in rows:
        if row['name'] == candidate:
            row[field] = value
    bad_csv = tmp_path / 'candidates.csv'
    with bad_csv.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    config['sources']['design_candidates'] = str(bad_csv)
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path


def test_candidate_delivery_rejects_nonempty_output(tmp_path):
    output = tmp_path / 'delivery'
    output.mkdir()
    (output / 'existing.txt').write_text('x')
    with pytest.raises(FileExistsError):
        build_candidate_delivery(ROOT / 'config/natv_candidate_delivery.yaml', output)


def test_candidate_delivery_rejects_wrong_edit(tmp_path):
    config_path = _tamper_config(tmp_path, 'A1', 'edits', 'L18I;D57S;R94N')
    with pytest.raises(ValueError):
        build_candidate_delivery(config_path, tmp_path / 'delivery')


def test_candidate_delivery_rejects_wrong_sequence_sha(tmp_path):
    config_path = _tamper_config(tmp_path, 'A2', 'sequence_sha256', '0' * 64)
    with pytest.raises(ValueError):
        build_candidate_delivery(config_path, tmp_path / 'delivery')


def test_candidate_delivery_preserves_source_candidate_field(tmp_path):
    config_path = _tamper_config(tmp_path, 'A1', 'source_candidate', '')
    build_candidate_delivery(config_path, tmp_path / 'delivery')
    by_id = {row['candidate_id']: row
             for row in _rows(tmp_path / 'delivery' / 'candidates.csv')}
    assert by_id['A1']['generation_candidate_id'] == ''
    assert by_id['A1']['source_table_record'] == 'candidates.csv:A1'
