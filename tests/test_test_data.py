import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from natvqvae.data import Record, mask_panels, sha256_file
from natvqvae.test_data import load_recovered_test, recover_test_data


def _write_csv(path, rows, fields):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path):
    sequences = ['ACDEFGHIKLMNPQRSTVWY', 'VCDEFGHIKLMNPQRSTVWY', 'GCDEFGHIKLMNPQRSTVWY']
    rows, raw_hits, splits = [], [], {}
    fields = ['sequence_id', 'component_v2', 'length', 'nmnat_cdd_from',
              'nmnat_cdd_to', 'nudix_cdd_from', 'nudix_cdd_to']
    for i, (role, seq) in enumerate(zip(('training', 'validation', 'test'), sequences)):
        uid = hashlib.sha256(seq.encode()).hexdigest()
        row = {'sequence_id': uid, 'sequence': seq, 'length': len(seq), 'component_v2': f'g{i}',
               'training_role_v2': role, 'split_v2': role,
               'family_eligible_v2': True, 'core_training_eligible_v2': True,
               'source_annotation_conflict': False, 'nmnat_cdd_accession': 'cd02168',
               'nmnat_cdd_from': 2, 'nmnat_cdd_to': 8, 'nmnat_cdd_specificity': 'Non-specific',
               'nudix_cdd_from': 12, 'nudix_cdd_to': 18, 'nudix_cdd_specificity': 'Specific'}
        rows.append(row)
        for domain, accession, name in [('nmnat', 'cd02168', 'NMNAT_Nudix'),
                                       ('nudix', 'cd18873', 'NUDIX_NadM_like')]:
            raw_hits.append({'sequence_id': uid, 'accession': accession, 'short_name': name,
                             'query_from': row[f'{domain}_cdd_from'], 'query_to': row[f'{domain}_cdd_to'],
                             'specificity_class': row[f'{domain}_cdd_specificity']})
        path = tmp_path / f'{role}.csv'
        keep = fields if role != 'test' else fields[:2]
        _write_csv(path, [{key: row[key] for key in keep}], keep)
        splits[role] = str(path)
    tables = {'roles': pd.DataFrame(rows), 'population': pd.DataFrame(rows)[['sequence_id', 'sequence', 'length']],
              'cdd': pd.DataFrame(raw_hits)}
    sources = {}
    for name, frame in tables.items():
        path = tmp_path / f'{name}.parquet'
        frame.to_parquet(path, index=False)
        sources[name] = {'path': str(path), 'sha256': sha256_file(path)}
    population = tmp_path / 'current.fasta'
    population.write_text(''.join(f'>{r["sequence_id"]}\n{r["sequence"]}\n' for r in rows[:2]))
    config = {'sources': sources, 'current': {'population': str(population), 'splits': splits},
              'expected_counts': {role: 1 for role in splits},
              'cdd_models': {'nmnat': {'accession': 'cd02168', 'short_name': 'NMNAT_Nudix'},
                             'nudix': {'accession': 'cd18873', 'short_name': 'NUDIX_NadM_like'}},
              'output': str(tmp_path / 'restored')}
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config))
    return config_path, config, rows


def _change_source(config_path, config, name, change):
    path = Path(config['sources'][name]['path'])
    frame = change(pd.read_parquet(path))
    frame.to_parquet(path, index=False)
    config['sources'][name]['sha256'] = sha256_file(path)
    config_path.write_text(json.dumps(config))


def _current_data(rows):
    records = [Record(r['sequence_id'], r['sequence'], r['component_v2'], r['length'],
                      r['nmnat_cdd_from'], r['nmnat_cdd_to'], r['nudix_cdd_from'], r['nudix_cdd_to'])
               for r in rows]
    return SimpleNamespace(train=records[:1], validation=records[1:2],
                           test_rows=[{'sequence_id': records[2].sequence_id,
                                       'component': records[2].component}])


def test_recovery_preserves_split_and_cdd_evidence_and_loads_maskable_records(tmp_path):
    config_path, config, rows = _fixture(tmp_path)
    result = recover_test_data(config_path)
    assert result['test_original_csv_had_cdd'] is False
    assert result['test_prediction_scoring'] is False
    assert result['historical_wt_alignment_used'] is False
    assert result['checked_splits']['training']['unique_original_cdd_hits'] == 2
    assert result['checked_splits']['test']['cdd_specificity']['nmnat'] == {'Non-specific': 1}
    records = load_recovered_test(_current_data(rows), config['output'])
    assert len(records) == 1 and records[0].sequence == rows[2]['sequence']
    assert records[0].component == 'g2'
    panels = mask_panels(records[0], 7401)
    assert len(panels) == 27
    assert all(0 <= p < records[0].length for panel in panels for p in panel['positions'])
    with pytest.raises(FileExistsError):
        recover_test_data(config_path)


@pytest.mark.parametrize('source,column,value,match', [
    ('roles', 'component_v2', 'wrong', 'component differs'),
    ('roles', 'split_v2', 'training', 'role or component'),
    ('roles', 'nmnat_cdd_from', float('nan'), 'invalid nmnat_cdd_from'),
    ('roles', 'source_annotation_conflict', True, 'unconflicted'),
    ('population', 'sequence', 'AAAAAAAAAAAAAAAAAAA', 'SHA256'),
])
def test_recovery_rejects_conflicting_original_fields(tmp_path, source, column, value, match):
    config_path, config, _ = _fixture(tmp_path)
    def change(frame):
        frame.loc[2, column] = value
        return frame
    _change_source(config_path, config, source, change)
    with pytest.raises(ValueError, match=match):
        recover_test_data(config_path)
    assert not Path(config['output']).exists()


@pytest.mark.parametrize('kind', ['missing_record', 'missing_cdd', 'duplicate_cdd'])
def test_recovery_refuses_missing_or_ambiguous_source_records(tmp_path, kind):
    config_path, config, _ = _fixture(tmp_path)
    if kind == 'missing_record':
        _change_source(config_path, config, 'roles', lambda frame: frame.iloc[:2])
    elif kind == 'missing_cdd':
        _change_source(config_path, config, 'cdd', lambda frame: frame.iloc[:-1])
    else:
        _change_source(config_path, config, 'cdd', lambda frame: pd.concat([frame, frame.iloc[-1:]]))
    with pytest.raises(ValueError):
        recover_test_data(config_path)
    assert not Path(config['output']).exists()


def test_recovery_checks_current_training_annotations_not_only_test_ids(tmp_path):
    config_path, config, _ = _fixture(tmp_path)
    path = Path(config['current']['splits']['training'])
    rows = list(csv.DictReader(path.open()))
    rows[0]['nudix_cdd_to'] = '17'
    _write_csv(path, rows, list(rows[0]))
    with pytest.raises(ValueError, match='Current training field differs'):
        recover_test_data(config_path)
    assert not Path(config['output']).exists()


def test_recovery_checks_pinned_source_hashes(tmp_path):
    config_path, config, _ = _fixture(tmp_path)
    config['sources']['roles']['sha256'] = '0' * 64
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='source SHA256'):
        recover_test_data(config_path)


@pytest.mark.parametrize('kind,match', [('sha', 'output SHA256'), ('group', 'component differs'),
                                       ('overlap', 'components overlap')])
def test_recovered_loader_rejects_changed_files_or_split_membership(tmp_path, kind, match):
    config_path, config, rows = _fixture(tmp_path)
    recover_test_data(config_path)
    data = _current_data(rows)
    if kind == 'sha':
        with (Path(config['output']) / 'test.fasta').open('a') as handle:
            handle.write('\n')
    elif kind == 'group':
        data.test_rows[0]['component'] = 'wrong'
    else:
        existing = data.train[0]
        data.train = [Record(existing.sequence_id, existing.sequence, 'g2', existing.length,
                             existing.nmnat_from, existing.nmnat_to, existing.nudix_from, existing.nudix_to)]
    with pytest.raises(ValueError, match=match):
        load_recovered_test(data, config['output'])
