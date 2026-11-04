import csv
import hashlib
import json

import pytest

from natvqvae.data import sha256_file
from natvqvae.test_data import inspect_test_history


def _csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _source(path):
    return {'path': str(path), 'sha256': sha256_file(path)}


def _fixture(tmp_path):
    ids = [hashlib.sha256(x.encode()).hexdigest() for x in ('protein-a', 'protein-b')]
    groups = ['a', 'b']
    directory = tmp_path / 'recovered'
    directory.mkdir()
    test_rows = [{'sequence_id': uid, 'component_v2': group} for uid, group in zip(ids, groups)]
    _csv(tmp_path / 'current.csv', test_rows)
    _csv(directory / 'test.csv', test_rows)
    (directory / 'test.fasta').write_text('>retained-original-file\nACDE\n')
    (directory / 'manifest.json').write_text(json.dumps({
        'status': 'COMPLETED', 'outputs': {
            name: _source(directory / name) for name in ('test.csv', 'test.fasta')}}))
    _csv(tmp_path / 'membership.csv', [
        {**row, 'prior_H1_39': str(i == 0), 'historically_pristine': 'False'}
        for i, row in enumerate(test_rows)])
    evaluations = []
    for name, subset, membership in [('full', test_rows, 'all_test'),
                                     ('earlier', test_rows[:1], 'prior_H1_39')]:
        (tmp_path / f'{name}.md').write_text('Completed evaluation was reported.')
        (tmp_path / f'{name}.json').write_text(json.dumps({'checkpoint_selection_calls': 0}))
        (tmp_path / f'{name}_status.json').write_text(json.dumps({
            'status': 'COMPLETE', 'started_unix': 1791083876.4805627}))
        # Deliberately unusable numbers: only IDs/model labels may be inspected.
        metrics = [{'sequence_id': row['sequence_id'], 'component': row['component_v2'],
                    'arm': 'S', 'seed': '1', 'CE_sum': 'not-a-number',
                    'targets': 'not-parsed', 'CE': 'not-a-number'} for row in subset]
        _csv(tmp_path / f'{name}_metrics.csv', metrics)
        evaluations.append({
            'name': name, 'membership': membership, 'model_description': 'Historical model',
            'report': _source(tmp_path / f'{name}.md'),
            'summary': _source(tmp_path / f'{name}.json'),
            'job_status': _source(tmp_path / f'{name}_status.json'),
            'models': [{'arm': 'S', 'seeds': [1]}],
            'metrics': [_source(tmp_path / f'{name}_metrics.csv')],
            'summary_fields': ['checkpoint_selection_calls']})
    config = {'current_test': _source(tmp_path / 'current.csv'), 'expected_test_sequences': 2,
              'expected_prior_h1_sequences': 1,
              'historical_membership': _source(tmp_path / 'membership.csv'),
              'prior_evaluations': evaluations}
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    return path, config, directory, ids


def test_history_checks_execution_members_without_using_scores(tmp_path):
    path, _, directory, ids = _fixture(tmp_path)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    result = inspect_test_history(path, directory)
    assert result['status'] == 'previously_evaluated'
    assert result['test_sequence_ids_sha256'] == hashlib.sha256(
        ('\n'.join(sorted(ids)) + '\n').encode()).hexdigest()
    assert result['prior_h1_sequence_count'] == 1
    assert result['prior_evaluations'][0]['sequence_count'] == 2
    assert result['flags']['historical_score_values_interpreted'] is False
    assert all((directory / name).read_bytes() == content for name, content in before.items())
    with pytest.raises(FileExistsError):
        inspect_test_history(path, directory)


@pytest.mark.parametrize('change', ['different_id', 'missing_id', 'duplicate_id', 'wrong_model'])
def test_history_rejects_incomplete_or_changed_metrics_membership(tmp_path, change):
    path, config, directory, _ = _fixture(tmp_path)
    metrics = tmp_path / 'full_metrics.csv'
    with metrics.open() as handle:
        rows = list(csv.DictReader(handle))
    if change == 'different_id':
        rows[0]['sequence_id'] = 'unexpected'
    elif change == 'missing_id':
        rows = rows[:1]
    elif change == 'duplicate_id':
        rows.append(rows[0])
    else:
        rows[0]['arm'] = 'unknown'
    _csv(metrics, rows)
    config['prior_evaluations'][0]['metrics'][0] = _source(metrics)
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='metrics|Duplicate historical'):
        inspect_test_history(path, directory)
    assert not (directory / 'historical_use.json').exists()


def test_report_and_completed_status_do_not_replace_actual_metrics(tmp_path):
    path, config, directory, _ = _fixture(tmp_path)
    config['prior_evaluations'][0]['metrics'] = []
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='report alone is not execution evidence'):
        inspect_test_history(path, directory)


def test_history_requires_source_hashes_and_confirmed_h1_membership(tmp_path):
    path, config, directory, _ = _fixture(tmp_path)
    config['historical_membership']['sha256'] = '0' * 64
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='source SHA256 differs'):
        inspect_test_history(path, directory)
    config['historical_membership'] = _source(tmp_path / 'membership.csv')
    config['prior_evaluations'] = config['prior_evaluations'][:1]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='H1 exposure flags lack matching'):
        inspect_test_history(path, directory)
