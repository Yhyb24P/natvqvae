"""Recover existing NatV test sequences and CDD annotations without scoring them."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from Bio import SeqIO

from .data import Record, load_yaml, sha256_file, write_json

_AA = set('ACDEFGHIKLMNPQRSTVWY')
_FIELDS = ['sequence_id', 'component_v2', 'length', 'nmnat_cdd_from',
           'nmnat_cdd_to', 'nudix_cdd_from', 'nudix_cdd_to']
_ROLES = ('training', 'validation', 'test')


def _integer(value, field, uid):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'Missing or invalid {field}: {uid}') from None
    if not math.isfinite(number) or number != int(number):
        raise ValueError(f'Missing or invalid {field}: {uid}')
    return int(number)


def _sequence(sequence, uid):
    if (not isinstance(sequence, str) or not sequence or not set(sequence) <= _AA or
            hashlib.sha256(sequence.encode()).hexdigest() != uid):
        raise ValueError(f'Sequence content or SHA256 differs: {uid}')
    return sequence


def _table_index(frame, name):
    if 'sequence_id' not in frame or frame.sequence_id.isna().any() or frame.sequence_id.duplicated().any():
        raise ValueError(f'Missing or duplicate sequence IDs in {name}')
    return frame.set_index('sequence_id', drop=False).to_dict('index')


def _checked_record(row, sequence):
    uid = row['sequence_id']
    _sequence(sequence, uid)
    numbers = {name: _integer(row[name], name, uid) for name in _FIELDS[2:]}
    if numbers['length'] != len(sequence) or not row['component_v2']:
        raise ValueError(f'Sequence length or component differs: {uid}')
    if not (1 <= numbers['nmnat_cdd_from'] <= numbers['nmnat_cdd_to'] <
            numbers['nudix_cdd_from'] <= numbers['nudix_cdd_to'] <= len(sequence)):
        raise ValueError(f'Invalid ordered CDD intervals: {uid}')
    if numbers['nudix_cdd_from'] - numbers['nmnat_cdd_to'] <= 1:
        raise ValueError(f'CDD intervals leave no intervening-region mask positions: {uid}')
    return Record(uid, sequence, row['component_v2'], numbers['length'],
                  numbers['nmnat_cdd_from'], numbers['nmnat_cdd_to'],
                  numbers['nudix_cdd_from'], numbers['nudix_cdd_to'])


def recover_test_data(config_path, output=None):
    """Validate all current splits against archived records, then export test inputs."""
    import pandas as pd

    config = load_yaml(config_path)
    if any(int(config['expected_counts'][role]) < 1 for role in _ROLES):
        raise ValueError('All three expected split counts must be positive')
    output = Path(output or config['output'])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Test recovery output already contains files: {output}')
    sources = config['sources']
    source_records = {}
    for name in ('roles', 'population', 'cdd'):
        item = sources[name]
        actual = sha256_file(item['path'])
        if actual != item['sha256']:
            raise ValueError(f'Original source SHA256 differs: {name}')
        source_records[name] = {'path': item['path'], 'sha256': actual,
                                'bytes': Path(item['path']).stat().st_size}
    roles = _table_index(pd.read_parquet(sources['roles']['path']), 'roles')
    source_population = _table_index(pd.read_parquet(sources['population']['path']), 'population')
    cdd = pd.read_parquet(sources['cdd']['path'])
    cdd_hits = Counter()
    for row in cdd.to_dict('records'):
        cdd_hits[(row['sequence_id'], row['accession'], row['short_name'],
                  _integer(row['query_from'], 'query_from', row['sequence_id']),
                  _integer(row['query_to'], 'query_to', row['sequence_id']),
                  row['specificity_class'])] += 1
    current_population = {}
    for record in SeqIO.parse(config['current']['population'], 'fasta'):
        if record.id in current_population:
            raise ValueError(f'Duplicate current population ID: {record.id}')
        current_population[record.id] = _sequence(str(record.seq), record.id)
    split_rows, seen_ids, component_roles = {}, set(), {}
    current_input_records = {}
    for role in _ROLES:
        path = config['current']['splits'][role]
        with open(path, newline='') as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            required = _FIELDS if role != 'test' else _FIELDS[:2]
            if not set(required) <= set(fields) or not set(fields) <= set(_FIELDS):
                raise ValueError(f'Current {role} split has missing or unexpected columns')
            rows = list(reader)
        if len(rows) != int(config['expected_counts'][role]):
            raise ValueError(f'Unexpected current split size: {role}')
        for row in rows:
            uid, group = row['sequence_id'], row['component_v2']
            if not uid or not group or uid in seen_ids:
                raise ValueError(f'Missing or duplicate/cross-split identity: {uid}')
            if group in component_roles and component_roles[group] != role:
                raise ValueError(f'Component occurs in multiple splits: {group}')
            seen_ids.add(uid)
            component_roles[group] = role
        source_ids = {uid for uid, row in roles.items() if row['training_role_v2'] == role}
        if source_ids != {row['sequence_id'] for row in rows}:
            raise ValueError(f'Original and current role membership differs: {role}')
        split_rows[role] = rows
        current_input_records[role] = {'path': path, 'sha256': sha256_file(path),
                                      'columns_checked': fields}
    checked, sequences, statistics = {}, {}, {}
    for role, rows in split_rows.items():
        checked[role] = []
        specificity = {domain: Counter() for domain in ('nmnat', 'nudix')}
        for current in rows:
            uid = current['sequence_id']
            source = roles[uid]
            if (source['training_role_v2'] != role or source['split_v2'] != role or
                    source['component_v2'] != current['component_v2']):
                raise ValueError(f'Original role or component differs: {uid}')
            if (source.get('family_eligible_v2') is not True or
                    source.get('core_training_eligible_v2') is not True or
                    source.get('source_annotation_conflict') is not False):
                raise ValueError(f'Original record is not an eligible, unconflicted sequence: {uid}')
            sequence = _sequence(source['sequence'], uid)
            if uid not in source_population:
                raise ValueError(f'Original population record missing: {uid}')
            original = source_population[uid]
            if (_sequence(original['sequence'], uid) != sequence or
                    _integer(original['length'], 'length', uid) != len(sequence)):
                raise ValueError(f'Original population and role sequence differ: {uid}')
            if role != 'test' and uid not in current_population:
                raise ValueError(f'Current population lacks {role} sequence: {uid}')
            if uid in current_population and current_population[uid] != sequence:
                raise ValueError(f'Current population sequence differs: {uid}')
            restored = {name: source[name] for name in _FIELDS[:2]}
            restored.update({name: _integer(source[name], name, uid) for name in _FIELDS[2:]})
            _checked_record(restored, sequence)
            for name in _FIELDS:
                if name not in current:
                    continue
                value = current[name] if name in _FIELDS[:2] else _integer(current[name], name, uid)
                if value != restored[name]:
                    raise ValueError(f'Current {role} field differs from original {name}: {uid}')
            for domain in ('nmnat', 'nudix'):
                model = config['cdd_models'][domain]
                if domain == 'nmnat' and source['nmnat_cdd_accession'] != model['accession']:
                    raise ValueError(f'Original NMNAT accession differs: {uid}')
                specific = source[f'{domain}_cdd_specificity']
                key = (uid, model['accession'], model['short_name'], restored[f'{domain}_cdd_from'],
                       restored[f'{domain}_cdd_to'], specific)
                if cdd_hits[key] != 1:
                    raise ValueError(f'Missing or duplicate original {domain} CDD hit: {uid}')
                specificity[domain][specific] += 1
            checked[role].append(restored)
            sequences[uid] = sequence
        statistics[role] = {'sequences': len(rows),
                            'groups': len({row['component_v2'] for row in rows}),
                            'fields_checked_against_current_csv': list(rows[0]),
                            'unique_original_cdd_hits': 2 * len(rows),
                            'cdd_specificity': {name: dict(counts) for name, counts in specificity.items()}}
    population_path = config['current']['population']
    current_input_records['population'] = {'path': population_path,
                                            'sha256': sha256_file(population_path)}
    manifest = {
        'status': 'COMPLETED', 'method': 'restore archived sequence and CDD fields; no new annotation',
        'sources': source_records, 'current_inputs': current_input_records,
        'config': {'path': str(config_path), 'sha256': sha256_file(config_path)},
        'code_sha256': sha256_file(__file__),
        'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'column_sources': {
            'sequence_id': 'current split ID; must equal SHA256 of the full amino-acid sequence',
            'sequence': 'roles.sequence, independently matched to population.sequence',
            'component_v2': 'current split component, verified against roles.component_v2',
            'length': 'roles.length, verified against sequence length and population.length',
            'nmnat_cdd_from/nmnat_cdd_to': 'roles fields of the same name; verified against original CDD query_from/query_to',
            'nudix_cdd_from/nudix_cdd_to': 'roles fields of the same name; verified against original CDD query_from/query_to',
            'split': 'current split membership, roles.training_role_v2 and roles.split_v2 must agree',
        },
        'coordinate_system': 'CDD sequence coordinates are 1-based inclusive',
        'cdd_models': config['cdd_models'], 'checked_splits': statistics,
        'test_original_csv_had_cdd': set(_FIELDS[2:]) <= set(split_rows['test'][0]),
        'test_prediction_scoring': False, 'training_performed': False,
        'historical_wt_alignment_used': False,
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'test.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(checked['test'])
    with (output / 'test.fasta').open('w') as handle:
        for row in checked['test']:
            uid = row['sequence_id']
            handle.write(f'>{uid}\n{sequences[uid]}\n')
    manifest['outputs'] = {name: {'path': str(output / name), 'sha256': sha256_file(output / name)}
                           for name in ('test.csv', 'test.fasta')}
    write_json(output / 'manifest.json', manifest)
    return manifest


def load_recovered_test(data, directory):
    """Explicitly load restored test records, checking identity and split separation."""
    import json

    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['status'] != 'COMPLETED':
        raise ValueError('Test recovery is not complete')
    for name in ('test.csv', 'test.fasta'):
        if sha256_file(directory / name) != manifest['outputs'][name]['sha256']:
            raise ValueError(f'Restored test output SHA256 differs: {name}')
    expected = {}
    for row in data.test_rows:
        if row['sequence_id'] in expected:
            raise ValueError('Current test split contains duplicate IDs')
        expected[row['sequence_id']] = row['component']
    with (directory / 'test.csv').open(newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != _FIELDS:
            raise ValueError('Restored test CSV columns differ')
        rows = list(reader)
    ids = [row['sequence_id'] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError('Restored and current test ID sets differ')
    population = {}
    for record in SeqIO.parse(directory / 'test.fasta', 'fasta'):
        if record.id in population:
            raise ValueError('Restored test FASTA has duplicate IDs')
        population[record.id] = _sequence(str(record.seq), record.id)
    if set(population) != set(expected):
        raise ValueError('Restored test FASTA and CSV ID sets differ')
    records = []
    for row in rows:
        if row['component_v2'] != expected[row['sequence_id']]:
            raise ValueError(f'Restored and current test component differs: {row["sequence_id"]}')
        records.append(_checked_record(row, population[row['sequence_id']]))
    fitted_records = [*data.train, *data.validation]
    if set(expected) & {record.sequence_id for record in fitted_records}:
        raise ValueError('Restored test sequences overlap training or validation')
    if {record.component for record in records} & {record.component for record in fitted_records}:
        raise ValueError('Restored test components overlap training or validation')
    counts = manifest['checked_splits']['test']
    if (len(records) != counts['sequences'] or
            len({record.component for record in records}) != counts['groups']):
        raise ValueError('Restored test counts differ from recovery manifest')
    return records


def inspect_test_history(config_path, recovered_dir):
    """Match archived evaluation membership without parsing any score values.

    Reports and completion flags are insufficient on their own: each declared
    model must also have a pinned, complete per-sequence metrics table.
    """
    config = load_yaml(config_path)
    directory = Path(recovered_dir)
    destination = directory / 'historical_use.json'
    if destination.exists():
        raise FileExistsError(f'Test history already exists: {destination}')

    def pinned(item):
        path = Path(item['path'])
        actual = sha256_file(path)
        if actual != item['sha256']:
            raise ValueError(f'Historical source SHA256 differs: {path}')
        return {'path': str(path), 'sha256': actual, 'bytes': path.stat().st_size}

    def rows_from(path):
        with Path(path).open(newline='') as handle:
            reader = csv.DictReader(handle)
            if 'sequence_id' not in (reader.fieldnames or []):
                raise ValueError(f'Missing sequence_id column: {path}')
            return list(reader)

    def unique_index(rows, label):
        result = {}
        for row in rows:
            uid = row['sequence_id']
            if not uid or uid in result:
                raise ValueError(f'Missing or duplicate sequence ID: {label}')
            result[uid] = row
        return result

    current = pinned(config['current_test'])
    test = unique_index(rows_from(current['path']), 'current test')
    ids = set(test)
    if len(ids) != int(config['expected_test_sequences']):
        raise ValueError('Unexpected current test sequence count')
    recovery = json.loads((directory / 'manifest.json').read_text())
    if recovery.get('status') != 'COMPLETED':
        raise ValueError('Test recovery is not complete')
    for name in ('test.csv', 'test.fasta'):
        if sha256_file(directory / name) != recovery['outputs'][name]['sha256']:
            raise ValueError(f'Restored test output SHA256 differs: {name}')
    restored = unique_index(rows_from(directory / 'test.csv'), 'recovered test')
    if set(restored) != ids or any(
            restored[uid]['component_v2'] != test[uid]['component_v2'] for uid in ids):
        raise ValueError('Restored and current test IDs or components differ')

    membership_source = pinned(config['historical_membership'])
    members = unique_index(rows_from(membership_source['path']), 'historical membership')
    if set(members) != ids or any(
            members[uid]['component_v2'] != test[uid]['component_v2'] for uid in ids):
        raise ValueError('Historical membership and current test IDs or components differ')
    if any(row.get('prior_H1_39') not in ('True', 'False') or
           row.get('historically_pristine') != 'False' for row in members.values()):
        raise ValueError('Historical exposure flags are missing or unexpected')
    prior_ids = {uid for uid, row in members.items() if row['prior_H1_39'] == 'True'}
    if len(prior_ids) != int(config['expected_prior_h1_sequences']):
        raise ValueError('Unexpected previously evaluated H1 sequence count')

    def seed_value(value):
        return None if value is None else int(value)

    allowed_summary_fields = {'checkpoint_selection_calls', 'test_evaluation_rounds',
                              'route_changed_by_test', 'H1_reconfirmation', 'route_actions',
                              'automatic_rescue_tuning', 'pristine_claim', 'prior_H1_39', 'evidence'}
    evaluations = []
    for entry in config['prior_evaluations']:
        if not entry.get('metrics'):
            raise ValueError('Historical report alone is not execution evidence: metrics required')
        if entry['membership'] not in ('all_test', 'prior_H1_39'):
            raise ValueError('Unknown historical evaluation membership')
        expected = ids if entry['membership'] == 'all_test' else prior_ids
        sources = {name: pinned(entry[name]) for name in ('report', 'summary', 'job_status')}
        status = json.loads(Path(sources['job_status']['path']).read_text())
        if status.get('status') != 'COMPLETE':
            raise ValueError(f'Historical execution did not complete: {entry["name"]}')
        started = float(status['started_unix'])
        if not math.isfinite(started):
            raise ValueError('Historical execution time is invalid')
        expected_models = {(model['arm'], seed_value(seed))
                           for model in entry['models'] for seed in model['seeds']}
        observed, metric_sources = {}, []
        for item in entry['metrics']:
            source = pinned(item)
            with Path(source['path']).open(newline='') as handle:
                reader = csv.DictReader(handle)
                fields = set(reader.fieldnames or [])
                if (not {'sequence_id', 'component', 'CE_sum', 'CE'} <= fields or
                        not {'targets', 'target_count'} & fields):
                    raise ValueError('Historical metrics lack per-sequence score columns')
                for row in reader:
                    # Score and target-count values are deliberately not accessed.
                    model = (row['arm'], seed_value(row['seed'])) if {'arm', 'seed'} <= fields else (
                        item['arm'], seed_value(item['seed']))
                    uid = row['sequence_id']
                    if model not in expected_models or uid not in expected:
                        raise ValueError('Historical metrics model or sequence ID differs')
                    if row['component'] != test[uid]['component_v2']:
                        raise ValueError('Historical metrics component differs')
                    observed.setdefault(model, set())
                    if uid in observed[model]:
                        raise ValueError('Duplicate historical sequence ID for a model')
                    observed[model].add(uid)
            metric_sources.append(source)
        if set(observed) != expected_models or any(found != expected for found in observed.values()):
            raise ValueError('Historical metrics do not contain the complete expected ID sets')
        selected_fields = entry.get('summary_fields', [])
        if not set(selected_fields) <= allowed_summary_fields:
            raise ValueError('Only historical-use metadata may be extracted from summaries')
        summary = json.loads(Path(sources['summary']['path']).read_text())
        metadata = {name: summary[name] for name in selected_fields}
        evaluations.append({
            'name': entry['name'], 'membership': entry['membership'],
            'sequence_count': len(expected), 'model_evaluations': len(observed),
            'models': entry['models'], 'model_description': entry['model_description'],
            'started_utc': datetime.fromtimestamp(started, timezone.utc).isoformat(),
            'execution_status': 'COMPLETE', 'use_metadata': metadata,
            'source_files': sources, 'per_sequence_metrics': metric_sources,
            'sequence_ids_sha256': hashlib.sha256(('\n'.join(sorted(expected)) + '\n').encode()).hexdigest(),
        })
    if not evaluations or not any(e['membership'] == 'all_test' for e in evaluations):
        raise ValueError('No complete historical evaluation of the current test set was verified')
    if not any(e['membership'] == 'prior_H1_39' for e in evaluations):
        raise ValueError('H1 exposure flags lack matching per-sequence evaluation evidence')
    result = {
        'status': 'previously_evaluated', 'test_sequence_count': len(ids),
        'test_sequence_ids_sha256': hashlib.sha256(('\n'.join(sorted(ids)) + '\n').encode()).hexdigest(),
        'sequence_ids_hash_definition': 'UTF-8 sorted sequence IDs, joined by newline, including final newline',
        'prior_h1_sequence_count': len(prior_ids), 'prior_evaluations': evaluations,
        'current_test': current, 'historical_membership': membership_source,
        'recovery_manifest': {'path': str(directory / 'manifest.json'),
                              'sha256': sha256_file(directory / 'manifest.json')},
        'config': {'path': str(config_path), 'sha256': sha256_file(config_path)},
        'code_sha256': sha256_file(__file__),
        'flags': {'historically_pristine': False, 'new_model_scoring_performed': False,
                  'historical_score_values_interpreted': False, 'recovery_files_changed': False},
        'limitations': [
            'Membership and completed-execution artifacts establish previous evaluation; score correctness was not reassessed.',
            'These sources do not establish an exhaustive history of sequence use or PLM pretraining exposure.',
            'Old S/SE model labels describe residual sequence/evolution models, not the current self/external pretraining conditions.',
            'Exclusion from current fitting and checkpoint selection does not make this a historically unused test set.',
        ],
    }
    write_json(destination, result)
    return result
