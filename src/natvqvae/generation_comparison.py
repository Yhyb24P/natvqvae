"""Compare fixed-background generation runs across model arms and distributions."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .data import AA, PopulationData, load_yaml, sha256_file, stable_seed, write_json
from .evaluate import _bootstrap
from .fragment_comparison import _generation_descriptors, _require
from .preprocessing import stable_alignment
from .test_data import load_recovered_test

_STRATA = ('0', '1-2', '3-4', '5-8', 'over_8')
AA_INDEX = {residue: index for index, residue in enumerate(AA)}
_METRICS = (
    ('mean_Hamming', 'mean_Hamming_left', 'mean_Hamming_right', 'mean_Hamming_difference'),
    ('unique_sequences', 'unique_left', 'unique_right', 'unique_difference'),
    ('site_edit_rate', 'site_edit_rate_left', 'site_edit_rate_right', 'site_edit_rate_difference'),
    ('site_entropy', 'site_entropy_left', 'site_entropy_right', 'site_entropy_difference'),
    ('pair_covariance', 'pair_covariance_left', 'pair_covariance_right',
     'pair_covariance_difference'),
    ('target_domain_distance', 'target_domain_distance_left', 'target_domain_distance_right',
     'target_domain_distance_difference'),
    ('edit_mismatch_fraction', 'edit_mismatch_fraction_left', 'edit_mismatch_fraction_right',
     'edit_mismatch_fraction_difference'),
)
_DISTANCE_FIELDS = (
    'target_id', 'alignment_identity', 'coverage_background', 'coverage_training',
    'n_optimal', 'ambiguous', 'qualified_references', 'domain_edits',
    'domain_mapped', 'domain_mismatches', 'domain_uncovered',
    'target_domain_distance', 'edit_mapped', 'edit_uncovered', 'edit_mismatch_fraction',
)


def _run_specs(config):
    specs = []
    for condition in config['conditions']:
        for seed in config['seeds']:
            specs.extend([
                {'run_id': f'QVAE_{condition}_seed{seed}_joint', 'arm': 'QVAE',
                 'condition': condition, 'seed': int(seed), 'distribution': 'joint'},
                {'run_id': f'QVAE_{condition}_seed{seed}_site_marginal', 'arm': 'QVAE',
                 'condition': condition, 'seed': int(seed), 'distribution': 'site_marginal'},
                {'run_id': f'deterministic_{condition}_seed{seed}_factorized', 'arm': 'deterministic',
                 'condition': condition, 'seed': int(seed), 'distribution': 'factorized'},
            ])
    return specs


def _hamming_bin(value):
    return ('0' if value == 0 else '1-2' if value <= 2 else '3-4' if value <= 4 else
            '5-8' if value <= 8 else 'over_8')


def _read_rows(path, spec, background, masks):
    _require(path.is_file(), f'Missing generation output: {path}')
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    _require(len(rows) == len(masks) * background['draws'], f'Draw count differs: {path}')
    expected = {(mask['id'], draw) for mask in masks for draw in range(background['draws'])}
    seen, by_key = set(), {}
    for row in rows:
        key = row['group'], int(row['draw'])
        _require(key in expected and key not in seen, f'Unexpected/duplicate draw: {path}, {key}')
        _require(row['arm'] == spec['arm'] and int(row['seed']) == spec['seed'] and
                 row['distribution'] == spec['distribution'], f'Run identity differs: {path}, {key}')
        _require(row['region'] and int(row['mask_size']) > 0, f'Missing mask metadata: {path}, {key}')
        seen.add(key); by_key[key] = row
    _require(seen == expected, f'Missing draws: {path}')
    return rows, by_key


def _verify_sequence(row, target, mask):
    sequence = row['sequence']
    _require(len(sequence) == len(target) and set(sequence) <= set(AA), f'Bad generated sequence: {row["candidate_id"]}')
    _require(hashlib.sha256(sequence.encode()).hexdigest() == row['sequence_sha256'], f'Sequence SHA differs: {row["candidate_id"]}')
    edits = [{'position': i + 1, 'from': a, 'to': b}
             for i, (a, b) in enumerate(zip(target, sequence)) if a != b]
    _require(int(row['Hamming']) == len(edits) and json.loads(row['edits']) == edits,
             f'Edits differ: {row["candidate_id"]}')
    _require({e['position'] - 1 for e in edits} <= set(mask['positions']), f'Edit outside mask: {row["candidate_id"]}')
    _require(all(sequence[p] == target[p] for p in range(len(target)) if p not in set(mask['positions'])),
             f'Mask-exterior sequence changed: {row["candidate_id"]}')


def _site_and_pair_stats(rows, masks, target):
    by_group = defaultdict(list)
    for row in rows:
        by_group[row['group']].append(row)
    sites, pairs = [], []
    for mask in masks:
        group_rows = by_group[mask['id']]
        positions = mask['positions']
        edited = np.asarray([[int(p in {e['position'] - 1 for e in json.loads(row['edits'])}) for p in positions]
                             for row in group_rows], dtype=float)
        for index, position in enumerate(positions):
            residues = Counter(row['sequence'][position] for row in group_rows)
            sites.append({'group': mask['id'], 'position': position + 1, 'wt': target[position],
                          'retention': int(residues.get(target[position], 0)), 'draws': len(group_rows),
                          'residues': json.dumps(dict(sorted(residues.items())))})
        covariances = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                p_i, p_j = edited[:, i].mean(), edited[:, j].mean()
                covariances.append(float((edited[:, i] * edited[:, j]).mean() - p_i * p_j))
        pairs.append({'group': mask['id'], 'region': mask['region'], 'mask_size': mask['mask_size'],
                      'pairs': len(covariances),
                      'mean_edit_covariance': float(np.mean(covariances)) if covariances else None,
                      'mean_abs_edit_covariance': float(np.mean(np.abs(covariances))) if covariances else None,
                      'positive_pair_fraction': float(np.mean(np.asarray(covariances) > 0)) if covariances else None})
    return sites, pairs


def _site_pair_aggregate(rows, masks):
    by_group = defaultdict(list)
    for row in rows:
        by_group[row['group']].append(row)
    site_rates, covariances, entropies = [], [], []
    for mask in masks:
        group_rows = by_group[mask['id']]
        if not group_rows:
            continue
        sites = mask['positions']
        edited = np.asarray([[int(p in {e['position'] - 1 for e in json.loads(row['edits'])}) for p in sites]
                             for row in group_rows], dtype=float)
        site_rates.extend(edited.mean(axis=0).tolist())
        for position in sites:
            counts = np.asarray(list(Counter(row['sequence'][position] for row in group_rows).values()), dtype=float)
            probabilities = counts / counts.sum()
            entropies.append(float(-(probabilities * np.log(probabilities)).sum()))
        for i in range(len(sites)):
            for j in range(i + 1, len(sites)):
                p_i, p_j = edited[:, i].mean(), edited[:, j].mean()
                covariances.append(float((edited[:, i] * edited[:, j]).mean() - p_i * p_j))
    return {'mean_site_edit_rate': float(np.mean(site_rates)) if site_rates else None,
            'mean_site_entropy': float(np.mean(entropies)) if entropies else None,
            'mean_pair_edit_covariance': float(np.mean(covariances)) if covariances else None,
            'mean_abs_pair_edit_covariance': float(np.mean(np.abs(covariances))) if covariances else None,
            'positive_pair_fraction': float(np.mean(np.asarray(covariances) > 0)) if covariances else None}


def _domain_bounds(background, data):
    if background['record'] is None:
        low, high = data.target.seed_domains['NMNAT']
        return low - 1, high
    low, high = background['record'].intervals()['NMNAT']
    return low, high


def _read_fasta(path):
    sequences, name, chunks = {}, None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith('>'):
            if name is not None:
                sequences[name] = ''.join(chunks)
            name, chunks = line[1:].split()[0], []
        else:
            chunks.append(line.strip())
    if name is not None:
        sequences[name] = ''.join(chunks)
    return sequences


def _alignment_tables(config):
    root = Path(config['neighbors'])
    sequences = _read_fasta(root / 'nmnat.fasta')
    training_ids = set()
    with (root / 'nmnat_both_nearest.csv').open() as handle:
        for row in csv.DictReader(handle):
            if row['role'] == 'training':
                _require(row['sequence_id'] in sequences, f'Nearest-neighbour source missing: {row["sequence_id"]}')
                training_ids.add(row['sequence_id'])
    training = {identifier: sequences[identifier] for identifier in sorted(training_ids)}
    _require(len(training) == 1644, f'Expected 1644 training NMNAT domains, found {len(training)}')
    return sequences, training


def _neighbor_table(config, background_ids):
    path = Path(config['neighbors']) / 'nmnat_both_nearest.csv'
    source_sha = sha256_file(path)
    table = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row['sequence_id'] in background_ids and row['target_role'] == 'training':
                _require(row['sequence_id'] not in table,
                         f'Duplicate nearest-neighbour row: {row["sequence_id"]}')
                _require(row['target_id'], f'Nearest-neighbour target missing: {row["sequence_id"]}')
                table[row['sequence_id']] = {
                    'target_id': row['target_id'],
                    'fident': float(row['fident']) if row['fident'] else None,
                    'qcov': float(row['qcov']) if row['qcov'] else None,
                    'tcov': float(row['tcov']) if row['tcov'] else None,
                    'source_sha256': source_sha}
    _require(set(table) == set(background_ids),
             f'Nearest-neighbour rows missing for backgrounds: {sorted(set(background_ids) - set(table))}')
    return table


def _verify_background_manifest(manifest):
    for path, digest in manifest.get('input_sha256', {}).items():
        source = Path(path)
        _require(source.is_file() and sha256_file(source) == digest,
                 f'Generation background manifest input SHA differs: {path}')


def _verify_test_manifest(config):
    root = Path(config['test_data'])
    manifest = json.loads((root / 'manifest.json').read_text())
    _require(manifest.get('status') == 'COMPLETED', 'Test recovery manifest is not completed')
    for name in ('test.csv', 'test.fasta'):
        recorded = manifest['outputs'][name]['sha256']
        _require(recorded == sha256_file(root / name), f'Test recovery output SHA differs: {name}')
    for name in ('training', 'validation', 'population'):
        entry = manifest.get('current_inputs', {}).get(name)
        if entry:
            path = Path(entry['path'])
            _require(path.is_file() and sha256_file(path) == entry['sha256'],
                     f'Test recovery input SHA differs: {name}')
    return manifest


def _verify_neighbor_identity(config):
    root = Path(config['neighbors'])
    identity = json.loads((root / 'identity.json').read_text())
    for path, digest in identity.get('input_sha256', {}).items():
        source = Path(path)
        _require(source.is_file() and sha256_file(source) == digest,
                 f'Nearest-neighbour input SHA differs: {path}')
    for name, digest in identity.get('alignment_sha256', {}).items():
        source = root / name
        _require(source.is_file() and sha256_file(source) == digest,
                 f'Nearest-neighbour alignment SHA differs: {name}')
    return identity


def _stable_targets(background_domain, training_domains, max_optimal, minimum_coverage=0.80):
    targets = []
    for target_id in sorted(training_domains):
        target_domain = training_domains[target_id]
        alignment = stable_alignment(background_domain, target_domain, max_optimal)
        stable = alignment['stable']
        if not stable:
            continue
        coverage_background = len(stable) / len(background_domain)
        coverage_training = len(set(stable.values())) / len(target_domain)
        if coverage_background < minimum_coverage or coverage_training < minimum_coverage:
            continue
        targets.append({'target_id': target_id, 'mapping': stable,
                        'coverage_background': coverage_background,
                        'coverage_training': coverage_training,
                        'identity': alignment['identity'], 'n_optimal': alignment['n_optimal'],
                        'ambiguous': alignment['ambiguous']})
    return targets


def _reference_matrix(background_domain, targets, training_domains):
    length = len(background_domain)
    residues = np.full((len(targets), length), -1, dtype=np.int16)
    for index, target in enumerate(targets):
        target_domain = training_domains[target['target_id']]
        for background_index, target_index in target['mapping'].items():
            residues[index, background_index] = AA_INDEX[target_domain[target_index]]
    background = np.asarray([AA_INDEX[residue] for residue in background_domain], dtype=np.int16)
    mapped = (residues >= 0).sum(axis=1)
    base_mismatches = ((residues >= 0) & (residues != background[None, :])).sum(axis=1)
    return {'residues': residues, 'mapped': mapped, 'base_mismatches': base_mismatches,
            'targets': targets, 'background': background, 'length': length}


def _candidate_distance(row, references, domain_low, domain_high):
    if not references or not references['targets']:
        return {'distance_available': False, 'distance_status': 'no_qualified_reference'}
    residues = references['residues']
    targets = references['targets']
    ed = [edit for edit in json.loads(row['edits'])
          if domain_low <= edit['position'] - 1 < domain_high]
    delta = np.zeros(len(targets), dtype=np.int32)
    edit_mismatches = np.zeros(len(targets), dtype=np.int32)
    valid_counts = np.full(len(targets), len(ed), dtype=np.int32)
    for edit in ed:
        position = edit['position'] - 1
        background_index = position - domain_low
        residues_at_site = residues[:, background_index]
        valid = residues_at_site >= 0
        valid_counts -= (~valid).astype(np.int32)
        candidate = AA_INDEX[row['sequence'][position]]
        background = references['background'][background_index]
        delta += np.where(valid, (residues_at_site != candidate).astype(np.int32) -
                          (residues_at_site != background).astype(np.int32), 0)
        edit_mismatches += (valid & (residues_at_site != candidate)).astype(np.int32)
    mismatches = references['base_mismatches'] + delta
    distances = np.where(references['mapped'] > 0, mismatches / np.maximum(references['mapped'], 1), np.inf)
    best = int(np.argmin(distances))
    if not np.isfinite(distances[best]):
        return {'distance_available': False, 'distance_status': 'no_mapped_reference'}
    target = targets[best]
    edit_fraction = (float(edit_mismatches[best] / len(ed))
                     if ed and valid_counts[best] == len(ed) else None)
    return {'distance_available': True, 'distance_status': 'aligned',
            'target_id': target['target_id'], 'alignment_identity': target['identity'],
            'coverage_background': target['coverage_background'],
            'coverage_training': target['coverage_training'],
            'n_optimal': target['n_optimal'], 'ambiguous': target['ambiguous'],
            'qualified_references': len(targets), 'domain_edits': len(ed),
            'domain_mapped': int(references['mapped'][best]),
            'domain_mismatches': int(mismatches[best]),
            'domain_uncovered': int(references['length'] - references['mapped'][best]),
            'target_domain_distance': float(distances[best]),
            'edit_mapped': int(valid_counts[best]) if ed else 0,
            'edit_uncovered': int(len(ed) - valid_counts[best]) if ed else 0,
            'edit_mismatch_fraction': edit_fraction}


def _mean_available(values, key):
    found = [value[key] for value in values if value.get(key) is not None]
    return float(np.mean(found)) if found else None


def _distance_aggregate(rows, distance_by_key):
    values = [distance_by_key[row['group'], int(row['draw'])] for row in rows]
    available = [value for value in values if value.get('distance_available')]
    if not available:
        return {'distance_available': False, 'distance_rows': 0, 'distance_missing_rows': len(values),
                'target_id_modal': None, 'target_id_unique': 0, 'qualified_references': None,
                'alignment_identity': None, 'coverage_background': None, 'coverage_training': None,
                'n_optimal': None, 'ambiguous_fraction': None,
                'target_domain_distance': None, 'edit_mismatch_fraction': None,
                'domain_mapped': None, 'domain_uncovered': None,
                'edit_mapped': None, 'edit_uncovered': None}
    counts = Counter(value['target_id'] for value in available)
    modal, _ = counts.most_common(1)[0]
    return {'distance_available': True, 'distance_rows': len(available),
            'distance_missing_rows': len(values) - len(available),
            'target_id_modal': modal, 'target_id_unique': len(counts),
            'qualified_references': _mean_available(available, 'qualified_references'),
            'alignment_identity': _mean_available(available, 'alignment_identity'),
            'coverage_background': _mean_available(available, 'coverage_background'),
            'coverage_training': _mean_available(available, 'coverage_training'),
            'n_optimal': _mean_available(available, 'n_optimal'),
            'ambiguous_fraction': float(np.mean([value['ambiguous'] for value in available])),
            'target_domain_distance': _mean_available(available, 'target_domain_distance'),
            'edit_mismatch_fraction': _mean_available(available, 'edit_mismatch_fraction'),
            'domain_mapped': _mean_available(available, 'domain_mapped'),
            'domain_uncovered': _mean_available(available, 'domain_uncovered'),
            'edit_mapped': _mean_available(available, 'edit_mapped'),
            'edit_uncovered': _mean_available(available, 'edit_uncovered')}


def _difference(left, right):
    if left is None or right is None or left == 'NA' or right == 'NA':
        return 'NA'
    return left - right


def _stratum_metrics(rows, masks, distance_by_key):
    if not rows:
        return {'attempts': 0, 'unique_sequences': 0, 'unique_fraction': 'NA',
                'mean_site_edit_rate': 'NA', 'mean_site_entropy': 'NA',
                'mean_pair_edit_covariance': 'NA', 'mean_abs_pair_edit_covariance': 'NA',
                'positive_pair_fraction': 'NA', 'target_domain_distance': 'NA',
                'edit_mismatch_fraction': 'NA'}
    unique = len({row['sequence_sha256'] for row in rows})
    metrics = _site_pair_aggregate(rows, masks)
    distance = _distance_aggregate(rows, distance_by_key)
    return {'attempts': len(rows), 'unique_sequences': unique,
            'unique_fraction': unique / len(rows),
            'mean_site_edit_rate': metrics['mean_site_edit_rate'],
            'mean_site_entropy': metrics['mean_site_entropy'],
            'mean_pair_edit_covariance': metrics['mean_pair_edit_covariance'],
            'mean_abs_pair_edit_covariance': metrics['mean_abs_pair_edit_covariance'],
            'positive_pair_fraction': metrics['positive_pair_fraction'],
            'target_domain_distance': distance['target_domain_distance'],
            'edit_mismatch_fraction': distance['edit_mismatch_fraction']}


def _matched_pair(left_rows, right_rows, stratum, rng_parts):
    left = [row for row in left_rows if _hamming_bin(int(row['Hamming'])) == stratum]
    right = [row for row in right_rows if _hamming_bin(int(row['Hamming'])) == stratum]
    paired = min(len(left), len(right))
    if not paired:
        return [], [], len(left), len(right), 0, None, None
    left_seed = stable_seed(*rng_parts, 'left')
    right_seed = stable_seed(*rng_parts, 'right')
    left_choice = np.sort(np.random.default_rng(left_seed).choice(len(left), paired, replace=False))
    right_choice = np.sort(np.random.default_rng(right_seed).choice(len(right), paired, replace=False))
    return ([left[int(index)] for index in left_choice],
            [right[int(index)] for index in right_choice],
            len(left), len(right), paired, left_seed, right_seed)


def _add_contrast(comparisons, hamming_rows, label, kind, condition, left, right,
                  background, seed, raw, distances):
    lrows, rrows = raw[left, background['id']], raw[right, background['id']]
    left_h = [int(row['Hamming']) for row in lrows]
    right_h = [int(row['Hamming']) for row in rrows]
    left_metrics = _site_pair_aggregate(lrows, background['masks'])
    right_metrics = _site_pair_aggregate(rrows, background['masks'])
    left_distance = _distance_aggregate(lrows, distances[left, background['id']])
    right_distance = _distance_aggregate(rrows, distances[right, background['id']])
    comparisons.append({
        'comparison': label, 'contrast_kind': kind, 'condition': condition,
        'left_run': left, 'right_run': right, 'background': background['id'],
        'component': background['component'], 'seed': seed,
        'attempts_left': len(lrows), 'attempts_right': len(rrows),
        'mean_Hamming_left': float(np.mean(left_h)),
        'mean_Hamming_right': float(np.mean(right_h)),
        'mean_Hamming_difference': float(np.mean(left_h) - np.mean(right_h)),
        'unique_left': len({row['sequence_sha256'] for row in lrows}),
        'unique_right': len({row['sequence_sha256'] for row in rrows}),
        'unique_difference': len({row['sequence_sha256'] for row in lrows}) -
                             len({row['sequence_sha256'] for row in rrows}),
        'site_edit_rate_left': left_metrics['mean_site_edit_rate'],
        'site_edit_rate_right': right_metrics['mean_site_edit_rate'],
        'site_edit_rate_difference': _difference(left_metrics['mean_site_edit_rate'],
                                                 right_metrics['mean_site_edit_rate']),
        'site_entropy_left': left_metrics['mean_site_entropy'],
        'site_entropy_right': right_metrics['mean_site_entropy'],
        'site_entropy_difference': _difference(left_metrics['mean_site_entropy'],
                                               right_metrics['mean_site_entropy']),
        'pair_covariance_left': left_metrics['mean_pair_edit_covariance'],
        'pair_covariance_right': right_metrics['mean_pair_edit_covariance'],
        'pair_covariance_difference': _difference(left_metrics['mean_pair_edit_covariance'],
                                                  right_metrics['mean_pair_edit_covariance']),
        'target_domain_distance_left': left_distance['target_domain_distance'],
        'target_domain_distance_right': right_distance['target_domain_distance'],
        'target_domain_distance_difference': _difference(left_distance['target_domain_distance'],
                                                         right_distance['target_domain_distance']),
        'edit_mismatch_fraction_left': left_distance['edit_mismatch_fraction'],
        'edit_mismatch_fraction_right': right_distance['edit_mismatch_fraction'],
        'edit_mismatch_fraction_difference': _difference(left_distance['edit_mismatch_fraction'],
                                                         right_distance['edit_mismatch_fraction']),
    })
    for mask in background['masks']:
        left_group = [row for row in lrows if row['group'] == mask['id']]
        right_group = [row for row in rrows if row['group'] == mask['id']]
        for stratum in _STRATA:
            lm, rm, left_original, right_original, paired, left_seed, right_seed = _matched_pair(
                left_group, right_group, stratum,
                (left, right, background['id'], mask['id'], stratum, seed))
            left_metric = _stratum_metrics(lm, [mask], distances[left, background['id']])
            right_metric = _stratum_metrics(rm, [mask], distances[right, background['id']])
            hamming_rows.append({
                'comparison': label, 'contrast_kind': kind, 'condition': condition,
                'left_run': left, 'right_run': right, 'background': background['id'],
                'component': background['component'], 'seed': seed,
                'mask': mask['id'], 'mask_size': mask['mask_size'], 'stratum': stratum,
                'attempts_left_original': left_original,
                'attempts_right_original': right_original,
                'paired_attempts': paired,
                'pair_seed_left': left_seed,
                'pair_seed_right': right_seed,
                'discarded_left': left_original - paired,
                'discarded_right': right_original - paired,
                'unique_left': left_metric['unique_sequences'] if paired else 'NA',
                'unique_right': right_metric['unique_sequences'] if paired else 'NA',
                'unique_difference': (_difference(left_metric['unique_sequences'],
                                                  right_metric['unique_sequences'])
                                      if paired else 'NA'),
                'site_edit_rate_left': left_metric['mean_site_edit_rate'] if paired else 'NA',
                'site_edit_rate_right': right_metric['mean_site_edit_rate'] if paired else 'NA',
                'site_edit_rate_difference': (_difference(left_metric['mean_site_edit_rate'],
                                                          right_metric['mean_site_edit_rate'])
                                              if paired else 'NA'),
                'site_entropy_left': left_metric['mean_site_entropy'] if paired else 'NA',
                'site_entropy_right': right_metric['mean_site_entropy'] if paired else 'NA',
                'site_entropy_difference': (_difference(left_metric['mean_site_entropy'],
                                                        right_metric['mean_site_entropy'])
                                            if paired else 'NA'),
                'pair_covariance_left': left_metric['mean_pair_edit_covariance'] if paired else 'NA',
                'pair_covariance_right': right_metric['mean_pair_edit_covariance'] if paired else 'NA',
                'pair_covariance_difference': (_difference(left_metric['mean_pair_edit_covariance'],
                                                           right_metric['mean_pair_edit_covariance'])
                                               if paired else 'NA'),
                'target_domain_distance_left': left_metric['target_domain_distance'] if paired else 'NA',
                'target_domain_distance_right': right_metric['target_domain_distance'] if paired else 'NA',
                'target_domain_distance_difference': (
                    _difference(left_metric['target_domain_distance'],
                                right_metric['target_domain_distance']) if paired else 'NA'),
                'edit_mismatch_fraction_left': left_metric['edit_mismatch_fraction'] if paired else 'NA',
                'edit_mismatch_fraction_right': right_metric['edit_mismatch_fraction'] if paired else 'NA',
                'edit_mismatch_fraction_difference': (
                    _difference(left_metric['edit_mismatch_fraction'],
                                right_metric['edit_mismatch_fraction']) if paired else 'NA'),
            })


def _component_effects(comparisons, entries, seeds, contrast_meta, config):
    analysis = config.get('uncertainty', {})
    repeats = int(analysis.get('bootstrap_repeats', 2000))
    bootstrap_seed = int(analysis.get('bootstrap_seed', 8802))
    _require(repeats > 0, 'Bootstrap repetitions must be positive')
    by_component = defaultdict(list)
    for entry in entries:
        if entry['record'] is not None:
            by_component[entry['component']].append(entry['id'])
    components = sorted(by_component)
    lookup = {(row['comparison'], row['contrast_kind'], row['condition'],
               row['background'], int(row['seed'])): row for row in comparisons}
    logical, seen = [], set()
    for label, kind, condition, _, _ in contrast_meta:
        key = label, kind, condition
        if key not in seen:
            seen.add(key); logical.append(key)
    rows = []

    def valid(values):
        return [float(value) for value in values if value != 'NA' and value is not None]

    def append_row(label, kind, condition, left, right, left_runs, right_runs,
                   unit, seed, metric, found, left_values, right_values):
        found = valid(found)
        left_values = valid(left_values)
        right_values = valid(right_values)
        is_kvp40 = unit == 'KVP40_application_background'
        n_found = (1 if found else 0) if is_kvp40 else len(found)
        row = {
            'comparison': label, 'contrast_kind': kind, 'condition': condition,
            'left_run': left, 'right_run': right,
            'left_runs': left_runs, 'right_runs': right_runs,
            'analysis_unit': unit, 'seed': seed,
            'metric': metric, 'components': n_found,
            'components_missing': (0 if found else 1) if is_kvp40 else len(components) - n_found,
            'left_mean': float(np.mean(left_values)) if left_values else 'NA',
            'right_mean': float(np.mean(right_values)) if right_values else 'NA',
        }
        if not found:
            row.update({'mean': 'NA', 'ci95_low': 'NA', 'ci95_high': 'NA',
                        'components_left_lower': 'NA', 'components_left_higher': 'NA'})
        elif is_kvp40:
            row.update({'mean': float(np.mean(found)), 'ci95_low': 'NA', 'ci95_high': 'NA',
                        'components_left_lower': 'NA', 'components_left_higher': 'NA'})
        elif len(found) == 1:
            row.update({'mean': found[0], 'ci95_low': 'NA', 'ci95_high': 'NA',
                        'components_left_lower': int(found[0] < 0),
                        'components_left_higher': int(found[0] > 0)})
        else:
            boot = _bootstrap(found, reps=repeats, seed=bootstrap_seed)
            row.update({'mean': boot['mean'], 'ci95_low': boot['ci95'][0],
                        'ci95_high': boot['ci95'][1],
                        'components_left_lower': sum(value < 0 for value in found),
                        'components_left_higher': sum(value > 0 for value in found)})
        rows.append(row)

    kvp40 = next((entry for entry in entries if entry['record'] is None), None)
    for label, kind, condition in logical:
        per_seed = {}
        for seed in [int(value) for value in seeds]:
            available = {}
            left_runs, right_runs = set(), set()
            for component in components:
                values = [lookup[label, kind, condition, background_id, seed]
                          for background_id in by_component[component]]
                for row in values:
                    left_runs.add(row['left_run']); right_runs.add(row['right_run'])
                available[component] = {
                    metric: (float(np.mean([row[diff_key] for row in values]))
                             if all(row[diff_key] not in ('NA', None) for row in values) else None)
                    for metric, _, _, diff_key in _METRICS}
                available[component].update({
                    f'{metric}_left': (float(np.mean([row[left_key] for row in values]))
                                       if all(row[left_key] not in ('NA', None) for row in values) else None)
                    for metric, left_key, _, _ in _METRICS})
                available[component].update({
                    f'{metric}_right': (float(np.mean([row[right_key] for row in values]))
                                        if all(row[right_key] not in ('NA', None) for row in values) else None)
                    for metric, _, right_key, _ in _METRICS})
            left_run = ';'.join(sorted(left_runs))
            right_run = ';'.join(sorted(right_runs))
            per_seed[seed] = (available, left_run, right_run)
            for metric, left_key, right_key, diff_key in _METRICS:
                found = [available[component][metric] for component in components
                         if available[component][metric] is not None]
                left_values = [available[component][f'{metric}_left'] for component in components
                               if available[component][f'{metric}_left'] is not None]
                right_values = [available[component][f'{metric}_right'] for component in components
                                if available[component][f'{metric}_right'] is not None]
                append_row(label, kind, condition, left_run, right_run, left_run, right_run,
                           'heldout_component', seed, metric, found, left_values, right_values)
            if kvp40 is not None:
                source = lookup[label, kind, condition, kvp40['id'], seed]
                for metric, left_key, right_key, diff_key in _METRICS:
                    found = [] if source[diff_key] == 'NA' else [source[diff_key]]
                    append_row(label, kind, condition, source['left_run'], source['right_run'],
                               source['left_run'], source['right_run'], 'KVP40_application_background',
                               seed, metric, found, [source[left_key]], [source[right_key]])
        pooled_left = sorted({per_seed[int(seed)][1] for seed in seeds})
        pooled_right = sorted({per_seed[int(seed)][2] for seed in seeds})
        for metric, left_key, right_key, diff_key in _METRICS:
            pooled = []
            for component in components:
                values = [lookup[label, kind, condition, background_id, int(seed)][diff_key]
                          for background_id in by_component[component]
                          for seed in seeds]
                pooled.append(float(np.mean(values)) if all(value != 'NA' for value in values) else None)
            found = [value for value in pooled if value is not None]
            append_row(label, kind, condition, 'pooled_across_seeds', 'pooled_across_seeds',
                       ';'.join(pooled_left), ';'.join(pooled_right),
                       'heldout_component', 'pooled', metric, found, [], [])
        if kvp40 is not None:
            for metric, left_key, right_key, diff_key in _METRICS:
                values = [lookup[label, kind, condition, kvp40['id'], int(seed)][diff_key]
                          for seed in seeds]
                found = [value for value in values if value != 'NA']
                append_row(label, kind, condition, 'pooled_across_seeds', 'pooled_across_seeds',
                           ';'.join(pooled_left), ';'.join(pooled_right),
                           'KVP40_application_background', 'pooled', metric, found, [], [])
    return rows, {'unit': 'component', 'repetitions': repeats, 'seed': bootstrap_seed,
                  'scope': 'within-seed equal-component bootstrap; pooled first averages seeds within component; '
                           'pooled run fields list all contributing runs and do not represent one run; '
                           'KVP40 is a separate application background and not a 45th component',
                  'components': len(components)}


def _write_csv(path, rows):
    _require(rows, f'No rows to write: {path}')
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _write_generation_outputs(component_rows, run_rows, hamming_rows, output, config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    metrics = [
        ('mean_Hamming', 'Mean substitutions'),
        ('unique_sequences', 'Unique sequences'),
        ('pair_covariance', 'Mean edit covariance'),
    ]
    groups = [
        ('External minus self', [
            ('QVAE_joint_external_minus_self', 'external_minus_self', 'QVAE joint'),
            ('deterministic_factorized_external_minus_self', 'external_minus_self', 'Deterministic'),
        ]),
        ('Joint minus own site marginals', [
            ('joint_minus_site_marginal', 'random', 'Random'),
            ('joint_minus_site_marginal', 'self', 'Self'),
            ('joint_minus_site_marginal', 'external', 'External'),
        ]),
    ]
    seeds = [int(value) for value in config['seeds']]
    heldout = [row for row in component_rows if row['analysis_unit'] == 'heldout_component']
    kvp40 = [row for row in component_rows if row['analysis_unit'] == 'KVP40_application_background']
    lookup = {(row['comparison'], row['condition'], row['seed'], row['metric']): row for row in heldout}
    kvp_lookup = {(row['comparison'], row['condition'], row['seed'], row['metric']): row
                  for row in kvp40}

    def numeric(value):
        return None if value in (None, 'NA') else float(value)

    def estimate(label, condition, metric):
        rows = [lookup[label, condition, seed, metric] for seed in seeds]
        pooled = lookup[label, condition, 'pooled', metric]
        return pooled, rows

    selected = []
    for _, specs in groups:
        for label, condition, _ in specs:
            for metric, _ in metrics:
                pooled, per_seed = estimate(label, condition, metric)
                selected.extend([pooled, *per_seed])
    _write_csv(output / 'generation_effects.csv', selected)

    colors = ('#2878A1', '#CA6A26', '#448C62')
    with plt.rc_context({'font.size': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig, axes = plt.subplots(2, 3, figsize=(11, 6), layout='constrained',
                                 gridspec_kw={'height_ratios': [2, 3]})
        for panel_index, (panel_title, specs) in enumerate(groups):
            for column, (metric, metric_label) in enumerate(metrics):
                axis = axes[panel_index, column]
                axis.axvline(0, color='.65', lw=.8, linestyle='--')
                for index, (label, condition, row_label) in enumerate(specs):
                    y = len(specs) - 1 - index
                    pooled, per_seed = estimate(label, condition, metric)
                    mean = numeric(pooled['mean'])
                    low, high = numeric(pooled['ci95_low']), numeric(pooled['ci95_high'])
                    if low is not None and high is not None:
                        axis.hlines(y, low, high, color='.15', lw=1.2)
                    if mean is not None:
                        axis.plot(mean, y, 'o', color='.15', ms=4)
                    for seed_index, row in enumerate(per_seed):
                        value = numeric(row['mean'])
                        if value is not None:
                            axis.plot(value, y - .20 + seed_index * .06, '^',
                                      color=colors[seed_index], ms=3)
                axis.set_yticks(range(len(specs)), [label for _, _, label in reversed(specs)])
                if column:
                    axis.tick_params(axis='y', labelleft=False)
                axis.set_ylim(-.5, len(specs) - .4)
                axis.set_title(metric_label)
                if column == 0:
                    axis.set_ylabel(panel_title)
                axis.set_xlabel('difference (first minus second)')
                axis.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3), useOffset=False)
                axis.spines[['top', 'right']].set_visible(False)
        handles = [Line2D([], [], marker='o', color='.15', label='Pooled equal-component mean and 95% interval')]
        handles.extend(Line2D([], [], marker='^', color=color, linestyle='', label=f'Seed {seed}')
                       for seed, color in zip(seeds, colors))
        fig.legend(handles=handles, loc='outside upper center', ncol=4, frameon=False)
        fig.supxlabel('Held-out components: differences are descriptive; intervals condition on three fitted pipeline replicates',
                      fontsize=9)
        for extension in ('pdf', 'png'):
            fig.savefig(output / f'generation_effects.{extension}', dpi=220, bbox_inches='tight')
        plt.close(fig)

    paired_total = sum(int(row['paired_attempts']) for row in hamming_rows
                       if row['paired_attempts'] != 'NA')
    empty_rows = sum(1 for row in hamming_rows if int(row['paired_attempts']) == 0)
    discarded = sum(int(row['discarded_left']) + int(row['discarded_right']) for row in hamming_rows)
    na_unique = sum(1 for row in hamming_rows if row['unique_difference'] == 'NA')
    runs = len({row['run'] for row in run_rows})
    backgrounds = len({row['background'] for row in run_rows})
    draws = int(config['draws_per_mask'])
    attempts_per_run = backgrounds * 6 * draws
    lines = ['# NatV 六格条件生成比较', '',
             f'评价对象：{backgrounds - 1} 个 held-out component 代表序列，加 KVP40 应用背景（KVP40 不是第 45 个 component）。'
             f'每个背景 6 个 NMNAT 掩码（8、16、32 位各 2 个），每个掩码 {draws} 次抽样，温度 1；'
             f'每背景 {6 * draws} 次、每个 fitted-model distribution {attempts_per_run} 次、'
             f'{runs} 个 run 共 {runs * attempts_per_run} 次核验尝试。', '',
             '差值为比较名称中前项减后项。Hamming 差值先在每背景的 384 次尝试上取平均替换数，'
             '再对 held-out component 等权；unique 差值是每背景 384 次尝试中不同整序列数之差；'
             'pair covariance 差值先对每掩码每对用 64 次抽样计算 E[I_i I_j]−E[I_i]E[I_j]，'
             '再对 6 个掩码的全部 1,288 个位点对等权（大掩码贡献更多对，不是每掩码等权）。'
             'pooled 值先在同 component 内平均三个下游 seed，再对 component 等权。'
             '区间是固定三个 fitted pipeline replicate 后的 equal-component bootstrap。'
             '逐 seed 三角是三个完整训练重复，不是独立生物样本。', '',
             '![条件生成差值](generation_effects.png)', '',
             '编辑数均值、唯一序列数和配对编辑协方差是描述量。编辑数只依赖逐位点边缘，'
             '不能据此证明共享潜变量的联合建模收益；唯一序列数接近抽样上限时区分力有限；'
             '协方差是掩码内位点对的经验描述，不是因果或功能指标。差异不自动表示生物质量、活性或候选优劣。', '',
             '## 原始总体与 Hamming 匹配', '',
             f'`run_summary.csv` 保留每个背景/run 的原始总体（未跨 component 汇总）；本报告只读取已保存的 '
             f'`component_effects.csv` 和 `hamming_paired.csv`，没有重采样或重新评分。',
             f'Hamming 匹配在每个 background/seed/mask/stratum 内独立抽样，每侧取 n=min(n_left,n_right)，'
             f'保持相等尝试预算；空层为 NA，丢弃量见 `discarded_left/right`。'
             f'本报告累计 {paired_total} 个匹配对、{empty_rows} 个空层、{na_unique} 个 unique 差值 NA、'
             f'丢弃计数合计 {discarded}；这些是多个比较条目复用同一批候选的累计抽样计数，'
             f'不是互不重叠的候选总数，也不是独立蛋白或生物学重复。', '',
             '## 候选距离', '',
             '候选距离使用背景序列的固定 stable alignment 映射：BLOSUM62、open gap −10、extend gap −0.5、'
             'max_optimal 来自 config，双方 stable coverage ≥0.80；候选编辑替换到固定映射后，对所有合格训练参考取最小值。'
             '`edit_mismatch_fraction` 描述候选编辑位点上候选与最近对齐训练参考的差异，'
             '不是以编辑位点单独重新搜索的最近邻，也不因候选编辑重新对齐。', '',
             '## Held-out component 估计', '',
             '| 比较 | 标签 | metric | 平均差值 | 95% 区间 | components |',
             '| --- | --- | --- | ---: | --- | ---: |']
    for panel_title, specs in groups:
        for label, condition, row_label in specs:
            for metric, metric_label in metrics:
                pooled, _ = estimate(label, condition, metric)
                low, high = numeric(pooled['ci95_low']), numeric(pooled['ci95_high'])
                interval = 'NA' if low is None or high is None else f'[{low:+.8f}, {high:+.8f}]'
                mean = numeric(pooled['mean'])
                lines.append(f'| {panel_title} | {row_label} | {metric_label} | '
                             f'{"NA" if mean is None else f"{mean:+.8f}"} | {interval} | {pooled["components"]} |')
    lines.extend(['', '## KVP40 应用背景（单独报告，不进入 component 区间）', '',
                  '| 比较 | 标签 | metric | 逐 seed 均值 | pooled 均值 | 有效 seed | 区间 |',
                  '| --- | --- | --- | --- | ---: | ---: | --- |'])
    for panel_title, specs in groups:
        for label, condition, row_label in specs:
            for metric, metric_label in metrics:
                values = [numeric(kvp_lookup[label, condition, seed, metric]['mean']) for seed in seeds]
                pooled = numeric(kvp_lookup[label, condition, 'pooled', metric]['mean'])
                shown = ', '.join('NA' if value is None else f'{value:+.8f}' for value in values)
                lines.append(f'| {panel_title} | {row_label} | {metric_label} | {shown} | '
                             f'{"NA" if pooled is None else f"{pooled:+.8f}"} | '
                             f'{sum(value is not None for value in values)} | NA |')
    lines.extend(['', 'KVP40 是单一应用背景，不是独立 component；其区间不定义，pooled 值只平均有效 seed。'
                  'KVP40 的逐 seed 差异不能当作 44 个 held-out component 的推断。', '',
                  '`generation_effects.csv` 保存本图使用的 pooled 与逐 seed 估计。'
                  '全部比较、分层和 KVP40 记录见 `component_effects.csv`。'])
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def compare_generation(config_path, output):
    config = load_yaml(config_path)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Generation comparison output already contains files: {output}')
    _require(set(config['conditions']) == {'random', 'self', 'external'},
             'Generation comparison requires random, self and external conditions')
    data = PopulationData(config['sources'], config['study_config'], config['target'])
    manifest = json.loads(Path(config['backgrounds']).read_text())
    _verify_background_manifest(manifest)
    test_manifest = _verify_test_manifest(config)
    _verify_neighbor_identity(config)
    records = {record.sequence_id: record for record in load_recovered_test(data, config['test_data'])}
    entries = []
    for row in manifest['backgrounds']:
        _require(row['sequence_id'] in records, f'Background sequence missing: {row["sequence_id"]}')
        record = records[row['sequence_id']]
        _require(hashlib.sha256(record.sequence.encode()).hexdigest() == row['sequence_id'],
                 f'Background sequence ID differs from sequence SHA: {row["sequence_id"]}')
        entries.append({'id': row['sequence_id'], 'component': row['component'], 'masks': row['masks'],
                        'generation': row['generation'], 'record': record})
    _require(len(entries) == len({entry['component'] for entry in entries}),
             'Held-out backgrounds must represent distinct components')
    _require(len(entries) == int(config.get('expected_components', len(entries))),
             f'Expected {config.get("expected_components", len(entries))} held-out components, '
             f'found {len(entries)}')
    entries.append({'id': config['kvp40_id'], 'component': config['kvp40_id'],
                    'masks': manifest['kvp40']['masks'],
                    'generation': manifest['kvp40']['generation'], 'record': None})
    specs = _run_specs(config)
    _require(len(specs) == 3 * len(config['conditions']) * len(config['seeds']), 'Run specification count differs')
    target_wt = data.target.sequence
    _, training_domains = _alignment_tables(config)
    neighbor = _neighbor_table(config, {entry['id'] for entry in entries if entry['record'] is not None})
    for background in entries:
        if background['record'] is not None:
            _require(neighbor[background['id']]['target_id'] in training_domains,
                     f'Nearest-neighbour target is not a training NMNAT domain: {background["id"]}')
    max_optimal = int(config.get('max_optimal_alignments', 16))
    domain_references, domain_bounds = {}, {}
    for background in entries:
        target = target_wt if background['record'] is None else background['record'].sequence
        low, high = _domain_bounds(background, data)
        domain = target[low:high]
        targets = _stable_targets(domain, training_domains, max_optimal)
        domain_references[background['id']] = _reference_matrix(domain, targets, training_domains)
        domain_bounds[background['id']] = (low, high)
    checkpoints, training_hashes = {}, {}
    for spec in specs:
        train_root = Path(config['training'][spec['arm']])
        name = f"{config.get('run_prefix', {}).get(spec['arm'], '')}{spec['condition']}_seed{spec['seed']}"
        summary_path = train_root / name / 'training_summary.json'
        summary = json.loads(summary_path.read_text())
        _require(summary['status'] == 'COMPLETED' and summary.get('arm', 'QVAE') == spec['arm'] and
                 summary['seed'] == spec['seed'] and summary['steps'] == int(config['downstream_steps']),
                 f'Training summary differs: {name}')
        best = summary['checkpoints']['best']
        _require(best['sha256'] == sha256_file(best['path']), f'Checkpoint SHA differs: {name}')
        checkpoints[spec['run_id']] = best
        training_hashes[str(summary_path)] = sha256_file(summary_path)
    run_rows, run_meta, site_rows, pair_rows = [], [], [], []
    comparisons, hamming_rows, candidate_rows = [], [], []
    raw, distances = {}, {}
    generation_hashes = {}
    for spec in specs:
        for background in entries:
            directory = Path(config['generation_root']) / spec['run_id'] / background['id']
            _require(directory.is_dir(), f'Missing generation directory: {directory}')
            masks = background['masks']
            _require(len(masks) == 6 and Counter(m['mask_size'] for m in masks) == Counter({8: 2, 16: 2, 32: 2}),
                     f'Background mask layout differs: {background["id"]}')
            target = target_wt if background['record'] is None else background['record'].sequence
            draws = int(background['generation']['draws_per_mask'])
            _require(draws == int(config['draws_per_mask']), f'Draw budget differs: {background["id"]}')
            rows, _ = _read_rows(directory / 'candidates.csv', spec, {**background, 'draws': draws}, masks)
            for name in ('candidates.csv', 'candidates.fasta', 'run_identity.json',
                         'generation.json', 'masks.json'):
                generation_hashes[str(directory / name)] = sha256_file(directory / name)
            identity = json.loads((directory / 'run_identity.json').read_text())
            background_identity = identity['background_id']
            expected_background = background['id'] if background['record'] is not None else None
            _require(identity['checkpoint_arm'] == spec['arm'] and identity['checkpoint_seed'] == spec['seed'] and
                     identity['distribution'] == spec['distribution'] and background_identity == expected_background and
                     identity['target_sha256'] == hashlib.sha256(target.encode()).hexdigest() and
                     identity['generation'] == background['generation'] and
                     sha256_file(directory / 'masks.json') == identity['masks_sha256'] and
                     identity['checkpoint_sha256'] == checkpoints[spec['run_id']]['sha256'],
                     f'Run identity differs: {directory}')
            _require(json.loads((directory / 'masks.json').read_text()) == masks, f'Mask identity differs: {directory}')
            generation = json.loads((directory / 'generation.json').read_text())
            _require(generation['status'] == 'COMPLETED' and generation['groups'] == 6 and
                     generation['attempts'] == 6 * draws and generation['draws_per_mask'] == draws and
                     generation['target_sha256'] == identity['target_sha256'] and
                     generation['checkpoint_sha256'] == checkpoints[spec['run_id']]['sha256'],
                     f'Incomplete generation: {directory}')
            by_group = defaultdict(list)
            for row in rows:
                mask = next(item for item in masks if item['id'] == row['group'])
                _verify_sequence(row, target, mask)
                by_group[row['group']].append(row)
            sites, pairs = _site_and_pair_stats(rows, masks, target)
            for row in sites:
                site_rows.append({'background': background['id'], 'component': background['component'],
                                  'run': spec['run_id'], **row})
            for row in pairs:
                pair_rows.append({'background': background['id'], 'component': background['component'],
                                  'run': spec['run_id'], **row})
            low, high = domain_bounds[background['id']]
            distance_by_key = {(row['group'], int(row['draw'])):
                               _candidate_distance(row, domain_references[background['id']], low, high)
                               for row in rows}
            distances[spec['run_id'], background['id']] = distance_by_key
            for row in rows:
                distance = distance_by_key[row['group'], int(row['draw'])]
                candidate_rows.append({
                    'background': background['id'], 'component': background['component'],
                    'run': spec['run_id'], 'arm': spec['arm'], 'condition': spec['condition'],
                    'seed': spec['seed'], 'distribution': spec['distribution'],
                    'group': row['group'], 'region': row['region'],
                    'mask_size': int(row['mask_size']), 'draw': int(row['draw']),
                    'candidate_id': row['candidate_id'], 'Hamming': int(row['Hamming']),
                    'distance_available': distance['distance_available'],
                    'distance_status': distance['distance_status'],
                    **{field: distance.get(field) for field in _DISTANCE_FIELDS}})
            descriptor = _generation_descriptors(rows)
            descriptor.update({'background': background['id'], 'component': background['component'],
                               'run': spec['run_id'], 'arm': spec['arm'],
                               'condition': spec['condition'], 'seed': spec['seed'],
                               'distribution': spec['distribution'], 'masks': 6, 'draws_per_mask': draws,
                               'background_nearest_training_id': (
                                   neighbor[background['id']]['target_id']
                                   if background['record'] is not None else None),
                               'background_nearest_training_identity': (
                                   neighbor[background['id']]['fident']
                                   if background['record'] is not None else None),
                               **_distance_aggregate(rows, distance_by_key)})
            run_rows.append(descriptor)
            for group in masks:
                group_rows = by_group[group['id']]
                descriptor = _generation_descriptors(group_rows)
                descriptor.update({'background': background['id'], 'component': background['component'],
                                   'run': spec['run_id'], 'group': group['id'], 'region': group['region'],
                                   'mask_size': group['mask_size'],
                                   **_distance_aggregate(group_rows, distance_by_key)})
                run_meta.append(descriptor)
            raw[spec['run_id'], background['id']] = rows
    # Common random numbers: the same uniform seed must pair every draw across all
    # conditions, arms and distributions, and must follow the generator formula.
    for background in entries:
        draws = int(background['generation']['draws_per_mask'])
        for seed in [int(value) for value in config['seeds']]:
            for group in background['masks']:
                expected = stable_seed(background['generation']['seed'], seed, group['id'], 'residues')
                for spec in specs:
                    if spec['seed'] != seed:
                        continue
                    group_rows = [row for row in raw[spec['run_id'], background['id']]
                                  if row['group'] == group['id']]
                    _require(len(group_rows) == draws,
                             f'Draw budget differs for uniform seed check: {spec["run_id"]}, {background["id"]}')
                    for row in group_rows:
                        _require(int(row['uniform_seed']) == expected,
                                 f'Uniform seed differs: {spec["run_id"]}, {background["id"]}, '
                                 f'{group["id"]}, draw {row["draw"]}')
    # Hamming strata are descriptive; draws are not biological replicates.
    strata = []
    for key, rows in raw.items():
        run_id, background_id = key
        background = next(entry for entry in entries if entry['id'] == background_id)
        distance_by_key = distances[key]
        for stratum in _STRATA:
            subset = [row for row in rows if _hamming_bin(int(row['Hamming'])) == stratum]
            strata.append({'background': background_id, 'component': background['component'],
                           'run': run_id, 'stratum': stratum,
                           **_stratum_metrics(subset, background['masks'], distance_by_key)})
    contrast_meta = []
    for seed in [int(value) for value in config['seeds']]:
        for condition in config['conditions']:
            contrast_meta.extend([
                ('joint_minus_site_marginal', 'within_condition', condition,
                 f'QVAE_{condition}_seed{seed}_joint', f'QVAE_{condition}_seed{seed}_site_marginal'),
                ('joint_minus_deterministic', 'within_condition', condition,
                 f'QVAE_{condition}_seed{seed}_joint', f'deterministic_{condition}_seed{seed}_factorized'),
                ('site_marginal_minus_deterministic', 'within_condition', condition,
                 f'QVAE_{condition}_seed{seed}_site_marginal',
                 f'deterministic_{condition}_seed{seed}_factorized'),
            ])
        for arm, distribution in (('QVAE', 'joint'), ('QVAE', 'site_marginal'),
                                  ('deterministic', 'factorized')):
            contrast_meta.append((
                f'{arm}_{distribution}_external_minus_self', 'external_minus_self',
                'external_minus_self', f'{arm}_external_seed{seed}_{distribution}',
                f'{arm}_self_seed{seed}_{distribution}'))
    seen_contrasts = set()
    for label, kind, condition, left, right in contrast_meta:
        key = label, left, right
        _require(key not in seen_contrasts, f'Duplicate contrast definition: {key}')
        seen_contrasts.add(key)
        seed = int(next(part[4:] for part in left.split('_') if part.startswith('seed')))
        for background in entries:
            _require((left, background['id']) in raw and (right, background['id']) in raw,
                     f'Missing contrast run: {left}, {right}, {background["id"]}')
            _add_contrast(comparisons, hamming_rows, label, kind, condition, left, right,
                          background, seed, raw, distances)
    component_rows, bootstrap = _component_effects(
        comparisons, entries, config['seeds'], contrast_meta, config)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / 'run_summary.csv', run_rows)
    _write_csv(output / 'group_summary.csv', run_meta)
    _write_csv(output / 'site_frequencies.csv', site_rows)
    _write_csv(output / 'pair_dependence.csv', pair_rows)
    _write_csv(output / 'hamming_strata.csv', strata)
    _write_csv(output / 'hamming_paired.csv', hamming_rows)
    _write_csv(output / 'paired_comparisons.csv', comparisons)
    _write_csv(output / 'component_effects.csv', component_rows)
    _write_csv(output / 'candidate_distances.csv', candidate_rows)
    _write_generation_outputs(component_rows, run_rows, hamming_rows, output, config)
    input_paths = [config_path, config['backgrounds'], config['sources'], config['target'],
                   config['study_config'], Path(config['neighbors']) / 'nmnat.fasta',
                   Path(config['neighbors']) / 'nmnat.tsv',
                   Path(config['neighbors']) / 'nmnat_both_nearest.csv',
                   Path(config['neighbors']) / 'identity.json']
    input_paths.extend(Path(config['test_data']) / name for name in
                       ('test.csv', 'test.fasta', 'manifest.json', 'historical_use.json'))
    input_hashes = {str(path): sha256_file(path) for path in input_paths}
    write_json(output / 'summary.json', {
        'status': 'COMPLETED', 'backgrounds': len(entries) - 1, 'kvp40': config['kvp40_id'],
        'run_ids': [spec['run_id'] for spec in specs], 'draws_per_mask': int(config['draws_per_mask']),
        'masks_per_background': 6, 'comparisons': len(comparisons),
        'hamming_comparisons': len(hamming_rows), 'component_effects': len(component_rows),
        'candidate_distance_rows': len(candidate_rows),
        'units': 'background component; draws are not biological replicates',
        'test_manifest': {
            'status': test_manifest.get('status'),
            'outputs': {name: test_manifest['outputs'][name]['sha256']
                        for name in ('test.csv', 'test.fasta')}},
        'bootstrap': bootstrap,
        'hamming_pairing': 'per background/seed/mask/stratum, independently sample n=min(n_left,n_right) '
                           'draws within each side using stable_seed(left_run,right_run,background,mask,'
                           'stratum,seed,"left"/"right"); equal attempt budget; the original overall '
                           'comparison keeps all draws under common uniform seeds',
        'distance': 'background-alignment-conditioned nearest-training distance: stable_alignment '
                    '(BLOSUM62, open gap -10, extend gap -0.5, max_optimal from config, both-direction '
                    'stable coverage >= 0.80); candidate edits are substituted on the fixed background mapping '
                    'and scored against all qualified training references; minimum over references; '
                    'uncovered positions and ambiguity are reported separately',
        'distance_limitations': 'The mapping is fixed from the background sequence; candidate edits do not '
                                'change the alignment. This is not a per-candidate realignment distance.',
        'edit_mismatch_fraction': 'fraction of the candidate domain edits at which the candidate differs '
                                  'from the nearest aligned training reference at those mapped positions; '
                                  'it is not an edit-site nearest-neighbour search',
        'background_nearest_identity_role': 'background property only; not used as a candidate distance',
        'checkpoints': {key: {'path': value['path'], 'sha256': value['sha256']}
                        for key, value in checkpoints.items()},
        'input_sha256': input_hashes,
        'training_summary_sha256': training_hashes,
        'generation_sha256': generation_hashes,
        'implementation_sha256': {name: sha256_file(Path(__file__).with_name(name))
                                  for name in ('generation_comparison.py', 'preprocessing.py')}})
    return {'status': 'COMPLETED', 'runs': len(run_rows), 'comparisons': len(comparisons),
            'component_effects': len(component_rows)}
