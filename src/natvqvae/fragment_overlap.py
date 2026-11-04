"""Check auxiliary fragments and derive sequence groups without fitting models."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import shutil
import subprocess
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
from Bio import SeqIO

from .data import load_yaml, sha256_file, write_json


def _fasta(path):
    """Strict reader for current population and auxiliary model inputs."""
    records = {}
    for record in SeqIO.parse(path, 'fasta'):
        if record.id in records:
            raise ValueError(f'Duplicate FASTA ID in {path}: {record.id}')
        sequence = str(record.seq).upper()
        digest = hashlib.sha256(sequence.encode()).hexdigest()
        named_digest = record.id.removeprefix('aux:')
        if not sequence or (re.fullmatch('[0-9a-f]{64}', named_digest)
                            and named_digest != digest):
            raise ValueError(f'Sequence SHA mismatch or empty sequence: {record.id}')
        records[record.id] = sequence
    return records


def _source_records(path, requested_ids, counts):
    """Read original libraries, including empty MMseqs cluster-title records."""
    counts.update(records=0, nonempty_records=0, empty_records_skipped=0,
                  duplicate_nonempty_ids=0)
    seen_nonempty = set()
    for record_index, record in enumerate(SeqIO.parse(path, 'fasta'), 1):
        counts['records'] += 1
        sequence = str(record.seq).upper()
        if not sequence:
            counts['empty_records_skipped'] += 1
            continue
        counts['nonempty_records'] += 1
        counts['duplicate_nonempty_ids'] += int(record.id in seen_nonempty)
        seen_nonempty.add(record.id)
        digest = hashlib.sha256(sequence.encode()).hexdigest()
        named_digest = record.id.removeprefix('aux:')
        if named_digest in requested_ids and named_digest != digest:
            raise ValueError(f'Sequence SHA mismatch for requested source ID: {record.id}')
        yield record_index, record.id, sequence


def _rows(path):
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows, fields):
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_overlap_inputs(config, source_read_counts=None):
    """Recover targets by sequence content; never infer an accession from a SHA."""
    auxiliary = _fasta(config['auxiliary']['domains'])
    selected_splits = set(config['auxiliary'].get('selected_splits', ['train']))
    if not selected_splits or not selected_splits <= {'train', 'val'}:
        raise ValueError('Auxiliary selected_splits must contain train and/or val')
    selected, seen = {}, set()
    for row in _rows(config['auxiliary']['split']):
        uid = row['aux_uid']
        if uid in seen:
            raise ValueError(f'Duplicate auxiliary split ID: {uid}')
        seen.add(uid)
        sequence = auxiliary[uid]
        if (hashlib.sha256(sequence.encode()).hexdigest() != row['sequence_sha256']
                or len(sequence) != int(row['length'])):
            raise ValueError(f'Auxiliary split identity mismatch: {uid}')
        if row['split'] in selected_splits:
            selected[uid] = {**row, 'sequence': sequence, 'kind': row['aux_kind']}
    if not selected:
        raise ValueError('No selected auxiliary sequences')

    targets, group_roles = {}, {}
    for role, path in config['natv']['splits'].items():
        for row in _rows(path):
            uid, component = row['sequence_id'], row['component_v2']
            if uid in targets:
                raise ValueError(f'Duplicate or cross-split target ID: {uid}')
            if component in group_roles and group_roles[component] != role:
                raise ValueError(f'Target component occurs in multiple splits: {component}')
            group_roles[component] = role
            targets[uid] = {**row, 'role': role, 'component': component}
    sources = [config['natv']['population'], *config['natv'].get('source_fastas', [])]
    provenance = []
    if source_read_counts is None:
        source_read_counts = {}
    for source_index, path in enumerate(sources):
        counts = source_read_counts.setdefault(str(path), {})
        if source_index == 0:
            population = _fasta(path)
            counts.update(records=len(population), nonempty_records=len(population),
                          empty_records_skipped=0, duplicate_nonempty_ids=0)
            entries = ((i, uid, seq) for i, (uid, seq) in enumerate(population.items(), 1))
        else:
            entries = _source_records(path, set(targets), counts)
        for record_index, source_id, sequence in entries:
            uid = hashlib.sha256(sequence.encode()).hexdigest()
            if uid not in targets:
                continue
            row = targets[uid]
            if row.get('length') and len(sequence) != int(row['length']):
                raise ValueError(f'Target length mismatch: {uid}')
            row['sequence'] = sequence
            provenance.append({'sequence_id': uid, 'role': row['role'],
                               'source_path': str(path), 'source_id': source_id,
                               'source_record_index': record_index,
                               'sequence_sha256': uid, 'length': len(sequence)})
    missing = [{'sequence_id': uid, 'role': row['role'], 'component': row['component']}
               for uid, row in sorted(targets.items()) if 'sequence' not in row]
    return selected, targets, provenance, missing


def exact_fragment_hits(auxiliary, targets):
    """Enumerate every full-fragment exact occurrence, with 1-based endpoints."""
    hits = []
    for uid, fragment in sorted(auxiliary.items()):
        query = fragment['sequence']
        for target_id, target in sorted(targets.items()):
            sequence = target.get('sequence', '')
            start = sequence.find(query)
            while start >= 0:
                end = start + len(query)
                domain_start, domain_end = target.get('nmnat_cdd_from'), target.get('nmnat_cdd_to')
                domain_length = int(domain_end) - int(domain_start) + 1 if domain_start else None
                overlap = (max(0, min(end, int(domain_end)) - max(start, int(domain_start) - 1))
                           if domain_start else None)
                hits.append({'aux_uid': uid, 'aux_kind': fragment['kind'],
                             'aux_split': fragment.get('split', 'train'),
                             'target_id': target_id, 'role': target['role'],
                             'component': target['component'], 'aux_length': len(query),
                             'target_length': len(sequence), 'target_start': start + 1,
                             'target_end': end, 'nmnat_length': domain_length,
                             'nmnat_covered_fraction': overlap / domain_length if domain_length else None})
                start = sequence.find(query, start + 1)
    return hits


def _hit_counts(hits, targets):
    result = {}
    for role in sorted({row['role'] for row in targets.values()}):
        rows = [row for row in hits if row['role'] == role]
        result[role] = {'hit_rows': len(rows), 'auxiliary_sequences': len({r['aux_uid'] for r in rows}),
                        'target_sequences': len({r['target_id'] for r in rows}),
                        'target_components': len({r['component'] for r in rows}),
                        'auxiliary_sequences_by_original_split': {
                            split: len({r['aux_uid'] for r in rows if r['aux_split'] == split})
                            for split in sorted({r['aux_split'] for r in rows})}}
    return result


def _diagnostic(path, excluded):
    data = json.loads(Path(path).read_text())['component_differences']['prior_joint_nll_per_residue']
    pairs = list(data.values())
    components = sorted(pairs[0]['NMNAT'])
    if any(set(pair['NMNAT']) != set(components) for pair in pairs):
        raise ValueError('Paired runs have different NMNAT components')
    means = {c: float(np.mean([pair['NMNAT'][c] for pair in pairs])) for c in components}
    unknown = excluded - set(means)
    if unknown:
        raise ValueError(f'Overlap components absent from evaluation: {sorted(unknown)}')
    def summarize(keep):
        values = np.array([means[c] for c in keep])
        if not len(values):
            return {'components': 0, 'mean': None, 'ci95': None, 'negative_components': 0}
        draws = np.random.default_rng(8802).integers(len(values), size=(2000, len(values)))
        return {'components': len(values), 'mean': float(values.mean()),
                'ci95': np.quantile(values[draws].mean(1), [0.025, 0.975]).tolist(),
                'negative_components': int((values < 0).sum())}
    return {'status': 'post_hoc_fixed_model_diagnostic_not_retraining',
            'score': 'NMNAT prior joint NLL per target residue; auxiliary minus random',
            'overlap_scope': 'Exact NatV validation matches to auxiliary sequences used in the original training split only.',
            'excluded_components': sorted(excluded), 'bootstrap_replicates': 2000, 'bootstrap_seed': 8802,
            'all_components': summarize(components),
            'without_exact_overlap_components': summarize([c for c in components if c not in excluded]),
            'limit': 'Does not remove information from fitted weights or undo validation-based selection.'}


ALIGNMENT_FIELDS = 'query,target,fident,qcov,tcov,qstart,qend,tstart,tend,qlen,tlen,alnlen,evalue,bits'


def _write_fasta(path, entries):
    with path.open('w') as handle:
        for uid, row in sorted(entries.items()):
            if 'sequence' in row:
                handle.write(f'>{uid}\n{row["sequence"]}\n')


def _run_search(parameters, output, query, target, stem, target_count):
    result = output / f'{stem}.tsv'
    command = [str(parameters['executable']), 'easy-search', str(query.resolve()), str(target.resolve()),
               str(result.resolve()), str((output / f'{stem}_tmp').resolve()),
               '--min-seq-id', str(parameters['min_identity']), '--cov-mode', '2',
               '-c', str(parameters['query_coverage']), '-e', str(parameters['evalue']),
               '-s', str(parameters['sensitivity']), '--max-seqs', str(target_count),
               '--alignment-mode', '3', '--seq-id-mode', '0',
               '--threads', str(parameters['threads']), '--format-output', ALIGNMENT_FIELDS]
    version = subprocess.check_output([parameters['executable'], 'version'], text=True).strip()
    with (output / f'{stem}.log').open('w') as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return result, {'command': command, 'version': version, 'columns': ALIGNMENT_FIELDS.split(','),
                    'limit': 'Heuristic local alignment search, not exhaustive detection of all homologous fragments.'}


def _similarity_search(parameters, output, auxiliary, targets):
    query, target = output / 'selected_auxiliary.fasta', output / 'available_natv.fasta'
    _write_fasta(query, auxiliary)
    _write_fasta(target, targets)
    result, metadata = _run_search(parameters, output, query, target, 'similarity_hits', len(targets))
    hits = []
    with result.open() as handle:
        for row in csv.DictReader(handle, fieldnames=ALIGNMENT_FIELDS.split(','), delimiter='\t'):
            uid, target_id = row.pop('query'), row.pop('target')
            hits.append({'aux_uid': uid, 'aux_split': auxiliary[uid]['split'], 'target_id': target_id,
                         'role': targets[target_id]['role'], 'component': targets[target_id]['component'], **row})
    _write_csv(output / 'similarity_hits.csv', hits,
               ['aux_uid', 'aux_split', 'target_id', 'role', 'component', *ALIGNMENT_FIELDS.split(',')[2:]])
    return {**metadata, 'counts': _hit_counts(hits, targets),
            'coverage': 'Fraction of the auxiliary query covered; no target coverage requirement.'}, hits


class _Groups:
    """Retain only a spanning forest, with evidence for each connecting edge."""
    def __init__(self, ids):
        self.parents = {uid: uid for uid in ids}
        self.forest = []

    def find(self, uid):
        while self.parents[uid] != uid:
            self.parents[uid] = self.parents[self.parents[uid]]
            uid = self.parents[uid]
        return uid

    def add(self, a, b, evidence, evidence_file, evidence_row):
        first, second = self.find(a), self.find(b)
        if first == second:
            return
        self.parents[max(first, second)] = min(first, second)
        self.forest.append({'edge_id': len(self.forest) + 1, 'aux_a': a, 'aux_b': b,
                            'evidence': evidence, 'evidence_file': evidence_file,
                            'evidence_row': evidence_row})


def _auxiliary_groups(auxiliary, parameters, output):
    groups, original = _Groups(auxiliary), {}
    for uid, row in sorted(auxiliary.items()):
        label = row.get('domain_cluster50')
        if label:
            if label in original:
                groups.add(original[label], uid, 'original_domain_cluster50', 'input_aux_split', label)
            else:
                original[label] = uid
    ids = sorted(auxiliary)
    exact, cross_split_exact = [], 0
    for index, a in enumerate(ids):
        for b in ids[index + 1:]:
            short, long = sorted((a, b), key=lambda uid: (len(auxiliary[uid]['sequence']), uid))
            position = auxiliary[long]['sequence'].find(auxiliary[short]['sequence'])
            if position < 0:
                continue
            cross_split_exact += auxiliary[a]['split'] != auxiliary[b]['split']
            exact.append({'short_aux_uid': short, 'long_aux_uid': long,
                          'short_original_split': auxiliary[short]['split'],
                          'long_original_split': auxiliary[long]['split'],
                          'short_length': len(auxiliary[short]['sequence']),
                          'long_length': len(auxiliary[long]['sequence']),
                          'long_start': position + 1,
                          'long_end': position + len(auxiliary[short]['sequence'])})
            groups.add(a, b, 'exact_containment', 'auxiliary_exact_pairs.csv', len(exact))
    fields = ['short_aux_uid', 'long_aux_uid', 'short_original_split', 'long_original_split',
              'short_length', 'long_length', 'long_start', 'long_end']
    _write_csv(output / 'auxiliary_exact_pairs.csv', exact, fields)
    query = output / 'selected_auxiliary.fasta'
    _write_fasta(query, auxiliary)
    result, metadata = _run_search(parameters, output, query, query, 'auxiliary_similarity', len(auxiliary))
    pair_ids, cross_split_pairs, accepted_rows, filtered = set(), 0, 0, 0
    indices = {uid: index for index, uid in enumerate(ids)}
    with result.open() as handle:
        for line, row in enumerate(csv.DictReader(handle, fieldnames=ALIGNMENT_FIELDS.split(','), delimiter='\t'), 1):
            a, b = row['query'], row['target']
            if a == b:
                continue
            short_coverage = float(row['qcov'] if int(row['qlen']) <= int(row['tlen']) else row['tcov'])
            if short_coverage < float(parameters['query_coverage']):
                filtered += 1
                continue
            accepted_rows += 1
            first, second = sorted((indices[a], indices[b]))
            pair_id = first * len(ids) + second
            if pair_id not in pair_ids:
                cross_split_pairs += auxiliary[a]['split'] != auxiliary[b]['split']
                pair_ids.add(pair_id)
            groups.add(a, b, 'local_similarity', 'auxiliary_similarity.tsv.gz', line)
    compressed = result.with_suffix('.tsv.gz')
    with result.open('rb') as source, gzip.GzipFile(filename=str(compressed), mode='wb', mtime=0) as target:
        shutil.copyfileobj(source, target)
    result.unlink()
    _write_csv(output / 'auxiliary_group_forest.csv', groups.forest,
               ['edge_id', 'aux_a', 'aux_b', 'evidence', 'evidence_file', 'evidence_row'])
    return groups, {**metadata, 'alignment_file': compressed.name,
                    'exact_containment_pairs': len(exact), 'exact_cross_original_split_pairs': cross_split_exact,
                    'similarity_pairs': len(pair_ids), 'similarity_cross_original_split_pairs': cross_split_pairs,
                    'accepted_directed_alignment_rows': accepted_rows, 'short_coverage_filtered_rows': filtered,
                    'group_definition': 'Union of original domain_cluster50 membership, exhaustive exact containment, and detected local alignments covering at least the configured fraction of the shorter fragment.',
                    'exact_coordinates': 'First occurrence per unordered pair, 1-based inclusive; all contained pairs enumerated.'}


def _group_membership(auxiliary, groups, target_hits):
    direct, group_roles, anchors = defaultdict(set), defaultdict(set), {}
    for row in target_hits:
        uid, role = row['aux_uid'], row['role']
        direct[uid].add(role)
        group_roles[groups.find(uid)].add(role)
        if role in {'validation', 'test'}:
            # Deterministic source choice; all direct matches remain in the hit tables.
            key = (role, row['target_id'], row['evidence_file'], row['evidence_row'])
            if uid not in anchors or key < anchors[uid]:
                anchors[uid] = key
    adjacency = defaultdict(list)
    for edge in groups.forest:
        a, b = edge['aux_a'], edge['aux_b']
        adjacency[a].append((b, edge['edge_id']))
        adjacency[b].append((a, edge['edge_id']))
    queue, paths = deque(sorted(anchors)), {uid: (uid, []) for uid in anchors}
    while queue:
        uid = queue.popleft()
        for neighbor, edge_id in adjacency[uid]:
            if neighbor not in paths:
                anchor, edge_ids = paths[uid]
                paths[neighbor] = (anchor, edge_ids + [edge_id])
                queue.append(neighbor)
    membership = []
    for uid, row in sorted(auxiliary.items()):
        anchor, edge_ids = paths.get(uid, ('', []))
        roles = direct[uid]
        reason = ('direct_validation_or_test_match' if uid in anchors else
                  'connected_to_validation_or_test_match' if anchor else
                  'training_related_only' if 'training' in group_roles[groups.find(uid)] else
                  'no_natv_match_detected_in_component')
        membership.append({'aux_uid': uid, 'sequence_sha256': hashlib.sha256(row['sequence'].encode()).hexdigest(),
                           'original_split': row['split'], 'original_domain_cluster50': row.get('domain_cluster50', ''),
                           'original_weight': row.get('weight', ''),
                           'aux_kind': row['kind'], 'length': len(row['sequence']), 'group': groups.find(uid),
                           'direct_natv_roles': ';'.join(sorted(roles)),
                           'group_natv_roles': ';'.join(sorted(group_roles[groups.find(uid)])),
                           'status': 'excluded' if anchor else 'retained', 'reason': reason,
                           'excluded_via_aux_uid': anchor, 'exclusion_path_edge_ids': json.dumps(edge_ids),
                           'anchor_target_role': anchors[anchor][0] if anchor else '',
                           'anchor_target_id': anchors[anchor][1] if anchor else '',
                           'anchor_evidence_file': anchors[anchor][2] if anchor else '',
                           'anchor_evidence_row': anchors[anchor][3] if anchor else ''})
    return membership


def _retained_pool(config, output, auxiliary, membership, missing):
    masks = json.loads(Path(config['auxiliary']['masks']).read_text())
    annotations, source_rows = defaultdict(set), defaultdict(list)
    records_path = config['auxiliary'].get('records')
    if records_path:
        opener = gzip.open if str(records_path).endswith('.gz') else open
        with opener(records_path, 'rt') as handle:
            for row in csv.DictReader(handle):
                if row.get('aux_uid') in auxiliary:
                    source_rows[row['aux_uid']].append(row)
                    annotations[row['aux_uid']].add(row['annotation_layer'])
    for row in membership:
        uid, sequence = row['aux_uid'], auxiliary[row['aux_uid']]['sequence']
        if uid not in masks:
            raise ValueError(f'Auxiliary mask missing: {uid}')
        mapped = {int(q) for q, _ in masks[uid]['pairs']}
        wt_positions = {int(w) for _, w in masks[uid]['pairs']}
        if any(q < 0 or q >= len(sequence) for q in mapped):
            raise ValueError(f'Auxiliary mapping outside sequence: {uid}')
        row.update(mapped_positions=len(mapped), mapped_fraction=len(mapped) / len(sequence),
                   wt_mapped_positions=len(wt_positions),
                   wt_mapped_fraction=len(wt_positions) / int(config['auxiliary_relations']['wt_domain_length']),
                   source_records=len(source_rows[uid]),
                   annotation_layers=';'.join(sorted(annotations[uid])))
    _write_csv(output / 'auxiliary_membership.csv', membership, list(membership[0]))
    retained = [row for row in membership if row['status'] == 'retained']
    ids = {row['aux_uid'] for row in retained}
    original_groups = defaultdict(list)
    for row in retained:
        original_groups[row['original_domain_cluster50']].append(row)
    group_rows = []
    for group, rows in sorted(original_groups.items()):
        lengths = [row['length'] for row in rows]
        group_rows.append({'original_domain_cluster50': group, 'sequences': len(rows),
                           'original_splits': ';'.join(sorted({row['original_split'] for row in rows})),
                           'relation_group': rows[0]['group'], 'length_min': min(lengths),
                           'length_median': float(np.median(lengths)), 'length_max': max(lengths),
                           'original_weight_sum': sum(float(row['original_weight'] or 0) for row in rows)})
    _write_csv(output / 'retained_original_groups.csv', group_rows,
               ['original_domain_cluster50', 'sequences', 'original_splits', 'relation_group',
                'length_min', 'length_median', 'length_max', 'original_weight_sum'])
    def distribution(key):
        values = np.array([row[key] for row in retained])
        return dict(zip(['min', 'median', 'max'], np.quantile(values, [0, .5, 1]).tolist())) if len(values) else {}
    summary = {'sequences': len(retained), 'groups': len({row['group'] for row in retained}),
               'group_sizes': sorted(Counter(row['group'] for row in retained).values(), reverse=True),
               'original_splits': dict(Counter(row['original_split'] for row in retained)),
               'kinds': dict(Counter(row['aux_kind'] for row in retained)),
               'length': distribution('length'), 'mapped_fraction': distribution('mapped_fraction'),
               'wt_mapped_positions': distribution('wt_mapped_positions'),
               'wt_mapped_fraction': distribution('wt_mapped_fraction'),
               'wt_domain_length': int(config['auxiliary_relations']['wt_domain_length']),
               'original_domain_cluster50_groups': len(original_groups),
               'original_weight_sum': sum(float(row['original_weight'] or 0) for row in retained),
               'original_weight_sum_by_split': {
                   split: sum(float(row['original_weight'] or 0) for row in retained if row['original_split'] == split)
                   for split in sorted({row['original_split'] for row in retained})},
               'weight_definition': 'Historical weights were normalized separately within original train/val. They are reported, not combined into new training weights. Relation groups do not define sampling weights.',
               'annotation_layers_sequence_counts': dict(Counter(label for uid in ids for label in annotations[uid])),
               'annotation_layers_source_counts': dict(Counter(r['annotation_layer'] for uid in ids for r in source_rows[uid])),
               'annotation_count_definition': 'Sequence counts may overlap when sources have different annotation layers; source counts count original records, not independent proteins.',
               'training_partition': 'Not assigned; original_split records past use only. No new auxiliary validation or checkpoint selection is defined.'}
    if missing:
        summary['export_status'] = 'not_exported_missing_natv_targets'
    else:
        _write_fasta(output / 'retained_aux_domains.fasta', {uid: auxiliary[uid] for uid in ids})
        write_json(output / 'retained_aux_masks.json', {uid: masks[uid] for uid in sorted(ids)})
        _write_csv(output / 'retained_auxiliary.csv', retained, list(membership[0]))
        summary['export_status'] = 'complete_relation_screened_pool_not_a_training_split'
    return summary


def check_fragment_overlap(config_path, output=None):
    config = load_yaml(config_path)
    relations = config.get('auxiliary_relations', {}).get('enabled', False)
    if relations and (set(config['auxiliary'].get('selected_splits', ['train'])) != {'train', 'val'}
                      or not config.get('mmseqs', {}).get('enabled')):
        raise ValueError('Auxiliary relations require both original train/val and enabled MMseqs search')
    output = Path(output or config['output'])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output already contains files: {output}')
    paths = [config_path, *(config['auxiliary'][key] for key in ('domains', 'split', 'masks', 'records')
                           if key in config['auxiliary']), config['natv']['population'],
             *config['natv']['splits'].values(), *config['natv'].get('source_fastas', [])]
    if config.get('paired_summary'):
        paths.append(config['paired_summary'])
    if config.get('aux_training_record'):
        paths.append(config['aux_training_record'])
        record = json.loads(Path(config['aux_training_record']).read_text())
        for key in ('domains', 'split', 'masks'):
            if sha256_file(config['auxiliary'][key]) != record['input_sha256'][key]:
                raise ValueError(f'Actual auxiliary training input SHA mismatch: {key}')
    inputs = {str(path): sha256_file(path) for path in paths}
    source_read_counts = {}
    auxiliary, targets, provenance, missing = load_overlap_inputs(config, source_read_counts)
    output.mkdir(parents=True, exist_ok=True)
    hits = exact_fragment_hits(auxiliary, targets)
    _write_csv(output / 'exact_hits.csv', hits,
               ['aux_uid', 'aux_kind', 'aux_split', 'target_id', 'role', 'component', 'aux_length', 'target_length',
                'target_start', 'target_end', 'nmnat_length', 'nmnat_covered_fraction'])
    _write_csv(output / 'target_sources.csv', provenance,
               ['sequence_id', 'role', 'source_path', 'source_id', 'source_record_index',
                'sequence_sha256', 'length'])
    _write_csv(output / 'missing_targets.csv', missing, ['sequence_id', 'role', 'component'])
    with (output / 'recovered_test.fasta').open('w') as handle:
        for uid, row in sorted(targets.items()):
            if row['role'] == 'test' and 'sequence' in row:
                handle.write(f'>{uid}\n{row["sequence"]}\n')
    summary = {'status': 'complete_available_targets' if missing else 'complete_all_targets',
               'inputs_sha256': inputs, 'implementation_sha256': sha256_file(__file__),
               'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
               'git_status': subprocess.check_output(['git', 'status', '--porcelain'], text=True).splitlines(),
               'selected_auxiliary_sequences': len(auxiliary),
               'auxiliary_training_sequences': sum(row['split'] == 'train' for row in auxiliary.values()),
               'auxiliary_original_split_counts': dict(Counter(row['split'] for row in auxiliary.values())),
               'requested_targets': dict(Counter(row['role'] for row in targets.values())),
               'available_targets': dict(Counter(row['role'] for row in targets.values() if 'sequence' in row)),
               'source_read_counts': source_read_counts,
               'missing_targets': missing, 'exact_hits': _hit_counts(hits, targets),
               'model_fitting_or_scoring': False}
    if config.get('paired_summary'):
        excluded = {r['component'] for r in hits if r['role'] == 'validation' and r['aux_split'] == 'train'}
        diagnostic = _diagnostic(config['paired_summary'], excluded)
        write_json(output / 'validation_diagnostic.json', diagnostic)
    if config.get('mmseqs', {}).get('enabled'):
        summary['similarity_search'], similar_hits = _similarity_search(config['mmseqs'], output, auxiliary, targets)
    if relations:
        groups, relation_summary = _auxiliary_groups(auxiliary, config['mmseqs'], output)
        target_hits = [{**row, 'evidence_file': filename, 'evidence_row': index}
                       for filename, rows in [('exact_hits.csv', hits), ('similarity_hits.csv', similar_hits)]
                       for index, row in enumerate(rows, 1)]
        membership = _group_membership(auxiliary, groups, target_hits)
        relation_summary.update(
            groups=len({row['group'] for row in membership}),
            group_sizes=sorted(Counter(row['group'] for row in membership).values(), reverse=True),
            reason_counts=dict(Counter(row['reason'] for row in membership)),
            excluded_sequences=sum(row['status'] == 'excluded' for row in membership),
            exclusion_policy='Exclude the entire auxiliary group with a detected NatV validation/test connection. NatV training connections are annotated, not exclusion grounds. NatV nodes do not connect auxiliary groups.',
            connection_path='Edge IDs give one path in the recorded spanning forest, from a directly matched auxiliary anchor to the listed fragment; not necessarily the shortest path in all detected relations. CSV evidence rows are 1-based data rows excluding the header. For original_domain_cluster50 edges, evidence_row is the original group label shared by both endpoints.')
        relation_summary['retained_pool'] = _retained_pool(config, output, auxiliary, membership, missing)
        summary['auxiliary_relations'] = relation_summary
        summary['result_sha256'] = {path.name: sha256_file(path) for path in output.iterdir()
                                   if path.is_file() and path.name != 'summary.json'}
    write_json(output / 'summary.json', summary)
    return summary
