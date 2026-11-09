"""Recover the historical component definition from its original alignment graph."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .data import load_yaml, sha256_file, write_json


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def nearest_cross_split_hits(hits, roles, coverage):
    """Choose maximum identity among qualifying returned local alignments."""
    best = {}
    for row in hits:
        query, target = row['query'], row['target']
        if roles[query] == roles[target]:
            continue
        qcov, tcov = float(row['qcov']), float(row['tcov'])
        if coverage == 'both':
            accepted = min(qcov, tcov) >= 0.8
        elif coverage == 'shorter':
            qlen, tlen = int(row['qlen']), int(row['tlen'])
            accepted = (min(qcov, tcov) if qlen == tlen else qcov if qlen < tlen else tcov) >= 0.8
        else:
            raise ValueError('Unknown alignment coverage rule')
        if not accepted:
            continue
        key = query, roles[target]
        rank = (-float(row['fident']), -float(row['bits']), target)
        if key not in best or rank < best[key][0]:
            best[key] = rank, row
    return {key: row for key, (_, row) in best.items()}


def analyze_sequence_neighbors(config_path, reuse_alignments=False):
    import subprocess
    from collections import Counter
    from .fragment_overlap import _fasta, _rows, _write_csv, _write_fasta, _run_search

    config = load_yaml(config_path)
    output = Path(config['output'])
    if output.exists() and any(output.iterdir()) and not reuse_alignments:
        raise FileExistsError(f'Sequence comparison output already contains files: {output}')
    sequences = {}
    for path in config['fastas']:
        for uid, sequence in _fasta(path).items():
            if uid in sequences and sequences[uid] != sequence:
                raise ValueError(f'Conflicting sequence: {uid}')
            sequences[uid] = sequence
    records, roles = {}, {}
    for role, path in config['splits'].items():
        for row in _rows(path):
            uid = row['sequence_id']
            if uid in roles or uid not in sequences:
                raise ValueError(f'Duplicate or unavailable split sequence: {uid}')
            roles[uid], records[uid] = role, row
    panels = {'full_length': {uid: {'sequence': sequences[uid]} for uid in roles}}
    for domain in ('nmnat', 'nudix'):
        panel = {}
        for uid, row in records.items():
            start, end = int(row[f'{domain}_cdd_from']), int(row[f'{domain}_cdd_to'])
            if not 1 <= start <= end <= len(sequences[uid]):
                raise ValueError(f'Invalid {domain} coordinates: {uid}')
            panel[uid] = {'sequence': sequences[uid][start - 1:end]}
        panels[domain] = panel
    inputs = [config_path, *config['fastas'], *config['splits'].values()]
    identity = {'input_sha256': {str(p): sha256_file(p) for p in inputs},
                'code_sha256': {name: sha256_file(Path(__file__).with_name(name))
                               for name in ('split_relations.py', 'fragment_overlap.py')},
                'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    previous = None
    if reuse_alignments:
        original = json.loads((output / 'identity.json').read_text())
        if original['input_sha256'] != identity['input_sha256']:
            raise ValueError('Cached alignment inputs differ from current inputs')
        previous = json.loads((output / 'summary.json').read_text())
        identity['previous_summary_sha256'] = sha256_file(output / 'summary.json')
    else:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / 'identity.json', identity)
    summaries = {}
    for name, panel in panels.items():
        fasta = output / f'{name}.fasta'
        if reuse_alignments:
            expected_fasta = ''.join(f'>{uid}\n{panel[uid]["sequence"]}\n' for uid in sorted(panel))
            if fasta.read_text() != expected_fasta:
                raise ValueError(f'Cached search panel differs: {name}')
            result = output / f'{name}.tsv'
            metadata = previous['regions'][name]['search']
        else:
            _write_fasta(fasta, panel)
            result, metadata = _run_search(config['mmseqs'], output, fasta, fasta, name, len(panel))
        summaries[name] = {'search': metadata}
        for coverage in ('both', 'shorter'):
            with result.open() as handle:
                best = nearest_cross_split_hits(
                    csv.DictReader(handle, fieldnames=metadata['columns'], delimiter='\t'), roles, coverage)
            rows = []
            for uid in sorted(roles):
                for target_role in sorted(set(roles.values()) - {roles[uid]}):
                    hit = best.get((uid, target_role))
                    rows.append({'sequence_id': uid, 'component': records[uid]['component_v2'],
                                 'role': roles[uid], 'target_role': target_role,
                                 'target_id': hit['target'] if hit else '',
                                 **{field: hit[field] if hit else '' for field in
                                    ('fident', 'qcov', 'tcov', 'alnlen', 'evalue', 'bits')}})
            _write_csv(output / f'{name}_{coverage}_nearest.csv', rows, list(rows[0]))
            counts = {}
            for role in sorted(set(roles.values())):
                for target_role in sorted(set(roles.values()) - {role}):
                    subset = [r for r in rows if r['role'] == role and r['target_role'] == target_role]
                    found = [float(r['fident']) for r in subset if r['target_id']]
                    counts[f'{role}_to_{target_role}'] = {
                        'queries': len(subset), 'with_qualifying_hit': len(found),
                        'identity_ge_50pct': sum(x >= 0.5 for x in found),
                        'identity_ge_70pct': sum(x >= 0.7 for x in found),
                        'maximum_identity': max(found) if found else None}
            summaries[name][coverage] = counts
    summary = {'status': 'completed', 'sequence_counts': dict(Counter(roles.values())),
               'regions': summaries,
               'definition': 'Maximum fident across returned local alignments to each other split; ties use bitscore then target ID. Both: each sequence >=80% covered. Shorter: shorter sequence >=80% covered; equal lengths require both.',
               'limits': ['MMseqs2 is heuristic; no qualifying returned alignment is not proof of no homolog.',
                          'Identity is local alignment identity, not global identity or functional similarity.',
                          'This compares sequence relationships without fitting models or scoring predictions.']}
    identity['alignment_sha256'] = {f'{name}.tsv': sha256_file(output / f'{name}.tsv') for name in panels}
    identity['reused_alignments'] = reuse_alignments
    write_json(output / 'summary_identity.json', identity)
    write_json(output / 'summary.json', summary)
    return summary


def reconstruct_components(identifiers, edges):
    """Keep excluded bridge nodes until after constructing the original graph."""
    parents = {uid: uid for uid in identifiers}

    def find(uid):
        while parents[uid] != uid:
            parents[uid] = parents[parents[uid]]
            uid = parents[uid]
        return uid

    for left, right in edges:
        if left not in parents or right not in parents:
            raise ValueError('Alignment edge outside the eligible family')
        left, right = find(left), find(right)
        if left != right:
            parents[max(left, right)] = min(left, right)
    return {uid: find(uid) for uid in parents}


def analyze_split_relations(config_path):
    import pandas as pd

    config = load_yaml(config_path)
    output = Path(config['output'])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Split analysis output already contains files: {output}')
    paths = {key: Path(value) for key, value in config['historical'].items()}
    manifest = json.loads(paths['manifest'].read_text())
    for key, field in [('family', 'family_manifest_sha256'),
                       ('edges', 'source_exact_trace_sha256')]:
        if sha256_file(paths[key]) != manifest[field]:
            raise ValueError(f'Historical {key} hash differs from split manifest')
    family = pd.read_parquet(paths['family'])
    if family.sequence_id.duplicated().any():
        raise ValueError('Duplicate historical family ID')
    trace = pd.read_parquet(paths['edges'])
    if not ((trace.alnlen > 0) & (trace.qlen > 0) & (trace.tlen > 0) &
            (100 * trace.nident >= 70 * trace.alnlen) &
            (100 * (trace.qend - trace.qstart + 1) >= 80 * trace.qlen) &
            (100 * (trace.tend - trace.tstart + 1) >= 80 * trace.tlen)).all():
        raise ValueError('Historical alignment violates the declared 70/80 predicate')
    population = set(family.sequence_id)
    if not (set(trace['query']) | set(trace.target)) <= population:
        raise ValueError('Historical alignment contains unknown population IDs')
    eligible = set(family.loc[family.family_eligible_v2, 'sequence_id'])
    trace = trace[trace['query'].isin(eligible) & trace.target.isin(eligible)]
    components = reconstruct_components(
        eligible, trace[['query', 'target']].itertuples(index=False, name=None))
    roles, counts = {}, {}
    for role, path in config['splits'].items():
        with open(path, newline='') as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f'Empty current split: {role}')
        for row in rows:
            uid, group = row['sequence_id'], row['component_v2']
            if uid in roles or components.get(uid) != group:
                raise ValueError(f'Duplicate current ID or reconstructed component mismatch: {uid}')
            if manifest['component_assignments'].get(group) != role:
                raise ValueError(f'Historical allocation differs for {uid}')
            roles[uid] = role
        counts[role] = {'sequences': len(rows),
                        'components': len({row['component_v2'] for row in rows})}
    cross_edges = sum(roles[a] != roles[b]
                      for a, b in trace[['query', 'target']].itertuples(index=False, name=None)
                      if a in roles and b in roles)
    if cross_edges:
        raise ValueError('Qualified historical edges cross current splits')
    inputs = [Path(config_path), *paths.values(), *map(Path, config['splits'].values())]
    result = {
        'status': 'historical_components_reconstructed',
        'definition': 'nident/alnlen >= 0.70; both inclusive aligned spans/sequence length >= 0.80',
        'component_label': 'lexicographically smallest sequence SHA256 in eligible-family component',
        'construction': 'Induce the graph on domain-eligible family nodes, then apply source-role exclusions without splitting components.',
        'eligible_family_sequences': len(eligible),
        'eligible_family_components': len(set(components.values())),
        'qualified_directed_edges_including_self': len(trace),
        'current_splits': counts, 'cross_split_qualified_edges': cross_edges,
        'input_sha256': {str(path): sha256_file(path) for path in inputs},
        'implementation_sha256': sha256_file(Path(__file__)),
        'limits': ['Uses the archived threshold-qualified alignment trace; does not independently establish search completeness.',
                   'Does not measure nearest-neighbor similarities below the graph threshold or domain/fragment isolation.',
                   'Does not establish historically unused test data or remote-homology generalization.'],
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'historical_components.json', result)
    return result


def containment_events(query, reference, max_events=None):
    """Report exact substring containment in both directions; not an alignment."""
    events = []
    for query_id, query_sequence in sorted(query.items()):
        for reference_id, reference_sequence in sorted(reference.items()):
            if len(query_sequence) <= len(reference_sequence):
                longer_id, longer = reference_id, reference_sequence
                shorter_id, shorter = query_id, query_sequence
                direction = 'query_in_reference'
            else:
                longer_id, longer = query_id, query_sequence
                shorter_id, shorter = reference_id, reference_sequence
                direction = 'reference_in_query'
            offset = longer.find(shorter)
            while offset >= 0:
                events.append({'query_id': query_id, 'reference_id': reference_id,
                               'direction': direction, 'shorter_length': len(shorter),
                               'longer_length': len(longer), 'offset_0based': offset,
                               'shorter_fraction_of_longer': len(shorter) / len(longer)})
                if max_events is not None and len(events) >= max_events:
                    return events, True
                offset = longer.find(shorter, offset + 1)
    return events, False


def analyze_fragment_containment(config_path):
    """Check exact short-fragment containment between auxiliary and NatV splits."""
    from .fragment_overlap import _fasta, _rows

    config = load_yaml(config_path)
    output = Path(config['output'])
    if output.exists():
        raise FileExistsError(f'Fragment containment output already exists: {output}')
    population = _fasta(config['population'])
    test_sequences = _fasta(config['test_fasta'])
    aux_domains = _fasta(config['aux_domains'])
    aux_roles = {}
    for row in _rows(config['aux_split']):
        _require(row['aux_uid'] in aux_domains, f'Auxiliary domain missing: {row["aux_uid"]}')
        _require(row['split'] in {'train', 'val'}, f'Unknown auxiliary split: {row["split"]}')
        aux_roles[row['aux_uid']] = row['split']
    natv = {}
    for role, path in config['natv_splits'].items():
        for row in _rows(path):
            uid = row['sequence_id']
            sequence = test_sequences.get(uid) if role == 'test' else population.get(uid)
            _require(sequence is not None, f'NatV sequence missing: {role}, {uid}')
            start, end = int(row['nmnat_cdd_from']), int(row['nmnat_cdd_to'])
            _require(1 <= start <= end <= len(sequence), f'Invalid NatV NMNAT interval: {role}, {uid}')
            natv[role, uid] = {'full': sequence, 'nmnat': sequence[start - 1:end],
                               'component': row['component_v2']}
    aux_sets = {
        role: {uid: aux_domains[uid] for uid, assigned in aux_roles.items() if assigned == role}
        for role in ('train', 'val')}
    result = {'status': 'completed',
              'method': 'exact amino-acid substring containment in both directions; no alignment',
              'auxiliary_sequences': {role: len(values) for role, values in aux_sets.items()},
              'natv_sequences': {role: sum(1 for r, _ in natv if r == role)
                                 for role in config['natv_splits']},
              'pairs': {}, 'inputs_sha256': {
                  str(path): sha256_file(path) for path in
                  [config_path, config['population'], config['test_fasta'], config['aux_domains'],
                   config['aux_split'], *config['natv_splits'].values()]},
              'implementation_sha256': sha256_file(Path(__file__))}
    for aux_role in ('train', 'val'):
        for natv_role in ('validation', 'test'):
            for region in ('full', 'nmnat'):
                query = aux_sets[aux_role]
                reference = {uid: values[region] for (role, uid), values in natv.items()
                             if role == natv_role}
                events, truncated = containment_events(query, reference, max_events=500)
                for event in events:
                    if event['direction'] == 'query_in_reference':
                        event['reference_component'] = natv[natv_role, event['reference_id']]['component']
                    else:
                        event['reference_component'] = natv[natv_role, event['reference_id']]['component']
                pair = f'aux_{aux_role}_vs_natv_{natv_role}_{region}'
                result['pairs'][pair] = {
                    'auxiliary_sequences': len(query), 'natv_sequences': len(reference),
                    'events': len(events), 'events_truncated': truncated,
                    'auxiliary_with_event': len({event['query_id'] for event in events}),
                    'natv_with_event': len({event['reference_id'] for event in events}),
                    'events_detail': events}
    known = [event for event in result['pairs']['aux_train_vs_natv_validation_nmnat']['events_detail']
             if 120 <= event['shorter_length'] <= 128]
    result['known_124_case_candidates'] = known
    write_json(output, result)
    return result
