"""Recover and stratify the old 8,849 PF01467-only sequences.

Reads only the stage3 whitelist and current training/validation inputs. It does
not read test, sealed, conflict or excluded sequences, and does not train.
"""
from __future__ import annotations

import csv
import json
import platform
import random
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

from Bio import SeqIO
from Bio.Align import PairwiseAligner, substitution_matrices

from .data import Target, load_auxiliary_inputs, load_yaml, load_fasta, sha256_file, write_json

ALIGNER = PairwiseAligner(mode='local', substitution_matrix=substitution_matrices.load('BLOSUM62'))
ALIGNER.open_gap_score = -10
ALIGNER.extend_gap_score = -0.5
WT = None


def _map_alignment(alignment, target_length, query_length):
    mapped = {}
    for (t0, t1), (q0, q1) in zip(alignment.aligned[0], alignment.aligned[1]):
        for offset in range(t1 - t0):
            target_index = t0 + offset
            query_index = q0 + offset
            target_index = int(target_index)
            query_index = int(query_index)
            if 0 <= target_index < target_length and 0 <= query_index < query_length:
                mapped[target_index] = query_index
    return mapped


def stable_alignment(target, query, max_optimal):
    alignments = ALIGNER.align(target, query)
    if len(alignments) == 0:
        return {'stable': {}, 'coverage': 0.0, 'identity': None, 'n_optimal': 0,
                'qstart': None, 'qend': None, 'ambiguous': True}
    first = alignments[0]
    first_map = _map_alignment(first, len(target), len(query))
    values = {position: set() for position in range(len(target))}
    n_optimal = len(alignments)
    if n_optimal <= max_optimal:
        for alignment in alignments:
            mapping = _map_alignment(alignment, len(target), len(query))
            for position in values:
                value = mapping.get(position)
                values[position].add(value)
    stable = {}
    if n_optimal <= max_optimal:
        for position, found in values.items():
            if len(found) == 1:
                query_index = next(iter(found))
                if query_index is not None and 0 <= query_index < len(query):
                    stable[position] = query_index
    if not first_map:
        return {'stable': stable, 'coverage': 0.0, 'identity': None, 'n_optimal': n_optimal,
                'qstart': None, 'qend': None, 'ambiguous': n_optimal > 1}
    target_span = max(first_map) - min(first_map) + 1
    target_covered = len(first_map)
    query_indices = [value for value in first_map.values()]
    counts = first.counts()
    return {
        'stable': stable,
        'coverage': target_covered / len(target),
        'target_span_fraction': target_span / len(target),
        'identity': counts.identities / counts.aligned if counts.aligned else None,
        'n_optimal': n_optimal,
        'qstart': min(query_indices) + 1,
        'qend': max(query_indices) + 1,
        'ambiguous': n_optimal > 1,
    }


def _flags(meta):
    evidence = meta.get('evidence') or {}
    ec = set(meta.get('uniprot_ec') or [])
    return {
        'ec_2_7_7_1': '2.7.7.1' in ec,
        'ec_2_7_7_18': '2.7.7.18' in ec,
        'ec_dual': '2.7.7.1' in ec and '2.7.7.18' in ec,
        'nadr_multifunction': bool(evidence.get('nadr_multifunction_annotation')) or
                              str(meta.get('uniprot_gene', '')).lower() == 'nadr',
        'text_nmnat': bool(evidence.get('name_nmnat_or_nadm')),
        'partial': bool(evidence.get('partial_or_fragment')),
        'annotation_conflict': bool(evidence.get('annotation_conflict')),
        'header_disagreement': bool(evidence.get('header_annotation_disagreement')),
        'unmapped': meta.get('stage1_final_status') == 'UNMAPPED',
    }


def _write_fasta(path, records):
    with open(path, 'w') as handle:
        for name, sequence in records:
            handle.write(f'>{name}\n{sequence}\n')


def _run(cmd, cwd):
    value = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if value.returncode:
        raise RuntimeError('command failed: ' + ' '.join(map(str, cmd)) + '\n' + value.stderr[-3000:])


def _clusters(path):
    mapping = {}
    members = defaultdict(list)
    with open(path) as handle:
        for line in handle:
            parts = line.rstrip().split('\t')
            if len(parts) < 2:
                continue
            rep, member = parts[0], parts[1]
            mapping[member] = rep
            members[rep].append(member)
    return mapping, members


def _cluster_metrics(mapping, members, old_prefix, train_prefix, val_prefix):
    result = {'total': len(members), 'old_only': 0, 'both': 0, 'current_only': 0,
              'old_classes': Counter()}
    for rep, values in members.items():
        old = any(value.startswith(old_prefix) for value in values)
        train = any(value.startswith(train_prefix) for value in values)
        val = any(value.startswith(val_prefix) for value in values)
        current = train or val
        if old and current:
            result['both'] += 1
        elif old:
            result['old_only'] += 1
        elif current:
            result['current_only'] += 1
        if old:
            if train and val:
                result['old_classes']['mixed_current'] += 1
            elif train:
                result['old_classes']['train_overlap'] += 1
            elif val:
                result['old_classes']['val_background'] += 1
            else:
                result['old_classes']['new'] += 1
    result['old_classes'] = dict(result['old_classes'])
    return result


def preprocess(config_path):
    global WT
    started = time.monotonic()
    config = load_yaml(config_path)
    params = config['parameters']
    source = config['source']
    current = config['current']
    output = Path(config['output'])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output already contains files: {output}')
    output.mkdir(parents=True, exist_ok=True)
    work = output / '_work'
    work.mkdir(exist_ok=True)
    target = Target.load(current['target'])
    WT = target.sequence[3:181]
    mmseqs = Path(params['mmseqs'])
    if not mmseqs.exists():
        raise FileNotFoundError(mmseqs)

    profile = {}
    with open(source['profile_manifest'], newline='') as handle:
        for row in csv.DictReader(handle):
            if row['architecture'] == 'nmnat_only':
                profile[row['seq_hash']] = row
    allowed = set(profile)
    if len(allowed) != 8849:
        raise ValueError(f'Expected 8849 PF01467-only records, got {len(allowed)}')
    corpus = {}
    with open(source['corpus_manifest']) as handle:
        for line in handle:
            item = json.loads(line)
            if item.get('seq_hash') in allowed:
                corpus[item['seq_hash']] = item
    index = SeqIO.index_db(str(work / 'stage3.sqlite'), source['stage3_fasta'], 'fasta')
    missing = allowed - set(index)
    if missing:
        raise ValueError(f'{len(missing)} whitelist IDs missing from stage3')
    sequences = {seq_hash: str(index[seq_hash].seq) for seq_hash in allowed}
    index.close()
    if len(sequences) != 8849:
        raise ValueError('Stage3 whitelist size differs')

    current_population = load_fasta(current['population'])
    current_train = _read_split(current['training'])
    current_val = _read_split(current['validation'])
    _read_ids(current['test'])  # IDs only; test sequences are not in the raw population
    current_records = []
    for role, rows in [('train', current_train), ('val', current_val)]:
        for row in rows:
            full = current_population[row['sequence_id']]
            if __import__('hashlib').sha256(full.encode()).hexdigest() != row['sequence_id']:
                raise ValueError(f'Current sequence SHA differs: {row["sequence_id"]}')
            domain = full[row['nmnat_from'] - 1:row['nmnat_to']]
            alignment = stable_alignment(WT, domain, int(params['max_optimal']))
            current_records.append({'role': role, 'sequence_id': row['sequence_id'],
                                    'component': row['component'], 'full': full,
                                    'domain': domain, 'alignment': alignment})
    current_train_residues = Counter()
    current_train_positions = Counter()
    current_train_pairs = set()
    for record in current_records:
        if record['role'] != 'train':
            continue
        substitutions = set()
        for wt_index, query_index in record['alignment']['stable'].items():
            position = wt_index + 4
            residue = record['domain'][query_index]
            current_train_positions[position] += 1
            current_train_residues[(position, residue)] += 1
            if residue != WT[wt_index]:
                substitutions.add((position, residue))
        ordered = sorted(substitutions)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                current_train_pairs.add(frozenset((ordered[i], ordered[j])))

    records = []
    source_metadata = {}
    aux_sequences = {}
    full_sequences = {}
    domain_sequences = {}
    aux_representatives = {}
    for seq_hash in sorted(allowed):
        sequence = sequences[seq_hash]
        if __import__('hashlib').sha256(sequence.encode()).hexdigest() != seq_hash:
            raise ValueError(f'Old sequence SHA differs: {seq_hash}')
        row = profile[seq_hash]
        meta = corpus.get(seq_hash, {})
        flags = _flags(meta)
        labels = []
        if flags['ec_2_7_7_1']:
            labels.append('ec_NMNAT')
        if flags['ec_dual']:
            labels.append('ec_dual')
        if flags['nadr_multifunction']:
            labels.append('NadR')
        if flags['text_nmnat']:
            labels.append('text')
        if not labels:
            labels.append('unknown')
        if flags['ec_dual']:
            annotation_layer = 'EC_dual'
        elif flags['ec_2_7_7_1']:
            annotation_layer = 'EC_NMNAT'
        elif flags['ec_2_7_7_18']:
            annotation_layer = 'EC_NaMNAT'
        elif flags['nadr_multifunction']:
            annotation_layer = 'NadR'
        elif flags['text_nmnat']:
            annotation_layer = 'text'
        else:
            annotation_layer = 'unknown'
        start = int(float(row['pf01467_best_alignment_start'])) if row['pf01467_best_alignment_start'] else None
        end = int(float(row['pf01467_best_alignment_end'])) if row['pf01467_best_alignment_end'] else None
        domain = sequence[start - 1:end] if start and end and 1 <= start <= end <= len(sequence) else ''
        full_alignment = stable_alignment(WT, sequence, int(params['max_optimal']))
        domain_alignment = stable_alignment(WT, domain, int(params['max_optimal'])) if domain else None
        aux_kind = None
        aux_sequence = ''
        aux_start = aux_end = None
        used_alignment = None
        reason = ''
        if (full_alignment['coverage'] >= float(params['full_context_target_coverage']) and
                full_alignment['identity'] is not None and
                full_alignment['identity'] >= float(params['min_identity']) and
                len(full_alignment['stable']) >= int(params['min_aux_positions'])):
            aux_kind = 'full_context_mapping'
            aux_start, aux_end = full_alignment['qstart'], full_alignment['qend']
            aux_sequence = sequence[aux_start - 1:aux_end]
            used_alignment = full_alignment
        elif (domain_alignment is not None and domain_alignment['identity'] is not None and
              domain_alignment['identity'] >= float(params['min_identity']) and
              len(domain_alignment['stable']) >= int(params['min_aux_positions'])):
            aux_kind = 'local_core'
            aux_start, aux_end = start, end
            aux_sequence = domain
            used_alignment = domain_alignment
        else:
            reason = 'no_reliable_WT_mapping'
        wt_residues = {}
        aux_positions = []
        aux_pairs = []
        if used_alignment is not None:
            offset = (aux_start - 1) if aux_kind == 'full_context_mapping' else 0
            for wt_index, query_index in used_alignment['stable'].items():
                query_index = int(query_index)
                position = int(wt_index) + 4
                residue = aux_sequence[query_index - offset] if 0 <= query_index - offset < len(aux_sequence) else None
                if residue is not None:
                    wt_residues[position] = residue
                    aux_positions.append(int(query_index - offset))
                    aux_pairs.append({'wt_position': int(position), 'query_index': int(query_index - offset)})
        full_sha = __import__('hashlib').sha256(sequence.encode()).hexdigest()
        domain_sha = __import__('hashlib').sha256(domain.encode()).hexdigest() if domain else None
        aux_sha = __import__('hashlib').sha256(aux_sequence.encode()).hexdigest() if aux_sequence else None
        aux_uid = f'aux:{aux_sha}' if aux_sha else None
        full_uid = f'full:{full_sha}'
        domain_uid = f'domain:{domain_sha}' if domain_sha else None
        record = {
            'seq_hash': seq_hash,
            'source_header': (meta.get('fasta_headers') or [''])[0],
            'source_header_count': len(meta.get('fasta_headers') or []),
            'functional_labels': ';'.join(labels),
            'annotation_layer': annotation_layer,
            'source_accessions': ';'.join(meta.get('source_accessions') or []),
            'uniprot_accession': meta.get('uniprot_accession'),
            'uniprot_gene': meta.get('uniprot_gene'),
            'uniprot_ec': ';'.join(meta.get('uniprot_ec') or []),
            'stage1_final_status': meta.get('stage1_final_status'),
            'architecture': row['architecture'],
            'profile_coverage': float(row['pf01467_best_profile_coverage']) if row['pf01467_best_profile_coverage'] else None,
            'query_coverage': float(row['pf01467_best_query_coverage']) if row['pf01467_best_query_coverage'] else None,
            'nmnat_hit_count': int(row['nmnat_hit_count']),
            'nmnat_intervals': row['nmnat_intervals'],
            'partial': flags['partial'],
            'multi_hit': int(row['nmnat_hit_count']) > 1,
            'ec_2_7_7_1': flags['ec_2_7_7_1'],
            'ec_2_7_7_18': flags['ec_2_7_7_18'],
            'ec_dual': flags['ec_dual'],
            'nadr_multifunction': flags['nadr_multifunction'],
            'text_nmnat': flags['text_nmnat'],
            'annotation_conflict': flags['annotation_conflict'],
            'header_disagreement': flags['header_disagreement'],
            'unmapped': flags['unmapped'],
            'full_length': len(sequence),
            'domain_start': start, 'domain_end': end, 'domain_length': len(domain),
            'domain_source': 'PF01467_best_alignment',
            'wt_target_coverage': full_alignment['coverage'],
            'wt_target_span_fraction': full_alignment.get('target_span_fraction'),
            'wt_identity': full_alignment['identity'],
            'wt_n_optimal': full_alignment['n_optimal'],
            'wt_ambiguous': full_alignment['ambiguous'],
            'wt_reliable_positions': len(full_alignment['stable']),
            'aux_kind': aux_kind,
            'aux_confidence': ('high_confidence' if aux_kind and not (flags['partial'] or int(row['nmnat_hit_count']) > 1)
                               else 'risk_partial_or_multihit' if aux_kind else 'rejected'),
            'completeness_risk': ('partial_or_multihit' if flags['partial'] or int(row['nmnat_hit_count']) > 1
                                  else 'none'),
            'aux_start': aux_start, 'aux_end': aux_end,
            'aux_length': len(aux_sequence),
            'aux_uid': aux_uid, 'full_uid': full_uid, 'domain_uid': domain_uid,
            'aux_eligible': aux_kind is not None,
            'aux_reason': reason,
            'main_task_eligible': False,
            'main_task_reason': 'PF01467-only; no ordered NMNAT-Nudix dual domain',
            'wt_residues_json': json.dumps(wt_residues, sort_keys=True),
            'aux_positions_json': json.dumps(aux_positions),
        }
        records.append(record)
        source_metadata[seq_hash] = {'headers': meta.get('fasta_headers') or [], 'accessions': meta.get('source_accessions') or [],
                                     'stage1_final_status': meta.get('stage1_final_status'),
                                     'uniprot_accession': meta.get('uniprot_accession'),
                                     'uniprot_gene': meta.get('uniprot_gene'),
                                     'uniprot_ec': meta.get('uniprot_ec') or [],
                                     'evidence': meta.get('evidence') or {}}
        full_sequences.setdefault(full_sha, sequence)
        if domain:
            domain_sequences.setdefault(domain_sha, domain)
        if aux_uid:
            aux_sequences.setdefault(aux_uid, aux_sequence)
            previous = aux_representatives.get(aux_uid)
            if previous is None or len(aux_positions) > len(previous['positions']):
                aux_representatives[aux_uid] = {'sequence': aux_sequence, 'kind': aux_kind,
                                                'positions': aux_positions, 'pairs': aux_pairs,
                                                'wt_residues': wt_residues,
                                                'aux_start': aux_start, 'aux_end': aux_end,
                                                'confidence': record['aux_confidence'],
                                                'partial': record['partial'], 'multi_hit': record['multi_hit']}
    # Deduplicated FASTA inputs.
    _write_fasta(output / 'full_length.fasta', [(f'full:{key}', value)
                                                for key, value in sorted(full_sequences.items())])
    _write_fasta(output / 'aux_domains.fasta', [(key, value)
                                                for key, value in sorted(aux_sequences.items())])
    _write_fasta(output / 'pf01467_core_domains.fasta', [(f'domain:{key}', value)
                                                         for key, value in sorted(domain_sequences.items())])
    masks = {}
    for key, value in sorted(aux_representatives.items()):
        masks[key] = {'kind': value['kind'], 'confidence': value['confidence'],
                      'partial': value['partial'], 'multi_hit': value['multi_hit'],
                      'positions': sorted(set(value['positions'])),
                      'wt_positions': sorted(value['wt_residues']),
                      'pairs': [[pair['query_index'], pair['wt_position']] for pair in value['pairs']],
                      'aux_start': value['aux_start'], 'aux_end': value['aux_end']}
    with open(output / 'aux_masks.json', 'w') as handle:
        json.dump(masks, handle, ensure_ascii=False, separators=(',', ':'))
        handle.write('\n')
    high = {key: value for key, value in aux_representatives.items() if value['confidence'] == 'high_confidence'}
    risk = {key: value for key, value in aux_representatives.items() if value['confidence'] == 'risk_partial_or_multihit'}
    _write_fasta(output / 'aux_domains_high_confidence.fasta', [(key, value['sequence']) for key, value in sorted(high.items())])
    _write_fasta(output / 'aux_domains_risk.fasta', [(key, value['sequence']) for key, value in sorted(risk.items())])
    (output / 'aux_high_confidence_ids.txt').write_text('\n'.join(sorted(high)) + '\n')
    (output / 'aux_risk_ids.txt').write_text('\n'.join(sorted(risk)) + '\n')
    write_json(output / 'source_metadata.json', source_metadata)

    # Joint clustering of old auxiliary domains with current train/val domains.
    domain_fasta = output / 'joint_domain_clustering.fasta'
    domain_entries = [(key, value) for key, value in sorted(aux_sequences.items())]
    domain_entries += [(f'current_{role}:{record["sequence_id"]}', record['domain'])
                       for record in current_records for role in [record['role']]]
    _write_fasta(domain_fasta, domain_entries)
    full_fasta = output / 'joint_full_clustering.fasta'
    full_entries = [(f'full:{key}', value) for key, value in sorted(full_sequences.items())]
    full_entries += [(f'current_{record["role"]}:{record["sequence_id"]}', record['full'])
                     for record in current_records]
    _write_fasta(full_fasta, full_entries)

    cluster_results = {'domain': {}, 'full': {}}
    cluster_maps = {'domain': {}, 'full': {}}
    cluster_members_all = {'domain': {}, 'full': {}}
    old_prefix = {'domain': 'aux:', 'full': 'full:'}
    targets = {'domain': (domain_fasta, 'domain'), 'full': (full_fasta, 'full')}
    for kind, (fasta, label) in targets.items():
        for identity in params['identities']:
            key = f'{identity:.2f}'
            prefix = (work / f'{label}_{key}').resolve()
            _run([str(mmseqs), 'easy-cluster', str(Path(fasta).resolve()), str(prefix), str(prefix) + '_tmp',
                  '--min-seq-id', str(identity), '--cov-mode', '0', '-c', str(params['coverage']),
                  '--cluster-mode', str(params['cluster_mode']), '--threads', str(params['threads'])],
                 cwd=work)
            mapping, members = _clusters(str(prefix) + '_cluster.tsv')
            cluster_maps[kind][key] = mapping
            cluster_members_all[kind][key] = members
            cluster_results[kind][key] = _cluster_metrics(
                mapping, members, old_prefix[kind], 'current_train:', 'current_val:')
            with open(output / f'{label}_clusters_{key.replace(".", "")}.tsv', 'w') as table:
                for rep, values in sorted(members.items()):
                    for member in values:
                        table.write(f'{rep}\t{member}\n')

    def class_for(uid, mapping, members):
        if uid is None or uid not in mapping:
            return 'unmapped', None
        rep = mapping[uid]
        values = members.get(rep, [])
        train = any(value.startswith('current_train:') for value in values)
        val = any(value.startswith('current_val:') for value in values)
        if train and val:
            return 'mixed_current', rep
        if train:
            return 'train_overlap', rep
        if val:
            return 'val_background', rep
        return 'new', rep

    aux_by_uid = defaultdict(list)
    for record in records:
        if record['aux_uid']:
            aux_by_uid[record['aux_uid']].append(record)
        for label, suffix in (('0.70', '70'), ('0.50', '50'), ('0.30', '30')):
            dclass, dcluster = class_for(record['aux_uid'], cluster_maps['domain'][label],
                                         cluster_members_all['domain'][label])
            fclass, fcluster = class_for(record['full_uid'], cluster_maps['full'][label],
                                         cluster_members_all['full'][label])
            record[f'domain_class{suffix}'] = dclass
            record[f'domain_cluster{suffix}'] = dcluster
            record[f'full_class{suffix}'] = fclass
            record[f'full_cluster{suffix}'] = fcluster

    # Auxiliary split uses the actual auxiliary domain sequence and the 50%
    # joint domain group; partial/multihit risk targets are kept as a separate
    # local input and do not enter the default high-confidence split.
    default_records = [record for record in records
                       if record['aux_eligible'] and record['aux_confidence'] == 'high_confidence'
                       and record['domain_class50'] == 'new']
    new_clusters = sorted({record['domain_cluster50'] for record in default_records
                           if record['domain_cluster50']})
    rng = random.Random(int(params['aux_split_seed']))
    rng.shuffle(new_clusters)
    n_val = max(1, round(len(new_clusters) * float(params['aux_val_fraction']))) if len(new_clusters) > 1 else 0
    val_clusters = set(new_clusters[:n_val])
    split_by_uid = {}
    for record in records:
        if not record['aux_uid']:
            record['aux_split'] = 'excluded'
            record['aux_weight'] = None
            record['aux_split_reason'] = 'no_reliable_WT_mapping'
            continue
        if record['aux_confidence'] != 'high_confidence':
            record['aux_split'] = 'excluded_risk'
            record['aux_weight'] = None
            record['aux_split_reason'] = 'partial_or_multihit_risk'
            continue
        if record['domain_class50'] != 'new':
            record['aux_split'] = 'excluded'
            record['aux_weight'] = None
            record['aux_split_reason'] = record['domain_class50'] + '_50_group'
            continue
        if n_val == 0:
            record['aux_split'] = 'excluded'
            record['aux_weight'] = None
            record['aux_split_reason'] = 'no_aux_validation_clusters'
            continue
        split = 'val' if record['domain_cluster50'] in val_clusters else 'train'
        split_by_uid.setdefault(record['aux_uid'], split)
        record['aux_split'] = split_by_uid[record['aux_uid']]
        record['aux_split_reason'] = 'new_50_group'
    unique_split = {record['aux_uid']: (record['aux_split'], record['domain_cluster50'])
                    for record in records if record['aux_uid']
                    and record['aux_split'] in {'train', 'val'}}
    cluster_sizes = Counter(unique_split.values())
    uid_weights = {uid: 1.0 / max(cluster_sizes[(split, cluster)], 1)
                   for uid, (split, cluster) in unique_split.items()}
    totals = Counter(split for split, _cluster in unique_split.values())
    means = {split: sum(uid_weights[uid] for uid, (value, _cluster) in unique_split.items()
                        if value == split) / totals[split]
             for split in ('train', 'val') if totals[split]}
    for uid in uid_weights:
        uid_weights[uid] /= means[unique_split[uid][0]]
    for record in records:
        record['aux_weight'] = uid_weights.get(record['aux_uid']) if record['aux_split'] in {'train', 'val'} else None
    split_rows = []
    risk_rows = []
    representative_by_uid = {}
    for record in records:
        if record['aux_uid'] and record['aux_uid'] not in representative_by_uid:
            representative_by_uid[record['aux_uid']] = record
    for uid, record in sorted(representative_by_uid.items()):
        if record['aux_split'] in {'train', 'val'}:
            split_rows.append({'aux_uid': uid, 'sequence_sha256': uid.split(':', 1)[1],
                               'aux_kind': record['aux_kind'], 'length': record['aux_length'],
                               'split': record['aux_split'], 'weight': uid_weights.get(uid),
                               'domain_class50': record['domain_class50'],
                               'domain_class70': record['domain_class70'],
                               'full_class50': record['full_class50'],
                               'domain_cluster50': record['domain_cluster50'],
                               'domain_cluster70': record['domain_cluster70'],
                               'n_source_records': len(aux_by_uid.get(uid, []))})
        elif record['aux_split'] == 'excluded_risk':
            risk_rows.append({'aux_uid': uid, 'sequence_sha256': uid.split(':', 1)[1],
                              'aux_kind': record['aux_kind'], 'length': record['aux_length'],
                              'confidence': record['aux_confidence'],
                              'domain_class50': record['domain_class50'],
                              'domain_cluster50': record['domain_cluster50'],
                              'partial': record['partial'], 'multi_hit': record['multi_hit'],
                              'reason': record['aux_split_reason']})
    for split in ('train', 'val'):
        rows = [row for row in split_rows if row['split'] == split]
        mean = sum(row['weight'] for row in rows) / len(rows) if rows else 1.0
        for row in rows:
            row['weight'] /= mean
            uid_weights[row['aux_uid']] = row['weight']
    _write_csv(output / 'aux_split.csv', split_rows)
    _write_csv(output / 'aux_risk.csv', risk_rows)
    load_auxiliary_inputs(output)

    # Increment events: current train residues and reliable WT positions only.
    event_clusters = defaultdict(set)
    event_records = defaultdict(int)
    event_kind = {}
    event_splits = defaultdict(set)
    event_confidence = defaultdict(set)
    for record in records:
        if not record['aux_eligible']:
            continue
        wt_residues = json.loads(record['wt_residues_json'])
        for position, residue in wt_residues.items():
            position = int(position)
            if residue == WT[position - 4]:
                continue
            key = (position, residue)
            event_clusters[key].add(record['domain_cluster50'])
            event_records[key] += 1
            event_splits[key].add(record['aux_split'])
            event_confidence[key].add(record['aux_confidence'])
            if current_train_positions[position] > 0:
                event_kind[key] = 'support_increase' if current_train_residues[key] else 'new'
            else:
                event_kind[key] = 'current_train_position_unmapped'
    event_rows = []
    for key, clusters in sorted(event_clusters.items()):
        splits = sorted(event_splits[key])
        confidence = sorted(event_confidence[key])
        event_rows.append({'wt_position': key[0], 'alt_residue': key[1],
                           'kind': event_kind[key], 'old_clusters': len(clusters),
                           'old_records': event_records[key],
                           'aux_splits': ';'.join(splits),
                           'aux_confidence': ';'.join(confidence),
                           'default_train_eligible': ('train' in splits and 'high_confidence' in confidence),
                           'current_train_count': current_train_residues[key],
                           'current_train_positions': current_train_positions[key[0]]})
    _write_csv(output / 'new_single_events.csv', [row for row in event_rows if row['old_clusters'] >= 2])
    single_summary = {
        'all_inventory_events': len(event_rows),
        'new_events_all_inventory': sum(1 for row in event_rows if row['kind'] == 'new'),
        'new_events_ge2_clusters': sum(1 for row in event_rows
                                       if row['kind'] == 'new' and row['old_clusters'] >= 2),
        'new_events_default_train': sum(1 for row in event_rows
                                        if row['kind'] == 'new' and row['default_train_eligible']),
        'new_events_aux_val': sum(1 for row in event_rows
                                  if row['kind'] == 'new' and 'val' in row['aux_splits'].split(';')),
        'support_increase_events': sum(1 for row in event_rows
                                       if row['kind'] == 'support_increase'),
        'current_train_position_unmapped': sum(1 for row in event_rows
                                               if row['kind'] == 'current_train_position_unmapped'),
    }
    # Double events only on full-context mapping, reliable WT positions.
    current_record_positions = []
    current_position_bits = defaultdict(int)
    for index, record in enumerate(current_records):
        if record['role'] != 'train':
            continue
        positions = {wt_index + 4 for wt_index in record['alignment']['stable']}
        current_record_positions.append(positions)
        for position in positions:
            current_position_bits[position] |= 1 << index
    old_pairs = defaultdict(set)
    pair_splits = defaultdict(set)
    pair_confidence = defaultdict(set)
    for record in records:
        if record['aux_kind'] != 'full_context_mapping':
            continue
        substitutions = sorted({(int(position), residue) for position, residue in
                                json.loads(record['wt_residues_json']).items()
                                if residue != WT[int(position) - 4]})
        for i in range(len(substitutions)):
            for j in range(i + 1, len(substitutions)):
                pair = frozenset((substitutions[i], substitutions[j]))
                old_pairs[pair].add(record['domain_cluster50'])
                pair_splits[pair].add(record['aux_split'])
                pair_confidence[pair].add(record['aux_confidence'])
    double_rows = []
    for pair, clusters in sorted(old_pairs.items(), key=lambda item: (-len(item[1]), str(item[0]))):
        positions = sorted({position for position, _residue in pair})
        both = (current_position_bits[positions[0]] & current_position_bits[positions[1]]).bit_count()
        seen = pair in current_train_pairs
        new = (not seen) and both > 0
        splits = sorted(pair_splits[pair])
        confidence = sorted(pair_confidence[pair])
        double_rows.append({'pair': json.dumps(sorted(map(list, pair))),
                            'old_clusters': len(clusters), 'current_train_both_mapped': both,
                            'current_train_seen': seen, 'new': new,
                            'aux_splits': ';'.join(splits),
                            'aux_confidence': ';'.join(confidence),
                            'default_train_eligible': ('train' in splits and 'high_confidence' in confidence)})
    supported_double = [row for row in double_rows if row['new'] and row['old_clusters'] >= 2]
    _write_csv(output / 'new_double_events_top.csv',
               sorted(supported_double, key=lambda row: (-row['old_clusters'], row['pair']))[:50])
    double_summary = {
        'all_inventory_pairs': len(double_rows),
        'new_pairs_observable': sum(1 for row in double_rows if row['new']),
        'new_pairs_ge2_clusters': len(supported_double),
        'new_pairs_default_train': sum(1 for row in double_rows
                                       if row['new'] and row['default_train_eligible']),
        'seen_pairs': sum(1 for row in double_rows if row['current_train_seen']),
        'unobservable_pairs_no_current_mapping': sum(1 for row in double_rows
                                                     if row['current_train_both_mapped'] == 0),
        'top_pairs_written': min(50, len(supported_double)),
        'support_bins': {
            '2_4': sum(1 for row in supported_double if 2 <= row['old_clusters'] < 5),
            '5_9': sum(1 for row in supported_double if 5 <= row['old_clusters'] < 10),
            'ge10': sum(1 for row in supported_double if row['old_clusters'] >= 10),
        },
    }

    # Functional-layer summary; annotation layer is a display grouping only.
    functional_groups = defaultdict(lambda: {'records': 0, 'aux_uids': set(),
                                               'clusters50': set(), 'new_clusters50': set(),
                                               'aux_train_records': 0, 'aux_val_records': 0})
    for record in records:
        key = (record['annotation_layer'], record['aux_kind'] or 'rejected',
               record['completeness_risk'], record['aux_confidence'])
        group = functional_groups[key]
        group['records'] += 1
        if record['aux_uid']:
            group['aux_uids'].add(record['aux_uid'])
        if record['domain_cluster50']:
            group['clusters50'].add(record['domain_cluster50'])
            if record['domain_class50'] == 'new':
                group['new_clusters50'].add(record['domain_cluster50'])
        if record['aux_split'] == 'train':
            group['aux_train_records'] += 1
        if record['aux_split'] == 'val':
            group['aux_val_records'] += 1
    functional_rows = []
    for (layer, kind, risk, confidence), group in sorted(functional_groups.items()):
        functional_rows.append({'annotation_layer': layer, 'aux_kind': kind,
                                'completeness_risk': risk, 'aux_confidence': confidence,
                                'records': group['records'],
                                'unique_aux_domains': len(group['aux_uids']),
                                'clusters50': len(group['clusters50']),
                                'new_clusters50': len(group['new_clusters50']),
                                'aux_train_records': group['aux_train_records'],
                                'aux_val_records': group['aux_val_records']})
    _write_csv(output / 'functional_layer_summary.csv', functional_rows)

    # Write the complete record table and summary last.
    record_fields = list(records[0])
    _write_csv(output / 'records.csv', records, record_fields)
    summary = {
        'status': 'COMPLETED',
        'inputs': {
            'stage3_fasta': {'path': source['stage3_fasta'], 'sha256': sha256_file(source['stage3_fasta'])},
            'profile_manifest': {'path': source['profile_manifest'], 'sha256': sha256_file(source['profile_manifest'])},
            'corpus_manifest': {'path': source['corpus_manifest'], 'sha256': sha256_file(source['corpus_manifest'])},
            'population': {'path': current['population'], 'sha256': sha256_file(current['population'])},
            'training_split': {'path': current['training'], 'sha256': sha256_file(current['training'])},
            'validation_split': {'path': current['validation'], 'sha256': sha256_file(current['validation'])},
            'target': {'path': current['target'], 'sha256': sha256_file(current['target'])},
            'config': {'path': config_path, 'sha256': sha256_file(config_path)},
        },
        'parameters': params,
        'runtime_seconds': round(time.monotonic() - started, 2),
        'counts': {
            'old_records': len(records),
            'main_task_new': 0,
            'aux_full_context_mapping': sum(1 for record in records if record['aux_kind'] == 'full_context_mapping'),
            'aux_local_core': sum(1 for record in records if record['aux_kind'] == 'local_core'),
            'aux_rejected': sum(1 for record in records if not record['aux_eligible']),
            'aux_unique_domains': len(aux_sequences),
            'full_unique_sequences': len(full_sequences),
            'pf01467_unique_domains': len(domain_sequences),
            'aux_split': dict(Counter(record['aux_split'] for record in records)),
            'aux_split_unique': dict(Counter(row['split'] for row in split_rows)),
            'aux_risk_unique': len(risk_rows),
            'aux_confidence': dict(Counter(record['aux_confidence'] for record in records)),
            'functional_overlap_counts': dict(Counter(label for record in records
                                                      for label in record['functional_labels'].split(';'))),
            'annotation_layer_counts': dict(Counter(record['annotation_layer'] for record in records)),
            'source_metadata_records': len(source_metadata),
            'aux_rejected_reasons': dict(Counter(record['aux_reason'] for record in records
                                                 if not record['aux_eligible'])),
            'current_train': len(current_train), 'current_validation': len(current_val),
            'test_ids': len(_read_ids(current['test'])),
            'domain_classes70': dict(Counter(record['domain_class70'] for record in records if record['aux_uid'])),
            'domain_classes50': dict(Counter(record['domain_class50'] for record in records if record['aux_uid'])),
            'domain_classes30': dict(Counter(record['domain_class30'] for record in records if record['aux_uid'])),
            'full_classes70': dict(Counter(record['full_class70'] for record in records)),
            'full_classes50': dict(Counter(record['full_class50'] for record in records)),
            'full_classes30': dict(Counter(record['full_class30'] for record in records)),
        },
        'cluster_metrics': cluster_results,
        'cluster_parameters': {'coverage': params['coverage'], 'identities': params['identities'],
                               'cluster_mode': params['cluster_mode'], 'threads': params['threads'],
                               'mmseqs': str(mmseqs)},
        'implementation_sha256': sha256_file(Path(__file__)),
        'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[2], text=True).strip(),
        'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=Path(__file__).resolve().parents[2], text=True).strip()),
        'tool_versions': {'python': platform.python_version(),
                           'biopython': __import__('Bio').__version__,
                           'mmseqs': subprocess.check_output([str(mmseqs), 'version'], text=True).strip()},
        'increment': {'single_events': single_summary, 'double_events_full_context': double_summary},
        'notes': [
            'main NatV task remains ordered NMNAT-Nudix and the original 1644/227 split; old 8849 contribute 0 main-task sequences',
            'aux_full_context requires WT target coverage >=0.8 with reliable positions; aux_local_core is a PF01467 local fragment, not a complete NatV domain',
            'aux split groups are actual auxiliary domain clusters at 50% identity; a group with current train/validation members is excluded from the default aux split',
            'test was read as IDs only; no test sequence homology check is claimed',
            'cluster counts and event counts are descriptive, not transfer or activity effects',
            'no training, generation, download or folding was performed',
        ],
    }
    summary['outputs'] = {path.name: sha256_file(path) for path in sorted(output.iterdir())
                          if path.is_file() and path.name != 'summary.json'}
    write_json(output / 'summary.json', summary)
    shutil.rmtree(work)
    return summary


def _read_split(path):
    rows = []
    with open(path, newline='') as handle:
        for row in csv.DictReader(handle):
            rows.append({'sequence_id': row['sequence_id'], 'component': row['component_v2'],
                         'nmnat_from': int(row['nmnat_cdd_from']),
                         'nmnat_to': int(row['nmnat_cdd_to'])})
    return rows


def _read_ids(path):
    with open(path, newline='') as handle:
        return [row['sequence_id'] for row in csv.DictReader(handle)]


def _write_csv(path, rows, fieldnames=None):
    if not rows:
        Path(path).write_text('')
        return
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
