#!/usr/bin/env python
"""Build a bounded NMNAT evolutionary design-evidence table for KVP40.

Reads existing audit outputs and the old reviewed NMNAT reference; no test,
sealed, conflict or excluded sequence is loaded. No MMseqs, download or folding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Align import PairwiseAligner, substitution_matrices

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from natvqvae.intake import sha256_file  # noqa: E402
from natvqvae.sequence import wt_sequence  # noqa: E402

AUDIT = ROOT / 'data/results/natv_evolution_data_audit/raw/pf01467_only_audit.csv'
OLD_FULL = ROOT / 'data/results/natv_evolution_data_audit/raw/old_only_full_proteins.fasta'
T80 = ROOT / 'data/results/natv_evolution_data_audit/raw/wt_full_context_hits_t80.csv'
T50 = ROOT / 'data/results/natv_evolution_data_audit/raw/wt_full_context_hits_t50.csv'
ANCHOR_AUDIT = ROOT / 'data/processed/anchor_mapping_audit.csv'
SAMPLES = ROOT / 'data/processed/samples.csv'
QV = ROOT / 'experiments/natv_conditional_qvae/representative_candidates.csv'
OLD_REF = Path('/home/yhshy/git_files/nad_qvae/experiments/run_reports/'
               'pretraining_scientific_validation_20260909/'
               'nmnat_specificity_reference_v1/domain_reference_v3')
REF_MANIFEST = OLD_REF / 'reference_manifest.csv'
REF_FASTA = OLD_REF / 'positive_domains.fasta'
REF_CLUSTERS = OLD_REF / 'cluster30_cov80/clusters_cluster.tsv'
RUN = ROOT / 'experiments/natv_evolution_informed_nmnat'
RAW = ROOT / 'data/results/natv_evolution_informed_nmnat/raw/phase_a'
ANCHORS = [14, 17, 135, 138]
DOMAIN_START, DOMAIN_END = 4, 181
WT = wt_sequence()
WT_DOMAIN = WT[DOMAIN_START - 1:DOMAIN_END]
WINDOW = 6
MAX_OPTIMAL = 32


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read_fasta(path):
    return {record.id: str(record.seq) for record in SeqIO.parse(str(path), 'fasta')}


def make_aligner():
    aligner = PairwiseAligner(mode='local', substitution_matrix=substitution_matrices.load('BLOSUM62'))
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    return aligner


def align_map(target, query, target_offset, count_optimal=False):
    alignments = make_aligner().align(target, query)
    alignment = alignments[0]
    n_optimal = len(alignments) if count_optimal else None
    mapped = {}
    query_indices = set()
    for (t0, t1), (q0, q1) in zip(alignment.aligned[0], alignment.aligned[1]):
        for offset in range(t1 - t0):
            target_index = t0 + offset
            query_index = q0 + offset
            if 0 <= target_index < len(target) and 0 <= query_index < len(query):
                mapped[target_offset + target_index] = query[query_index]
                query_indices.add(query_index)
    counts = alignment.counts()
    aligned_columns = int(counts.aligned)
    identity = counts.identities / aligned_columns if aligned_columns else None
    if mapped:
        target_span = max(mapped) - min(mapped) + 1
        query_span = max(query_indices) - min(query_indices) + 1
    else:
        target_span = 0
        query_span = 0
    return {
        'mapped': mapped,
        'target_span': target_span,
        'query_span': query_span,
        'query_start': min(query_indices) + 1 if query_indices else None,
        'query_end': max(query_indices) + 1 if query_indices else None,
        'aligned_columns': aligned_columns,
        'identity': identity,
        'n_optimal': n_optimal,
    }


def map_alignment(alignment, target, query, target_offset):
    mapped = {}
    for (t0, t1), (q0, q1) in zip(alignment.aligned[0], alignment.aligned[1]):
        for offset in range(t1 - t0):
            target_index = t0 + offset
            query_index = q0 + offset
            if 0 <= target_index < len(target) and 0 <= query_index < len(query):
                mapped[target_offset + target_index] = query[query_index]
    return mapped


def stable_positions(query, positions, max_optimal=MAX_OPTIMAL):
    alignments = make_aligner().align(WT_DOMAIN, query)
    n_optimal = len(alignments)
    if n_optimal > max_optimal:
        return {position: False for position in positions}, {}, n_optimal
    values = {position: set() for position in positions}
    for alignment in alignments:
        mapped = map_alignment(alignment, WT_DOMAIN, query, DOMAIN_START)
        for position in positions:
            values[position].add(mapped.get(position))
    stable = {position: len(values[position]) == 1 for position in positions}
    return stable, values, n_optimal


def variant_set(mapped, wt_domain, domain_start):
    result = set()
    for position, residue in mapped.items():
        offset = position - domain_start
        if 0 <= offset < len(wt_domain) and residue != wt_domain[offset]:
            result.add((position, wt_domain[offset], residue))
    return result


def anchor_fields(mapped):
    result = {}
    for anchor in ANCHORS:
        result[f'wt_{anchor}'] = mapped.get(anchor)
        result[f'wt_{anchor}_mapped'] = anchor in mapped
    return result


def window_fields(mapped):
    parts = []
    for position in range(DOMAIN_START, DOMAIN_END + 1):
        if min(abs(position - anchor) for anchor in ANCHORS) <= WINDOW and position in mapped:
            parts.append(f'{position}{mapped[position]}')
    return ';'.join(parts)


def region_of(position):
    if position <= 23:
        return 'N'
    if position >= 129:
        return 'C'
    return 'middle'


def aa_class(residue):
    if residue in 'AVILM':
        return 'hydrophobic'
    if residue in 'FWY':
        return 'aromatic'
    if residue in 'STNQ':
        return 'polar'
    if residue in 'KRH':
        return 'positive'
    if residue in 'DE':
        return 'negative'
    if residue in 'GP':
        return 'special'
    if residue == 'C':
        return 'cys'
    return 'other'


def conservative_change(wt_residue, alt_residue):
    if aa_class(wt_residue) == aa_class(alt_residue):
        return True
    return {wt_residue, alt_residue} in [
        {'D', 'E'}, {'K', 'R'}, {'S', 'T'}, {'N', 'Q'},
        {'I', 'V'}, {'L', 'M'}, {'F', 'Y'}, {'A', 'G'},
    ]


def load_reference():
    manifest = pd.read_csv(REF_MANIFEST)
    manifest = manifest[manifest.label == 'positive'].copy()
    sequences = read_fasta(REF_FASTA)
    cluster_of = {}
    for line in REF_CLUSTERS.read_text().splitlines():
        parts = line.split('\t')
        if len(parts) < 2:
            continue
        cluster_of[parts[1]] = parts[0]
    rows = []
    for item in manifest.itertuples():
        if item.accession not in sequences:
            continue
        rows.append({
            'record_id': f'ref:{item.accession}',
            'seq_hash': None,
            'accession': item.accession,
            'domain_sequence': sequences[item.accession],
            'full_sequence': None,
            'full_length': item.full_length,
            'domain_start': item.domain_start,
            'domain_end': item.domain_end,
            'domain_length': item.domain_length,
            'evidence_tier': item.evidence_tier,
            'ec_numbers': item.ec_numbers,
            'protein_existence': item.protein_existence,
            'cluster': 'ref:' + cluster_of.get(item.accession, item.accession),
            'source_sets': 'S4',
            'tier': 'T5',
            'full_context': False,
        })
    return pd.DataFrame(rows)


def load_qvae():
    frame = pd.read_csv(QV)
    single = set()
    pairs = set()
    for item in frame.itertuples():
        try:
            edits = json.loads(item.edits)
        except Exception:
            continue
        values = {(int(edit['position']), str(edit['to']))
                  for edit in edits if DOMAIN_START <= int(edit['position']) <= DOMAIN_END}
        single.update(values)
        ordered = sorted(values)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                pairs.add(frozenset((ordered[i], ordered[j])))
    return single, pairs


def choose_representatives(frame):
    rows = []
    for cluster, group in frame.groupby('cluster', sort=False):
        group = group.copy()
        group['tier_rank'] = group.tier.map({'T1': 0, 'T2': 1, 'T3': 2, 'T4': 3, 'T5': 4}).fillna(9)
        group['target_rank'] = pd.to_numeric(group.get('target_coverage'), errors='coerce').fillna(0)
        group['length_rank'] = pd.to_numeric(group.domain_length, errors='coerce').fillna(0)
        group['evidence_rank'] = group.get('evidence_tier', pd.Series('', index=group.index)).map(
            {'PROTEIN_LEVEL_REVIEWED': 0, 'REVIEWED_ANNOTATION': 1}).fillna(9)
        group = group.sort_values(['tier_rank', 'target_rank', 'length_rank', 'evidence_rank'],
                                  ascending=[True, False, False, True])
        chosen = group.iloc[0].to_dict()
        chosen['cluster_size'] = len(group)
        rows.append(chosen)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=RUN)
    args = parser.parse_args()
    started = time.monotonic()
    RAW.mkdir(parents=True, exist_ok=True)
    args.run.mkdir(parents=True, exist_ok=True)

    audit = pd.read_csv(AUDIT)
    roles = pd.read_csv(ROOT / 'data/processed/roles.csv')
    protected = set(roles.loc[roles.training_role_v2.isin(
        ['test', 'quarantined_old_sealed_component', 'annotation_conflict_quarantine',
         'excluded_noncanonical_or_domain']), 'sequence_id'])
    if set(audit.seq_hash) & protected:
        raise RuntimeError('old-only audit intersects protected current roles')
    audit['profile_coverage'] = pd.to_numeric(audit.profile_coverage, errors='coerce')
    audit['domain_length'] = pd.to_numeric(audit.domain_length, errors='coerce')
    audit['complete'] = (audit.profile_coverage >= 0.8) & (audit.domain_length >= 124)
    audit['ec_support'] = audit.ec_nmnat_only | audit.ec_dual
    full_sequences = read_fasta(OLD_FULL)
    t80 = pd.read_csv(T80)
    t50 = pd.read_csv(T50)
    t80_by_hash = t80.set_index('seq_hash').to_dict('index')
    s1 = audit[audit.complete & audit.ec_support].copy()
    s1_ec_nmnat = s1[s1.ec_nmnat_only].copy()
    t80_hashes = set(t80.seq_hash)
    s2_only = t80[~t80.seq_hash.isin(set(s1.seq_hash))].copy()

    cluster_column = '0.7_mode1_cluster'
    records = []
    for item in s1_ec_nmnat.itertuples():
        tier = 'T1' if item.seq_hash in t80_hashes else 'T2'
        records.append({
            'record_id': f'old:{item.seq_hash}', 'seq_hash': item.seq_hash,
            'accession': item.uniprot_accession, 'full_sequence': full_sequences.get(item.seq_hash),
            'domain_sequence': None, 'full_length': item.length,
            'domain_start': item.domain_start, 'domain_end': item.domain_end,
            'domain_length': item.domain_length,
            'source_sets': 'S1', 'tier': tier,
            'full_context': item.seq_hash in t80_hashes,
            'target_coverage': t80_by_hash.get(item.seq_hash, {}).get('target_coverage'),
            'profile_coverage': item.profile_coverage,
            'primary_layer': item.primary_layer, 'annotation_conflict': item.annotation_conflict,
        })
    for item in s1[s1.ec_dual].itertuples():
        records.append({
            'record_id': f'old:{item.seq_hash}', 'seq_hash': item.seq_hash,
            'accession': item.uniprot_accession, 'full_sequence': full_sequences.get(item.seq_hash),
            'domain_sequence': None, 'full_length': item.length,
            'domain_start': item.domain_start, 'domain_end': item.domain_end,
            'domain_length': item.domain_length,
            'source_sets': 'S1', 'tier': 'T4', 'full_context': False,
            'target_coverage': t80_by_hash.get(item.seq_hash, {}).get('target_coverage'),
            'profile_coverage': item.profile_coverage,
            'primary_layer': item.primary_layer, 'annotation_conflict': item.annotation_conflict,
        })
    for item in s2_only.itertuples():
        audit_row = audit[audit.seq_hash == item.seq_hash].iloc[0]
        records.append({
            'record_id': f'old:{item.seq_hash}', 'seq_hash': item.seq_hash,
            'accession': audit_row.uniprot_accession,
            'full_sequence': full_sequences.get(item.seq_hash), 'domain_sequence': None,
            'full_length': audit_row.length, 'domain_start': audit_row.domain_start,
            'domain_end': audit_row.domain_end, 'domain_length': audit_row.domain_length,
            'source_sets': 'S2', 'tier': 'T3',
            'full_context': True, 'target_coverage': item.target_coverage,
            'profile_coverage': audit_row.profile_coverage,
            'primary_layer': audit_row.primary_layer,
            'annotation_conflict': audit_row.annotation_conflict,
        })
    frame = pd.DataFrame(records)
    frame = frame.merge(
        audit[['seq_hash', cluster_column]].rename(columns={cluster_column: 'cluster'}),
        on='seq_hash', how='left', validate='one_to_one')
    frame['domain_sequence'] = None
    frame['full_context'] = frame.full_context.fillna(False).astype(bool)
    reference = load_reference()
    combined = pd.concat([frame, reference], ignore_index=True)

    source_inventory = pd.DataFrame([
        {'source': 'S1 complete+EC', 'records': int(len(s1)),
         'clusters_70_mode1': int(s1[cluster_column].nunique()),
         'note': '238 EC NMNAT-only + 318 EC dual; 44 clusters'},
        {'source': 'S1 EC NMNAT-only', 'records': int(len(s1_ec_nmnat)),
         'clusters_70_mode1': int(s1_ec_nmnat[cluster_column].nunique()),
         'note': 'used for target-aware alignment/hypotheses'},
        {'source': 'S2 full-context T80', 'records': int(len(t80)),
         'clusters_70_mode1': int(t80[cluster_column].nunique()),
         'note': '63 records/19 clusters; 6 overlap S1, not independent'},
        {'source': 'S2-only', 'records': int(len(s2_only)),
         'clusters_70_mode1': int(s2_only[cluster_column].nunique()),
         'note': 'T3 text/unknown full-context records outside S1'},
        {'source': 'S4 reviewed positives', 'records': int(len(reference)),
         'clusters_70_mode1': int(reference.cluster.nunique()),
         'note': '61 domains/7 clusters; cross-check only; classifier unvalidated'},
        {'source': 'S5 NatV training+validation', 'records': 1871,
         'clusters_70_mode1': None, 'note': '1,644 training + 227 validation; no test'},
        {'source': 'T50 sensitivity', 'records': int(len(t50)),
         'clusters_70_mode1': int(t50[cluster_column].nunique()),
         'note': '401/71; sensitivity only'},
    ])
    source_inventory.to_csv(RAW / 'source_inventory.csv', index=False)

    reps = choose_representatives(combined)
    reps.to_csv(RAW / 'cluster_representatives.csv', index=False)

    candidate_rows = combined[combined.tier.isin(['T1', 'T2', 'T3', 'T5'])].copy()
    align_rows = []
    variant_rows = []
    for item in candidate_rows.to_dict('records'):
        if not item.get('full_sequence') and not item.get('domain_sequence'):
            continue
        query = item.get('full_sequence') or item.get('domain_sequence')
        nmnat = align_map(WT_DOMAIN, query, DOMAIN_START, count_optimal=True)
        full = None
        if item.get('full_sequence'):
            full = align_map(WT, query, 1, count_optimal=False)
        mapped = nmnat['mapped']
        anchors = anchor_fields(mapped)
        domain_length = item.get('domain_length') or len(query)
        target_coverage = nmnat['target_span'] / len(WT_DOMAIN)
        query_coverage = nmnat['query_span'] / len(query)
        alignment_row = {
            'record_id': item['record_id'], 'seq_hash': item.get('seq_hash'),
            'tier': item['tier'], 'source_sets': item['source_sets'],
            'cluster': item['cluster'], 'full_context': item['full_context'],
            'query_length': len(query),
            'n_optimal_alignments': nmnat['n_optimal'],
            'target_start': min(mapped) if mapped else None,
            'target_end': max(mapped) if mapped else None,
            'target_span': nmnat['target_span'],
            'target_coverage': target_coverage,
            'aligned_columns': nmnat['aligned_columns'],
            'identity': nmnat['identity'],
            'query_start': nmnat['query_start'], 'query_end': nmnat['query_end'],
            'query_coverage': query_coverage,
            'full_target_span': full['target_span'] if full else None,
            'full_identity': full['identity'] if full else None,
            **anchors, 'anchor_window': window_fields(mapped),
        }
        align_rows.append(alignment_row)
        for position, wt_residue, alt_residue in variant_set(mapped, WT_DOMAIN, DOMAIN_START):
            variant_rows.append({
                'record_id': item['record_id'], 'cluster': item['cluster'],
                'tier': item['tier'], 'position': position,
                'wt_residue': wt_residue, 'alt_residue': alt_residue,
            })
    align_frame = pd.DataFrame(align_rows)
    align_frame.to_csv(RAW / 'target_aware_alignment.csv', index=False)
    align_quality = align_frame.set_index('record_id')[
        ['target_coverage', 'identity', 'n_optimal_alignments']].to_dict('index')
    variants = pd.DataFrame(variant_rows)
    record_sets = defaultdict(set)
    record_cluster = {}
    for item in variants.itertuples():
        record_sets[item.record_id].add((item.position, item.alt_residue))
        record_cluster[item.record_id] = item.cluster

    # S5 family baseline: map every training/validation NMNAT CDD domain to WT.
    samples = pd.read_csv(SAMPLES)
    if set(samples.training_role_v2) - {'training', 'validation'}:
        raise RuntimeError('S5 samples contain a non-training/validation role')
    samples = samples[samples.training_role_v2.isin(['training', 'validation'])].copy()
    s5_counts = defaultdict(Counter)
    s5_variant_counts = Counter()
    s5_sets = []
    s5_mapped_positions = []
    s5_queries = []
    for item in samples.itertuples():
        start, end = int(item.nmnat_cdd_from), int(item.nmnat_cdd_to)
        query = item.sequence[start - 1:end]
        s5_queries.append(query)
        mapping = align_map(WT_DOMAIN, query, DOMAIN_START, count_optimal=False)
        values = variant_set(mapping['mapped'], WT_DOMAIN, DOMAIN_START)
        s5_sets.append(values)
        s5_mapped_positions.append(set(mapping['mapped']))
        for position, residue in mapping['mapped'].items():
            if DOMAIN_START <= position <= DOMAIN_END:
                s5_counts[position][residue] += 1
        for value in values:
            s5_variant_counts[value] += 1
    s5_total = {position: sum(counter.values()) for position, counter in s5_counts.items()}
    s5_table = []
    for position in sorted(s5_counts):
        for residue, count in s5_counts[position].most_common():
            s5_table.append({'position': position, 'residue': residue, 'count': count,
                             'total': s5_total[position], 'frequency': count / s5_total[position]})
    pd.DataFrame(s5_table).to_csv(RAW / 's5_position_frequencies.csv', index=False)

    qvae_single, qvae_pairs = load_qvae()
    anchor_table = pd.read_csv(ANCHOR_AUDIT)
    anchor_distances = {}
    for _, item in anchor_table[anchor_table.domain == 'NMNAT'].iterrows():
        anchor_distances[int(item['map_wt_position'])] = {
            'distance': item['designated_min_ligand_distance_A'],
            'within_4_5': bool(item['2QJO_within_4.5A']),
            'within_8': bool(item['2QJO_within_8A']),
        }

    support = defaultdict(lambda: {'records': set(), 'clusters': set(), 'tiers': set()})
    for item in variants.itertuples():
        key = (item.position, item.wt_residue, item.alt_residue)
        support[key]['records'].add(item.record_id)
        support[key]['clusters'].add(item.cluster)
        support[key]['tiers'].add(item.tier)
    support_rows = []
    for (position, wt_residue, alt_residue), values in support.items():
        clusters = values['clusters']
        tiers = values['tiers']
        old_clusters = {cluster for cluster in clusters if not str(cluster).startswith('ref:')}
        t5_clusters = clusters - old_clusters
        old_records = [record_id for record_id in values['records']
                       if not str(record_id).startswith('ref:')]
        t5_records = [record_id for record_id in values['records']
                      if str(record_id).startswith('ref:')]
        anchor_distance = min(abs(position - anchor) for anchor in ANCHORS)
        nearest_anchor = min(ANCHORS, key=lambda anchor: abs(position - anchor))
        anchor_info = anchor_distances.get(nearest_anchor, {})
        qualities = [align_quality[record_id] for record_id in old_records
                     if record_id in align_quality]
        t5_qualities = [align_quality[record_id] for record_id in t5_records
                        if record_id in align_quality]
        support_rows.append({
            'position': position, 'wt_residue': wt_residue, 'alt_residue': alt_residue,
            'n_records': len(old_records), 'n_clusters': len(old_clusters),
            'old_clusters': ';'.join(sorted(map(str, old_clusters))),
            'clusters': ';'.join(sorted(map(str, clusters))),
            'tiers': ';'.join(sorted(tiers)),
            't5_clusters': len(t5_clusters),
            't5_records': len(t5_records),
            'support_target_coverage_median': float(np.median([q['target_coverage'] for q in qualities])) if qualities else None,
            'support_identity_median': float(np.median([q['identity'] for q in qualities if q['identity'] is not None])) if qualities else None,
            'support_n_optimal_median': float(np.median([q['n_optimal_alignments'] for q in qualities if q['n_optimal_alignments'] is not None])) if qualities else None,
            't5_support_target_coverage_median': float(np.median([q['target_coverage'] for q in t5_qualities])) if t5_qualities else None,
            't5_support_identity_median': float(np.median([q['identity'] for q in t5_qualities if q['identity'] is not None])) if t5_qualities else None,
            'conservative': conservative_change(wt_residue, alt_residue),
            's5_alt_count': s5_variant_counts.get((position, wt_residue, alt_residue), 0),
            's5_total': s5_total.get(position, 0),
            's5_frequency': (s5_variant_counts.get((position, wt_residue, alt_residue), 0) /
                             s5_total.get(position, 1)),
            'anchor_distance': anchor_distance, 'nearest_anchor': nearest_anchor,
            'nearest_anchor_template_distance_A': anchor_info.get('distance'),
            'nearest_anchor_first_shell_2QJO': anchor_info.get('within_4_5'),
            'nearest_anchor_pocket_2QJO': anchor_info.get('within_8'),
            'in_window': anchor_distance <= WINDOW,
            'region': region_of(position),
            'qvae_rep_match': (position, alt_residue) in qvae_single,
        })
    support_frame = pd.DataFrame(support_rows)
    support_frame.to_csv(RAW / 'anchor_neighbor_variation.csv', index=False)

    def eligible(row):
        if row.n_clusters < 2 or not row.in_window or row.region not in {'N', 'C'}:
            return False
        if not (set(row.tiers.split(';')) & {'T1', 'T2', 'T3'}):
            return False
        if row.s5_frequency > 0.05:
            return False
        if not conservative_change(row.wt_residue, row.alt_residue):
            return False
        if row.position in (14, 17, 138) and len(set(row.tiers.split(';'))) < 2:
            return False
        if row.position == 135 and row.alt_residue not in {'K', 'S', 'A'}:
            return False
        return True

    eligible_frame = support_frame[support_frame.apply(eligible, axis=1)].copy()
    eligible_frame['tier_score'] = eligible_frame.tiers.map(
        lambda value: 3 if 'T1' in value else 2 if 'T2' in value else 1 if 'T3' in value else 0)
    eligible_frame = eligible_frame.sort_values(
        ['tier_score', 'conservative', 'n_clusters', 'anchor_distance', 's5_frequency', 'position'],
        ascending=[False, False, False, True, True, True])

    hypotheses = []
    selected_positions = []
    selected_rows = []
    for region in ('N', 'C'):
        group = eligible_frame[eligible_frame.region == region]
        count = 0
        for row in group.itertuples():
            if count >= 3:
                break
            if any(row.region == prior_region and abs(row.position - prior_position) < 3
                   for prior_position, prior_region in selected_positions):
                continue
            selected_positions.append((row.position, row.region))
            selected_rows.append(row)
            count += 1
            hypotheses.append({
                'variant_id': f'H{len(hypotheses)+1:02d}',
                'substitutions': f'{row.wt_residue}{row.position}{row.alt_residue}',
                'positions': str(row.position), 'evidence_tiers': row.tiers,
                'n_clusters': int(row.n_clusters), 'n_records': int(row.n_records),
                't5_clusters': int(row.t5_clusters),
                'clusters': row.clusters,
                'support_target_coverage_median': row.support_target_coverage_median,
                'support_identity_median': row.support_identity_median,
                'support_n_optimal_median': row.support_n_optimal_median,
                'conservative': bool(row.conservative),
                'design_risk': ('conservative' if row.conservative else
                                'nonconservative side-chain change; higher local structural risk'),
                's5_frequency': float(row.s5_frequency),
                'anchor_distance': int(row.anchor_distance),
                'nearest_anchor': int(row.nearest_anchor),
                'nearest_anchor_template_distance_A': row.nearest_anchor_template_distance_A,
                'qvae_rep_match': bool(row.qvae_rep_match),
                'pair_clusters': None, 'pair_t5_clusters': None, 'pair_novel': None,
                'structural_note': structural_note(row),
                'unknown': 'activity, substrate specificity, kcat/Km, net NAD+',
            })

    # Pairs must cross the N- and C-terminal anchor neighborhoods and be observed
    # together in at least two independent clusters.
    if len(hypotheses) < 8:
        singles = [row._asdict() for row in selected_rows]
        pair_rows = []
        for i in range(len(singles)):
            for j in range(i + 1, len(singles)):
                a, b = singles[i], singles[j]
                if a['region'] == b['region'] or not (a['conservative'] and b['conservative']):
                    continue
                pair = ((int(a['position']), a['alt_residue']),
                        (int(b['position']), b['alt_residue']))
                pair_clusters = {
                    record_cluster[record_id]
                    for record_id, values in record_sets.items()
                    if pair[0] in values and pair[1] in values
                }
                old_pair_clusters = {cluster for cluster in pair_clusters
                                     if not str(cluster).startswith('ref:')}
                t5_pair_clusters = pair_clusters - old_pair_clusters
                if len(old_pair_clusters) < 2:
                    continue
                a_triple = (int(a['position']), a['wt_residue'], a['alt_residue'])
                b_triple = (int(b['position']), b['wt_residue'], b['alt_residue'])
                if any(a_triple in values and b_triple in values for values in s5_sets):
                    continue
                if frozenset(((int(a['position']), a['alt_residue']),
                              (int(b['position']), b['alt_residue']))) in qvae_pairs:
                    continue
                pair_rows.append((len(old_pair_clusters), len(t5_pair_clusters), a, b, pair_clusters))
        pair_rows.sort(key=lambda value: (-value[0], value[1], value[2]['anchor_distance'] + value[3]['anchor_distance']))
        used_components = set()
        for n_clusters, n_t5, a, b, pair_clusters in pair_rows:
            if len(hypotheses) >= 8:
                break
            components = {(int(a['position']), a['alt_residue']),
                          (int(b['position']), b['alt_residue'])}
            if components & used_components:
                continue
            used_components |= components
            hypotheses.append({
                'variant_id': f'H{len(hypotheses)+1:02d}',
                'substitutions': f"{a['wt_residue']}{a['position']}{a['alt_residue']}+{b['wt_residue']}{b['position']}{b['alt_residue']}",
                'positions': f"{a['position']};{b['position']}", 'evidence_tiers': ';'.join(sorted(set(a['tiers'].split(';')) | set(b['tiers'].split(';')))),
                'n_clusters': n_clusters, 'n_records': None,
                't5_clusters': n_t5,
                'clusters': ';'.join(sorted(map(str, pair_clusters))),
                'support_target_coverage_median': min(a['support_target_coverage_median'], b['support_target_coverage_median']),
                'support_identity_median': min(a['support_identity_median'], b['support_identity_median']),
                'support_n_optimal_median': max(a['support_n_optimal_median'], b['support_n_optimal_median']),
                'conservative': bool(a['conservative'] and b['conservative']),
                'design_risk': 'conservative pair; combined effects still UNKNOWN',
                's5_frequency': max(float(a['s5_frequency']), float(b['s5_frequency'])),
                'anchor_distance': max(int(a['anchor_distance']), int(b['anchor_distance'])),
                'nearest_anchor': None, 'nearest_anchor_template_distance_A': None,
                'qvae_rep_match': False,
                'pair_clusters': n_clusters, 'pair_t5_clusters': None, 'pair_novel': True,
                'structural_note': structural_note(a) + ' | ' + structural_note(b),
                'unknown': 'activity, substrate specificity, kcat/Km, net NAD+; pair co-occurrence is sequence evidence only',
            })

    # Limited stability check: for the selected positions only, require the
    # mapped residue to be identical across all (capped) optimal local alignments.
    selected_positions = sorted({int(position) for item in hypotheses
                                 for position in str(item['positions']).split(';')})
    record_queries = {}
    record_tier = {}
    for item in candidate_rows.to_dict('records'):
        query = item.get('full_sequence') or item.get('domain_sequence')
        if query:
            record_queries[item['record_id']] = query
            record_tier[item['record_id']] = item['tier']
    stable_record_sets = {}
    stable_record_nopt = {}
    for record_id, query in record_queries.items():
        stable, values, n_optimal = stable_positions(query, selected_positions)
        stable_set = set()
        for position in selected_positions:
            if stable[position] and len(values[position]) == 1:
                residue = next(iter(values[position]))
                if residue is not None and residue != WT_DOMAIN[position - DOMAIN_START]:
                    stable_set.add((position, WT_DOMAIN[position - DOMAIN_START], residue))
        stable_record_sets[record_id] = stable_set
        stable_record_nopt[record_id] = n_optimal

    s5_stable_sets = []
    s5_stable_mapped = []
    for query in s5_queries:
        stable, values, _n_optimal = stable_positions(query, selected_positions)
        mapped_positions = set()
        stable_set = set()
        for position in selected_positions:
            if stable[position] and len(values[position]) == 1:
                residue = next(iter(values[position]))
                if residue is not None:
                    mapped_positions.add(position)
                    if residue != WT_DOMAIN[position - DOMAIN_START]:
                        stable_set.add((position, WT_DOMAIN[position - DOMAIN_START], residue))
        s5_stable_sets.append(stable_set)
        s5_stable_mapped.append(mapped_positions)

    retained = []
    for item in hypotheses:
        triples = [(int(position), wt, alt) for wt, position, alt in
                   re.findall(r'([A-Z])(\d+)([A-Z])', item['substitutions'])]
        old_ids = [record_id for record_id, values in stable_record_sets.items()
                   if not record_id.startswith('ref:') and all(triple in values for triple in triples)]
        t5_ids = [record_id for record_id, values in stable_record_sets.items()
                  if record_id.startswith('ref:') and all(triple in values for triple in triples)]
        old_clusters = {record_cluster[record_id] for record_id in old_ids}
        t5_clusters = {record_cluster[record_id] for record_id in t5_ids}
        item['raw_n_clusters'] = item['n_clusters']
        item['raw_n_records'] = item['n_records']
        item['raw_evidence_tiers'] = item['evidence_tiers']
        item['raw_clusters'] = item['clusters']
        item['n_clusters'] = len(old_clusters)
        item['n_records'] = len(old_ids)
        item['t5_clusters'] = len(t5_clusters)
        stable_tiers = {record_tier[record_id] for record_id in old_ids + t5_ids
                        if record_id in record_tier}
        item['evidence_tiers'] = ';'.join(sorted(stable_tiers))
        item['clusters'] = ';'.join(sorted(map(str, old_clusters | t5_clusters)))
        item['stable_support_fraction'] = (len(old_ids) / item['raw_n_records']
                                           if item['raw_n_records'] else None)
        qualities = [align_quality[record_id] for record_id in old_ids if record_id in align_quality]
        item['support_target_coverage_median'] = (
            float(np.median([q['target_coverage'] for q in qualities])) if qualities else None)
        item['support_identity_median'] = (
            float(np.median([q['identity'] for q in qualities if q['identity'] is not None]))
            if qualities else None)
        item['support_n_optimal_median'] = (
            float(np.median([stable_record_nopt[record_id] for record_id in old_ids]))
            if old_ids else None)
        if item['pair_clusters'] is not None:
            item['pair_clusters'] = len(old_clusters)
            first, second = triples
            pair_count = sum(1 for values in s5_stable_sets
                             if first in values and second in values)
            pair_total = sum(1 for positions in s5_stable_mapped
                             if first[0] in positions and second[0] in positions)
            item['pair_s5_count'] = pair_count
            item['pair_s5_total'] = pair_total
            item['pair_s5_frequency'] = pair_count / pair_total if pair_total else None
            item['s5_frequency'] = max(
                (sum(1 for values in s5_stable_sets if triple in values) /
                 max(sum(1 for positions in s5_stable_mapped if triple[0] in positions), 1)
                 for triple in triples))
            item['s5_frequency_basis'] = 'max single-position stable frequency; pair frequency separate'
        else:
            triple = triples[0]
            count = sum(1 for values in s5_stable_sets if triple in values)
            total = sum(1 for positions in s5_stable_mapped if triple[0] in positions)
            item['s5_frequency'] = count / total if total else None
            item['s5_frequency_basis'] = 'single-position stable mapping'
            item['pair_s5_count'] = None
            item['pair_s5_total'] = None
            item['pair_s5_frequency'] = None
        context_supported = bool(stable_tiers & {'T1', 'T2', 'T5'})
        frequency_ok = item['s5_frequency'] is not None and item['s5_frequency'] <= 0.05
        pair_ok = True
        if item['pair_clusters'] is not None:
            pair_ok = (item['pair_s5_count'] == 0 and item['pair_s5_total'] > 0)
        if item['n_clusters'] >= 2 and context_supported and frequency_ok and pair_ok:
            retained.append(item)
    hypotheses = retained

    hypothesis_frame = pd.DataFrame(hypotheses)
    hypothesis_frame.to_csv(RAW / 'hypothesis_table.csv', index=False)

    supported = [h for h in hypotheses if set(h['evidence_tiers'].split(';')) & {'T1', 'T2', 'T5'}]
    gate_pass = len(hypotheses) >= 2 and len(supported) == len(hypotheses)
    reasons = []
    if len(hypotheses) < 2:
        reasons.append('fewer than 2 hypotheses with >=2 stable old-only clusters')
    if len(supported) != len(hypotheses):
        reasons.append('some hypotheses lack T1/T2/T5 context or reviewed evidence')
    if not gate_pass and not reasons:
        reasons.append('gate criteria not met')
    gate = {
        'status': 'PASS' if gate_pass else 'STOP',
        'n_hypotheses': len(hypotheses),
        'n_hypotheses_with_context_or_reviewed_evidence': len(supported),
        'evidence_by_hypothesis': [
            {key: item.get(key) for key in
             ['variant_id', 'substitutions', 'evidence_tiers', 'n_clusters', 'n_records',
              't5_clusters', 'stable_support_fraction', 's5_frequency', 'pair_s5_frequency']}
            for item in hypotheses
        ],
        'stable_mapping': {
            'selected_positions': selected_positions,
            'max_optimal_alignments': MAX_OPTIMAL,
            'note': 'positions unstable across capped optimal alignments are excluded from support',
        },
        'source_counts': {
            'S1_complete_ec': int(len(s1)),
            'S1_clusters': int(s1[cluster_column].nunique()),
            'S1_ec_nmnat_only': int(len(s1_ec_nmnat)),
            'S2_t80': int(len(t80)),
            'S2_clusters': int(t80[cluster_column].nunique()),
            'S2_only': int(len(s2_only)),
            'S4_positive': int(len(reference)),
            'S4_clusters': int(reference.cluster.nunique()),
        },
        'eligible_variants': int(len(eligible_frame)),
        'reasons': reasons,
        'note': ('Hypotheses require >=2 different old-only 70% sequence clusters and a position '
                 f'within {WINDOW} residues of H14/H17/T135/R138; S1/S2 overlap is not counted twice. '
                 'Sequence clusters are not independent evolutionary events. T5 is a reviewed-domain '
                 'cross-check; its ref prefix does not prove independence from old sequences, and it '
                 'does not establish function.'),
    }
    write_json(RAW / 'gate_A.json', gate)

    inputs = {
        'audit': {'path': str(AUDIT), 'sha256': sha256_file(AUDIT)},
        'old_full': {'path': str(OLD_FULL), 'sha256': sha256_file(OLD_FULL)},
        't80': {'path': str(T80), 'sha256': sha256_file(T80)},
        't50': {'path': str(T50), 'sha256': sha256_file(T50)},
        'anchor_audit': {'path': str(ANCHOR_AUDIT), 'sha256': sha256_file(ANCHOR_AUDIT)},
        'samples': {'path': str(SAMPLES), 'sha256': sha256_file(SAMPLES)},
        'qvae_representatives': {'path': str(QV), 'sha256': sha256_file(QV)},
        'reference_manifest': {'path': str(REF_MANIFEST), 'sha256': sha256_file(REF_MANIFEST)},
        'reference_fasta': {'path': str(REF_FASTA), 'sha256': sha256_file(REF_FASTA)},
        'reference_clusters': {'path': str(REF_CLUSTERS), 'sha256': sha256_file(REF_CLUSTERS)},
        'script': {'path': str(Path(__file__)), 'sha256': sha256_file(Path(__file__))},
        'wt_sequence_sha256': hashlib.sha256(WT.encode()).hexdigest(),
    }
    write_json(args.run / 'inputs.json', inputs)
    status = {
        'status': 'COMPLETED' if gate_pass else 'NEGATIVE',
        'gate': gate['status'],
        'n_hypotheses': len(hypotheses),
        'n_s1': int(len(s1)), 'n_s2': int(len(t80)), 'n_s4': int(len(reference)),
        'n_s5': int(len(samples)),
        'runtime_seconds': round(time.monotonic() - started, 2),
        'outputs': [str(RAW / name) for name in
                    ['source_inventory.csv', 'cluster_representatives.csv',
                     'target_aware_alignment.csv', 'anchor_neighbor_variation.csv',
                     's5_position_frequencies.csv', 'hypothesis_table.csv', 'gate_A.json']],
        'note': 'CPU only; no test/sealed/conflict/excluded sequence loaded; no MMseqs/download/folding.',
    }
    write_json(args.run / 'status.json', status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


def structural_note(row):
    def value(name):
        return row[name] if isinstance(row, dict) else getattr(row, name)

    position = value('position')
    if position in ANCHORS:
        return (f'anchor {position}; 2QJO NAD proximity recorded; side-chain change is a local '
                'hypothesis, not a substrate-geometry or activity claim')
    return (f'{value("anchor_distance")} aa from nearest anchor {value("nearest_anchor")}; '
            'anchor-neighborhood proxy only; no per-residue ligand distance or ATP/NMN geometry')


if __name__ == '__main__':
    main()
