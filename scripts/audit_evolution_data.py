#!/usr/bin/env python
"""Audit Stage1-3 PF01467 records against current NatV roles.

Read-only with respect to nad_qvae. Writes only under current data/results and
data/interim. Does not train, fold, or download.

Executed 2026-10-22: this version loaded all 13,972 Stage3 records and sliced
domains for all roles, including current test and sealed sequences. The executed
source is preserved in data/results/natv_evolution_data_audit/raw/. See
experiments/natv_evolution_data_audit/AUDIT.md for the scope correction. Do not
reuse this as a test-safe template; audit_wt_full_context.py implements the
whitelist loading used for the supplemental analysis.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Align import PairwiseAligner, substitution_matrices

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from natvqvae.sequence import wt_sequence  # noqa: E402

MMSEQS = Path('/home/yhshy/miniconda3/envs/nad_data/bin/mmseqs')
RAW = ROOT / 'data/results/natv_evolution_data_audit/raw'
INTERIM = ROOT / 'data/interim/natv_evolution_data_audit'
OLD_REPORT = Path('/home/yhshy/git_files/nad_qvae/experiments/run_reports/pretraining_scientific_validation_20260909')
OLD_STAGE3 = OLD_REPORT / 'stage3_accepted_sequences.fasta'
OLD_PROFILE = OLD_REPORT / 'domain_profile_manifest_v4.csv'
OLD_MANIFEST = OLD_REPORT / 'corpus_boundary_manifest.jsonl'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def run(cmd, cwd=None):
    log = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if log.returncode:
        raise RuntimeError('command failed: ' + ' '.join(map(str, cmd)) + '\n' + log.stderr[-3000:])
    return log.stdout


def functional_flags(row):
    ec = tuple(row.get('uniprot_ec') or ())
    evidence = row.get('evidence') or {}
    return {
        'ec_2_7_7_1': '2.7.7.1' in ec,
        'ec_2_7_7_18': '2.7.7.18' in ec,
        'ec_nmnat_only': '2.7.7.1' in ec and '2.7.7.18' not in ec,
        'ec_dual': '2.7.7.1' in ec and '2.7.7.18' in ec,
        'ec_namn_only': '2.7.7.18' in ec and '2.7.7.1' not in ec,
        'nadr': bool(evidence.get('nadr_multifunction_annotation')) or str(row.get('uniprot_gene', '')).lower() == 'nadr',
        'text_nmnat': bool(evidence.get('name_nmnat_or_nadm')),
        'partial': bool(evidence.get('partial_or_fragment')),
        'annotation_conflict': bool(evidence.get('annotation_conflict')),
        'header_disagreement': bool(evidence.get('header_annotation_disagreement')),
        'unmapped': row.get('stage1_final_status') == 'UNMAPPED',
        'has_uniprot': bool(row.get('uniprot_accession')),
    }


def primary_layer(flags):
    if flags['ec_dual']:
        return 'ec_dual'
    if flags['ec_nmnat_only']:
        return 'ec_nmnat_only'
    if flags['ec_namn_only']:
        return 'ec_namn_only'
    if flags['nadr']:
        return 'nadr_multifunctional'
    if flags['text_nmnat']:
        return 'text_only'
    return 'pf01467_only_unknown'


def parse_intervals(value):
    if not isinstance(value, str) or not value.strip():
        return []
    result = []
    for part in value.replace(';', ',').split(','):
        part = part.strip()
        if not part:
            continue
        low, high = part.split('-')
        result.append((int(low), int(high)))
    return result


def make_fasta(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as handle:
        for key, sequence in records:
            handle.write(f'>{key}\n{sequence}\n')


def parse_clusters(path):
    clusters = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rep, member = line.split('\t')[:2]
        clusters.setdefault(rep, []).append(member)
    return clusters


def cluster_summary(clusters, old_ids):
    old_ids = set(old_ids)
    result = {'clusters': len(clusters), 'old_only': 0, 'both': 0, 'current_only': 0,
              'old_domains_in_old_only': 0, 'old_only_sizes': [], 'both_sizes': []}
    for rep, members in clusters.items():
        old = [member for member in members if member in old_ids]
        current = [member for member in members if member.startswith('natv:')]
        if old and current:
            result['both'] += 1
            result['both_sizes'].append(len(members))
        elif old:
            result['old_only'] += 1
            result['old_domains_in_old_only'] += len(old)
            result['old_only_sizes'].append(len(old))
        else:
            result['current_only'] += 1
    for key in ('old_only_sizes', 'both_sizes'):
        values = result.pop(key)
        result[key] = {'n': len(values), 'median': float(np.median(values)) if values else None,
                       'max': int(max(values)) if values else None}
    return result


def best_hits(nn_path, qlen, tlen):
    best = {}
    for line in Path(nn_path).read_text().splitlines():
        parts = line.split('\t')
        if len(parts) < 9:
            continue
        query, target = parts[0], parts[1]
        fident = float(parts[2])
        qstart, qend, tstart, tend = map(int, parts[4:8])
        qcov = (qend - qstart + 1) / max(qlen.get(query, 1), 1)
        tcov = (tend - tstart + 1) / max(tlen.get(target, 1), 1)
        value = (fident, qcov, tcov, target)
        if query not in best or value[:3] > best[query][:3]:
            best[query] = value
    return best


def anchor_mapping(old_sequence, wt_domain, anchors, domain_start):
    aligner = PairwiseAligner(mode='global', substitution_matrix=substitution_matrices.load('BLOSUM62'))
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    try:
        alignment = aligner.align(wt_domain, old_sequence)[0]
    except Exception:
        return {anchor: None for anchor in anchors}
    mapped = {anchor: None for anchor in anchors}
    target_blocks = alignment.aligned[0]
    query_blocks = alignment.aligned[1]
    for anchor in anchors:
        target_index = anchor - domain_start
        for (t0, t1), (q0, q1) in zip(target_blocks, query_blocks):
            if t0 <= target_index < t1:
                query_index = int(q0 + (target_index - t0))
                if 0 <= query_index < len(old_sequence):
                    mapped[anchor] = str(old_sequence[query_index])
                break
    return mapped


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    INTERIM.mkdir(parents=True, exist_ok=True)
    if not MMSEQS.exists():
        raise FileNotFoundError(MMSEQS)
    version = run([str(MMSEQS), 'version']).strip()

    stage3 = {record.id: str(record.seq) for record in SeqIO.parse(str(OLD_STAGE3), 'fasta')}
    profile = pd.read_csv(OLD_PROFILE)
    profile = profile[profile.seq_hash.isin(stage3)].copy()
    manifest = {}
    for line in OLD_MANIFEST.read_text().splitlines():
        item = json.loads(line)
        if item.get('stage3_accepted'):
            manifest[item['seq_hash']] = item
    roles = pd.read_csv(ROOT / 'data/processed/roles.csv')
    role_by_id = roles.set_index('sequence_id').training_role_v2.to_dict()
    samples = pd.read_csv(ROOT / 'data/processed/samples.csv')

    rows = []
    for item in profile.itertuples():
        sequence = stage3[item.seq_hash]
        meta = manifest.get(item.seq_hash, {})
        flags = functional_flags(meta)
        intervals = parse_intervals(item.nmnat_intervals) if isinstance(item.nmnat_intervals, str) else []
        start = item.pf01467_best_alignment_start
        end = item.pf01467_best_alignment_end
        domain = ''
        if pd.notna(start) and pd.notna(end):
            start, end = int(start), int(end)
            if 1 <= start <= end <= len(sequence):
                domain = sequence[start-1:end]
        rows.append({
            'seq_hash': item.seq_hash, 'length': len(sequence), 'architecture': item.architecture,
            'current_role': role_by_id.get(item.seq_hash, 'stage3_only'),
            'stage1_final_status': meta.get('stage1_final_status'),
            'uniprot_accession': meta.get('uniprot_accession'),
            'uniprot_gene': meta.get('uniprot_gene'),
            'uniprot_ec': json.dumps(meta.get('uniprot_ec', []), ensure_ascii=False),
            'profile_coverage': item.pf01467_best_profile_coverage,
            'query_coverage': item.pf01467_best_query_coverage,
            'nmnat_hit_count': int(item.nmnat_hit_count),
            'nmnat_intervals': item.nmnat_intervals if isinstance(item.nmnat_intervals, str) else '',
            'domain_start': int(start) if pd.notna(start) else None,
            'domain_end': int(end) if pd.notna(end) else None,
            'domain_length': len(domain),
            'partial_flag': flags['partial'],
            'experimental_evidence': 'UNKNOWN',
            **flags, 'primary_layer': primary_layer(flags),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(RAW / 'stage3_audit.csv', index=False)

    # Current visible NatV NMNAT domains; test sequences are deliberately not read.
    current_records = []
    current_lengths = {}
    for row in samples.itertuples():
        start, end = int(row.nmnat_cdd_from), int(row.nmnat_cdd_to)
        domain = row.sequence[start-1:end]
        key = f'natv:{row.sequence_id}'
        current_records.append((key, domain))
        current_lengths[key] = len(domain)
    make_fasta(RAW / 'current_natv_nmnat_domains.fasta', current_records)

    only = frame[(frame.architecture == 'nmnat_only') & frame.domain_length.gt(0)].copy()
    old_records = [(f'old:{row.seq_hash}',
                    stage3[row.seq_hash][int(row.domain_start)-1:int(row.domain_end)])
                   for row in only.itertuples()]
    make_fasta(RAW / 'pf01467_only_domains.fasta', old_records)
    make_fasta(RAW / 'combined_nmnat_domains.fasta', old_records + current_records)
    old_lengths = {key: len(sequence) for key, sequence in old_records}

    thresholds = [0.7, 0.5, 0.3]
    cluster_results = {}
    cluster_maps = {}
    for threshold in thresholds:
        for mode in (0, 1):
            name = f'{threshold:.1f}_mode{mode}'
            work = INTERIM / 'mmseqs' / name
            if work.exists():
                shutil.rmtree(work)
            work.mkdir(parents=True)
            prefix = work / 'cluster'
            cmd = [str(MMSEQS), 'easy-cluster', str(RAW / 'combined_nmnat_domains.fasta'),
                   str(prefix), str(work / 'tmp'), '--min-seq-id', str(threshold),
                   '--cov-mode', '0', '-c', '0.8', '--cluster-mode', str(mode), '--threads', '8']
            run(cmd, cwd=work)
            clusters = parse_clusters(str(prefix) + '_cluster.tsv')
            cluster_results[name] = cluster_summary(clusters, [key for key, _ in old_records])
            mapping = {}
            for rep, members in clusters.items():
                for member in members:
                    mapping[member] = rep
            cluster_maps[name] = mapping

    nn_work = INTERIM / 'mmseqs' / 'nearest'
    if nn_work.exists():
        shutil.rmtree(nn_work)
    nn_work.mkdir(parents=True)
    nn_path = nn_work / 'hits.tsv'
    run([str(MMSEQS), 'easy-search', str(RAW / 'pf01467_only_domains.fasta'),
         str(RAW / 'current_natv_nmnat_domains.fasta'), str(nn_path), str(nn_work / 'tmp'),
         '--min-seq-id', '0.3', '--cov-mode', '2', '-c', '0.8', '--threads', '8',
         '--format-output', 'query,target,fident,alnlen,qstart,qend,tstart,tend,evalue'],
        cwd=nn_work)
    best = best_hits(str(nn_path), old_lengths, current_lengths)
    nearest_rows = []
    for key, _ in old_records:
        if key in best:
            fident, qcov, tcov, target = best[key]
            nearest_rows.append({'old_id': key, 'target': target, 'fident': fident,
                                 'query_coverage': qcov, 'target_coverage': tcov})
        else:
            nearest_rows.append({'old_id': key, 'target': None, 'fident': None,
                                 'query_coverage': None, 'target_coverage': None})
    nearest = pd.DataFrame(nearest_rows)
    nearest.to_csv(RAW / 'nearest_natv_nmnat.csv', index=False)

    for name, mapping in cluster_maps.items():
        only[name + '_cluster'] = only.seq_hash.map(lambda value: mapping.get('old:' + value))
    only.to_csv(RAW / 'pf01467_only_audit.csv', index=False)

    # Anchor correspondence for representatives of old-only clusters at 70% connected components.
    rep_map = {}
    for key, members in parse_clusters(str(INTERIM / 'mmseqs' / '0.7_mode1' / 'cluster_cluster.tsv')).items():
        old_members = [m for m in members if m.startswith('old:')]
        current_members = [m for m in members if m.startswith('natv:')]
        if old_members and not current_members:
            rep_map[key] = old_members[0]
    wt = wt_sequence()
    wt_domain = wt[3:181]
    anchor_rows = []
    for rep, old_id in rep_map.items():
        seq_hash = old_id.split(':', 1)[1]
        row = only[only.seq_hash == seq_hash].iloc[0]
        mapping = anchor_mapping(stage3[seq_hash][int(row.domain_start)-1:int(row.domain_end)], wt_domain,
                                 [14, 17, 135, 138], 4)
        anchor_rows.append({'cluster_rep': rep, 'old_id': old_id, 'domain_length': row.domain_length,
                            'profile_coverage': row.profile_coverage,
                            **{f'wt_{position}': value for position, value in mapping.items()}})
    pd.DataFrame(anchor_rows).to_csv(RAW / 'anchor_mapping_old_only_reps.csv', index=False)

    summary = {
        'status': 'COMPLETE_AUDIT_ONLY',
        'mmseqs_version': version,
        'inputs': {
            'stage3_fasta': {'path': str(OLD_STAGE3), 'sha256': sha256(OLD_STAGE3)},
            'profile_manifest': {'path': str(OLD_PROFILE), 'sha256': sha256(OLD_PROFILE)},
            'corpus_manifest': {'path': str(OLD_MANIFEST), 'sha256': sha256(OLD_MANIFEST)},
            'current_roles': {'path': str(ROOT / 'data/processed/roles.csv'),
                              'sha256': sha256(ROOT / 'data/processed/roles.csv')},
            'current_samples': {'path': str(ROOT / 'data/processed/samples.csv'),
                                'sha256': sha256(ROOT / 'data/processed/samples.csv')},
        },
        'population': {'stage3_accepted': len(stage3), 'profile_rows': len(profile),
                       'pf01467_only': int(len(only))},
        'architecture': frame.architecture.value_counts().to_dict(),
        'current_roles': {f'{architecture}|{role}': int(value)
                          for (architecture, role), value in
                          frame.groupby(['architecture', 'current_role']).size().items()},
        'functional_layers': only.primary_layer.value_counts().to_dict(),
        'functional_flags': {key: int(only[key].sum()) for key in
                             ['ec_2_7_7_1', 'ec_2_7_7_18', 'ec_nmnat_only', 'ec_dual',
                              'ec_namn_only', 'nadr', 'text_nmnat', 'partial',
                              'annotation_conflict', 'header_disagreement', 'unmapped']},
        'domain_completeness': {
            'profile_coverage_median': float(only.profile_coverage.median()),
            'profile_coverage_ge_0_8': int((only.profile_coverage >= 0.8).sum()),
            'domain_length_median': float(only.domain_length.median()),
            'domain_length_ge_124': int((only.domain_length >= 124).sum()),
            'multi_hit': int((only.nmnat_hit_count > 1).sum()),
            'operational_complete': int(((only.profile_coverage >= 0.8) &
                                        (only.domain_length >= 124)).sum()),
        },
        'clusters': cluster_results,
        'nearest_natv_nmnat': {
            'query_coverage_ge_0_8': int(nearest.query_coverage.ge(0.8).sum()),
            'ge70': int(nearest.fident.ge(0.7).sum()),
            'ge50': int(nearest.fident.ge(0.5).sum()),
            'ge30': int(nearest.fident.ge(0.3).sum()),
            'target_coverage_median': float(nearest.target_coverage.median()) if nearest.target_coverage.notna().any() else None,
            'none_ge30_qcov80': int(nearest.fident.isna().sum()),
            'coverage_definition': 'query = old PF01467 domain; target = current NatV NMNAT CDD domain; qcov>=0.8, tcov reported separately',
        },
        'old_only_70_reps': len(anchor_rows),
        'note': ('PF01467 alone is architecture evidence, not NMNAT function. '
                 'No current test sequence is read. Domain extraction uses PF01467 best alignment; '
                 'query coverage is full-protein fraction, not domain completeness. '
                 'Clustering uses bidirectional coverage 0.8 for old-old homology; '
                 'cross-annotation nearest neighbor uses old-domain query coverage 0.8 and reports target coverage separately.'),
    }
    write_json(RAW / 'audit_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
