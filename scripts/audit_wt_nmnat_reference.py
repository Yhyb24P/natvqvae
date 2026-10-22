#!/usr/bin/env python
"""Fixed-reference audit of old PF01467-only domains against NatV WT NMNAT.

Reads only the already-derived 8,849 old-only domain FASTA and audit table.
Does not read the Stage3 FASTA or any current test sequence.
"""
from __future__ import annotations

import hashlib
import json
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
WORK = ROOT / 'data/interim/natv_evolution_data_audit/wt_reference'
OLD_DOMAINS = RAW / 'pf01467_only_domains.fasta'
OLD_AUDIT = RAW / 'pf01467_only_audit.csv'
WT_LENGTH = 178  # NatV WT NMNAT CDD domain 4-181


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def run(cmd):
    log = subprocess.run(cmd, text=True, capture_output=True)
    if log.returncode:
        raise RuntimeError('command failed: ' + ' '.join(map(str, cmd)) + '\n' + log.stderr[-3000:])


def read_fasta(path):
    return {record.id: str(record.seq) for record in SeqIO.parse(str(path), 'fasta')}


def parse_best(path, domain_length):
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.split('\t')
        if len(parts) < 9:
            continue
        query, _target, fident, alnlen, qstart, qend, tstart, tend, _evalue = parts
        rows.append({
            'old_id': query,
            'fident': float(fident),
            'alnlen': int(alnlen),
            'qstart': int(qstart),
            'qend': int(qend),
            'tstart': int(tstart),
            'tend': int(tend),
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame['query_coverage'] = (frame.qend - frame.qstart + 1) / frame.old_id.map(domain_length)
    frame['target_coverage'] = frame.alnlen / WT_LENGTH
    frame['missing_n'] = frame.tstart - 1
    frame['missing_c'] = WT_LENGTH - frame.tend
    frame = frame.sort_values(['fident', 'query_coverage', 'target_coverage'],
                              ascending=False).drop_duplicates('old_id')
    return frame


def anchor_mapping(old_sequence, wt_domain, anchors):
    aligner = PairwiseAligner(mode='local', substitution_matrix=substitution_matrices.load('BLOSUM62'))
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(wt_domain, old_sequence)[0]
    mapped = {anchor: None for anchor in anchors}
    for anchor in anchors:
        index = anchor - 4
        for (t0, t1), (q0, q1) in zip(alignment.aligned[0], alignment.aligned[1]):
            if t0 <= index < t1:
                query_index = int(q0 + (index - t0))
                if 0 <= query_index < len(old_sequence):
                    mapped[anchor] = str(old_sequence[query_index])
                break
    return mapped


def main():
    if not MMSEQS.exists():
        raise FileNotFoundError(MMSEQS)
    WORK.mkdir(parents=True, exist_ok=True)
    domains = read_fasta(OLD_DOMAINS)
    audit = pd.read_csv(OLD_AUDIT)
    if len(domains) != len(audit):
        raise RuntimeError(f'domain/audit size mismatch: {len(domains)} != {len(audit)}')

    wt = wt_sequence()
    wt_domain = wt[3:181]
    if len(wt_domain) != WT_LENGTH:
        raise RuntimeError(len(wt_domain))
    wt_fasta = WORK / 'wt_nmnat_domain.fasta'
    wt_fasta.write_text('>natv:WT_NMNAT_4_181\n' + wt_domain + '\n')

    search_sets = {
        'strict': ('--min-seq-id', '0.3', '--cov-mode', '2', '-c', '0.8'),
        'diagnostic': ('--min-seq-id', '0.2', '--cov-mode', '2', '-c', '0.5'),
    }
    hits = {}
    for name, options in search_sets.items():
        prefix = WORK / name
        if prefix.exists():
            for child in prefix.glob('*'):
                if child.is_dir():
                    import shutil
                    shutil.rmtree(child)
                else:
                    child.unlink()
        prefix.mkdir(parents=True, exist_ok=True)
        out = prefix / 'hits.tsv'
        run([str(MMSEQS), 'easy-search', str(OLD_DOMAINS), str(wt_fasta), str(out),
             str(prefix / 'tmp'), *options, '--threads', '8',
             '--format-output', 'query,target,fident,alnlen,qstart,qend,tstart,tend,evalue'])
        hits[name] = parse_best(out, dict(zip(audit.seq_hash.map(lambda value: f'old:{value}'),
                                              audit.domain_length)))

    strict = hits['strict'].copy()
    strict['seq_hash'] = strict.old_id.str.split(':', n=1).str[1]
    strict = strict.merge(audit[['seq_hash', 'primary_layer', 'profile_coverage', 'domain_length',
                                 'ec_nmnat_only', 'ec_dual', 'nadr', 'text_nmnat',
                                 'annotation_conflict', '0.7_mode1_cluster', 'domain_start',
                                 'domain_end']], on='seq_hash', how='left')
    strict['complete'] = (strict.profile_coverage >= 0.8) & (strict.domain_length >= 124)
    strict['ec_support'] = strict.ec_nmnat_only | strict.ec_dual
    strict.to_csv(RAW / 'nearest_wt_nmnat.csv', index=False)

    # Anchor correspondence for the PF01467-complete, EC-supported strict hits.
    anchor_rows = []
    for _, row in strict[strict.complete & strict.ec_support].iterrows():
        old_sequence = domains[f'old:{row.seq_hash}']
        mapped = anchor_mapping(old_sequence, wt_domain, [14, 17, 135, 138])
        anchor_rows.append({'seq_hash': row.seq_hash, 'cluster': row['0.7_mode1_cluster'],
                            'fident': row.fident, 'target_coverage': row.target_coverage,
                            **{f'wt_{position}': value for position, value in mapped.items()}})
    anchor_frame = pd.DataFrame(anchor_rows)
    anchor_frame.to_csv(RAW / 'wt_hit_anchor_mapping.csv', index=False)

    audit['any_ec'] = audit.ec_2_7_7_1 | audit.ec_2_7_7_18
    overlaps = {
        'any_ec': int(audit.any_ec.sum()),
        'ec_and_nadr': int((audit.any_ec & audit.nadr).sum()),
        'ec_and_text': int((audit.any_ec & audit.text_nmnat).sum()),
        'ec_and_annotation_conflict': int((audit.any_ec & audit.annotation_conflict).sum()),
        'nadr_and_text': int((audit.nadr & audit.text_nmnat).sum()),
        'nadr_and_not_text': int((audit.nadr & ~audit.text_nmnat).sum()),
    }
    cluster_frame = strict.groupby('0.7_mode1_cluster').agg(
        members=('seq_hash', 'size'),
        any_ec=('ec_support', 'sum'),
        layers=('primary_layer', lambda values: ','.join(sorted(set(values)))),
    )
    complete_ec = strict[strict.complete & strict.ec_support]
    complete_ec_clusters = complete_ec['0.7_mode1_cluster'].nunique()
    summary = {
        'status': 'SUPPLEMENT_AUDIT_ONLY',
        'inputs': {
            'old_domains': {'path': str(OLD_DOMAINS), 'sha256': sha256(OLD_DOMAINS)},
            'old_audit': {'path': str(OLD_AUDIT), 'sha256': sha256(OLD_AUDIT)},
            'wt_domain': {'start': 4, 'end': 181, 'length': WT_LENGTH,
                          'sequence_sha256': hashlib.sha256(wt_domain.encode()).hexdigest()},
        },
        'strict_search': 'min-seq-id 0.3, cov-mode 2 query coverage 0.8',
        'strict_hits': int(len(strict)),
        'identity_bands': {
            'ge30': int((strict.fident >= 0.3).sum()),
            'ge50': int((strict.fident >= 0.5).sum()),
            'ge70': int((strict.fident >= 0.7).sum()),
        },
        'target_coverage': {
            'median': float(strict.target_coverage.median()),
            'max': float(strict.target_coverage.max()),
            'ge0_8': int((strict.target_coverage >= 0.8).sum()),
        },
        'clusters_70_mode1': int(strict['0.7_mode1_cluster'].nunique()),
        'cluster_composition': {
            'any_ec': int((cluster_frame.any_ec > 0).sum()),
            'pure_text': int((cluster_frame.layers == 'text_only').sum()),
            'mixed_layers': int(cluster_frame.layers.str.contains(',').sum()),
        },
        'complete_ec_hits': int(len(complete_ec)),
        'complete_ec_clusters': int(complete_ec_clusters),
        'complete_ec_identity_median': float(complete_ec.fident.median()) if len(complete_ec) else None,
        'complete_ec_target_coverage_median': float(complete_ec.target_coverage.median()) if len(complete_ec) else None,
        'complete_ec_missing_c_median': float(complete_ec.missing_c.median()) if len(complete_ec) else None,
        'anchor_mapping': {
            f'wt_{anchor}': {
                'mapped': int(anchor_frame[f'wt_{anchor}'].notna().sum()),
                'matches': int((anchor_frame[f'wt_{anchor}'] == residue).sum()),
            }
            for anchor, residue in [(14, 'H'), (17, 'H'), (135, 'T'), (138, 'R')]
        },
        'diagnostic_search': 'min-seq-id 0.2, cov-mode 2 query coverage 0.5',
        'diagnostic_hits': int(len(hits['diagnostic'])),
        'functional_overlaps': overlaps,
        'note': ('Reads only the derived old-only PF01467 domain FASTA; no Stage3 or current test '
                 'sequence is read. PF01467 HMM length is 133 aa; WT NatV NMNAT CDD domain is 178 aa, '
                 'so complete PF01467 domains are not complete NatV CDD domains.'),
    }
    (RAW / 'wt_reference_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
