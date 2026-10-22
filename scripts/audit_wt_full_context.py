#!/usr/bin/env python
"""Check old PF01467-only proteins in full context against the NatV WT NMNAT domain.

Only the 8,849 stage3-only PF01467 IDs are materialized from the read-only Stage3
FASTA. Current test, sealed, conflict and excluded sequences are never accessed.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from natvqvae.sequence import wt_sequence  # noqa: E402

MMSEQS = Path('/home/yhshy/miniconda3/envs/nad_data/bin/mmseqs')
RAW = ROOT / 'data/results/natv_evolution_data_audit/raw'
WORK = ROOT / 'data/interim/natv_evolution_data_audit/full_context'
OLD_REPORT = Path('/home/yhshy/git_files/nad_qvae/experiments/run_reports/pretraining_scientific_validation_20260909')
OLD_STAGE3 = OLD_REPORT / 'stage3_accepted_sequences.fasta'
AUDIT = RAW / 'pf01467_only_audit.csv'
WT_LENGTH = 178


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


def main():
    if not MMSEQS.exists():
        raise FileNotFoundError(MMSEQS)
    WORK.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(AUDIT)
    allowed = set(audit.seq_hash)
    roles = pd.read_csv(ROOT / 'data/processed/roles.csv')
    forbidden = set(roles.loc[roles.training_role_v2.isin(
        ['test', 'quarantined_old_sealed_component', 'annotation_conflict_quarantine',
         'excluded_noncanonical_or_domain']), 'sequence_id'])
    if allowed & forbidden:
        raise RuntimeError(f'whitelist intersects protected roles: {len(allowed & forbidden)}')

    # Local SQLite index; never writes next to the read-only source FASTA.
    index_path = WORK / 'stage3_index.sqlite'
    if index_path.exists():
        index_path.unlink()
    index = SeqIO.index_db(str(index_path), str(OLD_STAGE3), 'fasta')
    missing = allowed - set(index)
    if missing:
        raise RuntimeError(f'{len(missing)} whitelist IDs missing from source')
    full_fasta = WORK / 'old_only_full_proteins.fasta'
    with full_fasta.open('w') as handle:
        for seq_hash in audit.seq_hash:
            handle.write(f'>{seq_hash}\n{index[seq_hash].seq}\n')
    index.close()
    if sum(1 for _ in SeqIO.parse(str(full_fasta), 'fasta')) != len(audit):
        raise RuntimeError('full-protein FASTA count mismatch')

    wt = wt_sequence()
    wt_domain = wt[3:181]
    wt_fasta = WORK / 'wt_nmnat_domain.fasta'
    wt_fasta.write_text('>natv:WT_NMNAT_4_181\n' + wt_domain + '\n')

    summary = {}
    for name, cutoff in [('t50', '0.5'), ('t80', '0.8')]:
        prefix = WORK / name
        if prefix.exists():
            shutil.rmtree(prefix)
        prefix.mkdir(parents=True)
        out = prefix / 'hits.tsv'
        run([str(MMSEQS), 'easy-search', str(full_fasta), str(wt_fasta), str(out),
             str(prefix / 'tmp'), '--min-seq-id', '0.3', '--cov-mode', '1', '-c', cutoff,
             '--threads', '8', '--format-output',
             'query,target,fident,alnlen,qstart,qend,tstart,tend,evalue'])
        rows = []
        for line in out.read_text().splitlines():
            parts = line.split('\t')
            if len(parts) < 9:
                continue
            query, _target, fident, alnlen, qstart, qend, tstart, tend, _evalue = parts
            rows.append((query, float(fident), int(alnlen), int(qstart), int(qend),
                         int(tstart), int(tend)))
        hits = pd.DataFrame(rows, columns=['seq_hash', 'fident', 'alnlen', 'qstart', 'qend',
                                           'tstart', 'tend'])
        if hits.empty:
            summary[name] = {'hits': 0}
            continue
        hits['target_coverage'] = hits.alnlen / WT_LENGTH
        hits = hits.sort_values(['fident', 'target_coverage'],
                                ascending=False).drop_duplicates('seq_hash')
        hits = hits.merge(audit[['seq_hash', 'primary_layer', 'profile_coverage', 'domain_length',
                                 'domain_start', 'domain_end', 'length', 'partial',
                                 'nmnat_hit_count', 'ec_nmnat_only', 'ec_dual',
                                 'annotation_conflict', '0.7_mode1_cluster']], on='seq_hash', how='left')
        hits['complete'] = (hits.profile_coverage >= 0.8) & (hits.domain_length >= 124)
        hits['ec_support'] = hits.ec_nmnat_only | hits.ec_dual
        hits['outside_right'] = hits.qend > hits.domain_end
        hits['outside_left'] = hits.qstart < hits.domain_start
        hits['extension_right'] = np.maximum(0, hits.qend - hits.domain_end)
        hits['extension_left'] = np.maximum(0, hits.domain_start - hits.qstart)
        hits.to_csv(RAW / f'wt_full_context_hits_{name}.csv', index=False)
        summary[name] = {
            'hits': int(len(hits)),
            'clusters_70_mode1': int(hits['0.7_mode1_cluster'].nunique()),
            'identity_median': float(hits.fident.median()),
            'target_coverage_median': float(hits.target_coverage.median()),
            'target_coverage_ge_0_8': int((hits.target_coverage >= 0.8).sum()),
            'outside_right': int(hits.outside_right.sum()),
            'outside_left': int(hits.outside_left.sum()),
            'extension_right_median': float(hits.extension_right.median()),
            'extension_left_median': float(hits.extension_left.median()),
            'complete': int(hits.complete.sum()),
            'partial': int(hits.partial.sum()),
            'ec_support': int(hits.ec_support.sum()),
            'complete_ec': int((hits.complete & hits.ec_support).sum()),
            'complete_ec_clusters': int(hits.loc[hits.complete & hits.ec_support,
                                                '0.7_mode1_cluster'].nunique()),
            'layers': {key: int(value) for key, value in hits.primary_layer.value_counts().items()},
            'layer_clusters': {key: int(value) for key, value in
                               hits.groupby('primary_layer')['0.7_mode1_cluster'].nunique().items()},
        }

    result = {
        'status': 'SUPPLEMENT_AUDIT_ONLY',
        'inputs': {
            'old_stage3': {'path': str(OLD_STAGE3), 'sha256': sha256(OLD_STAGE3)},
            'old_audit': {'path': str(AUDIT), 'sha256': sha256(AUDIT)},
        },
        'whitelist': {
            'size': len(allowed),
            'protected_role_intersection': 0,
            'description': 'only the 8,849 stage3-only PF01467 IDs are materialized; test/sealed/conflict/excluded are not accessed',
        },
        'wt_domain': {'start': 4, 'end': 181, 'length': WT_LENGTH,
                      'sequence_sha256': hashlib.sha256(wt_domain.encode()).hexdigest()},
        'target_coverage_mode': 'cov-mode 1 (target coverage), min-seq-id 0.3',
        'results': summary,
        'note': ('Local PF01467 slices can be shorter than the full protein. This supplement '
                 'checks the full context; any extra target coverage comes from sequence outside '
                 'the PF01467 best-alignment slice and does not by itself establish NMNAT function.'),
    }
    (RAW / 'wt_full_context_summary.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
