#!/usr/bin/env python
"""CPU-only Phase B1 reranking diagnostic for existing NMNAT candidates."""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / 'data/results/natv_conditional_qvae/raw/annotated_candidates.parquet'
REPRESENTATIVES = ROOT / 'experiments/natv_conditional_qvae/representative_candidates.csv'
HYPOTHESES = ROOT / 'data/results/natv_evolution_informed_nmnat/raw/phase_a/hypothesis_table.csv'
RUN = ROOT / 'experiments/natv_evolution_informed_nmnat'
RAW = ROOT / 'data/results/natv_evolution_informed_nmnat/raw/phase_b1'
K = 3
SEED = 20261022
DOMAIN_START, DOMAIN_END = 4, 181


def parse_edits(value):
    try:
        edits = json.loads(value)
    except Exception:
        return set()
    return {(int(edit['position']), str(edit['from']), str(edit['to']))
            for edit in edits if DOMAIN_START <= int(edit['position']) <= DOMAIN_END}


def parse_hypotheses():
    frame = pd.read_csv(HYPOTHESES)
    singles = []
    pairs = []
    for item in frame.itertuples():
        triples = [(int(position), wt, alt)
                   for wt, position, alt in re.findall(r'([A-Z])(\d+)([A-Z])', item.substitutions)]
        if len(triples) == 1:
            singles.append((triples[0], int(item.n_clusters)))
        elif len(triples) == 2:
            pairs.append((tuple(triples), int(item.pair_clusters)))
    return singles, pairs


def score(subs, singles, pairs, weights=None):
    values = []
    for index, (triple, weight) in enumerate(singles):
        if triple in subs:
            values.append(weights[index] if weights is not None else weight)
    for pair, weight in pairs:
        if all(triple in subs for triple in pair):
            values.append(weight)
    return sum(values), len(values)


def select_top(frame, singles, pairs, weights=None, k=K):
    scored = frame.copy()
    parsed = [score(parse_edits(value), singles, pairs, weights) for value in scored.edits]
    scored['evidence_score'] = [value[0] for value in parsed]
    scored['evidence_matches'] = [value[1] for value in parsed]
    scored['nv'] = scored.edits.map(lambda value: len(parse_edits(value)))
    scored = scored.sort_values(['evidence_score', 'NMNAT_HMM_score', 'Hamming', 'candidate_id'],
                                ascending=[False, False, True, True])
    selected = scored.drop_duplicates('sequence_sha256').head(k)
    return selected, scored


def metrics(selected, singles, pairs):
    total = len(selected)
    matches = [parse_edits(value) for value in selected.edits] if total else []
    with_any = sum(1 for subs in matches if score(subs, singles, pairs)[0] > 0)
    with_pair = sum(1 for subs in matches if any(all(t in subs for t in pair) for pair, _ in pairs))
    supported = sum(score(subs, singles, pairs)[1] for subs in matches)
    return {
        'n_selected': total,
        'evidence_match_rate': with_any / total if total else None,
        'n_with_pair': with_pair,
        'n_supported_events': supported,
        'mean_supported_score': (sum(score(subs, singles, pairs)[0] for subs in matches) / total
                                 if total else None),
        'mean_Hamming': float(selected.Hamming.mean()) if total else None,
        'mean_nmnat_edits': (sum(len(parse_edits(value)) for value in selected.edits) / total
                             if total else None),
        'unique_sequences': int(selected.sequence_sha256.nunique()) if total else 0,
        'mean_NMNAT_HMM_score': float(selected.NMNAT_HMM_score.mean()) if total else None,
    }


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    singles, pairs = parse_hypotheses()
    frame = pd.read_parquet(CANDIDATES, columns=[
        'candidate_id', 'arm', 'region', 'edits', 'sequence_sha256', 'Hamming',
        'NMNAT_HMM_score'])
    frame = frame[frame.region == 'NMNAT'].copy()
    reps = pd.read_csv(REPRESENTATIVES)
    reps = reps[reps.region == 'NMNAT'][['candidate_id', 'arm']]

    supply = []
    for arm in ('QVAE', 'deterministic'):
        arm_frame = frame[frame.arm == arm]
        row = {'arm': arm, 'n_nmnat_candidates': len(arm_frame)}
        parsed = [parse_edits(value) for value in arm_frame.edits]
        for triple, _weight in singles:
            row[f'{triple[1]}{triple[0]}{triple[2]}'] = sum(1 for subs in parsed if triple in subs)
        for pair, _weight in pairs:
            label = '+'.join(f'{triple[1]}{triple[0]}{triple[2]}' for triple in pair)
            row[label] = sum(1 for subs in parsed if all(t in subs for t in pair))
        supply.append(row)
    supply_frame = pd.DataFrame(supply)
    supply_frame.to_csv(RAW / 'supply.csv', index=False)

    rng = random.Random(SEED)
    permuted_weights = [rng.choice([weight for _triple, weight in singles]) for _triple, _weight in singles]
    arms = {}
    selected_rows = []
    for arm in ('QVAE', 'deterministic'):
        arm_frame = frame[frame.arm == arm]
        current = arm_frame[arm_frame.candidate_id.isin(reps[reps.arm == arm].candidate_id)]
        top_evidence, scored = select_top(arm_frame, singles, pairs)
        top_permuted, _ = select_top(arm_frame, singles, pairs, permuted_weights)
        top_hmm = arm_frame.sort_values(['NMNAT_HMM_score', 'Hamming', 'candidate_id'],
                                        ascending=[False, True, True]).drop_duplicates('sequence_sha256').head(K)
        if arm == 'QVAE':
            arms['A_current'] = metrics(current, singles, pairs)
            arms['B_qvae_evidence'] = metrics(top_evidence, singles, pairs)
            arms['D_qvae_weight_permuted'] = metrics(top_permuted, singles, pairs)
            arms['A2_qvae_hmm_proxy'] = metrics(top_hmm, singles, pairs)
            selected = {'A_current': current, 'B_qvae_evidence': top_evidence,
                        'D_qvae_weight_permuted': top_permuted, 'A2_qvae_hmm_proxy': top_hmm}
        else:
            arms['C_deterministic_evidence'] = metrics(top_evidence, singles, pairs)
            selected = {'C_deterministic_evidence': top_evidence}
        for label, selected_frame in selected.items():
            for item in selected_frame.to_dict('records'):
                selected_rows.append({
                    'selection': label, 'candidate_id': item['candidate_id'], 'arm': arm,
                    'edits': item['edits'], 'sequence_sha256': item['sequence_sha256'],
                    'Hamming': item['Hamming'], 'NMNAT_HMM_score': item['NMNAT_HMM_score'],
                    'evidence_score': item.get('evidence_score'),
                    'evidence_matches': item.get('evidence_matches'),
                })

    selected_frame = pd.DataFrame(selected_rows)
    selected_frame.to_csv(RAW / 'selected_candidates.csv', index=False)
    pair_labels = ['+'.join(f'{triple[1]}{triple[0]}{triple[2]}' for triple in pair)
                   for pair, _weight in pairs]
    pair_supply_absent = {
        label: int(supply_frame[label].sum()) == 0 for label in pair_labels
    }
    summary = {
        'status': 'DESCRIPTIVE_ONLY',
        'gate_b1': 'not_evaluable',
        'held_out_evidence': False,
        'K_per_arm': K,
        'selection_scope': {
            'region': 'NMNAT',
            'candidate_pool_file': str(CANDIDATES),
            'nmnat_candidates_per_arm': int(len(frame[frame.arm == 'QVAE'])),
            'selection_count_per_arm': K,
            'deduplicate_within_arm_by': 'sequence_sha256',
            'no_new_generation': True,
        },
        'hypotheses': [
            {'substitutions': f'{triple[1]}{triple[0]}{triple[2]}', 'weight': weight}
            for triple, weight in singles
        ] + [
            {'substitutions': '+'.join(f'{triple[1]}{triple[0]}{triple[2]}' for triple in pair),
             'weight': weight} for pair, weight in pairs
        ],
        'supply': supply,
        'pair_supply_absent': pair_supply_absent,
        'arms': arms,
        'notes': [
            'A is the historical NMNAT representative selection (3 per arm) with a different selection objective; it is not a matched no-information baseline.',
            'B/C/D select 3 per arm from the same existing NMNAT pool; A/B/C/D all use the same 7 Phase A hypotheses, so match rate is descriptive enrichment, not held-out precision.',
            'D permutes weights among the same 7 events; it tests weight ordering, not an uninformative control over unrelated substitutions.',
            'B/C/D do not change the generative distribution or retrain; selected-candidate joint probability is not a model-likelihood gain.',
            'Raw supply counts are reported separately from selected-candidate match rates.',
            'Pair absence in the existing pool is a supply limitation, not evidence that the model cannot generate it under other masks; B2 is not started.',
            'Test/sealed sequences are not read; no new generation, training, download or folding.',
        ],
    }
    (RAW / 'rerank_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    status = {
        'status': summary['status'],
        'gate_b1': 'not_evaluable',
        'held_out_evidence': False,
        'K_per_arm': K,
        'runtime': 'CPU-only, existing candidate pool',
        'outputs': [str(RAW / name) for name in ['supply.csv', 'selected_candidates.csv', 'rerank_summary.json']],
    }
    (RUN / 'B1_STATUS.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
