"""Summarize actual edit coverage without treating attempts as new candidates."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

RUN = Path(__file__).parent
STRATA = [(0, 0), (1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30), (31, 34)]
ACTUAL_STRATA = STRATA + [(35, 341)]


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def stratum(value, bounds=STRATA):
    for low, high in bounds:
        if low <= value <= high:
            return str(low) if low == high else f'{low}-{high}'
    return 'out_of_range'


def region(position):
    if 1 <= position <= 3:
        return 'NTERM'
    if 4 <= position <= 181:
        return 'NMNAT'
    if 182 <= position <= 203:
        return 'LINKER'
    if 204 <= position <= 338:
        return 'NUDIX'
    if 339 <= position <= 341:
        return 'CTERM'
    raise ValueError('Position outside WT')


def load_rows():
    frame = pd.read_csv(RUN / 'raw_candidates.csv')
    for column in ['actual_positions_1based', 'mask_positions_1based', 'reversion_positions_1based', 'stage_change_counts', 'z_states']:
        frame[column] = frame[column].map(json.loads)
    return frame


def group_statistics(frame, keys):
    rows = []
    for key, group in frame.groupby(keys, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(keys, key))
        distances = group.distance.to_numpy()
        row.update({
            'attempts': len(group),
            'unique_sequences': int(group.sequence_sha256.nunique()),
            'new_unique': int((group.candidate_status == 'NEW_UNIQUE').sum()),
            'duplicates': int((group.candidate_status == 'DUPLICATE').sum()),
            'wt_reference': int((group.candidate_status == 'WT_REFERENCE').sum()),
            'failures': int((group.constraint_status == 'FAIL').sum()),
            'distance_min': int(distances.min()),
            'distance_median': float(np.median(distances)),
            'distance_max': int(distances.max()),
            'identity_min': float(group.WT_global_identity.min()),
            'actual_count_mean': float(group.actual_count.mean()),
            'actual_count_median': float(group.actual_count.median()),
            'mask_retained_mean': float(group.mask_retained_count.mean()),
            'reversion_count_mean': float(group.reversion_count.mean()),
            'stage_changes_mean': float(group.stage_change_counts.map(sum).mean()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    plan = json.loads((RUN / 'plan.json').read_text())
    frame = load_rows()
    if len(frame) != plan['execution']['attempts']:
        raise ValueError('Attempt denominator differs')
    out = RUN / 'coverage'
    out.mkdir(exist_ok=True)
    frame['distance_stratum'] = frame.distance.map(stratum)
    frame['actual_stratum'] = frame.actual_count.map(lambda value: stratum(value, ACTUAL_STRATA))

    group_statistics(frame, ['group_id', 'budget']).to_csv(out / 'per_group.csv', index=False)
    group_statistics(frame, ['checkpoint_seed', 'group_id', 'budget']).to_csv(out / 'per_group_checkpoint.csv', index=False)
    group_statistics(frame, ['checkpoint_seed']).to_csv(out / 'per_checkpoint.csv', index=False)
    group_statistics(frame, ['budget']).to_csv(out / 'per_budget.csv', index=False)

    strata_rows = []
    scopes = [('overall', frame)] + [(f'checkpoint_{seed}', group) for seed, group in frame.groupby('checkpoint_seed')]
    for scope, group in scopes:
        counts = group.distance_stratum.value_counts()
        for low, high in STRATA:
            label = stratum(low)
            attempts = int(counts.get(label, 0))
            strata_rows.append({'scope': scope, 'stratum': label, 'attempts': attempts,
                                'denominator': len(group), 'fraction': attempts / len(group)})
    pd.DataFrame(strata_rows).to_csv(out / 'edit_strata.csv', index=False)

    cross = pd.crosstab(frame.budget, frame.actual_stratum)
    cross = cross.reindex(columns=[stratum(low, ACTUAL_STRATA) for low, _ in ACTUAL_STRATA], fill_value=0)
    cross.reset_index().to_csv(out / 'budget_vs_actual.csv', index=False)

    masked = {p: 0 for p in range(1, 342)}
    changed = {p: 0 for p in range(1, 342)}
    reverted = {p: 0 for p in range(1, 342)}
    for row in frame.itertuples():
        positions = set(row.mask_positions_1based)
        actual = set(row.actual_positions_1based)
        back = set(row.reversion_positions_1based)
        for position in positions:
            masked[position] += 1
        for position in actual:
            changed[position] += 1
        for position in back:
            reverted[position] += 1
    positions = pd.DataFrame([{'position_1based': p, 'region': region(p), 'times_masked': masked[p],
                               'times_changed': changed[p], 'times_retained_when_masked': masked[p] - changed[p],
                               'times_reverted': reverted[p]} for p in range(1, 342)])
    positions.to_csv(out / 'position_coverage.csv', index=False)
    regions = positions.groupby('region').agg(positions=('position_1based', 'size'),
                                              positions_masked=('times_masked', lambda s: int((s > 0).sum())),
                                              positions_changed=('times_changed', lambda s: int((s > 0).sum())),
                                              masked_events=('times_masked', 'sum'),
                                              changed_events=('times_changed', 'sum'),
                                              reverted_events=('times_reverted', 'sum')).reset_index()
    regions.to_csv(out / 'region_coverage.csv', index=False)

    duplicates = frame[frame.candidate_status == 'DUPLICATE'][
        ['candidate_id', 'duplicate_of', 'sequence_sha256', 'checkpoint_seed', 'group_id', 'budget', 'draw']]
    duplicates.to_csv(out / 'duplicates.csv', index=False)

    diagnostic_rows = []
    for (seed, group_id), group in frame.groupby(['checkpoint_seed', 'group_id']):
        states = np.asarray(group.z_states.tolist(), dtype=np.int16)
        pair = (states[:, None, :] != states[None, :, :]).mean(-1)
        upper = pair[np.triu_indices(len(states), 1)]
        diagnostic = RUN / 'generation' / f'checkpoint_{seed}' / group_id / 'latent_diagnostics.json'
        record = {}
        if diagnostic.exists():
            entry = json.loads(diagnostic.read_text())[0]
            record = {'latent_disabled_logit_max_difference': entry['latent_disabled_logit_max_difference'],
                      'latent_disabled_logit_mean_abs_difference': entry['latent_disabled_logit_mean_abs_difference']}
        diagnostic_rows.append({'checkpoint_seed': seed, 'group_id': group_id, 'draws': len(states),
                                'unique_z_states': len({tuple(row) for row in states.tolist()}),
                                'mean_pairwise_z_hamming_fraction': float(upper.mean()), **record})
    pd.DataFrame(diagnostic_rows).to_csv(out / 'z_diagnostics.csv', index=False)

    with (out / 'candidates_all.fasta').open('w') as handle:
        for row in frame.itertuples():
            handle.write(f'>{row.candidate_id}|{row.candidate_status}|s{row.checkpoint_seed}|{row.group_id}|d{row.draw}|edits={row.distance}\n{row.sequence}\n')
    unique = frame[frame.candidate_status == 'NEW_UNIQUE']
    with (out / 'new_unique.fasta').open('w') as handle:
        for row in unique.itertuples():
            handle.write(f'>{row.candidate_id}|s{row.checkpoint_seed}|{row.group_id}|d{row.draw}|edits={row.distance}\n{row.sequence}\n')

    covered = set(frame.distance_stratum)
    summary = {
        'attempts': len(frame), 'groups': int(frame.group_id.nunique()),
        'group_runs': int(frame.groupby(['checkpoint_seed', 'group_id']).ngroups),
        'checkpoints': int(frame.checkpoint_seed.nunique()),
        'constraint_counts': {k: int(v) for k, v in frame.constraint_status.value_counts().items()},
        'candidate_status_counts': {k: int(v) for k, v in frame.candidate_status.value_counts().items()},
        'unique_sequences': int(frame.sequence_sha256.nunique()),
        'distance': {'min': int(frame.distance.min()), 'median': float(frame.distance.median()),
                     'max': int(frame.distance.max())},
        'identity_min': float(frame.WT_global_identity.min()),
        'actual_count': {'min': int(frame.actual_count.min()), 'median': float(frame.actual_count.median()),
                         'max': int(frame.actual_count.max())},
        'mask_retained': {'min': int(frame.mask_retained_count.min()),
                          'median': float(frame.mask_retained_count.median()),
                          'max': int(frame.mask_retained_count.max())},
        'reversion_candidates': int((frame.reversion_count > 0).sum()),
        'exact_training_matches': int(frame.training_exact_match.sum()),
        'exact_validation_matches': int(frame.validation_exact_match.sum()),
        'covered_distance_strata': [label for _, label in sorted((low, stratum(low)) for low, _ in STRATA)
                                    if label in covered],
        'missing_distance_strata': [label for _, label in sorted((low, stratum(low)) for low, _ in STRATA)
                                    if label not in covered],
        'scope': 'generation_coverage; attempts_include_WT_reference_duplicates; no_activity_claim',
    }
    write(out / 'coverage_summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
