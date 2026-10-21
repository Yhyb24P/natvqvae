"""Audit trajectories, explicit edit scripts and the attempt denominator."""
from pathlib import Path
import gzip
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.dont_write_bytecode = True

from natvqvae.intake import sha256_file
from natvqvae.sequence import wt_sequence
from natvqvae.conservative_editing import edit_distance, legacy_scripts
from natvqvae.conservative_generation import protected_positions_0based

RUN = Path(__file__).parent


def read_jsonl_gz(path):
    with gzip.open(path, 'rt') as handle:
        return [json.loads(line) for line in handle]


def main():
    plan = json.loads((RUN / 'plan.json').read_text())
    groups = json.loads((RUN / 'sampling_groups.json').read_text())
    identity = json.loads((RUN / 'run_identity.json').read_text())
    source = Path(json.loads((RUN / 'inputs/source_checks.json').read_text())['source_root'])
    _, apply_script = legacy_scripts(source)
    wt = wt_sequence()
    protected = set(protected_positions_0based(plan['protected_WT_positions_1based']))
    protected_1based = set(plan['protected_WT_positions_1based'])
    frame = pd.read_csv(RUN / 'raw_candidates.csv')
    for column in ['actual_positions_1based', 'mask_positions_1based', 'reversion_positions_1based', 'stage_change_counts', 'z_states']:
        frame[column] = frame[column].map(json.loads)
    table = {(row.checkpoint_seed, row.group_id, row.draw): row for row in frame.itertuples()}

    checks = {}
    for path, expected in identity['files'].items():
        checks[f'identity:{path}'] = sha256_file(ROOT / path) == expected
    checks['group_count'] = len(groups) == plan['execution']['groups']
    checks['attempt_denominator'] = len(frame) == identity['attempts'] == plan['execution']['attempts']
    checks['attempts_per_checkpoint'] = bool((frame.groupby('checkpoint_seed').size() ==
                                              plan['draws_per_shared_mask'] * len(groups)).all())
    checks['unique_candidate_ids'] = frame.candidate_id.is_unique
    checks['all_constraints_pass'] = bool((frame.constraint_status == 'PASS').all())
    checks['all_scripts_reconstruct'] = bool(frame.script_reconstructs.all())
    checks['all_protected_intact'] = bool(frame.protected_intact.all())
    seen = {}
    status_ok = True
    for row in frame.itertuples():
        duplicate_of = '' if pd.isna(row.duplicate_of) else str(row.duplicate_of)
        if row.distance == 0:
            status_ok &= row.candidate_status == 'WT_REFERENCE' and duplicate_of == ''
        elif row.sequence_sha256 in seen:
            status_ok &= row.candidate_status == 'DUPLICATE' and duplicate_of == seen[row.sequence_sha256]
        else:
            seen[row.sequence_sha256] = row.candidate_id
            status_ok &= row.candidate_status == 'NEW_UNIQUE' and duplicate_of == ''
    checks['candidate_status_consistent'] = bool(status_ok)

    group_map = {group['group_id']: group for group in groups}
    training = set()
    validation = set()
    for name, sink in [('training.fasta', training), ('validation.fasta', validation)]:
        current = []
        for line in (ROOT / 'discovery/natv_data_readiness' / name).read_text().splitlines():
            if line.startswith('>'):
                if current:
                    sink.add(''.join(current))
                current = []
            else:
                current.append(line.strip())
        if current:
            sink.add(''.join(current))

    row_checks = []
    for row in frame.itertuples():
        mask = set(row.mask_positions_1based)
        actual = set(row.actual_positions_1based)
        sequence = row.sequence
        states = np.asarray(row.z_states, dtype=np.int16)
        onehot = np.zeros((len(states), 8), dtype=np.uint8)
        onehot[np.arange(len(states)), states] = 1
        row_checks.append([
            row.length == 341,
            actual <= mask,
            not protected_1based & mask,
            all(sequence[p] == wt[p] for p in protected),
            edit_distance(sequence, wt) == row.distance,
            sum(a != b for a, b in zip(sequence, wt)) == row.actual_count,
            row.mask_retained_count == len(mask) - row.actual_count,
            int(row.actual_count + row.mask_retained_count) == row.mask_size,
            hashlib.sha256(onehot.tobytes()).hexdigest() == row.z_sha256,
            len(states) == plan['generator']['latent_dim'],
            bool(np.isin(states, range(plan['generator']['latent_states'])).all()),
            (sequence in training) == bool(row.training_exact_match),
            (sequence in validation) == bool(row.validation_exact_match),
        ])
    checks['row_level'] = bool(np.asarray(row_checks).all())

    traces_ok = True
    scripts_ok = True
    latents_ok = True
    steps = 0
    scripts = 0
    for group in groups:
        for seed in plan['seeds']:
            directory = RUN / 'generation' / f'checkpoint_{seed}' / group['group_id']
            traces = read_jsonl_gz(directory / 'trajectory.jsonl.gz')
            scripts_rows = read_jsonl_gz(directory / 'edit_scripts.jsonl.gz')
            steps += len(traces)
            scripts += len(scripts_rows)
            if len(traces) != plan['draws_per_shared_mask'] * len(group['stages']) or len(scripts_rows) != plan['draws_per_shared_mask']:
                traces_ok = False
            by_draw = {}
            for trace in traces:
                by_draw.setdefault(trace['draw'], []).append(trace)
            saved = np.load(directory / 'latents.npz')
            states, onehot = saved['states'], saved['onehot']
            if states.shape != (plan['draws_per_shared_mask'], plan['generator']['latent_dim']):
                latents_ok = False
            if onehot.shape != (plan['draws_per_shared_mask'], plan['generator']['latent_dim'], plan['generator']['latent_states']):
                latents_ok = False
            for draw, stages in by_draw.items():
                stages.sort(key=lambda item: item['stage'])
                row = table[(seed, group['group_id'], draw + 1)]
                if [item['stage'] for item in stages] != list(range(len(group['stages']))):
                    traces_ok = False
                    continue
                if stages[0]['before'] != wt or stages[0]['z_sha256'] != row.z_sha256:
                    traces_ok = False
                ever = set()
                for index, trace in enumerate(stages):
                    mask = group['stages'][index]
                    if trace['mask_positions_0based'] != mask or trace['prior_seed'] != group['prior_seeds'][str(seed)]:
                        traces_ok = False
                    if index and trace['before'] != stages[index - 1]['after']:
                        traces_ok = False
                    if any(trace['after'][p] != trace['before'][p] for p in range(341) if p not in mask):
                        traces_ok = False
                    if trace['masked_query'] != ''.join('X' if p in mask else trace['before'][p] for p in range(341)):
                        traces_ok = False
                    changed = [p for p in mask if trace['after'][p] != trace['before'][p]]
                    kept = [p for p in mask if trace['after'][p] == trace['before'][p]]
                    restored = [p for p in mask if trace['before'][p] != wt[p] and trace['after'][p] == wt[p]]
                    if changed != trace['changed_positions_0based'] or kept != trace['kept_positions_0based'] or restored != trace['restored_to_WT_positions_0based']:
                        traces_ok = False
                    if trace['uniform_draws'] != group['uniforms'][draw][index]:
                        traces_ok = False
                    ever.update(p for p in mask if trace['after'][p] != wt[p])
                final = stages[-1]['after']
                actual = {p for p in range(341) if final[p] != wt[p]}
                if final != row.sequence or sorted(ever - actual) != [p - 1 for p in row.reversion_positions_1based]:
                    traces_ok = False
                recomputed = hashlib.sha256(onehot[draw].tobytes()).hexdigest()
                if recomputed != row.z_sha256 or not np.array_equal(states[draw], onehot[draw].argmax(-1)):
                    latents_ok = False
                if not saved['prior_seed'][0] == group['prior_seeds'][str(seed)]:
                    latents_ok = False
            for script in scripts_rows:
                draw = int(script['candidate_id'].rsplit('_d', 1)[1]) - 1
                row = table[(seed, group['group_id'], draw + 1)]
                if apply_script(wt, script['operations']) != row.sequence or not script['reconstructs_exactly']:
                    scripts_ok = False
            if not (directory / 'latent_diagnostics.json').exists():
                latents_ok = False
    checks['trajectory_integrity'] = traces_ok
    checks['edit_script_reconstruction'] = scripts_ok
    checks['latent_record_integrity'] = latents_ok
    checks['step_denominator'] = steps == len(frame) * (1 + 2 * plan['update_sweeps'])
    checks['script_denominator'] = scripts == len(frame)

    historical = json.loads((ROOT / 'experiments/natv_conservative_editing/run_identity.json').read_text())
    for path in ['src/natvqvae/conservative_editing.py', 'src/natvqvae/conditional_backend.py']:
        if path in historical['files']:
            checks[f'historical_module:{path}'] = sha256_file(ROOT / path) == historical['files'][path]

    result = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
              'counts': {'attempts': len(frame), 'steps': steps, 'scripts': scripts,
                         'group_runs': len(groups) * len(plan['seeds']),
                         'new_unique': int((frame.candidate_status == 'NEW_UNIQUE').sum()),
                         'duplicates': int((frame.candidate_status == 'DUPLICATE').sum()),
                         'wt_reference': int((frame.candidate_status == 'WT_REFERENCE').sum())}}
    (RUN / 'result_checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    if result['status'] != 'PASS':
        raise SystemExit(json.dumps(result, ensure_ascii=False))
    print(json.dumps(result['counts']), flush=True)


if __name__ == '__main__':
    main()
