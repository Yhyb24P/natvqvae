"""Generate WT-conservative QVAE proposals with masks that may keep or restore WT."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('HF_HUB_OFFLINE','1')
import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.dont_write_bytecode = True

from natvqvae.intake import sha256_file
from natvqvae.sequence import wt_sequence
from natvqvae.conservative_editing import legacy_scripts
from natvqvae.conservative_generation import (
    candidate_audit,
    draw_letters,
    eligible_positions,
    protected_positions_0based,
    stage_schedule,
)
from natvqvae.conditional_backend import ConditionalBackend

RUN = Path(__file__).parent


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def derive_seed(*parts):
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1, dtype=np.uint32)[0])


def fasta_sequences(path):
    sequences = []
    current = []
    for line in Path(path).read_text().splitlines():
        if line.startswith('>'):
            if current:
                sequences.append(''.join(current))
            current = []
        else:
            current.append(line.strip())
    if current:
        sequences.append(''.join(current))
    return sequences


def build_groups(plan):
    protected_positions = plan['protected_WT_positions_1based']
    eligible = np.asarray(eligible_positions(protected_positions))
    protected = set(protected_positions_0based(protected_positions))
    draws = plan['draws_per_shared_mask']
    groups = []
    for budget in plan['mask_budgets']:
        for index in range(plan['shared_masks_per_budget']):
            site_rng = np.random.default_rng(derive_seed(20261021, budget, index))
            sites = sorted(int(p) for p in site_rng.choice(eligible, budget, replace=False))
            if set(sites) & protected:
                raise ValueError('Protected position selected')
            stages = stage_schedule(sites, plan['update_sweeps'])
            uniform_rng = np.random.default_rng(derive_seed(20261022, budget, index))
            uniforms = [[uniform_rng.random(len(stage)).tolist() for stage in stages] for _ in range(draws)]
            prior_seeds = {str(seed): derive_seed(20261023, budget, index, seed) for seed in plan['seeds']}
            groups.append({'group_id': f'b{budget}_g{index}', 'budget': budget, 'group': index,
                           'sites': sites, 'stages': stages, 'uniforms': uniforms, 'prior_seeds': prior_seeds})
    expected = len(plan['mask_budgets']) * plan['shared_masks_per_budget']
    if len(groups) != expected or len({g['group_id'] for g in groups}) != expected:
        raise ValueError('Sampling group count differs')
    return groups


def freeze_plan(plan):
    path = RUN / 'sampling_groups.json'
    if path.exists():
        raise ValueError('Sampling plan already frozen')
    if not (RUN / 'inputs/source_checks.json').exists():
        raise ValueError('Run prepare.py first')
    groups = build_groups(plan)
    write(path, groups)
    code = [Path(__file__), ROOT / 'src/natvqvae/conservative_generation.py',
            ROOT / 'src/natvqvae/conservative_editing.py', ROOT / 'src/natvqvae/conditional_backend.py',
            ROOT / 'src/natvqvae/sequence.py', ROOT / 'src/natvqvae/intake.py']
    inputs = [RUN / 'plan.json', RUN / 'inputs/source_checks.json', RUN / 'sampling_groups.json',
              ROOT / 'contracts/objective.yaml', ROOT / 'inputs/manifest.json',
              ROOT / 'discovery/natv_data_readiness/training.fasta',
              ROOT / 'discovery/natv_data_readiness/validation.fasta']
    files = [p for p in code + inputs if p.is_file()]
    attempts = len(groups) * len(plan['seeds']) * plan['draws_per_shared_mask']
    write(RUN / 'run_identity.json', {
        'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'files': {str(p.relative_to(ROOT)): sha256_file(p) for p in files},
        'groups': len(groups), 'attempts': attempts})
    return groups


def verify_identity():
    identity = json.loads((RUN / 'run_identity.json').read_text())
    for path, expected in identity['files'].items():
        if sha256_file(ROOT / path) != expected:
            raise ValueError('Run identity changed: ' + path)
    return identity


def run_job(backend, checkpoint, group, plan, destination, edit_script, apply_script):
    count = plan['draws_per_shared_mask']
    wt = wt_sequence()
    current = [wt] * count
    model = backend.model('QVAE', checkpoint)
    prior_seed = group['prior_seeds'][str(checkpoint)]
    generator = torch.Generator(device='cpu').manual_seed(prior_seed)
    z = model.sample_prior(count, generator=generator, device='cpu').cuda()
    z_states = z.argmax(-1).cpu().numpy().astype(np.int8)
    z_onehot = z.cpu().numpy().astype(np.uint8)
    z_hashes = [hashlib.sha256(z_onehot[i].tobytes()).hexdigest() for i in range(count)]
    traces = []
    stage_changes = [[] for _ in range(count)]
    ever_changed = [set() for _ in range(count)]
    diagnostics = []
    started = time.monotonic()
    for stage, wt_sites in enumerate(group['stages']):
        positions = list(wt_sites)
        masks = [positions] * count
        encoded, evolution, receipts, unique = yield current, masks
        logits, _ = backend.logits('QVAE', model, encoded, evolution, z)
        if stage == len(group['stages']) - 1:
            with torch.no_grad():
                disabled = model(encoded, encoded['target_mask'], evolution=evolution, disable_latent=True)['logits']
            diagnostics.append({'stage': stage,
                                'latent_disabled_logit_max_difference': float((logits - disabled).abs().max()),
                                'latent_disabled_logit_mean_abs_difference': float((logits - disabled).abs().mean()),
                                'interpretation': 'mechanism_probe_not_generation_gain'})
        for draw in range(count):
            before = current[draw]
            letters = draw_letters(logits[draw, positions].detach().float().cpu().numpy(),
                                   group['uniforms'][draw][stage], plan['temperature'])
            after = list(before)
            for position, letter in zip(positions, letters):
                after[position] = letter
            after = ''.join(after)
            changed = [p for p in positions if after[p] != before[p]]
            kept = [p for p in positions if after[p] == before[p]]
            restored = [p for p in positions if before[p] != wt[p] and after[p] == wt[p]]
            ever_changed[draw].update(p for p in positions if after[p] != wt[p])
            stage_changes[draw].append(len(changed))
            traces.append({'checkpoint_seed': checkpoint, 'group_id': group['group_id'], 'draw': draw,
                           'stage': stage, 'mask_positions_0based': positions, 'before': before, 'after': after,
                           'masked_query': ''.join('X' if j in positions else aa for j, aa in enumerate(before)),
                           'uniform_draws': group['uniforms'][draw][stage],
                           'z_sha256': z_hashes[draw], 'prior_seed': prior_seed, 'retrieval': receipts[draw],
                           'changed_positions_0based': changed, 'kept_positions_0based': kept,
                           'restored_to_WT_positions_0based': restored,
                           'fresh_current_context': True, 'unique_batch_contexts': unique})
            current[draw] = after
        del encoded, evolution, logits
    rows = []
    scripts = []
    for draw, sequence in enumerate(current):
        candidate = f'QVAE_s{checkpoint}_{group["group_id"]}_d{draw + 1:02d}'
        audit = candidate_audit(wt, sequence, group['sites'], sorted(ever_changed[draw]),
                                plan['protected_WT_positions_1based'])
        operations = edit_script(wt, sequence)
        reconstructs = apply_script(wt, operations) == sequence
        rows.append({
            'candidate_id': candidate, 'checkpoint_seed': checkpoint, 'group_id': group['group_id'],
            'budget': group['budget'], 'mask_size': len(group['sites']), 'draw': draw + 1,
            'sequence': sequence, 'sequence_sha256': hashlib.sha256(sequence.encode()).hexdigest(),
            'length': audit['length'], 'distance': audit['distance'],
            'WT_global_identity': audit['WT_global_identity'],
            'actual_count': audit['actual_count'],
            'actual_positions_1based': json.dumps([p + 1 for p in audit['actual_positions_0based']]),
            'mask_positions_1based': json.dumps([p + 1 for p in group['sites']]),
            'mask_retained_count': audit['mask_retained_count'],
            'reversion_count': audit['reversion_count'],
            'reversion_positions_1based': json.dumps([p + 1 for p in audit['reversion_positions_0based']]),
            'stage_change_counts': json.dumps(stage_changes[draw]),
            'prior_seed': prior_seed, 'z_states': json.dumps(z_states[draw].tolist()),
            'z_sha256': z_hashes[draw],
            'constraint_status': 'PASS' if audit['constraint_pass'] and reconstructs else 'FAIL',
            'protected_intact': bool(audit['protected_intact']),
            'script_reconstructs': bool(reconstructs), 'error': ''})
        scripts.append({'candidate_id': candidate, 'operations': operations, 'reconstructs_exactly': reconstructs,
                        'alignment': 'legacy_global_BLOSUM62; Levenshtein_distance_computed_separately',
                        'construction_event': None, 'WT_to_scaffold_current': list(range(len(wt)))})
    with gzip.open(destination / 'trajectory.jsonl.gz', 'wt') as handle:
        for row in traces:
            handle.write(json.dumps(row) + '\n')
    with gzip.open(destination / 'edit_scripts.jsonl.gz', 'wt') as handle:
        for row in scripts:
            handle.write(json.dumps(row) + '\n')
    np.savez_compressed(destination / 'latents.npz', states=z_states, onehot=z_onehot,
                        prior_seed=np.asarray([prior_seed], dtype=np.int64))
    write(destination / 'latent_diagnostics.json', diagnostics)
    write(destination / 'results.json', {
        'rows': rows, 'seconds': time.monotonic() - started,
        'artifacts': {p.name: sha256_file(p) for p in destination.glob('*.gz')} |
                     {p.name: sha256_file(p) for p in [destination / 'latents.npz', destination / 'latent_diagnostics.json']}})
    return rows


def load_group_rows(seed, group, destination):
    record = json.loads((destination / 'results.json').read_text())
    for name, expected in record.get('artifacts', {}).items():
        if sha256_file(destination / name) != expected:
            raise ValueError('Group artifact changed: ' + str(destination / name))
    return record['rows']


def assign_status(rows, training, validation):
    seen = {}
    for row in rows:
        row['training_exact_match'] = row['sequence'] in training
        row['validation_exact_match'] = row['sequence'] in validation
        if row['distance'] == 0:
            row['candidate_status'] = 'WT_REFERENCE'
            row['duplicate_of'] = ''
        elif row['sequence_sha256'] in seen:
            row['candidate_status'] = 'DUPLICATE'
            row['duplicate_of'] = seen[row['sequence_sha256']]
        else:
            seen[row['sequence_sha256']] = row['candidate_id']
            row['candidate_status'] = 'NEW_UNIQUE'
            row['duplicate_of'] = ''


def runtime_environment():
    return {'python': sys.version.split()[0], 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else '',
            'threads': torch.get_num_threads()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    plan = json.loads((RUN / 'plan.json').read_text())
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    if args.resume:
        identity = verify_identity()
        groups = json.loads((RUN / 'sampling_groups.json').read_text())
    else:
        groups = freeze_plan(plan)
        identity = json.loads((RUN / 'run_identity.json').read_text())
    if args.plan_only:
        print(json.dumps({'groups': len(groups), 'attempts': identity['attempts'],
                          'budgets': plan['mask_budgets']}), flush=True)
        return
    source = Path(json.loads((RUN / 'inputs/source_checks.json').read_text())['source_root'])
    if not torch.cuda.is_available():
        raise RuntimeError('This run requires an available GPU for frozen ESM-2')
    write(RUN / 'runtime_environment.json', runtime_environment())
    backend = ConditionalBackend(source, RUN)
    edit_script, apply_script = legacy_scripts(source)
    training = set(fasta_sequences(ROOT / 'discovery/natv_data_readiness/training.fasta'))
    validation = set(fasta_sequences(ROOT / 'discovery/natv_data_readiness/validation.fasta'))
    generation = RUN / 'generation'
    generation.mkdir(exist_ok=args.resume)
    rows = []
    started = time.monotonic()
    completed = 0
    ledger = []
    try:
        for group in groups:
            group_id = group['group_id']
            destinations = [generation / f'checkpoint_{seed}' / group_id for seed in plan['seeds']]
            features = generation / 'features' / group_id
            if all((destination / 'results.json').exists() for destination in destinations):
                if not args.resume:
                    raise ValueError('Generation already exists: ' + group_id)
                for seed, destination in zip(plan['seeds'], destinations):
                    rows.extend(load_group_rows(seed, group, destination))
                completed += len(destinations)
                ledger.append({'group_id': group_id, 'budget': group['budget'], 'sites': len(group['sites']),
                               'attempts': len(destinations) * plan['draws_per_shared_mask'], 'status': 'PRESENT'})
                continue
            if any(destination.exists() for destination in destinations) or features.exists():
                raise ValueError('Incomplete group requires inspection: ' + group_id)
            jobs = []
            for seed, destination in zip(plan['seeds'], destinations):
                destination.mkdir(parents=True)
                job = run_job(backend, seed, group, plan, destination, edit_script, apply_script)
                jobs.append((seed, job, next(job)))
            pending = []
            stage = 0
            while jobs:
                sequences = []
                masks = []
                sizes = []
                for seed, job, (current, group_masks) in jobs:
                    sizes.append(len(current))
                    sequences.extend(current)
                    masks.extend(group_masks)
                encoded, evolution, receipts, unique = backend.features(
                    sequences, masks, features / f'stage{stage}')
                following = []
                begin = 0
                for (seed, job, request), size in zip(jobs, sizes):
                    end = begin + size
                    enc = None if encoded is None else {
                        key: value[begin:end] if isinstance(value, torch.Tensor) else value
                        for key, value in encoded.items()}
                    response = (enc, evolution[begin:end], receipts[begin:end], unique)
                    try:
                        following.append((seed, job, job.send(response)))
                    except StopIteration as done:
                        rows.extend(done.value)
                        completed += 1
                    begin = end
                jobs = following
                stage += 1
                del encoded, evolution
            ledger.append({'group_id': group_id, 'budget': group['budget'], 'sites': len(group['sites']),
                           'attempts': len(destinations) * plan['draws_per_shared_mask'], 'status': 'GENERATED'})
            write(RUN / 'run_status.json', {'status': 'RUNNING', 'completed_group_runs': completed,
                                            'attempts': len(rows), 'elapsed_seconds': time.monotonic() - started})
            print(json.dumps({'completed_group_runs': completed, 'group_runs': len(groups) * len(plan['seeds']),
                              'attempts': len(rows), 'elapsed_seconds': time.monotonic() - started}), flush=True)
        assign_status(rows, training, validation)
        frame = pd.DataFrame(rows)
        frame.to_csv(RUN / 'raw_candidates.csv', index=False)
        if len(frame) != identity['attempts']:
            raise ValueError('Attempt denominator differs')
        write(RUN / 'group_ledger.json', ledger)
        write(RUN / 'generation_receipt.json', {
            'loaded_models': backend.loaded, 'imported_modules': backend.module_hashes,
            'unique_actual_PLM_sequence_encodings': backend.forward_sequences,
            'retrieval_queries': backend.retrieval_sequences, 'stage_timing': backend.timing,
            'PLM_trainable_parameters': 0, 'elapsed_seconds': time.monotonic() - started,
            'raw_candidates_sha256': sha256_file(RUN / 'raw_candidates.csv')})
        write(RUN / 'run_status.json', {'status': 'GENERATION_COMPLETED', 'attempts': len(rows),
                                        'constraint_counts': frame.constraint_status.value_counts().to_dict(),
                                        'candidate_counts': frame.candidate_status.value_counts().to_dict()})
    except BaseException as exc:
        write(RUN / 'run_status.json', {'status': 'FAILED', 'completed_group_runs': completed,
                                        'attempts_preserved': len(rows), 'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
