"""Independent conditional-prior generation from a saved QVAE checkpoint."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .data import PopulationData, generation_masks, sha256_file, stable_seed, write_json
from .model import (FrozenEncoder, categorical_log_prob, categorical_samples,
                    load_checkpoint, build_model, sample_letters, target_tensors)


def _background_generation(generation, record):
    intervals = record.intervals()
    return {**generation, 'seed': stable_seed(generation['seed'], 'background', record.sequence_id),
            'regions_1based': {region: [intervals[region][0] + 1, intervals[region][1]]
                              for region in generation['regions_1based']}}


def prepare_generation_backgrounds(config_path, sources_path, target_path, test_data, output):
    """Fix one sequence per held-out component using IDs, without prediction scores."""
    from .test_data import load_recovered_test

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Background manifest already exists: {output}')
    data = PopulationData(sources_path, config_path, target_path)
    records = load_recovered_test(data, test_data)
    selected = {}
    for record in sorted(records, key=lambda row: row.sequence_id):
        selected.setdefault(record.component, record)
    generation = data.study['generation']
    entries = [{'sequence_id': record.sequence_id, 'component': component,
                'length': record.length, 'generation': _background_generation(generation, record),
                'masks': generation_masks(_background_generation(generation, record), record.sequence)}
               for component, record in sorted(selected.items())]
    inputs = [config_path, sources_path, target_path, data.sources['population'], *data.sources['splits'].values(),
              *(Path(test_data) / name for name in ('test.csv', 'test.fasta', 'manifest.json', 'historical_use.json'))]
    result = {'selection': 'lexicographically smallest sequence SHA256 in each held-out component',
              'test_data': str(test_data), 'available_sequences': len(records), 'backgrounds': entries,
              'kvp40': {'sequence_sha256': hashlib.sha256(data.target.sequence.encode()).hexdigest(),
                        'generation': generation, 'masks': generation_masks(generation, data.target.sequence)},
              'input_sha256': {str(path): sha256_file(path) for path in inputs},
              'implementation_sha256': sha256_file(Path(__file__)),
              'code_sha256': {name: sha256_file(Path(__file__).with_name(name))
                              for name in ('generate.py', 'test_data.py', 'data.py')},
              'prediction_scores_used': False,
              'limit': 'These held-out sequences have prior evaluation history; one representative per component does not describe all within-component variation.'}
    write_json(output / 'backgrounds.json', result)
    return result


def generate(config_path, sources_path, target_path, checkpoint, output, device=None,
             group_filter=None, draws=None, distribution='joint', background_id=None, test_data=None,
             backgrounds_manifest=None):
    if distribution not in {'joint', 'site_marginal'}:
        raise ValueError('Unknown generation distribution')
    if backgrounds_manifest is not None:
        if background_id is not None or test_data is None or group_filter is not None:
            raise ValueError('Batch generation requires test_data and no background_id or group filter')
    elif (background_id is None) != (test_data is None):
        raise ValueError('Held-out generation requires both background_id and test_data')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Generation output already contains files: {output}')
    data = PopulationData(sources_path, config_path, target_path)
    study = data.study
    generation = dict(study['generation'])
    if draws is not None:
        generation['draws_per_mask'] = int(draws)
    if int(generation['draws_per_mask']) < 1:
        raise ValueError('Generation requires positive draws per mask')
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    payload = load_checkpoint(checkpoint, device='cpu')
    arm = payload.get('arm')
    if arm not in {'QVAE', 'deterministic'}:
        raise ValueError('Generation requires a QVAE or deterministic checkpoint')
    if distribution == 'site_marginal' and arm != 'QVAE':
        raise ValueError('Site-marginal ablation requires QVAE; deterministic is already factorized')
    records = {}
    if test_data is not None:
        from .test_data import load_recovered_test
        records = {record.sequence_id: record for record in load_recovered_test(data, test_data)}
    if background_id is not None and background_id not in records:
        raise ValueError('Generation background is not in the recovered test set')
    input_paths = [config_path, sources_path, target_path, checkpoint, data.sources['population'],
                   *data.sources['splits'].values()]
    if test_data is not None:
        input_paths.extend(Path(test_data) / name for name in ('test.csv', 'test.fasta', 'manifest.json', 'historical_use.json'))
    input_hashes = {str(path): sha256_file(path) for path in input_paths}
    manifest = None
    background_ids = [background_id]
    if backgrounds_manifest is not None:
        manifest = json.loads(Path(backgrounds_manifest).read_text())
        current = {str(Path(path).resolve()): digest for path, digest in input_hashes.items()}
        recorded = {str(Path(path).resolve()): digest for path, digest in manifest['input_sha256'].items()}
        expected_paths = set(current) - {str(Path(checkpoint).resolve())}
        if set(recorded) != expected_paths or any(recorded[path] != current[path] for path in recorded):
            raise ValueError('Generation background manifest input hashes differ')
        selected = {}
        for record in sorted(records.values(), key=lambda row: row.sequence_id):
            selected.setdefault(record.component, record.sequence_id)
        members = [(row['component'], row['sequence_id']) for row in manifest['backgrounds']]
        if members != sorted(selected.items()):
            raise ValueError('Background manifest does not contain the fixed component representatives')
        background_ids = [uid for _, uid in members] + [None]
        input_hashes[str(backgrounds_manifest)] = sha256_file(backgrounds_manifest)
    jobs = []
    for uid in background_ids:
        record = records.get(uid)
        sequence = record.sequence if record else data.target.sequence
        settings = _background_generation(generation, record) if record else generation
        groups = generation_masks(settings, sequence)
        if group_filter:
            groups = [group for group in groups if group['id'] == group_filter]
            if not groups:
                raise ValueError(f'Unknown generation group: {group_filter}')
        if manifest is not None:
            fixed = (next(row for row in manifest['backgrounds'] if row['sequence_id'] == uid)
                     if uid else manifest['kvp40'])
            if fixed['generation'] != settings or fixed['masks'] != groups:
                raise ValueError('Batch generation masks or settings differ from fixed manifest')
            if uid is None and fixed['sequence_sha256'] != hashlib.sha256(sequence.encode()).hexdigest():
                raise ValueError('Background manifest KVP40 sequence differs')
        jobs.append({'output': output / (uid or 'KVP40') if manifest else output,
                     'sequence': sequence, 'generation': settings, 'groups': groups,
                     'background_id': uid, 'background_component': record.component if record else None})
    esm_path = Path(data.sources['esm_model'])
    identity = {
        'input_sha256': input_hashes,
        'esm_files_sha256': {path.name: sha256_file(path) for path in sorted(esm_path.iterdir())
                             if path.suffix in {'.json', '.safetensors'} or path.name == 'vocab.txt'},
        'code_sha256': {name: sha256_file(Path(__file__).with_name(name))
                       for name in ('generate.py', 'model.py', 'data.py') +
                       (('test_data.py',) if test_data is not None else ())},
        'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'group_filter': group_filter, 'draws_override': draws,
        'backgrounds_manifest': str(backgrounds_manifest) if backgrounds_manifest is not None else None,
        'distribution': distribution if arm == 'QVAE' else 'factorized',
        'latent_layout': 'independent_per_site' if distribution == 'site_marginal' else
                         'shared_across_sites' if arm == 'QVAE' else None,
        'log_prior_z_definition': 'sum of independent site-code log probabilities' if distribution == 'site_marginal' else
                                  'shared-code log probability' if arm == 'QVAE' else None,
        'checkpoint': str(checkpoint), 'checkpoint_seed': int(payload['seed']),
        'esm_model': str(data.sources['esm_model']),
        'checkpoint_sha256': input_hashes[str(checkpoint)],
        'checkpoint_arm': payload['arm'], 'checkpoint_step': payload.get('step'),
        'mask_batch_size': 16, 'device': str(device), 'amino_acid_output_count': 20,
        'precision': 'bfloat16 forward, float32 logits, float64 residue sampling' if device.type == 'cuda' else
                     'float32 logits, float64 residue sampling',
    }
    model = build_model(payload, device=device)
    encoder = FrozenEncoder(data.sources['esm_model'], device=device)
    attempts = 0
    for job in jobs:
        rows = _generate_background(model, encoder, payload, device, job, distribution, identity)
        attempts += len(rows)
    if manifest is None:
        return rows
    summary = {'status': 'COMPLETED', 'backgrounds': len(jobs), 'attempts': attempts,
               'checkpoint_sha256': identity['checkpoint_sha256'],
               'distribution': identity['distribution'],
               'backgrounds_manifest_sha256': input_hashes[str(backgrounds_manifest)],
               'input_sha256': input_hashes}
    write_json(output / 'batch_generation.json', summary)
    return summary


def _generate_background(model, encoder, payload, device, job, distribution, identity):
    output, target_sequence = job['output'], job['sequence']
    generation, groups, background_id = job['generation'], job['groups'], job['background_id']
    arm = payload['arm']
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'masks.json', groups)
    write_json(output / 'run_identity.json', {
        **identity, 'generation': generation, 'background_id': background_id,
        'background_component': job['background_component'],
        'target_sha256': hashlib.sha256(target_sequence.encode()).hexdigest(),
        'masks_sha256': sha256_file(output / 'masks.json'), 'groups': len(groups),
    })
    rows = []
    started = time.monotonic()
    draws_n = int(generation['draws_per_mask'])
    for start in range(0, len(groups), 16):
        batch = groups[start:start + 16]
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                           enabled=device.type == 'cuda'):
            h, valid, mask = encoder([target_sequence] * len(batch),
                                     [group['positions'] for group in batch])
            h = h.to(device); valid = valid.to(device); mask = mask.to(device)
            context = model.encode_context(h, valid, mask)
            labels = torch.zeros(len(batch), len(target_sequence), dtype=torch.long, device=device)
            targets, _, _, _ = target_tensors(context, mask, labels)
            if arm == 'QVAE':
                prior = model.prior(context, valid, mask)
            else:
                latent = model.deterministic_context(context, valid, mask)
                deterministic_logits = model.decode(targets, latent.unsqueeze(0))[0]
            for index, group in enumerate(batch):
                prior_seed = stable_seed(generation['seed'], int(payload['seed']), group['id'], 'latent')
                if distribution == 'site_marginal':
                    prior_seed = stable_seed(generation['seed'], int(payload['seed']), group['id'], 'independent_sites')
                uniform_seed = stable_seed(generation['seed'], int(payload['seed']), group['id'], 'residues')
                if arm == 'QVAE':
                    generator = torch.Generator(device=device.type)
                    generator.manual_seed(prior_seed)
                    if distribution == 'site_marginal':
                        sites = len(group['positions'])
                        site_prior = prior[index:index + 1].expand(sites, -1, -1)
                        codes = categorical_samples(site_prior, draws_n, generator)
                        # Each site's independent code integrates to that same site's marginal.
                        logits = model.decode_codes(targets[index, :sites].unsqueeze(1), codes)
                        logits = logits[:, :, 0].float().cpu().numpy()
                        logp = categorical_log_prob(site_prior, codes).sum(1).cpu().numpy()
                        z_codes = codes.cpu().tolist()
                    else:
                        codes = categorical_samples(prior[index:index + 1], draws_n, generator)
                        logits = model.decode_codes(targets[index:index + 1], codes)[:, 0].float().cpu().numpy()
                        logp = categorical_log_prob(prior[index:index + 1], codes)[:, 0].cpu().numpy()
                        z_codes = codes[:, 0].cpu().tolist()
                else:
                    prior_seed = None
                    logits = np.broadcast_to(deterministic_logits[index].float().cpu().numpy(),
                                             (draws_n, targets.shape[1], 20))
                    logp, z_codes = [None] * draws_n, [None] * draws_n
                uniforms = np.random.default_rng(uniform_seed).random((draws_n, len(group['positions'])))
                logits = logits[:, :len(group['positions'])]
                letters = sample_letters(logits, uniforms, float(generation['temperature']))
                for draw in range(draws_n):
                    chars = list(target_sequence)
                    for position, letter in zip(group['positions'], letters[draw]):
                        chars[position] = str(letter)
                    sequence = ''.join(chars)
                    digest = hashlib.sha256(sequence.encode()).hexdigest()
                    edits = [{'position': i + 1, 'from': a, 'to': b}
                             for i, (a, b) in enumerate(zip(target_sequence, sequence)) if a != b]
                    rows.append({
                        'candidate_id': f"{background_id + '_' if background_id else ''}{arm}{'_site_marginal' if distribution == 'site_marginal' else ''}_s{payload['seed']}_{group['id']}_d{draw:02d}",
                        'arm': arm, 'seed': int(payload['seed']), 'group': group['id'],
                        'distribution': distribution if arm == 'QVAE' else 'factorized',
                        'region': group['region'], 'mask_size': group['mask_size'], 'draw': draw,
                        'sequence': sequence, 'sequence_sha256': digest,
                        'Hamming': len(edits), 'edits': json.dumps(edits, ensure_ascii=False),
                        'z_codes': json.dumps(z_codes[draw]),
                        'log_prior_z': float(logp[draw]) if logp[draw] is not None else None,
                        'prior_seed': prior_seed, 'uniform_seed': uniform_seed,
                    })
        print(json.dumps({'groups': min(start + 16, len(groups)), 'total': len(groups),
                          'seconds': time.monotonic() - started}), flush=True)
    with (output / 'candidates.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (output / 'candidates.fasta').open('w') as handle:
        for row in rows:
            handle.write(f">{row['candidate_id']}\n{row['sequence']}\n")
    write_json(output / 'generation.json', {
        'status': 'COMPLETED', 'checkpoint': identity['checkpoint'],
        'distribution': distribution if arm == 'QVAE' else 'factorized',
        'checkpoint_sha256': identity['checkpoint_sha256'],
        'target_sha256': hashlib.sha256(target_sequence.encode()).hexdigest(),
        'esm_model': identity['esm_model'], 'device': str(device),
        'groups': len(groups), 'draws_per_mask': draws_n, 'attempts': len(rows),
        'seconds': time.monotonic() - started,
    })
    return rows
