"""Independent conditional-prior generation from a saved QVAE checkpoint."""
from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import PopulationData, generation_masks, sha256_file, stable_seed, write_json
from .model import (FrozenEncoder, categorical_log_prob, categorical_samples,
                    load_checkpoint, build_model, sample_letters, target_tensors)


def generate(config_path, sources_path, target_path, checkpoint, output, device=None,
             group_filter=None, draws=None):
    data = PopulationData(sources_path, config_path, target_path)
    study = data.study
    generation = dict(study['generation'])
    if draws is not None:
        generation['draws_per_mask'] = int(draws)
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    payload = load_checkpoint(checkpoint, device='cpu')
    if payload.get('arm') != 'QVAE':
        raise ValueError('Generation requires a QVAE checkpoint')
    model = build_model(payload, device=device)
    target_sequence = data.target.sequence
    groups = generation_masks(generation, target_sequence)
    if group_filter:
        groups = [group for group in groups if group['id'] == group_filter]
        if not groups:
            raise ValueError(f'Unknown generation group: {group_filter}')
    encoder = FrozenEncoder(data.sources['esm_model'], device=device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'masks.json', groups)
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
            prior = model.prior(context, valid, mask)
            for index, group in enumerate(batch):
                prior_seed = stable_seed(generation['seed'], int(payload['seed']), group['id'], 'latent')
                uniform_seed = stable_seed(generation['seed'], int(payload['seed']), group['id'], 'residues')
                generator = torch.Generator(device=device.type)
                generator.manual_seed(prior_seed)
                codes = categorical_samples(prior[index:index + 1], draws_n, generator)
                logits = model.decode_codes(targets[index:index + 1], codes)[:, 0].float().cpu().numpy()
                logp = categorical_log_prob(prior[index:index + 1], codes)[:, 0].cpu().numpy()
                uniforms = np.random.default_rng(uniform_seed).random((draws_n, len(group['positions'])))
                letters = sample_letters(logits, uniforms, float(generation['temperature']))
                z_codes = codes[:, 0].cpu().tolist()
                for draw in range(draws_n):
                    chars = list(target_sequence)
                    for position, letter in zip(group['positions'], letters[draw]):
                        chars[position] = str(letter)
                    sequence = ''.join(chars)
                    digest = hashlib.sha256(sequence.encode()).hexdigest()
                    edits = [{'position': i + 1, 'from': a, 'to': b}
                             for i, (a, b) in enumerate(zip(target_sequence, sequence)) if a != b]
                    rows.append({
                        'candidate_id': f"QVAE_s{payload['seed']}_{group['id']}_d{draw:02d}",
                        'arm': 'QVAE', 'seed': int(payload['seed']), 'group': group['id'],
                        'region': group['region'], 'mask_size': group['mask_size'], 'draw': draw,
                        'sequence': sequence, 'sequence_sha256': digest,
                        'Hamming': len(edits), 'edits': json.dumps(edits, ensure_ascii=False),
                        'z_codes': json.dumps(z_codes[draw]), 'log_prior_z': float(logp[draw]),
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
        'status': 'COMPLETED', 'checkpoint': str(checkpoint),
        'checkpoint_sha256': sha256_file(checkpoint),
        'target_sha256': hashlib.sha256(target_sequence.encode()).hexdigest(),
        'esm_model': data.sources['esm_model'], 'device': str(device),
        'groups': len(groups), 'draws_per_mask': draws_n, 'attempts': len(rows),
        'seconds': time.monotonic() - started,
    })
    return rows
