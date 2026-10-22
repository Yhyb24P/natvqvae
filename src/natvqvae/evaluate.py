"""Validation evaluation for the QVAE checkpoint."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from .data import PopulationData, sha256_file, stable_seed, write_json
from .model import (FrozenEncoder, build_model, categorical_kl, categorical_log_prob,
                    categorical_samples, load_checkpoint, reconstruction_nll, target_tensors)


def evaluate(config_path, sources_path, target_path, checkpoint, output, device=None,
             split='validation', limit=None):
    if split != 'validation':
        raise ValueError('Only the validation split is evaluated; test remains sealed')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Evaluation output already contains files: {output}')
    data = PopulationData(sources_path, config_path, target_path)
    study = data.study
    config = study['evaluation']
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    payload = load_checkpoint(checkpoint, device='cpu')
    if payload.get('arm') != 'QVAE':
        raise ValueError('Evaluation requires a QVAE checkpoint')
    model = build_model(payload, device=device)
    encoder = FrozenEncoder(data.sources['esm_model'], device=device)
    queries = data.validation_queries()
    if limit is not None:
        queries = queries[:int(limit)]
    totals = []
    started = time.monotonic()
    batch_size = int(config['batch_size'])
    sample_count = int(config['sample_count'])
    repeats = int(config['repeats'])
    for start in range(0, len(queries), batch_size):
        batch = queries[start:start + batch_size]
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                           enabled=device.type == 'cuda'):
            context, full, valid, mask, labels = data.batch(batch, encoder, device, need_full=True)
            hidden = model.encode_context(context, valid, mask)
            targets, y, use, _ = target_tensors(hidden, mask, labels)
            prior = model.prior(hidden, valid, mask)
            posterior = model.posterior(full, hidden, valid, mask)
            for repeat in range(repeats):
                generator = torch.Generator(device=device.type)
                generator.manual_seed(stable_seed(config['seed'], 'evaluation', start, repeat))
                codes_prior = categorical_samples(prior, sample_count, generator)
                codes_post = categorical_samples(posterior, sample_count, generator)
                prior_logits = model.decode_codes(targets, codes_prior)
                post_logits = model.decode_codes(targets, codes_post)
                prior_nll = reconstruction_nll(prior_logits, y, use).mean(0)
                post_nll = reconstruction_nll(post_logits, y, use).mean(0)
                sites = use.sum(1)
                for index, query in enumerate(batch):
                    totals.append({
                        'panel': query['panel']['panel'],
                        'weight': float(query['panel']['weight']),
                        'sites': int(sites[index].item()),
                        'prior_nll_per_residue': float((prior_nll[index] / sites[index]).item()),
                        'posterior_nll_per_residue': float((post_nll[index] / sites[index]).item()),
                    })
    if not totals:
        raise ValueError('No validation queries evaluated')
    weighted = sum(item['weight'] for item in totals)
    summary = {
        'status': 'COMPLETED', 'split': split, 'checkpoint': str(checkpoint),
        'checkpoint_sha256': sha256_file(checkpoint),
        'queries': len(totals), 'sample_count': sample_count, 'repeats': repeats,
        'prior_nll_per_residue': sum(item['weight'] * item['prior_nll_per_residue']
                                     for item in totals) / weighted,
        'posterior_nll_per_residue': sum(item['weight'] * item['posterior_nll_per_residue']
                                         for item in totals) / weighted,
        'seconds': time.monotonic() - started,
        'esm_model': data.sources['esm_model'], 'target_sha256': sha256_file(target_path),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'evaluation.json', summary)
    with (output / 'per_query.jsonl').open('w') as handle:
        for item in totals:
            handle.write(json.dumps(item, ensure_ascii=False) + '\n')
    return summary
