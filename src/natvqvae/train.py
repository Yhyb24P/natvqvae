"""Formal QVAE training entry."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_

from .data import PopulationData, sha256_file, stable_seed, write_json
from .model import (ConditionalModel, FrozenEncoder, categorical_kl, categorical_log_prob,
                    categorical_samples, reconstruction_nll, rloo_objective, target_tensors)


def _learning_rate_at_epoch(base_lr, epoch_number, warmup_epochs):
    warmup = min(1.0, epoch_number / warmup_epochs) if warmup_epochs > 0 else 1.0
    return float(base_lr) * warmup


def _beta_at_epoch(epoch_number, warmup_epochs):
    return min(1.0, epoch_number / warmup_epochs) if warmup_epochs > 0 else 1.0


def _train_step(model, data, batch, weights, beta, generator, sample_count, encoder, device):
    context, full, valid, mask, labels = data.batch(batch, encoder, device, need_full=model.arm == 'QVAE')
    hidden = model.encode_context(context, valid, mask)
    targets, labels_target, use, _ = target_tensors(hidden, mask, labels)
    if model.arm == 'QVAE':
        prior = model.prior(hidden, valid, mask)
        posterior = model.posterior(full, hidden, valid, mask)
        codes = categorical_samples(posterior, sample_count, generator)
        logq = categorical_log_prob(posterior, codes)
        values = []
        for start in range(0, len(codes), 32):
            values.append(reconstruction_nll(model.decode_codes(targets, codes[start:start + 32]),
                                             labels_target, use))
        nll = torch.cat(values, 0)
        kl = categorical_kl(posterior, prior)
        loss = rloo_objective(nll, logq, kl, beta, weights)
        reconstruction = (nll.mean(0).detach() * weights).sum()
        denominator = (use.sum(1) * weights).sum()
        return loss, {
            'reconstruction_per_residue': float((reconstruction / denominator.clamp_min(1)).item()),
            'kl_per_factor': float((kl.detach() * weights).sum().item() /
                                   (model.factors * weights.sum().clamp_min(1).item())),
            'target_sites': int(use.sum().item()),
        }
    latent = model.deterministic_context(hidden, valid, mask)
    nll = reconstruction_nll(model.decode(targets, latent.unsqueeze(0)), labels_target, use)[0]
    loss = (nll * weights).mean()
    denominator = (use.sum(1) * weights).sum()
    return loss, {
        'reconstruction_per_residue': float((nll.detach() * weights).sum().item() /
                                            denominator.clamp_min(1).item()),
        'kl_per_factor': None, 'target_sites': int(use.sum().item()),
    }


def train(config_path, sources_path, target_path, output, device=None):
    data = PopulationData(sources_path, config_path, target_path)
    study = data.study
    config = study['training']
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(config['seed']))
    encoder = FrozenEncoder(data.sources['esm_model'], device=device)
    model = ConditionalModel('QVAE', **study['model']).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['learning_rate']),
                                  weight_decay=float(config['weight_decay']))
    steps_per_epoch = math.ceil(len(data.train) / int(config['batch_size']))
    max_epochs = math.ceil(int(config['max_steps']) / steps_per_epoch)
    metrics_path = output / 'metrics.jsonl'
    started = time.monotonic()
    step = 0
    with metrics_path.open('w') as metrics:
        for epoch in range(1, max_epochs + 1):
            beta = _beta_at_epoch(epoch, int(config['beta_warmup_epochs']))
            lr = _learning_rate_at_epoch(float(config['learning_rate']), epoch,
                                         int(config['learning_rate_warmup_epochs']))
            for group in optimizer.param_groups:
                group['lr'] = lr
            queries = data.training_queries(epoch - 1, int(config['seed']))
            epoch_loss = 0.0
            epoch_count = 0
            for start in range(0, len(queries), int(config['batch_size'])):
                if step >= int(config['max_steps']):
                    break
                batch = queries[start:start + int(config['batch_size'])]
                weights = torch.tensor([data.weights[q['record'].sequence_id] for q in batch],
                                       dtype=torch.float32, device=device)
                generator = torch.Generator(device=device.type)
                generator.manual_seed(stable_seed(config['seed'], 'latent', step) % (2**63 - 1))
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == 'cuda'):
                    loss, values = _train_step(model, data, batch, weights, beta, generator,
                                               int(config['K']), encoder, device)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at step {step + 1}')
                loss.backward()
                norm = clip_grad_norm_(model.parameters(), float(config['gradient_clip']))
                if not torch.isfinite(norm):
                    raise FloatingPointError(f'Nonfinite gradient at step {step + 1}')
                optimizer.step()
                step += 1
                epoch_loss += float(loss.detach().item())
                epoch_count += 1
                metrics.write(json.dumps({'epoch': epoch, 'step': step, 'loss': float(loss.detach().item()),
                                          'beta': beta, 'learning_rate': lr, **values},
                                         ensure_ascii=False) + '\n')
                metrics.flush()
                print(json.dumps({'step': step, 'loss': float(loss.detach().item()),
                                  'beta': beta, 'seconds': time.monotonic() - started}), flush=True)
            if step >= int(config['max_steps']):
                break
    payload = {
        'state_dict': model.state_dict(),
        'arm': 'QVAE',
        'seed': int(config['seed']),
        'model_config': dict(study['model']),
        'step': step,
        'epoch': max_epochs,
        'config': dict(config),
    }
    checkpoint = output / 'checkpoint.pt'
    torch.save(payload, checkpoint)
    write_json(output / 'training_summary.json', {
        'status': 'COMPLETED', 'steps': step, 'checkpoint': str(checkpoint),
        'checkpoint_sha256': sha256_file(checkpoint),
        'population': str(sources_path), 'target': str(target_path),
        'train_sequences': len(data.train), 'seconds': time.monotonic() - started,
        'esm_model': data.sources['esm_model'],
    })
    return checkpoint
