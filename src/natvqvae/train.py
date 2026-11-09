"""Formal QVAE training entry."""
from __future__ import annotations

import json
import math
import platform
import subprocess
import time
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_

from .data import PopulationData, sha256_file, stable_seed, write_json
from .model import (ConditionalModel, FrozenEncoder, categorical_kl, categorical_log_prob,
                    categorical_samples, load_checkpoint, reconstruction_nll, rloo_objective,
                    target_tensors)


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


def _state_cpu(model):
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def _save_checkpoint(path, model, config, study, step, epoch, initialization, selection):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': _state_cpu(model), 'arm': model.arm,
                'seed': int(config['seed']), 'model_config': dict(study['model']),
                'step': int(step), 'epoch': int(epoch), 'config': dict(config),
                'initialization': initialization, 'selection': selection}, path)


def _score_validation(model, data, encoder, queries, device, sample_count, seed):
    from .evaluate import conditional_scores
    model.eval()
    rows = []
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                         enabled=device.type == 'cuda'):
        for start in range(0, len(queries), 8):
            batch = queries[start:start + 8]
            context, full, valid, mask, labels = data.batch(
                batch, encoder, device, need_full=model.arm == 'QVAE')
            generator = torch.Generator(device=device.type)
            generator.manual_seed(stable_seed(seed, 'selection', start))
            scores = conditional_scores(model, context, full, valid, mask, labels,
                                        sample_count, generator)
            sites = scores['sites']
            for index, query in enumerate(batch):
                site = int(sites[index].item())
                if site < 1:
                    continue
                row = {'sequence_id': query['record'].sequence_id,
                       'component': query['record'].component,
                       'panel': query['panel']['panel'], 'sites': site,
                       'weight': float(query.get('weight', 1.0))}
                for name in ('prior_joint_nll', 'posterior_joint_nll', 'marginal_nll',
                             'prior_expected_reconstruction_nll',
                             'posterior_expected_reconstruction_nll'):
                    row[name + '_per_residue'] = float(scores[name][index].item()) / site
                row['kl'] = float(scores['kl'][index].item())
                if scores['ess'] is not None:
                    row['ess'] = float(scores['ess'][index].item())
                    row['max_weight'] = float(scores['max_weight'][index].item())
                rows.append(row)
    model.train()
    total_sites = sum(row['sites'] for row in rows)
    weight_sum = sum(row['weight'] for row in rows)
    if not rows or total_sites < 1 or weight_sum <= 0:
        raise ValueError('Validation selection produced no scored sites')
    summary = {'queries': len(rows), 'sites': total_sites,
               'weighted_sites': sum(row['weight'] * row['sites'] for row in rows),
               'selection_metric': 'prior_joint_nll_per_residue'}
    for name in ('prior_joint_nll', 'posterior_joint_nll', 'marginal_nll',
                 'prior_expected_reconstruction_nll',
                 'posterior_expected_reconstruction_nll'):
        summary[name + '_per_residue'] = sum(
            row[name + '_per_residue'] * row['weight'] for row in rows) / weight_sum
    summary['kl_mean'] = sum(row['kl'] * row['weight'] for row in rows) / weight_sum
    if rows[0].get('ess') is not None:
        summary['ess_mean'] = sum(row['ess'] * row['weight']
                                  for row in rows) / weight_sum
        summary['max_weight_mean'] = sum(row['max_weight'] * row['weight']
                                         for row in rows) / weight_sum
    return summary, rows


def train(config_path, sources_path, target_path, output, device=None,
          init_checkpoint=None, seed=None, arm='QVAE'):
    if arm not in {'QVAE', 'deterministic'}:
        raise ValueError('Training supports QVAE and deterministic arms')
    data = PopulationData(sources_path, config_path, target_path)
    study = data.study
    config = dict(study['training'])
    if seed is not None:
        config['seed'] = int(seed)
    selection_config = config.get('selection')
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    output = Path(output)
    checkpoint_root = Path(data.sources['checkpoint_dir']) / output.name
    if (output.exists() and any(output.iterdir())) or (
            checkpoint_root.exists() and any(checkpoint_root.iterdir())):
        raise FileExistsError(f'Run output or weights already exist: {output.name}')
    output.mkdir(parents=True, exist_ok=True)
    input_paths = [config_path, sources_path, target_path, data.sources['population'],
                   data.sources['splits']['training'], data.sources['splits']['validation']]
    if init_checkpoint:
        input_paths.append(init_checkpoint)
    esm_path = Path(data.sources['esm_model'])
    write_json(output / 'run_identity.json', {
        'input_sha256': {str(path): sha256_file(path) for path in input_paths},
        'esm_files_sha256': {path.name: sha256_file(path) for path in sorted(esm_path.iterdir())
                             if path.suffix in {'.json', '.safetensors'} or path.name == 'vocab.txt'},
        'code_sha256': {name: sha256_file(Path(__file__).with_name(name))
                       for name in ('train.py', 'model.py', 'data.py', 'evaluate.py')},
        'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--porcelain'], text=True).splitlines(),
        'python': platform.python_version(), 'torch': torch.__version__,
        'effective_training': config, 'seed_override': seed, 'device': str(device),
        'arm': arm,
        'initialization': str(init_checkpoint) if init_checkpoint else 'random',
    })
    torch.manual_seed(int(config['seed']))
    encoder = FrozenEncoder(data.sources['esm_model'], device=device)
    model = ConditionalModel(arm, **study['model']).to(device).train()
    if init_checkpoint:
        init_payload = load_checkpoint(init_checkpoint, device='cpu')
        if init_payload.get('arm') != arm:
            raise ValueError(f'Initialization checkpoint arm must be {arm}')
        model.load_state_dict(init_payload['state_dict'], strict=True)
        initialization = {'mode': 'checkpoint', 'path': str(init_checkpoint),
                          'sha256': sha256_file(init_checkpoint),
                          'checkpoint_seed': int(init_payload['seed']),
                          'checkpoint_step': int(init_payload['step'])}
    else:
        initialization = {'mode': 'random'}
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['learning_rate']),
                                  weight_decay=float(config['weight_decay']))
    steps_per_epoch = math.ceil(len(data.train) / int(config['batch_size']))
    max_epochs = math.ceil(int(config['max_steps']) / steps_per_epoch)
    metrics_path = output / 'metrics.jsonl'
    started = time.monotonic()
    step = 0
    curve = []
    paths = ({kind: checkpoint_root / f'{kind}.pt' for kind in ('initial', 'best', 'last')}
             if selection_config else {'last': checkpoint_root / 'checkpoint.pt'})
    selection_queries = []
    selected_records = []
    min_improvement = 0.0
    if selection_config:
        panel_names = list(selection_config.get('panels') or [selection_config['panel']])
        min_improvement = float(selection_config.get('min_improvement', 0.0))
        selected_records, selection_queries = data.validation_selection(
            int(selection_config['records']), panel_names)
        write_json(output / 'validation_selection.json', {
            'rule': 'component-balanced fixed selection; three NMNAT mask tasks equal; '
                    'each selected component equal; subset recomputed',
            'panels': panel_names, 'records': len(selected_records),
            'items': [{'sequence_id': query['record'].sequence_id,
                       'component': query['record'].component,
                       'panel': query['panel']['panel'],
                       'positions': query['panel']['positions'],
                       'weight': query['weight']}
                      for query in selection_queries],
        })
        initial_scores, _ = _score_validation(
            model, data, encoder, selection_queries, device,
            int(selection_config['sample_count']),
            stable_seed(config['seed'], 'selection', 0))
        _save_checkpoint(paths['initial'], model, config, study, 0, 0, initialization, initial_scores)
        _save_checkpoint(paths['best'], model, config, study, 0, 0, initialization, initial_scores)
        curve.append({'stage': 'initial', 'epoch': 0, 'step': 0, **initial_scores})
        best_scores, best_epoch = initial_scores, 0
        last_scores = initial_scores
        evaluated_epochs = {0}
    else:
        best_scores = best_epoch = last_scores = None
        evaluated_epochs = set()
    curve_path = output / 'validation_curve.jsonl'
    with metrics_path.open('w') as metrics, curve_path.open('w') as curve_file:
        if curve:
            curve_file.write(json.dumps(curve[0], ensure_ascii=False) + '\n')
            curve_file.flush()
        for epoch in range(1, max_epochs + 1):
            beta = _beta_at_epoch(epoch, int(config['beta_warmup_epochs']))
            lr = _learning_rate_at_epoch(float(config['learning_rate']), epoch,
                                         int(config['learning_rate_warmup_epochs']))
            for group in optimizer.param_groups:
                group['lr'] = lr
            queries = data.training_queries(epoch - 1, int(config['seed']))
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
                metrics.write(json.dumps({'epoch': epoch, 'step': step,
                                          'loss': float(loss.detach().item()), 'beta': beta,
                                          'learning_rate': lr, **values},
                                         ensure_ascii=False) + '\n')
                metrics.flush()
                print(json.dumps({'step': step, 'loss': float(loss.detach().item()),
                                  'beta': beta, 'seconds': time.monotonic() - started}), flush=True)
            if selection_config:
                interval = int(selection_config['interval_epochs'])
                if interval > 0 and epoch % interval == 0:
                    scores, _ = _score_validation(
                        model, data, encoder, selection_queries, device,
                        int(selection_config['sample_count']),
                        stable_seed(config['seed'], 'selection', epoch))
                    entry = {'stage': 'epoch', 'epoch': epoch, 'step': step, **scores}
                    curve.append(entry)
                    curve_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    curve_file.flush()
                    last_scores = scores
                    evaluated_epochs.add(epoch)
                    if (scores['prior_joint_nll_per_residue']
                            < best_scores['prior_joint_nll_per_residue'] - min_improvement):
                        best_scores, best_epoch = scores, epoch
                        _save_checkpoint(paths['best'], model, config, study, step, epoch,
                                         initialization, best_scores)
            if step >= int(config['max_steps']):
                break
        if selection_config and epoch not in evaluated_epochs:
            scores, _ = _score_validation(
                model, data, encoder, selection_queries, device,
                int(selection_config['sample_count']),
                stable_seed(config['seed'], 'selection', epoch))
            entry = {'stage': 'last', 'epoch': epoch, 'step': step, **scores}
            curve.append(entry)
            curve_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
            curve_file.flush()
            last_scores = scores
            if (scores['prior_joint_nll_per_residue']
                    < best_scores['prior_joint_nll_per_residue'] - min_improvement):
                best_scores, best_epoch = scores, epoch
                _save_checkpoint(paths['best'], model, config, study, step, epoch,
                                 initialization, best_scores)
    _save_checkpoint(paths['last'], model, config, study, step, epoch,
                     initialization, last_scores)
    returned = paths['best'] if selection_config else paths['last']
    summary = {
        'status': 'COMPLETED', 'steps': step, 'seed': int(config['seed']), 'arm': arm,
        'initialization': initialization, 'checkpoint': str(returned),
        'checkpoint_sha256': sha256_file(returned),
        'checkpoints': {kind: {'path': str(path), 'sha256': sha256_file(path)}
                        for kind, path in paths.items()},
        'population': str(sources_path), 'target': str(target_path),
        'train_sequences': len(data.train), 'validation_sequences': len(data.validation),
        'seconds': time.monotonic() - started, 'esm_model': data.sources['esm_model'],
    }
    if selection_config:
        summary.update({
            'selection': {'records': len(selected_records),
                          'queries': len(selection_queries),
                          'panels': panel_names,
                          'sample_count': int(selection_config['sample_count']),
                          'interval_epochs': int(selection_config['interval_epochs']),
                          'min_improvement': min_improvement,
                          'balance_key': 'component; three mask tasks equal'},
            'initial_validation': curve[0], 'best_validation': best_scores,
            'best_epoch': best_epoch, 'last_validation': last_scores,
            'validation_curve': curve,
        })
    write_json(output / 'training_summary.json', summary)
    return returned
