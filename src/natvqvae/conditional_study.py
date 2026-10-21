"""NatV 条件类别 QVAE 的校准、选参与正式训练。

本模块只负责训练端：固定预算内的 4 次校准、后验重构梯度诊断、似然校准、
每 arm 的 6 配置 x 2 种子选参及 5 个正式种子。生成与最终交付由
conditional_outputs.finish_study 承担。
"""
from __future__ import annotations

import gzip
import gc
import hashlib
import json
import math
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from .conditional_data import FeatureData, stable_seed, status, write_json
from .conditional_evaluation import (aggregate_panel_records, compare_joint_scores,
                                     deterministic_joint_probability, posterior_joint_probability,
                                     validation_score)
from .conditional_generation import (ConditionalModel, categorical_kl, categorical_log_prob,
                                     categorical_samples, reconstruction_nll, rloo_objective,
                                     target_tensors)
from .intake import sha256_file

ROOT = Path(__file__).resolve().parents[2]
_EPS = 1e-12


class BudgetExceeded(RuntimeError):
    pass


class FitFailed(RuntimeError):
    pass


def _json_float(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _seed_all(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _numpy_state():
    state = np.random.get_state()
    return {'state': state[1].tolist(), 'pos': int(state[2]),
            'has_gauss': int(state[3]), 'cached_gaussian': float(state[4])}


def rng_state():
    state = {'python': list(random.getstate()), 'numpy': _numpy_state(),
             'torch': torch.get_rng_state().tolist()}
    if torch.cuda.is_available():
        state['cuda'] = [item.tolist() for item in torch.cuda.get_rng_state_all()]
    return state


def set_rng_state(state):
    random.setstate((state['python'][0], tuple(state['python'][1]), state['python'][2]))
    numpy_state = state['numpy']
    np.random.set_state(('MT19937', np.asarray(numpy_state['state'], dtype=np.uint32),
                         numpy_state['pos'], numpy_state['has_gauss'],
                         numpy_state['cached_gaussian']))
    torch.set_rng_state(torch.tensor(state['torch'], dtype=torch.uint8))
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([torch.tensor(item, dtype=torch.uint8)
                                      for item in state['cuda']])


def budget_steps(plan, T):
    """B=max(5200,4*T)，但不超过绝对上限；T 是校准确认最佳步数最大值。"""
    floor = int(plan['budget']['calibration_steps_per_fit'])
    ceiling = int(plan['budget']['absolute_steps_per_remaining_fit_max'])
    return min(ceiling, max(floor, 4 * int(T)))


def expected_fit_count(plan):
    arms = len(plan['arms'])
    calibration = arms * len(plan['calibration']['rates'])
    search = arms * len(plan['search_rate_multipliers']) * len(plan['search_weight_decay']) * len(plan['selection_seeds'])
    formal = arms * len(plan['formal_seeds'])
    return calibration, search, formal, calibration + search + formal


def learning_rate_at_epoch(base_lr, epoch_number, warmup_epochs, lr_factor=1.):
    warmup = min(1., epoch_number / warmup_epochs) if warmup_epochs > 0 else 1.
    return float(base_lr) * warmup * float(lr_factor)


def beta_at_epoch(epoch_number, warmup_epochs):
    return min(1., epoch_number / warmup_epochs) if warmup_epochs > 0 else 1.


def initial_selection_state():
    return {'best': None, 'best_step': None, 'best_epoch': None,
            'since_improve': 0, 'lr_factor': 1., 'lr_reductions': 0,
            'lr_reductions_since_improve': 0, 'last_lr_reduction_since_improve': 0,
            'last_lr_reduction_epoch': None, 'last_score': None,
            'last_score_reliable': None, 'stop_reason': None}


def update_selection(state, score, step, epoch_number, delta, beta_ready,
                     lr_patience=8, stop_patience=30):
    """更新 best/早停计数；返回 (state, improved)。

    只有 score 与 best 的差超过 delta 才更新 best 和重置计数。连续 lr_patience
    轮无改善降学习率；beta=1 后连续 stop_patience 轮无改善且至少两次降率无效时早停。
    """
    state = dict(state)
    state['last_score'] = _json_float(score)
    if not beta_ready:
        return state, False
    finite = score is not None and math.isfinite(float(score))
    if finite and (state['best'] is None or state['best'] - float(score) > float(delta)):
        state.update(best=float(score), best_step=int(step), best_epoch=int(epoch_number),
                     since_improve=0, lr_reductions_since_improve=0,
                     last_lr_reduction_since_improve=0)
        return state, True
    state['since_improve'] = int(state['since_improve']) + 1
    if (state['best'] is not None and state['since_improve'] >= lr_patience and
            state['since_improve'] - state['last_lr_reduction_since_improve'] >= lr_patience):
        state['lr_factor'] = float(state['lr_factor']) * .5
        state['lr_reductions'] = int(state['lr_reductions']) + 1
        state['lr_reductions_since_improve'] = int(state['lr_reductions_since_improve']) + 1
        state['last_lr_reduction_since_improve'] = int(state['since_improve'])
        state['last_lr_reduction_epoch'] = int(epoch_number)
    if (state['since_improve'] >= stop_patience and
            state['lr_reductions_since_improve'] >= 2):
        state['stop_reason'] = 'early_stopping'
    return state, False


def _checkpoint_paths(root, run_name, fit_key):
    directory = root / 'data/models' / run_name / fit_key
    return directory / 'best.pt', directory / 'last.pt'


def _torch_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def checkpoint_payload(model, optimizer, arm, seed, model_config, step, epoch,
                       metric, config, include_optimizer=True):
    payload = {'state_dict': model.state_dict(), 'arm': arm, 'seed': int(seed),
               'model_config': model_config, 'step': int(step), 'epoch': int(epoch),
               'selected_metric': _json_float(metric), 'config': config,
               'rng': rng_state()}
    if include_optimizer:
        payload['optimizer'] = optimizer.state_dict()
    return payload


def load_checkpoint(path, map_location='cpu'):
    return torch.load(path, map_location=map_location, weights_only=True)


def _append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def _fit_weights(data, batch, device):
    return torch.tensor([float(data.weights[query['row']]) for query in batch],
                        dtype=torch.float32, device=device)


def _decode_nll(model, targets, labels, use, codes, sample_chunk=32):
    values = []
    for start in range(0, len(codes), sample_chunk):
        logits = model.decode_codes(targets, codes[start:start+sample_chunk])
        values.append(reconstruction_nll(logits, labels, use))
    return torch.cat(values, 0)


def _autocast(device):
    if device.type == 'cuda':
        return torch.autocast('cuda', dtype=torch.bfloat16)
    return torch.autocast('cpu', enabled=False)


def _train_step(model, data, batch, weights, beta, generator, sample_count, device):
    need_full = model.arm == 'QVAE'
    context, full, valid, mask, labels = data.batch(batch, device=device, need_full=need_full)
    with _autocast(device):
        hidden = model.encode_context(context, valid, mask)
        targets, labels_target, use, _ = target_tensors(hidden, mask, labels)
        if model.arm == 'QVAE':
            prior = model.prior(hidden, valid, mask)
            posterior = model.posterior(full, hidden, valid, mask)
            codes = categorical_samples(posterior, sample_count, generator)
            logq = categorical_log_prob(posterior, codes)
            nll = _decode_nll(model, targets, labels_target, use, codes)
            kl = categorical_kl(posterior, prior)
            loss = rloo_objective(nll, logq, kl, beta, weights)
            reconstruction_numerator = (nll.mean(0).detach() * weights).sum()
            reconstruction_denominator = (use.sum(1) * weights).sum()
            kl_numerator = (kl.detach() * weights).sum() / model.factors
            kl_denominator = weights.sum()
            metrics = {
                'reconstruction_numerator': float(reconstruction_numerator.item()),
                'reconstruction_denominator': float(reconstruction_denominator.item()),
                'reconstruction_per_residue': float(
                    (reconstruction_numerator / reconstruction_denominator.clamp_min(1.)).item()),
                'kl_numerator': float(kl_numerator.item()),
                'kl_denominator': float(kl_denominator.item()),
                'kl_per_factor': float(
                    (kl_numerator / kl_denominator.clamp_min(1.)).item()),
                'target_sites': int(use.sum().item()),
            }
        else:
            latent = model.deterministic_context(hidden, valid, mask)
            logits = model.decode(targets, latent.unsqueeze(0))
            nll = reconstruction_nll(logits, labels_target, use)[0]
            loss = (nll * weights).mean()
            reconstruction_numerator = (nll.detach() * weights).sum()
            reconstruction_denominator = (use.sum(1) * weights).sum()
            metrics = {
                'reconstruction_numerator': float(reconstruction_numerator.item()),
                'reconstruction_denominator': float(reconstruction_denominator.item()),
                'reconstruction_per_residue': float(
                    (reconstruction_numerator / reconstruction_denominator.clamp_min(1.)).item()),
                'kl_numerator': None, 'kl_denominator': None, 'kl_per_factor': None,
                'target_sites': int(use.sum().item()),
            }
    return loss, metrics


def _compact_summary(summary):
    return {'score': _json_float(summary.get('score')), 'se': _json_float(summary.get('se')),
            'finite': bool(summary.get('finite')), 'repeats': summary.get('repeats'),
            'input_ess_mean': _json_float(summary.get('input_ess_mean')),
            'input_ess_min': _json_float(summary.get('input_ess_min')),
            'input_max_normalized_weight': _json_float(summary.get('input_max_normalized_weight')),
            'input_low_ess_fraction': _json_float(summary.get('input_low_ess_fraction')),
            'input_mc_reliable': bool(summary.get('input_mc_reliable')),
            'aggregate_score_defined': bool(summary.get('aggregate_score_defined')),
            'panels': {panel: {'score': _json_float(value.get('score')),
                               'se': _json_float(value.get('se')),
                               'weight': value.get('weight'),
                               'normalized_weight': value.get('normalized_weight')}
                       for panel, value in summary.get('panels', {}).items()}}


def _validation_records(path, fit_key, epoch_number, step, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'at') as handle:
        for record in result['records']:
            compact = {'fit': fit_key, 'epoch': epoch_number, 'step': step,
                       'row': record['row'], 'sequence_id': record['sequence_id'],
                       'component': record['component'], 'panel': record['panel'],
                       'weight': record['weight'], 'sites': record['sites'],
                       'context_sites': record.get('context_sites'),
                       'nats_per_residue': record['nats_per_residue'],
                       'ess': record.get('ess'), 'max_normalized_weight': record.get('max_normalized_weight'),
                       'low_ess': record.get('low_ess'), 'finite': record.get('finite')}
            handle.write(json.dumps(compact, ensure_ascii=False, allow_nan=False) + '\n')


def _grouped_parameters(model):
    if model.arm != 'QVAE':
        return {}
    return {
        'posterior_output': list(model.posterior_head[-1].parameters()),
        'posterior_encoder_last': list(model.posterior_encoder.layers[-1].parameters()),
        'full_projection': list(model.full_projection.parameters()),
    }


def _diagnostic_queries(data, seed, count=27):
    """固定诊断输入，优先覆盖不同面板，便于 K 比较不依赖单一面板。"""
    queries = data.training_queries(0, seed)
    by_panel = {}
    for query in queries:
        by_panel.setdefault(query['panel'], query)
    selected = [by_panel[panel] for panel in sorted(by_panel)[:count]]
    chosen = {id(query) for query in selected}
    if len(selected) < min(count, len(queries)):
        for query in queries:
            if id(query) in chosen:
                continue
            selected.append(query)
            chosen.add(id(query))
            if len(selected) >= count:
                break
    return selected


def _flat_group_grads(loss, groups, retain_graph):
    names = list(groups)
    parameters = [parameter for name in names for parameter in groups[name]]
    grads = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    output = {}
    index = 0
    for name in names:
        parts = []
        for parameter in groups[name]:
            value = grads[index]
            index += 1
            parts.append(torch.zeros(parameter.numel(), device=parameter.device)
                         if value is None else value.detach().reshape(-1).float())
        output[name] = torch.cat(parts) if parts else torch.zeros(0, device=parameters[0].device)
    return output


def _gradient_metrics(stack, projections):
    stack = torch.stack(stack)
    mean = stack.mean(0)
    std = stack.std(0, unbiased=True) if len(stack) > 1 else torch.zeros_like(mean)
    repetitions = len(stack)
    signal = torch.linalg.vector_norm(mean)
    standard_error = torch.linalg.vector_norm(std) / math.sqrt(repetitions)
    snr = (signal / standard_error) if standard_error > 0 else (
        torch.tensor(float('inf'), device=mean.device) if signal > 0 else torch.tensor(0., device=mean.device))
    relative_error = (standard_error / signal) if signal > 0 else torch.tensor(float('inf'), device=mean.device)
    if repetitions > 1:
        half = repetitions // 2
        first = stack[:half].mean(0)
        second = stack[half:].mean(0)
        denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
        direction = (first @ second / denominator) if denominator > 0 else torch.tensor(float('nan'), device=mean.device)
    else:
        direction = torch.tensor(float('nan'), device=mean.device)
    projected = stack @ projections.T
    projected_mean = projected.mean(0)
    projected_se = projected.std(0, unbiased=True) / math.sqrt(repetitions) if repetitions > 1 else torch.zeros_like(projected_mean)
    projected_snr = torch.where(projected_se > 0, projected_mean.abs() / projected_se,
                                torch.where(projected_mean.abs() > 0,
                                            torch.full_like(projected_mean, float('inf')),
                                            torch.zeros_like(projected_mean)))
    sign_agreement = (torch.sign(projected) == torch.sign(projected_mean)[None]).float().mean(0)
    return {'mean_norm': float(signal.item()), 'se_norm': float(standard_error.item()),
            'snr': _json_float(snr.item()), 'relative_error': _json_float(relative_error.item()),
            'direction_stability': _json_float(direction.item()),
            'projections': [_json_float(value) for value in projected_mean.tolist()],
            'projection_snr': [_json_float(value) for value in projected_snr.tolist()],
            'projection_sign_agreement': [_json_float(value) for value in sign_agreement.tolist()]}


def _run_gradient_diagnostics(model, data, queries, plan, seed, point, step, device):
    """后验重构梯度、直接路径与 KL 的 SNR/方向/固定投影/耗时。"""
    if model.arm != 'QVAE':
        return {'point': point, 'step': int(step), 'not_applicable': True}
    calibration = plan['calibration']
    candidates = [int(value) for value in calibration['K_candidates']]
    repetitions = int(calibration['gradient_repetitions'])
    projection_count = int(calibration['gradient_projections'])
    batch_size = min(27, len(queries))
    selected = list(queries)[:batch_size]
    context, full, valid, mask, labels = data.batch(selected, device=device, need_full=True)
    weights = _fit_weights(data, selected, device)
    groups = _grouped_parameters(model)
    projection_generator = torch.Generator(device='cpu')
    projections = {}
    for name, parameters in groups.items():
        size = sum(parameter.numel() for parameter in parameters)
        projection_generator.manual_seed(stable_seed(seed, 'gradient_projection', name) % (2 ** 63 - 1))
        matrix = torch.randn(projection_count, size, generator=projection_generator)
        matrix = matrix / matrix.norm(dim=1, keepdim=True).clamp_min(_EPS)
        projections[name] = matrix.to(device)
    records = []
    model.train()
    for K in candidates:
        collected = {name: {'rloo': [], 'direct': [], 'kl': []} for name in groups}
        times = []
        for repetition in range(repetitions):
            generator = torch.Generator(device=('cuda' if device.type == 'cuda' else 'cpu'))
            generator.manual_seed(stable_seed(seed, 'gradient_sample', point, K, repetition) % (2 ** 63 - 1))
            start = time.monotonic()
            with _autocast(device):
                hidden = model.encode_context(context, valid, mask)
                targets, y, use, _ = target_tensors(hidden, mask, labels)
                prior = model.prior(hidden, valid, mask)
                posterior = model.posterior(full, hidden, valid, mask)
                codes = categorical_samples(posterior, K, generator)
                logq = categorical_log_prob(posterior, codes)
                nll = _decode_nll(model, targets, y, use, codes)
                kl = categorical_kl(posterior, prior)
                loss = rloo_objective(nll, logq, kl, 1., weights)
            loss.backward(retain_graph=True)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            times.append(time.monotonic() - start)
            model.zero_grad(set_to_none=True)
            # 用同一计算图重新求三条路径的梯度；rloo/direct/kl 分开记录。
            rloo_grads = _flat_group_grads(
                rloo_objective(nll, logq, kl, 0., weights), groups, retain_graph=True)
            direct_grads = _flat_group_grads(
                (nll.mean(0) * weights).mean(), groups, retain_graph=True)
            kl_grads = _flat_group_grads(
                (kl * weights).mean(), groups, retain_graph=False)
            for name in groups:
                collected[name]['rloo'].append(rloo_grads[name].cpu())
                collected[name]['direct'].append(direct_grads[name].cpu())
                collected[name]['kl'].append(kl_grads[name].cpu())
            del hidden, targets, prior, posterior, codes, logq, nll, kl, loss
        group_metrics = {}
        for name in groups:
            group_metrics[name] = {}
            for path in ('rloo', 'direct', 'kl'):
                group_metrics[name][path] = _gradient_metrics(
                    collected[name][path], projections[name].cpu())
        records.append({'point': point, 'step': int(step), 'K': int(K),
                        'repetitions': repetitions, 'queries': [q['row'] for q in selected],
                        'time_per_repetition': float(np.mean(times)),
                        'time_standard_deviation': float(np.std(times, ddof=1)) if len(times) > 1 else 0.,
                        'groups': group_metrics})
    model.eval()
    return {'point': point, 'step': int(step), 'records': records,
            'projection_count': projection_count, 'batch_queries': [q['row'] for q in selected]}


def _select_K(diagnostics, plan):
    calibration = plan['calibration']
    minimum_snr = float(calibration['signal_SNR_min'])
    tie_fraction = float(calibration['efficiency_tie_fraction'])
    default = int(calibration['default_K'])
    candidates = [int(value) for value in calibration['K_candidates']]
    summary = {}
    for K in candidates:
        relative_errors = []
        times = []
        snrs = []
        groups = 0
        for point in diagnostics:
            for record in point.get('records', []):
                if record['K'] != K:
                    continue
                times.append(record['time_per_repetition'])
                for metrics in record['groups'].values():
                    value = metrics['rloo']
                    snr = value.get('snr')
                    direction = value.get('direction_stability')
                    if (snr is not None and snr >= minimum_snr and
                            direction is not None and direction >= .5 and
                            value.get('relative_error') is not None and
                            math.isfinite(value['relative_error'])):
                        relative_errors.append(value['relative_error'])
                        snrs.append(snr)
                        groups += 1
        if relative_errors:
            mean_relative = float(np.mean(relative_errors))
            mean_time = float(np.mean(times))
            summary[str(K)] = {'mean_relative_error': mean_relative,
                               'mean_time_per_repetition': mean_time,
                               'mean_snr': float(np.mean(snrs)),
                               'signal_groups': groups,
                               'score': mean_relative * mean_time}
        else:
            summary[str(K)] = {'mean_relative_error': None,
                               'mean_time_per_repetition': float(np.mean(times)) if times else None,
                               'mean_snr': None, 'signal_groups': 0, 'score': None}
    valid = {int(K): value for K, value in summary.items() if value['score'] is not None}
    if not valid:
        return default, {'evidence': 'indistinguishable', 'chosen': default,
                         'signal_SNR_min': minimum_snr, 'direction_stability_min': .5,
                         'efficiency_tie_fraction': tie_fraction, 'ranking': summary,
                         'note': 'no parameter group reached the SNR/direction threshold; use default K'}
    best_score = min(value['score'] for value in valid.values())
    eligible = [K for K, value in valid.items() if value['score'] <= best_score * (1. + tie_fraction)]
    chosen = min(eligible)
    return chosen, {'evidence': 'signal_detected', 'chosen': chosen,
                    'signal_SNR_min': minimum_snr, 'direction_stability_min': .5,
                    'efficiency_tie_fraction': tie_fraction,
                    'best_score': best_score, 'eligible_K': eligible, 'ranking': summary}


def _calibration_queries(data, count=3):
    by_row = {}
    for query in data.validation:
        by_row.setdefault(int(query['row']), []).append(query)
    rows = []
    for row, queries in by_row.items():
        first = data.rows.iloc[row]
        rows.append((str(first.component_v2), str(first.sequence_id), row, queries))
    rows.sort(key=lambda item: (item[0], item[1]))
    selected_components = []
    selected_sequence_ids = []
    selected = []
    for component, sequence_id, row, queries in rows:
        if component in selected_components:
            continue
        selected_components.append(component)
        selected_sequence_ids.append(sequence_id)
        selected.extend(queries)
        if len(selected_components) == count:
            break
    if not selected:
        raise ValueError('No validation queries for calibration')
    panels = {query['panel'] for query in selected}
    if len(panels) != 27:
        raise ValueError(f'Calibration inputs must cover 27 panels, got {len(panels)}')
    selected.sort(key=lambda query: (query['panel'], query['row']))
    return selected, {'components': selected_components,
                      'sequence_ids': selected_sequence_ids,
                      'queries': len(selected), 'panels': len(panels)}


def _query_hash(queries, data):
    payload = []
    for query in queries:
        row = data.rows.iloc[query['row']]
        payload.append({'sequence_id': str(row.sequence_id), 'panel': query['panel'],
                        'positions': list(query['positions']),
                        'subset_positions': list(query.get('subset_positions', []))})
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _calibration_fit(run, plan, data, root, device, arm, learning_rate, fixed_queries,
                     attempts, failures, save_records):
    fit_key = f'calibration/{arm}/lr{learning_rate:g}'
    seed = int(plan['calibration']['seed'])
    config = {'learning_rate': float(learning_rate),
              'weight_decay': float(plan['search_weight_decay'][0]),
              'K': int(plan['calibration']['default_K'])}
    record = _run_fit(run, plan, data, root, device, fit_key, 'calibration', arm, seed,
                      config, int(plan['calibration']['max_steps']), fixed_queries,
                      int(plan['calibration']['likelihood_counts'][0]), 4, None,
                      attempts, failures, save_records=save_records, diagnostics=True,
                      diagnostic_seed=seed)
    return record


def _calibrate(run, plan, data, root, device, attempts, failures, save_records):
    raw = root / 'data/results' / run.name / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    status(run, 'CALIBRATING', fits=len(plan['arms']) * len(plan['calibration']['rates']))
    fixed_queries, input_receipt = _calibration_queries(data)
    write_json(raw / 'calibration_inputs.json',
               {**input_receipt, 'sha256': _query_hash(fixed_queries, data)})
    fits = {}
    for arm in plan['arms']:
        for learning_rate in plan['calibration']['rates']:
            record = _calibration_fit(run, plan, data, root, device, arm,
                                      float(learning_rate), fixed_queries, attempts,
                                      failures, save_records)
            if record is not None:
                fits[(arm, float(learning_rate))] = record
    # 每个校准最佳权重上做 64/256/1024/4096 样本的似然校准。
    likelihood = {}
    for (arm, learning_rate), record in fits.items():
        if record.get('best_path') is None:
            continue
        model = _load_model_for_evaluation(record, plan, data, device)
        evaluations = {}
        if arm == 'QVAE':
            for sample_count in plan['calibration']['likelihood_counts']:
                result = posterior_joint_probability(
                    model, data, fixed_queries, sample_count=int(sample_count),
                    repeats=int(plan['calibration']['likelihood_repetitions']),
                    seed=stable_seed(plan['calibration']['seed'], 'likelihood', arm, learning_rate, sample_count),
                    batch_size=8)
                evaluations[int(sample_count)] = result
                write_json(raw / 'calibration' /
                           f'likelihood_{arm}_lr{learning_rate:g}_{sample_count}.json',
                           {'summary': result['summary'], 'records': result['records']})
        else:
            from .conditional_evaluation import deterministic_joint_probability
            result = deterministic_joint_probability(model, data, fixed_queries)
            evaluations['exact'] = result
            write_json(raw / 'calibration' / f'likelihood_{arm}_lr{learning_rate:g}_exact.json',
                       {'summary': result['summary'], 'records': result['records']})
        likelihood[(arm, learning_rate)] = evaluations
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    base_lr = {}
    base_lr_scores = {}
    base_lr_tolerance = {}
    for arm in plan['arms']:
        candidates = {lr: values for (candidate_arm, lr), values in likelihood.items()
                      if candidate_arm == arm}
        if not candidates:
            raise FitFailed(f'No calibration likelihood for arm {arm}')
        scores = {}
        for lr, values in candidates.items():
            if arm == 'QVAE':
                result = values.get(int(plan['calibration']['likelihood_counts'][-1]))
            else:
                result = values.get('exact')
            scores[lr] = result['summary']['score'] if result else None
        base_lr_scores[arm] = {str(lr): _json_float(score) for lr, score in scores.items()}
        finite = {lr: score for lr, score in scores.items() if score is not None}
        if not finite:
            raise FitFailed(f'No finite calibration score for arm {arm}')
        ordered = sorted(finite.items(), key=lambda item: item[1])
        best_lr, best_score = ordered[0]
        tolerance = 1e-4
        if len(ordered) > 1:
            other_lr, other_score = ordered[1]
            if arm == 'QVAE':
                result = candidates[best_lr].get(int(plan['calibration']['likelihood_counts'][-1]))
                tolerance = max(tolerance, 2 * (result['summary'].get('se') or 0.))
            if abs(other_score - best_score) <= tolerance:
                best_lr = min(best_lr, other_lr)
        base_lr[arm] = float(best_lr)
        base_lr_tolerance[arm] = float(tolerance)
    K_choice = {}
    K_decision = {}
    likelihood_decision = {}
    for arm in plan['arms']:
        selected_lr = base_lr[arm]
        record = fits.get((arm, selected_lr))
        diagnostics = record.get('gradient_diagnostics', []) if record else []
        if not diagnostics:
            other = [fit for (candidate_arm, _), fit in fits.items()
                     if candidate_arm == arm and fit.get('gradient_diagnostics')]
            diagnostics = other[0]['gradient_diagnostics'] if other else []
        K_choice[arm], K_decision[arm] = _select_K(diagnostics, plan)
        values = likelihood.get((arm, selected_lr), {})
        likelihood_decision[arm] = _summarize_likelihood(values, plan, arm)
    base_lr_evidence = {
        arm: {'scores': base_lr_scores.get(arm, {}), 'selected': base_lr[arm],
              'tolerance': base_lr_tolerance.get(arm, 1e-4),
              'evaluation_stable': likelihood_decision[arm].get('evaluation_stable'),
              'evidence': 'stable' if likelihood_decision[arm].get('evaluation_stable') else 'UNKNOWN'}
        for arm in plan['arms']}
    best_steps = [record['best_step'] for record in fits.values() if record.get('best_step') is not None]
    if not best_steps:
        raise FitFailed('Calibration produced no confirmed best step')
    T = max(best_steps)
    B = budget_steps(plan, T)
    decision = {
        'calibration_seed': int(plan['calibration']['seed']),
        'calibration_steps_per_fit_max': int(plan['calibration']['max_steps']),
        'calibration_inputs': {**input_receipt, 'sha256': _query_hash(fixed_queries, data)},
        'base_learning_rate': base_lr,
        'base_learning_rate_evidence': base_lr_evidence,
        'K': K_choice,
        'K_decision': K_decision,
        'T': int(T),
        'B': int(B),
        'budget_rule': 'min(absolute_max, max(5200, 4*T))',
        'likelihood': likelihood_decision,
        'selection_metric': 'panel_q_weighted_component_balanced_nll_per_residue',
        'selection_metric_sign': 'lower_is_better',
        'provisional_calibration_selection': {
            'sample_count': int(plan['calibration']['likelihood_counts'][0]),
            'repeats': 4,
            'delta_rule': 'max(1e-4, 2*SE_current)',
            'note': 'used only to locate calibration best steps; final delta is frozen below',
        },
        'search_rule': 'each arm selects the configuration with the lowest two-seed mean best validation score; within selection_delta choose lower lr multiplier then larger weight decay',
        'formal_rule': 'run 5 paired seeds from scratch with the selected configuration',
    }
    write_json(raw / 'calibration_decision.json', decision)
    interface = {
        'task': decision['task'] if 'task' in decision else plan['task'],
        'git_revision': _git_revision(root),
        'full_decision': {'path': str((raw / 'calibration_decision.json').relative_to(root)),
                          'sha256': sha256_file(raw / 'calibration_decision.json')},
        'base_learning_rates': base_lr,
        'max_steps_per_fit': int(B),
        'arms': {},
    }
    for arm in plan['arms']:
        values = likelihood_decision[arm]
        interface['arms'][arm] = {
            'base_learning_rate': float(base_lr[arm]),
            'posterior_sample_count': int(K_choice[arm]) if arm == 'QVAE' else 0,
            'selection_sample_count': int(values['selection_sample_count']),
            'evaluation_sample_count': int(values['evaluation_sample_count']),
            'validation_delta': float(values['selection_delta']),
            'evaluation_delta': float(values['evaluation_delta']),
            'max_steps_per_fit': int(B),
            'selection_stable': bool(values['selection_stable']),
            'evaluation_stable': bool(values.get('evaluation_stable', False)),
            'exact': bool(values.get('exact', False)),
            'selected_metric': decision['selection_metric'],
        }
    write_json(run / 'calibration_decision.json', interface)
    status(run, 'CALIBRATION_READY', base_learning_rate=base_lr, K=K_choice, T=int(T), B=int(B),
           likelihood={arm: {'selection_sample_count': likelihood_decision[arm]['selection_sample_count'],
                              'evaluation_sample_count': likelihood_decision[arm]['evaluation_sample_count'],
                              'selection_delta': likelihood_decision[arm]['selection_delta'],
                              'evaluation_delta': likelihood_decision[arm]['evaluation_delta'],
                              'selection_stable': likelihood_decision[arm]['selection_stable']}
                       for arm in likelihood_decision})
    return {'decision': decision, 'fits': fits, 'likelihood': likelihood,
            'fixed_queries': fixed_queries, 'inputs': input_receipt}


def _summarize_likelihood(values, plan, arm):
    tolerance = float(plan['calibration']['likelihood_tolerance_per_residue'])
    if arm != 'QVAE':
        result = values.get('exact')
        if result is None:
            raise FitFailed('Missing deterministic likelihood calibration')
        return {'exact': True, 'selection_sample_count': 0, 'evaluation_sample_count': 0,
                'selection_delta': 1e-4, 'evaluation_delta': 1e-4,
                'selection_stable': True, 'evaluation_stable': True,
                'selection_stability': 'exact',
                'unstable_panels': [], 'unstable_panels_at_selection': [],
                'unstable_panels_at_evaluation': [], 'unstable_at_all_scales': [],
                'counts': {}, 'summary': _compact_summary(result['summary'])}
    counts = sorted(int(key) for key in values)
    if not counts:
        raise FitFailed('Missing QVAE likelihood calibration')
    summaries = {count: values[count]['summary'] for count in counts}
    reference = summaries[max(counts)]
    stable_counts = []
    stable_panels_by_count = {}
    nested = {}
    all_panels = set()
    for count in counts:
        summary = summaries[count]
        se = summary.get('se')
        score = summary.get('score')
        difference = None if score is None or reference.get('score') is None else score - reference['score']
        nested[str(count)] = {'score_difference_from_largest': _json_float(difference),
                              'note': 'nested_sample_log_difference; not an exact log p bias estimate'}
        stable = (summary.get('finite') and se is not None and 2 * se <= tolerance and
                  difference is not None and abs(difference) <= tolerance)
        if stable:
            stable_counts.append(count)
        stable_panels = set()
        for panel, value in summary.get('panels', {}).items():
            all_panels.add(panel)
            panel_se = value.get('se')
            reference_panel = reference.get('panels', {}).get(panel, {})
            panel_difference = (None if value.get('score') is None or reference_panel.get('score') is None
                                else value['score'] - reference_panel['score'])
            if (panel_se is not None and 2 * panel_se <= tolerance and
                    panel_difference is not None and abs(panel_difference) <= tolerance):
                stable_panels.add(panel)
        stable_panels_by_count[count] = stable_panels
    selection_count = min(stable_counts) if stable_counts else max(counts)
    selection_stable = bool(stable_counts)
    selection_summary = summaries[selection_count]
    evaluation_count = max(counts)
    evaluation_summary = summaries[evaluation_count]
    evaluation_stable = evaluation_count in stable_counts
    selection_unstable = sorted(all_panels - stable_panels_by_count[selection_count])
    evaluation_unstable = sorted(all_panels - stable_panels_by_count[evaluation_count])
    all_scale_unstable = sorted(panel for panel in all_panels
                                if all(panel not in stable_panels_by_count[count] for count in counts))
    selection_delta = max(1e-4, 2 * (selection_summary.get('se') or 0.))
    evaluation_delta = max(1e-4, 2 * (evaluation_summary.get('se') or 0.))
    return {'exact': False, 'selection_sample_count': int(selection_count),
            'evaluation_sample_count': int(evaluation_count),
            'selection_delta': float(selection_delta),
            'evaluation_delta': float(evaluation_delta),
            'selection_stable': selection_stable,
            'evaluation_stable': evaluation_stable,
            'selection_stability': 'smallest count with 2*SE<=tolerance and |score-score_4096|<=tolerance'
                                   if selection_stable else 'UNKNOWN: no count met the aggregate stability rule',
            'tolerance': tolerance,
            'unstable_panels': evaluation_unstable,
            'unstable_panels_at_selection': selection_unstable,
            'unstable_panels_at_evaluation': evaluation_unstable,
            'unstable_at_all_scales': all_scale_unstable,
            'nested_sample_log_difference': nested,
            'counts': {str(count): {'score': _json_float(summaries[count].get('score')),
                                    'se': _json_float(summaries[count].get('se')),
                                    'input_ess_mean': _json_float(summaries[count].get('input_ess_mean')),
                                    'input_ess_min': _json_float(summaries[count].get('input_ess_min')),
                                    'input_max_normalized_weight': _json_float(summaries[count].get('input_max_normalized_weight')),
                                    'input_low_ess_fraction': _json_float(summaries[count].get('input_low_ess_fraction')),
                                    'input_mc_reliable': bool(summaries[count].get('input_mc_reliable'))}
                       for count in counts},
            'summary': _compact_summary(selection_summary)}


def _load_model_for_evaluation(record, plan, data, device):
    model_config = record['model_config']
    model = ConditionalModel(record['arm'], **model_config).to(device)
    path = record.get('best_absolute_path') or record['best_path']
    payload = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval()
    return model


def _run_fit(run, plan, data, root, device, fit_key, phase, arm, seed, config, max_steps,
             validation_queries, sample_count, repeats, delta, attempts, failures,
             save_records, diagnostics=False, diagnostic_seed=None):
    if int(attempts['count']) >= int(plan['budget']['max_fits']):
        raise BudgetExceeded('Authorized fit count exhausted')
    attempts['count'] = int(attempts['count']) + 1
    started = time.monotonic()
    try:
        record = fit_model(run, plan, data, root, device, fit_key, phase, arm, seed,
                           config, max_steps, validation_queries, sample_count,
                           repeats, delta, save_records=save_records,
                           diagnostics=diagnostics, diagnostic_seed=diagnostic_seed)
        record['attempt'] = attempts['count']
        if record.get('best_path') is None or record.get('best_score') is None:
            failures.append({'fit': fit_key, 'reason': 'no_confirmed_best',
                             'attempt': attempts['count'], 'seconds': time.monotonic() - started})
        return record
    except Exception as exc:
        failure = {'fit': fit_key, 'phase': phase, 'arm': arm, 'seed': int(seed),
                   'attempt': attempts['count'], 'reason': repr(exc),
                   'traceback': traceback.format_exc(),
                   'seconds': time.monotonic() - started}
        failures.append(failure)
        _append_jsonl(root / 'data/results' / run.name / 'raw' / 'failures.jsonl', failure)
        return None


def fit_model(run, plan, data, root, device, fit_key, phase, arm, seed, config,
              max_steps, validation_queries, sample_count, repeats, delta,
              save_records=True, diagnostics=False, diagnostic_seed=None):
    """一个从头拟合；保存 best/last、曲线与完整验证记录。"""
    started = time.monotonic()
    _seed_all(seed)
    model_config = {'hidden': int(plan['model']['hidden']),
                    'factors': int(plan['model']['factors']),
                    'states': int(plan['model']['states']),
                    'feature_dim': int(getattr(data, 'feature_dim',
                                               getattr(data, 'features', np.zeros((1, 1280))).shape[1]))}
    model = ConditionalModel(arm, **model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['learning_rate']),
                                  weight_decay=float(config['weight_decay']))
    batch_size = int(plan['batch_size'])
    train_rows = len(data.weights)
    steps_per_epoch = max(1, math.ceil(train_rows / batch_size))
    lr_warmup = int(plan['learning_rate_warmup_epochs'])
    beta_warmup = int(plan['beta_warmup_epochs'])
    lr_patience = int(plan['lr_patience'])
    stop_patience = int(plan['stop_patience'])
    state = initial_selection_state()
    history = []
    diagnostic_records = []
    diagnostics_errors = []
    best_path, last_path = _checkpoint_paths(root, run.name, fit_key)
    raw = root / 'data/results' / run.name / 'raw'
    curve_path = raw / 'curves' / f'{fit_key.replace("/", "_")}.jsonl'
    validation_path = raw / 'validation_records' / f'{fit_key.replace("/", "_")}.jsonl.gz'
    if validation_path.exists():
        validation_path.unlink()
    step = 0
    epoch_number = 0
    stop_reason = None
    training_seconds = 0.
    validation_seconds = 0.
    diagnostics_seconds = 0.
    diagnostic_steps = set()
    diagnostic_attempted = set()
    last_saved = False
    diagnostic_queries = _diagnostic_queries(data, seed) if diagnostics else []
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    model.train()

    def diagnostics_at(point):
        nonlocal diagnostics_seconds
        if not diagnostics:
            return
        diagnostic_attempted.add(point)
        try:
            started_diag = time.monotonic()
            result = _run_gradient_diagnostics(model, data, diagnostic_queries,
                                               plan, diagnostic_seed or seed, point, step, device)
            diagnostics_seconds += time.monotonic() - started_diag
            diagnostic_records.append(result)
            diagnostic_steps.add(step)
        except Exception as exc:
            diagnostics_errors.append({'point': point, 'step': step, 'error': repr(exc),
                                       'traceback': traceback.format_exc()})
        model.train()

    try:
        diagnostics_at('init')
        max_epochs = math.ceil(max_steps / steps_per_epoch) + 2
        while step < max_steps and stop_reason is None and epoch_number < max_epochs:
            epoch_number += 1
            beta = beta_at_epoch(epoch_number, beta_warmup)
            learning_rate = learning_rate_at_epoch(float(config['learning_rate']),
                                                   epoch_number, lr_warmup, state['lr_factor'])
            for group in optimizer.param_groups:
                group['lr'] = learning_rate
            queries = data.training_queries(epoch_number - 1, seed)
            if len(queries) != train_rows:
                raise ValueError('Training query count differs from weight population')
            epoch_loss = 0.
            epoch_samples = 0
            reconstruction_numerator = 0.
            reconstruction_denominator = 0.
            kl_numerator = 0.
            kl_denominator = 0.
            train_started = time.monotonic()
            for start in range(0, len(queries), batch_size):
                if step >= max_steps:
                    break
                batch = queries[start:start+batch_size]
                weights = _fit_weights(data, batch, device)
                generator = torch.Generator(device=('cuda' if device.type == 'cuda' else 'cpu'))
                generator.manual_seed(stable_seed(seed, 'latent', step) % (2 ** 63 - 1))
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = _train_step(model, data, batch, weights, beta,
                                            generator, int(config['K']), device)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite training objective at step {step + 1}')
                loss.backward()
                norm = clip_grad_norm_(model.parameters(), float(plan['gradient_clip']))
                if not torch.isfinite(norm):
                    raise FloatingPointError(f'Nonfinite gradient at step {step + 1}')
                optimizer.step()
                step += 1
                epoch_loss += float(loss.detach().item()) * len(batch)
                epoch_samples += len(batch)
                reconstruction_numerator += metrics['reconstruction_numerator']
                reconstruction_denominator += metrics['reconstruction_denominator']
                if metrics['kl_numerator'] is not None:
                    kl_numerator += metrics['kl_numerator']
                    kl_denominator += metrics['kl_denominator']
                if (diagnostics and 'midpoint' not in diagnostic_attempted
                        and step >= max_steps // 2):
                    diagnostics_at('midpoint')
            training_seconds += time.monotonic() - train_started
            if epoch_samples == 0:
                break
            beta_ready = epoch_number >= beta_warmup
            beta_ready_diagnostics = beta_ready and 'beta_ready' not in diagnostic_attempted
            if beta_ready and beta_ready_diagnostics:
                diagnostics_at('beta_ready')
            validation_summary = None
            validation_result = None
            if beta_ready:
                validation_started = time.monotonic()
                try:
                    validation_summary, validation_result = validation_score(
                        model, data, validation_queries, sample_count=sample_count,
                        repeats=repeats, seed=stable_seed(seed, 'validation', epoch_number),
                        batch_size=16 if arm == 'QVAE' else 128)
                finally:
                    validation_seconds += time.monotonic() - validation_started
                validation_summary = _compact_summary(validation_summary)
                if save_records:
                    _validation_records(validation_path, fit_key, epoch_number, step, validation_result)
                score = validation_summary.get('score')
                effective_delta = (max(1e-4, 2 * (validation_summary.get('se') or 0.))
                                   if delta is None else float(delta))
                previous_best = state['best_step']
                state, improved = update_selection(
                    state, score, step, epoch_number, effective_delta, True,
                    lr_patience=lr_patience, stop_patience=stop_patience)
                if improved and state['best_step'] != previous_best:
                    payload = checkpoint_payload(
                        model, optimizer, arm, seed, model_config, step, epoch_number,
                        state['best'], config, include_optimizer=False)
                    _torch_save(best_path, payload)
            else:
                effective_delta = None
            history.append({
                'fit': fit_key, 'phase': phase, 'arm': arm, 'seed': int(seed),
                'epoch': epoch_number, 'step': step,
                'learning_rate': learning_rate, 'beta': beta,
                'train_loss': epoch_loss / epoch_samples,
                'train_reconstruction_per_residue': (
                    reconstruction_numerator / reconstruction_denominator
                    if reconstruction_denominator else None),
                'train_reconstruction_numerator': reconstruction_numerator,
                'train_reconstruction_denominator': reconstruction_denominator,
                'train_kl_per_factor': (kl_numerator / kl_denominator
                                        if kl_denominator else None),
                'train_kl_numerator': kl_numerator,
                'train_kl_denominator': kl_denominator,
                'validation': validation_summary,
                'effective_delta': _json_float(effective_delta),
                'best': _json_float(state['best']), 'best_step': state['best_step'],
                'since_improve': state['since_improve'], 'lr_factor': state['lr_factor'],
                'lr_reductions': state['lr_reductions'],
                'stop_reason': state['stop_reason'],
            })
            _append_jsonl(curve_path, history[-1])
            print(json.dumps({'fit': fit_key, 'epoch': epoch_number, 'step': step,
                              'train_loss': history[-1]['train_loss'],
                              'validation': validation_summary.get('score') if validation_summary else None,
                              'best': state['best'], 'lr': learning_rate,
                              'stale': state['since_improve'], 'seconds': time.monotonic() - started},
                             ensure_ascii=False), flush=True)
            if step >= max_steps:
                stop_reason = 'budget_truncated'
            elif state['stop_reason']:
                stop_reason = state['stop_reason']
        if stop_reason is None:
            stop_reason = 'budget_truncated' if step >= max_steps else 'completed'
        last_payload = checkpoint_payload(
            model, optimizer, arm, seed, model_config, step, epoch_number,
            state['best'], config, include_optimizer=True)
        _torch_save(last_path, last_payload)
        last_saved = True
        last_rng = last_payload['rng']
        if diagnostics and best_path.exists() and state['best_step'] not in diagnostic_steps:
            payload = torch.load(best_path, map_location='cpu', weights_only=True)
            model.load_state_dict(payload['state_dict'], strict=True)
            diagnostics_at('best')
            # last 已保存为训练末态；best 诊断只读取权重，不回写 last。
    except Exception:
        try:
            if not last_saved:
                _torch_save(last_path, checkpoint_payload(
                    model, optimizer, arm, seed, model_config, step, epoch_number,
                    state.get('best'), config, include_optimizer=True))
        except Exception:
            pass
        raise
    finally:
        history_path = raw / 'curves' / f'{fit_key.replace("/", "_")}.json'
        write_json(history_path, {'fit': fit_key, 'history': history})
    record = {
        'fit': fit_key, 'phase': phase, 'arm': arm, 'seed': int(seed),
        'config': config, 'model_config': model_config,
        'max_steps': int(max_steps), 'steps': int(step), 'epochs': int(epoch_number),
        'steps_per_epoch': int(steps_per_epoch),
        'stop_reason': stop_reason, 'best_step': state['best_step'],
        'best_epoch': state['best_epoch'], 'best_score': _json_float(state['best']),
        'since_improve': state['since_improve'], 'lr_factor': state['lr_factor'],
        'lr_reductions': state['lr_reductions'],
        'best_path': str(best_path) if best_path.exists() else None,
        'best_sha256': sha256_file(best_path) if best_path.exists() else None,
        'last_path': str(last_path) if last_path.exists() else None,
        'last_sha256': sha256_file(last_path) if last_path.exists() else None,
        'training_seconds': training_seconds, 'validation_seconds': validation_seconds,
        'diagnostics_seconds': diagnostics_seconds,
        'total_seconds': time.monotonic() - started,
        'steps_per_second': step / max(time.monotonic() - started, _EPS),
        'max_memory_mib': (torch.cuda.max_memory_allocated(device) / 2 ** 20
                           if device.type == 'cuda' else 0.),
        'gradient_diagnostics': diagnostic_records,
        'gradient_diagnostics_errors': diagnostics_errors,
        'diagnostic_queries': [{'row': query['row'], 'panel': query['panel'],
                                'positions': query['positions']}
                               for query in diagnostic_queries],
        'history_path': str(curve_path),
        'validation_records_path': str(validation_path) if save_records else None,
        'rng_state': last_rng if last_saved else rng_state(),
    }
    record['best_absolute_path'] = str(best_path)
    record['last_absolute_path'] = str(last_path)
    for key, path in (('best_path', best_path), ('last_path', last_path)):
        if record[key]:
            try:
                record[key] = str(Path(record[key]).relative_to(root))
            except ValueError:
                record[key] = str(Path(record[key]))
    write_json(raw / 'fits' / f'{fit_key.replace("/", "_")}.json', record)
    return record


def _search(run, plan, data, root, device, calibration, attempts, failures, save_records):
    decision = calibration['decision']
    raw = root / 'data/results' / run.name / 'raw'
    status(run, 'SEARCHING')
    search_records = []
    for arm in plan['arms']:
        base_lr = float(decision['base_learning_rate'][arm])
        for multiplier in plan['search_rate_multipliers']:
            for weight_decay in plan['search_weight_decay']:
                config = {'learning_rate': base_lr * float(multiplier),
                          'weight_decay': float(weight_decay), 'K': int(decision['K'][arm]),
                          'multiplier': float(multiplier)}
                for seed in plan['selection_seeds']:
                    fit_key = (f'search/{arm}/x{float(multiplier):g}_wd{float(weight_decay):g}'
                               f'/seed{seed}')
                    record = _run_fit(
                        run, plan, data, root, device, fit_key, 'search', arm, int(seed),
                        config, int(decision['B']), data.validation,
                        int(decision['likelihood'][arm]['selection_sample_count']), 1,
                        float(decision['likelihood'][arm]['selection_delta']),
                        attempts, failures, save_records)
                    if record is not None:
                        search_records.append(record)
    write_json(raw / 'search_fits.json', search_records)
    selected = {}
    for arm in plan['arms']:
        candidates = {}
        for record in search_records:
            if record['arm'] != arm or record.get('best_score') is None:
                continue
            key = (float(record['config']['multiplier']), float(record['config']['weight_decay']))
            candidates.setdefault(key, []).append(record)
        if not candidates:
            raise FitFailed(f'No successful search fit for arm {arm}')
        means = {key: float(np.mean([item['best_score'] for item in items]))
                 for key, items in candidates.items()}
        best_mean = min(means.values())
        tolerance = float(decision['likelihood'][arm]['selection_delta'])
        eligible = {key: value for key, value in means.items() if value <= best_mean + tolerance}
        chosen_key = min(eligible, key=lambda key: (key[0], -key[1]))
        chosen_records = candidates[chosen_key]
        selected[arm] = {
            'arm': arm, 'multiplier': chosen_key[0], 'weight_decay': chosen_key[1],
            'learning_rate': float(decision['base_learning_rate'][arm]) * chosen_key[0],
            'K': int(decision['K'][arm]),
            'mean_best_score': means[chosen_key],
            'seed_scores': {str(item['seed']): item['best_score'] for item in chosen_records},
            'seed_steps': {str(item['seed']): item['best_step'] for item in chosen_records},
            'selection_tolerance': tolerance,
            'candidates': {str(key): value for key, value in means.items()},
        }
    write_json(raw / 'search_selection.json', selected)
    status(run, 'SEARCH_READY', selected={arm: {'multiplier': value['multiplier'],
                                                'weight_decay': value['weight_decay'],
                                                'mean_best_score': value['mean_best_score']}
                                          for arm, value in selected.items()})
    return selected, search_records


def _formal(run, plan, data, root, device, calibration, selected, attempts, failures,
            save_records):
    decision = calibration['decision']
    raw = root / 'data/results' / run.name / 'raw'
    status(run, 'FORMAL_TRAINING')
    formal = {}
    formal_records = []
    for arm in plan['arms']:
        config = {'learning_rate': float(selected[arm]['learning_rate']),
                  'weight_decay': float(selected[arm]['weight_decay']),
                  'K': int(selected[arm]['K']),
                  'multiplier': float(selected[arm]['multiplier'])}
        for seed in plan['formal_seeds']:
            fit_key = f'formal/{arm}/seed{seed}'
            record = _run_fit(
                run, plan, data, root, device, fit_key, 'formal', arm, int(seed),
                config, int(decision['B']), data.validation,
                int(decision['likelihood'][arm]['selection_sample_count']), 1,
                float(decision['likelihood'][arm]['selection_delta']),
                attempts, failures, save_records)
            if record is None or record.get('best_path') is None:
                continue
            formal_records.append(record)
            best_path = root / record['best_path']
            last_path = root / record['last_path']
            formal[f'{arm}/{seed}'] = {
                'path': record['best_path'], 'sha256': sha256_file(best_path),
                'arm': arm, 'seed': int(seed), 'model_config': record['model_config'],
                'selected_metric': _json_float(record['best_score']),
                'selected_metric_name': 'panel_q_weighted_component_balanced_nll_per_residue',
                'selection_sample_count': int(decision['likelihood'][arm]['selection_sample_count']),
                'selection_delta': float(decision['likelihood'][arm]['selection_delta']),
                'step': int(record['best_step']), 'epoch': int(record['best_epoch']),
                'last_path': record['last_path'], 'last_sha256': sha256_file(last_path),
                'config': config, 'steps': int(record['steps']),
                'stop_reason': record['stop_reason'],
            }
    write_json(raw / 'formal_fits.json', formal_records)
    write_json(raw / 'formal_manifest.json', formal)
    expected = {f'{arm}/{seed}' for arm in plan['arms'] for seed in plan['formal_seeds']}
    missing = sorted(expected - set(formal))
    if missing:
        raise FitFailed('Formal fits missing: ' + ', '.join(missing))
    status(run, 'FORMAL_READY', checkpoints=len(formal))
    return formal, formal_records


_FORMAL_METHODS = ('shared_joint', 'marginal_product', 'deterministic',
                   'marginal_minus_shared', 'deterministic_minus_shared',
                   'marginal_minus_deterministic')
_POSTERIOR_METHODS = ('posterior_is_joint', 'posterior_minus_shared',
                      'posterior_minus_marginal', 'posterior_minus_deterministic')
_QVAE_METHODS = _FORMAL_METHODS + _POSTERIOR_METHODS


def _formal_difference(left, right):
    output = []
    for a, b in zip(left, right):
        output.append(None if a is None or b is None else float(a) - float(b))
    return output


def _summary_for_values(records, data, value_key, ess_key='posterior_ess',
                        max_key='posterior_max_normalized_weight', low_key='posterior_low_ess'):
    values = [{'row': record['row'], 'component': record['component'],
               'panel': record['panel'], 'weight': record['weight'],
               'nats_per_residue': record[value_key],
               'ess': record.get(ess_key), 'max_normalized_weight': record.get(max_key),
               'low_ess': record.get(low_key)}
              for record in records]
    return aggregate_panel_records(values, data)


def _formal_eval_paths(root, run_name, key):
    safe = key.replace('/', '_')
    base = root / 'data/results' / run_name / 'raw'
    return (base / 'formal_evaluation' / f'{safe}.jsonl.gz',
            base / 'formal_evaluation' / f'{safe}_summary.json',
            base / 'formal_evaluation_receipts' / f'{safe}.json')


def _load_formal_model(entry, plan, root, device):
    path = root / entry['path']
    if sha256_file(path) != entry['sha256']:
        raise ValueError('Formal checkpoint identity differs: ' + str(path))
    saved = torch.load(path, map_location='cpu', weights_only=True)
    arm = saved.get('arm', entry['arm'])
    if arm != entry['arm']:
        raise ValueError('Formal checkpoint arm differs')
    model_config = saved.get('model_config', entry.get('model_config', plan['model']))
    model = ConditionalModel(arm, **model_config).to(device).eval()
    model.load_state_dict(saved['state_dict'], strict=True)
    return model


def _compact_formal_summary(summary):
    return {'score': _json_float(summary.get('score')), 'se': _json_float(summary.get('se')),
            'finite': bool(summary.get('finite')),
            'components': {component: _json_float(value)
                           for component, value in summary.get('components', {}).items()},
            'panels': {panel: {'score': _json_float(value.get('score')),
                               'se': _json_float(value.get('se'))}
                       for panel, value in summary.get('panels', {}).items()},
            'input_ess_mean': _json_float(summary.get('input_ess_mean')),
            'input_ess_min': _json_float(summary.get('input_ess_min')),
            'input_max_normalized_weight': _json_float(summary.get('input_max_normalized_weight')),
            'input_low_ess_fraction': _json_float(summary.get('input_low_ess_fraction')),
            'input_mc_reliable': bool(summary.get('input_mc_reliable')),
            'aggregate_score_defined': bool(summary.get('aggregate_score_defined'))}


def _write_formal_records(path, key, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt') as handle:
        for record in records:
            handle.write(json.dumps({'fit': key, **record}, ensure_ascii=False,
                                    allow_nan=False) + '\n')


def _formal_receipt(root, receipt_path, checkpoint_sha, paired_sha, sample_count,
                    repeats, query_hash):
    if not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text())
    expected = {'checkpoint_sha256': checkpoint_sha, 'paired_deterministic_sha256': paired_sha,
                'sample_count': int(sample_count), 'repeats': int(repeats),
                'query_hash': query_hash}
    if any(receipt.get(key) != value for key, value in expected.items()):
        return None
    for kind in ('records', 'summary'):
        path = root / receipt[f'{kind}_path']
        if not path.exists() or sha256_file(path) != receipt[f'{kind}_sha256']:
            return None
    return receipt


def _write_formal_receipt(path, root, key, checkpoint_sha, paired_sha, sample_count,
                          repeats, query_hash, records_path, summary_path):
    write_json(path, {'fit': key, 'checkpoint_sha256': checkpoint_sha,
                      'paired_deterministic_sha256': paired_sha,
                      'sample_count': int(sample_count), 'repeats': int(repeats),
                      'query_hash': query_hash,
                      'records_path': str(records_path.relative_to(root)),
                      'records_sha256': sha256_file(records_path),
                      'summary_path': str(summary_path.relative_to(root)),
                      'summary_sha256': sha256_file(summary_path)})


def _paired_component_bootstrap(component_scores, mc_se, seeds_by_arm, methods_by_arm,
                                repeats, seed):
    """按同源组分重采样、正式种子配对；QVAE/deterministic 使用同一组分抽样。"""
    component_sets = []
    for arm, seeds in seeds_by_arm.items():
        for model_seed in seeds:
            for method in methods_by_arm[arm]:
                scores = component_scores.get((arm, model_seed), {}).get(method)
                if scores:
                    component_sets.append(set(scores))
    components = sorted(set.intersection(*component_sets)) if component_sets else []
    if not components:
        return {'method': 'component_seed_paired_bootstrap', 'components': [],
                'bootstrap_repeats': repeats, 'arms': {},
                'unknown': 'no common components for paired bootstrap'}
    generator = np.random.default_rng(stable_seed(seed, 'formal_bootstrap') % (2 ** 63 - 1))
    draws = {arm: {method: [] for method in methods} for arm, methods in methods_by_arm.items()}
    for _ in range(int(repeats)):
        indices = generator.integers(0, len(components), len(components))
        sampled = [components[index] for index in indices]
        for arm, seeds in seeds_by_arm.items():
            for method in methods_by_arm[arm]:
                per_seed = []
                for model_seed in seeds:
                    scores = component_scores[(arm, model_seed)].get(method)
                    if scores is not None:
                        per_seed.append(float(np.mean([scores[component] for component in sampled])))
                if per_seed:
                    draws[arm][method].append(float(np.mean(per_seed)))
    arms = {}
    for arm, methods in methods_by_arm.items():
        arms[arm] = {}
        for method in methods:
            values = np.asarray(draws[arm][method], dtype=float)
            values = values[np.isfinite(values)]
            points = []
            for model_seed in seeds_by_arm[arm]:
                scores = component_scores[(arm, model_seed)].get(method)
                if scores is not None:
                    points.append(float(np.mean([scores[component] for component in components])))
            mc_values = []
            for model_seed in seeds_by_arm[arm]:
                value = mc_se.get((arm, model_seed), {}).get(method)
                if value is not None and math.isfinite(float(value)):
                    mc_values.append(float(value))
            deterministic = method == 'deterministic'
            arms[arm][method] = {
                'point': float(np.mean(points)) if points else None,
                'bootstrap_mean': float(np.mean(values)) if len(values) else None,
                'bootstrap_sd': float(np.std(values, ddof=1)) if len(values) > 1 else None,
                'ci95': [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
                        if len(values) else None,
                'mc_se_mean': float(np.mean(mc_values)) if mc_values else (0. if deterministic else None),
                'mc_se_range': [float(np.min(mc_values)), float(np.max(mc_values))]
                               if mc_values else ([0., 0.] if deterministic else None),
                'unknown': bool(not points or not len(values)),
            }
    return {'method': 'component_seed_paired_bootstrap', 'components': components,
            'component_count': len(components), 'bootstrap_repeats': int(repeats),
            'seeds': {arm: list(seeds) for arm, seeds in seeds_by_arm.items()},
            'paired_seed_policy': 'same component resample across arms; QVAE seed paired with deterministic seed',
            'arms': arms, 'draws': draws}


def evaluate_formal_models(run, plan, data, formal_manifest, root=None, device=None,
                           repeats=None, bootstrap_repeats=None, force=False):
    """正式种子的全 M 联合概率评价，供 finish_study 调用。

    QVAE 每个正式种子与同 seed 的确定性权重配对，调用 compare_joint_scores；
    评价样本数取校准接口的 evaluation_sample_count，重复 8 次。逐输入记录保留
    shared/own-marginal/deterministic 的 nats/residue、ESS 与低 ESS 标记。汇总用
    同源组分重采样、正式种子配对 bootstrap 2000 次，并单独报告 MC SE。已有
    checkpoint/输入/记录 SHA receipt 匹配时跳过重算。
    """
    root = Path(root or ROOT)
    run = Path(run)
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    repeats = int(repeats if repeats is not None
                  else plan.get('calibration', {}).get('likelihood_repetitions', 8))
    bootstrap_repeats = int(bootstrap_repeats if bootstrap_repeats is not None
                            else plan.get('statistics', {}).get('bootstrap_repeats', 2000))
    interface_path = run / 'calibration_decision.json'
    if not interface_path.exists():
        raise FileNotFoundError('Missing output interface: ' + str(interface_path))
    interface = json.loads(interface_path.read_text())
    sample_counts = {arm: int(values['evaluation_sample_count'])
                     for arm, values in interface['arms'].items()}
    if 'QVAE' not in sample_counts or 'deterministic' not in sample_counts:
        raise ValueError('Calibration interface must contain both arms')
    groups = {}
    for key, entry in formal_manifest.items():
        arm = entry['arm']
        groups.setdefault(arm, {})[int(entry['seed'])] = entry
    if 'QVAE' not in groups or 'deterministic' not in groups:
        raise ValueError('Formal manifest must contain both arms')
    paired_seeds = sorted(set(groups['QVAE']) & set(groups['deterministic']))
    if not paired_seeds:
        raise ValueError('No paired formal seeds')
    query_hash = _query_hash(data.validation, data)
    raw = root / 'data/results' / run.name / 'raw'
    component_scores = {}
    mc_se = {}
    records_paths = {}
    summary_paths = {}
    receipt_paths = {}
    seed_summaries = {}
    evaluated = []
    skipped = []

    def store(key, arm, model_seed, summary, records_path, summary_path, receipt_path):
        component_scores[(arm, model_seed)] = summary['components_by_method']
        mc_se[(arm, model_seed)] = summary['mc_se']
        records_paths[key] = str(records_path.relative_to(root))
        summary_paths[key] = str(summary_path.relative_to(root))
        receipt_paths[key] = str(receipt_path.relative_to(root))
        seed_summaries[key] = {'path': summary_paths[key],
                               'methods': {method: {'score': value.get('score'),
                                                    'se': value.get('se')}
                                           for method, value in summary['methods'].items()}}

    for model_seed in paired_seeds:
        key = f'QVAE/{model_seed}'
        entry = groups['QVAE'][model_seed]
        deterministic_entry = groups['deterministic'][model_seed]
        records_path, summary_path, receipt_path = _formal_eval_paths(root, run.name, key)
        receipt = None if force else _formal_receipt(
            root, receipt_path, entry['sha256'], deterministic_entry['sha256'],
            sample_counts['QVAE'], repeats, query_hash)
        if receipt is not None:
            summary = json.loads((root / receipt['summary_path']).read_text())
            summary_paths[key] = receipt['summary_path']
            skipped.append(key)
            store(key, 'QVAE', model_seed, summary, records_path, summary_path, receipt_path)
            continue
        qvae = _load_formal_model(entry, plan, root, device)
        deterministic = _load_formal_model(deterministic_entry, plan, root, device)
        comparison = compare_joint_scores(
            qvae, deterministic, data, data.validation,
            sample_count=sample_counts['QVAE'], repeats=repeats,
            seed=stable_seed(plan.get('statistics', {}).get('seed', 0),
                             'formal_evaluation', 'QVAE', model_seed),
            batch_size=16, sample_chunk=32)
        posterior = posterior_joint_probability(
            qvae, data, data.validation, sample_count=sample_counts['QVAE'],
            repeats=repeats,
            seed=stable_seed(plan.get('statistics', {}).get('seed', 0),
                             'formal_evaluation', 'posterior', model_seed),
            batch_size=16, sample_chunk=32)
        combined = []
        for base, post in zip(comparison['records'], posterior['records']):
            if (base['row'], base['panel']) != (post['row'], post['panel']):
                raise ValueError('Posterior and comparison query order differ')
            posterior_values = post['nats_per_residue']
            deterministic_values = [base['deterministic'][0]] * len(posterior_values)
            combined.append({
                **base,
                'posterior_is_joint': posterior_values,
                'posterior_ess': post.get('ess'),
                'posterior_max_normalized_weight': post.get('max_normalized_weight'),
                'posterior_low_ess': post.get('low_ess'),
                'posterior_minus_shared': _formal_difference(posterior_values, base['shared_joint']),
                'posterior_minus_marginal': _formal_difference(posterior_values, base['marginal_product']),
                'posterior_minus_deterministic': _formal_difference(
                    posterior_values, deterministic_values),
            })
        _write_formal_records(records_path, key, combined)
        methods = {method: _compact_formal_summary(comparison['summary'][method])
                   for method in _FORMAL_METHODS}
        methods['posterior_is_joint'] = _compact_formal_summary(posterior['summary'])
        for method in ('posterior_minus_shared', 'posterior_minus_marginal',
                       'posterior_minus_deterministic'):
            methods[method] = _compact_formal_summary(
                _summary_for_values(combined, data, method))
        summary = {'fit': key, 'arm': 'QVAE', 'seed': int(model_seed),
                   'checkpoint_sha256': entry['sha256'],
                   'paired_deterministic_sha256': deterministic_entry['sha256'],
                   'sample_count': int(sample_counts['QVAE']), 'repeats': int(repeats),
                   'query_hash': query_hash, 'methods': methods,
                   'components_by_method': {method: methods[method]['components']
                                            for method in methods},
                   'mc_se': {method: methods[method]['se'] for method in methods},
                   'records_path': str(records_path.relative_to(root))}
        write_json(summary_path, summary)
        _write_formal_receipt(receipt_path, root, key, entry['sha256'],
                              deterministic_entry['sha256'], sample_counts['QVAE'],
                              repeats, query_hash, records_path, summary_path)
        del qvae, deterministic
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        evaluated.append(key)
        store(key, 'QVAE', model_seed, summary, records_path, summary_path, receipt_path)

    for model_seed in sorted(groups['deterministic']):
        key = f'deterministic/{model_seed}'
        entry = groups['deterministic'][model_seed]
        records_path, summary_path, receipt_path = _formal_eval_paths(root, run.name, key)
        receipt = None if force else _formal_receipt(
            root, receipt_path, entry['sha256'], None, 0, 1, query_hash)
        if receipt is not None:
            summary = json.loads((root / receipt['summary_path']).read_text())
            summary_paths[key] = receipt['summary_path']
            skipped.append(key)
            store(key, 'deterministic', model_seed, summary, records_path, summary_path,
                  receipt_path)
            continue
        deterministic = _load_formal_model(entry, plan, root, device)
        result = deterministic_joint_probability(deterministic, data, data.validation)
        _write_formal_records(records_path, key, result['records'])
        methods = {'deterministic': _compact_formal_summary(result['summary'])}
        summary = {'fit': key, 'arm': 'deterministic', 'seed': int(model_seed),
                   'checkpoint_sha256': entry['sha256'], 'paired_deterministic_sha256': None,
                   'sample_count': 0, 'repeats': 1, 'query_hash': query_hash,
                   'methods': methods,
                   'components_by_method': {'deterministic': methods['deterministic']['components']},
                   'mc_se': {'deterministic': 0.},
                   'records_path': str(records_path.relative_to(root))}
        write_json(summary_path, summary)
        _write_formal_receipt(receipt_path, root, key, entry['sha256'], None, 0, 1,
                              query_hash, records_path, summary_path)
        del deterministic
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        evaluated.append(key)
        store(key, 'deterministic', model_seed, summary, records_path, summary_path,
              receipt_path)

    bootstrap = _paired_component_bootstrap(
        component_scores, mc_se,
        {'QVAE': paired_seeds, 'deterministic': sorted(groups['deterministic'])},
        {'QVAE': list(_QVAE_METHODS), 'deterministic': ['deterministic']},
        bootstrap_repeats, plan.get('statistics', {}).get('seed', 0))
    draws = bootstrap.pop('draws', None)
    draws_path = raw / 'formal_evaluation' / 'bootstrap_draws.json'
    if draws is not None:
        write_json(draws_path, {'components': bootstrap['components'],
                                'bootstrap_repeats': bootstrap['bootstrap_repeats'],
                                'draws': draws})
    aggregate = {
        'method': 'formal_full_M_paired_joint_evaluation',
        'validation_queries': len(data.validation), 'repeats': repeats,
        'bootstrap_repeats': bootstrap_repeats,
        'evaluation_sample_count': sample_counts,
        'records': records_paths, 'summary_paths': summary_paths,
        'receipts': receipt_paths, 'seed_summaries': seed_summaries,
        'bootstrap': bootstrap, 'query_hash': query_hash,
        'evaluated': evaluated, 'skipped': skipped,
        'bootstrap_draws_path': str(draws_path.relative_to(root)) if draws is not None else None,
    }
    aggregate_path = raw / 'formal_evaluation_summary.json'
    write_json(aggregate_path, aggregate)
    aggregate['path'] = str(aggregate_path.relative_to(root))
    aggregate['sha256'] = sha256_file(aggregate_path)
    return aggregate


def _write_curve_scaling_note(raw, revision):
    history_path = raw / 'calibration_history.json'
    history = None
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text())
        except Exception:
            history = None
    if history is None:
        candidates = sorted((raw / 'curves').glob('calibration_*.json'))
        if candidates:
            try:
                history = json.loads(candidates[0].read_text()).get('history')
            except Exception:
                history = None
    corrected = None
    if history:
        corrected = bool('train_reconstruction_numerator' in history[0])
    note = {
        'fixed_in_git_revision': revision,
        'calibration_curves_corrected': corrected,
        'issue': ('earlier calibration curves stored per-epoch means of batch-level weighted '
                  'per-residue reconstruction and KL without dividing by total sample counts'),
        'impact': 'loss.backward, component weights and validation scores are unaffected',
        'corrected_fields': ['train_reconstruction_numerator', 'train_reconstruction_denominator',
                             'train_kl_numerator', 'train_kl_denominator'],
        'calibration_raw_records_preserved': True,
    }
    write_json(raw / 'curve_scaling_note.json', note)
    return note


def _source_hashes(run, plan):
    paths = [Path(__file__), ROOT / 'src/natvqvae/conditional_evaluation.py',
             ROOT / 'src/natvqvae/conditional_generation.py',
             ROOT / 'src/natvqvae/conditional_data.py',
             ROOT / 'src/natvqvae/conditional_outputs.py',
             ROOT / 'scripts/run_conditional_study.py',
             run / 'PLAN.md', run / 'plan.json']
    output = {}
    for path in paths:
        if path.exists():
            try:
                key = str(path.relative_to(ROOT))
            except ValueError:
                key = str(path)
            output[key] = sha256_file(path)
    return output


def run_study(run, plan, data=None, root=None, device=None, calibrate_only=False,
              from_calibration=False, train_only=False):
    """训练端入口；生成与最终交付由 conditional_outputs.finish_study 完成。

    calibrate_only=True 时只执行四次校准、写出冻结决策与校准清单后退出，便于
    在长流程前核对 T/B、样本数与吞吐，不消耗 search/formal 拟合预算。
    from_calibration=True 时复用已完成校准，只运行 search/formal；train_only=True
    在正式权重与清单落盘后退出，不调用 finish_study。
    """
    root = Path(root or ROOT)
    run = Path(run)
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if plan.get('task') != 'NatV_conditional_categorical_joint_generation':
        raise ValueError('Plan task differs from this study')
    if calibrate_only and (from_calibration or train_only):
        raise ValueError('calibrate_only cannot be combined with continuation flags')
    calibration_count, search_count, formal_count, total = expected_fit_count(plan)
    if total > int(plan['budget']['max_fits']):
        raise ValueError(f'Scheduled fits {total} exceed authorized {plan["budget"]["max_fits"]}')
    raw = root / 'data/results' / run.name / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    models = root / 'data/models' / run.name
    models.mkdir(parents=True, exist_ok=True)
    attempts = {'count': 0}
    failures = []
    save_records = bool(plan.get('save_validation_records', True))
    started = time.monotonic()
    if data is None:
        data = FeatureData(run)
    if len(data.rows[data.rows.training_role_v2 == 'training']) != int(plan['training_population']):
        raise ValueError('Training population differs from plan')
    if len(data.rows[data.rows.training_role_v2 == 'validation']) != int(plan['validation_population']):
        raise ValueError('Validation population differs from plan')
    input_files = {}
    if (run / 'inputs').exists():
        for path in sorted((run / 'inputs').glob('*.json')):
            try:
                key = str(path.relative_to(root))
            except ValueError:
                key = str(path)
            input_files[key] = sha256_file(path)
    feature_manifest = raw / 'feature_manifest.json'
    statistical_manifest = raw / 'statistical_feature_manifest.json'
    identity = {
        'task': plan['task'], 'git_revision': _git_revision(root),
        'source_sha256': _source_hashes(run, plan),
        'plan_sha256': sha256_file(run / 'plan.json'),
        'input_sha256': input_files,
        'feature_manifest_sha256': sha256_file(feature_manifest) if feature_manifest.exists() else None,
        'statistical_feature_manifest_sha256': (sha256_file(statistical_manifest)
                                                if statistical_manifest.exists() else None),
        'training_population': int(plan['training_population']),
        'validation_population': int(plan['validation_population']),
        'scheduled_fits': {'calibration': calibration_count, 'search': search_count,
                           'formal': formal_count, 'total': total},
        'device': str(device),
        'environment': {'python': sys.version, 'torch': torch.__version__,
                        'numpy': np.__version__, 'cuda': torch.version.cuda},
    }
    mode = ('calibrate_only' if calibrate_only else
            ('from_calibration' if from_calibration else 'full'))
    if train_only:
        mode += '+train_only'
    start_path = raw / 'study_start.json'
    if start_path.exists():
        initial_sha = sha256_file(start_path)
        initial = json.loads(start_path.read_text())
        stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
        continuation_path = raw / 'study_continuations' / f'study_start_{stamp}.json'
        index = 1
        while continuation_path.exists():
            continuation_path = raw / 'study_continuations' / f'study_start_{stamp}_{index}.json'
            index += 1
        write_json(continuation_path, {**identity, 'mode': mode,
                                       'initial_identity_sha256': initial_sha})
        identity_record = {
            'initial_identity': {'path': str(start_path.relative_to(root)),
                                 'sha256': initial_sha,
                                 'git_revision': initial.get('git_revision')},
            'continuation_identity': {'path': str(continuation_path.relative_to(root)),
                                      'sha256': sha256_file(continuation_path),
                                      'git_revision': identity.get('git_revision')},
            'initial_git_revision': initial.get('git_revision'),
            'current_git_revision': identity.get('git_revision'),
        }
    else:
        identity['mode'] = mode
        write_json(start_path, identity)
        initial_sha = sha256_file(start_path)
        identity_record = {
            'initial_identity': {'path': str(start_path.relative_to(root)),
                                 'sha256': initial_sha,
                                 'git_revision': identity.get('git_revision')},
            'continuation_identity': None,
            'initial_git_revision': identity.get('git_revision'),
            'current_git_revision': identity.get('git_revision'),
        }
    try:
        if from_calibration:
            interface = json.loads((run / 'calibration_decision.json').read_text())
            full_decision_path = root / interface['full_decision']['path']
            if sha256_file(full_decision_path) != interface['full_decision']['sha256']:
                raise ValueError('Full calibration decision changed')
            decision = json.loads(full_decision_path.read_text())
            calibration = {'decision': decision}
            calibration_manifest = json.loads((raw / 'calibration_manifest.json').read_text())
            attempts = {'count': int(calibration_manifest['attempts'])}
            failures = list(calibration_manifest.get('failures', []))
            if attempts['count'] != calibration_count:
                raise FitFailed('Calibration manifest attempt count differs from schedule')
            start_snapshot = raw / 'study_start.json'
            if start_snapshot.exists():
                initial = json.loads(start_snapshot.read_text())
                if interface.get('git_revision') != initial.get('git_revision'):
                    write_json(raw / 'calibration_identity_correction.json', {
                        'interface_git_revision': interface.get('git_revision'),
                        'initial_git_revision': initial.get('git_revision'),
                        'calibration_manifest_git_revision': calibration_manifest.get('git_revision'),
                        'note': ('interface git_revision was evaluated when the interface was written; '
                                 'calibration provenance is the initial study_start and calibration manifest'),
                    })
            _write_curve_scaling_note(raw, identity_record['current_git_revision'])
        else:
            calibration = _calibrate(run, plan, data, root, device, attempts, failures,
                                     save_records)
            if attempts['count'] != calibration_count:
                raise FitFailed('Calibration fit count differs from schedule')
            interface_path = run / 'calibration_decision.json'
            if interface_path.exists():
                interface = json.loads(interface_path.read_text())
                if interface.get('git_revision') != identity.get('git_revision'):
                    interface['git_revision'] = identity.get('git_revision')
                    write_json(interface_path, interface)
            _write_curve_scaling_note(raw, identity_record['current_git_revision'])
            if calibrate_only:
                calibration_records = list(calibration['fits'].values())
                write_json(raw / 'calibration_fits.json', calibration_records)
                combined_history = []
                for path in sorted((raw / 'curves').glob('*.json')):
                    combined_history.extend(json.loads(path.read_text())['history'])
                write_json(raw / 'calibration_history.json', combined_history)
                _write_curve_scaling_note(raw, identity_record['current_git_revision'])
                wall_seconds = time.monotonic() - started
                cost = {'attempted_fits': attempts['count'],
                        'authorized_fits': int(plan['budget']['max_fits']),
                        'wall_seconds': wall_seconds,
                        'fit_seconds': sum(record['total_seconds'] for record in calibration_records),
                        'failures': failures}
                write_json(raw / 'calibration_cost.json', cost)
                decision_path = raw / 'calibration_decision.json'
                manifest = {
                    'task': plan['task'], 'status': 'calibration_complete',
                    'git_revision': identity_record['current_git_revision'],
                    'initial_git_revision': identity_record['initial_git_revision'],
                    'identity': identity_record,
                    'plan_sha256': identity['plan_sha256'],
                    'decision': {'path': str(decision_path.relative_to(root)),
                                 'sha256': sha256_file(decision_path)},
                    'fit_status': {'path': str((raw / 'calibration_fits.json').relative_to(root)),
                                   'sha256': sha256_file(raw / 'calibration_fits.json')},
                    'cost': cost,
                    'attempts': attempts['count'],
                    'failures': failures,
                }
                if (raw / 'curve_scaling_note.json').exists():
                    manifest['curve_scaling_note'] = {
                        'path': str((raw / 'curve_scaling_note.json').relative_to(root)),
                        'sha256': sha256_file(raw / 'curve_scaling_note.json')}
                write_json(raw / 'calibration_manifest.json', manifest)
                status(run, 'CALIBRATION_READY', mode='calibration_only',
                       attempts=attempts['count'], wall_seconds=wall_seconds)
                return manifest
        search_selected, search_records = _search(
            run, plan, data, root, device, calibration, attempts, failures, save_records)
        if attempts['count'] != calibration_count + search_count:
            raise FitFailed('Search fit count differs from schedule')
        formal, formal_records = _formal(
            run, plan, data, root, device, calibration, search_selected,
            attempts, failures, save_records)
        if attempts['count'] != total:
            raise FitFailed('Total fit count differs from schedule')
        if calibration.get('fits'):
            calibration_records = list(calibration['fits'].values())
        else:
            calibration_fits_path = raw / 'calibration_fits.json'
            calibration_records = (json.loads(calibration_fits_path.read_text())
                                   if calibration_fits_path.exists() else [])
        fit_records = calibration_records + search_records + formal_records
        combined_history = []
        for path in sorted((raw / 'curves').glob('*.json')):
            combined_history.extend(json.loads(path.read_text())['history'])
        write_json(raw / 'training_history.json', combined_history)
        fit_status = [{key: record.get(key) for key in
                       ('fit', 'phase', 'arm', 'seed', 'steps', 'epochs', 'stop_reason',
                        'best_step', 'best_epoch', 'best_score', 'best_path', 'best_sha256',
                        'last_path', 'last_sha256', 'total_seconds', 'steps_per_second',
                        'max_memory_mib', 'config')}
                      for record in fit_records]
        write_json(raw / 'fit_status.json', fit_status)
        wall_seconds = time.monotonic() - started
        cost = {'attempted_fits': attempts['count'],
                'authorized_fits': int(plan['budget']['max_fits']),
                'wall_seconds': wall_seconds,
                'fit_seconds': sum(record['total_seconds'] for record in fit_records),
                'max_memory_mib': max([record.get('max_memory_mib', 0.) for record in fit_records] or [0.]),
                'failures': failures}
        write_json(raw / 'cost.json', cost)
        artifacts = {}
        for name, path in [('study_start', raw / 'study_start.json'),
                           ('calibration_decision', raw / 'calibration_decision.json'),
                           ('curve_scaling_note', raw / 'curve_scaling_note.json'),
                           ('search_selection', raw / 'search_selection.json'),
                           ('formal_manifest', raw / 'formal_manifest.json'),
                           ('fit_status', raw / 'fit_status.json'),
                           ('cost', raw / 'cost.json'),
                           ('training_history', raw / 'training_history.json')]:
            if path.exists():
                artifacts[name] = {'path': str(path.relative_to(root)),
                                   'sha256': sha256_file(path)}
        interface_path = run / 'calibration_decision.json'
        if interface_path.exists():
            artifacts['calibration_interface'] = {
                'path': str(interface_path.relative_to(root)),
                'sha256': sha256_file(interface_path)}
        if identity_record['continuation_identity'] is not None:
            artifacts['study_continuation'] = identity_record['continuation_identity']
        if train_only:
            training_manifest = {
                'task': plan['task'], 'status': 'training_complete',
                'git_revision': identity_record['current_git_revision'],
                'initial_git_revision': identity_record['initial_git_revision'],
                'identity': identity_record,
                'plan_sha256': identity['plan_sha256'],
                'formal_entries': len(formal), 'attempts': attempts['count'],
                'failures': failures, 'artifacts': artifacts,
            }
            write_json(raw / 'training_manifest.json', training_manifest)
            status(run, 'TRAIN_READY', fits=attempts['count'], wall_seconds=wall_seconds)
            return training_manifest
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        status(run, 'FINISHING', fits=len(formal))
        from .conditional_outputs import finish_study
        finish = finish_study(run, plan, data, formal)
        manifest = {
            'task': plan['task'], 'status': 'completed',
            'git_revision': identity_record['current_git_revision'],
            'initial_git_revision': identity_record['initial_git_revision'],
            'identity': identity_record,
            'plan_sha256': identity['plan_sha256'],
            'formal_entries': len(formal),
            'artifacts': artifacts,
            'finish_returned': repr(finish)[:2000],
        }
        write_json(raw / 'study_manifest.json', manifest)
        status(run, 'COMPLETED', fits=attempts['count'], wall_seconds=time.monotonic() - started)
        return manifest
    except Exception as exc:
        failure = {'phase': 'study', 'reason': repr(exc), 'traceback': traceback.format_exc(),
                   'attempted_fits': attempts['count'], 'wall_seconds': time.monotonic() - started,
                   'failures': failures}
        _append_jsonl(raw / 'failures.jsonl', failure)
        write_json(raw / 'study_failure.json', failure)
        status(run, 'FAILED', reason=repr(exc), attempted_fits=attempts['count'])
        raise


def _git_revision(root):
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    except BaseException:
        return None
