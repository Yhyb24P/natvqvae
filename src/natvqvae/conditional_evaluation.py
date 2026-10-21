"""条件联合概率评价：先验 joint、后验重要性采样、边缘乘积与真实目标分布。

所有函数接受 FeatureData 风格的查询列表（row/offset/length/positions/panel/weight），
因此可复用于额外统计 mask。逐查询结果保留有限样本诊断，聚合时按 q.weight 与
component_v2 做同源组分平衡，不对面板等权平均。
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import torch

from .conditional_data import stable_seed
from .conditional_generation import (AA, categorical_log_prob, categorical_samples,
                                     reconstruction_nll, target_tensors)

EPS = 1e-12


def _device(model):
    return next(model.parameters()).device


def _generator(device, seed):
    kind = 'cuda' if device.type == 'cuda' else 'cpu'
    generator = torch.Generator(device=kind)
    generator.manual_seed(int(seed) % (2**63 - 1))
    return generator


def _chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start+size]


def _scored_positions(query):
    positions = query.get('subset_positions', query['positions'])
    if not positions:
        raise ValueError('Scoring positions must be nonempty')
    return positions


def _metadata(data, queries):
    output = []
    for query in queries:
        row = data.rows.iloc[query['row']]
        scored = _scored_positions(query)
        if len(set(scored)) != len(scored) or not set(scored) <= set(query['positions']):
            raise ValueError('subset_positions must be a subset of the conditioning mask')
        output.append({'row': int(query['row']), 'sequence_id': str(row.sequence_id),
                       'component': str(row.component_v2), 'panel': str(query['panel']),
                       'weight': float(query.get('weight', 0.)),
                       'sites': int(len(scored)), 'context_sites': int(len(query['positions']))})
    return output


def _prepare(model, data, queries, device, need_full):
    context, full, valid, mask, labels = data.batch(queries, device=device, need_full=need_full)
    hidden = model.encode_context(context, valid, mask)
    score_mask = torch.zeros_like(mask)
    for index, query in enumerate(queries):
        score_mask[index, _scored_positions(query)] = True
    targets, y, use, _ = target_tensors(hidden, score_mask, labels)
    return hidden, full, valid, mask, labels, targets, y, use


def _decode_nll(model, targets, y, use, codes, sample_chunk):
    values = []
    for start in range(0, len(codes), sample_chunk):
        logits = model.decode_codes(targets, codes[start:start+sample_chunk])
        values.append(reconstruction_nll(logits, y, use))
    return torch.cat(values, 0)


def _finite(value):
    return bool(torch.isfinite(value).all())


def _float_list(value):
    if value is None:
        return [None]
    return [float(x) if math.isfinite(float(x)) else None for x in value]


def _records(meta, repeats, n_sites, sampled=True):
    """repeats: 每项为 {'log_prob': B, 'ess': B, 'max_weight': B}。"""
    output = []
    for index, item in enumerate(meta):
        log_prob = [float(rep['log_prob'][index]) for rep in repeats]
        sites = float(n_sites[index].item()) if torch.is_tensor(n_sites) else float(n_sites[index])
        nats = [(-value / sites) if math.isfinite(value) else None for value in log_prob]
        ess_raw = [float(rep['ess'][index]) for rep in repeats]
        max_weight_raw = [float(rep['max_weight'][index]) for rep in repeats]
        ess = [value if math.isfinite(value) else None for value in ess_raw]
        max_weight = [value if math.isfinite(value) else None for value in max_weight_raw]
        low_ess = sampled and any(value is None or value < 2 or weight is None or weight > .5
                                  for value, weight in zip(ess, max_weight))
        output.append({
            **item,
            'log_prob': [value if math.isfinite(value) else None for value in log_prob],
            'nats_per_residue': nats,
            'ess': ess,
            'max_normalized_weight': max_weight,
            'low_ess': bool(low_ess),
        })
    return output


def _method_record(method, sample_count, repeats, records, data, extra=None):
    summary = aggregate_panel_records(records, data)
    return {'method': method, 'sample_count': int(sample_count), 'repeats': int(repeats),
            'records': records, 'summary': summary, **(extra or {})}


@torch.no_grad()
def posterior_joint_probability(model, data, queries, sample_count=256, repeats=8,
                                seed=0, batch_size=8, sample_chunk=32):
    """完整目标 log p(x_M|c,M) 的后验重要性采样估计。

    返回逐查询 log_prob 与 nats/residue（负数 log 概率取负）、ESS、最大归一化权重
    以及重复间 SE（在 summary 中）。估计不可用时相应字段为 None，summary['finite']
    为 False。有限样本 log 偏差不在单个估计中宣称；校准阶段用相邻样本数的
    nested_sample_log_difference 报告收敛差。
    """
    if model.arm != 'QVAE':
        raise ValueError('Posterior importance sampling requires the QVAE arm')
    if sample_count < 1 or repeats < 1:
        raise ValueError('sample_count and repeats must be positive')
    device = _device(model)
    output = []
    for batch_index, batch in enumerate(_chunks(list(queries), batch_size)):
        hidden, full, valid, mask, labels, targets, y, use = _prepare(
            model, data, batch, device, need_full=True)
        q_logits = model.posterior(full, hidden, valid, mask)
        p_logits = model.prior(hidden, valid, mask)
        sites = use.sum(1)
        per_repeat = []
        for repeat in range(repeats):
            generator = _generator(device, stable_seed(seed, 'posterior', batch_index, repeat))
            codes = categorical_samples(q_logits, sample_count, generator)
            logq = categorical_log_prob(q_logits, codes)
            logpz = categorical_log_prob(p_logits, codes)
            nll = _decode_nll(model, targets, y, use, codes, sample_chunk)
            log_weight = -nll + logpz - logq
            finite = _finite(log_weight)
            if finite:
                normalizer = torch.logsumexp(log_weight, 0)
                log_prob = normalizer - math.log(sample_count)
                ess = torch.exp(2 * normalizer - torch.logsumexp(2 * log_weight, 0))
                max_weight = torch.exp(log_weight.max(0).values - normalizer)
            else:
                log_prob = torch.full_like(nll.mean(0), float('nan'))
                ess = torch.full_like(log_prob, float('nan'))
                max_weight = torch.full_like(log_prob, float('nan'))
            per_repeat.append({'log_prob': log_prob, 'ess': ess,
                               'max_weight': max_weight, 'finite': finite})
        meta = _metadata(data, batch)
        for record, item in zip(_records(meta, per_repeat, sites), meta):
            record['finite'] = all(rep['finite'] for rep in per_repeat)
            output.append(record)
    return _method_record('posterior_importance_sampling', sample_count, repeats, output, data)


@torch.no_grad()
def prior_joint_probability(model, data, queries, sample_count=256, repeats=8,
                            seed=0, batch_size=8, sample_chunk=32):
    """条件先验共享 z 的 joint score：log E_{z~p(z|c,M)} p(x_M|c,M,z)。"""
    if model.arm != 'QVAE':
        raise ValueError('Prior joint probability requires the QVAE arm')
    if sample_count < 1 or repeats < 1:
        raise ValueError('sample_count and repeats must be positive')
    device = _device(model)
    output = []
    for batch_index, batch in enumerate(_chunks(list(queries), batch_size)):
        hidden, _, valid, mask, _, targets, y, use = _prepare(
            model, data, batch, device, need_full=False)
        p_logits = model.prior(hidden, valid, mask)
        sites = use.sum(1)
        per_repeat = []
        for repeat in range(repeats):
            generator = _generator(device, stable_seed(seed, 'prior', batch_index, repeat))
            codes = categorical_samples(p_logits, sample_count, generator)
            nll = _decode_nll(model, targets, y, use, codes, sample_chunk)
            log_px = -nll
            finite = _finite(log_px)
            if finite:
                normalizer = torch.logsumexp(log_px, 0)
                log_prob = normalizer - math.log(sample_count)
                ess = torch.exp(2 * normalizer - torch.logsumexp(2 * log_px, 0))
                max_weight = torch.exp(log_px.max(0).values - normalizer)
            else:
                log_prob = torch.full_like(nll.mean(0), float('nan'))
                ess = torch.full_like(log_prob, float('nan'))
                max_weight = torch.full_like(log_prob, float('nan'))
            per_repeat.append({'log_prob': log_prob, 'ess': ess,
                               'max_weight': max_weight, 'finite': finite})
        meta = _metadata(data, batch)
        for record, item in zip(_records(meta, per_repeat, sites), meta):
            record['finite'] = all(rep['finite'] for rep in per_repeat)
            output.append(record)
    return _method_record('prior_joint_shared_z', sample_count, repeats, output, data)


@torch.no_grad()
def marginal_probability(model, data, queries, sample_count=256, repeats=8,
                         seed=0, batch_size=8, sample_chunk=32):
    """同模型单位点边缘乘积：sum_i log E_{z~p(z|c,M)} p(x_i|c,M,z)。

    边缘在每个目标位点独立估计后相乘；同一次重复内所有位点共享同一组先验 z，
    因此该 score 只去掉位点间协变，不改变 z 的先验边缘化。
    """
    if model.arm != 'QVAE':
        raise ValueError('Marginal probability requires the QVAE arm')
    if sample_count < 1 or repeats < 1:
        raise ValueError('sample_count and repeats must be positive')
    device = _device(model)
    output = []
    for batch_index, batch in enumerate(_chunks(list(queries), batch_size)):
        hidden, _, valid, mask, _, targets, y, use = _prepare(
            model, data, batch, device, need_full=False)
        p_logits = model.prior(hidden, valid, mask)
        sites = use.sum(1)
        per_repeat = []
        for repeat in range(repeats):
            generator = _generator(device, stable_seed(seed, 'marginal', batch_index, repeat))
            codes = categorical_samples(p_logits, sample_count, generator)
            weight_sum = torch.zeros_like(use, dtype=torch.float32)
            weight_square = torch.zeros_like(weight_sum)
            weight_max = torch.zeros_like(weight_sum)
            count = 0
            for start in range(0, len(codes), sample_chunk):
                chunk = codes[start:start+sample_chunk]
                logits = model.decode_codes(targets, chunk)
                probabilities = logits.float().softmax(-1)
                target_index = y[None].expand(len(chunk), -1, -1)[..., None]
                target_probability = probabilities.gather(-1, target_index).squeeze(-1)
                weight_sum += target_probability.sum(0)
                weight_square += target_probability.square().sum(0)
                weight_max = torch.maximum(weight_max, target_probability.max(0).values)
                count += len(chunk)
            average = weight_sum / count
            finite = _finite(average)
            if finite:
                log_probability = (average.clamp_min(EPS).log() * use).sum(1)
                ess = weight_sum.square() / weight_square.clamp_min(EPS)
                max_weight = weight_max / weight_sum.clamp_min(EPS)
            else:
                log_probability = torch.full_like(weight_sum.sum(1), float('nan'))
                ess = torch.full_like(log_probability, float('nan'))
                max_weight = torch.full_like(log_probability, float('nan'))
                ess = ess[:, None].expand_as(use)
                max_weight = max_weight[:, None].expand_as(use)
            used_ess = torch.where(use, ess, torch.full_like(ess, float('inf'))).min(1).values
            used_max = torch.where(use, max_weight, torch.full_like(max_weight, -float('inf'))).max(1).values
            per_repeat.append({'log_prob': log_probability, 'ess': used_ess,
                               'max_weight': used_max, 'finite': finite})
        meta = _metadata(data, batch)
        for record, item in zip(_records(meta, per_repeat, sites), meta):
            record['finite'] = all(rep['finite'] for rep in per_repeat)
            output.append(record)
    return _method_record('marginal_product', sample_count, repeats, output, data)


@torch.no_grad()
def deterministic_joint_probability(model, data, queries, batch_size=64, **_):
    """确定性 arm 的精确 joint score；逐位点条件独立，joint 等于边缘乘积。"""
    if model.arm != 'deterministic':
        raise ValueError('Deterministic joint probability requires the deterministic arm')
    device = _device(model)
    output = []
    for batch in _chunks(list(queries), batch_size):
        hidden, _, valid, mask, _, targets, y, use = _prepare(
            model, data, batch, device, need_full=False)
        latent = model.deterministic_context(hidden, valid, mask)
        logits = model.decode(targets, latent.unsqueeze(0))
        nll = reconstruction_nll(logits, y, use)[0]
        sites = use.sum(1)
        per_repeat = [{'log_prob': -nll, 'ess': torch.ones_like(nll),
                       'max_weight': torch.ones_like(nll),
                       'finite': _finite(nll)}]
        for record, item in zip(_records(_metadata(data, batch), per_repeat, sites, sampled=False),
                                _metadata(data, batch)):
            record['finite'] = bool(per_repeat[0]['finite'])
            output.append(record)
    return _method_record('deterministic_exact', 1, 1, output, data,
                          {'exact': True, 'marginal_equals_joint': True})


def compare_joint_scores(qvae, deterministic, data, queries, sample_count=256, repeats=8,
                         seed=0, batch_size=8, sample_chunk=32):
    """同一 c,M 上比较共享 z joint、同模型边缘乘积与独立确定性模型。

    差值为 nats/residue：marginal-shared、deterministic-shared、marginal-deterministic；
    负值表示前者的完整目标负对数似然更低。
    """
    shared = prior_joint_probability(qvae, data, queries, sample_count, repeats,
                                     seed, batch_size, sample_chunk)
    marginal = marginal_probability(qvae, data, queries, sample_count, repeats,
                                    seed, batch_size, sample_chunk)
    exact = deterministic_joint_probability(deterministic, data, queries)
    comparisons = []
    for a, b, c in zip(shared['records'], marginal['records'], exact['records']):
        if (a['row'], a['panel']) != (b['row'], b['panel']) or (a['row'], a['panel']) != (c['row'], c['panel']):
            raise ValueError('Paired evaluation order differs')
        shared_values = np.asarray([np.nan if x is None else x for x in a['nats_per_residue']], dtype=float)
        marginal_values = np.asarray([np.nan if x is None else x for x in b['nats_per_residue']], dtype=float)
        deterministic_value = c['nats_per_residue'][0]
        deterministic_values = np.full_like(shared_values, np.nan if deterministic_value is None else deterministic_value)
        comparisons.append({
            'row': a['row'], 'sequence_id': a['sequence_id'], 'component': a['component'],
            'panel': a['panel'], 'weight': a['weight'], 'sites': a['sites'],
            'shared_joint': a['nats_per_residue'],
            'marginal_product': b['nats_per_residue'],
            'deterministic': c['nats_per_residue'],
            'marginal_minus_shared': _float_list(marginal_values - shared_values),
            'deterministic_minus_shared': _float_list(deterministic_values - shared_values),
            'marginal_minus_deterministic': _float_list(marginal_values - deterministic_values),
        })
    def aggregate(key):
        records = [{'row': item['row'], 'component': item['component'], 'panel': item['panel'],
                    'weight': item['weight'], 'nats_per_residue': item[key]}
                   for item in comparisons]
        return aggregate_panel_records(records, data)
    return {'method': 'paired_shared_marginal_deterministic',
            'sample_count': int(sample_count), 'repeats': int(repeats),
            'records': comparisons,
            'summary': {'shared_joint': shared['summary'], 'marginal_product': marginal['summary'],
                        'deterministic': exact['summary'],
                        'marginal_minus_shared': aggregate('marginal_minus_shared'),
                        'deterministic_minus_shared': aggregate('deterministic_minus_shared'),
                        'marginal_minus_deterministic': aggregate('marginal_minus_deterministic')}}


def aggregate_panel_records(records, data):
    """按 q.weight 汇总面板，并在每个面板内先做 component_v2 平衡。

    records 需含 row/panel/weight/nats_per_residue（长度为重复数的列表）。
    返回总 score、重复间 SE 及逐面板/逐组分结果；不把面板或序列数当作独立样本。
    逐输入低 ESS 与聚合选择稳定性分开：前者见 input_mc_reliable，后者由校准阶段
    依据聚合重复 SE 判断。
    """
    if not records:
        raise ValueError('No evaluation records')
    repeats = len(records[0]['nats_per_residue'])
    for record in records:
        if len(record['nats_per_residue']) != repeats:
            raise ValueError('Inconsistent repeat counts')
    panel_rows = {}
    panel_weights = {}
    for record in records:
        panel = str(record['panel'])
        panel_rows.setdefault(panel, []).append(record)
        panel_weights.setdefault(panel, []).append(float(record.get('weight', 0.)))
    weights = {panel: float(np.mean(panel_weights[panel])) for panel in panel_rows}
    positive = {panel: value for panel, value in weights.items() if value > 0}
    if not positive:
        normalized = {panel: 1. / len(panel_rows) for panel in panel_rows}
        weight_status = 'all_zero_equal_fallback'
    else:
        total = sum(positive.values())
        normalized = {panel: (positive[panel] / total if panel in positive else 0.)
                      for panel in panel_rows}
        weight_status = 'q_weight_normalized'
    per_panel = {}
    panel_component_repeats = {}
    component_panel_values = {}
    for panel, part in panel_rows.items():
        grouped = {}
        for record in part:
            component = str(record.get('component') or data.rows.iloc[record['row']].component_v2)
            grouped.setdefault(component, []).append(record)
        component_means = {}
        for component, items in grouped.items():
            values = np.asarray([[np.nan if x is None else x for x in item['nats_per_residue']]
                                 for item in items], dtype=float)
            component_means[component] = np.nanmean(values, axis=0)
        panel_component_repeats[panel] = component_means
        stacked = np.stack(list(component_means.values()))
        repeat_scores = np.nanmean(stacked, axis=0)
        per_panel[panel] = {
            'weight': weights[panel], 'normalized_weight': normalized[panel],
            'score': float(np.nanmean(repeat_scores)),
            'se': _se(repeat_scores),
            'components': {component: float(np.nanmean(value))
                           for component, value in component_means.items()},
        }
        for component, value in component_means.items():
            component_panel_values.setdefault(component, []).append((panel, value))
    total_by_repeat = np.zeros(repeats, dtype=float)
    for panel, component_means in panel_component_repeats.items():
        total_by_repeat += normalized[panel] * np.nanmean(
            np.stack(list(component_means.values())), axis=0)
    per_component = {}
    for component, values in component_panel_values.items():
        weighted = sum(normalized[panel] * value for panel, value in values)
        per_component[component] = float(np.nanmean(weighted))
    finite = bool(np.isfinite(total_by_repeat).any())
    ess_values = []
    max_weights = []
    low_ess = []
    for record in records:
        ess_values.extend([float(x) for x in record.get('ess', [])
                           if x is not None and math.isfinite(float(x))])
        max_weights.extend([float(x) for x in record.get('max_normalized_weight', [])
                            if x is not None and math.isfinite(float(x))])
        low_ess.append(bool(record.get('low_ess', False)))
    input_mc_reliable = not any(low_ess)
    return {'score': float(np.nanmean(total_by_repeat)) if finite else None,
            'se': _se(total_by_repeat),
            'repeats': repeats, 'panels': per_panel, 'components': per_component,
            'panel_weights': normalized, 'panel_weight_status': weight_status,
            'input_ess_mean': float(np.mean(ess_values)) if ess_values else None,
            'input_ess_min': float(np.min(ess_values)) if ess_values else None,
            'input_max_normalized_weight': float(np.max(max_weights)) if max_weights else None,
            'input_low_ess_fraction': float(np.mean(low_ess)) if low_ess else None,
            'input_mc_reliable': bool(input_mc_reliable), 'finite': finite,
            'aggregate_score_defined': bool(finite)}


def _se(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return None
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def validation_score(model, data, queries, sample_count=64, repeats=1, seed=0,
                     batch_size=None, sample_chunk=32):
    """训练用验证 score；QVAE 用后验 IS，确定性 arm 用精确 joint。"""
    if model.arm == 'QVAE':
        result = posterior_joint_probability(model, data, queries, sample_count=sample_count,
                                             repeats=repeats, seed=seed,
                                             batch_size=batch_size or 8, sample_chunk=sample_chunk)
    else:
        result = deterministic_joint_probability(model, data, queries)
    return result['summary'], result


def target_aa_distribution(data, queries, weights=None, panel_weighted=False):
    """真实目标 AA 计数/频率，支持 q.weight 与 component 平衡。

    weights 为逐查询权重；panel_weighted=True 时使用 q['weight']。频率按位点计；
    component_balanced_frequencies 先求每个 component 内的加权频率，再等权平均 component。
    """
    if not queries:
        raise ValueError('No queries')
    if weights is None:
        weights = [float(q.get('weight', 1.)) if panel_weighted else 1. for q in queries]
    if len(weights) != len(queries):
        raise ValueError('Weight count differs')
    counts = np.zeros(len(AA), dtype=float)
    per_query = []
    component_samples = {}
    panel_samples = {}
    for query, weight in zip(queries, weights):
        if weight < 0:
            raise ValueError('Negative weight')
        row = data.rows.iloc[query['row']]
        sequence = str(row.sequence)
        positions = _scored_positions(query)
        letters = [sequence[position] for position in positions]
        if not set(letters) <= set(AA):
            raise ValueError('Invalid residue in target positions')
        local = np.bincount([AA.index(letter) for letter in letters], minlength=len(AA)).astype(float)
        counts += weight * local
        frequencies = local / max(len(letters), 1)
        item = {'row': int(query['row']), 'sequence_id': str(row.sequence_id),
                'component': str(row.component_v2), 'panel': str(query['panel']),
                'weight': float(weight), 'sites': len(letters),
                'context_sites': len(query['positions']),
                'counts': local.astype(int).tolist(), 'frequencies': frequencies.tolist()}
        per_query.append(item)
        component_samples.setdefault(str(row.component_v2), []).append((frequencies, weight))
        panel_samples.setdefault(str(query['panel']), []).append((frequencies, weight))
    total = counts.sum()
    frequencies = counts / total if total > 0 else counts
    def balanced(samples):
        component_values = []
        for part in samples.values():
            local_weights = np.asarray([max(weight, 0.) for _, weight in part], dtype=float)
            if local_weights.sum() <= 0:
                local_weights = np.ones(len(part), dtype=float)
            local_weights /= local_weights.sum()
            component_values.append(np.sum([weight * freq for weight, (freq, _) in
                                            zip(local_weights, part)], axis=0))
        return np.mean(np.stack(component_values), axis=0) if component_values else frequencies
    component_values = {component: balanced({component: part})
                        for component, part in component_samples.items()}
    component_balanced = balanced(component_samples)
    panel_frequencies = {panel: balanced({panel: part}) for panel, part in panel_samples.items()}
    denominator = sum((weight ** 2) * len(_scored_positions(query))
                      for query, weight in zip(queries, weights))
    effective_positions = (counts.sum() ** 2 / denominator) if denominator > 0 else 0.
    return {'aa': list(AA), 'counts': counts.tolist(), 'frequencies': frequencies.tolist(),
            'component_balanced_frequencies': component_balanced.tolist(),
            'component_frequencies': {key: value.tolist() for key, value in component_values.items()},
            'panel_frequencies': {key: value.tolist() for key, value in panel_frequencies.items()},
            'positions': int(sum(len(_scored_positions(query)) for query in queries)),
            'effective_positions': float(effective_positions),
            'per_query': per_query}
