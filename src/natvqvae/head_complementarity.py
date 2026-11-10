"""Decompose existing head-mixture scores without fitting or forwarding a model."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .data import load_yaml, sha256_file, write_json
from .evaluate import _bootstrap
from .fragment_comparison import _require
from .head_mixture import REGIONS, _csv, _probability_inputs, _read_probabilities

METHODS = ('single_mixture', 'region_calibrated_mixture')
METRICS = ('mass', 'esm_nll', 'task_nll', 'mixture_nll', 'mixture_minus_esm',
           'mixture_minus_task', 'weighted_head_cost', 'jensen_gap', 'esm_accuracy',
           'task_accuracy', 'mixture_accuracy', 'accuracy_change', 'esm_rank',
           'task_rank', 'mixture_rank', 'rank_change')


def site_metrics(esm, task, labels, alpha, esm_temperature=1., task_temperature=1.):
    """Rank ties use the stored AA order, consistently with argmax."""
    esm, task = np.asarray(esm, float), np.asarray(task, float)
    labels = np.asarray(labels)
    _require(esm.shape == task.shape == (len(labels), 20) and labels.dtype.kind in 'iu' and
             ((labels >= 0) & (labels < 20)).all(), 'Invalid complete distributions or labels')
    for values in (esm, task):
        _require(np.isfinite(values).all() and
                 np.allclose(np.logaddexp.reduce(values, axis=1), 0, atol=2e-6, rtol=0),
                 'Expected normalized complete probabilities')
    alpha = np.broadcast_to(np.asarray(alpha, float), (len(labels),))
    _require(np.isfinite(alpha).all() and ((alpha > 0) & (alpha < 1)).all(),
             'Explanation requires interior mixture weights')
    scaled = []
    for values, temperature in ((esm, esm_temperature), (task, task_temperature)):
        temperature = np.broadcast_to(np.asarray(temperature, float), (len(labels),))
        _require(np.isfinite(temperature).all() and (temperature > 0).all(), 'Invalid temperature')
        values = values / temperature[:, None]
        scaled.append(values - np.logaddexp.reduce(values, axis=1)[:, None])
    esm, task = scaled
    mixture = np.logaddexp(np.log1p(-alpha)[:, None] + esm, np.log(alpha)[:, None] + task)
    top = [values.argmax(1) for values in (esm, task, mixture)]
    correct = [values == labels for values in top]
    ranks = [1 + ((values > values[np.arange(len(labels)), labels, None]) |
                  ((values == values[np.arange(len(labels)), labels, None]) &
                   (np.arange(20)[None] < labels[:, None]))).sum(1) for values in (esm, task, mixture)]
    losses = [-values[np.arange(len(labels)), labels] for values in (esm, task, mixture)]
    e, d, m = losses
    cost, gap = alpha * (d - e), (1 - alpha) * e + alpha * d - m
    agreement = top[0] == top[1]
    _require(np.all(gap >= -1e-12) and np.allclose(m - e, cost - gap, atol=1e-12) and
             np.all(top[2][agreement] == top[0][agreement]), 'Mixture invariants failed')
    metrics = np.stack([np.ones(len(labels)), e, d, m, m - e, m - d, cost, gap,
                        *correct, correct[2].astype(float) - correct[0], *ranks, ranks[2] - ranks[0]], axis=1)
    return metrics, agreement, correct, np.exp(-e), np.exp(-d)


def component_contributions(metrics, rows, boundaries, selector, region):
    """Keep excluded sites at zero; never renormalize a stratum's contribution."""
    groups = sorted({row['component'] for row in rows})
    ids = np.array([groups.index(row['component']) for row in rows])
    included = np.array([region == 'all_panels' or row['region'] == region for row in rows])
    weights = np.array([row['weight'] if region == 'all_panels' else 1. for row in rows]) * included
    totals = np.bincount(ids, weights=weights, minlength=len(groups))
    _require((totals > 0).all(), 'Missing component/region')
    lengths = np.diff(boundaries)
    _require(len(lengths) == len(rows) and (lengths > 0).all() and boundaries[0] == 0 and
             boundaries[-1] == len(metrics) == len(selector), 'Site boundaries differ')
    site_ids = np.repeat(ids, lengths)
    site_weights = np.repeat(weights / totals[ids] / lengths, lengths) * selector
    values = np.stack([np.bincount(site_ids, weights=site_weights * metric, minlength=len(groups))
                       for metric in metrics.T], axis=1)
    counts = np.bincount(site_ids, weights=selector * np.repeat(included, lengths), minlength=len(groups))
    return groups, values, counts


def _partitions(agreement, correct, esm_probability, task_probability, bins):
    partitions = {'all': {'all': np.ones(len(agreement), bool)},
                  'head_top1': {'same': agreement, 'different': ~agreement},
                  'esm_to_mixture_top1': {
                      f'esm_{e}_mixture_{m}': (correct[0] == (e == 'correct')) &
                                            (correct[2] == (m == 'correct'))
                      for e in ('correct', 'wrong') for m in ('correct', 'wrong')}}
    for name, probability in (('esm_true_probability', esm_probability), ('task_true_probability', task_probability)):
        partitions[name] = {'below_0.01': probability < bins[0],
                            '0.01_to_0.05': (probability >= bins[0]) & (probability < bins[1]),
                            'at_least_0.05': probability >= bins[1]}
    return partitions


def run_head_complementarity(config_path, output, split='test'):
    config = load_yaml(config_path)
    _require(split in ('validation', 'test') and config['true_probability_bins'] == [.01, .05],
             'Unexpected descriptive protocol')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    calibration = load_yaml(config['calibration_config'])
    # Existing loader verifies task identities, model hashes, normalization, and old-score parity.
    spec, rows, _, boundaries, parity, identity = _read_probabilities(calibration, split, [1.])
    score_root = Path(config['scores'][split])
    summary = json.loads((score_root / 'summary.json').read_text())
    fitted = json.loads(Path(config['fitted']).read_text())
    module = Path(__file__).with_name('head_mixture.py')
    _require(summary['status'] == 'COMPLETED' and fitted['status'] == 'FITTED_ON_VALIDATION' and
             summary['split'] == split and
             summary['implementation_sha256'] == fitted['implementation_sha256'] == sha256_file(module) and
             summary['weights'] == fitted['weights'] and fitted['fit_split'] == 'validation',
             'Frozen scoring protocol differs')
    for path, digest in summary['input_sha256'].items():
        _require(sha256_file(path) == digest, f'Scoring input changed: {path}')
    reference = {(r['method'], int(r['seed']), r['component'], r['region']): float(r['nll'])
                 for r in csv.DictReader((score_root / 'group_scores.csv').open())}
    groups = sorted({row['component'] for row in rows})
    lengths = np.diff(boundaries)
    distributions = Path(calibration['probabilities'][split]) / 'site_log_probs.npz'
    all_groups, strata, robustness, checks = [], [], {}, {}
    with np.load(distributions, allow_pickle=False) as archive:
        labels, esm = archive['labels'], archive['esm']
        for method in METHODS:
            by_cell, count_cell = {}, {}
            for seed in spec['seeds']:
                choices = []
                for row in rows:
                    fit = summary['weights'] if split == 'test' else summary['excluded_component_weights'][row['component']]
                    choice = fit['single']['all_panels'] if method == 'single_mixture' else fit['region'][row['region']]
                    choices.append((choice['uncalibrated_alpha'], 1., 1.) if method == 'single_mixture' else
                                   (choice['alpha'], choice['esm_temperature'], choice['task_temperature']))
                parameters = np.repeat(np.asarray(choices), lengths, axis=0)
                metrics, agreement, correct, pe, pd = site_metrics(
                    esm, archive[str(seed)], labels, *parameters.T)
                partitions = _partitions(agreement, correct, pe, pd, config['true_probability_bins'])
                for region in REGIONS:
                    total = None
                    for axis, categories in partitions.items():
                        reconstructed = []
                        for category, selector in categories.items():
                            _, values, counts = component_contributions(metrics, rows, boundaries, selector, region)
                            key = region, axis, category
                            by_cell.setdefault(key, []).append(values)
                            count_cell.setdefault(key, []).append(counts)
                            reconstructed.append(values)
                            if axis == 'all':
                                total = values
                                for g, group in enumerate(groups):
                                    expected = reference[method, seed, group, region]
                                    _require(abs(values[g, 3] - expected) < 2e-6, 'Group NLL differs from frozen scoring')
                                    all_groups.append({'method': method, 'seed': seed, 'component': group,
                                                       'region': region, **dict(zip(METRICS, values[g].tolist()))})
                        _require(np.allclose(sum(reconstructed), total, atol=1e-11, rtol=0),
                                 'Partition contributions do not reconstruct overall scores')
            for (region, axis, category), seed_values in by_cell.items():
                values = np.mean(seed_values, axis=0)
                means = values.mean(0)
                counts = np.asarray(count_cell[region, axis, category])
                estimate = _bootstrap(values[:, 4], spec['bootstrap_repeats'], spec['bootstrap_seed'])
                strata.append({'method': method, 'region': region, 'partition': axis, 'category': category,
                               'site_observations_across_seeds': int(counts.sum()),
                               'components_with_sites': int((counts.sum(0) > 0).sum()),
                               **dict(zip(METRICS, means.tolist())),
                               'delta_ci_low': estimate['ci95'][0], 'delta_ci_high': estimate['ci95'][1],
                               'conditional_delta': float(means[4] / means[0]) if means[0] else None})
                if axis == 'all':
                    delta = values[:, 4]
                    oracle = values[:, 3] - np.minimum(values[:, 1], values[:, 2])
                    robustness[f'{method}:{region}'] = {
                        'groups_improved': int((delta < 0).sum()), 'groups': len(groups),
                        'median_difference': float(np.median(delta)),
                        'mean_without_most_improved_group': float(np.delete(delta, delta.argmin()).mean()),
                        'groups_better_than_groupwise_best_head': int((oracle < 0).sum()),
                        'mean_minus_groupwise_best_head': float(oracle.mean()),
                        'per_seed_difference': np.asarray(seed_values)[:, :, 4].mean(1).tolist()}
            checks[method] = 'All group NLLs match the frozen scoring; every partition reconstructs all metrics.'
    paths = _probability_inputs(calibration, split) | {
        str(config_path), config['calibration_config'], config['fitted'],
        str(score_root / 'summary.json'), str(score_root / 'group_scores.csv')}
    result = {'status': 'COMPLETED', 'split': split, 'groups': len(groups), 'queries': len(rows),
              'sites_per_seed': len(labels), 'seeds': spec['seeds'], 'methods': METHODS,
              'input_sha256': {p: sha256_file(p) for p in sorted(paths)},
              'implementation_sha256': {str(p): sha256_file(p) for p in (Path(__file__), module)},
              'baseline_parity': parity, 'checks': checks, 'robustness': robustness, 'strata': strata,
              'definitions': {
                  'aggregation': 'Mean sites within query, equal queries within component/region, panel-weighted queries for all_panels; then equal components and seeds.',
                  'contributions': 'A subset keeps its original denominator; absent sites contribute zero. Partition masses sum to 1 and metric contributions sum to the overall metric.',
                  'conditional_delta': 'Mean additive contribution divided by mean weighted mass; not an equal average over nonempty components.',
                  'counts': 'Target-site observations across queries and three seeds, not unique residues or independent observations.',
                  'jensen': 'mixture_minus_esm = alpha*(task_nll-esm_nll) - ((1-alpha)*esm_nll+alpha*task_nll-mixture_nll), evaluated at each site before aggregation.',
                  'rank': 'One-based rank; exact ties broken by stored AA order, also used for top1.',
                  'uncertainty': '2000 conditional component bootstrap samples of additive contributions after seed averaging; no refitting or site bootstrap.',
                  'oracle': 'Choose lower component-mean single-head NLL after averaging seeds; descriptive hindsight reference, not a deployable model.',
                  'limits': 'Post hoc decomposition, overlapping mask contexts, historically evaluated held-out set. Neither independent confirmation nor causal biological information attribution. Panel parameters apply to the entire query, not residue-specific domain routing.'}}
    output.mkdir(parents=True, exist_ok=True)
    _csv(output / 'group_scores.csv', all_groups)
    _csv(output / 'stratum_contributions.csv', strata)
    lines = ['# 预测头互补的位点分解', '', result['definitions']['aggregation'], '',
             result['definitions']['contributions'], '', result['definitions']['limits'], '',
             '| 方法 | 区域 | 分层 | 类别 | 加权占比 | 对总NLL差的贡献 | 条件组区间 |',
             '|---|---|---|---|---:|---:|---|']
    for row in strata:
        if row['region'] in ('NMNAT', 'Nudix') and row['partition'] in ('all', 'head_top1', 'esm_true_probability'):
            lines.append(f'| {row["method"]} | {row["region"]} | {row["partition"]} | {row["category"]} | '
                         f'{row["mass"]:.6f} | {row["mixture_minus_esm"]:+.6f} | '
                         f'[{row["delta_ci_low"]:+.6f}, {row["delta_ci_high"]:+.6f}] |')
    lines += ['', '## NMNAT整体诊断', '',
              '以下准确率与真实残基平均排序沿用相同query/component权重；排序1为最高。', '',
              '| 方法 | 加权单头代价 | Jensen差 | ESM/任务/混合 top1准确率 | 混合减ESM平均排序 | 改善组数 | 去掉最大改善组的均差 | 优于事后逐组最佳单头 |',
              '|---|---:|---:|---|---:|---:|---:|---|']
    for row in strata:
        if row['region'] == 'NMNAT' and row['partition'] == 'all':
            robust = robustness[f'{row["method"]}:NMNAT']
            lines.append(f'| {row["method"]} | {row["weighted_head_cost"]:+.6f} | {row["jensen_gap"]:.6f} | '
                         f'{row["esm_accuracy"]:.6f}/{row["task_accuracy"]:.6f}/{row["mixture_accuracy"]:.6f} | '
                         f'{row["rank_change"]:+.6f} | {robust["groups_improved"]}/{len(groups)} | '
                         f'{robust["mean_without_most_improved_group"]:+.6f} | '
                         f'{robust["groups_better_than_groupwise_best_head"]}/{len(groups)}, '
                         f'均差{robust["mean_minus_groupwise_best_head"]:+.6f} |')
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    write_json(output / 'summary.json', result)
    return {'status': 'COMPLETED', 'split': split, 'output': str(output)}
