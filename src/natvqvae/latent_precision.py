"""Separate repeated Monte Carlo error from biological-group variation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .data import sha256_file, write_json
from .esm_baseline import _metric_components
from .evaluate import _bootstrap, _panel_region
from .fragment_comparison import _CONDITIONS, _METRICS, _require


def _pool_query_repeats(rows):
    """Combine equal-size prior samples in probability space, not by averaging NLL."""
    sites = rows[0]['sites']
    log_marginals = np.asarray([row['prior_site_marginal_log_probs'] for row in rows], dtype=float)
    _require(log_marginals.shape == (len(rows), sites) and np.isfinite(log_marginals).all() and
             np.all(log_marginals <= 1e-6) and np.allclose(
                 -log_marginals.mean(axis=1), [row[_METRICS[1]] for row in rows], rtol=0, atol=2e-6),
             'Stored site marginals do not match their NLL')
    joint = np.asarray([-sites * row[_METRICS[0]] for row in rows])
    return {**rows[0],
            _METRICS[0]: float(-(np.logaddexp.reduce(joint) - np.log(len(rows))) / sites),
            _METRICS[1]: float(-(np.logaddexp.reduce(log_marginals, axis=0) - np.log(len(rows))).mean())}


def _repeat_scores(path, repeats):
    rows = [[] for _ in range(repeats)]
    tasks = [dict() for _ in range(repeats)]
    with Path(path).open() as handle:
        for line in handle:
            row = json.loads(line)
            repeat = row['repeat']
            _require(type(repeat) is int and 0 <= repeat < repeats, 'Unexpected MC repeat')
            key = row['sequence_id'], row['panel']
            _require(key not in tasks[repeat], 'Duplicate query/repeat')
            positions = tuple(row['positions_0based'])
            _require(len(positions) == row['sites'] > 0 and len(set(positions)) == len(positions),
                     'Invalid target positions')
            _require(np.isfinite(row['weight']) and row['weight'] > 0 and
                     all(np.isfinite(row[metric]) for metric in _METRICS), 'Invalid query score or weight')
            tasks[repeat][key] = (row['component'], positions, row['weight'])
            rows[repeat].append({**row, 'region': _panel_region(row['panel'])})
    _require(bool(tasks[0]) and all(task == tasks[0] for task in tasks), 'Incomplete paired MC repeats')
    # Order determines evaluation batches and therefore the shared random streams.
    _require(all(list(task) == list(tasks[0]) for task in tasks), 'MC query order differs')
    scores = {metric: [_metric_components(batch, metric) for batch in rows] for metric in _METRICS}
    groups = sorted(scores[_METRICS[0]][0])
    regions = sorted({region for group in scores[_METRICS[0]][0].values() for region in group})
    _require(all(set(group) == set(regions) for group in scores[_METRICS[0]][0].values()),
             'Components have different region coverage')
    arrays = {
        (metric, region): np.asarray([[sample[group][region] for group in groups] for sample in samples])
        for metric, samples in scores.items() for region in regions}
    pooled = {1: arrays}
    for factor in range(2, repeats + 1):
        if repeats % factor:
            continue
        pooled_rows = [[_pool_query_repeats(list(query)) for query in zip(*rows[start:start + factor])]
                       for start in range(0, repeats, factor)]
        pooled_scores = {metric: [_metric_components(batch, metric) for batch in pooled_rows] for metric in _METRICS}
        pooled[factor] = {(metric, region): np.asarray([[sample[group][region] for group in groups] for sample in samples])
                          for metric, samples in pooled_scores.items() for region in regions}
    return tasks[0], groups, arrays, pooled


def _difference_summary(values, bootstrap_repeats, bootstrap_seed):
    """Input axes are MC repeat and component, with pipeline seeds already paired."""
    values = np.asarray(values, dtype=float)
    _require(values.ndim == 2 and values.shape[0] >= 1 and values.shape[1] > 0 and
             np.isfinite(values).all(), 'MC precision needs finite paired repeats and components')
    repeat_means = values.mean(axis=1)
    group_means = values.mean(axis=0)
    interval = (_bootstrap(group_means, bootstrap_repeats, bootstrap_seed)['ci95']
                if len(group_means) > 1 else None)
    return {'groups': len(group_means), 'mc_repeats': len(repeat_means),
            'mean': float(group_means.mean()), 'group_bootstrap_ci95': interval,
            'repeat_estimates': repeat_means.tolist(),
            'mc_standard_error': (float(repeat_means.std(ddof=1) / np.sqrt(len(repeat_means)))
                                  if len(repeat_means) > 1 else None)}


def summarize_latent_precision(comparison_dir, output):
    """Reuse a completed fragment comparison and its hash-verified scoring inputs."""
    comparison_dir, output = Path(comparison_dir), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Latent precision output already contains files: {output}')
    summary_path, identity_path = (comparison_dir / name for name in ('summary.json', 'identity.json'))
    summary = json.loads(summary_path.read_text())
    identity = json.loads(identity_path.read_text())
    _require(summary['status'] == 'COMPLETED' and summary.get('model_arm', 'QVAE') == 'QVAE',
             'A completed QVAE fragment comparison is required')
    repeats = summary['data']['mc_repeats']
    _require(repeats >= 2, 'At least two MC repeats are required')
    hashes = {str(Path(path).resolve()): digest for path, digest in identity['input_sha256'].items()}
    for path, digest in hashes.items():
        _require(sha256_file(path) == digest, f'Comparison input SHA256 differs: {path}')
    inputs = {str(path.resolve()): sha256_file(path) for path in (summary_path, identity_path)}
    seeds = [int(pair['downstream']) for pair in summary['pipeline_seed_pairs']]
    _require(len(seeds) == len(set(seeds)) == 3, 'Three distinct pipeline seeds are required')
    scores, pooled, run_inputs = {}, {}, {}
    reference_tasks = reference_groups = reference_config = reference_code = None
    for name, run in summary['runs'].items():
        key = run['condition'], int(run['downstream_seed'])
        _require(key not in scores, 'Duplicate condition/pipeline seed')
        candidates = [Path(path) for path in hashes
                      if Path(path).name == 'per_query.jsonl' and Path(path).parent.name == name]
        _require(len(candidates) == 1, f'Expected one verified query file for {name}')
        path = candidates[0]
        metadata = []
        for filename in ('identity.json', 'evaluation.json'):
            item = path.with_name(filename)
            _require(str(item) in hashes, f'Unverified evaluation metadata: {item}')
            metadata.append(json.loads(item.read_text()))
        evaluation_identity, evaluation = metadata
        _require(evaluation['status'] == 'COMPLETED' and evaluation['split'] == summary['split'] and
                 evaluation_identity.get('arm', 'QVAE') == 'QVAE' and
                 evaluation_identity['limit'] is None and evaluation['repeats'] == repeats and
                 evaluation['checkpoint_sha256'] == run['checkpoint']['sha256'],
                 f'Evaluation identity differs: {name}')
        config = evaluation_identity['config']
        _require(config['repeats'] == repeats and config['sample_count'] == evaluation['sample_count'],
                 f'Evaluation sampling configuration differs: {name}')
        tasks, groups, scores[key], pooled[key] = _repeat_scores(path, repeats)
        _require(len(tasks) == summary['data']['queries'] and len(groups) == summary['data']['groups'] and
                 len({uid for uid, _ in tasks}) == summary['data']['sequences'] and
                 evaluation['queries'] == len(tasks) * repeats, f'Comparison task counts differ: {name}')
        if reference_tasks is None:
            reference_tasks, reference_groups, reference_config = tasks, groups, config
            reference_code = evaluation_identity['code_sha256']
        _require(tasks == reference_tasks and list(tasks) == list(reference_tasks) and
                 groups == reference_groups and config == reference_config and
                 evaluation_identity['code_sha256'] == reference_code,
                 f'Paired tasks, sampling settings or scoring implementation differ: {name}')
        for metric in _METRICS:
            _require(np.isclose(scores[key][metric, 'all_panels'].mean(), evaluation[metric], rtol=1e-9),
                     f'Reconstructed score differs from evaluation summary: {name}')
        run_inputs[name] = {'per_query': str(path), 'sample_count': evaluation['sample_count'],
                            'precision': evaluation_identity['precision'], 'checkpoint': run['checkpoint']}
    _require(set(scores) == {(condition, seed) for condition in _CONDITIONS for seed in seeds},
             'Incomplete three-condition, three-seed comparison')
    bootstrap = summary['bootstrap']
    _require(bootstrap['repetitions'] > 0, 'Bootstrap repetitions must be positive')
    contrasts = [(f'{condition}:joint_minus_site_marginal', condition, _METRICS[0], condition, _METRICS[1])
                 for condition in _CONDITIONS]
    contrasts += [(f'external_minus_self:{metric}', 'external', metric, 'self', metric) for metric in _METRICS]
    estimates, group_rows, sensitivity = [], [], []
    regions = sorted({region for _, region in next(iter(scores.values()))})
    for name, left, left_metric, right, right_metric in contrasts:
        for region in regions:
            paired = [scores[left, seed][left_metric, region] - scores[right, seed][right_metric, region]
                      for seed in seeds]
            # Same evaluation RNG seeds are used across fitted models; average before computing SE.
            scopes = [(str(seed), values) for seed, values in zip(seeds, paired)]
            scopes.append(('mean_of_pipeline_seeds', np.mean(paired, axis=0)))
            for scope, values in scopes:
                estimates.append({'comparison': name, 'region': region, 'scope': scope,
                                  **_difference_summary(values, bootstrap['repetitions'], bootstrap['seed'])})
                for repeat, row in enumerate(values):
                    group_rows.extend({'comparison': name, 'region': region, 'scope': scope,
                                       'repeat': repeat, 'component': group, 'difference': float(value)}
                                      for group, value in zip(reference_groups, row))
            reference_mean = float(np.mean(paired))
            for factor in sorted(next(iter(pooled.values()))):
                values = np.mean([pooled[left, seed][factor][left_metric, region] -
                                  pooled[right, seed][factor][right_metric, region] for seed in seeds], axis=0)
                value = _difference_summary(values, bootstrap['repetitions'], bootstrap['seed'])
                sensitivity.append({'comparison': name, 'region': region,
                                    'samples_per_estimate': reference_config['sample_count'] * factor,
                                    'pooled_original_repeats': factor, **value,
                                    'change_from_unpooled_mean': value['mean'] - reference_mean})
    result = {'status': 'COMPLETED', 'split': summary['split'], 'data_status': summary['data_status'],
              'data': summary['data'], 'units': 'nats per target residue', 'runs': run_inputs,
              'sampling_config': reference_config, 'estimates': estimates, 'sample_count_sensitivity': sensitivity,
              'method': {'mc_standard_error': 'SD of paired repeat estimates / sqrt(repeats)',
                         'aggregation': 'equal component means within region; panel weights only for all_panels',
                         'pipeline_seeds': 'paired before averaging; shared evaluation random streams retained',
                         'group_bootstrap': 'repeat-averaged component scores; conditional on fitted pipeline seeds',
                         'sample_pooling': 'logmeanexp of joint probabilities and of each site marginal across consecutive equal-size repeats; no additional draws',
                         'limitations': [f'SE is estimated from only {repeats} Monte Carlo repeats.',
                                         'Repeat variability does not measure finite-sample log-probability bias.',
                                         'Pooled estimates share samples across sample counts; they do not establish asymptotic convergence.',
                                         'Numerical forward/probability precision is not tested by repeated sampling.',
                                         'No test-based model selection or new probability scoring is performed.']}}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'summary.json', result)
    write_json(output / 'identity.json', {'input_sha256': {**hashes, **inputs},
               'implementation_sha256': {name: sha256_file(Path(__file__).with_name(name))
                                         for name in ('latent_precision.py', 'esm_baseline.py', 'evaluate.py')}})
    with (output / 'group_repeat_differences.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(group_rows)
    lines = ['# 潜变量与迁移比较的 Monte Carlo 精度', '',
             f'{summary["split"]}：{summary["data"]["sequences"]} 条序列、'
             f'{summary["data"]["groups"]} 个组，每任务 {repeats} 次重复、'
             f'每次 {reference_config["sample_count"]} 个潜变量样本。单位为 nats/目标残基；负差值表示前者 NLL 较低。', '',
             '下表为 NMNAT，先在同一重复内平均三个配对训练种子的组级差值，再对组等权平均。'
             'MC SE 为重复均值的样本标准差除以重复数平方根。组区间先平均 MC 重复，再重采样组；'
             '两者回答不同问题，不把 MC 重复当作独立序列组。', '',
             '| 比较 | 均值 | MC SE | 95% 组 bootstrap 区间 |', '| --- | ---: | ---: | --- |']
    for row in estimates:
        if row['scope'] == 'mean_of_pipeline_seeds' and row['region'] == 'NMNAT':
            interval = row['group_bootstrap_ci95']
            interval_text = 'NA' if interval is None else f'[{interval[0]:+.8f}, {interval[1]:+.8f}]'
            lines.append(f'| {row["comparison"]} | {row["mean"]:+.8f} | {row["mc_standard_error"]:.8f} | {interval_text} |')
    lines += ['', '将连续重复的联合概率及每个位点边缘概率分别用 logmeanexp 合并，得到较大样本量的估计；'
              '这不是平均 NLL，也没有生成新样本。不同样本量之间共享原始样本，仅用于检查本次有限样本敏感性。'
              '全部重复合并后只剩一次估计，MC SE 记为 NA，不能据此声称已收敛或偏差消失。', '',
              '| NMNAT 比较 | 每次样本数 | 剩余独立重复数 | 均值 | 相对未合并均值变化 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for row in sensitivity:
        if row['region'] == 'NMNAT':
            lines.append(f'| {row["comparison"]} | {row["samples_per_estimate"]} | {row["mc_repeats"]} | '
                         f'{row["mean"]:+.8f} | {row["change_from_unpooled_mean"]:+.8f} |')
    lines += ['', f'原始每任务仅有 {repeats} 次重复，SE 本身仍不稳定；它不衡量有限样本 log 概率估计偏差，也不检验 '
              'bfloat16/float32 数值误差。区间以当前拟合的三个训练重复为条件，不覆盖新的训练过程。'
              '因此，差值超过 MC SE 不能单独证明精确分布的联合建模收益。逐种子、逐区域及逐重复差值保存在 JSON/CSV 中。', '']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    return result
