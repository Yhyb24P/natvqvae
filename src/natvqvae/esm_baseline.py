"""Evaluate the frozen ESM-2 amino-acid head under the NatV masking task."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import transformers

from .data import PopulationData, load_yaml, mask_panels, sha256_file, write_json
from .evaluate import _bootstrap
from .model import AA, FrozenEncoder


def score_sites(log_probs, sequence, positions):
    """Return target-site log probabilities, never an unmasked pseudo-likelihood."""
    if not positions or len(set(positions)) != len(positions):
        raise ValueError('Scoring needs a nonempty set of unique target positions')
    if min(positions) < 0 or max(positions) >= len(sequence):
        raise ValueError('Target position outside sequence')
    labels = torch.tensor([AA.index(sequence[i]) for i in positions], device=log_probs.device)
    return log_probs[positions].gather(-1, labels[:, None]).squeeze(-1)


def summarize_rows(rows):
    """Match the existing equal-group, per-residue masking-panel aggregation."""
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row['component']][row['region']].append(row)
    component_scores = {}
    for group, regions in groups.items():
        component_scores[group] = {}
        all_rows = []
        for region, entries in regions.items():
            component_scores[group][region] = float(np.mean(
                [row['factorized_nll_per_residue'] for row in entries]))
            all_rows.extend(entries)
        component_scores[group]['all_panels'] = sum(
            row['weight'] * row['factorized_nll_per_residue'] for row in all_rows
        ) / sum(row['weight'] for row in all_rows)
    regions = sorted({region for scores in component_scores.values() for region in scores})
    return {'component_scores': component_scores,
            'mean_nll_per_residue': {
                region: float(np.mean([scores[region] for scores in component_scores.values()
                                       if region in scores])) for region in regions}}


def evaluate_esm(config_path, sources_path, target_path, output, device=None,
                 split='validation', test_data=None):
    """Score validation by default; test requires explicit recovered input data."""
    if split not in ('validation', 'test') or (split == 'test') != (test_data is not None):
        raise ValueError('Test scoring requires explicit test data; validation must not use it')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'ESM evaluation output already contains files: {output}')
    data = PopulationData(sources_path, config_path, target_path)
    if split == 'test':
        from .test_data import load_recovered_test
        records = load_recovered_test(data, test_data)
        queries = [{'record': record, 'panel': panel} for record in records
                   for panel in mask_panels(record, data.study['training']['mask_seed'])]
    else:
        records, queries = data.validation, data.validation_queries()
    if not queries:
        raise ValueError('No evaluation queries')
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    batch_size = int(data.study['evaluation']['batch_size'])
    if batch_size < 1:
        raise ValueError('Evaluation batch size must be positive')
    esm_path = Path(data.sources['esm_model'])
    input_paths = [config_path, sources_path, target_path, data.sources['population'],
                   data.sources['splits']['training'], data.sources['splits']['validation'],
                   *sorted(esm_path.glob('*.json')), esm_path / 'vocab.txt',
                   esm_path / 'model.safetensors']
    if split == 'test':
        input_paths.extend([data.sources['splits']['test'],
                            *(Path(test_data) / name for name in ('test.csv', 'test.fasta', 'manifest.json'))])
    identity = {
        'command': 'esm-evaluate', 'split': split,
        'input_sha256': {str(path): sha256_file(path) for path in input_paths},
        'code_sha256': {name: sha256_file(Path(__file__).with_name(name))
                       for name in ('esm_baseline.py', 'model.py', 'data.py') +
                       (('test_data.py',) if split == 'test' else ())},
        'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--porcelain'], text=True).splitlines(),
        'python': platform.python_version(), 'torch': torch.__version__,
        'transformers': transformers.__version__, 'device': str(device),
        'batch_size': batch_size, 'amino_acid_order': AA,
        'precision': 'bfloat16 forward, float32 log-softmax' if device.type == 'cuda' else 'float32',
        'normalization': 'restrict logits to the same 20 amino acids before log-softmax',
        'masking': 'all positions in the query masked jointly before ESM encoding',
        'test_scoring': split == 'test', 'training_performed': False,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'identity.json', identity)
    encoder = FrozenEncoder(esm_path, device=device)
    started = time.monotonic()
    rows = []
    with (output / 'per_query.jsonl').open('w') as handle:
        for start in range(0, len(queries), batch_size):
            batch = queries[start:start + batch_size]
            sequences = [q['record'].sequence for q in batch]
            positions = [q['panel']['positions'] for q in batch]
            probabilities, valid, mask = encoder.amino_acid_log_probs(sequences, positions)
            for index, query in enumerate(batch):
                sites = positions[index]
                if int(mask[index].sum()) != len(sites) or not bool(valid[index, sites].all()):
                    raise ValueError('Encoded targets do not match the scoring positions')
                site_logp = score_sites(probabilities[index], sequences[index], sites)
                if not bool(torch.isfinite(site_logp).all()):
                    raise FloatingPointError('Nonfinite ESM site log probability')
                row = {
                    'sequence_id': query['record'].sequence_id,
                    'component': query['record'].component,
                    'panel': query['panel']['panel'], 'region': query['panel']['region'],
                    'sites': len(sites), 'positions_0based': sites,
                    'target_site_log_probs': site_logp.cpu().tolist(),
                    'factorized_nll_per_residue': float(-site_logp.mean().item()),
                    'weight': float(query['panel']['weight']),
                }
                rows.append(row)
                handle.write(json.dumps(row) + '\n')
            handle.flush()
            print(json.dumps({'completed_queries': len(rows), 'total_queries': len(queries),
                              'seconds': time.monotonic() - started}), flush=True)
    summary = {
        'status': 'COMPLETED', 'split': split, 'sequences': len(records),
        'groups': len(Counter(r.component for r in records)),
        'queries': len(rows), 'model': 'frozen ESM-2 masked-amino-acid head',
        'estimator': 'exact product of 20-amino-acid site probabilities for the jointly masked input',
        'aggregation': 'site mean within query, panel mean within region, sequence mean within group, equal groups',
        'seconds': time.monotonic() - started, 'identity': 'identity.json',
        **summarize_rows(rows),
    }
    write_json(output / 'evaluation.json', summary)
    return summary


_COMPARISON_REGIONS = ('NMNAT', 'Nudix', 'all_panels')
_COMPARISON_METRICS = ('prior_joint_nll_per_residue', 'marginal_nll_per_residue')


def plot_esm_comparison(comparison_dir):
    """Export the existing group comparisons without fitting or rescoring."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    directory = Path(comparison_dir)
    source = directory / 'comparison.json'
    result = json.loads(source.read_text())
    outputs = [directory / f'esm_comparison.{suffix}' for suffix in ('pdf', 'png')]
    if any(path.exists() for path in outputs):
        raise FileExistsError('ESM comparison figure already exists')
    titles = ['NMNAT', 'Nudix', 'All masks']
    colors = ['#286D9B', '#B96524']
    with plt.rc_context({'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.6), sharey=True, layout='constrained')
        for axis, region, title in zip(axes, _COMPARISON_REGIONS, titles):
            axis.axvline(0, color='.6', lw=.8, linestyle='--', zorder=1)
            for y, (arm, color) in enumerate(zip(('random', 'auxinit'), colors)):
                item = result['arms'][arm]['comparisons']['prior_joint_nll_per_residue'][region]
                mean, (low, high) = item['mean'], item['ci95']
                axis.errorbar(mean, 1-y, xerr=[[mean-low], [high-mean]], fmt='o',
                              color=color, capsize=3, lw=1.4, markersize=5, zorder=2)
            axis.set_title(title)
            axis.set_xlim(-.15, .20)
            axis.set_ylim(-.65, 1.65)
            axis.set_xticks([-.1, 0, .1, .2])
            axis.spines[['top', 'right']].set_visible(False)
            axis.set_yticks([1, 0], ['Random initialization', 'Auxiliary initialization'])
        fig.supxlabel('NatV model − frozen ESM-2 NLL (nats per target residue)', fontsize=9)
        fig.savefig(outputs[0], bbox_inches='tight')
        fig.savefig(outputs[1], dpi=300, bbox_inches='tight')
        plt.close(fig)
    identity = {'source': str(source), 'source_sha256': sha256_file(source),
                'implementation_sha256': sha256_file(__file__),
                'matplotlib': matplotlib.__version__,
                'interval': '95% group bootstrap, conditional on the three fitted training seeds',
                'outputs': {path.name: sha256_file(path) for path in outputs}}
    write_json(directory / 'figure_identity.json', identity)
    return identity


def _query_key(row):
    return row['sequence_id'], row['panel']


def _checked_query_rows(path, expected, component_sizes, repeats=None):
    """Check task identity, then average repeated NLL estimates within a query."""
    if not expected or (repeats is not None and repeats < 1):
        raise ValueError('Comparison needs nonempty queries and a positive repeat count')
    grouped = defaultdict(dict)
    stored_positions = 0
    metrics = _COMPARISON_METRICS if repeats is not None else ('factorized_nll_per_residue',)
    with Path(path).open() as handle:
        for line in handle:
            row = json.loads(line)
            key = _query_key(row)
            if key not in expected:
                raise ValueError(f'Unexpected query in {path}: {key}')
            reference = expected[key]
            if (row['component'] != reference['component'] or
                    int(row['sites']) != len(reference['positions_0based'])):
                raise ValueError(f'Query group or site count differs in {path}: {key}')
            if 'positions_0based' in row:
                stored_positions += 1
                if row['positions_0based'] != reference['positions_0based']:
                    raise ValueError(f'Masked positions differ in {path}: {key}')
            elif repeats is None:
                raise ValueError(f'ESM query lacks masked positions: {key}')
            expected_weight = reference['weight']
            if repeats is not None:
                expected_weight /= component_sizes[reference['component']]
            if not np.isclose(float(row['weight']), expected_weight, rtol=1e-10, atol=1e-12):
                raise ValueError(f'Query weight differs in {path}: {key}')
            repeat = int(row['repeat']) if repeats is not None else 0
            if repeat in grouped[key]:
                raise ValueError(f'Duplicate query/repeat in {path}: {key}, {repeat}')
            if any(not np.isfinite(float(row[metric])) for metric in metrics):
                raise ValueError(f'Nonfinite query score in {path}: {key}')
            if repeats is None:
                logp = np.asarray(row['target_site_log_probs'], dtype=float)
                if (row['region'] != reference['region'] or len(logp) != row['sites'] or
                        not np.isfinite(logp).all() or
                        not np.isclose(-logp.mean(), row['factorized_nll_per_residue'], atol=2e-6)):
                    raise ValueError(f'ESM target probabilities differ from its query score: {key}')
            grouped[key][repeat] = row
    if set(grouped) != set(expected):
        raise ValueError(f'Missing queries in {path}: {len(set(expected) - set(grouped))}')
    required_repeats = set(range(repeats if repeats is not None else 1))
    averaged = []
    for key in sorted(expected):
        if set(grouped[key]) != required_repeats:
            raise ValueError(f'Incomplete or unexpected repeats in {path}: {key}')
        row = dict(expected[key])
        row.update({metric: float(np.mean([item[metric] for item in grouped[key].values()]))
                    for metric in metrics})
        averaged.append(row)
    return averaged, {'queries': len(averaged), 'recorded_rows': sum(map(len, grouped.values())),
                      'rows_with_stored_positions': stored_positions,
                      'repeats_per_query': len(required_repeats)}


def _metric_components(rows, metric):
    # Reuse the same panel/group aggregation as the ESM evaluation above.
    return summarize_rows([{**row, 'factorized_nll_per_residue': row[metric]}
                           for row in rows])['component_scores']


def _compare_components(model, esm, region):
    if set(model) != set(esm):
        raise ValueError('Model and ESM group sets differ')
    groups = sorted(esm)
    differences = [model[group][region] - esm[group][region] for group in groups]
    return {**_bootstrap(differences),
            'model_mean': float(np.mean([model[group][region] for group in groups])),
            'esm_mean': float(np.mean([esm[group][region] for group in groups])),
            'groups_model_lower': sum(value < 0 for value in differences),
            'groups_model_higher': sum(value > 0 for value in differences)}


def _comparison_report(result):
    lines = ['# ESM-2 与 NatV 任务模型的遮蔽预测比较', '',
             '差值为 QVAE − ESM-2，单位为 nats/目标残基；正值表示 QVAE 的 NLL 较高。'
             '两者使用相同验证序列和共同遮蔽输入，输出均按 20 种氨基酸归一化。', '',
             '| 区域 | 初始化 | ESM-2 NLL | QVAE 联合 NLL | 差值 | 95% 组区间 |',
             '| --- | --- | ---: | ---: | ---: | --- |']
    labels = {'NMNAT': 'NMNAT', 'Nudix': 'Nudix', 'all_panels': '全部掩码加权总体'}
    for region in _COMPARISON_REGIONS:
        for arm in ('random', 'auxinit'):
            values = result['arms'][arm]['comparisons']['prior_joint_nll_per_residue'][region]
            low, high = values['ci95']
            lines.append(f'| {labels[region]} | {arm} | {values["esm_mean"]:.6f} | '
                         f'{values["model_mean"]:.6f} | {values["mean"]:+.6f} | [{low:+.6f}, {high:+.6f}] |')
    lines.extend(['', 'QVAE 的逐位点边缘概率乘积也与 ESM-2 配对比较：', '',
                  '| 区域 | random − ESM-2 | auxinit − ESM-2 |', '| --- | ---: | ---: |'])
    for region in _COMPARISON_REGIONS:
        values = [result['arms'][arm]['comparisons']['marginal_nll_per_residue'][region]['mean']
                  for arm in ('random', 'auxinit')]
        lines.append(f'| {labels[region]} | {values[0]:+.6f} | {values[1]:+.6f} |')
    lines.extend(['',
        f'每个模型均比较 {result["validation"]["queries"]:,} 条遮蔽任务。'
        '先平均旧 QVAE 每条任务的 4 次 Monte Carlo NLL，再计算各组的掩码和序列均值。'
        f'总体差值先在组内平均三个训练种子，再对 {result["validation"]["groups"]} 个组重新抽样 '
        '2,000 次（种子 8802）。区间以既有训练模型为条件，不包含训练种子变异；'
        '三个训练种子不是三个独立生物样本。逐种子、逐组分数见 `group_scores.csv`。', '',
        'ESM-2 分数来自共同遮蔽输入下的位点概率乘积；QVAE 联合分数包含共享潜变量求和。'
        '二者是不同条件分布的对数评分。边缘概率对照另行保留，不将概率分解方式混称为同一模型。', '',
        '旧 QVAE 文件没有逐行保存遮蔽位置。本次以匹配 SHA256 的配置、序列和划分重建位置，'
        '并核对 ESM-2 保存的位置、旧 QVAE 的序列/组/掩码键及目标数。'
        '历史 ESM-2 权重文件 SHA256 未记录，不能宣称已逐字验证历史运行时权重；'
        '本次 ESM-2 权重 SHA256 和 BF16/20AA 设置保存在输入记录中。', '',
        '这些是开发验证结果，仍受原有验证集选权重和辅助片段重叠影响。'
        '本次未训练模型、未评分测试集，也不以 NLL 判断酶活。', '',
        '`comparison.json` 保存完整区间和逐模型结果；`comparisons.csv` 为汇总表，'
        '`identity.json` 保存输入与实现的 SHA256。', ''])
    return '\n'.join(lines)


def compare_esm_baseline(config_path, esm_dir, output):
    """Compare an existing transfer manifest with existing ESM validation scores."""
    manifest = load_yaml(config_path)
    esm_dir, output = Path(esm_dir), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Comparison output already contains files: {output}')
    runs = manifest['runs']
    if len({run['name'] for run in runs}) != len(runs):
        raise ValueError('Duplicate run names')
    seeds = {arm: sorted(int(run['seed']) for run in runs if run['arm'] == arm)
             for arm in ('random', 'auxinit')}
    if (set(run['arm'] for run in runs) != set(seeds) or len(runs) != 6 or
            len(set(seeds['random'])) != 3 or seeds['random'] != seeds['auxinit']):
        raise ValueError('Comparison requires three matched seeds in each original transfer arm')
    old_identity_path = Path(config_path).parent / 'run_identity.json'
    old_identity = json.loads(old_identity_path.read_text())
    if sha256_file(config_path) != old_identity['manifest']['sha256']:
        raise ValueError('Transfer manifest differs from its recorded SHA256')
    esm_identity_path = esm_dir / 'identity.json'
    esm_identity = json.loads(esm_identity_path.read_text())
    esm_summary_path = esm_dir / 'evaluation.json'
    esm_summary = json.loads(esm_summary_path.read_text())
    training_summaries = {run['name']: json.loads(Path(run['training_summary']).read_text())
                          for run in runs}
    sources = {item['population'] for item in training_summaries.values()}
    if len(sources) != 1:
        raise ValueError('Transfer runs used different source configurations')
    sources_path = next(iter(sources))
    data = PopulationData(sources_path, manifest['config'], manifest['target'])
    common_inputs = [manifest['config'], sources_path, manifest['target'], data.sources['population'],
                     data.sources['splits']['training'], data.sources['splits']['validation']]
    input_paths = [config_path, old_identity_path, esm_identity_path, esm_summary_path,
                   esm_dir / 'per_query.jsonl', *common_inputs]
    for path in common_inputs:
        actual = sha256_file(path)
        if (old_identity['inputs'].get(str(path)) != actual or
                esm_identity['input_sha256'].get(str(path)) != actual):
            raise ValueError(f'Historical and current task input SHA differs: {path}')
    if (esm_identity['split'] != 'validation' or esm_identity['amino_acid_order'] != AA or
            esm_summary['status'] != 'COMPLETED' or esm_summary['split'] != 'validation'):
        raise ValueError('ESM result is not a completed 20AA validation evaluation')
    expected = {}
    for query in data.validation_queries():
        record, panel = query['record'], query['panel']
        if hashlib.sha256(record.sequence.encode()).hexdigest() != record.sequence_id:
            raise ValueError(f'Validation sequence SHA differs: {record.sequence_id}')
        row = {'sequence_id': record.sequence_id, 'component': record.component,
               'panel': panel['panel'], 'region': panel['region'],
               'positions_0based': panel['positions'], 'weight': float(panel['weight'])}
        key = _query_key(row)
        if key in expected:
            raise ValueError(f'Duplicate expected query: {key}')
        expected[key] = row
    component_sizes = Counter(record.component for record in data.validation)
    esm_rows, esm_checks = _checked_query_rows(esm_dir / 'per_query.jsonl', expected, component_sizes)
    esm_components = _metric_components(esm_rows, 'factorized_nll_per_residue')
    if (esm_summary['queries'] != len(expected) or esm_summary['sequences'] != len(data.validation) or
            esm_summary['groups'] != len(component_sizes)):
        raise ValueError('ESM summary counts differ from its query records')
    for region in _COMPARISON_REGIONS:
        actual = np.mean([scores[region] for scores in esm_components.values()])
        if not np.isclose(actual, esm_summary['mean_nll_per_residue'][region], rtol=1e-10):
            raise ValueError(f'ESM summary score differs from its query records: {region}')
    result = {
        'status': 'COMPLETED', 'difference': 'QVAE minus ESM; positive means higher QVAE NLL',
        'units': 'nats per target residue',
        'validation': {'sequences': len(data.validation), 'groups': len(component_sizes),
                       'queries': len(expected), 'mask_seed': int(data.study['training']['mask_seed'])},
        'bootstrap': {'unit': 'sequence-similarity component', 'replicates': 2000, 'seed': 8802,
                      'scope': 'resample groups after averaging the fixed three training seeds; '
                               'not a training-seed bootstrap or independent biological seed replicates'},
        'pairing': {'esm': esm_checks,
                    'historical_positions': 'not saved per query; reconstructed from SHA-matched inputs',
                    'query_key': ['sequence_id', 'panel'],
                    'checked_fields': ['component', 'sites', 'weight', 'repeats', 'positions when recorded']},
        'estimators': {'ESM': 'product of 20AA site probabilities for the jointly masked input',
                       'prior_joint_nll_per_residue': 'QVAE shared-latent joint, mean of four MC NLL scores',
                       'marginal_nll_per_residue': 'product of QVAE site marginals, mean of four MC NLL scores'},
        'training_performed': False, 'test_scoring': False, 'runs': {}, 'arms': {},
    }
    group_rows, comparison_rows, run_components = [], [], {}
    for run in runs:
        name = run['name']
        if name not in old_identity['runs']:
            raise ValueError(f'Run absent from the recorded transfer identity: {name}')
        for field in ('per_query', 'evaluation', 'training_summary'):
            path = run[field]
            if sha256_file(path) != old_identity['runs'][name][field]['sha256']:
                raise ValueError(f'Recorded transfer file SHA differs: {path}')
            input_paths.append(path)
        summary = json.loads(Path(run['evaluation']).read_text())
        training = training_summaries[name]
        if (summary['status'] != 'COMPLETED' or summary['split'] != 'validation' or
                summary['repeats'] != 4 or int(training['seed']) != int(run['seed']) or
                summary['checkpoint_sha256'] != training['checkpoint_sha256'] or
                summary['esm_model'] != data.sources['esm_model']):
            raise ValueError(f'Transfer run identity or evaluation settings differ: {name}')
        rows, checks = _checked_query_rows(run['per_query'], expected, component_sizes, repeats=4)
        if summary['queries'] != checks['recorded_rows']:
            raise ValueError(f'Transfer row count differs: {name}')
        run_components[name] = {}
        result['runs'][name] = {'arm': run['arm'], 'seed': int(run['seed']),
                                'pairing': checks, 'comparisons': {}}
        for metric in _COMPARISON_METRICS:
            components = _metric_components(rows, metric)
            run_components[name][metric] = components
            result['runs'][name]['comparisons'][metric] = {}
            for region in _COMPARISON_REGIONS:
                values = _compare_components(components, esm_components, region)
                result['runs'][name]['comparisons'][metric][region] = values
                comparison_rows.append({'scope': 'run', 'model': name, 'arm': run['arm'],
                                        'seed': int(run['seed']), 'metric': metric, 'region': region,
                                        **{k: v for k, v in values.items() if k != 'ci95'},
                                        'ci95_lower': values['ci95'][0], 'ci95_upper': values['ci95'][1]})
                for group in sorted(esm_components):
                    model_value, esm_value = components[group][region], esm_components[group][region]
                    group_rows.append({'run': name, 'arm': run['arm'], 'seed': int(run['seed']),
                                       'component': group, 'metric': metric, 'region': region,
                                       'qvae_nll_per_residue': model_value, 'esm_nll_per_residue': esm_value,
                                       'difference': model_value - esm_value})
    for arm in seeds:
        arm_runs = [run['name'] for run in runs if run['arm'] == arm]
        result['arms'][arm] = {'seeds': seeds[arm], 'comparisons': {}}
        for metric in _COMPARISON_METRICS:
            mean_components = {group: {
                region: float(np.mean([run_components[name][metric][group][region] for name in arm_runs]))
                for region in _COMPARISON_REGIONS} for group in sorted(esm_components)}
            result['arms'][arm]['comparisons'][metric] = {}
            for region in _COMPARISON_REGIONS:
                values = _compare_components(mean_components, esm_components, region)
                result['arms'][arm]['comparisons'][metric][region] = values
                comparison_rows.append({'scope': 'fixed_three_seed_mean', 'model': arm, 'arm': arm,
                                        'seed': '', 'metric': metric, 'region': region,
                                        **{k: v for k, v in values.items() if k != 'ci95'},
                                        'ci95_lower': values['ci95'][0], 'ci95_upper': values['ci95'][1]})
    identity = {
        'input_sha256': {str(path): sha256_file(path) for path in input_paths},
        'code_sha256': {name: sha256_file(Path(__file__).with_name(name))
                       for name in ('esm_baseline.py', 'evaluate.py', 'data.py')},
        'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'precision': esm_identity['precision'], 'normalization': esm_identity['normalization'],
        'esm_evaluation_record': esm_identity,
        'historical_esm_weight_sha256': None,
        'historical_esm_weight_limit': 'Only the historical ESM path was recorded; current and '
                                       'historical weight-file identity is not independently verified.',
        'old_transfer_revision': old_identity['git_revision'],
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'identity.json', identity)
    write_json(output / 'comparison.json', result)
    for filename, rows in [('group_scores.csv', group_rows), ('comparisons.csv', comparison_rows)]:
        with (output / filename).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    (output / 'REPORT.md').write_text(_comparison_report(result), encoding='utf-8')
    return result
