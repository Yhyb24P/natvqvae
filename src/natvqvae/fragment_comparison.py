"""Paired prediction comparison for the three NatV fragment-pretraining conditions."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .data import (PopulationData, Target, generation_masks, load_yaml, mask_panels,
                   sha256_file, stable_seed, write_json)
from .esm_baseline import _checked_query_rows, _metric_components
from .evaluate import _bootstrap
from .model import AA
from .test_data import load_recovered_test

_CONDITIONS = ('random', 'self', 'external')
_REGIONS = ('NMNAT', 'Nudix', 'global', 'linker', 'random', 'all_panels')
_METRICS = ('prior_joint_nll_per_residue', 'marginal_nll_per_residue')
_CONTRASTS = (('external', 'self'), ('self', 'random'), ('external', 'random'),
              ('random', 'frozen_esm2'), ('self', 'frozen_esm2'), ('external', 'frozen_esm2'))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _input_hashes(identity, paths, digest):
    recorded = {str(Path(path).resolve()): value for path, value in identity['input_sha256'].items()}
    for path in paths:
        _require(recorded.get(str(Path(path).resolve())) == digest(path), f'Input SHA256 differs: {path}')


def _paired_difference(left, right, region, repeats, seed):
    _require(set(left) == set(right), 'Compared component sets differ')
    groups = sorted(left)
    differences = [left[group][region] - right[group][region] for group in groups]
    return {**_bootstrap(differences, reps=repeats, seed=seed),
            'left_mean': float(np.mean([left[group][region] for group in groups])),
            'right_mean': float(np.mean([right[group][region] for group in groups])),
            'groups_left_lower': sum(value < 0 for value in differences),
            'groups_left_higher': sum(value > 0 for value in differences)}


def _logged_training_budget(path, steps, seconds):
    counts = []
    with Path(path).open() as handle:
        for line in handle:
            row = json.loads(line)
            _require(type(row['step']) is int and row['step'] == len(counts) + 1,
                     f'Training log step order differs: {path}')
            count = row['target_sites']
            _require(type(count) is int and count > 0, f'Invalid logged target count: {path}')
            counts.append(count)
    _require(len(counts) == steps and np.isfinite(seconds) and seconds > 0,
             f'Training log is incomplete or elapsed time is invalid: {path}')
    return {'updates': steps, 'target_residue_exposures': sum(counts),
            'elapsed_seconds': seconds, 'metrics_file': str(path)}, counts


def _identity_stratum(value):
    if value is None:
        return 'no_qualifying_hit'
    _require(np.isfinite(value) and 0 <= value <= 1, 'Invalid nearest-neighbor identity')
    return ('below_0.5' if value < .5 else '0.5_to_0.7' if value < .7 else
            '0.7_to_0.9' if value < .9 else 'at_least_0.9')


def _stratified_components(rows, neighbors, metric):
    """Keep a component as the resampling unit within each descriptive stratum."""
    grouped = defaultdict(list)
    for row in rows:
        uid, region = row['sequence_id'], row['region']
        reference = region if region in ('NMNAT', 'Nudix') else 'full_length'
        _require(uid in neighbors[reference], f'Nearest-neighbor query missing: {uid}')
        strata = ('all_identities', _identity_stratum(neighbors[reference][uid]))
        for stratum in strata:
            for size in ('all_sizes', str(len(row['positions_0based']))):
                grouped[region, stratum, size, row['component']].append((uid, row[metric]))
    return [{'region': region, 'identity_stratum': stratum, 'target_sites': size,
             'component': group, 'sequences': len({uid for uid, _ in values}),
             'queries': len(values), 'nll': float(np.mean([value for _, value in values]))}
            for (region, stratum, size, group), values in sorted(grouped.items())]


def _report(result):
    primary = result['overall']['external_minus_self'][_METRICS[0]]['NMNAT']
    deterministic = result['model_arm'] == 'deterministic'
    lines = ['# NatV 外部片段、自身片段与随机初始化的预测比较', '',
             f'本次比较 {result["split"]} 集的 {result["data"]["sequences"]} 条序列、'
             f'{result["data"]["groups"]} 个序列组和 {result["data"]["queries"]} 条遮蔽任务。'
             '差值为前一条件减去后一条件，单位为 nats/目标残基；负值表示前者 NLL 较低。', '',
             f'主要比较为 NMNAT 的 external − self：{primary["mean"]:+.6f}，'
             f'95% 组重采样区间 [{primary["ci95"][0]:+.6f}, {primary["ci95"][1]:+.6f}]。', '',
             '| 比较 | NMNAT 差值 [95% 区间] | Nudix 差值 [95% 区间] | 全部掩码差值 [95% 区间] |',
             '| --- | --- | --- | --- |']
    for name, metrics in result['overall'].items():
        cells = []
        for region in ('NMNAT', 'Nudix', 'all_panels'):
            value = metrics[_METRICS[0]][region]
            cells.append(f'{value["mean"]:+.6f} [{value["ci95"][0]:+.6f}, {value["ci95"][1]:+.6f}]')
        lines.append('| ' + name.replace('_minus_', ' − ') + ' | ' + ' | '.join(cells) + ' |')
    lines.extend(['',
        ('确定性模型按逐位点概率乘积精确评分，再计算序列组均值。' if deterministic else
         f'QVAE 的每条任务先平均 {result["data"]["mc_repeats"]} 次 Monte Carlo NLL，再计算序列组均值。') +
        '总体比较先在各组内平均三对训练种子的差值，再按组重采样。'
        '区间以这三对已训练模型为条件，不包含训练种子不确定性，也不将重复任务视为独立蛋白。', '',
        'NMNAT 的 external − self 为预定主要比较；其他结果为补充比较，区间未作多重比较调整。', '',
        ('ESM-2 与确定性模型均采用共同遮蔽输入下的位点概率乘积。' if deterministic else
         'ESM-2 采用共同遮蔽输入下的位点概率乘积；QVAE 联合分数包含共享潜变量。'
         'QVAE 边缘概率乘积的结果单独保存在 summary.json，不混同两种分布。'), '',
        '逐训练种子结果见 summary.json，逐组分数与差值见 group_scores.csv 和 group_differences.csv。'
        '所有九份模型评价均已完成，并核对相同任务、输入、ESM 权重及各自选定的模型权重。', '',
        ('这些验证序列参与了下游权重选择，结果不能代替独立测试。' if result['split'] == 'validation' else
         '这些保留序列未用于本轮训练，但此前已有预测评价，且部分序列曾影响方法选择，'
         '不能称为首次检验或方法开发从未接触的数据。本次使用下游验证已选定的权重，'
         '汇总时不依据保留集分数重新选择模型。具体历史记录与限制保存在 summary.json 的 historical_use 中。'),
        '本程序只汇总已保存分数，没有执行训练或新的预测。NLL 不表示酶学活性。', ''])
    lines += ['实际训练预算直接汇总逐更新日志；同一配对种子的 external/self 辅助训练，以及'
              '三个初始化的下游训练，均核对每一步预测残基数相同。残基暴露计数按每次更新中'
              '被预测的残基累加，不乘潜变量样本数，不包含验证评分。耗时来自运行总结，包含该训练'
              '计时段内的验证/保存开销；特征缓存与并行资源竞争不同，不能据此单独归因模型速度。', '',
              '| 运行 | 辅助预测残基数 | 辅助秒数 | 下游预测残基数 | 下游秒数 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for name, run in result['runs'].items():
        auxiliary, downstream = (run['training_budget'][key] for key in ('auxiliary', 'downstream'))
        lines.append(f'| {name} | {auxiliary["target_residue_exposures"] if auxiliary else 0} | '
                     f'{auxiliary["elapsed_seconds"] if auxiliary else 0:.2f} | '
                     f'{downstream["target_residue_exposures"]} | {downstream["elapsed_seconds"]:.2f} |')
    return '\n'.join(lines) + '\n'


def compare_fragment_transfer(config_path, evaluation_dir, esm_dir, output,
                              split='validation', test_data=None, neighbors_dir=None):
    """Summarize nine completed runs in an explicit split directory; never score models."""
    _require(split in ('validation', 'test') and ((split == 'test') == (test_data is not None)),
             'Test comparison requires explicit recovered test data; validation must not use it')
    config = load_yaml(config_path)
    model_arm = config.get('model_arm', 'QVAE')
    _require(model_arm in {'QVAE', 'deterministic'}, 'Unsupported comparison model arm')
    output, evaluation_dir, esm_dir = Path(output), Path(evaluation_dir), Path(esm_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Fragment comparison output already contains files: {output}')
    pairs = config['seed_pairs']
    _require(len(pairs) == 3 and len({p['auxiliary'] for p in pairs}) == 3 and
             len({p['downstream'] for p in pairs}) == 3, 'Three distinct paired pipeline seeds are required')
    _require(len(config['conditions']) == 3 and set(config['conditions']) == set(_CONDITIONS),
             'The comparison requires random, self and external conditions')
    _require(config['primary_comparison'] == 'external_minus_self' and
             config['primary_metric'] == 'NMNAT_prior_joint_nll_per_target_residue',
             'Primary comparison differs from the specified NMNAT external-minus-self question')
    _require(config['additional_comparisons'] == ['self_minus_random', 'external_minus_random', 'each_minus_frozen_esm2'],
             'Additional comparisons differ from the specified fragment comparison design')
    _require(config['auxiliary_checkpoint'] == 'last' and config['auxiliary_updates'] == 750 and
             config['auxiliary_selection'] == 'fixed_updates_without_validation' and
             config['downstream_checkpoint'] == 'best_by_existing_nmnat_validation_rule',
             'Checkpoint rules differ from the fragment comparison design')
    hashes = {}
    auxiliary_exposures, downstream_exposures = {}, {}

    def digest(path):
        key = str(Path(path).resolve())
        if key not in hashes:
            hashes[key] = sha256_file(path)
        return hashes[key]

    def read(path):
        digest(path)
        return json.loads(Path(path).read_text())

    digest(config_path)
    data = PopulationData(config['sources'], config['downstream_config'], config['target'])
    study = data.study
    records = data.validation if split == 'validation' else load_recovered_test(data, test_data)
    neighbors, stratified_rows = {}, []
    if neighbors_dir is not None:
        neighbor_identity = read(Path(neighbors_dir) / 'identity.json')
        for path, expected_hash in neighbor_identity['input_sha256'].items():
            _require(digest(path) == expected_hash, f'Nearest-neighbor source has changed: {path}')
        record_groups = {record.sequence_id: record.component for record in records}
        for region, name in [('full_length', 'full_length'), ('NMNAT', 'nmnat'), ('Nudix', 'nudix')]:
            path = Path(neighbors_dir) / f'{name}_both_nearest.csv'
            digest(path)
            neighbors[region] = {}
            with path.open() as handle:
                for row in csv.DictReader(handle):
                    if row['role'] != split or row['target_role'] != 'training':
                        continue
                    uid = row['sequence_id']
                    _require(uid not in neighbors[region] and record_groups.get(uid) == row['component'],
                             'Nearest-neighbor query duplicate or component mismatch')
                    value = float(row['fident']) if row['target_id'] else None
                    _identity_stratum(value)
                    neighbors[region][uid] = value
            _require(set(neighbors[region]) == set(record_groups), 'Nearest-neighbor membership differs')
    historical_use = None
    if split == 'test':
        historical_use = read(Path(test_data) / 'historical_use.json')
        _require(historical_use['status'] == 'previously_evaluated',
                 'Test comparison requires the recorded prior evaluation history')
        test_ids = sorted(record.sequence_id for record in records)
        test_ids_sha256 = hashlib.sha256(('\n'.join(test_ids) + '\n').encode()).hexdigest()
        _require(historical_use['test_sequence_count'] == len(test_ids) and
                 historical_use['test_sequence_ids_sha256'] == test_ids_sha256,
                 'Prior evaluation history belongs to different test sequences')
    expected = {}
    for record in records:
        _require(hashlib.sha256(record.sequence.encode()).hexdigest() == record.sequence_id,
                 f'Sequence SHA differs: {record.sequence_id}')
        for panel in mask_panels(record, study['training']['mask_seed']):
            key = record.sequence_id, panel['panel']
            _require(key not in expected, f'Duplicate expected query: {key}')
            expected[key] = {'sequence_id': record.sequence_id, 'component': record.component,
                             'panel': panel['panel'], 'region': panel['region'],
                             'positions_0based': panel['positions'], 'weight': float(panel['weight'])}
    component_sizes = Counter(record.component for record in records)
    common_inputs = [config['downstream_config'], config['sources'], config['target'],
                     data.sources['population'], data.sources['splits']['training'],
                     data.sources['splits']['validation']]
    evaluation_inputs = [*common_inputs, data.sources['splits']['test']]
    if split == 'test':
        evaluation_inputs.extend(Path(test_data) / name for name in ('test.csv', 'test.fasta', 'manifest.json'))
    esm_path = Path(data.sources['esm_model'])
    esm_files = {path.name: digest(path) for path in sorted(esm_path.iterdir())
                 if path.suffix in {'.json', '.safetensors'} or path.name == 'vocab.txt'}
    _require('model.safetensors' in esm_files and 'vocab.txt' in esm_files, 'Incomplete ESM weight identity')
    esm_identity, esm_summary = read(esm_dir / 'identity.json'), read(esm_dir / 'evaluation.json')
    _require(esm_summary['status'] == 'COMPLETED' and esm_summary['split'] == split and
             esm_identity['split'] == split and esm_identity['amino_acid_order'] == AA,
             'ESM evaluation split, completion or amino-acid vocabulary differs')
    _input_hashes(esm_identity, evaluation_inputs if split == 'test' else common_inputs, digest)
    _input_hashes(esm_identity, [esm_path / name for name in esm_files], digest)
    esm_rows, esm_checks = _checked_query_rows(esm_dir / 'per_query.jsonl', expected, component_sizes)
    digest(esm_dir / 'per_query.jsonl')
    esm_components = _metric_components(esm_rows, 'factorized_nll_per_residue')
    if neighbors:
        stratified_rows.extend({**row, 'condition': 'frozen_esm2', 'downstream_seed': '',
                                'metric': 'factorized_nll_per_residue'} for row in
                               _stratified_components(esm_rows, neighbors, 'factorized_nll_per_residue'))
    _require(esm_summary['queries'] == len(expected) and esm_summary['sequences'] == len(records) and
             esm_summary['groups'] == len(component_sizes), 'ESM summary counts differ')
    for region in _REGIONS:
        _require(np.isclose(np.mean([g[region] for g in esm_components.values()]),
                            esm_summary['mean_nll_per_residue'][region], rtol=1e-10),
                 f'ESM summary score differs: {region}')
    train_root = Path(config['training_output'])
    model_scores, run_records, auxiliary_initials, selection_hashes = {}, {}, {}, set()
    auxiliary_pair_inputs = set()
    repeats = 1 if model_arm == 'deterministic' else int(study['evaluation']['repeats'])
    for pair in pairs:
        auxiliary_seed, downstream_seed = int(pair['auxiliary']), int(pair['downstream'])
        for condition in _CONDITIONS:
            name = f'{config.get("run_prefix", "fragment_")}{condition}_seed{downstream_seed}'
            directory, train_directory = evaluation_dir / name, train_root / name
            training = read(train_directory / 'training_summary.json')
            train_identity = read(train_directory / 'run_identity.json')
            _require(training.get('arm', 'QVAE') == model_arm and
                     train_identity.get('arm', 'QVAE') == model_arm,
                     f'Downstream model arm differs: {name}')
            expected_training = {**study['training'], 'seed': downstream_seed}
            _require(training['status'] == 'COMPLETED' and training['seed'] == downstream_seed and
                     training['steps'] == int(study['training']['max_steps']) and
                     train_identity['effective_training'] == expected_training,
                     f'Incomplete or different downstream training: {name}')
            _input_hashes(train_identity, common_inputs, digest)
            metrics_path = train_directory / 'metrics.jsonl'
            digest(metrics_path)
            downstream_budget, counts = _logged_training_budget(metrics_path, training['steps'], training['seconds'])
            if downstream_seed in downstream_exposures:
                _require(counts == downstream_exposures[downstream_seed],
                         f'Downstream target-residue exposure differs across conditions: {downstream_seed}')
            downstream_exposures[downstream_seed] = counts
            auxiliary_budget = None
            _require(train_identity['esm_files_sha256'] == esm_files, f'Training ESM files differ: {name}')
            best = training['checkpoints']['best']
            _require(Path(best['path']).resolve() == Path(training['checkpoint']).resolve() and
                     best['sha256'] == training['checkpoint_sha256'] == digest(best['path']),
                     f'Selected downstream checkpoint differs: {name}')
            selection_hashes.add(digest(train_directory / 'validation_selection.json'))
            initialization = training['initialization']
            if condition == 'random':
                _require(initialization == {'mode': 'random'} and train_identity['initialization'] == 'random',
                         f'Random condition has pretrained initialization: {name}')
            else:
                aux_config_path = config['auxiliary_configs'][condition]
                aux_config = load_yaml(aux_config_path)
                _require(aux_config['fragment_condition'] == condition,
                         f'Auxiliary configuration condition differs: {name}')
                auxiliary_directory = train_root / f'{condition}_pretraining_seed{auxiliary_seed}'
                auxiliary = read(auxiliary_directory / 'training_summary.json')
                auxiliary_identity = read(auxiliary_directory / 'run_identity.json')
                aux_training = {**aux_config['training'], 'seed': auxiliary_seed}
                _require(model_arm in auxiliary['arms'], f'Auxiliary model arm differs: {name}')
                arm = auxiliary['arms'][model_arm]
                metrics_path = auxiliary_directory / f'{model_arm}_metrics.jsonl'
                digest(metrics_path)
                auxiliary_budget, counts = _logged_training_budget(metrics_path, arm['steps'], arm['seconds'])
                if auxiliary_seed in auxiliary_exposures:
                    _require(counts == auxiliary_exposures[auxiliary_seed],
                             f'Auxiliary target-residue exposure differs across conditions: {auxiliary_seed}')
                auxiliary_exposures[auxiliary_seed] = counts
                _require(auxiliary['status'] == 'COMPLETED' and auxiliary['seed'] == auxiliary_seed and
                         auxiliary['fragment_condition'] == condition and arm['steps'] == 750 and
                         arm['checkpoint_rule'] == 'fixed_updates' and arm['selection_metric'] is None and
                         arm['validation_curve'] == [] and
                         all(arm.get(key) is None for key in ('initial_validation', 'best_validation', 'last_validation', 'best_epoch')) and
                         auxiliary['validation_sequences'] == 0 and
                         auxiliary_identity['effective_training'] == aux_training and
                         auxiliary_identity['checkpoint_rule'] == 'fixed_updates',
                         f'Auxiliary seed, budget or selection differs: {name}')
                _require(auxiliary_identity['esm_files_sha256'] == esm_files,
                         f'Auxiliary ESM files differ: {name}')
                for key, path in auxiliary['input_paths'].items():
                    _require(Path(path).resolve() == Path(aux_config['paths'][key]).resolve() and
                             auxiliary_identity['input_sha256'][key] == digest(path),
                             f'Auxiliary input differs: {name}, {key}')
                _require(auxiliary_identity['input_sha256']['config'] == digest(aux_config_path),
                         f'Auxiliary configuration differs: {name}')
                auxiliary_pair_inputs.add(tuple(auxiliary_identity['input_sha256'][key]
                                                for key in ('pair_manifest', 'pair_panels', 'a_fragments', 's_fragments')))
                last = arm['checkpoints']['last']
                _require(initialization['mode'] == 'checkpoint' and
                         initialization['checkpoint_seed'] == auxiliary_seed and
                         initialization['checkpoint_step'] == 750 and
                         Path(initialization['path']).resolve() == Path(last['path']).resolve() and
                         initialization['sha256'] == last['sha256'] == digest(last['path']),
                         f'Downstream initialization is not its paired 750-update last checkpoint: {name}')
                _require(Path(train_identity['initialization']).resolve() == Path(last['path']).resolve(),
                         f'Downstream recorded initialization path differs: {name}')
                _input_hashes(train_identity, [last['path']], digest)
                initial = arm['checkpoints']['initial']
                _require(initial['sha256'] == digest(initial['path']), f'Auxiliary initial checkpoint differs: {name}')
                auxiliary_initials[condition, auxiliary_seed] = initial['sha256']
            identity, summary = read(directory / 'identity.json'), read(directory / 'evaluation.json')
            _require(identity.get('arm', 'QVAE') == model_arm and
                     summary.get('arm', 'QVAE') == model_arm, f'Evaluation model arm differs: {name}')
            _require(summary['status'] == 'COMPLETED' and summary['split'] == split and
                     identity['split'] == split and identity['limit'] is None and
                     identity['config'] == study['evaluation'] and
                     identity['mask_seed'] == study['training']['mask_seed'] and
                     identity['amino_acid_output_count'] == 20 and
                     summary['repeats'] == repeats and
                     summary['sample_count'] == (None if model_arm == 'deterministic' else
                                                 study['evaluation']['sample_count']),
                     f'Evaluation task or completion differs: {name}')
            _input_hashes(identity, [*evaluation_inputs, best['path']], digest)
            _require(identity['esm_files_sha256'] == esm_files and
                     summary['checkpoint_sha256'] == best['sha256'] and
                     Path(summary['checkpoint']).resolve() == Path(best['path']).resolve(),
                     f'Evaluation encoder or selected checkpoint differs: {name}')
            rows, checks = _checked_query_rows(directory / 'per_query.jsonl', expected, component_sizes, repeats)
            digest(directory / 'per_query.jsonl')
            _require(checks['recorded_rows'] == summary['queries'] and
                     checks['rows_with_stored_positions'] == checks['recorded_rows'] and
                     identity['sequences'] == len(records) and identity['groups'] == len(component_sizes),
                     f'Incomplete task counts or stored target positions: {name}')
            model_scores[condition, downstream_seed] = {metric: _metric_components(rows, metric) for metric in _METRICS}
            if neighbors:
                for metric in _METRICS:
                    stratified_rows.extend({**row, 'condition': condition, 'downstream_seed': downstream_seed,
                                            'metric': metric} for row in
                                           _stratified_components(rows, neighbors, metric))
            if model_arm == 'deterministic':
                _require(all(np.isclose(row[_METRICS[0]], row[_METRICS[1]], rtol=0, atol=1e-7)
                             for row in rows), f'Deterministic joint and marginal scores differ: {name}')
            for metric in _METRICS:
                actual = np.mean([g['all_panels'] for g in model_scores[condition, downstream_seed][metric].values()])
                _require(np.isclose(actual, summary[metric], rtol=1e-9), f'Evaluation summary score differs: {name}, {metric}')
            run_records[name] = {'condition': condition, 'downstream_seed': downstream_seed,
                                 'auxiliary_seed': auxiliary_seed if condition != 'random' else None,
                                 'checkpoint': best, 'initialization': initialization, 'pairing': checks,
                                 'precision': identity['precision'],
                                 'training_budget': {'auxiliary': auxiliary_budget, 'downstream': downstream_budget},
                                 'training_code_sha256': train_identity['code_sha256'],
                                 'evaluation_code_sha256': identity['code_sha256']}
    precisions = {row['precision'].split(' forward')[0] for row in run_records.values()}
    _require(precisions == {esm_identity['precision'].split(' forward')[0]},
             'Task models and ESM were evaluated with different numerical precisions')
    _require(len(selection_hashes) == 1, 'Downstream validation-selection tasks differ across runs')
    _require(len(auxiliary_pair_inputs) == 1, 'Auxiliary runs used different paired fragment data')
    for pair in pairs:
        seed = int(pair['auxiliary'])
        _require(auxiliary_initials['external', seed] == auxiliary_initials['self', seed],
                 f'External/self auxiliary initial states differ: {seed}')
    bootstrap_repeats, bootstrap_seed = (int(config['uncertainty'][key]) for key in ('bootstrap_repeats', 'bootstrap_seed'))
    _require(bootstrap_repeats > 0, 'Bootstrap repetitions must be positive')
    result = {'status': 'COMPLETED', 'split': split, 'model_arm': model_arm,
              'primary_comparison': 'external_minus_self',
              'data_status': 'historically_evaluated_heldout' if split == 'test' else 'development_validation',
              'historical_use': historical_use,
              'primary_metric': 'NMNAT_prior_joint_nll_per_target_residue', 'units': 'nats per target residue',
              'data': {'sequences': len(records), 'groups': len(component_sizes), 'queries': len(expected),
                       'mc_repeats': repeats, 'mask_seed': study['training']['mask_seed'],
                       'query_tasks_sha256': hashlib.sha256(json.dumps(
                           [expected[key] for key in sorted(expected)], sort_keys=True).encode()).hexdigest()},
              'pipeline_seed_pairs': pairs, 'runs': run_records, 'esm_pairing': esm_checks,
              'bootstrap': {'unit': 'component_v2', 'repetitions': bootstrap_repeats, 'seed': bootstrap_seed,
                            'scope': 'conditional on three fitted pipeline replicates; group bootstrap after averaging seed differences'},
              'distributions': {'prior_joint_nll_per_residue': ('deterministic site probability product' if model_arm == 'deterministic' else 'shared-latent QVAE joint'),
                                'marginal_nll_per_residue': ('deterministic site probability product' if model_arm == 'deterministic' else 'product of QVAE site marginals'),
                                'frozen_esm2': 'product of site probabilities for the jointly masked input'},
              'prediction_scoring_performed': False, 'training_performed': False,
              'per_pipeline_seed': {}, 'overall': {}}
    group_rows, difference_rows, summary_rows = [], [], []
    for group in sorted(esm_components):
        for region in _REGIONS:
            group_rows.append({'condition': 'frozen_esm2', 'downstream_seed': '', 'auxiliary_seed': '',
                               'component': group, 'metric': 'factorized_nll_per_residue', 'region': region,
                               'nll': esm_components[group][region]})
    for pair in pairs:
        seed = int(pair['downstream'])
        result['per_pipeline_seed'][str(seed)] = {'auxiliary_seed': int(pair['auxiliary']), 'comparisons': {}}
        for condition in _CONDITIONS:
            for metric, components in model_scores[condition, seed].items():
                for group in sorted(components):
                    for region in _REGIONS:
                        group_rows.append({'condition': condition, 'downstream_seed': seed,
                                           'auxiliary_seed': pair['auxiliary'] if condition != 'random' else '',
                                           'component': group, 'metric': metric, 'region': region,
                                           'nll': components[group][region]})
        for left_name, right_name in _CONTRASTS:
            contrast = left_name + '_minus_' + right_name
            by_metric = result['per_pipeline_seed'][str(seed)]['comparisons'].setdefault(contrast, {})
            for metric in _METRICS:
                left = model_scores[left_name, seed][metric]
                right = esm_components if right_name == 'frozen_esm2' else model_scores[right_name, seed][metric]
                by_metric[metric] = {}
                for region in _REGIONS:
                    value = _paired_difference(left, right, region, bootstrap_repeats, bootstrap_seed)
                    by_metric[metric][region] = value
                    summary_rows.append({'scope': 'pipeline_seed', 'downstream_seed': seed,
                                         'comparison': contrast, 'metric': metric, 'region': region,
                                         'mean': value['mean'], 'ci95_low': value['ci95'][0], 'ci95_high': value['ci95'][1]})
                    for group in sorted(left):
                        difference_rows.append({'downstream_seed': seed, 'auxiliary_seed': pair['auxiliary'],
                                                'comparison': contrast, 'metric': metric, 'region': region,
                                                'component': group, 'left': left[group][region],
                                                'right': right[group][region], 'difference': left[group][region] - right[group][region]})
    means = {condition: {metric: {group: {region: float(np.mean([
        model_scores[condition, int(pair['downstream'])][metric][group][region] for pair in pairs]))
        for region in _REGIONS} for group in sorted(esm_components)} for metric in _METRICS} for condition in _CONDITIONS}
    for left_name, right_name in _CONTRASTS:
        contrast = left_name + '_minus_' + right_name
        result['overall'][contrast] = {}
        for metric in _METRICS:
            left = means[left_name][metric]
            right = esm_components if right_name == 'frozen_esm2' else means[right_name][metric]
            result['overall'][contrast][metric] = {}
            for region in _REGIONS:
                value = _paired_difference(left, right, region, bootstrap_repeats, bootstrap_seed)
                result['overall'][contrast][metric][region] = value
                summary_rows.append({'scope': 'fixed_three_pipeline_seed_mean', 'downstream_seed': '',
                                     'comparison': contrast, 'metric': metric, 'region': region,
                                     'mean': value['mean'], 'ci95_low': value['ci95'][0], 'ci95_high': value['ci95'][1]})
    identity = {'input_sha256': hashes,
                'implementation_sha256': {name: sha256_file(Path(__file__).with_name(name))
                                          for name in ('fragment_comparison.py', 'esm_baseline.py', 'evaluate.py', 'data.py')},
                'training_selection_sha256': next(iter(selection_hashes)),
                'esm_files_sha256': esm_files, 'esm_evaluation_identity': esm_identity}
    if split == 'test':
        identity['implementation_sha256']['test_data.py'] = sha256_file(Path(__file__).with_name('test_data.py'))
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'summary.json', result)
    write_json(output / 'identity.json', identity)
    for name, rows in [('group_scores.csv', group_rows), ('group_differences.csv', difference_rows),
                       ('comparisons.csv', summary_rows)] + (
                           [('stratified_group_scores.csv', stratified_rows)] if neighbors else []):
        with (output / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    (output / 'REPORT.md').write_text(_report(result), encoding='utf-8')
    return result


def _six_cell_contrasts(tables, seeds, bootstrap_repeats, bootstrap_seed):
    scores = {}
    counts = {}
    for arm, rows in tables.items():
        for row in rows:
            cell = row['region'], row['identity_stratum'], str(row['target_sites']), row['component']
            count = int(row['sequences']), int(row['queries'])
            _require(cell not in counts or counts[cell] == count, 'Stratum membership differs between models')
            counts[cell] = count
            if row['condition'] == 'frozen_esm2':
                for seed in seeds:
                    key = 'frozen_esm2', 'frozen_esm2', seed, _METRICS[0], cell
                    _require(key not in scores or scores[key] == float(row['nll']), 'Frozen ESM stratum differs')
                    scores[key] = float(row['nll'])
            else:
                key = arm, row['condition'], int(row['downstream_seed']), row['metric'], cell
                _require(key not in scores, 'Duplicate stratified score')
                scores[key] = float(row['nll'])
    joint, marginal = _METRICS
    specs = {}
    for condition in _CONDITIONS:
        specs[f'{condition}:QVAE_minus_deterministic'] = [('QVAE', condition, joint, 1),
                                                        ('deterministic', condition, joint, -1)]
        specs[f'{condition}:joint_minus_site_marginal'] = [('QVAE', condition, joint, 1),
                                                         ('QVAE', condition, marginal, -1)]
    for arm in ('QVAE', 'deterministic'):
        for left, right in _CONTRASTS[:3]:
            specs[f'{arm}:{left}_minus_{right}'] = [(arm, left, joint, 1), (arm, right, joint, -1)]
        for condition in _CONDITIONS:
            specs[f'{arm}:{condition}_minus_frozen_esm2'] = [
                (arm, condition, joint, 1), ('frozen_esm2', 'frozen_esm2', joint, -1)]
    specs['external_minus_self:QVAE_minus_deterministic'] = [
        ('QVAE', 'external', joint, 1), ('QVAE', 'self', joint, -1),
        ('deterministic', 'external', joint, -1), ('deterministic', 'self', joint, 1)]
    summaries, group_rows = [], []
    strata = sorted({cell[:3] for cell in counts})
    for name, terms in specs.items():
        for region, stratum, size in strata:
            cells = sorted(cell for cell in counts if cell[:3] == (region, stratum, size))
            seed_values = []
            for seed in seeds:
                values = []
                for cell in cells:
                    keys = [(arm, condition, seed, metric, cell) for arm, condition, metric, _ in terms]
                    _require(all(key in scores for key in keys), 'Incomplete paired stratum/model/seed scores')
                    value = sum(term[3] * scores[key] for key, term in zip(keys, terms))
                    values.append(value)
                    group_rows.append({'comparison': name, 'region': region, 'identity_stratum': stratum,
                                       'target_sites': size, 'seed': seed, 'component': cell[3],
                                       'difference': value})
                seed_values.append(values)
            for scope, values in [('mean_of_pipeline_seeds', np.mean(seed_values, axis=0)),
                                  *[(str(seed), values) for seed, values in zip(seeds, seed_values)]]:
                estimate = _bootstrap(values, reps=bootstrap_repeats, seed=bootstrap_seed)
                if len(values) < 2:
                    estimate['ci95'] = None
                summaries.append({'comparison': name, 'region': region, 'identity_stratum': stratum,
                                  'target_sites': size, 'scope': scope, **estimate,
                                  'sequences': sum(counts[cell][0] for cell in cells),
                                  'queries': sum(counts[cell][1] for cell in cells)})
    return summaries, group_rows


def compare_six_cells(config_path, output, split='validation'):
    """Revalidate both three-condition studies before paired six-cell contrasts."""
    config = load_yaml(config_path)
    _require(split in ('validation', 'test'), 'Unknown six-cell evaluation split')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Six-cell output already contains files: {output}')
    results, tables, identities = {}, {}, {}
    for arm in ('QVAE', 'deterministic'):
        source = config['models'][arm]
        results[arm] = compare_fragment_transfer(
            source['config'], Path(source['evaluation_root']) / split, config['esm'][split], output / arm,
            split, config['test_data'] if split == 'test' else None, config['neighbors'])
        _require(results[arm]['model_arm'] == arm, 'Six-cell arm configuration mismatch')
        with (output / arm / 'stratified_group_scores.csv').open() as handle:
            tables[arm] = list(csv.DictReader(handle))
        identities[arm] = json.loads((output / arm / 'identity.json').read_text())
    q, d = results['QVAE'], results['deterministic']
    for field in ('query_tasks_sha256', 'mask_seed', 'sequences', 'groups', 'queries'):
        _require(q['data'][field] == d['data'][field], f'Six-cell task differs: {field}')
    _require(q['pipeline_seed_pairs'] == d['pipeline_seed_pairs'], 'Six-cell pipeline seeds differ')
    _require(q['bootstrap'] == d['bootstrap'], 'Six-cell bootstrap rules differ')
    for field in ('training_selection_sha256', 'esm_files_sha256', 'esm_evaluation_identity'):
        _require(identities['QVAE'][field] == identities['deterministic'][field],
                 f'Six-cell comparison conditions differ: {field}')
    seeds = [int(pair['downstream']) for pair in q['pipeline_seed_pairs']]
    summaries, groups = _six_cell_contrasts(tables, seeds, q['bootstrap']['repetitions'], q['bootstrap']['seed'])
    identity_bins = ['below_0.5', '0.5_to_0.7', '0.7_to_0.9', 'at_least_0.9', 'no_qualifying_hit']
    observed = {(row['region'], row['identity_stratum'], row['target_sites']) for row in summaries}
    empty_strata = [{'region': region, 'identity_stratum': stratum, 'target_sites': size,
                     'groups': 0, 'sequences': 0, 'queries': 0, 'mean': None, 'ci95': None}
                    for region, size in sorted({(r, size) for r, _, size in observed})
                    for stratum in identity_bins if (region, stratum, size) not in observed]
    result = {'split': split, 'status': 'COMPLETED', 'pipeline_seed_pairs': q['pipeline_seed_pairs'],
              'comparisons': summaries,
              'identity_bins': identity_bins, 'empty_strata': empty_strata,
              'aggregation': 'mean queries within component/region/size/identity stratum; mean paired seeds within component; equal-component bootstrap',
              'identity_reference': 'NMNAT/Nudix use their CDD domain; other masks use full length; nearest training local alignment with both coverages >=80%',
              'limits': ['Intervals condition on three fitted pipeline replicates; seed differences are shown separately.',
                         'Components may occur in multiple strata; strata are not independent studies. Empty strata have no estimate; one-component strata have no interval.',
                         'Size/identity strata and model/latent contrasts are supplementary, without multiplicity adjustment.',
                         'Stratified self-minus-random and external-minus-random contrasts were added after examining aggregate QVAE validation regional scores; they are exploratory reference comparisons.',
                         'Small joint-minus-marginal differences need Monte Carlo error assessment before interpretation.'],
              'input_sha256': {str(config_path): sha256_file(config_path), **{
                  str(output / arm / name): sha256_file(output / arm / name)
                  for arm in results for name in ('summary.json', 'identity.json', 'stratified_group_scores.csv')}},
              'implementation_sha256': sha256_file(Path(__file__))}
    with (output / 'paired_group_differences.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=groups[0])
        writer.writeheader()
        writer.writerows(groups)
    _write_six_cell_outputs(result, output)
    write_json(output / 'summary.json', result)
    return result


def _write_six_cell_outputs(result, output):
    """Display the existing paired estimates without reweighting or rescoring."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    panels = [
        ('External minus self', [('QVAE:external_minus_self', 'QVAE'),
                                 ('deterministic:external_minus_self', 'Deterministic')]),
        ('QVAE minus deterministic', [(f'{c}:QVAE_minus_deterministic', c.capitalize())
                                      for c in _CONDITIONS]),
        ('Joint minus own site marginal', [(f'{c}:joint_minus_site_marginal', c.capitalize())
                                           for c in _CONDITIONS]),
        ('Task model minus frozen ESM-2', [
            (f'{arm}:{c}_minus_frozen_esm2', f'{arm}: {c}')
            for arm in ('QVAE', 'deterministic') for c in _CONDITIONS]),
    ]
    selected = [row for row in result['comparisons']
                if row['identity_stratum'] == 'all_identities' and row['target_sites'] == 'all_sizes']
    lookup = {(row['comparison'], row['region'], row['scope']): row for row in selected}
    _require(len(lookup) == len(selected), 'Duplicate six-cell display estimate')
    regions = ('NMNAT', 'Nudix', 'global')
    seeds = [str(pair['downstream']) for pair in result['pipeline_seed_pairs']]
    required = [(name, region, scope) for _, specs in panels for name, _ in specs
                for region in regions for scope in ['mean_of_pipeline_seeds', *seeds]]
    _require(all(key in lookup for key in required), 'Incomplete six-cell figure estimates')
    colors = ('#2878A1', '#CA6A26', '#448C62')
    with plt.rc_context({'font.size': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig, axes = plt.subplots(4, 3, figsize=(9, 9), layout='constrained',
                                 gridspec_kw={'height_ratios': [2, 3, 3, 6]})
        for panel_index, (title, specs) in enumerate(panels):
            for column, region in enumerate(regions):
                axis = axes[panel_index, column]
                axis.axvline(0, color='.65', lw=.8, linestyle='--')
                for index, (name, label) in enumerate(specs):
                    y = len(specs) - 1 - index
                    estimate = lookup[name, region, 'mean_of_pipeline_seeds']
                    interval = estimate['ci95']
                    if interval is not None:
                        axis.hlines(y, *interval, color='.15', lw=1.2)
                    axis.plot(estimate['mean'], y, 'o', color='.15', ms=4)
                    for seed_index, seed in enumerate(seeds):
                        axis.plot(lookup[name, region, seed]['mean'], y - .22 + seed_index * .06,
                                  '^', color=colors[seed_index], ms=3)
                axis.set_yticks(range(len(specs)), [label for _, label in reversed(specs)])
                if column:
                    axis.tick_params(axis='y', labelleft=False)
                axis.set_ylim(-.5, len(specs) - .4)
                groups = lookup[specs[0][0], region, 'mean_of_pipeline_seeds']['groups']
                axis.set_title(f'{region} masks (n={groups} groups)' if panel_index == 0 else title)
                if panel_index == 0 and column == 0:
                    axis.set_ylabel(title)
                axis.set_xlabel('NLL difference')
                axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
                axis.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3), useOffset=False)
                axis.spines[['top', 'right']].set_visible(False)
        handles = [Line2D([], [], marker='o', color='.15', label='Paired mean and 95% group interval')]
        handles.extend(Line2D([], [], marker='^', color=color, linestyle='', label=f'Seed {seed}')
                       for seed, color in zip(seeds, colors))
        fig.legend(handles=handles, loc='outside upper center', ncol=4, frameon=False)
        fig.supxlabel(f'NatV {result["split"]}: nats per target residue; negative favors the first term; fixed pipeline replicates', fontsize=9)
        for extension in ('pdf', 'png'):
            fig.savefig(output / f'prediction_effects.{extension}', dpi=220, bbox_inches='tight')
        plt.close(fig)
    with (output / 'prediction_effects.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=selected[0])
        writer.writeheader()
        writer.writerows(selected)
    lines = ['# NatV 六格条件预测比较', '',
             f'评价划分：{result["split"]}。差值为比较名称中前项减后项，单位为 nats/目标残基；'
             '负值表示前项 NLL 较低。主要问题是 NMNAT 的 external − self；'
             '其他区域、模型差异、与冻结 ESM-2 的差异及交互项为补充比较。', '',
             '图中黑点先在每个序列组内平均配对训练重复，再对组等权汇总；横线为组 bootstrap 95% 区间。'
             '彩色三角为各完整训练重复的组均值，不是新增独立序列组。'
             '区间以这些已拟合模型为条件，未覆盖训练随机性或作多重比较调整。', '',
             '![条件预测差值](prediction_effects.png)', '',
             '各图列仅显示该区域的掩码，global 不是全部掩码的加权总体。'
             '坐标范围随比较变化；约 10⁻⁴ 的 joint − marginal 差值须结合另行输出的 Monte Carlo 精度分析，'
             '不能由区间与零的关系直接解释为有用的生成依赖。', '',
             '| 比较 | 区域 | 组数 | 平均差值 | 95% 组区间 |',
             '| --- | --- | ---: | ---: | --- |']
    for row in selected:
        if row['scope'] != 'mean_of_pipeline_seeds':
            continue
        interval = row['ci95']
        bounds = f'[{interval[0]:+.8f}, {interval[1]:+.8f}]' if interval is not None else 'NA'
        lines.append(f'| {row["comparison"]} | {row["region"]} | {row["groups"]} | '
                     f'{row["mean"]:+.8f} | {bounds} |')
    lines.extend(['', '全部相似性与位点数分层、空层及单组层见 summary.json；'
                  '逐组配对差值见 paired_group_differences.csv。'
                  'prediction_effects.csv 保留全部相似性、全部位点数范围内的区域估计和逐种子结果，'
                  '其中 linker、random 及模型交互项列于本表，但未绘入主图。', '',
                  '自身减随机与外部减随机的分层对照，是观察到 QVAE 验证总体区域差异后补充的探索性参照；'
                  '用于区分自身预训练的退化与外部相对随机基线的改善，不替代预定主要比较。', '',
                  *[f'- {limit}' for limit in result['limits']]])
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def plot_six_cells(comparison_dir):
    """Re-export reviewed comparisons without changing the analysis provenance."""
    directory = Path(comparison_dir)
    source = directory / 'summary.json'
    result = json.loads(source.read_text())
    _require(result['status'] == 'COMPLETED', 'Six-cell comparison is not completed')
    _write_six_cell_outputs(result, directory)
    identity = {'summary_sha256': sha256_file(source),
                'implementation_sha256': sha256_file(Path(__file__)),
                'outputs_sha256': {name: sha256_file(directory / name) for name in
                                  ('prediction_effects.csv', 'prediction_effects.pdf',
                                   'prediction_effects.png', 'REPORT.md')},
                'prediction_scoring_performed': False}
    write_json(directory / 'figure_identity.json', identity)
    return identity


def _generation_descriptors(rows):
    """Describe the observed draws; variance uses divisor n and is not uncertainty."""
    values = np.asarray([int(row['Hamming']) for row in rows], dtype=float)
    unique = len({row['sequence_sha256'] for row in rows})
    return {'attempts': len(rows), 'mean_Hamming': float(values.mean()),
            'variance_Hamming': float(values.var(ddof=0)), 'unique_sequences': unique,
            'duplicate_attempts_beyond_first': len(rows) - unique,
            'wt_returns': int(np.count_nonzero(values == 0))}


def _generation_subsets(rows, masks):
    yield 'all_masks', '', rows
    for size in (8, 16, 32):
        yield 'mask_size', str(size), [row for row in rows if int(row['mask_size']) == size]
    for group in masks:
        yield 'mask', group, [row for row in rows if row['group'] == group]


def summarize_fragment_generation(config_path, generation_dir, output):
    """Check and summarize nine stored KVP40 generation runs without sampling."""
    config = load_yaml(config_path)
    generation_dir, output = Path(generation_dir), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Generation summary output already contains files: {output}')
    pairs = config['seed_pairs']
    _require(len(pairs) == 3 and len({p['downstream'] for p in pairs}) == 3 and
             len({p['auxiliary'] for p in pairs}) == 3 and
             len(config['conditions']) == 3 and set(config['conditions']) == set(_CONDITIONS),
             'Generation comparison requires three paired seeds and all three conditions')
    study = load_yaml(config['downstream_config'])
    generation = study['generation']
    sources = load_yaml(config['sources'])
    wt = Target.load(config['target']).sequence
    target_hash = hashlib.sha256(wt.encode()).hexdigest()
    _require(generation['masks_per_condition'] == 16 and generation['draws_per_mask'] == 64 and
             generation['mask_sizes'] == {'NMNAT': [8, 16, 32]} and
             generation['regions_1based'] == {'NMNAT': [4, 181]},
             'Expected 48 NMNAT masks and 64 draws per mask in the KVP40 context')
    groups = generation_masks(generation, wt)
    masks = {group['id']: group for group in groups}
    _require(len(groups) == len(masks) == 48, 'Generation mask IDs are not unique')
    hashes = {}

    def digest(path):
        key = str(Path(path).resolve())
        if key not in hashes:
            hashes[key] = sha256_file(path)
        return hashes[key]

    def read(path):
        digest(path)
        return json.loads(Path(path).read_text())

    digest(config_path)
    common_inputs = [config['downstream_config'], config['sources'], config['target'],
                     sources['population'], *sources['splits'].values()]
    esm_path = Path(sources['esm_model'])
    esm_files = {path.name: digest(path) for path in sorted(esm_path.iterdir())
                 if path.suffix in {'.json', '.safetensors'} or path.name == 'vocab.txt'}
    _require('model.safetensors' in esm_files and 'vocab.txt' in esm_files,
             'Incomplete generation ESM weight identity')
    rows_by_run, run_records, statistics = {}, {}, []
    precision = set()
    for pair in pairs:
        seed = int(pair['downstream'])
        expected_keys = {(group, draw) for group in masks for draw in range(64)}
        for condition in _CONDITIONS:
            name = f'fragment_{condition}_seed{seed}'
            directory = generation_dir / name
            training = read(Path(config['training_output']) / name / 'training_summary.json')
            best = training['checkpoints']['best']
            _require(training['status'] == 'COMPLETED' and training['seed'] == seed and
                     Path(best['path']).resolve() == Path(training['checkpoint']).resolve() and
                     best['sha256'] == training['checkpoint_sha256'] == digest(best['path']),
                     f'Generation training/checkpoint identity differs: {name}')
            initialization = training['initialization']
            if condition == 'random':
                _require(initialization == {'mode': 'random'}, f'Random initialization differs: {name}')
            else:
                auxiliary = read(Path(config['training_output']) /
                                 f'{condition}_pretraining_seed{pair["auxiliary"]}' / 'training_summary.json')
                last = auxiliary['arms']['QVAE']['checkpoints']['last']
                _require(auxiliary['status'] == 'COMPLETED' and auxiliary['fragment_condition'] == condition and
                         auxiliary['seed'] == int(pair['auxiliary']) and
                         initialization['mode'] == 'checkpoint' and
                         initialization['checkpoint_seed'] == int(pair['auxiliary']) and
                         initialization['checkpoint_step'] == 750 and
                         Path(initialization['path']).resolve() == Path(last['path']).resolve() and
                         initialization['sha256'] == last['sha256'] == digest(last['path']),
                         f'Generation auxiliary initialization differs: {name}')
            identity, summary = read(directory / 'run_identity.json'), read(directory / 'generation.json')
            actual_masks = read(directory / 'masks.json')
            _require(actual_masks == groups and identity['masks_sha256'] == digest(directory / 'masks.json'),
                     f'Generation mask positions differ: {name}')
            _require(identity['generation'] == generation and identity['group_filter'] is None and
                     identity['draws_override'] in (None, 64) and identity['mask_batch_size'] == 16 and
                     identity['checkpoint_seed'] == seed and identity['checkpoint_arm'] == 'QVAE' and
                     identity['amino_acid_output_count'] == 20 and identity['groups'] == 48 and
                     identity['target_sha256'] == target_hash and
                     identity['esm_files_sha256'] == esm_files,
                     f'Generation settings, target or encoder differ: {name}')
            _input_hashes(identity, [*common_inputs, best['path']], digest)
            _require(summary['status'] == 'COMPLETED' and summary['groups'] == 48 and
                     summary['draws_per_mask'] == 64 and summary['attempts'] == 3072 and
                     summary['target_sha256'] == target_hash and summary['checkpoint_sha256'] == best['sha256'] and
                     Path(summary['checkpoint']).resolve() == Path(best['path']).resolve() and
                     Path(identity['checkpoint']).resolve() == Path(best['path']).resolve(),
                     f'Incomplete generation or wrong selected checkpoint: {name}')
            candidate_path = directory / 'candidates.csv'
            digest(candidate_path)
            with candidate_path.open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            by_key = {}
            for row in rows:
                key = row['group'], int(row['draw'])
                _require(key in expected_keys and key not in by_key, f'Unexpected/duplicate generation draw: {name}, {key}')
                group, draw = key
                mask = masks[group]
                _require(row['candidate_id'] == f'QVAE_s{seed}_{group}_d{draw:02d}' and
                         row['arm'] == 'QVAE' and int(row['seed']) == seed and
                         row['region'] == mask['region'] and int(row['mask_size']) == mask['mask_size'] and
                         int(row['prior_seed']) == stable_seed(generation['seed'], seed, group, 'latent') and
                         int(row['uniform_seed']) == stable_seed(generation['seed'], seed, group, 'residues'),
                         f'Generation draw identifiers or paired random seeds differ: {name}, {key}')
                sequence = row['sequence']
                _require(len(sequence) == len(wt) and set(sequence) <= set(AA) and
                         hashlib.sha256(sequence.encode()).hexdigest() == row['sequence_sha256'],
                         f'Generated sequence identity differs: {name}, {key}')
                edits = [{'position': i + 1, 'from': a, 'to': b}
                         for i, (a, b) in enumerate(zip(wt, sequence)) if a != b]
                _require(int(row['Hamming']) == len(edits) and json.loads(row['edits']) == edits and
                         {e['position'] - 1 for e in edits} <= set(mask['positions']),
                         f'Candidate edits or mask membership differ: {name}, {key}')
                by_key[key] = row
            _require(set(by_key) == expected_keys and len(rows) == 3072, f'Missing generation draws: {name}')
            rows_by_run[condition, seed] = rows
            for scope, label, subset in _generation_subsets(rows, masks):
                statistics.append({'condition': condition, 'downstream_seed': seed,
                                   'scope': scope, 'mask': label, **_generation_descriptors(subset)})
            run_records[name] = {'condition': condition, 'downstream_seed': seed,
                                 'auxiliary_seed': pair['auxiliary'] if condition != 'random' else None,
                                 'checkpoint': best, 'generation_code_sha256': identity['code_sha256'],
                                 'precision': identity['precision'], **_generation_descriptors(rows)}
            precision.add(identity['precision'])
    _require(len(precision) == 1, 'Generation numerical precisions differ across runs')
    condition_stats, paired_stats = {}, []
    for condition in _CONDITIONS:
        pooled = [row for pair in pairs for row in rows_by_run[condition, int(pair['downstream'])]]
        condition_stats[condition] = _generation_descriptors(pooled)
        for scope, label, subset in _generation_subsets(pooled, masks):
            statistics.append({'condition': condition, 'downstream_seed': 'pooled_three_seeds',
                               'scope': scope, 'mask': label, **_generation_descriptors(subset)})
    for left, right in _CONTRASTS[:3]:
        all_differences = []
        for pair in pairs:
            seed = int(pair['downstream'])
            right_by_key = {(row['group'], int(row['draw'])): row for row in rows_by_run[right, seed]}
            differences = []
            for row in rows_by_run[left, seed]:
                other = right_by_key[row['group'], int(row['draw'])]
                _require(all(row[key] == other[key] for key in ('candidate_id', 'prior_seed', 'uniform_seed')),
                         f'Conditional generation attempts are not paired: {left}, {right}, {seed}')
                differences.append({'group': row['group'], 'mask_size': row['mask_size'],
                                    'difference': int(row['Hamming']) - int(other['Hamming'])})
            all_differences.extend(differences)
            for scope, label, subset in _generation_subsets(differences, masks):
                values = [row['difference'] for row in subset]
                paired_stats.append({'comparison': f'{left}_minus_{right}', 'downstream_seed': seed,
                                     'scope': scope, 'mask': label, 'paired_draws': len(values),
                                     'mean_Hamming_difference': float(np.mean(values)),
                                     'variance_Hamming_difference': float(np.var(values, ddof=0))})
        for scope, label, subset in _generation_subsets(all_differences, masks):
            values = [row['difference'] for row in subset]
            paired_stats.append({'comparison': f'{left}_minus_{right}', 'downstream_seed': 'pooled_three_seeds',
                                 'scope': scope, 'mask': label, 'paired_draws': len(values),
                                 'mean_Hamming_difference': float(np.mean(values)),
                                 'variance_Hamming_difference': float(np.var(values, ddof=0))})
    result = {'status': 'COMPLETED', 'backgrounds': 1, 'target_sha256': target_hash,
              'pipeline_seed_pairs': pairs, 'masks': groups, 'generation': generation,
              'runs': run_records, 'pooled_conditions': condition_stats,
              'statistics': statistics, 'paired_Hamming_differences': paired_stats,
              'variance_definition': 'empirical variance with divisor n, not a confidence interval',
              'generation_performed_by_summary': False,
              'interpretation': 'One KVP40 WT context; mask-specific descriptors are conditional on each fixed mask. '
                                'Pooled variances mix masks and fitted models. Neither estimates residue dependence or enzyme function.'}
    identity = {'input_sha256': hashes, 'esm_files_sha256': esm_files,
                'implementation_sha256': {name: sha256_file(Path(__file__).with_name(name))
                                          for name in ('fragment_comparison.py', 'generate.py', 'data.py')}}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'generation_summary.json', result)
    write_json(output / 'generation_identity.json', identity)
    for filename, rows in [('generation_statistics.csv', statistics), ('generation_paired_differences.csv', paired_stats)]:
        with (output / filename).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(rows)
    lines = ['# KVP40 条件生成的三种初始化比较', '',
             '九次生成使用同一个 KVP40 WT 背景，各有 48 个 NMNAT 掩码、每掩码 64 次抽样。'
             '同一下游种子内，三种初始化使用相同掩码、候选编号及潜码和残基抽样种子；'
             '共享随机数不要求抽得相同潜码或氨基酸。', '',
             '| 初始化 | 训练种子 | 次数 | Hamming 均值 | Hamming 方差 | 唯一序列 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for run in run_records.values():
        lines.append(f'| {run["condition"]} | {run["downstream_seed"]} | {run["attempts"]} | '
                     f'{run["mean_Hamming"]:.6f} | {run["variance_Hamming"]:.6f} | {run["unique_sequences"]} |')
    lines.extend(['', 'Hamming 是相对 WT 的替换数；方差为抽样记录的经验方差（分母 n），不是置信区间。'
                  '表中总体方差混合了不同掩码；合并三个种子的方差还包括模型间差异。'
                  '每个具体掩码的 48×9 组统计及按 8/16/32 位汇总保存在 generation_statistics.csv。', '',
                  '均值主要反映各位点的边缘概率，不能据此声称共享潜变量的联合建模收益。'
                  '同一 WT 的重复抽样不是独立蛋白样本；本汇总不计算生物样本置信区间，也不进行活性排序。', '',
                  '每种条件合并三次训练的唯一序列数按序列内容取并集，不累加各次唯一数。'
                  '配对 Hamming 差见 generation_paired_differences.csv，具体输入与权重校验值见 generation_identity.json。', ''])
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    return result
