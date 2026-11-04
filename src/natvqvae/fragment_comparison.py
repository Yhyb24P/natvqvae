"""Paired prediction comparison for the three NatV fragment-pretraining conditions."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .data import PopulationData, load_yaml, mask_panels, sha256_file, write_json
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


def _report(result):
    primary = result['overall']['external_minus_self'][_METRICS[0]]['NMNAT']
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
        f'QVAE 的每条任务先平均 {result["data"]["mc_repeats"]} 次 Monte Carlo NLL，再计算序列组均值。'
        '总体比较先在各组内平均三对训练种子的差值，再按组重采样。'
        '区间以这三对已训练模型为条件，不包含训练种子不确定性，也不将重复任务视为独立蛋白。', '',
        'NMNAT 的 external − self 为预定主要比较；其他结果为补充比较，区间未作多重比较调整。', '',
        'ESM-2 采用共同遮蔽输入下的位点概率乘积；QVAE 联合分数包含共享潜变量。'
        'QVAE 边缘概率乘积的结果单独保存在 summary.json，不混同两种分布。', '',
        '逐训练种子结果见 summary.json，逐组分数与差值见 group_scores.csv 和 group_differences.csv。'
        '所有九份模型评价均已完成，并核对相同任务、输入、ESM 权重及各自选定的模型权重。', '',
        ('这些验证序列参与了下游权重选择，结果不能代替独立测试。' if result['split'] == 'validation' else
         '这些保留序列未用于本轮训练，但此前已有预测评价，且部分序列曾影响方法选择，'
         '不能称为首次检验或方法开发从未接触的数据。本次使用下游验证已选定的权重，'
         '汇总时不依据保留集分数重新选择模型。具体历史记录与限制保存在 summary.json 的 historical_use 中。'),
        '本程序只汇总已保存分数，没有执行训练或新的预测。NLL 不表示酶学活性。', ''])
    return '\n'.join(lines)


def compare_fragment_transfer(config_path, evaluation_dir, esm_dir, output,
                              split='validation', test_data=None):
    """Summarize nine completed runs in an explicit split directory; never score models."""
    _require(split in ('validation', 'test') and ((split == 'test') == (test_data is not None)),
             'Test comparison requires explicit recovered test data; validation must not use it')
    config = load_yaml(config_path)
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
    _require(esm_summary['queries'] == len(expected) and esm_summary['sequences'] == len(records) and
             esm_summary['groups'] == len(component_sizes), 'ESM summary counts differ')
    for region in _REGIONS:
        _require(np.isclose(np.mean([g[region] for g in esm_components.values()]),
                            esm_summary['mean_nll_per_residue'][region], rtol=1e-10),
                 f'ESM summary score differs: {region}')
    train_root = Path(config['training_output'])
    model_scores, run_records, auxiliary_initials, selection_hashes = {}, {}, {}, set()
    auxiliary_pair_inputs = set()
    repeats = int(study['evaluation']['repeats'])
    for pair in pairs:
        auxiliary_seed, downstream_seed = int(pair['auxiliary']), int(pair['downstream'])
        for condition in _CONDITIONS:
            name = f'fragment_{condition}_seed{downstream_seed}'
            directory, train_directory = evaluation_dir / name, train_root / name
            training = read(train_directory / 'training_summary.json')
            train_identity = read(train_directory / 'run_identity.json')
            expected_training = {**study['training'], 'seed': downstream_seed}
            _require(training['status'] == 'COMPLETED' and training['seed'] == downstream_seed and
                     training['steps'] == int(study['training']['max_steps']) and
                     train_identity['effective_training'] == expected_training,
                     f'Incomplete or different downstream training: {name}')
            _input_hashes(train_identity, common_inputs, digest)
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
                arm = auxiliary['arms']['QVAE']
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
            _require(summary['status'] == 'COMPLETED' and summary['split'] == split and
                     identity['split'] == split and identity['limit'] is None and
                     identity['config'] == study['evaluation'] and
                     identity['mask_seed'] == study['training']['mask_seed'] and
                     identity['amino_acid_output_count'] == 20 and
                     summary['repeats'] == repeats and
                     summary['sample_count'] == study['evaluation']['sample_count'],
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
            for metric in _METRICS:
                actual = np.mean([g['all_panels'] for g in model_scores[condition, downstream_seed][metric].values()])
                _require(np.isclose(actual, summary[metric], rtol=1e-9), f'Evaluation summary score differs: {name}, {metric}')
            run_records[name] = {'condition': condition, 'downstream_seed': downstream_seed,
                                 'auxiliary_seed': auxiliary_seed if condition != 'random' else None,
                                 'checkpoint': best, 'initialization': initialization, 'pairing': checks,
                                 'precision': identity['precision'],
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
    result = {'status': 'COMPLETED', 'split': split, 'primary_comparison': 'external_minus_self',
              'data_status': 'historically_evaluated_heldout' if split == 'test' else 'development_validation',
              'historical_use': historical_use,
              'primary_metric': 'NMNAT_prior_joint_nll_per_target_residue', 'units': 'nats per target residue',
              'data': {'sequences': len(records), 'groups': len(component_sizes), 'queries': len(expected),
                       'mc_repeats': repeats, 'mask_seed': study['training']['mask_seed']},
              'pipeline_seed_pairs': pairs, 'runs': run_records, 'esm_pairing': esm_checks,
              'bootstrap': {'unit': 'component_v2', 'repetitions': bootstrap_repeats, 'seed': bootstrap_seed,
                            'scope': 'conditional on three fitted pipeline replicates; group bootstrap after averaging seed differences'},
              'distributions': {'prior_joint_nll_per_residue': 'shared-latent QVAE joint',
                                'marginal_nll_per_residue': 'product of QVAE site marginals',
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
                       ('comparisons.csv', summary_rows)]:
        with (output / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    (output / 'REPORT.md').write_text(_report(result), encoding='utf-8')
    return result
