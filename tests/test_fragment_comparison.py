import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from natvqvae import fragment_comparison as comparison
from natvqvae.data import Record, mask_panels, sha256_file
from natvqvae.esm_baseline import _metric_components, summarize_rows
from natvqvae.model import AA


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _file(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return str(path)


def _hashes(paths):
    return {str(path): sha256_file(path) for path in paths}


def _fixture(tmp_path, monkeypatch, split='validation'):
    sources = {'population': _file(tmp_path / 'population.fasta', 'population'),
               'splits': {name: _file(tmp_path / f'{name}.csv', name)
                          for name in ('training', 'validation', 'test')},
               'esm_model': str(tmp_path / 'esm')}
    for name in ('model.safetensors', 'config.json', 'vocab.txt'):
        _file(tmp_path / 'esm' / name, name)
    esm_files = {p.name: sha256_file(p) for p in (tmp_path / 'esm').iterdir()}
    study = {'training': {'seed': 0, 'mask_seed': 7, 'max_steps': 10, 'selection': {'panels': ['NMNAT_8_count']}},
             'evaluation': {'repeats': 2, 'sample_count': 4, 'batch_size': 8, 'seed': 15}}
    study_path, source_path, target_path = [tmp_path / name for name in ('study.json', 'sources.json', 'target.json')]
    for path, value in [(study_path, study), (source_path, sources), (target_path, {'sequence': AA})]:
        _json(path, value)
    records = []
    for i, first in enumerate('ACD'):
        sequence = first + AA[1:]
        uid = hashlib.sha256(sequence.encode()).hexdigest()
        records.append(Record(uid, sequence, 'large' if i < 2 else 'small', len(sequence), 2, 8, 12, 18))
    data = SimpleNamespace(study=study, sources=sources, validation=records)
    monkeypatch.setattr(comparison, 'PopulationData', lambda *args: data)
    called_test = []
    monkeypatch.setattr(comparison, 'load_recovered_test', lambda data, path: called_test.append(path) or records)
    common = [study_path, source_path, target_path, sources['population'],
              sources['splits']['training'], sources['splits']['validation']]
    evaluation_inputs = [*common, sources['splits']['test']]
    test_data = None
    if split == 'test':
        test_data = tmp_path / 'restored'
        evaluation_inputs.extend(_file(test_data / name, name) for name in ('test.csv', 'test.fasta', 'manifest.json'))
        _json(test_data / 'historical_use.json', {'status': 'previously_evaluated', 'prior_evaluations': ['old-study'],
                                               'test_sequence_count': len(records),
                                               'test_sequence_ids_sha256': hashlib.sha256(
                                                   ('\n'.join(sorted(r.sequence_id for r in records)) + '\n').encode()).hexdigest(),
                                               'flags': {'previous_method_selection': True},
                                               'limitations': ['Previously used during method development.']})
    rows = []
    for record in records:
        for panel in mask_panels(record, 7):
            rows.append({'sequence_id': record.sequence_id, 'component': record.component,
                         'panel': panel['panel'], 'region': panel['region'], 'positions_0based': panel['positions'],
                         'sites': len(panel['positions']), 'weight': panel['weight'],
                         'factorized_nll_per_residue': 1.75,
                         'target_site_log_probs': [-1.75] * len(panel['positions'])})
    esm_dir = tmp_path / 'esm_evaluated'
    esm_input_paths = (evaluation_inputs if split == 'test' else common) + list((tmp_path / 'esm').iterdir())
    _json(esm_dir / 'identity.json', {'split': split, 'amino_acid_order': AA, 'input_sha256': _hashes(esm_input_paths),
                                    'precision': 'bfloat16 forward, float32 log-softmax'})
    _json(esm_dir / 'evaluation.json', {'status': 'COMPLETED', 'split': split, 'sequences': 3,
                                      'groups': 2, 'queries': len(rows), **summarize_rows(rows)})
    _file(esm_dir / 'per_query.jsonl', ''.join(json.dumps(r) + '\n' for r in rows))
    shared = {name: _file(tmp_path / name, name)
              for name in ('pair_manifest', 'pair_panels', 'a_fragments', 's_fragments')}
    shared['target'] = str(target_path)
    train_root, eval_root = tmp_path / 'trained', tmp_path / 'evaluated'
    config = {'downstream_config': str(study_path), 'sources': str(source_path), 'target': str(target_path),
              'training_output': str(train_root), 'seed_pairs': [{'auxiliary': i, 'downstream': i + 10} for i in (1, 2, 3)],
              'conditions': ['random', 'self', 'external'], 'auxiliary_checkpoint': 'last', 'auxiliary_updates': 750,
              'auxiliary_selection': 'fixed_updates_without_validation',
              'downstream_checkpoint': 'best_by_existing_nmnat_validation_rule',
              'primary_comparison': 'external_minus_self',
              'primary_metric': 'NMNAT_prior_joint_nll_per_target_residue',
              'additional_comparisons': ['self_minus_random', 'external_minus_random', 'each_minus_frozen_esm2'],
              'uncertainty': {'bootstrap_repeats': 2000, 'bootstrap_seed': 8802}, 'auxiliary_configs': {}}
    for condition in ('self', 'external'):
        aux_config_path = tmp_path / f'{condition}.json'
        aux_config = {'fragment_condition': condition, 'paths': shared,
                      'training': {'seed': 1, 'max_steps': 750, 'checkpoint_selection': 'fixed_updates'}}
        _json(aux_config_path, aux_config)
        config['auxiliary_configs'][condition] = str(aux_config_path)
        for seed in (1, 2, 3):
            auxiliary_dir = train_root / f'{condition}_pretraining_seed{seed}'
            initial = _file(auxiliary_dir / 'initial.pt', f'initial_{seed}')
            last = _file(auxiliary_dir / 'last.pt', f'{condition}_{seed}_last')
            checkpoints = {key: {'path': path, 'sha256': sha256_file(path)} for key, path in [('initial', initial), ('last', last)]}
            inputs = {key: sha256_file(path) for key, path in shared.items()}
            inputs['config'] = sha256_file(aux_config_path)
            _json(auxiliary_dir / 'run_identity.json', {
                'effective_training': {**aux_config['training'], 'seed': seed}, 'checkpoint_rule': 'fixed_updates',
                'input_sha256': inputs, 'esm_files_sha256': esm_files})
            _json(auxiliary_dir / 'training_summary.json', {
                'status': 'COMPLETED', 'seed': seed, 'fragment_condition': condition, 'validation_sequences': 0,
                'input_paths': shared, 'arms': {'QVAE': {'steps': 750, 'checkpoint_rule': 'fixed_updates',
                                                       'selection_metric': None, 'validation_curve': [], 'checkpoints': checkpoints}}})
    sizes = Counter(r.component for r in records)
    for i, pair in enumerate(config['seed_pairs']):
        for condition in config['conditions']:
            seed, aux_seed = pair['downstream'], pair['auxiliary']
            name = f'fragment_{condition}_seed{seed}'
            trained, evaluated = train_root / name, eval_root / name
            checkpoint = _file(trained / 'best.pt', name)
            best = {'path': checkpoint, 'sha256': sha256_file(checkpoint)}
            initialization, extra_inputs = {'mode': 'random'}, []
            if condition != 'random':
                last = str(train_root / f'{condition}_pretraining_seed{aux_seed}' / 'last.pt')
                initialization = {'mode': 'checkpoint', 'path': last, 'sha256': sha256_file(last),
                                  'checkpoint_seed': aux_seed, 'checkpoint_step': 750}
                extra_inputs = [last]
            _json(trained / 'training_summary.json', {
                'status': 'COMPLETED', 'seed': seed, 'steps': 10, 'checkpoint': checkpoint,
                'checkpoint_sha256': best['sha256'], 'checkpoints': {'best': best}, 'initialization': initialization})
            _json(trained / 'run_identity.json', {'effective_training': {**study['training'], 'seed': seed},
                  'input_sha256': _hashes(common + extra_inputs), 'esm_files_sha256': esm_files,
                  'initialization': extra_inputs[0] if extra_inputs else 'random', 'code_sha256': {'train.py': 'training-version'}})
            _json(trained / 'validation_selection.json', {'same_selection': True})
            model_rows, averaged_rows = [], []
            for row in rows:
                base = 2.0 if condition == 'random' else 1.8 + .1 * i
                if condition == 'external':
                    base += (.6 if row['component'] == 'large' else -1.0) + (.1 * i - .1)
                averaged_rows.append({**row, 'prior_joint_nll_per_residue': base, 'marginal_nll_per_residue': base + .01})
                for repeat in (0, 1):
                    value = base + (-.02 if repeat == 0 else .02)
                    model_rows.append({**row, 'weight': row['weight'] / sizes[row['component']], 'repeat': repeat,
                                       'prior_joint_nll_per_residue': value, 'marginal_nll_per_residue': value + .01})
            summary = {'status': 'COMPLETED', 'split': split, 'checkpoint': checkpoint,
                       'checkpoint_sha256': best['sha256'], 'repeats': 2, 'sample_count': 4, 'queries': len(model_rows)}
            for metric in comparison._METRICS:
                values = _metric_components(averaged_rows, metric)
                summary[metric] = sum(v['all_panels'] for v in values.values()) / len(values)
            _json(evaluated / 'evaluation.json', summary)
            _json(evaluated / 'identity.json', {'split': split, 'limit': None, 'config': study['evaluation'],
                  'mask_seed': 7, 'amino_acid_output_count': 20, 'input_sha256': _hashes(evaluation_inputs + [checkpoint]),
                  'esm_files_sha256': esm_files, 'sequences': 3, 'groups': 2,
                  'precision': 'bfloat16 forward, float32 probabilities', 'code_sha256': {'evaluate.py': 'evaluation-version'}})
            _file(evaluated / 'per_query.jsonl', ''.join(json.dumps(r) + '\n' for r in model_rows))
    path = tmp_path / 'comparison.json'
    _json(path, config)
    return path, eval_root, esm_dir, tmp_path / 'summary', test_data, called_test


@pytest.mark.parametrize('split', ['validation', 'test'])
def test_complete_comparison_averages_seeds_then_equal_groups(tmp_path, monkeypatch, split):
    config, evaluated, esm, output, recovered, calls = _fixture(tmp_path, monkeypatch, split)
    result = comparison.compare_fragment_transfer(config, evaluated, esm, output, split, recovered)
    value = result['overall']['external_minus_self']['prior_joint_nll_per_residue']['NMNAT']
    assert value['mean'] == pytest.approx(-.2)
    assert value['groups'] == 2
    assert value['ci95'] == pytest.approx([-1., .6])
    for seed, expected in [(11, -.3), (12, -.2), (13, -.1)]:
        assert result['per_pipeline_seed'][str(seed)]['comparisons']['external_minus_self'][
            'prior_joint_nll_per_residue']['NMNAT']['mean'] == pytest.approx(expected)
    assert len(result['runs']) == 9 and result['prediction_scoring_performed'] is False
    assert bool(calls) == (split == 'test')
    if split == 'test':
        assert result['data_status'] == 'historically_evaluated_heldout'
        assert result['historical_use']['prior_evaluations'] == ['old-study']
        identity = json.loads((output / 'identity.json').read_text())
        assert str((recovered / 'historical_use.json').resolve()) in identity['input_sha256']
    assert (output / 'group_differences.csv').exists()
    with pytest.raises(FileExistsError):
        comparison.compare_fragment_transfer(config, evaluated, esm, output, split, recovered)


@pytest.mark.parametrize('kind,match', [('missing', 'training_summary'), ('positions', 'Masked positions'),
                                       ('checkpoint', 'checkpoint differs'), ('weights', 'ESM files differ'),
                                       ('seed', 'downstream training')])
def test_incomplete_or_inconsistent_runs_never_produce_formal_summary(tmp_path, monkeypatch, kind, match):
    config, evaluated, esm, output, _, _ = _fixture(tmp_path, monkeypatch)
    directory = evaluated / 'fragment_external_seed13'
    trained = tmp_path / 'trained/fragment_external_seed13'
    if kind == 'missing':
        (trained / 'training_summary.json').unlink()
    elif kind == 'positions':
        path = directory / 'per_query.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['positions_0based'][0] = 999
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    else:
        path = (directory / 'evaluation.json' if kind == 'checkpoint' else
                trained / ('run_identity.json' if kind == 'weights' else 'training_summary.json'))
        value = json.loads(path.read_text())
        if kind == 'checkpoint':
            value['checkpoint_sha256'] = 'wrong'
        elif kind == 'weights':
            value['esm_files_sha256']['model.safetensors'] = 'wrong'
        else:
            value['seed'] = 99
        _json(path, value)
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        comparison.compare_fragment_transfer(config, evaluated, esm, output)
    assert not output.exists()


def test_test_directory_is_never_selected_implicitly(tmp_path):
    with pytest.raises(ValueError, match='explicit recovered'):
        comparison.compare_fragment_transfer('absent', 'absent', 'absent', tmp_path, split='test')


def test_prior_test_use_cannot_be_silently_omitted(tmp_path, monkeypatch):
    config, evaluated, esm, output, recovered, _ = _fixture(tmp_path, monkeypatch, 'test')
    _json(recovered / 'historical_use.json', {'status': 'unknown'})
    with pytest.raises(ValueError, match='prior evaluation history'):
        comparison.compare_fragment_transfer(config, evaluated, esm, output, 'test', recovered)
    assert not output.exists()


def test_prior_history_must_describe_the_same_test_sequences(tmp_path, monkeypatch):
    config, evaluated, esm, output, recovered, _ = _fixture(tmp_path, monkeypatch, 'test')
    path = recovered / 'historical_use.json'
    history = json.loads(path.read_text())
    history['test_sequence_ids_sha256'] = hashlib.sha256(b'another sequence collection\n').hexdigest()
    _json(path, history)
    with pytest.raises(ValueError, match='different test sequences'):
        comparison.compare_fragment_transfer(config, evaluated, esm, output, 'test', recovered)
    assert not output.exists()
