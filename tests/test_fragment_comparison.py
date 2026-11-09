import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from natvqvae import fragment_comparison as comparison
from natvqvae.data import Record, generation_masks, mask_panels, sha256_file, stable_seed
from natvqvae.esm_baseline import _metric_components, summarize_rows
from natvqvae.model import AA


def test_identity_strata_and_component_means_preserve_biological_unit():
    assert [comparison._identity_stratum(x) for x in (None, .49, .5, .7, .9)] == [
        'no_qualifying_hit', 'below_0.5', '0.5_to_0.7', '0.7_to_0.9', 'at_least_0.9']
    rows = [{'sequence_id': uid, 'region': 'NMNAT', 'component': group,
             'positions_0based': [1, 2], 'score': value}
            for uid, group, value in [('a', 'large', 1), ('b', 'large', 3), ('c', 'small', 7)]]
    neighbors = {'NMNAT': {'a': .6, 'b': .6, 'c': .6}}
    values = comparison._stratified_components(rows, neighbors, 'score')
    selected = [row for row in values if row['identity_stratum'] == '0.5_to_0.7' and row['target_sites'] == '2']
    assert [(row['component'], row['nll'], row['sequences']) for row in selected] == [('large', 2., 2), ('small', 7., 1)]
    with pytest.raises(ValueError, match='missing'):
        comparison._stratified_components(rows, {'NMNAT': {}}, 'score')


def test_six_cell_interaction_and_latent_contrasts_are_paired():
    tables = {'QVAE': [], 'deterministic': []}
    seeds = [1, 2, 3]
    for arm in tables:
        tables[arm].append({'condition': 'frozen_esm2', 'downstream_seed': '', 'metric': 'factorized_nll_per_residue',
                            'region': 'NMNAT', 'identity_stratum': 'below_0.5', 'target_sites': '8',
                            'component': 'one', 'sequences': 2, 'queries': 4, 'nll': 1.2})
        for condition in comparison._CONDITIONS:
            for seed in seeds:
                for metric in comparison._METRICS:
                    score = 1.0
                    if condition == 'external':
                        score -= .2 if arm == 'QVAE' else .1
                    if metric == 'marginal_nll_per_residue' and arm == 'QVAE':
                        score += .01
                    tables[arm].append({'condition': condition, 'downstream_seed': seed, 'metric': metric,
                                        'region': 'NMNAT', 'identity_stratum': 'below_0.5', 'target_sites': '8',
                                        'component': 'one', 'sequences': 2, 'queries': 4, 'nll': score})
    summary, groups = comparison._six_cell_contrasts(tables, seeds, 100, 7)
    estimates = {row['comparison']: row for row in summary if row['scope'] == 'mean_of_pipeline_seeds'}
    assert estimates['external_minus_self:QVAE_minus_deterministic']['mean'] == pytest.approx(-.1)
    assert estimates['external:joint_minus_site_marginal']['mean'] == pytest.approx(-.01)
    assert all(row['ci95'] is None and row['groups'] == 1 for row in summary)
    assert estimates['deterministic:random_minus_frozen_esm2']['mean'] == pytest.approx(-.2)
    assert len(groups) == 19 * 3
    tables['deterministic'].pop(-2)
    with pytest.raises(ValueError, match='Incomplete'):
        comparison._six_cell_contrasts(tables, seeds, 100, 7)


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _file(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return str(path)


def _hashes(paths):
    return {str(path): sha256_file(path) for path in paths}


def _fixture(tmp_path, monkeypatch, split='validation', arm='QVAE'):
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
              'model_arm': arm, 'run_prefix': '' if arm == 'deterministic' else 'fragment_',
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
                'input_paths': shared, 'arms': {arm: {'steps': 750, 'seconds': 12., 'checkpoint_rule': 'fixed_updates',
                                                       'selection_metric': None, 'validation_curve': [], 'checkpoints': checkpoints}}})
            _file(auxiliary_dir / f'{arm}_metrics.jsonl', ''.join(
                json.dumps({'step': step, 'target_sites': 16}) + '\n' for step in range(1, 751)))
    sizes = Counter(r.component for r in records)
    for i, pair in enumerate(config['seed_pairs']):
        for condition in config['conditions']:
            seed, aux_seed = pair['downstream'], pair['auxiliary']
            name = f'{config["run_prefix"]}{condition}_seed{seed}'
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
                'status': 'COMPLETED', 'arm': arm, 'seed': seed, 'steps': 10, 'seconds': 30., 'checkpoint': checkpoint,
                'checkpoint_sha256': best['sha256'], 'checkpoints': {'best': best}, 'initialization': initialization})
            _file(trained / 'metrics.jsonl', ''.join(
                json.dumps({'step': step, 'target_sites': 32}) + '\n' for step in range(1, 11)))
            _json(trained / 'run_identity.json', {'arm': arm, 'effective_training': {**study['training'], 'seed': seed},
                  'input_sha256': _hashes(common + extra_inputs), 'esm_files_sha256': esm_files,
                  'initialization': extra_inputs[0] if extra_inputs else 'random', 'code_sha256': {'train.py': 'training-version'}})
            _json(trained / 'validation_selection.json', {'same_selection': True})
            model_rows, averaged_rows = [], []
            for row in rows:
                base = 2.0 if condition == 'random' else 1.8 + .1 * i
                if condition == 'external':
                    base += (.6 if row['component'] == 'large' else -1.0) + (.1 * i - .1)
                marginal_delta = 0 if arm == 'deterministic' else .01
                averaged_rows.append({**row, 'prior_joint_nll_per_residue': base, 'marginal_nll_per_residue': base + marginal_delta})
                for repeat in range(1 if arm == 'deterministic' else 2):
                    value = base if arm == 'deterministic' else base + (-.02 if repeat == 0 else .02)
                    model_rows.append({**row, 'weight': row['weight'] / sizes[row['component']], 'repeat': repeat,
                                       'prior_joint_nll_per_residue': value, 'marginal_nll_per_residue': value + marginal_delta,
                                       'prior_site_marginal_log_probs': [-(value + marginal_delta)] * row['sites']})
            summary = {'status': 'COMPLETED', 'arm': arm, 'split': split, 'checkpoint': checkpoint,
                       'checkpoint_sha256': best['sha256'], 'repeats': 1 if arm == 'deterministic' else 2,
                       'sample_count': None if arm == 'deterministic' else 4, 'queries': len(model_rows)}
            for metric in comparison._METRICS:
                values = _metric_components(averaged_rows, metric)
                summary[metric] = sum(v['all_panels'] for v in values.values()) / len(values)
            _json(evaluated / 'evaluation.json', summary)
            _json(evaluated / 'identity.json', {'arm': arm, 'split': split, 'limit': None, 'config': study['evaluation'],
                  'mask_seed': 7, 'amino_acid_output_count': 20, 'input_sha256': _hashes(evaluation_inputs + [checkpoint]),
                  'esm_files_sha256': esm_files, 'sequences': 3, 'groups': 2,
                  'precision': 'bfloat16 forward, float32 probabilities', 'code_sha256': {'evaluate.py': 'evaluation-version'}})
            _file(evaluated / 'per_query.jsonl', ''.join(json.dumps(r) + '\n' for r in model_rows))
    path = tmp_path / 'comparison.json'
    _json(path, config)
    return path, eval_root, esm_dir, tmp_path / 'summary', test_data, called_test


def test_validated_comparison_emits_group_scores_with_identity_and_mask_strata(tmp_path, monkeypatch):
    config, evaluated, esm, output, _, _ = _fixture(tmp_path, monkeypatch)
    data = comparison.PopulationData(None, None, None)
    neighbors = tmp_path / 'neighbors'
    _json(neighbors / 'identity.json', {'input_sha256': _hashes([config])})
    for name in ('full_length', 'nmnat', 'nudix'):
        rows = [{'sequence_id': record.sequence_id, 'component': record.component,
                 'role': 'validation', 'target_role': 'training', 'target_id': 'training_neighbor',
                 'fident': .4} for record in data.validation]
        with (neighbors / f'{name}_both_nearest.csv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    comparison.compare_fragment_transfer(config, evaluated, esm, output, neighbors_dir=neighbors)
    with (output / 'stratified_group_scores.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row['condition'] == 'external' and row['downstream_seed'] == '11'
                and row['metric'] == 'prior_joint_nll_per_residue' and row['region'] == 'NMNAT'
                and row['identity_stratum'] == 'below_0.5' and row['target_sites'] == 'all_sizes']
    assert {row['component']: int(row['sequences']) for row in selected} == {'large': 2, 'small': 1}
    identity = json.loads((output / 'identity.json').read_text())
    assert str((neighbors / 'nmnat_both_nearest.csv').resolve()) in identity['input_sha256']


@pytest.mark.parametrize('split', ['validation', 'test'])
@pytest.mark.parametrize('arm', ['QVAE', 'deterministic'])
def test_complete_comparison_averages_seeds_then_equal_groups(tmp_path, monkeypatch, split, arm):
    config, evaluated, esm, output, recovered, calls = _fixture(tmp_path, monkeypatch, split, arm)
    result = comparison.compare_fragment_transfer(config, evaluated, esm, output, split, recovered)
    value = result['overall']['external_minus_self']['prior_joint_nll_per_residue']['NMNAT']
    assert value['mean'] == pytest.approx(-.2)
    assert value['groups'] == 2
    assert value['ci95'] == pytest.approx([-1., .6])
    for seed, expected in [(11, -.3), (12, -.2), (13, -.1)]:
        assert result['per_pipeline_seed'][str(seed)]['comparisons']['external_minus_self'][
            'prior_joint_nll_per_residue']['NMNAT']['mean'] == pytest.approx(expected)
    assert len(result['runs']) == 9 and result['prediction_scoring_performed'] is False
    budget = next(run['training_budget'] for run in result['runs'].values() if run['condition'] == 'external')
    assert budget['auxiliary']['target_residue_exposures'] == 12000
    assert budget['downstream']['target_residue_exposures'] == 320
    assert result['model_arm'] == arm
    if arm == 'deterministic':
        assert result['data']['mc_repeats'] == 1
        assert '精确评分' in (output / 'REPORT.md').read_text()
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


@pytest.mark.parametrize('stage', ['auxiliary', 'downstream'])
def test_comparison_rejects_unmatched_actual_target_exposures(tmp_path, monkeypatch, stage):
    config, evaluated, esm, output, _, _ = _fixture(tmp_path, monkeypatch)
    directory, filename = (('external_pretraining_seed1', 'QVAE_metrics.jsonl') if stage == 'auxiliary'
                           else ('fragment_external_seed11', 'metrics.jsonl'))
    path = tmp_path / 'trained' / directory / filename
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]['target_sites'] += 1
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match='target-residue exposure differs'):
        comparison.compare_fragment_transfer(config, evaluated, esm, output)


def test_logged_budget_rejects_missing_steps(tmp_path):
    path = tmp_path / 'metrics.jsonl'
    path.write_text(json.dumps({'step': 2, 'target_sites': 8}) + '\n')
    with pytest.raises(ValueError, match='step order'):
        comparison._logged_training_budget(path, 2, 1.)
    path.write_text(json.dumps({'step': 1, 'target_sites': 8}) + '\n')
    with pytest.raises(ValueError, match='incomplete'):
        comparison._logged_training_budget(path, 2, 1.)


def _generation_fixture(tmp_path, monkeypatch):
    config_path, _, _, output, _, _ = _fixture(tmp_path, monkeypatch)
    config = json.loads(config_path.read_text())
    study_path = Path(config['downstream_config'])
    study = json.loads(study_path.read_text())
    study['generation'] = {'seed': 8401, 'masks_per_condition': 16, 'draws_per_mask': 64,
                           'temperature': 1.0, 'regions_1based': {'NMNAT': [4, 181]},
                           'mask_sizes': {'NMNAT': [8, 16, 32]}}
    _json(study_path, study)
    wt = (AA * 18)[:341]
    _json(Path(config['target']), {'sequence': wt, 'length': len(wt), 'name': 'test-WT',
                                 'domains': {}, 'seed_domains': {}, 'anchors': {}})
    sources = json.loads(Path(config['sources']).read_text())
    esm_files = {p.name: sha256_file(p) for p in Path(sources['esm_model']).iterdir()}
    target_hash = hashlib.sha256(wt.encode()).hexdigest()
    groups = generation_masks(study['generation'], wt)
    generated = tmp_path / 'generated'
    for pair in config['seed_pairs']:
        seed = pair['downstream']
        for condition in config['conditions']:
            name = f'fragment_{condition}_seed{seed}'
            directory = generated / name
            trained = json.loads((Path(config['training_output']) / name / 'training_summary.json').read_text())
            best = trained['checkpoints']['best']
            inputs = [config['downstream_config'], config['sources'], config['target'], best['path'],
                      sources['population'], *sources['splits'].values()]
            _json(directory / 'masks.json', groups)
            _json(directory / 'run_identity.json', {
                'generation': study['generation'], 'group_filter': None, 'draws_override': None,
                'checkpoint': best['path'], 'checkpoint_seed': seed, 'checkpoint_arm': 'QVAE',
                'target_sha256': target_hash, 'groups': 48, 'mask_batch_size': 16,
                'masks_sha256': sha256_file(directory / 'masks.json'), 'amino_acid_output_count': 20,
                'esm_files_sha256': esm_files, 'input_sha256': _hashes(inputs),
                'code_sha256': {'generate.py': 'generation-code'}, 'precision': 'float32'})
            _json(directory / 'generation.json', {'status': 'COMPLETED', 'groups': 48, 'draws_per_mask': 64,
                   'attempts': 3072, 'target_sha256': target_hash, 'checkpoint': best['path'],
                   'checkpoint_sha256': best['sha256']})
            rows = []
            for group in groups:
                for draw in range(64):
                    count = draw % 3 + {'random': 0, 'self': 1, 'external': 2}[condition]
                    sequence = list(wt)
                    for position in group['positions'][:count]:
                        sequence[position] = 'C' if wt[position] == 'A' else 'A'
                    sequence = ''.join(sequence)
                    edits = [{'position': i + 1, 'from': a, 'to': b}
                             for i, (a, b) in enumerate(zip(wt, sequence)) if a != b]
                    rows.append({'candidate_id': f'QVAE_s{seed}_{group["id"]}_d{draw:02d}',
                                 'arm': 'QVAE', 'seed': seed, 'group': group['id'], 'draw': draw,
                                 'region': 'NMNAT', 'mask_size': group['mask_size'], 'sequence': sequence,
                                 'sequence_sha256': hashlib.sha256(sequence.encode()).hexdigest(),
                                 'Hamming': count, 'edits': json.dumps(edits),
                                 'prior_seed': stable_seed(8401, seed, group['id'], 'latent'),
                                 'uniform_seed': stable_seed(8401, seed, group['id'], 'residues')})
            with (directory / 'candidates.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
    return config_path, generated, output


def test_generation_statistics_preserve_fixed_masks_and_cross_seed_union(tmp_path, monkeypatch):
    config, generated, output = _generation_fixture(tmp_path, monkeypatch)
    result = comparison.summarize_fragment_generation(config, generated, output)
    assert len(result['runs']) == 9 and result['backgrounds'] == 1
    fixed = [r for r in result['statistics'] if r['scope'] == 'mask' and r['downstream_seed'] != 'pooled_three_seeds']
    assert len(fixed) == 48 * 9
    assert all(r['attempts'] == 64 for r in fixed)
    values = [n % 3 for n in range(64)]
    expected_mean = sum(values) / 64
    expected_variance = sum((x - expected_mean) ** 2 for x in values) / 64
    run = result['runs']['fragment_random_seed11']
    assert run['mean_Hamming'] == pytest.approx(expected_mean)
    assert run['variance_Hamming'] == pytest.approx(expected_variance)
    assert result['pooled_conditions']['random']['attempts'] == 9216
    assert result['pooled_conditions']['random']['unique_sequences'] == run['unique_sequences']
    differences = [r for r in result['paired_Hamming_differences'] if r['comparison'] == 'external_minus_self']
    assert all(r['mean_Hamming_difference'] == 1 and r['variance_Hamming_difference'] == 0 for r in differences)
    assert not any('ci95' in r for r in result['statistics'])
    assert (output / 'generation_statistics.csv').exists()
    assert result['generation_performed_by_summary'] is False


@pytest.mark.parametrize('change,match', [('seed', 'random seeds'), ('missing', 'Missing generation draws'),
                                        ('hamming', 'Candidate edits'), ('mask', 'mask positions')])
def test_generation_summary_rejects_unpaired_or_inconsistent_records(tmp_path, monkeypatch, change, match):
    config, generated, output = _generation_fixture(tmp_path, monkeypatch)
    directory = generated / 'fragment_external_seed13'
    if change == 'mask':
        path = directory / 'masks.json'
        masks = json.loads(path.read_text())
        masks[0]['positions'][0] += 1
        _json(path, masks)
    else:
        path = directory / 'candidates.csv'
        with path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        if change == 'missing':
            rows.pop()
        elif change == 'seed':
            rows[0]['prior_seed'] = str(int(rows[0]['prior_seed']) + 1)
        else:
            rows[0]['Hamming'] = str(int(rows[0]['Hamming']) + 1)
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    with pytest.raises(ValueError, match=match):
        comparison.summarize_fragment_generation(config, generated, output)
    assert not output.exists()
