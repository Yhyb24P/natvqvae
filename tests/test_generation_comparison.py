import json

import pytest

from natvqvae.generation_comparison import (_hamming_bin, _run_specs, _site_and_pair_stats,
                                            _verify_sequence)

TARGET = 'ACDEFGHIKLMNPQRSTVWY'
MASK = {'id': 'NMNAT_m02_r00', 'region': 'NMNAT', 'mask_size': 2, 'positions': [2, 7]}


def _row(sequence, group='NMNAT_m02_r00'):
    edits = [{'position': i + 1, 'from': a, 'to': b}
             for i, (a, b) in enumerate(zip(TARGET, sequence)) if a != b]
    return {'candidate_id': 'x', 'group': group, 'sequence': sequence,
            'sequence_sha256': __import__('hashlib').sha256(sequence.encode()).hexdigest(),
            'Hamming': len(edits), 'edits': json.dumps(edits)}


def test_sequence_verifier_accepts_mask_edits_and_rejects_outside_change():
    _verify_sequence(_row('ACAEFGHIKLMNPQRSTVWY'), TARGET, MASK)
    with pytest.raises(ValueError):
        _verify_sequence(_row('ACDEFGHIKLNNPQRSTVWY'), TARGET, MASK)


def test_site_and_pair_statistics():
    rows = [_row('ACAEFGHIKLMNPQRSTVWY'), _row('ACDEFGHIKLMNPQRSTVWY'),
            _row('ACAEFGHIKLMNPQRSTVWY'), _row('ACAEFGHIKLMNPQRSTVWY')]
    sites, pairs = _site_and_pair_stats(rows, [MASK], TARGET)
    assert {row['position'] for row in sites} == {3, 8}
    assert all(row['draws'] == 4 for row in sites)
    assert pairs[0]['pairs'] == 1
    assert pairs[0]['mean_edit_covariance'] == pytest.approx(0.0)


def test_hamming_bins_and_run_specs():
    assert [_hamming_bin(v) for v in (0, 1, 2, 3, 5, 9)] == ['0', '1-2', '1-2', '3-4', '5-8', 'over_8']
    specs = _run_specs({'conditions': ['random', 'self'], 'seeds': [9201]})
    assert len(specs) == 6
    assert {spec['distribution'] for spec in specs} == {'joint', 'site_marginal', 'factorized'}


def test_candidate_distance_uses_fixed_stable_mapping():
    import json
    import numpy as np
    from natvqvae.generation_comparison import AA_INDEX, _candidate_distance
    background = 'ACDE'
    targets = [
        {'target_id': 't1', 'mapping': {0: 0, 1: 1, 2: 2, 3: 3}, 'coverage_background': 1.0,
         'coverage_training': 1.0, 'identity': 1.0, 'n_optimal': 1, 'ambiguous': False},
        {'target_id': 't2', 'mapping': {0: 0, 1: 1, 2: 2, 3: 3}, 'coverage_background': 1.0,
         'coverage_training': 1.0, 'identity': 1.0, 'n_optimal': 1, 'ambiguous': False},
    ]
    residues = np.asarray([[AA_INDEX[c] for c in 'ACDE'], [AA_INDEX[c] for c in 'ACDF']], dtype=np.int16)
    references = {'residues': residues, 'mapped': np.asarray([4, 4]),
                  'base_mismatches': np.asarray([0, 1]), 'targets': targets,
                  'background': np.asarray([AA_INDEX[c] for c in background], dtype=np.int16),
                  'length': 4}
    row = {'sequence': 'ACDF', 'edits': json.dumps([{'position': 4, 'from': 'E', 'to': 'F'}])}
    result = _candidate_distance(row, references, 0, 4)
    assert result['target_id'] == 't2'
    assert result['target_domain_distance'] == 0.0
    assert result['edit_mismatch_fraction'] == 0.0
    assert result['domain_mapped'] == 4


def test_candidate_distance_marks_unmapped_edit():
    import json
    import numpy as np
    from natvqvae.generation_comparison import AA_INDEX, _candidate_distance
    target = {'target_id': 't1', 'mapping': {0: 0, 1: 1}, 'coverage_background': 0.5,
              'coverage_training': 1.0, 'identity': 1.0, 'n_optimal': 1, 'ambiguous': False}
    references = {'residues': np.asarray([[AA_INDEX['A'], AA_INDEX['C'], -1, -1]], dtype=np.int16),
                  'mapped': np.asarray([2]), 'base_mismatches': np.asarray([0]),
                  'targets': [target], 'background': np.asarray([AA_INDEX[c] for c in 'ACDE'], dtype=np.int16),
                  'length': 4}
    row = {'sequence': 'ACDF', 'edits': json.dumps([{'position': 4, 'from': 'E', 'to': 'F'}])}
    result = _candidate_distance(row, references, 0, 4)
    assert result['edit_mapped'] == 0
    assert result['edit_uncovered'] == 1
    assert result['edit_mismatch_fraction'] is None


def _sha(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


def test_matched_pair_samples_disjoint_draws():
    from natvqvae.generation_comparison import _matched_pair

    left = [_row('ACAEFGHIKLMNPQRSTVWY'), _row('ACAEFGHIKLMNPQRSTVWY')]
    right = [_row('ACAEFGHIKLMNPQRSTVWY'), _row('ACAEFGHIKLMNPQRSTVWY')]
    for row, draw in zip(left, (0, 1)):
        row['draw'] = draw
    for row, draw in zip(right, (2, 3)):
        row['draw'] = draw
    lm, rm, left_n, right_n, paired, left_seed, right_seed = _matched_pair(
        left, right, '1-2', ('left', 'right', 'background', 'mask', '1-2', 9201))
    assert (left_n, right_n, paired) == (2, 2, 2)
    assert len(lm) == len(rm) == 2
    assert {row['draw'] for row in lm} <= {0, 1}
    assert {row['draw'] for row in rm} <= {2, 3}
    assert left_seed is not None and right_seed is not None


def _integration_fixture(root):
    import csv
    import json
    from pathlib import Path

    from natvqvae.data import Record, sha256_file, stable_seed

    seq1 = 'ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY'
    seq2 = 'ACDEYGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWA'
    target = seq1
    masks = [
        {'id': 'NMNAT_m08_r00', 'region': 'NMNAT', 'mask_size': 8, 'positions': list(range(0, 8))},
        {'id': 'NMNAT_m08_r01', 'region': 'NMNAT', 'mask_size': 8, 'positions': list(range(8, 16))},
        {'id': 'NMNAT_m16_r00', 'region': 'NMNAT', 'mask_size': 16, 'positions': list(range(0, 16))},
        {'id': 'NMNAT_m16_r01', 'region': 'NMNAT', 'mask_size': 16, 'positions': list(range(16, 32))},
        {'id': 'NMNAT_m32_r00', 'region': 'NMNAT', 'mask_size': 32, 'positions': list(range(0, 32))},
        {'id': 'NMNAT_m32_r01', 'region': 'NMNAT', 'mask_size': 32, 'positions': list(range(8, 40))},
    ]
    backgrounds = [
        {'sequence_id': _sha(seq1), 'component': 'comp1', 'length': 40,
         'generation': {'seed': 12345, 'draws_per_mask': 1, 'temperature': 1.0},
         'masks': masks},
        {'sequence_id': _sha(seq2), 'component': 'comp2', 'length': 40,
         'generation': {'seed': 67890, 'draws_per_mask': 1, 'temperature': 1.0},
         'masks': masks},
    ]
    kvp40 = {'masks': masks, 'generation': {'seed': 54321, 'draws_per_mask': 1, 'temperature': 1.0}}
    background_path = root / 'backgrounds.json'
    background_path.write_text(json.dumps({'backgrounds': backgrounds, 'kvp40': kvp40}))

    test_data = root / 'test_data'
    test_data.mkdir()
    (test_data / 'test.csv').write_text('sequence_id\n')
    (test_data / 'test.fasta').write_text('>x\nACDE\n')
    (test_data / 'historical_use.json').write_text('{}\n')
    (test_data / 'manifest.json').write_text(json.dumps({
        'status': 'COMPLETED',
        'outputs': {name: {'sha256': sha256_file(test_data / name)}
                    for name in ('test.csv', 'test.fasta')}}))

    neighbors = root / 'neighbors'
    neighbors.mkdir()
    (neighbors / 'nmnat.fasta').write_text(f'>train1\n{seq1}\n')
    (neighbors / 'nmnat.tsv').write_text('query\ttarget\n')
    (neighbors / 'identity.json').write_text(json.dumps({'input_sha256': {}, 'alignment_sha256': {}}))
    with (neighbors / 'nmnat_both_nearest.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            'sequence_id', 'component', 'role', 'target_role', 'target_id', 'fident', 'qcov', 'tcov'])
        writer.writeheader()
        for item in backgrounds:
            writer.writerow({'sequence_id': item['sequence_id'], 'component': item['component'],
                             'role': 'test', 'target_role': 'training', 'target_id': 'train1',
                             'fident': 1.0, 'qcov': 1.0, 'tcov': 1.0})

    for name in ('sources.yaml', 'target.yaml', 'study.yaml'):
        (root / name).write_text('placeholder: true\n')

    training = {}
    for arm, prefix in (('QVAE', 'fragment_'), ('deterministic', '')):
        arm_root = root / f'training_{arm.lower()}'
        arm_root.mkdir()
        training[arm] = str(arm_root)
        for condition in ('random', 'self', 'external'):
            checkpoint = root / f'checkpoint_{arm}_{condition}.pt'
            checkpoint.write_text(f'{arm}-{condition}\n')
            name = f'{prefix}{condition}_seed9201'
            (arm_root / name).mkdir()
            (arm_root / name / 'training_summary.json').write_text(json.dumps({
                'status': 'COMPLETED', 'arm': arm, 'seed': 9201, 'steps': 2704,
                'checkpoints': {'best': {'path': str(checkpoint),
                                         'sha256': sha256_file(checkpoint)}}}))

    generation_root = root / 'generation'
    for condition in ('random', 'self', 'external'):
        for arm, distribution in (('QVAE', 'joint'), ('QVAE', 'site_marginal'),
                                  ('deterministic', 'factorized')):
            run_id = f'{arm}_{condition}_seed9201_{distribution}'
            checkpoint = root / f'checkpoint_{arm}_{condition}.pt'
            checkpoint_sha = sha256_file(checkpoint)
            for item in backgrounds + [{'sequence_id': 'KVP40', 'component': 'KVP40',
                                        'generation': kvp40['generation'], 'masks': masks,
                                        'sequence': target}]:
                background_id = None if item['sequence_id'] == 'KVP40' else item['sequence_id']
                sequence = target if background_id is None else (
                    seq1 if item['component'] == 'comp1' else seq2)
                directory = generation_root / run_id / item['sequence_id']
                directory.mkdir(parents=True)
                (directory / 'masks.json').write_text(json.dumps(masks))
                identity = {
                    'checkpoint_arm': arm, 'checkpoint_seed': 9201,
                    'distribution': distribution, 'background_id': background_id,
                    'target_sha256': _sha(sequence), 'generation': item['generation'],
                    'masks_sha256': sha256_file(directory / 'masks.json'),
                    'checkpoint_sha256': checkpoint_sha}
                (directory / 'run_identity.json').write_text(json.dumps(identity))
                (directory / 'generation.json').write_text(json.dumps({
                    'status': 'COMPLETED', 'groups': 6, 'attempts': 6, 'draws_per_mask': 1,
                    'target_sha256': identity['target_sha256'],
                    'checkpoint_sha256': checkpoint_sha}))
                rows = []
                for mask in masks:
                    position = mask['positions'][0]
                    replacement = 'A' if sequence[position] != 'A' else 'C'
                    candidate = list(sequence)
                    candidate[position] = replacement
                    candidate = ''.join(candidate)
                    edits = [{'position': position + 1, 'from': sequence[position], 'to': replacement}]
                    rows.append({
                        'candidate_id': f'{item["sequence_id"]}_{arm}_{condition}_{mask["id"]}_d00',
                        'arm': arm, 'seed': 9201, 'group': mask['id'],
                        'distribution': distribution, 'region': mask['region'],
                        'mask_size': mask['mask_size'], 'draw': 0,
                        'sequence': candidate, 'sequence_sha256': _sha(candidate),
                        'Hamming': 1, 'edits': json.dumps(edits),
                        'uniform_seed': stable_seed(item['generation']['seed'], 9201,
                                                    mask['id'], 'residues')})
                with (directory / 'candidates.csv').open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader(); writer.writerows(rows)
                (directory / 'candidates.fasta').write_text(''.join(
                    f'>{row["candidate_id"]}\n{row["sequence"]}\n' for row in rows))
    config = {
        'backgrounds': str(background_path), 'generation_root': str(generation_root),
        'sources': str(root / 'sources.yaml'), 'target': str(root / 'target.yaml'),
        'study_config': str(root / 'study.yaml'), 'test_data': str(test_data),
        'neighbors': str(neighbors), 'training': training,
        'run_prefix': {'QVAE': 'fragment_', 'deterministic': ''},
        'conditions': ['random', 'self', 'external'], 'seeds': [9201],
        'downstream_steps': 2704, 'draws_per_mask': 1, 'expected_components': 2,
        'max_optimal_alignments': 16, 'kvp40_id': 'KVP40',
        'uncertainty': {'bootstrap_repeats': 20, 'bootstrap_seed': 3}}
    config_path = root / 'comparison.yaml'
    config_path.write_text(json.dumps(config))
    records = [Record(backgrounds[0]['sequence_id'], seq1, 'comp1', 40, 1, 10, 11, 20),
               Record(backgrounds[1]['sequence_id'], seq2, 'comp2', 40, 1, 10, 11, 20)]
    return config_path, records, seq1


def test_compare_generation_full_flow(tmp_path, monkeypatch):
    import csv
    import json

    import natvqvae.generation_comparison as module

    config_path, records, target = _integration_fixture(tmp_path)

    class FakeTarget:
        sequence = target

    class FakeData:
        def __init__(self, *args, **kwargs):
            self.target = FakeTarget()

    monkeypatch.setattr(module, 'PopulationData', FakeData)
    monkeypatch.setattr(module, 'load_recovered_test', lambda data, directory: records)
    monkeypatch.setattr(module, '_alignment_tables', lambda config: ({}, {'train1': target}))
    monkeypatch.setattr(module, '_domain_bounds', lambda background, data: (0, 40))
    monkeypatch.setattr(module, '_stable_targets', lambda domain, training, maximum, minimum_coverage=0.8: [{
        'target_id': 'train1', 'mapping': {index: index for index in range(len(domain))},
        'coverage_background': 1.0, 'coverage_training': 1.0, 'identity': 1.0,
        'n_optimal': 1, 'ambiguous': False}])

    output = tmp_path / 'comparison'
    result = module.compare_generation(config_path, output)
    assert result['status'] == 'COMPLETED'
    for name in ('run_summary.csv', 'group_summary.csv', 'site_frequencies.csv',
                 'pair_dependence.csv', 'hamming_strata.csv', 'hamming_paired.csv',
                 'paired_comparisons.csv', 'component_effects.csv',
                 'candidate_distances.csv', 'summary.json'):
        assert (output / name).stat().st_size > 0

    with (output / 'paired_comparisons.csv').open(newline='') as handle:
        comparisons = list(csv.DictReader(handle))
    keys = [(row['comparison'], row['left_run'], row['right_run'], row['background'], row['seed'])
            for row in comparisons]
    assert len(keys) == len(set(keys))
    assert all(row['condition'] and row['left_run'] and row['right_run'] for row in comparisons)
    assert any(row['contrast_kind'] == 'external_minus_self' and
               row['condition'] == 'external_minus_self' for row in comparisons)

    with (output / 'hamming_paired.csv').open(newline='') as handle:
        paired = list(csv.DictReader(handle))
    paired_keys = [(row['comparison'], row['condition'], row['left_run'], row['right_run'],
                    row['background'], row['mask'], row['stratum'])
                   for row in paired]
    assert len(paired_keys) == len(set(paired_keys))
    assert all(row['paired_attempts'] == '1' and row['discarded_left'] == '0'
               and row['discarded_right'] == '0'
               for row in paired if row['stratum'] == '1-2')
    assert all(row['paired_attempts'] == '0' and row['unique_difference'] == 'NA'
               for row in paired if row['stratum'] != '1-2')

    with (output / 'candidate_distances.csv').open(newline='') as handle:
        distances = list(csv.DictReader(handle))
    assert len(distances) == 9 * 3 * 6
    assert all(row['distance_available'] == 'True' for row in distances)
    assert all(row['target_id'] == 'train1' for row in distances)

    summary = json.loads((output / 'summary.json').read_text())
    assert summary['bootstrap']['unit'] == 'component'
    assert summary['candidate_distance_rows'] == len(distances)
    assert len(summary['generation_sha256']) == 9 * 3 * 5
    assert len(summary['training_summary_sha256']) == 6
    for name in ('generation_effects.csv', 'generation_effects.pdf',
                 'generation_effects.png', 'REPORT.md'):
        assert (output / name).stat().st_size > 0
    with (output / 'generation_effects.csv').open(newline='') as handle:
        effects = list(csv.DictReader(handle))
    assert len(effects) == (2 + 3) * 3 * 2
    assert {row['metric'] for row in effects} == {'mean_Hamming', 'unique_sequences', 'pair_covariance'}
    report = (output / 'REPORT.md').read_text()
    assert 'KVP40' in report and '第 45' in report and 'discarded_left/right' in report

    with (output / 'component_effects.csv').open(newline='') as handle:
        effects = list(csv.DictReader(handle))
    heldout = [row for row in effects if row['analysis_unit'] == 'heldout_component']
    assert heldout and all(row['components'] in ('2', '0') for row in heldout)
    assert any(row['analysis_unit'] == 'KVP40_application_background' for row in effects)


def test_generation_report_writes_effect_figure_and_kvp40_table(tmp_path):
    from natvqvae.generation_comparison import _write_generation_outputs

    definitions = [
        ('QVAE_joint_external_minus_self', 'external_minus_self'),
        ('deterministic_factorized_external_minus_self', 'external_minus_self'),
        ('joint_minus_site_marginal', 'random'),
        ('joint_minus_site_marginal', 'self'),
        ('joint_minus_site_marginal', 'external'),
    ]
    metrics = ('mean_Hamming', 'unique_sequences', 'pair_covariance')
    rows = []
    for label, condition in definitions:
        for metric in metrics:
            rows.append({'comparison': label, 'condition': condition, 'seed': 9201,
                         'metric': metric, 'analysis_unit': 'heldout_component',
                         'mean': 0.25, 'ci95_low': 0.10, 'ci95_high': 0.40, 'components': 2})
            rows.append({'comparison': label, 'condition': condition, 'seed': 'pooled',
                         'metric': metric, 'analysis_unit': 'heldout_component',
                         'mean': 0.30, 'ci95_low': 0.12, 'ci95_high': 0.45, 'components': 2})
            rows.append({'comparison': label, 'condition': condition, 'seed': 9201,
                         'metric': metric, 'analysis_unit': 'KVP40_application_background',
                         'mean': 0.50, 'ci95_low': 'NA', 'ci95_high': 'NA', 'components': 1})
            rows.append({'comparison': label, 'condition': condition, 'seed': 'pooled',
                         'metric': metric, 'analysis_unit': 'KVP40_application_background',
                         'mean': 0.50, 'ci95_low': 'NA', 'ci95_high': 'NA', 'components': 1})
    run_rows = [{'run': 'run1', 'background': 'background1'}, {'run': 'run1', 'background': 'KVP40'}]
    hamming_rows = [{'paired_attempts': 10, 'discarded_left': 1, 'discarded_right': 2,
                     'unique_difference': '0'},
                    {'paired_attempts': 0, 'discarded_left': 0, 'discarded_right': 0,
                     'unique_difference': 'NA'}]
    _write_generation_outputs(rows, run_rows, hamming_rows, tmp_path,
                              {'seeds': [9201], 'draws_per_mask': 1})
    for name in ('generation_effects.csv', 'generation_effects.pdf',
                 'generation_effects.png', 'REPORT.md'):
        assert (tmp_path / name).stat().st_size > 0
    report = (tmp_path / 'REPORT.md').read_text()
    assert '有效 seed' in report and '+0.50000000' in report and '| NA |' in report
    assert '10 个匹配对' in report and '丢弃计数合计 3' in report


def test_component_effects_preserve_condition_and_seed_identity():
    from natvqvae.generation_comparison import _METRICS, _component_effects

    effects = {'random': 1.0, 'self': 2.0, 'external': 3.0}
    rows = []
    for condition, effect in effects.items():
        for seed in (9201, 9202):
            for background in ('comp1', 'comp2', 'KVP40'):
                row = {'comparison': 'joint_minus_site_marginal', 'contrast_kind': 'within_condition',
                       'condition': condition, 'background': background, 'seed': seed,
                       'left_run': f'left_{condition}_{seed}', 'right_run': f'right_{condition}_{seed}'}
                for metric, left_key, right_key, diff_key in _METRICS:
                    row[left_key] = 0.0
                    row[right_key] = 0.0
                    row[diff_key] = effect if metric == 'mean_Hamming' else 0.0
                rows.append(row)
    entries = [{'id': 'comp1', 'component': 'comp1', 'record': object()},
               {'id': 'comp2', 'component': 'comp2', 'record': object()},
               {'id': 'KVP40', 'component': 'KVP40', 'record': None}]
    meta = [(label, 'within_condition', condition, f'left_{condition}_{seed}',
             f'right_{condition}_{seed}')
            for seed in (9201, 9202) for condition in ('random', 'self', 'external')
            for label in ('joint_minus_site_marginal',)]
    result, _ = _component_effects(rows, entries, [9201, 9202], meta,
                                   {'uncertainty': {'bootstrap_repeats': 5, 'bootstrap_seed': 1}})
    keys = [(row['comparison'], row['contrast_kind'], row['condition'], row['analysis_unit'],
             row['seed'], row['metric']) for row in result]
    assert len(keys) == len(set(keys))
    for seed in (9201, 9202):
        for condition, effect in effects.items():
            row = next(row for row in result if row['condition'] == condition and row['seed'] == seed and
                       row['analysis_unit'] == 'heldout_component' and row['metric'] == 'mean_Hamming')
            assert row['mean'] == pytest.approx(effect)
            assert row['left_run'] == f'left_{condition}_{seed}'
            assert row['right_run'] == f'right_{condition}_{seed}'
            kvp40 = next(row for row in result if row['condition'] == condition and row['seed'] == seed and
                         row['analysis_unit'] == 'KVP40_application_background' and
                         row['metric'] == 'mean_Hamming')
            assert kvp40['mean'] == pytest.approx(effect)
    pooled = [row for row in result if row['seed'] == 'pooled' and row['condition'] == 'external' and
              row['analysis_unit'] == 'heldout_component' and row['metric'] == 'mean_Hamming']
    assert len(pooled) == 1
    assert pooled[0]['mean'] == pytest.approx(3.0)
    assert pooled[0]['left_run'] == 'pooled_across_seeds'
    assert set(pooled[0]['left_runs'].split(';')) == {'left_external_9201', 'left_external_9202'}


def test_component_effects_treat_kvp40_as_single_background_without_bootstrap():
    from natvqvae.generation_comparison import _METRICS, _component_effects

    rows = []
    for seed in (9201, 9202):
        for background, value in (('comp1', 2.0 * (seed - 9200)), ('KVP40', 3.0 + (seed - 9201))):
            row = {'comparison': 'joint_minus_site_marginal', 'contrast_kind': 'within_condition',
                   'condition': 'external', 'background': background, 'seed': seed,
                   'left_run': f'left_{seed}', 'right_run': f'right_{seed}'}
            for metric, left_key, right_key, diff_key in _METRICS:
                row[left_key] = 1.0
                row[right_key] = 1.0
                row[diff_key] = 'NA' if metric == 'unique_sequences' else value
            rows.append(row)
    entries = [{'id': 'comp1', 'component': 'comp1', 'record': object()},
               {'id': 'KVP40', 'component': 'KVP40', 'record': None}]
    meta = [('joint_minus_site_marginal', 'within_condition', 'external',
             'left_9201', 'right_9201')]
    result, _ = _component_effects(rows, entries, [9201, 9202], meta,
                                   {'uncertainty': {'bootstrap_repeats': 5, 'bootstrap_seed': 1}})

    def find(unit, seed, metric):
        return next(row for row in result if row['analysis_unit'] == unit and
                    row['seed'] == seed and row['metric'] == metric)

    for seed, expected in ((9201, 2.0), (9202, 4.0)):
        row = find('heldout_component', seed, 'mean_Hamming')
        assert row['mean'] == pytest.approx(expected)
        assert row['ci95_low'] == row['ci95_high'] == 'NA'
        assert row['components'] == 1 and row['components_missing'] == 0
    assert find('KVP40_application_background', 9201, 'mean_Hamming')['mean'] == pytest.approx(3.0)
    assert find('KVP40_application_background', 9202, 'mean_Hamming')['mean'] == pytest.approx(4.0)
    pooled = find('KVP40_application_background', 'pooled', 'mean_Hamming')
    assert pooled['mean'] == pytest.approx(3.5)
    assert pooled['ci95_low'] == pooled['ci95_high'] == 'NA'
    assert pooled['components'] == 1 and pooled['components_missing'] == 0
    missing = find('KVP40_application_background', 9201, 'unique_sequences')
    assert missing['mean'] == 'NA' and missing['components'] == 0 and missing['components_missing'] == 1
    heldout_missing = find('heldout_component', 9201, 'unique_sequences')
    assert heldout_missing['mean'] == 'NA' and heldout_missing['components'] == 0
    assert heldout_missing['components_missing'] == 1


def test_component_effects_drop_none_side_values_without_crashing():
    from natvqvae.generation_comparison import _METRICS, _component_effects

    rows = []
    for seed in (9201, 9202):
        for background in ('comp1', 'comp2'):
            row = {'comparison': 'joint_minus_site_marginal', 'contrast_kind': 'within_condition',
                   'condition': 'external', 'background': background, 'seed': seed,
                   'left_run': f'left_{seed}', 'right_run': f'right_{seed}'}
            for metric, left_key, right_key, diff_key in _METRICS:
                if background == 'comp1' and metric == 'target_domain_distance':
                    row[left_key] = None
                    row[right_key] = None
                    row[diff_key] = 'NA'
                else:
                    row[left_key] = 1.0
                    row[right_key] = 2.0
                    row[diff_key] = 1.0
            rows.append(row)
    entries = [{'id': 'comp1', 'component': 'comp1', 'record': object()},
               {'id': 'comp2', 'component': 'comp2', 'record': object()}]
    meta = [('joint_minus_site_marginal', 'within_condition', 'external', 'left', 'right')]
    result, _ = _component_effects(rows, entries, [9201, 9202], meta,
                                   {'uncertainty': {'bootstrap_repeats': 5, 'bootstrap_seed': 1}})
    row = next(item for item in result if item['analysis_unit'] == 'heldout_component' and
               item['metric'] == 'target_domain_distance' and item['seed'] == 'pooled')
    assert row['mean'] == pytest.approx(1.0)
    assert row['left_mean'] == 'NA' and row['right_mean'] == 'NA'
    assert row['components'] == 1 and row['components_missing'] == 1
