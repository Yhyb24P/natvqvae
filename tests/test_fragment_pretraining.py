import csv
import hashlib
import json

import pytest
import yaml

from natvqvae.data import sha256_file
from natvqvae.fragment_pretraining import (build_fragment_pretraining, extend_within_domain,
                                          pair_fragments, paired_panels)


def _sha(sequence):
    return hashlib.sha256(sequence.encode()).hexdigest()


def _write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path):
    target = 'ACDEFGHIKLMN'
    a_sequence = target[:10]
    a_id = 'aux:' + _sha(a_sequence)
    source_id = _sha(target)
    relation = tmp_path / 'relations'
    relation.mkdir()
    (relation / 'retained_aux_domains.fasta').write_text(f'>{a_id}\n{a_sequence}\n')
    (relation / 'retained_aux_masks.json').write_text(json.dumps({a_id: {'pairs': [[i, i + 1] for i in range(10)]}}))
    _write_csv(relation / 'retained_auxiliary.csv', [
        {'aux_uid': a_id, 'sequence_sha256': _sha(a_sequence), 'length': len(a_sequence),
         'status': 'retained', 'original_domain_cluster50': 'a_group', 'original_split': 'train', 'group': 'relation_group'}])
    files = list(relation.iterdir())
    (relation / 'summary.json').write_text(json.dumps({
        'status': 'complete_all_targets',
        'auxiliary_relations': {'retained_pool': {'export_status': 'complete_relation_screened_pool_not_a_training_split'}},
        'result_sha256': {path.name: sha256_file(path) for path in files}}))
    (tmp_path / 'population.fasta').write_text(f'>{source_id}\n{target}\n')
    _write_csv(tmp_path / 'train.csv', [
        {'sequence_id': source_id, 'component_v2': 's_group', 'length': len(target),
         'nmnat_cdd_from': 1, 'nmnat_cdd_to': len(target)}])
    (tmp_path / 'target.yaml').write_text(yaml.safe_dump({
        'name': 'example', 'sequence': target, 'length': len(target),
        'domains': {'NMNAT': [1, len(target)]}, 'seed_domains': {}, 'anchors': {}}))
    config = {'inputs': {'relation_directory': str(relation), 'population': str(tmp_path / 'population.fasta'),
                         'training': str(tmp_path / 'train.csv'), 'target': str(tmp_path / 'target.yaml')},
              'selection': {'min_shared_fraction': .8, 'relative_length_tolerance': .2,
                            'max_optimal_alignments': 16, 'seed': 19},
              'masking': {'sizes': [8, 16, 32], 'seed': 20},
              'training_design': {'updates': 750, 'auxiliary_validation': False},
              'output': str(tmp_path / 'output')}
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    return config, path, a_id, source_id


@pytest.mark.parametrize(('interval', 'domain_length', 'desired', 'expected'), [
    ((10, 20), 50, 30, (0, 30)),
    ((2, 12), 50, 30, (0, 30)),
    ((35, 45), 50, 30, (20, 50)),
    ((10, 20), 25, 40, (0, 25)),
    ((5, 20), 25, 10, (5, 20)),
    ((10, 20), 50, 13, (9, 22)),
])
def test_extension_preserves_common_region_and_stays_in_domain(interval, domain_length, desired, expected):
    assert extend_within_domain(*interval, domain_length, desired) == expected


def _manual_pair_inputs():
    auxiliary = {uid: {'sequence': sequence, 'mapping': {i + 1: i for i in range(4)},
                      'original_domain_cluster50': group, 'original_split': 'train'}
                 for uid, sequence, group in [('a1', 'ACDE', 'g1'), ('a2', 'ACDF', 'g1'), ('a3', 'ACDG', 'g2')]}
    training = [{'sequence_id': uid, 'component': component, 'full_length': 4,
                 'domain_start': 0, 'domain_end': 4, 'domain': 'ACDE', 'mapping': {i + 1: i for i in range(4)}}
                for uid, component in [('s1', 'c1'), ('s2', 'c2'), ('s3', 'c2')]]
    selection = {'min_shared_fraction': .8, 'relative_length_tolerance': .2, 'seed': 12}
    return auxiliary, training, selection


def test_both_arms_share_a_group_weight_and_selection_is_order_independent():
    auxiliary, training, selection = _manual_pair_inputs()
    rows, unmatched = pair_fragments(auxiliary, training, 4, selection)
    reversed_rows, _ = pair_fragments(dict(reversed(list(auxiliary.items()))), list(reversed(training)), 4, selection)
    assert not unmatched and rows == reversed_rows
    assert sum(row['weight'] for row in rows) == pytest.approx(3)
    assert {row['weight'] for row in rows if row['a_original_group'] == 'g1'} == {.75}
    assert {row['weight'] for row in rows if row['a_original_group'] == 'g2'} == {1.5}
    assert all('s_weight' not in row for row in rows)
    assert len({row['s_component'] for row in rows}) == 2


def test_relative_length_tolerance_has_no_absolute_five_residue_allowance():
    auxiliary, training, selection = _manual_pair_inputs()
    auxiliary = {'a1': auxiliary['a1']}
    training = [{**training[0], 'domain': 'ACXDEX', 'full_length': 6, 'domain_end': 6,
                 'mapping': {1: 0, 2: 1, 3: 3, 4: 4}}]
    rows, unmatched = pair_fragments(auxiliary, training, 4, selection)
    assert not rows and unmatched[0]['reason'] == 'no_source_with_required_coverage_and_length'


def test_common_mask_columns_match_and_leave_a_visible_column():
    auxiliary, training, selection = _manual_pair_inputs()
    rows, _ = pair_fragments(auxiliary, training, 4, selection)
    panels = paired_panels(rows[0], [2, 8], 81)
    assert panels == paired_panels(rows[0], [2, 8], 81)
    assert [panel['actual_size'] for panel in panels['panels']] == [2, 3]
    a_map = dict(zip(panels['a_common_positions'], panels['common_wt_columns']))
    s_map = dict(zip(panels['s_common_positions'], panels['common_wt_columns']))
    for panel in panels['panels']:
        assert [a_map[p] for p in panel['a_positions']] == panel['wt_columns']
        assert [s_map[p] for p in panel['s_positions']] == panel['wt_columns']
        assert panel['actual_size'] < len(panels['common_wt_columns'])


def test_prepare_retains_full_a_and_emits_actual_panels_and_source_identity(tmp_path):
    config, path, a_id, source_id = _fixture(tmp_path)
    summary = build_fragment_pretraining(path)
    assert summary['status'] == 'prepared_not_trained'
    assert summary['matched_pairs'] == summary['a_original_groups'] == 1
    assert summary['actual_mask_sizes']['16'] == {9: 1}
    output = tmp_path / 'output'
    row = next(csv.DictReader((output / 'pair_manifest.csv').open()))
    assert row['a_sequence_id'] == a_id and row['s_source_sequence_id'] == source_id
    assert row['a_sequence_sha256'] == a_id.split(':')[1]
    assert int(row['a_length']) == 10 and int(row['s_length']) == 10
    assert row['weight'] == '1.0'
    assert int(row['s_source_start_1based']) == 1 and int(row['s_source_end_1based']) == 10
    assert not (output / 'aux_split.csv').exists()
    assert not (output / 'validation.csv').exists()
    with pytest.raises(FileExistsError):
        build_fragment_pretraining(path)


def test_changed_retained_inputs_and_incomplete_relation_check_are_rejected(tmp_path):
    config, path, _, _ = _fixture(tmp_path)
    relation = tmp_path / 'relations'
    summary_path = relation / 'summary.json'
    summary = json.loads(summary_path.read_text())
    summary['status'] = 'complete_available_targets'
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='incomplete'):
        build_fragment_pretraining(path)
    summary['status'] = 'complete_all_targets'
    summary_path.write_text(json.dumps(summary))
    (relation / 'retained_aux_domains.fasta').write_text('>changed\nACDE\n')
    with pytest.raises(ValueError, match='SHA mismatch'):
        build_fragment_pretraining(path)


def test_source_training_identity_and_fixed_update_design_are_checked(tmp_path):
    config, path, _, source_id = _fixture(tmp_path)
    config['training_design']['auxiliary_validation'] = True
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='without auxiliary validation'):
        build_fragment_pretraining(path)
    config['training_design']['auxiliary_validation'] = False
    path.write_text(yaml.safe_dump(config))
    (tmp_path / 'population.fasta').write_text(f'>{source_id}\nWRONG\n')
    with pytest.raises(ValueError, match='SHA mismatch'):
        build_fragment_pretraining(path)
