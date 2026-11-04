import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from natvqvae.fragment_overlap import (ALIGNMENT_FIELDS, _Groups, _auxiliary_groups, _diagnostic,
                                      _group_membership, _retained_pool, check_fragment_overlap,
                                      exact_fragment_hits, load_overlap_inputs)


def _sha(sequence):
    return hashlib.sha256(sequence.encode()).hexdigest()


def _csv(path, rows, fields):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path):
    fragment = 'ACDEFGHIKLMNPQRSTVWY' * 6 + 'ACDE'
    full = 'MMMMMMMMM' + fragment + 'Y' * 49
    query_id, target_id = 'aux:' + _sha(fragment), _sha(full)
    (tmp_path / 'aux.fasta').write_text(f'>{query_id}\n{fragment}\n')
    (tmp_path / 'population.fasta').write_text(f'>{target_id}\n{full}\n')
    split = {'aux_uid': query_id, 'sequence_sha256': _sha(fragment), 'length': len(fragment),
             'aux_kind': 'local_core', 'split': 'train'}
    _csv(tmp_path / 'aux.csv', [split], list(split))
    rows = [{'sequence_id': target_id, 'component_v2': 'g1', 'length': len(full),
             'nmnat_cdd_from': 1, 'nmnat_cdd_to': 182}]
    _csv(tmp_path / 'validation.csv', rows, list(rows[0]))
    config = {'auxiliary': {'domains': str(tmp_path / 'aux.fasta'), 'split': str(tmp_path / 'aux.csv')},
              'natv': {'population': str(tmp_path / 'population.fasta'),
                       'splits': {'validation': str(tmp_path / 'validation.csv')}},
              'output': str(tmp_path / 'result')}
    return config, fragment, full, query_id, target_id


def test_exact_containment_is_not_bidirectional_domain_coverage(tmp_path):
    config, fragment, full, query_id, target_id = _fixture(tmp_path)
    auxiliary, targets, _, missing = load_overlap_inputs(config)
    hits = exact_fragment_hits(auxiliary, targets)
    assert not missing and len(hits) == 1
    assert hits[0]['target_start'] == 10 and hits[0]['target_end'] == 133
    assert hits[0]['nmnat_covered_fraction'] == pytest.approx(124 / 182)
    assert hits[0]['nmnat_covered_fraction'] < 0.8


def test_duplicate_split_and_bad_auxiliary_hash_are_rejected(tmp_path):
    config, _, _, _, _ = _fixture(tmp_path)
    path = Path(config['auxiliary']['split'])
    rows = list(csv.DictReader(path.open()))
    _csv(path, rows * 2, list(rows[0]))
    with pytest.raises(ValueError, match='Duplicate auxiliary'):
        load_overlap_inputs(config)
    rows[0]['sequence_sha256'] = '0' * 64
    _csv(path, rows, list(rows[0]))
    with pytest.raises(ValueError, match='identity mismatch'):
        load_overlap_inputs(config)


def test_duplicate_fasta_id_is_rejected(tmp_path):
    config, fragment, _, query_id, _ = _fixture(tmp_path)
    Path(config['auxiliary']['domains']).write_text(f'>{query_id}\n{fragment}\n' * 2)
    with pytest.raises(ValueError, match='Duplicate FASTA'):
        load_overlap_inputs(config)


def test_recovery_by_content_hash_keeps_multiple_sources_and_missing_ids(tmp_path):
    config, _, _, _, _ = _fixture(tmp_path)
    test_sequence = 'Q' * 40
    test_id, missing_id = _sha(test_sequence), _sha('H' * 41)
    rows = [{'sequence_id': test_id, 'component_v2': 'g2'},
            {'sequence_id': missing_id, 'component_v2': 'g3'}]
    _csv(tmp_path / 'test.csv', rows, list(rows[0]))
    config['natv']['splits']['test'] = str(tmp_path / 'test.csv')
    for name, header in [('source1', 'original_accession'), ('source2', test_id)]:
        (tmp_path / f'{name}.fasta').write_text(f'>{header}\n{test_sequence}\n')
    config['natv']['source_fastas'] = [str(tmp_path / f'source{i}.fasta') for i in [1, 2]]
    _, targets, provenance, missing = load_overlap_inputs(config)
    assert targets[test_id]['sequence'] == test_sequence
    assert [r['sequence_id'] for r in missing] == [missing_id]
    assert len([r for r in provenance if r['sequence_id'] == test_id]) == 2
    (tmp_path / 'source2.fasta').write_text(f'>{test_id}\nWRONG\n')
    with pytest.raises(ValueError, match='SHA mismatch'):
        load_overlap_inputs(config)


def test_target_ids_and_components_cannot_cross_splits(tmp_path):
    config, _, _, _, target_id = _fixture(tmp_path)
    path = tmp_path / 'training.csv'
    _csv(path, [{'sequence_id': target_id, 'component_v2': 'g1'}], ['sequence_id', 'component_v2'])
    config['natv']['splits']['training'] = str(path)
    with pytest.raises(ValueError, match='cross-split target ID'):
        load_overlap_inputs(config)
    _csv(path, [{'sequence_id': _sha('Q'), 'component_v2': 'g1'}], ['sequence_id', 'component_v2'])
    with pytest.raises(ValueError, match='component occurs in multiple'):
        load_overlap_inputs(config)


def test_unrequested_source_hash_label_does_not_override_sequence_identity(tmp_path):
    config, _, _, _, _ = _fixture(tmp_path)
    test_sequence = 'Q' * 40
    test_id = _sha(test_sequence)
    rows = [{'sequence_id': test_id, 'component_v2': 'g2'}]
    _csv(tmp_path / 'test.csv', rows, list(rows[0]))
    config['natv']['splits']['test'] = str(tmp_path / 'test.csv')
    source = tmp_path / 'source.fasta'
    source.write_text(f'>{"0" * 64}\nUNRELATED\n>{"1" * 64}\n{test_sequence}\n')
    config['natv']['source_fastas'] = [str(source)]
    _, targets, provenance, missing = load_overlap_inputs(config)
    assert not missing
    assert targets[test_id]['sequence'] == test_sequence
    recovered = [row for row in provenance if row['sequence_id'] == test_id]
    assert len(recovered) == 1
    assert recovered[0]['source_id'] == '1' * 64
    assert recovered[0]['sequence_sha256'] == test_id
    source.write_text(f'>{test_id}\nWRONG\n')
    with pytest.raises(ValueError, match='SHA mismatch'):
        load_overlap_inputs(config)


def test_cluster_fasta_empty_titles_and_duplicate_records_are_preserved(tmp_path):
    config, _, full, _, target_id = _fixture(tmp_path)
    source = tmp_path / 'cluster.fasta'
    source.write_text(f'>{target_id}\n>{target_id}\n{full}\n>{target_id}\n{full}\n')
    config['natv']['source_fastas'] = [str(source)]
    counts = {}
    _, targets, provenance, missing = load_overlap_inputs(config, counts)
    assert not missing and targets[target_id]['sequence'] == full
    source_rows = [row for row in provenance if row['source_path'] == str(source)]
    assert [row['source_record_index'] for row in source_rows] == [2, 3]
    assert counts[str(source)] == {'records': 3, 'nonempty_records': 2,
                                  'empty_records_skipped': 1, 'duplicate_nonempty_ids': 1}
    # The same format is not allowed for an actual current model input.
    config['natv']['population'] = str(source)
    with pytest.raises(ValueError, match='empty sequence'):
        load_overlap_inputs(config)


def test_complete_check_and_existing_output_protection(tmp_path):
    config, _, _, _, _ = _fixture(tmp_path)
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    summary = check_fragment_overlap(path)
    assert summary['status'] == 'complete_all_targets'
    assert summary['exact_hits']['validation']['target_components'] == 1
    assert not summary['model_fitting_or_scoring']
    assert json.loads((tmp_path / 'result/summary.json').read_text())['implementation_sha256']
    with pytest.raises(FileExistsError):
        check_fragment_overlap(path)


def test_validation_exclusion_is_a_fixed_model_diagnostic(tmp_path):
    path = tmp_path / 'paired.json'
    path.write_text(json.dumps({'component_differences': {'prior_joint_nll_per_residue': {
        'seed1': {'NMNAT': {'overlap': -0.2, 'other': 0.1}},
        'seed2': {'NMNAT': {'overlap': 0.0, 'other': 0.3}},
    }}}))
    diagnostic = _diagnostic(path, {'overlap'})
    assert diagnostic['all_components']['mean'] == pytest.approx(0.05)
    assert diagnostic['all_components']['negative_components'] == 1
    remaining = diagnostic['without_exact_overlap_components']
    assert remaining['components'] == 1
    assert remaining['mean'] == pytest.approx(0.2)
    assert remaining['ci95'] == pytest.approx([0.2, 0.2])
    assert 'not_retraining' in diagnostic['status']


def test_changed_historical_training_inputs_are_rejected(tmp_path):
    config, _, _, _, _ = _fixture(tmp_path)
    record = tmp_path / 'training.json'
    record.write_text(json.dumps({'input_sha256': {'domains': '0' * 64}}))
    config['aux_training_record'] = str(record)
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='Actual auxiliary training input SHA mismatch'):
        check_fragment_overlap(path)


def test_select_both_original_splits_keeps_roles_and_excludes_unsplit_risk(tmp_path):
    config, _, _, _, _ = _fixture(tmp_path)
    extra, risk = 'QQQQQQQQ', 'HHHHHHHH'
    uid, risk_id = 'aux:' + _sha(extra), 'aux:' + _sha(risk)
    with Path(config['auxiliary']['domains']).open('a') as handle:
        handle.write(f'>{uid}\n{extra}\n>{risk_id}\n{risk}\n')
    path = Path(config['auxiliary']['split'])
    rows = list(csv.DictReader(path.open()))
    rows.append({'aux_uid': uid, 'sequence_sha256': _sha(extra), 'length': len(extra),
                 'aux_kind': 'local_core', 'split': 'val'})
    _csv(path, rows, list(rows[0]))
    original, _, _, _ = load_overlap_inputs(config)
    assert len(original) == 1
    config['auxiliary']['selected_splits'] = ['train', 'val']
    both, _, _, _ = load_overlap_inputs(config)
    assert len(both) == 2 and both[uid]['split'] == 'val' and risk_id not in both


def test_group_exclusion_propagates_but_training_match_does_not_exclude():
    auxiliary = {uid: {'sequence': uid * 10, 'kind': 'local_core', 'split': split}
                 for uid, split in [('A', 'train'), ('B', 'val'), ('C', 'train'),
                                    ('D', 'val'), ('E', 'train')]}
    groups = _Groups(auxiliary)
    groups.add('A', 'B', 'exact_containment', 'pairs.csv', 1)
    groups.add('B', 'C', 'local_similarity', 'hits.tsv.gz', 2)
    groups.add('D', 'E', 'original_domain_cluster50', 'input_aux_split', 'old_group')
    hits = [{'aux_uid': uid, 'role': role, 'target_id': uid + '_target',
             'evidence_file': 'similarity_hits.csv', 'evidence_row': index}
            for index, (uid, role) in enumerate([('A', 'validation'), ('D', 'training')], 1)]
    rows = {row['aux_uid']: row for row in _group_membership(auxiliary, groups, hits)}
    assert rows['A']['reason'] == 'direct_validation_or_test_match'
    assert rows['C']['reason'] == 'connected_to_validation_or_test_match'
    assert rows['C']['excluded_via_aux_uid'] == 'A'
    assert json.loads(rows['C']['exclusion_path_edge_ids']) == [1, 2]
    assert rows['E']['status'] == 'retained' and rows['E']['reason'] == 'training_related_only'
    assert rows['E']['direct_natv_roles'] == '' and rows['E']['group_natv_roles'] == 'training'


def test_auxiliary_graph_keeps_old_groups_exact_containment_and_short_coverage(tmp_path, monkeypatch):
    auxiliary = {
        'A': {'sequence': 'ACDEFG', 'kind': 'local_core', 'split': 'train', 'domain_cluster50': 'old'},
        'B': {'sequence': 'WWWWWWWWWW', 'kind': 'local_core', 'split': 'val', 'domain_cluster50': 'old'},
        'C': {'sequence': 'MMACDEFGYYY', 'kind': 'local_core', 'split': 'val', 'domain_cluster50': 'separate'},
        'D': {'sequence': 'HHHHH', 'kind': 'local_core', 'split': 'train', 'domain_cluster50': 'fourth'},
        'E': {'sequence': 'IIIII', 'kind': 'local_core', 'split': 'train', 'domain_cluster50': 'fifth'},
    }
    def fake_search(parameters, output, query, target, stem, target_count):
        result = output / f'{stem}.tsv'
        # C is longer: covering C alone must not replace the required D coverage.
        result.write_text('C\tD\t0.5\t0.9\t0.6\t1\t10\t1\t3\t11\t5\t10\t1e-5\t30\n'
                          'D\tE\t0.8\t0.8\t0.8\t1\t4\t1\t4\t5\t5\t4\t1e-5\t30\n')
        return result, {'columns': ALIGNMENT_FIELDS.split(',')}
    monkeypatch.setattr('natvqvae.fragment_overlap._run_search', fake_search)
    groups, summary = _auxiliary_groups(auxiliary, {'query_coverage': .8}, tmp_path)
    assert groups.find('A') == groups.find('B') == groups.find('C')
    assert groups.find('D') == groups.find('E') != groups.find('A')
    assert summary['exact_containment_pairs'] == 1
    assert summary['exact_cross_original_split_pairs'] == 1
    assert summary['short_coverage_filtered_rows'] == 1 and summary['similarity_pairs'] == 1
    assert (tmp_path / 'auxiliary_similarity.tsv.gz').exists()
    assert not (tmp_path / 'auxiliary_similarity.tsv').exists()


def test_retained_pool_never_turns_historical_split_or_weight_into_new_training_split(tmp_path):
    auxiliary = {uid: {'sequence': seq, 'kind': 'local_core', 'split': split,
                      'domain_cluster50': uid, 'weight': weight}
                 for uid, seq, split, weight in [('A', 'ACDE', 'train', '2'), ('B', 'FGHI', 'val', '3')]}
    groups = _Groups(auxiliary)
    groups.add('A', 'B', 'local_similarity', 'pairs.tsv.gz', 1)
    membership = _group_membership(auxiliary, groups, [])
    masks = tmp_path / 'input_masks.json'
    masks.write_text(json.dumps({uid: {'pairs': [[0, 4], [1, 5]]} for uid in auxiliary}))
    config = {'auxiliary': {'masks': str(masks)}, 'auxiliary_relations': {'wt_domain_length': 178}}
    summary = _retained_pool(config, tmp_path, auxiliary, membership, [])
    assert summary['groups'] == 1 and summary['original_domain_cluster50_groups'] == 2
    assert summary['original_weight_sum'] == 5
    assert summary['original_weight_sum_by_split'] == {'train': 2, 'val': 3}
    assert (tmp_path / 'retained_aux_domains.fasta').exists()
    assert not (tmp_path / 'aux_split.csv').exists()
    saved = list(csv.DictReader((tmp_path / 'retained_auxiliary.csv').open()))
    assert {row['original_split'] for row in saved} == {'train', 'val'}
    assert all('split' not in row and 'weight' not in row for row in saved)


def test_missing_targets_block_retained_pool_export(tmp_path):
    auxiliary = {'A': {'sequence': 'ACDE', 'kind': 'local_core', 'split': 'train'}}
    membership = _group_membership(auxiliary, _Groups(auxiliary), [])
    masks = tmp_path / 'input_masks.json'
    masks.write_text(json.dumps({'A': {'pairs': [[0, 4], [1, 5]]}}))
    config = {'auxiliary': {'masks': str(masks)}, 'auxiliary_relations': {'wt_domain_length': 178}}
    summary = _retained_pool(config, tmp_path, auxiliary, membership, [{'role': 'test'}])
    assert summary['export_status'] == 'not_exported_missing_natv_targets'
    assert not (tmp_path / 'retained_aux_domains.fasta').exists()


def test_relations_require_both_auxiliary_splits_and_similarity_search(tmp_path):
    config, _, _, _, _ = _fixture(tmp_path)
    config['auxiliary_relations'] = {'enabled': True}
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='both original train/val'):
        check_fragment_overlap(path)
