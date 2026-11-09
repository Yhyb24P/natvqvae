import pytest

from natvqvae.split_relations import reconstruct_components, nearest_cross_split_hits


def test_component_reconstruction_keeps_excluded_bridge_and_stable_labels():
    # b is later excluded from fitting, but still connects a and c in the original graph.
    edges = [('c', 'b'), ('b', 'a'), ('a', 'a')]
    expected = {'a': 'a', 'b': 'a', 'c': 'a', 'd': 'd'}
    assert reconstruct_components('abcd', edges) == expected
    assert reconstruct_components('dcba', reversed(edges)) == expected


def test_component_reconstruction_rejects_unknown_endpoint():
    with pytest.raises(ValueError, match='outside'):
        reconstruct_components('ab', [('a', 'c')])


def test_neighbor_selection_distinguishes_containment_and_both_coverage():
    roles = {'query': 'test', 'same': 'test', 'long': 'training', 'full': 'training'}
    base = {'query': 'query', 'qlen': '100', 'bits': '100'}
    hits = [dict(base, target='same', tlen='100', fident='1', qcov='1', tcov='1'),
            dict(base, target='long', tlen='200', fident='1', qcov='1', tcov='0.5'),
            dict(base, target='full', tlen='100', fident='0.6', qcov='0.9', tcov='0.9')]
    assert nearest_cross_split_hits(hits, roles, 'both')[('query', 'training')]['target'] == 'full'
    assert nearest_cross_split_hits(hits, roles, 'shorter')[('query', 'training')]['target'] == 'long'


def test_shorter_coverage_uses_length_not_larger_coverage():
    hit = dict(query='q', target='t', qlen='50', tlen='100', qcov='0.7', tcov='0.9',
               fident='0.9', bits='100')
    assert nearest_cross_split_hits([hit], {'q': 'test', 't': 'training'}, 'shorter') == {}
    hit.update(qlen='100', qcov='0.9', tcov='0.5')
    assert nearest_cross_split_hits([hit], {'q': 'test', 't': 'training'}, 'shorter') == {}


def test_sequence_neighbor_run_records_unmatched_queries_and_refuses_overwrite(tmp_path, monkeypatch):
    import csv
    import hashlib
    import json
    import yaml
    from natvqvae import fragment_overlap
    from natvqvae.split_relations import analyze_sequence_neighbors

    sequences = ['ACDEFGHIKLMN', 'ACDEFGHIKLMP']
    ids = [hashlib.sha256(sequence.encode()).hexdigest() for sequence in sequences]
    fasta = tmp_path / 'population.fasta'
    fasta.write_text(''.join(f'>{uid}\n{sequence}\n' for uid, sequence in zip(ids, sequences)))
    splits = {}
    for role, uid in zip(('training', 'test'), ids):
        path = tmp_path / f'{role}.csv'
        row = dict(sequence_id=uid, component_v2=uid, nmnat_cdd_from=1,
                   nmnat_cdd_to=6, nudix_cdd_from=7, nudix_cdd_to=12)
        with path.open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=row)
            writer.writeheader()
            writer.writerow(row)
        splits[role] = str(path)

    def search(parameters, output, query, target, stem, count):
        assert count == 2 and query == target
        result = output / f'{stem}.tsv'
        fields = fragment_overlap.ALIGNMENT_FIELDS.split(',')
        row = dict(query=ids[1], target=ids[0], fident='0.75', qcov='1', tcov='1',
                   qstart='1', qend='6', tstart='1', tend='6', qlen='6', tlen='6',
                   alnlen='6', evalue='0.0001', bits='20')
        with result.open('w') as handle:
            csv.DictWriter(handle, fieldnames=fields, delimiter='\t').writerow(row)
        return result, {'columns': fields, 'version': 'mock'}

    monkeypatch.setattr(fragment_overlap, '_run_search', search)
    config = tmp_path / 'config.yaml'
    output = tmp_path / 'result'
    config.write_text(yaml.safe_dump(dict(fastas=[str(fasta)], splits=splits,
                                        mmseqs={}, output=str(output))))
    result = analyze_sequence_neighbors(config)
    for region in ('full_length', 'nmnat', 'nudix'):
        counts = result['regions'][region]['both']
        assert counts['test_to_training']['identity_ge_70pct'] == 1
        assert counts['training_to_test']['with_qualifying_hit'] == 0
        assert counts['training_to_test']['maximum_identity'] is None
    identity = json.loads((output / 'identity.json').read_text())
    assert str(fasta) in identity['input_sha256']
    with pytest.raises(FileExistsError):
        analyze_sequence_neighbors(config)
    monkeypatch.setattr(fragment_overlap, '_run_search', lambda *args: pytest.fail('Search must not rerun'))
    assert analyze_sequence_neighbors(config, reuse_alignments=True) == result
    splits_path = tmp_path / 'test.csv'
    splits_path.write_text(splits_path.read_text() + '\n')
    with pytest.raises(ValueError, match='inputs differ'):
        analyze_sequence_neighbors(config, reuse_alignments=True)


def test_containment_events_reports_both_directions_and_offsets():
    from natvqvae.split_relations import containment_events

    events, truncated = containment_events({'a': 'ACDE'}, {'b': 'XXACDEYY'})
    assert not truncated and len(events) == 1
    assert events[0]['direction'] == 'query_in_reference'
    assert events[0]['offset_0based'] == 2
    assert events[0]['shorter_fraction_of_longer'] == 0.5
    reverse, _ = containment_events({'long': 'XXACDEYY'}, {'short': 'ACDE'})
    assert reverse[0]['direction'] == 'reference_in_query'
    repeated, truncated = containment_events({'a': 'AC'}, {'b': 'ACAC'}, max_events=1)
    assert truncated and len(repeated) == 1
