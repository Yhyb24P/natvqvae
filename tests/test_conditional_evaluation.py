from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')
from natvqvae.conditional_evaluation import (aggregate_panel_records, compare_joint_scores,
    deterministic_joint_probability, marginal_probability, posterior_joint_probability,
    prior_joint_probability, target_aa_distribution)
from natvqvae.conditional_generation import ConditionalModel, target_tensors


class TinyData:
    def __init__(self, feature_dim=6):
        self.feature_dim = feature_dim
        self.features = np.zeros((1, feature_dim), dtype=np.float32)
        self.rows = pd.DataFrame([
            {'sequence_id': 'train_a', 'sequence': 'ACDEFG', 'length': 6,
             'training_role_v2': 'training', 'component_v2': 'c1'},
            {'sequence_id': 'train_b', 'sequence': 'GHIKLM', 'length': 6,
             'training_role_v2': 'training', 'component_v2': 'c2'},
            {'sequence_id': 'val_a', 'sequence': 'NQRSTV', 'length': 6,
             'training_role_v2': 'validation', 'component_v2': 'c1'},
            {'sequence_id': 'val_b', 'sequence': 'WYACDE', 'length': 6,
             'training_role_v2': 'validation', 'component_v2': 'c2'},
        ])
        rng = np.random.default_rng(7)
        self.values = {row: rng.standard_normal((6, feature_dim)).astype(np.float32)
                       for row in range(len(self.rows))}
        self.weights = {0: 1., 1: 1.}
        self.training_by_row = {
            0: {'row': 0, 'offset': 0, 'length': 6, 'positions': [1, 2],
                'panel': 'p1', 'weight': .4},
            1: {'row': 1, 'offset': 6, 'length': 6, 'positions': [3, 4],
                'panel': 'p2', 'weight': .6},
        }
        self.validation = []
        for row, panel, positions, weight in [(2, 'p1', [1, 2], .4),
                                              (3, 'p1', [1, 2], .4),
                                              (2, 'p2', [3, 4], .6),
                                              (3, 'p2', [3, 4], .6)]:
            self.validation.append({'row': row, 'offset': row * 6, 'length': 6,
                                    'positions': positions, 'panel': panel,
                                    'weight': weight})
        self.validation.sort(key=lambda q: (q['panel'], q['row']))

    def training_queries(self, epoch, seed):
        rng = np.random.default_rng(seed + epoch)
        values = [dict(self.training_by_row[row]) for row in sorted(self.training_by_row)]
        return [values[i] for i in rng.permutation(len(values))]

    def batch(self, queries, device='cpu', need_full=True):
        width = max(q['length'] for q in queries)
        n = len(queries)
        context = torch.zeros(n, width, self.feature_dim, dtype=torch.bfloat16)
        full = torch.zeros_like(context) if need_full else None
        valid = torch.zeros(n, width, dtype=torch.bool)
        mask = torch.zeros_like(valid)
        labels = torch.zeros(n, width, dtype=torch.long)
        from natvqvae.conditional_generation import AA
        for i, query in enumerate(queries):
            row = self.rows.iloc[query['row']]
            values = torch.from_numpy(self.values[query['row']])
            context[i, :query['length']] = values.to(torch.bfloat16)
            context[i, query['positions']] = .5
            if need_full:
                full[i, :query['length']] = values.to(torch.bfloat16)
            valid[i, :query['length']] = True
            mask[i, query['positions']] = True
            labels[i, :query['length']] = torch.tensor([AA.index(letter) for letter in row.sequence])
        return context.to(device), (full.to(device) if full is not None else None), \
            valid.to(device), mask.to(device), labels.to(device)


def tiny_model(arm, feature_dim=6):
    return ConditionalModel(arm, hidden=8, factors=2, states=3, feature_dim=feature_dim)


def test_deterministic_exact_joint_matches_manual_and_subset():
    data = TinyData()
    model = tiny_model('deterministic')
    query = dict(data.validation[0])
    query['positions'] = [0, 1, 2, 3]
    query['subset_positions'] = [1, 2]
    result = deterministic_joint_probability(model, data, [query])
    record = result['records'][0]
    assert record['sites'] == 2 and record['context_sites'] == 4
    context, _, valid, mask, labels = data.batch([query], need_full=False)
    hidden = model.encode_context(context, valid, mask)
    score_mask = torch.zeros_like(mask)
    score_mask[0, query['subset_positions']] = True
    targets, y, use, _ = target_tensors(hidden, score_mask, labels)
    latent = model.deterministic_context(hidden, valid, mask)
    logits = model.decode(targets, latent.unsqueeze(0))[0]
    manual = (logits.float().log_softmax(-1).gather(-1, y[..., None]).squeeze(-1) * use).sum()
    assert record['log_prob'][0] == pytest.approx(float(manual.detach()), abs=1e-5)
    assert record['nats_per_residue'][0] == pytest.approx(-float(manual.detach()) / 2, abs=1e-5)


def test_posterior_prior_marginal_and_paired_comparison():
    data = TinyData()
    torch.manual_seed(11)
    qvae = tiny_model('QVAE')
    deterministic = tiny_model('deterministic')
    queries = data.validation[:2]
    posterior = posterior_joint_probability(qvae, data, queries, sample_count=8, repeats=2,
                                           seed=3, batch_size=2, sample_chunk=3)
    prior = prior_joint_probability(qvae, data, queries, sample_count=8, repeats=2,
                                    seed=3, batch_size=2, sample_chunk=3)
    marginal = marginal_probability(qvae, data, queries, sample_count=8, repeats=2,
                                    seed=3, batch_size=2, sample_chunk=3)
    for result in (posterior, prior, marginal):
        assert len(result['records']) == 2
        assert result['summary']['finite']
        assert abs(sum(result['summary']['panel_weights'].values()) - 1) < 1e-9
        for record in result['records']:
            assert len(record['nats_per_residue']) == 2
            assert all(value is not None for value in record['nats_per_residue'])
            assert all(1 - 1e-6 <= value <= 8 + 1e-6 for value in record['ess'])
            assert all(0 <= value <= 1 + 1e-6 for value in record['max_normalized_weight'])
    comparison = compare_joint_scores(qvae, deterministic, data, queries, sample_count=8,
                                      repeats=2, seed=4, batch_size=2, sample_chunk=3)
    assert set(comparison['summary']) == {'shared_joint', 'marginal_product', 'deterministic',
                                         'marginal_minus_shared', 'deterministic_minus_shared',
                                         'marginal_minus_deterministic'}
    assert comparison['summary']['marginal_minus_shared']['finite']


def test_aggregate_panel_records_uses_panel_weights_and_components_not_equal_panels():
    data = TinyData()
    records = []
    for panel, weight, c1, c2 in [('p1', .25, 1., 3.), ('p2', .75, 5., 7.)]:
        for component, value in [('c1', c1), ('c2', c2)]:
            records.append({'row': 0, 'component': component, 'panel': panel,
                            'weight': weight, 'nats_per_residue': [value, value + 1],
                            'ess': [8., 8.], 'max_normalized_weight': [.2, .2],
                            'low_ess': False})
    summary = aggregate_panel_records(records, data)
    assert summary['score'] == pytest.approx(5.5)
    assert summary['panels']['p1']['score'] == pytest.approx(2.5)
    assert summary['panels']['p2']['score'] == pytest.approx(6.5)
    assert summary['components']['c1'] == pytest.approx(4.5)
    assert summary['components']['c2'] == pytest.approx(6.5)
    assert summary['se'] is not None
    assert summary['input_low_ess_fraction'] == 0
    records[0]['ess'] = [1.]
    records[0]['max_normalized_weight'] = [1.]
    records[0]['low_ess'] = True
    summary = aggregate_panel_records(records, data)
    assert summary['input_low_ess_fraction'] == pytest.approx(.25)
    assert not summary['input_mc_reliable']
    assert summary['score'] is not None


def test_target_aa_distribution_counts_frequencies_and_component_balance():
    data = TinyData()
    queries = [data.validation[0], data.validation[1]]
    result = target_aa_distribution(data, queries)
    assert result['positions'] == 4
    assert sum(result['counts']) == 4
    assert pytest.approx(1.) == sum(result['frequencies'])
    assert pytest.approx(1.) == sum(result['component_balanced_frequencies'])
    assert result['effective_positions'] > 0
    weighted = target_aa_distribution(data, queries, weights=[1., 3.])
    assert weighted['positions'] == 4
    assert pytest.approx(1.) == sum(weighted['frequencies'])
