import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from natvqvae.esm_baseline import (
    _checked_query_rows, _compare_components, _metric_components, score_sites, summarize_rows,
)
from natvqvae.model import AA, FrozenEncoder


def test_esm_head_uses_masked_features_and_twenty_amino_acids():
    class FakeEncoder(FrozenEncoder):
        def __init__(self):
            self.device = torch.device('cpu')
            self.tokenizer = SimpleNamespace(
                convert_tokens_to_ids=lambda values: list(range(3, 23)),
                convert_ids_to_tokens=lambda values: list(AA))
            self.model = SimpleNamespace(lm_head=lambda h: h)

        def __call__(self, sequences, positions):
            assert sequences == ['AC'] and positions == [[1]]
            logits = torch.zeros(1, 2, 33)
            logits[..., :3] = 1000  # Non-amino-acid tokens must not absorb mass.
            return logits, torch.tensor([[True, True]]), torch.tensor([[False, True]])

    logp, _, mask = FakeEncoder().amino_acid_log_probs(['AC'], [[1]])
    assert mask.tolist() == [[False, True]]
    assert logp.shape == (1, 2, 20)
    assert torch.allclose(logp.exp().sum(-1), torch.ones(1, 2))
    assert torch.allclose(score_sites(logp[0], 'AC', [1]), torch.tensor([-np.log(20)], dtype=torch.float32))


def test_target_scoring_only_uses_requested_residue_labels():
    logp = torch.full((3, 20), -9.)
    logp[0, AA.index('A')] = -1.
    logp[2, AA.index('D')] = -3.
    assert score_sites(logp, 'ACD', [2, 0]).tolist() == [-3., -1.]
    with pytest.raises(ValueError):
        score_sites(logp, 'ACD', [1, 1])
    with pytest.raises(ValueError):
        score_sites(logp, 'ACD', [3])


def test_group_average_does_not_count_more_queries_as_more_groups():
    rows = [{'component': 'large', 'region': 'NMNAT', 'weight': 1.,
             'factorized_nll_per_residue': 1.} for _ in range(8)]
    rows.append({'component': 'small', 'region': 'NMNAT', 'weight': 1.,
                 'factorized_nll_per_residue': 3.})
    result = summarize_rows(rows)
    assert result['mean_nll_per_residue']['NMNAT'] == 2.
    assert result['mean_nll_per_residue']['all_panels'] == 2.


def _comparison_task():
    return {('seq1', 'NMNAT_8_0'): {
        'sequence_id': 'seq1', 'component': 'group1', 'panel': 'NMNAT_8_0',
        'region': 'NMNAT', 'positions_0based': [1, 3], 'weight': .25}}


def _model_rows():
    return [{'sequence_id': 'seq1', 'component': 'group1', 'panel': 'NMNAT_8_0',
             'sites': 2, 'weight': .125, 'repeat': repeat,
             'prior_joint_nll_per_residue': float(repeat + 1),
             'marginal_nll_per_residue': float(repeat + 2)} for repeat in range(4)]


def _write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def test_comparison_averages_mc_scores_not_probabilities(tmp_path):
    path = tmp_path / 'model.jsonl'
    _write_jsonl(path, _model_rows())
    rows, checks = _checked_query_rows(path, _comparison_task(), {'group1': 2}, repeats=4)
    assert rows[0]['prior_joint_nll_per_residue'] == 2.5
    assert rows[0]['marginal_nll_per_residue'] == 3.5
    assert rows[0]['weight'] == .25  # Group-size scaling was checked, not retained twice.
    assert checks == {'queries': 1, 'recorded_rows': 4, 'rows_with_stored_positions': 0,
                      'repeats_per_query': 4}


@pytest.mark.parametrize('change,match', [
    ({'component': 'wrong_group'}, 'group or site count'),
    ({'sites': 3}, 'group or site count'),
    ({'positions_0based': [1, 4]}, 'Masked positions'),
    ({'weight': .25}, 'weight differs'),
    ({'sequence_id': 'another_seq'}, 'Unexpected query'),
])
def test_comparison_rejects_changed_task_identity(tmp_path, change, match):
    rows = _model_rows()
    rows[0].update(change)
    path = tmp_path / 'model.jsonl'
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match=match):
        _checked_query_rows(path, _comparison_task(), {'group1': 2}, repeats=4)


@pytest.mark.parametrize('kind,match', [('duplicate', 'Duplicate'), ('missing', 'Incomplete')])
def test_comparison_does_not_count_mc_repeats_as_new_queries(tmp_path, kind, match):
    rows = _model_rows()
    rows = rows + [rows[0]] if kind == 'duplicate' else rows[:-1]
    path = tmp_path / 'model.jsonl'
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match=match):
        _checked_query_rows(path, _comparison_task(), {'group1': 2}, repeats=4)


def test_comparison_checks_recorded_esm_positions_and_site_scores(tmp_path):
    row = {**next(iter(_comparison_task().values())), 'sites': 2,
           'target_site_log_probs': [-1., -3.], 'factorized_nll_per_residue': 2.}
    path = tmp_path / 'esm.jsonl'
    _write_jsonl(path, [row])
    rows, checks = _checked_query_rows(path, _comparison_task(), {'group1': 2})
    assert rows[0]['factorized_nll_per_residue'] == 2.
    assert checks['rows_with_stored_positions'] == 1
    row['factorized_nll_per_residue'] = 1.
    _write_jsonl(path, [row])
    with pytest.raises(ValueError, match='target probabilities'):
        _checked_query_rows(path, _comparison_task(), {'group1': 2})


def test_comparison_preserves_panel_weights_but_equalizes_groups():
    rows = [
        {'component': 'large', 'region': 'NMNAT', 'weight': 1., 'score': 1.},
        {'component': 'large', 'region': 'NMNAT', 'weight': 1., 'score': 3.},
        {'component': 'large', 'region': 'Nudix', 'weight': 3., 'score': 4.},
        {'component': 'large', 'region': 'Nudix', 'weight': 3., 'score': 6.},
        {'component': 'small', 'region': 'NMNAT', 'weight': 1., 'score': 4.},
        {'component': 'small', 'region': 'Nudix', 'weight': 3., 'score': 4.},
    ]
    model = _metric_components(rows, 'score')
    assert model['large'] == {'NMNAT': 2., 'Nudix': 5., 'all_panels': 4.25}
    esm = {'large': {'all_panels': 3.25}, 'small': {'all_panels': 5.}}
    comparison = _compare_components(model, esm, 'all_panels')
    assert comparison['groups'] == 2
    assert comparison['mean'] == 0.
    assert comparison['model_mean'] == 4.125
    assert comparison['groups_model_lower'] == comparison['groups_model_higher'] == 1
