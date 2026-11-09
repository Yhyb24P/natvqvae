import json

import numpy as np
import pytest

from natvqvae.fragment_comparison import compare_fragment_transfer
from natvqvae.latent_precision import (_difference_summary, _pool_query_repeats,
                                      _repeat_scores, summarize_latent_precision)
from test_fragment_comparison import _fixture


def test_precision_preserves_paired_repeats_and_equal_component_weights():
    values = np.array([[1., 9.], [3., 11.], [5., 13.], [7., 15.]])
    result = _difference_summary(values, 2000, 7)
    assert result['mean'] == 8
    assert result['repeat_estimates'] == [5, 7, 9, 11]
    assert result['mc_standard_error'] == pytest.approx(np.sqrt(20 / 3) / 2)
    assert result['group_bootstrap_ci95'] == [4, 12]
    assert _difference_summary(values[:, :1], 20, 7)['group_bootstrap_ci95'] is None
    noise = np.array([10., -10., 5., -5.])[:, None]
    left, right = noise + np.array([1., 3.]), noise + np.array([2., 4.])
    assert _difference_summary(left - right, 20, 7)['mc_standard_error'] == 0
    # Shared random streams across pipeline seeds must not be treated as independent noise.
    averaged_seeds = np.mean([values, values + 2, values + 4], axis=0)
    assert _difference_summary(averaged_seeds, 20, 7)['mc_standard_error'] == result['mc_standard_error']
    assert _difference_summary([[1, 2]], 20, 7)['mc_standard_error'] is None
    with pytest.raises(ValueError, match='finite paired'):
        _difference_summary([[]], 20, 7)


def test_pooling_reconstructs_concatenated_samples_not_mean_nll():
    probabilities = np.array([[.9, .9], [.1, .1], [.5, .8], [.7, .2]])
    rows = [{'sites': 2, 'prior_joint_nll_per_residue': -np.log(batch.prod(axis=1).mean()) / 2,
             'marginal_nll_per_residue': -np.log(batch.mean(axis=0)).mean(),
             'prior_site_marginal_log_probs': np.log(batch.mean(axis=0)).tolist()}
            for batch in np.split(probabilities, 2)]
    pooled = _pool_query_repeats(rows)
    assert pooled['prior_joint_nll_per_residue'] == pytest.approx(-np.log(probabilities.prod(axis=1).mean()) / 2)
    assert pooled['marginal_nll_per_residue'] == pytest.approx(-np.log(probabilities.mean(axis=0)).mean())
    assert pooled['prior_joint_nll_per_residue'] != pytest.approx(np.mean([row['prior_joint_nll_per_residue'] for row in rows]))
    rows[0]['prior_site_marginal_log_probs'][0] -= 1
    with pytest.raises(ValueError, match='site marginals'):
        _pool_query_repeats(rows)


@pytest.mark.parametrize('split', ['validation', 'test'])
def test_precision_reuses_verified_comparison_and_rejects_changed_inputs(tmp_path, monkeypatch, split):
    config, evaluated, esm, comparison, recovered, _ = _fixture(tmp_path, monkeypatch, split)
    compare_fragment_transfer(config, evaluated, esm, comparison, split, recovered)
    output = tmp_path / 'precision'
    result = summarize_latent_precision(comparison, output)
    estimates = {row['comparison']: row for row in result['estimates']
                 if row['scope'] == 'mean_of_pipeline_seeds' and row['region'] == 'NMNAT'}
    assert estimates['external_minus_self:prior_joint_nll_per_residue']['mean'] == pytest.approx(-.2)
    assert estimates['external:joint_minus_site_marginal']['mean'] == pytest.approx(-.01)
    assert all(row['mc_standard_error'] < 1e-14 for row in estimates.values())
    assert estimates['external_minus_self:prior_joint_nll_per_residue']['group_bootstrap_ci95'] == pytest.approx([-1., .6])
    assert result['data']['groups'] == 2 and result['split'] == split
    assert len(result['estimates']) == 5 * 6 * 4
    assert {row['samples_per_estimate'] for row in result['sample_count_sensitivity']} == {4, 8}
    assert all(row['mc_standard_error'] is None for row in result['sample_count_sensitivity']
               if row['samples_per_estimate'] == 8)
    assert (output / 'group_repeat_differences.csv').exists()
    with pytest.raises(FileExistsError):
        summarize_latent_precision(comparison, output)
    path = evaluated / 'fragment_random_seed11/per_query.jsonl'
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='input SHA256 differs'):
        summarize_latent_precision(comparison, tmp_path / 'changed')
    assert not (tmp_path / 'changed').exists()


@pytest.mark.parametrize('fault,match', [('missing', 'Incomplete'), ('duplicate', 'Duplicate'),
                                       ('nonfinite', 'Invalid query'), ('order', 'order differs')])
def test_repeat_reader_rejects_unpaired_tasks(tmp_path, fault, match):
    rows = [{'sequence_id': uid, 'component': 'g', 'panel': 'NMNAT_8_count',
             'positions_0based': [0], 'sites': 1, 'weight': 1., 'repeat': repeat,
             'prior_joint_nll_per_residue': 1., 'marginal_nll_per_residue': 1.1}
            for repeat in range(2) for uid in ('a', 'b')]
    if fault == 'missing':
        rows.pop()
    elif fault == 'duplicate':
        rows.append(rows[0])
    elif fault == 'nonfinite':
        rows[0]['prior_joint_nll_per_residue'] = float('nan')
    else:
        rows[-2:] = reversed(rows[-2:])
    path = tmp_path / 'queries.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match=match):
        _repeat_scores(path, 2)
