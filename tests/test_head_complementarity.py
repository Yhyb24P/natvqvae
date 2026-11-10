import numpy as np
import pytest

from natvqvae.head_complementarity import component_contributions, site_metrics


def _distribution(first):
    values = np.full(20, (1 - sum(first)) / (20 - len(first)))
    values[:len(first)] = first
    return values


def test_jensen_identity_ranking_and_unchanged_top1():
    esm = np.array([_distribution([.8, .001]), _distribution([.4, .5]), np.full(20, .05)])
    task = np.array([_distribution([.6, .2]), _distribution([.6, .2]), np.full(20, .05)])
    labels = np.array([1, 0, 3])
    alpha = np.array([.5, .3, .5])
    values, agreement, correct, _, _ = site_metrics(np.log(esm), np.log(task), labels, alpha)
    np.testing.assert_allclose(values[:, 4], values[:, 6] - values[:, 7], atol=1e-12)
    assert (values[:, 7] >= -1e-12).all()
    assert agreement.tolist() == [True, False, True]
    assert correct[0][0] == correct[2][0] == False
    assert values[0, 4] < -4  # Large score gain while both heads and mixture remain wrong at top1.
    assert values[2, 12:15].tolist() == [4, 4, 4]  # Stable AA-order tie break.
    scaled, _, _, _, _ = site_metrics(np.log(esm), np.log(task), labels, alpha, 2, 1.25)
    np.testing.assert_array_equal(scaled[:, 12:14], values[:, 12:14])
    with pytest.raises(ValueError, match='normalized'):
        site_metrics(np.log(esm) + 1, np.log(task), labels, alpha)


def test_partition_contributions_keep_original_denominators():
    rows = [{'component': 'a', 'region': 'NMNAT', 'weight': 1},
            {'component': 'a', 'region': 'Nudix', 'weight': 3},
            {'component': 'b', 'region': 'NMNAT', 'weight': 1},
            {'component': 'b', 'region': 'Nudix', 'weight': 3}]
    boundaries = np.array([0, 1, 4, 5, 6])
    metrics = np.column_stack([np.ones(6), [1, 2, 2, 2, 4, 8]])
    selected = np.array([True, False, True, False, False, False])
    groups, total, _ = component_contributions(metrics, rows, boundaries, np.ones(6, bool), 'all_panels')
    _, first, counts = component_contributions(metrics, rows, boundaries, selected, 'all_panels')
    _, second, _ = component_contributions(metrics, rows, boundaries, ~selected, 'all_panels')
    assert groups == ['a', 'b'] and counts.tolist() == [2, 0]
    np.testing.assert_allclose(total, [[1, 1.75], [1, 7]])
    np.testing.assert_allclose(first, [[.5, .75], [0, 0]])
    np.testing.assert_allclose(first + second, total)
    _, nmnat, _ = component_contributions(metrics, rows, boundaries, np.ones(6, bool), 'NMNAT')
    np.testing.assert_allclose(nmnat, [[1, 1], [1, 4]])
