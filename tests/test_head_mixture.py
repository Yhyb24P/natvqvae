import json

import numpy as np
import pytest
import yaml

from natvqvae.head_mixture import (REGIONS, _group_grid, _selected_queries, calibrate,
                                  mixture_nll, run_head_mixture)


def test_mixture_uses_probabilities_and_retains_mc_repeats():
    esm = np.log([.2, .8])
    task = np.log([[.8, .2], [.4, .6]])
    actual = mixture_nll(esm, task, [0, .5, 1])
    assert actual[0] == pytest.approx(-esm.mean())
    assert actual[-1] == pytest.approx(-task.mean())
    assert actual[1] == pytest.approx(-np.log((np.exp(esm) + np.exp(task)) / 2).mean())
    assert not np.isclose(actual[1], (actual[0] + actual[-1]) / 2)
    with pytest.raises(ValueError):
        mixture_nll(esm, [[0.1, -2]], [.5])
    with pytest.raises(ValueError):
        mixture_nll(esm, task, [1.2])


def test_calibration_excludes_whole_component_and_averages_seeds():
    grid = np.zeros((3, 2, 3, 6))
    grid[:, 0] = np.array([0, 5, 5])[None, :, None]
    grid[:, 1] = np.array([100, 0, 0])[None, :, None]
    full, excluded = calibrate(grid)
    assert (full == 0).all()
    assert (excluded[0] == 1).all()
    assert (excluded[1:] == 0).all()
    changed = grid.copy()
    changed[:, :, 0] += 1000
    assert np.array_equal(calibrate(changed)[1][0], excluded[0])


def _fixture():
    rows = [{'sequence_id': str(g), 'component': str(g), 'panel': region,
             'region': region, 'positions_0based': [0], 'weight': (2 if region == 'NMNAT' else 1)}
            for g in range(2) for region in REGIONS[:-1]]
    cube = np.ones((3, 3, len(rows)))
    cube[:, 1, :] = .8
    cube[:, 2, :] = 1.2
    neighbors = {r: {str(g): .6 for g in range(2)} for r in ('NMNAT', 'Nudix', 'full_length')}
    return rows, cube, neighbors


def test_region_and_global_weights_select_before_all_panel_aggregation():
    rows, cube, _ = _fixture()
    groups, _ = _group_grid(cube, rows)
    indices = np.zeros((2, len(REGIONS)), dtype=int)
    indices[:, 0] = 1
    indices[:, -1] = 2
    regional = _selected_queries(cube, rows, indices, groups)
    global_ = _selected_queries(cube, rows, indices, groups, True)
    assert (regional[:, [0, 5]] == .8).all()
    assert (global_ == 1.2).all()
    _, aggregated = _group_grid(regional[:, None, :], rows)
    assert aggregated[0, 0, 0, -1] == pytest.approx((2 * .8 + 4) / 6)


def test_fit_never_loads_test_and_evaluation_checks_frozen_protocol(monkeypatch, tmp_path):
    import natvqvae.head_mixture as module
    config = {'alpha_grid': [0, .5, 1], 'models': {'QVAE': {}, 'deterministic': {}},
              'seeds': [1, 2, 3], 'bootstrap_repeats': 5, 'bootstrap_seed': 1}
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    splits = []

    def load(config, split, hashes):
        splits.append(split)
        rows, cube, neighbors = _fixture()
        return rows, {'QVAE': cube, 'deterministic': cube}, neighbors, {'models': 'same'}

    monkeypatch.setattr(module, '_load_split', load)
    monkeypatch.setattr(module, '_validation_input_paths', lambda *args: {str(path)})
    fit = tmp_path / 'fit'
    run_head_mixture(path, fit)
    assert splits == ['validation']
    frozen = json.loads((fit / 'fitted_weights.json').read_text())
    assert frozen['weights']['QVAE']['full_alphas']['NMNAT'] == .5
    run_head_mixture(path, tmp_path / 'test', fit / 'fitted_weights.json')
    assert splits == ['validation', 'test']
    truncated = dict(frozen, input_sha256={})
    bad_fit = tmp_path / 'truncated.json'
    bad_fit.write_text(json.dumps(truncated))
    with pytest.raises(ValueError, match='incomplete'):
        run_head_mixture(path, tmp_path / 'incomplete', bad_fit)
    path.write_text(path.read_text() + '\n# changed protocol\n')
    with pytest.raises(ValueError, match='protocol'):
        run_head_mixture(path, tmp_path / 'bad_test', fit / 'fitted_weights.json')
    assert splits == ['validation', 'test']


def test_cli_separates_fit_from_heldout_evaluation():
    from natvqvae.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(['head-mixture-fit', '--output', 'data/results/evaluation/mixture'])
    assert not hasattr(args, 'fit')
    with pytest.raises(SystemExit):
        parser.parse_args(['head-mixture-evaluate', '--output', 'data/results/evaluation/mixture'])
