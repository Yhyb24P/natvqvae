import json
import random

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')
from natvqvae.conditional_generation import AA, ConditionalModel
from natvqvae.conditional_study import (budget_steps, checkpoint_payload, expected_fit_count,
    fit_model, initial_selection_state, learning_rate_at_epoch, load_checkpoint, rng_state,
    set_rng_state, update_selection, _run_gradient_diagnostics, _torch_save)


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
        rng = np.random.default_rng(9)
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
        panels = [f'p{index:02d}' for index in range(27)]
        for row in (2, 3):
            for panel in panels:
                self.validation.append({'row': row, 'offset': row * 6, 'length': 6,
                                        'positions': [1, 2], 'panel': panel,
                                        'weight': 1. / 27})

    def training_queries(self, epoch, seed):
        rng = np.random.default_rng(seed + epoch)
        values = [dict(self.training_by_row[row]) for row in sorted(self.training_by_row)]
        return [values[i] for i in rng.permutation(len(values))]

    def batch(self, queries, device='cpu', need_full=True):
        width = max(query['length'] for query in queries)
        n = len(queries)
        context = torch.zeros(n, width, self.feature_dim, dtype=torch.bfloat16)
        full = torch.zeros_like(context) if need_full else None
        valid = torch.zeros(n, width, dtype=torch.bool)
        mask = torch.zeros_like(valid)
        labels = torch.zeros(n, width, dtype=torch.long)
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


def toy_plan(arm='deterministic'):
    return {
        'task': 'NatV_conditional_categorical_joint_generation',
        'model': {'hidden': 8, 'factors': 2, 'states': 3},
        'arms': [arm],
        'batch_size': 1,
        'beta_warmup_epochs': 1,
        'learning_rate_warmup_epochs': 1,
        'lr_patience': 8,
        'stop_patience': 30,
        'gradient_clip': 1.0,
        'calibration': {'rates': [1e-4], 'seed': 1, 'max_steps': 2, 'default_K': 2,
                        'K_candidates': [2], 'gradient_repetitions': 2,
                        'gradient_projections': 4, 'signal_SNR_min': 2.0,
                        'efficiency_tie_fraction': .2, 'likelihood_counts': [2, 4],
                        'likelihood_repetitions': 2,
                        'likelihood_tolerance_per_residue': .01},
        'search_rate_multipliers': [1.0],
        'search_weight_decay': [1e-4],
        'selection_seeds': [2],
        'formal_seeds': [3],
        'budget': {'max_fits': 38, 'calibration_steps_per_fit': 5200,
                   'absolute_steps_per_remaining_fit_max': 20800},
    }


def test_run_study_calibration_only(tmp_path):
    from natvqvae.conditional_study import run_study

    torch.set_num_threads(1)
    data = TinyData()
    plan = toy_plan('deterministic')
    plan.update({'training_population': 2, 'validation_population': 2})
    plan['budget'] = {'max_fits': 38, 'calibration_steps_per_fit': 2,
                      'absolute_steps_per_remaining_fit_max': 4}
    plan['calibration']['max_steps'] = 2
    plan['save_validation_records'] = False
    run = tmp_path / 'experiments' / 'toy'
    run.mkdir(parents=True)
    (run / 'plan.json').write_text(json.dumps(plan))
    manifest = run_study(run, plan, data=data, root=tmp_path, device='cpu',
                         calibrate_only=True)
    assert manifest['status'] == 'calibration_complete'
    assert manifest['attempts'] == 1
    assert manifest['decision']['sha256']
    assert (run / 'calibration_decision.json').exists()
    assert json.loads((run / 'status.json').read_text())['phase'] == 'CALIBRATION_READY'


def test_run_study_tiny_end_to_end(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from natvqvae.conditional_study import run_study

    torch.set_num_threads(1)
    data = TinyData()
    plan = toy_plan('deterministic')
    plan.update({'training_population': 2, 'validation_population': 2})
    plan['budget'] = {'max_fits': 38, 'calibration_steps_per_fit': 2,
                      'absolute_steps_per_remaining_fit_max': 4}
    plan['calibration']['max_steps'] = 2
    plan['save_validation_records'] = True
    run = tmp_path / 'experiments' / 'toy'
    run.mkdir(parents=True)
    (run / 'plan.json').write_text(json.dumps(plan))
    monkeypatch.setitem(sys.modules, 'natvqvae.conditional_outputs',
                        SimpleNamespace(finish_study=lambda *args: {'ok': True}))
    manifest = run_study(run, plan, data=data, root=tmp_path, device='cpu')
    assert manifest['status'] == 'completed'
    for name in ('calibration_decision', 'search_selection', 'formal_manifest',
                 'fit_status', 'cost', 'training_history', 'calibration_interface'):
        assert manifest['artifacts'][name]['sha256']
        assert (tmp_path / manifest['artifacts'][name]['path']).exists()
    interface = json.loads((run / 'calibration_decision.json').read_text())
    assert interface['arms']['deterministic']['max_steps_per_fit'] == 4
    assert interface['arms']['deterministic']['evaluation_sample_count'] == 0
    assert interface['full_decision']['sha256']
    cost = json.loads((tmp_path / manifest['artifacts']['cost']['path']).read_text())
    formal = json.loads((tmp_path / manifest['artifacts']['formal_manifest']['path']).read_text())
    assert cost['attempted_fits'] == 3
    assert set(formal) == {'deterministic/3'}
    assert (tmp_path / 'data/results/toy/raw/calibration_decision.json').exists()
    assert (tmp_path / 'data/results/toy/raw/formal_manifest.json').exists()
    assert (tmp_path / 'data/results/toy/raw/study_manifest.json').exists()
    assert json.loads((run / 'status.json').read_text())['phase'] == 'COMPLETED'


def test_budget_and_fit_count_and_schedules():
    plan = toy_plan()
    assert budget_steps(plan, 100) == 5200
    assert budget_steps(plan, 2000) == 8000
    assert budget_steps(plan, 6000) == 20800
    plan['arms'] = ['QVAE', 'deterministic']
    plan['calibration']['rates'] = [1e-4, 3e-4]
    plan['search_rate_multipliers'] = [.5, 1., 2.]
    plan['search_weight_decay'] = [1e-4, 1e-2]
    plan['selection_seeds'] = [8201, 8202]
    plan['formal_seeds'] = [8301, 8302, 8303, 8304, 8305]
    assert expected_fit_count(plan) == (4, 24, 10, 38)
    assert learning_rate_at_epoch(1e-4, 1, 5, .5) == pytest.approx(1e-5)
    assert learning_rate_at_epoch(1e-4, 5, 5, .5) == pytest.approx(5e-5)


def test_selection_tolerance_lr_halving_and_early_stopping():
    state = initial_selection_state()
    state, improved = update_selection(state, 1.0, 10, 1, .01, True)
    assert improved and state['best'] == 1.0
    state, improved = update_selection(state, 1.005, 20, 2, .01, True)
    assert not improved and state['best'] == 1.0 and state['since_improve'] == 1
    state = initial_selection_state()
    state, _ = update_selection(state, 1.0, 10, 1, .01, True)
    for epoch in range(2, 18):
        state, improved = update_selection(state, 1.0, epoch * 10, epoch, .01, True)
        assert not improved
    assert state['lr_factor'] == pytest.approx(.25)
    assert state['lr_reductions_since_improve'] == 2
    for epoch in range(18, 32):
        state, _ = update_selection(state, 1.0, epoch * 10, epoch, .01, True)
    assert state['stop_reason'] == 'early_stopping'
    state, improved = update_selection(state, .5, 999, 32, .01, True)
    assert improved and state['best'] == .5 and state['since_improve'] == 0


def test_checkpoint_rng_roundtrip(tmp_path):
    model = ConditionalModel('deterministic', hidden=8, factors=2, states=3, feature_dim=6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    random.seed(3); np.random.seed(3); torch.manual_seed(3)
    payload = checkpoint_payload(model, optimizer, 'deterministic', 7,
                                 {'hidden': 8, 'factors': 2, 'states': 3, 'feature_dim': 6},
                                 5, 1, 1.25, {'learning_rate': 1e-3}, include_optimizer=True)
    expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    path = tmp_path / 'checkpoint.pt'
    _torch_save(path, payload)
    loaded = load_checkpoint(path)
    assert loaded['arm'] == 'deterministic' and loaded['seed'] == 7
    assert 'optimizer' in loaded and 'rng' in loaded
    set_rng_state(loaded['rng'])
    actual = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize('arm', ['deterministic', 'QVAE'])
def test_fit_model_smoke(tmp_path, arm):
    torch.set_num_threads(1)
    data = TinyData()
    plan = toy_plan(arm)
    run = tmp_path / 'experiments' / 'toy'
    run.mkdir(parents=True)
    config = {'learning_rate': 1e-3, 'weight_decay': 1e-4, 'K': 2}
    record = fit_model(run, plan, data, tmp_path, torch.device('cpu'),
                       f'formal/{arm}/seed1', 'formal', arm, 1, config, 2,
                       data.validation, 2, 1, 1e-4, save_records=False,
                       diagnostics=False)
    assert record['steps'] == 2 and record['best_score'] is not None
    assert record['best_path'] and record['last_path']
    assert (tmp_path / record['best_path']).exists()
    last = torch.load(tmp_path / record['last_path'], map_location='cpu', weights_only=True)
    assert last['arm'] == arm and last['seed'] == 1
    assert 'optimizer' in last and 'rng' in last
    assert record['best_sha256'] and record['last_sha256']


def test_gradient_diagnostics_tiny():
    torch.set_num_threads(1)
    data = TinyData()
    plan = toy_plan('QVAE')
    plan['calibration']['K_candidates'] = [2, 3]
    model = ConditionalModel('QVAE', hidden=8, factors=2, states=3, feature_dim=6)
    result = _run_gradient_diagnostics(model, data, data.training_queries(0, 1),
                                       plan, 1, 'init', 0, torch.device('cpu'))
    assert len(result['records']) == 2
    for record in result['records']:
        assert record['K'] in (2, 3)
        assert set(record['groups']) == {'posterior_output', 'posterior_encoder_last',
                                         'full_projection'}
        for metrics in record['groups'].values():
            for path in ('rloo', 'direct', 'kl'):
                assert 'snr' in metrics[path]
                assert len(metrics[path]['projections']) == 4
        assert record['time_per_repetition'] > 0
