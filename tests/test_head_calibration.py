import numpy as np
import pytest
import json
import yaml

from natvqvae.head_mixture import REGIONS, calibrated_predictions, temperature_targets


def test_temperature_renormalizes_all_classes_and_self_mixture_is_invariant():
    p = np.array([[.81, *([.19 / 19] * 19)], [.05] * 20])
    values = temperature_targets(np.log(p), np.array([0, 1]), [.5, 1, 2])
    np.testing.assert_allclose(values[1], np.log([.81, .05]))
    assert values[0, 0] == pytest.approx(np.log(.81 ** 2 / (p[0] ** 2).sum()))
    with pytest.raises(ValueError, match='normalized'):
        temperature_targets(np.log(p) + 1, np.array([0, 1]), [1])
    rows = [{'component': str(g), 'region': r, 'weight': 1} for g in range(3) for r in REGIONS[:-1]]
    targets = np.full((4, 1, len(rows)), -.7)
    result, _ = calibrated_predictions(targets, np.arange(len(rows) + 1), rows, [1], [0, .5, 1])
    for scores in result.values():
        np.testing.assert_allclose(scores, .7)


def test_loco_temperature_and_alpha_both_exclude_held_component():
    rows = [{'component': str(g), 'region': r, 'weight': 1} for g in range(3) for r in REGIONS[:-1]]
    rng = np.random.default_rng(8)
    targets = -rng.uniform(.2, 3, (4, 3, len(rows)))
    boundaries = np.arange(len(rows) + 1)
    scores, fits = calibrated_predictions(targets, boundaries, rows, [.5, 1, 2], [0, .5, 1])
    changed = targets.copy()
    changed[:, :, :5] = -rng.uniform(4, 9, (4, 3, 5))
    _, changed_fits = calibrated_predictions(changed, boundaries, rows, [.5, 1, 2], [0, .5, 1])
    assert fits['0'] == changed_fits['0']
    applied, _ = calibrated_predictions(targets, boundaries, rows, [.5, 1, 2], [0, .5, 1], fits['0'])
    for name in scores:
        np.testing.assert_allclose(scores[name][:, :5], applied[name][:, :5])
    assert all(np.isfinite(value).all() for value in applied.values())


def test_calibration_runner_freezes_complete_inputs_before_test(monkeypatch, tmp_path):
    import natvqvae.head_mixture as module

    source = tmp_path / 'inputs.txt'
    source.write_text('frozen input')
    source_config = tmp_path / 'mixture.yaml'
    source_config.write_text('original config')
    config = tmp_path / 'calibration.yaml'
    config.write_text(yaml.safe_dump({'temperatures': [.5, 1, 2], 'mixture_config': str(source_config),
                                     'probabilities': {'validation': str(tmp_path)}}))
    training = tmp_path / 'random1'
    training.mkdir()
    training_path = training / 'training_summary.json'
    training_path.write_text(json.dumps({'checkpoints': {'best': {'path': str(source)}}}))
    identity = {'input_sha256': {str(source): module.sha256_file(source),
                                 str(training_path): module.sha256_file(training_path)},
                'implementation_sha256': {'export': 'frozen'}, 'precision': 'float32', 'definition': '20AA',
                'esm_files_sha256': {'model': 'same'}}
    (tmp_path / 'identity.json').write_text(json.dumps(identity))
    rows = [{'component': str(g), 'region': r, 'weight': 1} for g in range(3) for r in REGIONS[:-1]]
    targets = -np.random.default_rng(2).uniform(.1, 2, (4, 3, len(rows)))
    spec = {'seeds': [1], 'alpha_grid': [0, .5, 1], 'bootstrap_repeats': 10, 'bootstrap_seed': 1,
            'models': {'deterministic': {'training': str(tmp_path), 'prefix': 'random'}}}
    targets = targets[:2]
    splits = []

    def read(config, split, temperatures):
        splits.append(split)
        return spec, rows, targets, np.arange(len(rows) + 1), {'esm': 0}, identity.copy()

    monkeypatch.setattr(module, '_probability_inputs', lambda *args: {str(source)})
    monkeypatch.setattr(module, '_read_probabilities', read)
    fitted_dir = tmp_path / 'validation'
    module.run_head_calibration(config, fitted_dir)
    assert splits == ['validation']
    fitted = fitted_dir / 'fitted_weights.json'
    module.run_head_calibration(config, tmp_path / 'test', fitted)
    assert splits == ['validation', 'test']
    identity['precision'] = 'changed'
    with pytest.raises(ValueError, match='precision'):
        module.run_head_calibration(config, tmp_path / 'changed_export', fitted)
    identity['precision'] = 'float32'
    broken = json.loads(fitted.read_text())
    broken['input_sha256'].pop(str(source))
    fitted.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match='manifest'):
        module.run_head_calibration(config, tmp_path / 'incomplete', fitted)
    assert splits == ['validation', 'test', 'test']


def test_failed_head_cli_marks_attempt_without_overwriting(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import natvqvae.cli as cli
    from natvqvae.data import write_json
    monkeypatch.setattr(cli, '_result_output', lambda path, _: tmp_path / path)
    args = SimpleNamespace(config='config', output='attempt')

    def fail(config, output):
        output.mkdir()
        write_json(output / 'identity.json', {'status': 'RUNNING', 'split': 'validation'})
        raise ValueError('invalid probabilities')

    with pytest.raises(ValueError):
        cli._head_run(fail, args)
    identity = json.loads((tmp_path / 'attempt/identity.json').read_text())
    assert identity['status'] == 'FAILED' and identity['split'] == 'validation'
    with pytest.raises(FileExistsError):
        cli._head_run(fail, args)
    assert json.loads((tmp_path / 'attempt/identity.json').read_text()) == identity
