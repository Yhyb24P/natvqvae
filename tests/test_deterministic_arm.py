import pytest
import torch

from natvqvae.cli import build_parser
from natvqvae.evaluate import conditional_scores
from natvqvae.model import ConditionalModel, load_checkpoint
from natvqvae.train import _save_checkpoint, train

MODEL_CONFIG = {'hidden': 16, 'factors': 4, 'states': 4, 'feature_dim': 32}


def _tiny_model(arm):
    return ConditionalModel(arm, **MODEL_CONFIG).eval()


def test_deterministic_arm_rejects_latent_heads():
    model = _tiny_model('deterministic')
    features = torch.randn(2, 5, 32)
    valid = torch.ones(2, 5, dtype=torch.bool)
    mask = torch.zeros_like(valid)
    mask[0, [1, 3]] = True
    mask[1, [0, 4]] = True
    with torch.no_grad():
        context = model.encode_context(features, valid, mask)
        latent = model.deterministic_context(context, valid, mask)
    assert latent.shape == (2, 16)
    with pytest.raises(ValueError):
        model.prior(context, valid, mask)
    with pytest.raises(ValueError):
        model.posterior(features, context, valid, mask)


def test_deterministic_scoring_uses_one_distribution():
    torch.manual_seed(0)
    model = _tiny_model('deterministic')
    features = torch.randn(2, 5, 32)
    valid = torch.ones(2, 5, dtype=torch.bool)
    mask = torch.zeros_like(valid)
    mask[0, [1, 3]] = True
    mask[1, [0, 4]] = True
    labels = torch.zeros(2, 5, dtype=torch.long)
    with torch.no_grad():
        scores = conditional_scores(model, features, None, valid, mask, labels, 4,
                                    torch.Generator().manual_seed(1))
    assert scores['sites'].tolist() == [2, 2]
    assert torch.equal(scores['prior_joint_nll'], scores['posterior_joint_nll'])
    assert torch.equal(scores['prior_joint_nll'], scores['marginal_nll'])
    assert scores['ess'] is None and scores['max_weight'] is None


def test_checkpoint_records_model_arm(tmp_path):
    for arm in ('QVAE', 'deterministic'):
        model = _tiny_model(arm)
        path = tmp_path / f'{arm}.pt'
        _save_checkpoint(path, model, {'seed': 1}, {'model': MODEL_CONFIG},
                         3, 1, {'mode': 'random'}, None)
        payload = load_checkpoint(path, device='cpu')
        assert payload['arm'] == arm
        assert payload['step'] == 3


def test_train_rejects_unknown_arm():
    with pytest.raises(ValueError):
        train('config/transfer.yaml', 'config/sources.yaml', 'config/target.yaml',
              'unused', arm='marginal')


def test_train_cli_exposes_arm_choice():
    args = build_parser().parse_args(['train', '--output', 'data/results/training/run'])
    assert args.arm == 'QVAE'
    args = build_parser().parse_args(
        ['train', '--output', 'data/results/training/run', '--arm', 'deterministic'])
    assert args.arm == 'deterministic'


def test_evaluate_entry_runs_deterministic_checkpoint(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace

    import natvqvae.evaluate as evaluate_module

    config_path, sources_path, target_path = (tmp_path / name for name in
                                               ('config.yaml', 'sources.yaml', 'target.yaml'))
    for path in (config_path, sources_path, target_path):
        path.write_text('placeholder')
    population = tmp_path / 'population.fasta'
    population.write_text('>x\nA\n')
    splits = {}
    for role in ('training', 'validation', 'test'):
        path = tmp_path / f'{role}.csv'
        path.write_text('placeholder')
        splits[role] = path
    esm_dir = tmp_path / 'esm'
    esm_dir.mkdir()
    (esm_dir / 'config.json').write_text('{}')
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'placeholder')

    records = [SimpleNamespace(sequence_id=f'seq{i}', component=f'group{i}') for i in range(2)]
    queries = [{'record': record, 'panel': {'panel': 'NMNAT_8_count',
                                            'positions': [1, 2], 'weight': 1.0}}
               for record in records]
    data = SimpleNamespace(
        study={'evaluation': {'sample_count': 8, 'repeats': 4, 'batch_size': 1, 'seed': 8501},
               'training': {'mask_seed': 7401}},
        validation=records, validation_queries=lambda: queries,
        sources={'esm_model': str(esm_dir), 'population': str(population), 'splits': splits})

    def batch(items, encoder, device, need_full=True):
        size, length = len(items), 4
        features = torch.randn(size, length, 32, device=device)
        valid = torch.ones(size, length, dtype=torch.bool, device=device)
        mask = torch.zeros_like(valid)
        mask[:, 1:3] = True
        labels = torch.zeros(size, length, dtype=torch.long, device=device)
        return features, (features if need_full else None), valid, mask, labels

    data.batch = batch
    model = _tiny_model('deterministic')
    monkeypatch.setattr(evaluate_module, 'PopulationData', lambda *args, **kwargs: data)
    monkeypatch.setattr(evaluate_module, 'FrozenEncoder', lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluate_module, 'load_checkpoint',
                        lambda *args, **kwargs: {'arm': 'deterministic'})
    monkeypatch.setattr(evaluate_module, 'build_model',
                        lambda payload, device=None: model.to(device))

    output = tmp_path / 'evaluation'
    summary = evaluate_module.evaluate(str(config_path), str(sources_path), str(target_path),
                                       str(checkpoint), str(output), device='cpu')
    identity = json.loads((output / 'identity.json').read_text())
    rows = [json.loads(line) for line in (output / 'per_query.jsonl').read_text().splitlines()]

    assert summary['arm'] == 'deterministic'
    assert summary['sample_count'] is None
    assert summary['repeats'] == 1
    assert 'exact factorized' in summary['estimator']
    assert identity['arm'] == 'deterministic'
    assert 'exact factorized' in identity['estimator']
    assert identity['effective_repeats'] == 1
    assert len(rows) == len(queries)
    assert all(len(row['prior_site_marginal_log_probs']) == 2 for row in rows)
    assert all(row['kl'] == 0.0 and 'ess' not in row for row in rows)
