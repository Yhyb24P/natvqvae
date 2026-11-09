import json
from types import SimpleNamespace

import numpy as np
import torch

from natvqvae.model import ConditionalModel


def test_export_preserves_target_order_and_shared_features(monkeypatch, tmp_path):
    import natvqvae.head_probabilities as module

    calls = []
    records = [SimpleNamespace(sequence_id='a', component='g'),
               SimpleNamespace(sequence_id='b', component='h')]
    panels = {'a': [0, 2], 'b': [1]}
    context = torch.arange(24).reshape(2, 3, 4).float() / 24
    mask = torch.tensor([[True, False, True], [False, True, False]])
    labels = torch.tensor([[0, 1, 2], [3, 4, 5]])
    valid = torch.ones(2, 3, dtype=torch.bool)

    def batch(queries, encoder, device, need_full):
        calls.append('encoder')
        assert len(queries) == 2 and not need_full
        return context, None, valid, mask, labels

    data = SimpleNamespace(validation=records, study={'training': {'mask_seed': 1},
                           'evaluation': {'batch_size': 2}}, batch=batch,
                           sources={'population': 'population', 'splits': {'training': 'train',
                                    'validation': 'val'}, 'esm_model': str(tmp_path)})
    config = {'sources': 'sources', 'study': 'study', 'target': 'target', 'seeds': [1, 2, 3],
              'models': {'deterministic': {'prefix': 'random', 'training': str(tmp_path)}}}
    for seed in config['seeds']:
        root = tmp_path / f'random{seed}'
        root.mkdir()
        (root / 'training_summary.json').write_text(json.dumps({
            'status': 'COMPLETED', 'arm': 'deterministic', 'seed': seed,
            'initialization': {'mode': 'random'}, 'checkpoints': {'best': {'path': 'ckpt', 'sha256': 'hash'}}}))
    model = ConditionalModel('deterministic', hidden=8, feature_dim=4).eval()
    esm = torch.arange(120).reshape(2, 3, 20).float().log_softmax(-1)
    monkeypatch.setattr(module, 'load_yaml', lambda _: config)
    monkeypatch.setattr(module, 'PopulationData', lambda *args: data)
    monkeypatch.setattr(module, 'sha256_file', lambda _: 'hash')
    monkeypatch.setattr(module, 'mask_panels', lambda record, _: [
        {'panel': 'p', 'region': 'NMNAT', 'positions': panels[record.sequence_id], 'weight': 1}])
    monkeypatch.setattr(module, 'load_checkpoint', lambda *args: {})
    monkeypatch.setattr(module, 'build_model', lambda *args: model)
    monkeypatch.setattr(module, 'FrozenEncoder', lambda *args: SimpleNamespace(
        amino_acid_log_probs_from_features=lambda h: esm))
    output = tmp_path / 'output'
    result = module.export_head_probabilities('config', output, device='cpu')
    assert calls == ['encoder'] and result['sites'] == 3
    with np.load(output / 'site_log_probs.npz') as values:
        assert values['boundaries'].tolist() == [0, 2, 3]
        assert values['labels'].tolist() == [0, 2, 4]
        np.testing.assert_allclose(values['esm'], esm[mask].numpy())
        for seed in config['seeds']:
            assert values[str(seed)].shape == (3, 20)
            np.testing.assert_allclose(np.exp(values[str(seed)]).sum(-1), 1, atol=1e-6)
    assert [q['positions_0based'] for q in json.loads((output / 'queries.json').read_text())] == [[0, 2], [1]]
