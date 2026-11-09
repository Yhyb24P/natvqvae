import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from natvqvae import evaluate as evaluation
from natvqvae.data import Record
from natvqvae.model import AA, ConditionalModel


@pytest.mark.parametrize('split', ['validation', 'test'])
def test_shared_encoder_preserves_individual_scores_and_random_streams(tmp_path, monkeypatch, split):
    paths = {}
    for name in ('config', 'sources', 'target', 'population', 'training', 'validation', 'test'):
        path = tmp_path / name
        path.write_text(name)
        paths[name] = str(path)
    esm = tmp_path / 'esm'
    esm.mkdir()
    (esm / 'config.json').write_text('{}')
    restored = tmp_path / 'restored'
    restored.mkdir()
    for name in ('test.csv', 'test.fasta', 'manifest.json'):
        (restored / name).write_text(name)
    records = [Record(hashlib.sha256(seq.encode()).hexdigest(), seq, str(i // 2), len(seq), 1, 3, 4, len(seq))
               for i, seq in enumerate(('ACDE', 'ACDEFG', 'ACDEF'))]
    panel = {'panel': 'NMNAT_2_count', 'positions': [1, 2], 'weight': .5}
    queries = [{'record': record, 'panel': panel} for record in records]
    calls, loads = [], []

    class Encoder:
        def __init__(self, *args, **kwargs):
            loads.append(1)

        def __call__(self, sequences, positions):
            calls.append(tuple(tuple(sites) for sites in positions))
            width = max(map(len, sequences))
            h = torch.zeros(len(sequences), width, 4)
            valid = torch.zeros(len(sequences), width, dtype=torch.bool)
            mask = torch.zeros_like(valid)
            for i, (sequence, sites) in enumerate(zip(sequences, positions)):
                valid[i, :len(sequence)] = True
                h[i, :len(sequence)] = torch.tensor([[AA.index(aa) / 20] * 4 for aa in sequence])
                h[i, sites] = 0
                mask[i, sites] = True
            return h, valid, mask

    def batch(items, encoder, device, need_full):
        sequences = [q['record'].sequence for q in items]
        context, valid, mask = encoder(sequences, [q['panel']['positions'] for q in items])
        full = encoder(sequences, [[] for _ in items])[0] if need_full else None
        labels = torch.zeros(valid.shape, dtype=torch.long)
        for i, sequence in enumerate(sequences):
            labels[i, :len(sequence)] = torch.tensor([AA.index(aa) for aa in sequence])
        return context, full, valid, mask, labels

    data = SimpleNamespace(study={'evaluation': {'sample_count': 5, 'repeats': 3, 'batch_size': 2, 'seed': 8501},
                                 'training': {'mask_seed': 7401}},
                           validation=records, validation_queries=lambda: queries, batch=batch,
                           sources={'esm_model': str(esm), 'population': paths['population'],
                                    'splits': {role: paths[role] for role in ('training', 'validation', 'test')}})
    monkeypatch.setattr(evaluation, 'PopulationData', lambda *args: data)
    monkeypatch.setattr(evaluation, 'FrozenEncoder', Encoder)
    monkeypatch.setattr(evaluation, 'mask_panels', lambda *args: [panel])
    from natvqvae import test_data
    monkeypatch.setattr(test_data, 'load_recovered_test', lambda *args: records)
    checkpoints = []
    for i, arm in enumerate(('QVAE', 'deterministic', 'QVAE')):
        torch.manual_seed(20 + i)
        config = {'hidden': 4, 'feature_dim': 4, 'factors': 2, 'states': 2}
        model = ConditionalModel(arm, **config)
        path = tmp_path / f'{i}.pt'
        torch.save({'state_dict': model.state_dict(), 'arm': arm, 'model_config': config}, path)
        checkpoints.append(path)
    options = {'device': 'cpu', 'split': split, 'test_data': restored if split == 'test' else None}
    singles = [evaluation.evaluate(paths['config'], paths['sources'], paths['target'], checkpoint,
                                   tmp_path / f'single_{i}', **options)
               for i, checkpoint in enumerate(checkpoints)]
    assert len(calls) == 10 and len(loads) == 3
    calls.clear(); loads.clear()
    runs = [{'checkpoint': checkpoint, 'output': tmp_path / f'batch_{i}'}
            for i, checkpoint in enumerate(checkpoints)]
    summaries = evaluation.evaluate_many(paths['config'], paths['sources'], paths['target'], runs, **options)
    assert len(calls) == 4 and len(loads) == 1
    for i, (single, combined) in enumerate(zip(singles, summaries)):
        assert (tmp_path / f'single_{i}/per_query.jsonl').read_bytes() == (tmp_path / f'batch_{i}/per_query.jsonl').read_bytes()
        for key in single.keys() - {'seconds', 'shared_encoder_checkpoints'}:
            assert single[key] == combined[key]
        identity = json.loads((tmp_path / f'batch_{i}/identity.json').read_text())
        assert identity['shared_encoder_checkpoints'] == 3
        assert combined['sample_count'] is None if i == 1 else combined['sample_count'] == 5
    calls.clear()
    reversed_runs = [{**run, 'output': tmp_path / f'reversed_{i}'} for i, run in reversed(list(enumerate(runs)))]
    evaluation.evaluate_many(paths['config'], paths['sources'], paths['target'], reversed_runs, **options)
    for i in range(3):
        assert (tmp_path / f'batch_{i}/per_query.jsonl').read_bytes() == (tmp_path / f'reversed_{i}/per_query.jsonl').read_bytes()
    assert len(calls) == 4
    with pytest.raises(FileExistsError):
        evaluation.evaluate_many(paths['config'], paths['sources'], paths['target'], runs, **options)
    with pytest.raises(ValueError, match='non-overlapping'):
        evaluation.evaluate_many(paths['config'], paths['sources'], paths['target'],
                                 [runs[0], {**runs[1], 'output': Path(runs[0]['output']) / 'child'}], **options)
