import json
import csv
import hashlib
from pathlib import Path

import pytest
import torch

from natvqvae import auxiliary


def test_update_warmup_is_independent_of_fragment_epoch_length():
    config = {'learning_rate': .01, 'learning_rate_warmup_steps': 4,
              'beta_warmup_steps': 8}
    assert auxiliary._aux_schedule(config, 2, 1) == (.005, .25)
    assert auxiliary._aux_schedule(config, 2, 9) == (.005, .25)
    assert auxiliary._aux_schedule(config, 9, 3) == (.01, 1.0)
    with pytest.raises(ValueError, match='Both'):
        auxiliary._aux_schedule({'learning_rate': .01, 'beta_warmup_steps': 8}, 1, 1)


def test_paired_fragments_share_order_weights_and_masked_wt_sites(tmp_path):
    sequences = {'a': ['ACDEFGH', 'VCDEFGH'], 's': ['DACDEFGH', 'DACDEFGH']}
    columns = [4, 5, 7, 10]
    positions = {'a': [0, 1, 3, 6], 's': [1, 2, 4, 7]}
    rows, panels, paths = [], {}, {}
    ids = {}
    for side in ('a', 's'):
        ids[side] = [(('aux:' if side == 'a' else 'self:') + hashlib.sha256(seq.encode()).hexdigest())
                     for seq in sequences[side]]
        path = tmp_path / f'{side}.fasta'
        path.write_text(''.join(f'>{uid}\n{seq}\n' for uid, seq in dict(zip(ids[side], sequences[side])).items()))
        paths[f'{side}_fragments'] = str(path)
    for i in range(2):
        uid = f'pair{i}'
        row = {'pair_id': uid, 'weight': [.75, 1.25][i], 'common_count': len(columns)}
        for side in ('a', 's'):
            row.update({f'{side}_sequence_id': ids[side][i],
                        f'{side}_sequence_sha256': ids[side][i].split(':')[1],
                        f'{side}_length': len(sequences[side][i])})
        rows.append(row)
        item = {'common_wt_columns': columns, 'a_common_positions': positions['a'],
                's_common_positions': positions['s'], 'panels': []}
        for nominal, chosen in [(2, [4, 7]), (8, [4, 5, 10])]:
            panel = {'panel': f'mask_{nominal}', 'nominal_size': nominal,
                     'actual_size': len(chosen), 'wt_columns': chosen}
            for side in ('a', 's'):
                panel[f'{side}_positions'] = [dict(zip(columns, positions[side]))[w] for w in chosen]
            item['panels'].append(panel)
        panels[uid] = item
    manifest = tmp_path / 'pairs.csv'
    with manifest.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    panel_path = tmp_path / 'panels.json'
    panel_path.write_text(json.dumps(panels))
    paths.update(pair_manifest=str(manifest), pair_panels=str(panel_path),
                 target=str(Path('config/target.yaml').resolve()))
    config = {'paths': paths, 'training': {'seed': 13, 'mask_sizes': [2, 8]}}
    data = {}
    for condition in ('external', 'self'):
        config['fragment_condition'] = condition
        config_path = tmp_path / f'{condition}.json'
        config_path.write_text(json.dumps(config))
        data[condition] = auxiliary.AuxiliaryData(config_path)
    # The repeated S fragment remains two paired training queries, with the same
    # weights and masks as A; it is not counted as two unique protein sequences.
    assert len(data['self'].domains) == 1
    assert len(data['self'].train) == 2
    for epoch in range(3):
        a, s = [data[c].training_queries(epoch, 13) for c in ('external', 'self')]
        assert [(q['record']['uid'], q['panel']['panel'], q['weight']) for q in a] == [
            (q['record']['uid'], q['panel']['panel'], q['weight']) for q in s]
        assert [len(q['panel']['positions']) for q in a] == [len(q['panel']['positions']) for q in s]
    panels['pair0']['panels'][0]['s_positions'][0] = 0
    panel_path.write_text(json.dumps(panels))
    with pytest.raises(ValueError, match='same WT columns'):
        auxiliary.AuxiliaryData(tmp_path / 'self.json')


@pytest.mark.parametrize('fixed', [True, False])
def test_auxiliary_training_obeys_checkpoint_rule(tmp_path, monkeypatch, fixed):
    paths = {}
    for name in ('domains', 'masks', 'split', 'target'):
        path = tmp_path / name
        path.write_text(name)
        paths[name] = str(path)
    esm = tmp_path / 'esm'
    esm.mkdir()
    paths.update(esm_model=str(esm), checkpoint_dir=str(tmp_path / 'weights'))
    config = {'paths': paths, 'model': {'hidden': 16, 'factors': 2, 'states': 2,
                                      'feature_dim': 32},
              'generation': {'balance_key': 'domain_cluster50'},
              'training': {'seed': 9, 'arms': ['QVAE'], 'batch_size': 2, 'max_steps': 3,
                           'K': 2, 'learning_rate': .001, 'weight_decay': .01,
                           'gradient_clip': 1., 'mask_sizes': [2],
                           'learning_rate_warmup_steps': 2, 'beta_warmup_steps': 3}}
    if fixed:
        config['training']['checkpoint_selection'] = 'fixed_updates'
    else:
        config['training']['selection'] = {'records': 1, 'panel': 'mask_2',
                                           'sample_count': 2, 'interval_epochs': 1}
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config))
    calls = []

    class Data:
        def __init__(self, path):
            self.config = json.loads(path.read_text())
            self.input_paths = {k: paths[k] for k in ('domains', 'masks', 'split', 'target')}
            self.train = [{'uid': str(i), 'weight': 1., 'sequence': 'ACDE'} for i in range(3)]
            self.validation = [] if fixed else self.train[:1]
            self.panels = {r['uid']: [{'panel': 'mask_2', 'positions': [0, 2]}]
                           for r in self.train}
            self.split = {'0': {'domain_cluster50': 'g'}}
            self.nonstandard_domains = self.excluded_masked_positions = 0

        def training_queries(self, epoch, seed):
            calls.append((epoch, seed, self.config['training']['seed']))
            return [{'record': r, 'panel': self.panels[r['uid']][0], 'weight': 1.}
                    for r in self.train]

        def validation_selection(self, limit, panel):
            assert not fixed, 'Fixed-update training must never select validation queries'
            return self.validation, [{'record': self.validation[0],
                                      'panel': self.panels['0'][0], 'weight': 1.}]

        def batch(self, queries, encoder, device, need_full):
            n = len(queries)
            features = torch.randn(n, 4, 32, generator=torch.Generator().manual_seed(42))
            valid = torch.ones(n, 4, dtype=torch.bool)
            mask = torch.zeros_like(valid)
            mask[:, [0, 2]] = True
            labels = torch.zeros(n, 4, dtype=torch.long)
            return features, features.clone() if need_full else None, valid, mask, labels

    validation_calls = []

    def score(*args):
        assert not fixed, 'Fixed-update training must never score validation'
        validation_calls.append(1)
        return {'prior_joint_nll_per_residue': 5. - len(validation_calls)}, []

    monkeypatch.setattr(auxiliary, 'AuxiliaryData', Data)
    monkeypatch.setattr(auxiliary, 'FrozenEncoder', lambda *a, **k: None)
    monkeypatch.setattr(auxiliary, '_score_validation', score)
    summary = auxiliary.train_aux(config_path, tmp_path / 'run', 'cpu', seed=17)
    result = summary['arms']['QVAE']
    assert result['steps'] == 3
    assert calls == [(0, 17, 17), (1, 17, 17)]
    initial = torch.load(result['checkpoints']['initial']['path'], weights_only=False)
    last = torch.load(result['checkpoints']['last']['path'], weights_only=False)
    assert last['step'] == 3 and last['seed'] == 17
    assert any(not torch.equal(initial['state_dict'][k], v)
               for k, v in last['state_dict'].items())
    metrics = [json.loads(line) for line in (tmp_path / 'run/QVAE_metrics.jsonl').read_text().splitlines()]
    assert [row['learning_rate'] for row in metrics] == [.0005, .001, .001]
    assert [row['beta'] for row in metrics] == [1/3, 2/3, 1.]
    if fixed:
        assert set(result['checkpoints']) == {'initial', 'last'}
        assert result['best_epoch'] is None and result['validation_curve'] == []
        assert not (tmp_path / 'run/validation_selection.json').exists()
    else:
        assert len(validation_calls) == 3
        assert result['best_epoch'] == 2
        assert torch.load(result['checkpoints']['best']['path'], weights_only=False)['step'] == 3
