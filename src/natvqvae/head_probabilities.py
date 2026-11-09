"""Export aligned complete site distributions for calibration experiments."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .data import PopulationData, load_yaml, mask_panels, sha256_file, write_json
from .fragment_comparison import _require
from .model import AA, FrozenEncoder, build_model, load_checkpoint, target_tensors
from .test_data import load_recovered_test


def export_head_probabilities(config_path, output, split='validation', device='cuda'):
    config = load_yaml(config_path)
    _require(split in ('validation', 'test'), 'Unsupported split')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    data = PopulationData(config['sources'], config['study'], config['target'])
    records = data.validation if split == 'validation' else load_recovered_test(data, config['test_data'])
    queries = [{'record': record, 'panel': panel} for record in records
               for panel in mask_panels(record, data.study['training']['mask_seed'])]
    paths = [config_path, config['sources'], config['study'], config['target'], data.sources['population'],
             data.sources['splits']['training'], data.sources['splits']['validation']]
    if split == 'test':
        paths += [Path(config['test_data']) / name for name in
                  ('test.csv', 'test.fasta', 'manifest.json', 'historical_use.json')]
    device = torch.device(device)
    spec = config['models']['deterministic']
    models = {}
    for seed in config['seeds']:
        run = spec['prefix'] + str(seed)
        path = Path(spec['training']) / run / 'training_summary.json'
        paths.append(path)
        training = json.loads(path.read_text())
        best = training['checkpoints']['best']
        _require(training['status'] == 'COMPLETED' and training['arm'] == 'deterministic' and
                 training['seed'] == seed and training['initialization'] == {'mode': 'random'} and
                 sha256_file(best['path']) == best['sha256'], 'Selected deterministic checkpoint differs')
        paths.append(best['path'])
        models[str(seed)] = build_model(load_checkpoint(best['path'], device), device)
    model_names = ['esm', *models]
    tasks, arrays, labels = [], {name: [] for name in model_names}, []
    esm_path = Path(data.sources['esm_model'])
    identity = {'status': 'RUNNING', 'split': split, 'amino_acid_order': AA,
                'input_sha256': {str(path): sha256_file(path) for path in paths},
                'esm_files_sha256': {p.name: sha256_file(p) for p in esm_path.iterdir()
                                     if p.suffix in ('.json', '.safetensors') or p.name == 'vocab.txt'},
                'implementation_sha256': {name: sha256_file(Path(__file__).with_name(name))
                                         for name in ('head_probabilities.py', 'model.py', 'data.py')},
                'precision': 'bfloat16 forward, float32 probabilities' if device.type == 'cuda' else 'float32',
                'definition': '20AA log probabilities at all targeted positions under shared masked ESM features',
                'model_order': model_names, 'queries': len(queries)}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'identity.json', identity)
    encoder = FrozenEncoder(esm_path, device)
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                               enabled=device.type == 'cuda'):
        for start in range(0, len(queries), data.study['evaluation']['batch_size']):
            batch = queries[start:start + data.study['evaluation']['batch_size']]
            context, _, valid, mask, target_labels = data.batch(batch, encoder, device, need_full=False)
            _, y, use, indices = target_tensors(context, mask, target_labels)
            esm = encoder.amino_acid_log_probs_from_features(context)
            probs = {'esm': esm.gather(1, indices[..., None].expand(-1, -1, 20))}
            for name, model in models.items():
                hidden = model.encode_context(context, valid, mask)
                targets, _, _, _ = target_tensors(hidden, mask, target_labels)
                logits = model.decode(targets, model.deterministic_context(hidden, valid, mask).unsqueeze(0))[0]
                probs[name] = logits.float().log_softmax(-1)
            for name in model_names:
                values = probs[name][use].float()
                _require(bool(torch.isfinite(values).all()) and
                         bool(torch.allclose(values.exp().sum(-1), torch.ones(len(values), device=device),
                                             atol=2e-6)), 'Invalid exported probability normalization')
                arrays[name].append(values.cpu().numpy())
            labels.append(y[use].cpu().numpy())
            for query in batch:
                record, panel = query['record'], query['panel']
                tasks.append({'sequence_id': record.sequence_id, 'component': record.component,
                              'panel': panel['panel'], 'region': panel['region'],
                              'positions_0based': panel['positions'], 'weight': panel['weight']})
            if start % 800 == 0:
                print(json.dumps({'completed_queries': len(tasks), 'total_queries': len(queries)}), flush=True)
    boundaries = np.cumsum([0, *[len(task['positions_0based']) for task in tasks]])
    np.savez_compressed(output / 'site_log_probs.npz',
                        **{name: np.concatenate(values) for name, values in arrays.items()},
                        labels=np.concatenate(labels), boundaries=boundaries)
    write_json(output / 'queries.json', tasks)
    identity.update({'status': 'COMPLETED', 'sites': int(boundaries[-1]),
                     'output_sha256': {name: sha256_file(output / name) for name in
                                       ('site_log_probs.npz', 'queries.json')}})
    write_json(output / 'identity.json', identity)
    return {'status': 'COMPLETED', 'split': split, 'sites': int(boundaries[-1]), 'output': str(output)}
