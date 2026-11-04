"""NMNAT auxiliary-domain QVAE: data, training, generation, and evaluation."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from Bio import SeqIO
from torch.nn.utils import clip_grad_norm_

from .data import (Target, load_auxiliary_inputs, load_yaml, sha256_file, stable_seed,
                   write_json)
from .model import (AA, ConditionalModel, FrozenEncoder, build_model, categorical_log_prob,
                    categorical_samples, load_checkpoint, sample_letters, target_tensors)
from .train import _beta_at_epoch, _learning_rate_at_epoch, _train_step

ANCHORS = [14, 17, 135, 138]


class AuxiliaryData:
    def __init__(self, config_path):
        self.config = load_yaml(config_path)
        paths = self.config['paths']
        self.target = Target.load(paths['target'])
        self.study = self.config
        self.sources = paths
        if 'pair_manifest' in paths:
            self._paired_inputs(paths)
            return
        self.input_paths = {key: paths[key] for key in ('domains', 'masks', 'split', 'target')}
        for key, name in [('domains', 'aux_domains.fasta'), ('masks', 'aux_masks.json'),
                          ('split', 'aux_split.csv')]:
            if Path(paths[key]).resolve() != (Path(paths['domains']).parent / name).resolve():
                raise ValueError(f'Auxiliary {key} path differs from the actual input filename')
        validated = load_auxiliary_inputs(Path(paths['domains']).parent)
        self.domains = validated['domains']
        self.masks = validated['masks']
        rows = validated['split']
        self.split = {}
        for row in rows:
            uid = row['aux_uid']
            if uid in self.split or uid not in self.domains or uid not in self.masks:
                raise ValueError(f'Invalid auxiliary split row: {uid}')
            self.split[uid] = row
        self.train = self._records('train')
        self.validation = self._records('val')
        self.train_ids = {record['uid'] for record in self.train}
        self.validation_ids = {record['uid'] for record in self.validation}
        self.training_weights = {record['uid']: float(self.split[record['uid']]['weight'])
                                 for record in self.train}
        self.nonstandard_domains = sum(any(residue not in AA for residue in sequence)
                                       for sequence in self.domains.values())
        self.excluded_masked_positions = sum(
            1 for uid, item in self.masks.items() for position in item['positions']
            if self.domains[uid][position] not in AA)
        self.panels = {}
        for record in self.train + self.validation:
            self.panels[record['uid']] = self._make_panels(record)
            if not self.panels[record['uid']]:
                raise ValueError(f'No valid masking panel for auxiliary record {record["uid"]}')

    def _paired_inputs(self, paths):
        condition = self.config.get('fragment_condition')
        if condition not in ('external', 'self'):
            raise ValueError('Paired fragments require external or self condition')
        self.input_paths = {key: paths[key] for key in
                            ('pair_manifest', 'pair_panels', 'a_fragments', 's_fragments', 'target')}
        arm = 'a' if condition == 'external' else 's'
        domains = {side: {uid: str(record.seq).upper() for uid, record in
                          SeqIO.to_dict(SeqIO.parse(paths[f'{side}_fragments'], 'fasta')).items()}
                   for side in ('a', 's')}
        for side in domains:
            for uid, sequence in domains[side].items():
                if not sequence or uid.split(':', 1)[-1] != hashlib.sha256(sequence.encode()).hexdigest():
                    raise ValueError(f'Paired fragment sequence identity differs: {uid}')
        with open(paths['pair_manifest']) as handle:
            rows = list(csv.DictReader(handle))
        panels = json.loads(Path(paths['pair_panels']).read_text())
        self.domains, self.masks, self.split, self.panels = domains[arm], {}, {}, {}
        self.train, self.validation = [], []
        lower, upper = self.target.domains['NMNAT']
        for row in sorted(rows, key=lambda item: item['pair_id']):
            uid = row['pair_id']
            if uid in self.split or uid not in panels:
                raise ValueError(f'Duplicate pair or missing panels: {uid}')
            weight = float(row['weight'])
            if not np.isfinite(weight) or weight <= 0:
                raise ValueError(f'Invalid paired weight: {uid}')
            item = panels[uid]
            columns = item['common_wt_columns']
            if (len(columns) < 2 or columns != sorted(set(columns)) or
                    min(columns) < lower or max(columns) > upper or
                    len(columns) != int(row['common_count'])):
                raise ValueError(f'Invalid common WT columns: {uid}')
            for side in ('a', 's'):
                sequence = domains[side][row[f'{side}_sequence_id']]
                if (hashlib.sha256(sequence.encode()).hexdigest() != row[f'{side}_sequence_sha256'] or
                        len(sequence) != int(row[f'{side}_length'])):
                    raise ValueError(f'Paired sequence does not match manifest: {uid}, {side}')
                positions = item[f'{side}_common_positions']
                if (len(positions) != len(columns) or positions != sorted(set(positions)) or
                        min(positions) < 0 or max(positions) >= len(sequence) or
                        any(sequence[position] not in AA for position in positions)):
                    raise ValueError(f'Invalid paired target mapping: {uid}, {side}')
            saved = item['panels']
            nominal = list(self.config['training']['mask_sizes'])
            if [panel['nominal_size'] for panel in saved] != nominal:
                raise ValueError(f'Panel sizes differ from training configuration: {uid}')
            for panel in saved:
                count = min(int(panel['nominal_size']), len(columns) - 1)
                wt_columns = panel['wt_columns']
                if (count < 1 or panel['actual_size'] != count or
                        len(wt_columns) != count or wt_columns != sorted(set(wt_columns)) or
                        not set(wt_columns) <= set(columns) or
                        panel['panel'] != f'mask_{panel["nominal_size"]}'):
                    raise ValueError(f'Invalid paired masking panel: {uid}')
                for side in ('a', 's'):
                    mapping = dict(zip(columns, item[f'{side}_common_positions']))
                    if panel[f'{side}_positions'] != [mapping[w] for w in wt_columns]:
                        raise ValueError(f'Paired mask does not use the same WT columns: {uid}, {side}')
            sequence = domains[arm][row[f'{arm}_sequence_id']]
            pairs = list(zip(item[f'{arm}_common_positions'], columns))
            self.train.append({'uid': uid, 'sequence': sequence, 'pairs': pairs,
                               'kind': condition, 'weight': weight,
                               'fragment_id': row[f'{arm}_sequence_id']})
            self.split[uid] = row
            self.masks[uid] = {'pairs': pairs}
            self.panels[uid] = [{'panel': p['panel'], 'positions': p[f'{arm}_positions']}
                                for p in saved]
        if not rows or set(panels) != set(self.split):
            raise ValueError('Empty pairs or panel identities differ from manifest')
        if not np.isclose(np.mean([r['weight'] for r in self.train]), 1., rtol=1e-8):
            raise ValueError('Paired training weights must have mean one')
        self.train_ids, self.validation_ids = set(self.split), set()
        self.training_weights = {r['uid']: r['weight'] for r in self.train}
        self.nonstandard_domains = sum(any(residue not in AA for residue in seq)
                                       for seq in self.domains.values())
        self.excluded_masked_positions = 0

    def _records(self, split):
        output = []
        for uid, row in sorted(self.split.items()):
            if row['split'] != split:
                continue
            sequence = self.domains[uid]
            pairs = self.masks[uid]['pairs']
            output.append({'uid': uid, 'sequence': sequence,
                           'pairs': [(int(q), int(w)) for q, w in pairs],
                           'kind': row['aux_kind'], 'weight': float(row['weight'])})
        return output

    def _make_panels(self, record):
        available = [pair[0] for pair in record['pairs']
                     if record['sequence'][pair[0]] in AA]
        if len(available) < 2:
            return []
        panels = []
        for size in self.config['training']['mask_sizes']:
            count = min(int(size), len(available) - 1)
            if count < 1:
                continue
            rng = np.random.default_rng(stable_seed(self.config['training']['mask_seed'],
                                                    record['uid'], size))
            positions = sorted(map(int, rng.choice(available, count, replace=False)))
            panels.append({'panel': f'mask_{size}', 'positions': positions})
        return panels

    def _query(self, record, panel, weight):
        return {'record': record, 'panel': panel, 'weight': float(weight)}

    def training_queries(self, epoch, seed):
        rng = np.random.default_rng(stable_seed(self.config['training']['seed'], 'aux_order', epoch))
        queries = []
        for record in self.train:
            panels = self.panels[record['uid']]
            if not panels:
                continue
            queries.append(self._query(record, panels[int(rng.integers(len(panels)))],
                                       record['weight']))
        return [queries[i] for i in rng.permutation(len(queries))]

    def validation_queries(self):
        return [self._query(record, panel, record['weight'] / len(self.panels[record['uid']]))
                for record in self.validation
                for panel in self.panels[record['uid']]]

    def balanced_records(self, records, limit, key='domain_cluster50'):
        groups = defaultdict(list)
        for record in records:
            groups[self.split[record['uid']][key]].append(record)
        ordered = [groups[name] for name in sorted(groups)]
        limit = min(limit, sum(len(group) for group in ordered))
        base, extra = divmod(limit, len(ordered))
        pointers = [0] * len(ordered)
        selected = []
        for index, group in enumerate(ordered):
            take = min(len(group), base + (1 if index < extra else 0))
            selected.extend(group[:take])
            pointers[index] = take
        while len(selected) < limit:
            added = False
            for index, group in enumerate(ordered):
                if pointers[index] < len(group):
                    selected.append(group[pointers[index]])
                    pointers[index] += 1
                    added = True
                    if len(selected) >= limit:
                        break
            if not added:
                break
        return selected[:limit]

    def validation_selection(self, limit, panel_name):
        selected = self.balanced_records(self.validation, limit)
        queries = []
        for record in selected:
            panels = self.panels[record['uid']]
            panel = next((item for item in panels if item['panel'] == panel_name),
                         panels[0] if panels else None)
            if panel is not None:
                queries.append(self._query(record, panel, record['weight']))
        return selected, queries

    def batch(self, queries, encoder, device, need_full):
        sequences = [query['record']['sequence'] for query in queries]
        context, valid, mask = encoder(sequences, [query['panel']['positions'] for query in queries])
        full = None
        if need_full:
            full, _, _ = encoder(sequences, [[] for _ in queries])
        width = context.shape[1]
        labels = torch.zeros(len(queries), width, dtype=torch.long, device=device)
        for index, query in enumerate(queries):
            sequence = query['record']['sequence']
            labels[index, :len(sequence)] = torch.tensor(
                [AA.index(residue) if residue in AA else 0 for residue in sequence], device=device)
        return context.to(device), None if full is None else full.to(device), valid.to(device), mask.to(device), labels


def _checkpoint_paths(config, arm, seed):
    directory = Path(config['paths']['checkpoint_dir'])
    directory.mkdir(parents=True, exist_ok=True)
    return {kind: directory / f'{arm}_seed{seed}_{kind}.pt'
            for kind in ('initial', 'best', 'last')}


def _state_cpu(model):
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def _save_checkpoint(path, model, arm, seed, step, epoch, model_config, training,
                     selection):
    torch.save({'state_dict': _state_cpu(model), 'arm': arm, 'seed': int(seed),
                'model_config': dict(model_config), 'step': int(step), 'epoch': int(epoch),
                'config': dict(training), 'selection': selection}, path)


def _score_validation(model, data, encoder, queries, device, sample_count, seed):
    from .evaluate import conditional_scores
    model.eval()
    rows = []
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                         enabled=device.type == 'cuda'):
        for start in range(0, len(queries), 8):
            batch = queries[start:start + 8]
            context, full, valid, mask, labels = data.batch(
                batch, encoder, device, need_full=model.arm == 'QVAE')
            generator = torch.Generator(device=device.type)
            generator.manual_seed(stable_seed(seed, 'selection', start))
            scores = conditional_scores(model, context, full, valid, mask, labels,
                                        sample_count, generator)
            sites = scores['sites']
            for index, query in enumerate(batch):
                site = int(sites[index].item())
                if site < 1:
                    continue
                row = {'uid': query['record']['uid'], 'panel': query['panel']['panel'],
                       'sites': site, 'weight': float(query['weight'])}
                for name in ('prior_joint_nll', 'posterior_joint_nll', 'marginal_nll',
                             'prior_expected_reconstruction_nll',
                             'posterior_expected_reconstruction_nll'):
                    row[name + '_per_residue'] = float(scores[name][index].item()) / site
                row['kl'] = float(scores['kl'][index].item())
                if scores['ess'] is not None:
                    row['ess'] = float(scores['ess'][index].item())
                    row['max_weight'] = float(scores['max_weight'][index].item())
                rows.append(row)
    model.train()
    total_sites = sum(row['sites'] for row in rows)
    if not rows or total_sites < 1:
        raise ValueError('Validation selection produced no scored sites')
    summary = {'queries': len(rows), 'sites': total_sites,
               'selection_metric': 'prior_joint_nll_per_residue'}
    for name in ('prior_joint_nll', 'posterior_joint_nll', 'marginal_nll',
                 'prior_expected_reconstruction_nll',
                 'posterior_expected_reconstruction_nll'):
        summary[name + '_per_residue'] = sum(
            row[name + '_per_residue'] * row['sites'] for row in rows) / total_sites
    summary['kl_mean'] = sum(row['kl'] for row in rows) / len(rows)
    if rows[0].get('ess') is not None:
        summary['ess_mean'] = sum(row['ess'] for row in rows) / len(rows)
        summary['max_weight_mean'] = sum(row['max_weight'] for row in rows) / len(rows)
    return summary, rows


def _aux_schedule(training, update, epoch):
    """Use update-based warmup when declared, otherwise preserve historical epochs."""
    if update < 1 or epoch < 1:
        raise ValueError('Updates and epochs are one-based')
    step_keys = ('learning_rate_warmup_steps', 'beta_warmup_steps')
    if any(key in training for key in step_keys):
        if not all(key in training for key in step_keys):
            raise ValueError('Both update-based warmup lengths are required')
        if any(int(training[key]) < 0 for key in step_keys):
            raise ValueError('Warmup lengths must be nonnegative')
        lr = _learning_rate_at_epoch(float(training['learning_rate']), update,
                                     int(training[step_keys[0]]))
        beta = _beta_at_epoch(update, int(training[step_keys[1]]))
    else:
        lr = _learning_rate_at_epoch(float(training['learning_rate']), epoch,
                                     int(training['learning_rate_warmup_epochs']))
        beta = _beta_at_epoch(epoch, int(training['beta_warmup_epochs']))
    return lr, beta


def train_aux(config_path, output, device=None, seed=None):
    data = AuxiliaryData(config_path)
    study = data.config
    training = dict(study['training'])
    if seed is not None:
        training['seed'] = int(seed)
    # AuxiliaryData also uses this seed for query ordering and panel selection.
    study['training'] = training
    fixed = training.get('checkpoint_selection') == 'fixed_updates'
    if training.get('checkpoint_selection') not in (None, 'fixed_updates'):
        raise ValueError('Unknown auxiliary checkpoint selection rule')
    if fixed and training.get('selection') is not None:
        raise ValueError('Fixed updates cannot also select a checkpoint using validation')
    selection_config = None if fixed else training['selection']
    if not data.train or int(training['max_steps']) < 1 or int(training['batch_size']) < 1:
        raise ValueError('Training requires records, positive updates and batch size')
    if not fixed and not data.validation:
        raise ValueError('Validation checkpoint selection requires validation records')
    _aux_schedule(training, 1, 1)
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Training output already contains files: {output}')
    planned_checkpoints = {arm: _checkpoint_paths(study, arm, int(training['seed']))
                           for arm in training['arms']}
    existing = [str(path) for paths in planned_checkpoints.values()
                for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f'Auxiliary checkpoint already exists: {existing}')
    if fixed:
        for paths in planned_checkpoints.values():
            del paths['best']
    output.mkdir(parents=True, exist_ok=True)
    input_hashes = {key: sha256_file(path) for key, path in data.input_paths.items()}
    input_hashes['config'] = sha256_file(config_path)
    esm_path = Path(study['paths']['esm_model'])
    write_json(output / 'run_identity.json', {
        'input_sha256': input_hashes,
        'esm_files_sha256': {path.name: sha256_file(path) for path in sorted(esm_path.iterdir())
                             if path.suffix in {'.json', '.safetensors'} or path.name == 'vocab.txt'},
        'code_sha256': {name: sha256_file(Path(__file__).with_name(name))
                       for name in ('auxiliary.py', 'train.py', 'model.py', 'data.py')},
        'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--porcelain'], text=True).splitlines(),
        'python': platform.python_version(), 'torch': torch.__version__,
        'effective_training': training, 'seed_override': seed, 'device': str(device),
        'checkpoint_rule': 'fixed_updates' if fixed else 'auxiliary_validation',
    })
    encoder = FrozenEncoder(study['paths']['esm_model'], device=device)
    selected_records, selection_queries = [], []
    if not fixed:
        selected_records, selection_queries = data.validation_selection(
            int(selection_config['records']), selection_config['panel'])
        write_json(output / 'validation_selection.json', {
            'rule': f'round-robin by {study["generation"]["balance_key"]} over sorted validation records',
            'panel': selection_config['panel'], 'records': len(selected_records),
            'items': [{'uid': record['uid'],
                       'cluster': data.split[record['uid']][study['generation']['balance_key']],
                       'weight': record['weight'], 'panel': query['panel']['panel'],
                       'positions': query['panel']['positions']}
                      for record, query in zip(selected_records, selection_queries)],
        })
    summaries = {}
    for arm in training['arms']:
        started = time.monotonic()
        torch.manual_seed(int(training['seed']))
        model = ConditionalModel(arm, **study['model']).to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(training['learning_rate']),
                                      weight_decay=float(training['weight_decay']))
        steps_per_epoch = math.ceil(len(data.train) / int(training['batch_size']))
        max_epochs = math.ceil(int(training['max_steps']) / steps_per_epoch)
        metrics_path = output / f'{arm}_metrics.jsonl'
        curve_path = output / f'{arm}_validation_curve.jsonl'
        step, curve = 0, []
        initial_scores = None
        if not fixed:
            initial_scores, _ = _score_validation(
                model, data, encoder, selection_queries, device,
                int(selection_config['sample_count']),
                stable_seed(training['seed'], arm, 'selection', 0))
            _save_checkpoint(planned_checkpoints[arm]['best'], model, arm,
                             training['seed'], 0, 0, study['model'], training, initial_scores)
            curve.append({'stage': 'initial', 'epoch': 0, 'step': 0, **initial_scores})
            curve_path.write_text(json.dumps(curve[0], ensure_ascii=False) + '\n')
        _save_checkpoint(planned_checkpoints[arm]['initial'], model, arm,
                         training['seed'], 0, 0, study['model'], training, initial_scores)
        best_scores, best_epoch = initial_scores, None if fixed else 0
        last_scores = initial_scores
        evaluated_epochs = {0}

        def score_epoch(epoch, stage):
            nonlocal best_scores, best_epoch, last_scores
            scores, _ = _score_validation(
                model, data, encoder, selection_queries, device,
                int(selection_config['sample_count']),
                stable_seed(training['seed'], arm, 'selection', epoch))
            entry = {'stage': stage, 'epoch': epoch, 'step': step, **scores}
            curve.append(entry)
            with curve_path.open('a') as curve_file:
                curve_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
            last_scores = scores
            evaluated_epochs.add(epoch)
            if scores['prior_joint_nll_per_residue'] < best_scores['prior_joint_nll_per_residue']:
                best_scores, best_epoch = scores, epoch
                _save_checkpoint(planned_checkpoints[arm]['best'], model, arm,
                                 training['seed'], step, epoch, study['model'], training, scores)

        with metrics_path.open('w') as metrics:
            for epoch in range(1, max_epochs + 1):
                queries = data.training_queries(epoch - 1, int(training['seed']))
                if not queries:
                    raise ValueError('Training epoch produced no queries')
                for start in range(0, len(queries), int(training['batch_size'])):
                    if step >= int(training['max_steps']):
                        break
                    lr, beta = _aux_schedule(training, step + 1, epoch)
                    for group in optimizer.param_groups:
                        group['lr'] = lr
                    batch = queries[start:start + int(training['batch_size'])]
                    weights = torch.tensor([q['weight'] for q in batch],
                                           dtype=torch.float32, device=device)
                    generator = torch.Generator(device=device.type)
                    generator.manual_seed(stable_seed(training['seed'], arm, 'latent', step)
                                          % (2**63 - 1))
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                        enabled=device.type == 'cuda'):
                        loss, values = _train_step(model, data, batch, weights, beta, generator,
                                                   int(training['K']), encoder, device)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Nonfinite loss at {arm} step {step + 1}')
                    loss.backward()
                    norm = clip_grad_norm_(model.parameters(), float(training['gradient_clip']))
                    if not torch.isfinite(norm):
                        raise FloatingPointError(f'Nonfinite gradient at {arm} step {step + 1}')
                    optimizer.step()
                    step += 1
                    metrics.write(json.dumps({'arm': arm, 'epoch': epoch, 'step': step,
                                              'loss': float(loss.detach().item()), 'beta': beta,
                                              'learning_rate': lr, **values},
                                             ensure_ascii=False) + '\n')
                    metrics.flush()
                if not fixed:
                    interval = int(selection_config['interval_epochs'])
                    if interval > 0 and epoch % interval == 0:
                        score_epoch(epoch, 'epoch')
                if step >= int(training['max_steps']):
                    break
            if not fixed and epoch not in evaluated_epochs:
                score_epoch(epoch, 'last')
        if step != int(training['max_steps']):
            raise RuntimeError(f'Training ended at {step}, expected {training["max_steps"]}')
        _save_checkpoint(planned_checkpoints[arm]['last'], model, arm,
                         training['seed'], step, epoch, study['model'], training, last_scores)
        summaries[arm] = {
            'steps': step, 'epochs': epoch, 'seconds': time.monotonic() - started,
            'selection_metric': None if fixed else 'prior_joint_nll_per_residue',
            'checkpoint_rule': 'fixed_updates' if fixed else 'auxiliary_validation',
            'initial_validation': initial_scores, 'best_validation': best_scores,
            'best_epoch': best_epoch, 'last_validation': last_scores,
            'checkpoints': {kind: {'path': str(path), 'sha256': sha256_file(path)}
                            for kind, path in planned_checkpoints[arm].items()},
            'validation_curve': curve,
        }
        del model
        torch.cuda.empty_cache()
    summary = {
        'status': 'COMPLETED', 'device': str(device), 'seed': int(training['seed']),
        'train_records': len(data.train),
        'train_sequences': len({r['sequence'] for r in data.train}),
        'validation_sequences': len(data.validation),
        'train_panels': sum(len(data.panels[record['uid']]) for record in data.train),
        'validation_panels': sum(len(data.panels[record['uid']]) for record in data.validation),
        'nonstandard_domains': data.nonstandard_domains,
        'excluded_masked_positions': data.excluded_masked_positions,
        'mask_sizes': list(training['mask_sizes']), 'K': int(training['K']),
        'steps_per_arm': int(training['max_steps']), 'arms': summaries,
        'selection': {'method': 'fixed_updates', 'updates': int(training['max_steps'])} if fixed else {
            'records': len(selected_records), 'queries': len(selection_queries),
            'panel': selection_config['panel'], 'sample_count': int(selection_config['sample_count']),
            'interval_epochs': int(selection_config['interval_epochs']),
            'balance_key': study['generation']['balance_key']},
        'input_paths': data.input_paths,
        'input_domains': study['paths'].get('domains'), 'input_masks': study['paths'].get('masks'),
        'input_split': study['paths'].get('split'), 'esm_model': study['paths']['esm_model'],
        'fragment_condition': study.get('fragment_condition'),
        'input_sha256': input_hashes,
    }
    write_json(output / 'training_summary.json', summary)
    return summary


def _wt_panels(domain_length, generation, mask_sizes):
    panels = []
    for size in mask_sizes:
        for set_index in range(int(generation['wt_mask_sets'])):
            rng = np.random.default_rng(stable_seed(generation['seed'], 'wt_mask', size, set_index))
            positions = sorted(map(int, rng.choice(domain_length, int(size), replace=False)))
            panels.append({'panel': f'wt_mask_{size}_set{set_index:02d}', 'positions': positions})
    return panels


def generate_aux(config_path, checkpoint, output, device=None, limit=None, draws=None,
                 source='aux'):
    data = AuxiliaryData(config_path)
    study = data.config
    if source not in {'aux', 'wt_kvp40'}:
        raise ValueError(f'Unknown generation source: {source}')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Generation output already contains files: {output}')
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    payload = load_checkpoint(checkpoint, device='cpu')
    arm = payload.get('arm')
    if arm not in {'QVAE', 'deterministic'}:
        raise ValueError('Unsupported checkpoint arm')
    if int(payload['seed']) != int(study['training']['seed']):
        raise ValueError('Checkpoint seed differs from the configured auxiliary seed')
    model = build_model(payload, device=device)
    encoder = FrozenEncoder(study['paths']['esm_model'], device=device)
    generation = study['generation']
    full_sequence = data.target.sequence
    domain_start = int(generation['wt_domain_start'])
    domain_end = int(generation['wt_domain_end'])
    wt_domain = full_sequence[domain_start - 1:domain_end]
    if len(wt_domain) != domain_end - domain_start + 1:
        raise ValueError('WT domain slice is out of range')
    if source == 'aux':
        limit = int(limit if limit is not None else generation['limit'])
        selected = data.balanced_records(data.validation, limit, generation['balance_key'])
        backgrounds = [{'id': record['uid'], 'sequence': record['sequence'],
                        'panels': data.panels[record['uid']],
                        'mapping': {q: w for q, w in record['pairs']}}
                       for record in selected]
        draws_n = int(draws if draws is not None else generation['draws_per_mask'])
    else:
        panels = _wt_panels(len(wt_domain), generation, study['training']['mask_sizes'])
        backgrounds = [{'id': 'KVP40_WT_NMNAT_4_181', 'sequence': wt_domain,
                        'panels': panels,
                        'mapping': {position: domain_start + position
                                    for position in range(len(wt_domain))}}]
        draws_n = int(draws if draws is not None else generation['wt_draws_per_mask'])
    train_sequences = {record['sequence'] for record in data.train}
    rows = []
    anchor_changes = 0
    anchor_tests = 0
    panel_count = 0
    mask_entropy = []
    for background in backgrounds:
        for panel in background['panels']:
            if not panel['positions']:
                continue
            panel_count += 1
            background_id = background['id']
            prior_seed = stable_seed(generation['seed'], int(payload['seed']), source,
                                     background_id, panel['panel'], 'latent')
            uniform_seed = stable_seed(generation['seed'], int(payload['seed']), source,
                                       background_id, panel['panel'], 'residues')
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                                 enabled=device.type == 'cuda'):
                h, valid, mask = encoder([background['sequence']], [panel['positions']])
                context = model.encode_context(h, valid, mask)
                labels = torch.zeros(1, len(background['sequence']), dtype=torch.long,
                                     device=device)
                targets, _, _, _ = target_tensors(context, mask, labels)
                if arm == 'QVAE':
                    prior = model.prior(context, valid, mask)
                    generator = torch.Generator(device=device.type)
                    generator.manual_seed(prior_seed)
                    codes = categorical_samples(prior, draws_n, generator)
                    logits = model.decode_codes(targets, codes)[:, 0].float().cpu().numpy()
                    logp = categorical_log_prob(prior, codes)[:, 0].cpu().numpy()
                else:
                    latent = model.deterministic_context(context, valid, mask)
                    base_logits = model.decode(targets, latent.unsqueeze(0))[:, 0]
                    logits = np.repeat(base_logits.float().cpu().numpy(), draws_n, axis=0)
                    logp = [None] * draws_n
                uniforms = np.random.default_rng(uniform_seed).random(
                    (draws_n, len(panel['positions'])))
                letters = sample_letters(logits, uniforms, float(generation['temperature']))
            position_counts = defaultdict(Counter)
            masked_wt = [background['mapping'][position] for position in panel['positions']]
            is_mask16 = panel['panel'] == 'mask_16' or panel['panel'].startswith('wt_mask_16_')
            for draw in range(draws_n):
                chars = list(background['sequence'])
                edits = []
                for position, letter in zip(panel['positions'], letters[draw]):
                    letter = str(letter)
                    position_counts[position][letter] += 1
                    if background['sequence'][position] != letter:
                        edits.append({'query_position': position,
                                      'wt_position': background['mapping'][position],
                                      'from': background['sequence'][position], 'to': letter})
                    chars[position] = letter
                domain_sequence = ''.join(chars)
                full_chars = list(full_sequence)
                kvp40_edits = []
                for edit in edits:
                    wt_position = edit['wt_position']
                    wt_from = full_sequence[wt_position - 1]
                    full_chars[wt_position - 1] = edit['to']
                    kvp40_edits.append({
                        'position': wt_position, 'wt_from': wt_from, 'to': edit['to'],
                        'aux_from': edit['from'],
                        'direct_wt_substitution': wt_from == edit['from'],
                        'wt_changed': wt_from != edit['to'],
                    })
                kvp40_sequence = ''.join(full_chars)
                short_id = (background_id.split(':')[1][:10]
                            if background_id.startswith('aux:') else 'KVP40WT')
                rows.append({
                    'candidate_id': f"{arm}_s{payload['seed']}_{source}_{short_id}_"
                                    f"{panel['panel']}_d{draw:02d}",
                    'arm': arm, 'seed': int(payload['seed']), 'source': source,
                    'background_id': background_id,
                    'aux_uid': background_id if source == 'aux' else None,
                    'source_sha256': hashlib.sha256(background['sequence'].encode()).hexdigest(),
                    'source_sequence': background['sequence'],
                    'panel': panel['panel'], 'mask_size': len(panel['positions']), 'draw': draw,
                    'sequence': domain_sequence,
                    'sequence_sha256': hashlib.sha256(domain_sequence.encode()).hexdigest(),
                    'length': len(domain_sequence),
                    'kvp40_sequence': kvp40_sequence,
                    'kvp40_sequence_sha256': hashlib.sha256(kvp40_sequence.encode()).hexdigest(),
                    'kvp40_length': len(kvp40_sequence),
                    'mask_positions': json.dumps(panel['positions']),
                    'wt_positions': json.dumps(masked_wt),
                    'edits': json.dumps(edits, ensure_ascii=False),
                    'kvp40_edits': json.dumps(kvp40_edits, ensure_ascii=False),
                    'prior_seed': prior_seed, 'uniform_seed': uniform_seed,
                    'log_prior_z': float(logp[draw]) if logp[draw] is not None else None,
                    'novel_vs_aux_train': domain_sequence not in train_sequences,
                })
            for counts in position_counts.values():
                total = sum(counts.values())
                mask_entropy.append(-sum((count / total) * math.log(count / total)
                                         for count in counts.values() if count))
            if is_mask16:
                for index, position in enumerate(panel['positions']):
                    if background['mapping'][position] in ANCHORS:
                        anchor_tests += 1
                        if any(str(letters[draw][index]) != background['sequence'][position]
                               for draw in range(draws_n)):
                            anchor_changes += 1
    if not rows:
        raise ValueError('No auxiliary generation attempts were produced')
    mask_consistent = 0
    for row in rows:
        positions = set(json.loads(row['mask_positions']))
        if all(row['sequence'][i] == row['source_sequence'][i]
               for i in range(len(row['sequence'])) if i not in positions):
            mask_consistent += 1
    if mask_consistent != len(rows):
        raise ValueError('Generated sequence changed a non-masked position')
    with (output / 'candidates.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    with (output / 'candidates.fasta').open('w') as handle:
        for row in rows:
            handle.write(f">{row['candidate_id']}\n{row['sequence']}\n")
    with (output / 'candidates_kvp40.fasta').open('w') as handle:
        for row in rows:
            handle.write(f">{row['candidate_id']}\n{row['kvp40_sequence']}\n")
    summary = {
        'status': 'COMPLETED', 'source': source, 'arm': arm,
        'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256_file(checkpoint),
        'checkpoint_seed': int(payload['seed']),
        'backgrounds': len(backgrounds), 'panels': panel_count,
        'draws_per_mask': draws_n, 'attempts': len(rows),
        'unique_domain_sequences': len({row['sequence_sha256'] for row in rows}),
        'unique_kvp40_sequences': len({row['kvp40_sequence_sha256'] for row in rows}),
        'novel_vs_aux_train': sum(1 for row in rows if row['novel_vs_aux_train']),
        'rows_with_wt_change': sum(1 for row in rows if any(
            edit['wt_changed'] for edit in json.loads(row['kvp40_edits']))),
        'rows_with_direct_wt_change': sum(1 for row in rows if any(
            edit['direct_wt_substitution'] and edit['wt_changed']
            for edit in json.loads(row['kvp40_edits']))),
        'mask_consistent': mask_consistent,
        'mean_masked_position_entropy': float(np.mean(mask_entropy)) if mask_entropy else None,
        'anchor_tests_mask16': anchor_tests,
        'anchor_variants_mask16': anchor_changes,
        'seconds': time.monotonic() - started,
        'device': str(device),
        'wt_domain': {'start': domain_start, 'end': domain_end,
                      'length': len(wt_domain),
                      'sha256': hashlib.sha256(wt_domain.encode()).hexdigest()},
        'generation': {key: generation[key] for key in
                       ('seed', 'temperature', 'balance_key', 'limit', 'draws_per_mask',
                        'wt_draws_per_mask', 'wt_mask_sets')},
        'input_sha256': {
            'domains': sha256_file(study['paths']['domains']),
            'masks': sha256_file(study['paths']['masks']),
            'split': sha256_file(study['paths']['split']),
            'target': sha256_file(study['paths']['target']),
            'config': sha256_file(config_path),
        },
    }
    write_json(output / 'generation.json', summary)
    return summary
