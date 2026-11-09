"""Validation-fitted mixtures of frozen ESM and existing task-head site probabilities."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import PopulationData, load_yaml, mask_panels, sha256_file, write_json
from .esm_baseline import _checked_query_rows, _metric_components
from .evaluate import _bootstrap
from .fragment_comparison import _input_hashes, _require, _stratified_components
from .model import AA
from .test_data import load_recovered_test

REGIONS = ('NMNAT', 'Nudix', 'global', 'linker', 'random', 'all_panels')


def _validation_input_paths(config_path, config):
    data = PopulationData(config['sources'], config['study'], config['target'])
    paths = [config_path, config['sources'], config['study'], config['target'],
             data.sources['population'], *[data.sources['splits'][s] for s in ('training', 'validation')]]
    paths += [Path(config['esm']['validation']) / name for name in
              ('identity.json', 'evaluation.json', 'per_query.jsonl')]
    for spec in config['models'].values():
        for seed in config['seeds']:
            run = spec['prefix'] + str(seed)
            paths += [Path(spec['evaluation']) / 'validation' / run / name for name in
                      ('identity.json', 'evaluation.json', 'per_query.jsonl')]
            training = Path(spec['training']) / run / 'training_summary.json'
            paths += [training, json.loads(training.read_text())['checkpoints']['best']['path']]
    paths += [Path(config['neighbors']) / f'{region}_both_nearest.csv'
              for region in ('nmnat', 'nudix', 'full_length')]
    return {str(path) for path in paths}


def mixture_nll(esm_logp, task_logp, alphas):
    """Mix probabilities at each site, then average site NLL and MC repeats."""
    esm, task, alphas = map(lambda x: np.asarray(x, dtype=float), (esm_logp, task_logp, alphas))
    _require(esm.ndim == 1 and task.ndim == 2 and task.shape[1] == len(esm) and len(esm) > 0,
             'Site probability shapes differ')
    _require(np.isfinite(esm).all() and np.isfinite(task).all() and
             (esm <= 1e-6).all() and (task <= 1e-6).all(), 'Invalid log probabilities')
    _require(alphas.ndim == 1 and len(alphas) > 0 and np.isfinite(alphas).all() and
             ((alphas >= 0) & (alphas <= 1)).all(), 'Invalid mixture weights')
    with np.errstate(divide='ignore'):
        mixed = np.logaddexp(np.log1p(-alphas)[:, None, None] + esm[None, None, :],
                             np.log(alphas)[:, None, None] + task[None, :, :])
    return -mixed.mean(axis=(1, 2))


def calibrate(group_grid):
    """Choose shared-across-seed weights; leave each component out of weight fitting."""
    _require(group_grid.ndim == 4 and group_grid.shape[2] > 1 and np.isfinite(group_grid).all(),
             'Calibration requires seed × alpha × component × region scores')
    loss = group_grid.mean(axis=0)
    full = loss.mean(axis=1).argmin(axis=0)
    excluded = (loss.sum(axis=1)[:, None, :] - loss) / (loss.shape[1] - 1)
    return full, excluded.argmin(axis=0)


def _group_grid(cube, rows):
    groups = sorted({row['component'] for row in rows})
    result = np.empty((*cube.shape[:2], len(groups), len(REGIONS)))
    for g, group in enumerate(groups):
        for r, region in enumerate(REGIONS):
            indices = [i for i, row in enumerate(rows) if row['component'] == group and
                       (region == 'all_panels' or row['region'] == region)]
            _require(bool(indices), f'Missing component/region: {group}/{region}')
            weights = [rows[i]['weight'] for i in indices] if region == 'all_panels' else None
            result[:, :, g, r] = np.average(cube[:, :, indices], axis=2, weights=weights)
    return groups, result


def _selected_queries(cube, rows, indices, groups, global_weight=False):
    # Region selection happens before the all-panel weighted aggregation.
    by_group = {group: i for i, group in enumerate(groups)}
    choices = [int(indices[by_group[row['component']], -1 if global_weight else
                           REGIONS.index(row['region'])]) for row in rows]
    return cube[:, choices, np.arange(len(rows))]


def _load_split(config, split, hashes):
    def digest(path):
        path = str(path)
        if path not in hashes:
            hashes[path] = sha256_file(path)
        return hashes[path]

    def read(path):
        digest(path)
        return json.loads(Path(path).read_text())

    data = PopulationData(config['sources'], config['study'], config['target'])
    records = data.validation if split == 'validation' else load_recovered_test(data, config['test_data'])
    expected = {}
    for record in records:
        for panel in mask_panels(record, data.study['training']['mask_seed']):
            key = record.sequence_id, panel['panel']
            expected[key] = {'sequence_id': record.sequence_id, 'component': record.component,
                             'panel': panel['panel'], 'region': panel['region'],
                             'positions_0based': panel['positions'], 'weight': panel['weight']}
    keys = sorted(expected)
    rows = [expected[key] for key in keys]
    sizes = Counter(record.component for record in records)
    common = [config['sources'], config['study'], config['target'], data.sources['population'],
              data.sources['splits']['training'], data.sources['splits']['validation']]
    if split == 'test':
        common.extend(Path(config['test_data']) / name for name in ('test.csv', 'test.fasta', 'manifest.json'))
        history = read(Path(config['test_data']) / 'historical_use.json')
        _require(history['status'] == 'previously_evaluated', 'Missing held-out evaluation history')
    esm_root = Path(config['esm'][split])
    esm_identity, esm_summary = (read(esm_root / name) for name in ('identity.json', 'evaluation.json'))
    _require(esm_identity['split'] == split and esm_identity['amino_acid_order'] == AA and
             esm_summary['status'] == 'COMPLETED' and esm_summary['queries'] == len(keys),
             'ESM evaluation identity differs')
    _input_hashes(esm_identity, common, digest)
    esm_files = {Path(path).name: value for path, value in esm_identity['input_sha256'].items()
                 if Path(path).parent == Path(data.sources['esm_model'])}
    _require('model.safetensors' in esm_files, 'ESM checkpoint identity missing')
    esm_path = esm_root / 'per_query.jsonl'
    _checked_query_rows(esm_path, expected, sizes)
    digest(esm_path)
    esm = { (row['sequence_id'], row['panel']): row['target_site_log_probs']
            for row in map(json.loads, esm_path.read_text().splitlines()) }
    cubes, checkpoints = {}, {}
    for arm, spec in config['models'].items():
        values = []
        for seed in config['seeds']:
            run = spec['prefix'] + str(seed)
            root = Path(spec['evaluation']) / split / run
            identity, summary = (read(root / name) for name in ('identity.json', 'evaluation.json'))
            training = read(Path(spec['training']) / run / 'training_summary.json')
            best = training['checkpoints']['best']
            _require(training['initialization'] == {'mode': 'random'} and training['seed'] == seed and
                     training.get('arm', 'QVAE') == arm and training['status'] == 'COMPLETED',
                     f'Expected selected random-initialization model: {run}')
            _require(summary['checkpoint_sha256'] == best['sha256'] == digest(best['path']) and
                     summary['status'] == 'COMPLETED' and identity['split'] == split and
                     identity.get('arm', 'QVAE') == arm and identity['limit'] is None and
                     identity['esm_files_sha256'] == esm_files and
                     identity['precision'].split(' forward')[0] == esm_identity['precision'].split(' forward')[0],
                     f'Model evaluation identity differs: {run}')
            _input_hashes(identity, [*common, best['path']], digest)
            repeats = summary['repeats']
            query_path = root / 'per_query.jsonl'
            checked, _ = _checked_query_rows(query_path, expected, sizes, repeats)
            digest(query_path)
            by_key = defaultdict(dict)
            for row in map(json.loads, query_path.read_text().splitlines()):
                key = row['sequence_id'], row['panel']
                logp = row['prior_site_marginal_log_probs']
                _require(np.isclose(-np.mean(logp), row['marginal_nll_per_residue'], atol=3e-6),
                         f'Site probabilities disagree with NLL: {run}/{key}')
                by_key[key][row['repeat']] = logp
            scores = np.stack([mixture_nll(esm[key], [by_key[key][r] for r in range(repeats)],
                                          config['alpha_grid']) for key in keys], axis=1)
            _require(np.allclose(scores[-1], [row['marginal_nll_per_residue'] for row in checked],
                                 rtol=0, atol=3e-6), 'Task-only mixture endpoint changed')
            values.append(scores)
            checkpoints[f'{arm}:{seed}'] = best['sha256']
        cubes[arm] = np.stack(values)
    neighbors = {}
    for region, name in [('NMNAT', 'nmnat'), ('Nudix', 'nudix'), ('full_length', 'full_length')]:
        path = Path(config['neighbors']) / f'{name}_both_nearest.csv'
        digest(path)
        neighbors[region] = {row['sequence_id']: float(row['fident']) if row['target_id'] else None
                             for row in csv.DictReader(path.open())
                             if row['role'] == split and row['target_role'] == 'training'}
        _require(set(neighbors[region]) == {record.sequence_id for record in records},
                 'Nearest-neighbor members differ')
    return rows, cubes, neighbors, checkpoints


def _csv(path, rows):
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def temperature_targets(log_probs, labels, temperatures):
    """Renormalize all 20 amino acids before selecting the observed label."""
    log_probs = np.asarray(log_probs, dtype=float)
    labels = np.asarray(labels)
    temperatures = np.asarray(temperatures, dtype=float)
    _require(log_probs.shape == (len(labels), 20) and np.isfinite(log_probs).all() and
             np.allclose(np.logaddexp.reduce(log_probs, axis=-1), 0, atol=2e-6, rtol=0),
             'Expected normalized finite 20AA log probabilities')
    _require(labels.dtype.kind in 'iu' and ((labels >= 0) & (labels < 20)).all() and
             temperatures.ndim == 1 and np.isfinite(temperatures).all() and (temperatures > 0).all(),
             'Invalid labels or temperatures')
    return np.stack([(log_probs / t)[np.arange(len(labels)), labels] -
                     np.logaddexp.reduce(log_probs / t, axis=-1) for t in temperatures])


def calibrated_predictions(targets, boundaries, rows, temperatures, alphas, frozen=None):
    """Fit T then alpha within each training fold; never use another fold's fitted T."""
    targets = np.asarray(targets, dtype=float)  # model (ESM first), T, site
    temperatures, alphas = np.asarray(temperatures), np.asarray(alphas)
    boundaries = np.asarray(boundaries)
    _require(targets.ndim == 3 and targets.shape[1] == len(temperatures) and
             len(boundaries) == len(rows) + 1 and boundaries[0] == 0 and
             boundaries[-1] == targets.shape[-1] and (np.diff(boundaries) > 0).all(),
             'Calibration arrays do not align')
    unit = np.flatnonzero(temperatures == 1)
    _require(len(unit) == 1 and alphas[0] == 0 and alphas[-1] == 1 and (np.diff(alphas) > 0).all(),
             'Calibration grids must include T=1 and ordered alpha endpoints')
    unit = int(unit[0])
    query = -np.add.reduceat(targets, boundaries[:-1], axis=-1) / np.diff(boundaries)
    seeds = len(targets) - 1
    groups, task_grid = _group_grid(query[1:], rows)
    _, esm_grid = _group_grid(np.repeat(query[:1], seeds, axis=0), rows)
    baseline = {'esm': np.repeat(query[:1, unit], seeds, axis=0), 'task': query[1:, unit]}
    selected = {**baseline}
    for scheme in ('single', 'region'):
        for method in ('calibrated_esm', 'calibrated_task', 'mixture', 'calibrated_mixture'):
            selected[f'{scheme}_{method}'] = np.full((seeds, len(rows)), np.nan)
    cache = {}

    def mixture(te, tt):
        key = te, tt
        if key not in cache:
            scores = []
            with np.errstate(divide='ignore'):
                for task in targets[1:, tt]:
                    mixed = np.logaddexp(np.log1p(-alphas)[:, None] + targets[0, te],
                                         np.log(alphas)[:, None] + task)
                    scores.append(-np.add.reduceat(mixed, boundaries[:-1], axis=1) / np.diff(boundaries))
            cube = np.asarray(scores)
            cache[key] = cube, _group_grid(cube, rows)[1]
        return cache[key]

    raw_cube, raw_grid = mixture(unit, unit)
    folds = [None] if frozen is not None else [None, *range(len(groups))]
    fits = {}
    for held in folds:
        train = np.arange(len(groups)) != held if held is not None else np.ones(len(groups), dtype=bool)
        fold = {}
        for scheme in ('single', 'region'):
            region_ids = [len(REGIONS) - 1] if scheme == 'single' else list(range(len(REGIONS) - 1))
            fold[scheme] = {}
            for r in region_ids:
                region = REGIONS[r]
                if frozen is None:
                    te = int(esm_grid[:, :, train, r].mean(axis=(0, 2)).argmin())
                    tt = int(task_grid[:, :, train, r].mean(axis=(0, 2)).argmin())
                    cube, grid = mixture(te, tt)
                    a = int(grid[:, :, train, r].mean(axis=(0, 2)).argmin())
                    raw_a = int(raw_grid[:, :, train, r].mean(axis=(0, 2)).argmin())
                    fit = {'esm_temperature_index': te, 'task_temperature_index': tt,
                           'alpha_index': a, 'uncalibrated_alpha_index': raw_a,
                           'esm_temperature': float(temperatures[te]),
                           'task_temperature': float(temperatures[tt]),
                           'alpha': float(alphas[a]), 'uncalibrated_alpha': float(alphas[raw_a])}
                else:
                    fit = frozen[scheme][region]
                    te, tt, a, raw_a = [fit[k] for k in ('esm_temperature_index', 'task_temperature_index',
                                                       'alpha_index', 'uncalibrated_alpha_index')]
                    _require(all(type(v) is int for v in (te, tt, a, raw_a)) and
                             0 <= te < len(temperatures) and 0 <= tt < len(temperatures) and
                             0 <= a < len(alphas) and 0 <= raw_a < len(alphas), 'Invalid frozen indices')
                    cube, _ = mixture(te, tt)
                fold[scheme][region] = fit
                if held is not None or frozen is not None:
                    indices = [i for i, row in enumerate(rows) if
                               (held is None or row['component'] == groups[held]) and
                               (scheme == 'single' or row['region'] == region)]
                    selected[f'{scheme}_calibrated_esm'][:, indices] = query[0, te, indices]
                    selected[f'{scheme}_calibrated_task'][:, indices] = query[1:, tt][:, indices]
                    selected[f'{scheme}_mixture'][:, indices] = raw_cube[:, raw_a, indices]
                    selected[f'{scheme}_calibrated_mixture'][:, indices] = cube[:, a, indices]
        fits['full' if held is None else groups[held]] = fold
    _require(all(np.isfinite(value).all() for value in selected.values()), 'Incomplete calibration predictions')
    return selected, fits


def _probability_inputs(config, split):
    """A complete small manifest binds the export and the protocol used to produce it."""
    root = Path(config['probabilities'][split])
    identity = json.loads((root / 'identity.json').read_text())
    paths = {str(root / name) for name in ('identity.json', 'queries.json', 'site_log_probs.npz')}
    paths.update(identity['input_sha256'])
    return paths


def _read_probabilities(config, split, temperatures):
    spec = load_yaml(config['mixture_config'])
    root = Path(config['probabilities'][split])
    identity = json.loads((root / 'identity.json').read_text())
    names = ['esm', *map(str, spec['seeds'])]
    _require(identity['status'] == 'COMPLETED' and identity['split'] == split and
             identity['amino_acid_order'] == AA and identity['model_order'] == names,
             'Probability export identity differs')
    for name in ('queries.json', 'site_log_probs.npz'):
        _require(sha256_file(root / name) == identity['output_sha256'][name], 'Export output changed')
    for path, digest in identity['input_sha256'].items():
        _require(sha256_file(path) == digest, f'Export input changed: {path}')
    data = PopulationData(spec['sources'], spec['study'], spec['target'])
    expected_inputs = {config['mixture_config'], spec['sources'], spec['study'], spec['target'],
                       data.sources['population'], data.sources['splits']['training'],
                       data.sources['splits']['validation']}
    model_spec = spec['models']['deterministic']
    for seed in spec['seeds']:
        training_path = Path(model_spec['training']) / (model_spec['prefix'] + str(seed)) / 'training_summary.json'
        training = json.loads(training_path.read_text())
        _require(training['status'] == 'COMPLETED' and training['arm'] == 'deterministic' and
                 training['seed'] == seed and training['initialization'] == {'mode': 'random'},
                 'Export checkpoint selection differs')
        checkpoint = training['checkpoints']['best']
        expected_inputs.update((str(training_path), checkpoint['path']))
        _require(identity['input_sha256'].get(checkpoint['path']) == checkpoint['sha256'],
                 'Export checkpoint hash differs')
    if split == 'test':
        expected_inputs.update(str(Path(spec['test_data']) / name) for name in
                               ('test.csv', 'test.fasta', 'manifest.json', 'historical_use.json'))
    _require(set(identity['input_sha256']) == expected_inputs, 'Incomplete export input manifest')
    records = data.validation if split == 'validation' else load_recovered_test(data, spec['test_data'])
    expected = [{'sequence_id': rec.sequence_id, 'component': rec.component, 'panel': panel['panel'],
                 'region': panel['region'], 'positions_0based': panel['positions'], 'weight': panel['weight']}
                for rec in records for panel in mask_panels(rec, data.study['training']['mask_seed'])]
    rows = json.loads((root / 'queries.json').read_text())
    _require(rows == expected and identity['queries'] == len(rows), 'Export tasks differ')
    by_id = {record.sequence_id: record.sequence for record in records}
    labels = np.array([AA.index(by_id[row['sequence_id']][p]) for row in rows for p in row['positions_0based']])
    boundaries = np.cumsum([0, *[len(row['positions_0based']) for row in rows]])
    targets = []
    parity = {}
    with np.load(root / 'site_log_probs.npz', allow_pickle=False) as archive:
        _require(set(archive.files) == {*names, 'labels', 'boundaries'} and
                 np.array_equal(archive['labels'], labels) and np.array_equal(archive['boundaries'], boundaries),
                 'Export labels or boundaries differ')
        for name in names:
            targets.append(temperature_targets(archive[name], labels, temperatures))
            if name == 'esm':
                source = Path(spec['esm'][split]) / 'per_query.jsonl'
                field = 'target_site_log_probs'
            else:
                model = spec['models']['deterministic']
                source = Path(model['evaluation']) / split / (model['prefix'] + name) / 'per_query.jsonl'
                field = 'prior_site_marginal_log_probs'
            stored = {(r['sequence_id'], r['panel']): r[field]
                      for r in map(json.loads, source.read_text().splitlines())}
            original = np.concatenate([stored[(r['sequence_id'], r['panel'])] for r in rows])
            actual = archive[name][np.arange(len(labels)), labels]
            parity[name] = float(np.max(np.abs(original - actual)))
            _require(parity[name] <= 3e-6, f'Shared forward differs from existing scores: {name}')
    return spec, rows, np.asarray(targets), boundaries, parity, identity


def run_head_calibration(config_path, output, fitted_path=None):
    config = load_yaml(config_path)
    temperatures = np.asarray(config['temperatures'], dtype=float)
    _require(len(temperatures) > 1 and (np.diff(temperatures) > 0).all(), 'Invalid temperature grid')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    fitted = json.loads(Path(fitted_path).read_text()) if fitted_path else None
    implementation = sha256_file(__file__)
    validation_paths = _probability_inputs(config, 'validation') | {str(config_path), config['mixture_config']}
    if fitted:
        _require(fitted['status'] == 'FITTED_ON_VALIDATION' and fitted['fit_split'] == 'validation' and
                 fitted['implementation_sha256'] == implementation and
                 set(fitted['input_sha256']) == validation_paths, 'Calibration protocol or manifest differs')
        for path, digest in fitted['input_sha256'].items():
            _require(sha256_file(path) == digest, f'Calibration input changed: {path}')
    split = 'test' if fitted else 'validation'
    spec, rows, targets, boundaries, parity, identity = _read_probabilities(config, split, temperatures)
    if fitted:
        _require(identity['esm_files_sha256'] == fitted['esm_files_sha256'], 'ESM checkpoint changed')
        validation_identity = json.loads((Path(config['probabilities']['validation']) / 'identity.json').read_text())
        for field in ('implementation_sha256', 'precision', 'definition'):
            _require(identity[field] == validation_identity[field], f'Probability export {field} changed')
        model_spec = spec['models']['deterministic']
        for seed in spec['seeds']:
            path = str(Path(model_spec['training']) / (model_spec['prefix'] + str(seed)) / 'training_summary.json')
            checkpoint = json.loads(Path(path).read_text())['checkpoints']['best']['path']
            for item in (path, checkpoint):
                _require(identity['input_sha256'][item] == validation_identity['input_sha256'][item],
                         f'Training input changed for seed {seed}: {item}')
    scores, fits = calibrated_predictions(targets, boundaries, rows, temperatures, spec['alpha_grid'],
                                         fitted['weights'] if fitted else None)
    groups = sorted({row['component'] for row in rows})
    aggregates = {method: _group_grid(values[:, None], rows)[1][:, 0] for method, values in scores.items()}
    contrasts = [('task', 'esm')]
    for scheme in ('single', 'region'):
        contrasts += [(f'{scheme}_calibrated_task', f'{scheme}_calibrated_esm')]
        contrasts += [(f'{scheme}_calibrated_{head}', head) for head in ('esm', 'task')]
        contrasts += [(f'{scheme}_mixture', head) for head in ('esm', 'task')]
        contrasts += [(f'{scheme}_calibrated_mixture', f'{scheme}_calibrated_{head}') for head in ('esm', 'task')]
        contrasts += [(f'{scheme}_calibrated_mixture', f'{scheme}_mixture')]
    contrasts += [(f'region_{method}', f'single_{method}') for method in
                  ('calibrated_esm', 'calibrated_task', 'mixture', 'calibrated_mixture')]
    effects = {}
    for method, reference in contrasts:
        delta = aggregates[method] - aggregates[reference]
        effects[f'{method}_minus_{reference}'] = {
            region: {**_bootstrap(delta[:, :, r].mean(0), spec['bootstrap_repeats'], spec['bootstrap_seed']),
                     'method_mean': float(aggregates[method][:, :, r].mean()),
                     'reference_mean': float(aggregates[reference][:, :, r].mean()),
                     'per_seed': delta[:, :, r].mean(1).tolist()} for r, region in enumerate(REGIONS)}
    paths = _probability_inputs(config, split) | {str(config_path), config['mixture_config']}
    if fitted_path:
        paths.add(str(fitted_path))
    hashes = {p: sha256_file(p) for p in sorted(paths)}
    result = {'status': 'COMPLETED', 'split': split, 'input_sha256': hashes,
              'implementation_sha256': implementation, 'effects': effects,
              'esm_files_sha256': identity['esm_files_sha256'], 'baseline_max_abs_logp_difference': parity,
              'protocol': 'Fit T for ESM and seed-averaged deterministic head, then alpha on the same training components; single all_panels objective or separate five-region objectives. Validation excludes the entire scored component from both fits.',
              'aggregation': 'Equal components; equal queries within each region; panel-weighted all_panels. Per-query mean target NLL in nats.',
              'limits': 'Exploratory after six-cell results. Original checkpoint selection used validation; test has prior use. Conditional component bootstrap does not refit heads or calibration parameters. Temperature grid cannot exclude every possible calibration explanation. Temperatures optimize single heads before alpha, not a joint optimum; mixture distributions are not themselves temperature-calibrated.',
              'temperatures': temperatures.tolist(), 'alpha_grid': spec['alpha_grid'],
              'weights': fits['full'], 'excluded_component_weights': {g: w for g, w in fits.items() if g != 'full'}}
    output.mkdir(parents=True, exist_ok=True)
    _csv(output / 'group_scores.csv', [
        {'method': method, 'seed': seed, 'component': group, 'region': region, 'nll': float(values[s, g, r])}
        for method, values in aggregates.items() for s, seed in enumerate(spec['seeds'])
        for g, group in enumerate(groups) for r, region in enumerate(REGIONS)])
    if not fitted:
        write_json(output / 'fitted_weights.json', {k: result[k] for k in
                   ('input_sha256', 'implementation_sha256', 'esm_files_sha256', 'weights')} |
                   {'status': 'FITTED_ON_VALIDATION', 'fit_split': 'validation'})
    lines = ['# 温度校准与预测概率互补', '', result['protocol'], '', result['aggregation'], '', result['limits'], '',
             '| 对比 | 区域 | NLL差 | 95%条件组区间 |', '|---|---|---:|---|']
    for contrast, regions in effects.items():
        for region in ('NMNAT', 'Nudix', 'all_panels'):
            value = regions[region]
            lines.append(f'| {contrast} | {region} | {value["mean"]:+.6f} | {value["ci95"]} |')
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    write_json(output / 'summary.json', result)
    return {'status': 'COMPLETED', 'split': split, 'output': str(output)}


def run_head_mixture(config_path, output, fitted_path=None):
    config = load_yaml(config_path)
    alphas = np.asarray(config['alpha_grid'], dtype=float)
    _require(len(alphas) > 1 and alphas[0] == 0 and alphas[-1] == 1 and
             (np.diff(alphas) > 0).all(), 'Alpha grid must increase from zero to one')
    _require(set(config['models']) == {'QVAE', 'deterministic'} and
             len(set(config['seeds'])) == len(config['seeds']) == 3, 'Expected two arms and three seeds')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    hashes = {str(config_path): sha256_file(config_path)}
    fitted = None
    if fitted_path:
        fitted = json.loads(Path(fitted_path).read_text())
        _require(fitted['status'] == 'FITTED_ON_VALIDATION' and
                 fitted.get('fit_split') == 'validation' and
                 fitted['config_sha256'] == hashes[str(config_path)] and
                 fitted['implementation_sha256'] == sha256_file(__file__), 'Fitted mixture protocol differs')
        _require(set(fitted['input_sha256']) == _validation_input_paths(config_path, config),
                 'Calibration input manifest is incomplete')
        for path, digest in fitted['input_sha256'].items():
            _require(sha256_file(path) == digest, f'Calibration input changed: {path}')
        hashes[str(fitted_path)] = sha256_file(fitted_path)
    split = 'test' if fitted else 'validation'
    rows, cubes, neighbors, checkpoints = _load_split(config, split, hashes)
    if fitted:
        _require(checkpoints == fitted['checkpoints'], 'Held-out checkpoints differ from calibration')
    group_rows, strata_rows, effects, weights = [], [], {}, {}
    for arm, cube in cubes.items():
        groups, grid = _group_grid(cube, rows)
        if fitted:
            raw_indices = fitted['weights'][arm]['full_indices']
            _require(len(raw_indices) == len(REGIONS) and
                     all(type(i) is int and 0 <= i < len(alphas) for i in raw_indices),
                     'Invalid fitted mixture indices')
            full = np.asarray(raw_indices, dtype=int)
            selected = np.tile(full, (len(groups), 1))
        else:
            full, selected = calibrate(grid)
            weights[arm] = {'full_indices': full.tolist(), 'full_alphas': dict(zip(REGIONS, alphas[full].tolist())),
                            'excluded_group_indices': dict(zip(groups, selected.tolist()))}
        distributions = {'esm': cube[:, 0], 'task_marginal': cube[:, -1],
                         'global_mixture': _selected_queries(cube, rows, selected, groups, True),
                         'region_mixture': _selected_queries(cube, rows, selected, groups)}
        scores = {}
        for method, query_scores in distributions.items():
            seed_scores = []
            for i, seed in enumerate(config['seeds']):
                scored = [{**row, 'nll': float(value)} for row, value in zip(rows, query_scores[i])]
                component = _metric_components(scored, 'nll')
                seed_scores.append(component)
                group_rows.extend({'arm': arm, 'method': method, 'seed': seed, 'component': group,
                                   'region': region, 'nll': component[group][region]}
                                  for group in groups for region in REGIONS)
                strata_rows.extend({'arm': arm, 'method': method, 'seed': seed, **row}
                                   for row in _stratified_components(scored, neighbors, 'nll'))
            scores[method] = np.asarray([[component[group][region] for group in groups for region in REGIONS]
                                        for component in seed_scores]).reshape(len(config['seeds']), len(groups), -1)
        effects[arm] = {}
        for method in ('task_marginal', 'global_mixture', 'region_mixture'):
            for reference in ('esm', 'task_marginal', 'global_mixture'):
                if method == reference:
                    continue
                if reference == 'global_mixture' and method != 'region_mixture':
                    continue
                difference = (scores[method] - scores[reference]).mean(axis=0)
                effects[arm][f'{method}_minus_{reference}'] = {
                    region: {**_bootstrap(difference[:, r], config['bootstrap_repeats'], config['bootstrap_seed']),
                             'method_mean': float(scores[method][:, :, r].mean()),
                             'reference_mean': float(scores[reference][:, :, r].mean()),
                             'per_seed': (scores[method][:, :, r] - scores[reference][:, :, r]).mean(axis=1).tolist()}
                    for r, region in enumerate(REGIONS)}
    output.mkdir(parents=True, exist_ok=True)
    _csv(output / 'group_scores.csv', group_rows)
    _csv(output / 'stratified_group_scores.csv', strata_rows)
    result = {'status': 'COMPLETED', 'split': split, 'config_sha256': hashes[str(config_path)],
              'completed_utc': datetime.now(timezone.utc).isoformat(),
              'input_sha256': hashes, 'implementation_sha256': sha256_file(__file__),
              'checkpoints': checkpoints, 'effects': effects,
              'probability_definition': '(1-alpha)*p_ESM(site) + alpha*p_task_marginal(site); product across sites',
              'mc_definition': 'Compute mixture NLL in each stored repeat then average NLL; no new samples',
              'aggregation': 'Equal components; equal queries within component/region; panel-weighted queries for all_panels. Single global alpha optimizes all_panels, not equal regions.',
              'normalization_check': 'Stored target-label probabilities alone cannot verify full 20AA normalization; new complete-distribution export checks it at every site.',
              'interpretation': 'Exploratory complementarity diagnostic, not an architecture-causal or independent-generalization test',
              'calibration': 'Validation leave-one-component-out weights only; original checkpoints were selected using validation',
              'uncertainty': 'Equal component bootstrap conditional on fitted heads, weights and stored MC draws',
              'identity_use': 'Descriptive strata only; full-query nearest identity is never a mixture predictor'}
    write_json(output / 'summary.json', result)
    if not fitted:
        write_json(output / 'fitted_weights.json', {
            **{key: result[key] for key in ('config_sha256', 'input_sha256', 'implementation_sha256',
                                            'completed_utc', 'checkpoints')},
            'status': 'FITTED_ON_VALIDATION', 'fit_split': 'validation', 'weights': weights})
    lines = ['# ESM 与 NatV 任务头的概率互补诊断', '',
             f'划分：{split}。验证阶段仅对混合权重做留一component交叉验证；任务模型的既有checkpoint使用过验证集。',
             '保留集已有评价历史。本实验是在查看六格结果后设计，不能称为独立确认，也不单独证明预测头替换导致信息丢失。',
             '混合逐位点概率后取对数，不混合NLL；QVAE使用自身边缘，未改变共享潜变量。', '',
             '各MC重复分别混合并计算NLL，再平均NLL；不是先跨重复平均概率。区域内等query平均，再等component平均；all_panels使用panel权重。单一global_mixture权重按all_panels目标拟合，不是五区域等权。旧产物只存目标标签概率，不能据此检查完整20AA归一化。', '',
             '| 模型 | 对比 | 区域 | NLL差 | 95%组区间 |', '|---|---|---|---:|---|']
    for arm, comparisons in effects.items():
        for comparison, regions in comparisons.items():
            for region in ('NMNAT', 'Nudix', 'all_panels'):
                value = regions[region]
                lines.append(f'| {arm} | {comparison} | {region} | {value["mean"]:+.6f} | '
                             f'[{value["ci95"][0]:+.6f}, {value["ci95"][1]:+.6f}] |')
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return {'status': result['status'], 'split': split, 'output': str(output)}
