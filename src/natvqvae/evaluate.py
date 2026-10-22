"""Validation evaluation for the QVAE checkpoint."""
from __future__ import annotations

import json
import math
import time
import csv
import hashlib
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from .data import PopulationData, sha256_file, stable_seed, write_json
from .model import (FrozenEncoder, build_model, categorical_kl, categorical_log_prob,
                    categorical_samples, load_checkpoint, reconstruction_nll, target_tensors)


@torch.no_grad()
def conditional_scores(model, context_features, full_features, valid, mask, labels,
                       sample_count, generator, chunk_size=16):
    """Score the actual categorical mixture, separately from expected CE.

    Returned NLLs sum target sites for each input. Posterior IS and direct-prior
    Monte Carlo estimate the same shared-z distribution; neither is an exact
    finite-sample log likelihood. The marginal reference uses the prior samples
    but averages each site's probability before taking the product.
    """
    if sample_count < 1 or chunk_size < 1:
        raise ValueError('Probability estimation requires positive sample/chunk counts')
    hidden = model.encode_context(context_features, valid, mask)
    targets, y, use, _ = target_tensors(hidden, mask, labels)
    sites = use.sum(1)
    if model.arm == 'deterministic':
        latent = model.deterministic_context(hidden, valid, mask)
        nll = reconstruction_nll(model.decode(targets, latent.unsqueeze(0)), y, use)[0]
        return {'sites': sites, 'prior_joint_nll': nll, 'posterior_joint_nll': nll,
                'marginal_nll': nll, 'prior_expected_reconstruction_nll': nll,
                'posterior_expected_reconstruction_nll': nll,
                'kl': torch.zeros_like(nll), 'ess': None, 'max_weight': None}
    if full_features is None:
        raise ValueError('Posterior importance scoring requires full target features')
    prior = model.prior(hidden, valid, mask)
    posterior = model.posterior(full_features, hidden, valid, mask)
    codes_prior = categorical_samples(prior, sample_count, generator)
    codes_post = categorical_samples(posterior, sample_count, generator)

    def target_log_probs(codes):
        values = []
        for start in range(0, sample_count, chunk_size):
            logits = model.decode_codes(targets, codes[start:start + chunk_size])
            index = y[None, ..., None].expand(len(logits), -1, -1, 1)
            values.append(logits.float().log_softmax(-1).gather(-1, index).squeeze(-1))
        return torch.cat(values, 0)

    prior_site_logp = target_log_probs(codes_prior)
    prior_logp = (prior_site_logp * use).sum(-1)
    post_logp = (target_log_probs(codes_post) * use).sum(-1)
    logw = (post_logp + categorical_log_prob(prior, codes_post)
            - categorical_log_prob(posterior, codes_post))
    normalized_weights = logw.softmax(0)
    return {
        'sites': sites,
        'prior_joint_nll': -(torch.logsumexp(prior_logp, 0) - math.log(sample_count)),
        'posterior_joint_nll': -(torch.logsumexp(logw, 0) - math.log(sample_count)),
        'marginal_nll': -((torch.logsumexp(prior_site_logp, 0)
                           - math.log(sample_count)) * use).sum(-1),
        'prior_expected_reconstruction_nll': -prior_logp.mean(0),
        'posterior_expected_reconstruction_nll': -post_logp.mean(0),
        'kl': categorical_kl(posterior, prior),
        'ess': normalized_weights.square().sum(0).reciprocal(),
        'max_weight': normalized_weights.max(0).values,
    }


def evaluate(config_path, sources_path, target_path, checkpoint, output, device=None,
             split='validation', limit=None):
    if split != 'validation':
        raise ValueError('Only the validation split is evaluated; test remains sealed')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Evaluation output already contains files: {output}')
    data = PopulationData(sources_path, config_path, target_path)
    study = data.study
    config = study['evaluation']
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    payload = load_checkpoint(checkpoint, device='cpu')
    if payload.get('arm') != 'QVAE':
        raise ValueError('Evaluation requires a QVAE checkpoint')
    model = build_model(payload, device=device)
    encoder = FrozenEncoder(data.sources['esm_model'], device=device)
    queries = data.validation_queries()
    component_sizes = Counter(record.component for record in data.validation)
    if limit is not None:
        queries = queries[:int(limit)]
    totals = []
    started = time.monotonic()
    batch_size = int(config['batch_size'])
    sample_count = int(config['sample_count'])
    repeats = int(config['repeats'])
    for start in range(0, len(queries), batch_size):
        batch = queries[start:start + batch_size]
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                           enabled=device.type == 'cuda'):
            context, full, valid, mask, labels = data.batch(batch, encoder, device, need_full=True)
            for repeat in range(repeats):
                generator = torch.Generator(device=device.type)
                generator.manual_seed(stable_seed(config['seed'], 'evaluation', start, repeat))
                scores = conditional_scores(model, context, full, valid, mask, labels,
                                            sample_count, generator)
                for index, query in enumerate(batch):
                    sites = int(scores['sites'][index].item())
                    item = {
                        'sequence_id': query['record'].sequence_id,
                        'component': query['record'].component,
                        'panel': query['panel']['panel'],
                        'repeat': repeat,
                        'weight': float(query['panel']['weight']) /
                                  component_sizes[query['record'].component],
                        'sites': sites,
                    }
                    for name, values in scores.items():
                        if name == 'sites' or values is None:
                            continue
                        value = float(values[index].item())
                        item[name + '_per_residue' if name.endswith('nll') else name] = (
                            value / sites if name.endswith('nll') else value)
                    totals.append(item)
    if not totals:
        raise ValueError('No validation queries evaluated')
    weighted = sum(item['weight'] for item in totals)
    summary = {
        'status': 'COMPLETED', 'split': split, 'checkpoint': str(checkpoint),
        'checkpoint_sha256': sha256_file(checkpoint),
        'queries': len(totals), 'sample_count': sample_count, 'repeats': repeats,
        'aggregation': 'component-weighted mean of per-target-residue query scores',
        'estimator': 'posterior importance sampling and direct-prior categorical Monte Carlo',
        'seconds': time.monotonic() - started,
        'esm_model': data.sources['esm_model'], 'target_sha256': sha256_file(target_path),
    }
    for name in totals[0]:
        if name.endswith('_per_residue') or name in {'kl', 'ess', 'max_weight'}:
            summary[name] = sum(item['weight'] * item[name] for item in totals) / weighted
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'evaluation.json', summary)
    with (output / 'per_query.jsonl').open('w') as handle:
        for item in totals:
            handle.write(json.dumps(item, ensure_ascii=False) + '\n')
    return summary


def evaluate_auxiliary(config_path, checkpoint, output, device=None):
    """Evaluate both auxiliary arms on identical held-out domain masks."""
    from .auxiliary import AuxiliaryData

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Auxiliary evaluation already contains files: {output}')
    data = AuxiliaryData(config_path)
    config = data.config['evaluation']
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    payload = load_checkpoint(checkpoint, device='cpu')
    model = build_model(payload, device=device)
    encoder = FrozenEncoder(data.sources['esm_model'], device=device)
    queries = data.validation_queries()
    sample_count, repeats = int(config['sample_count']), int(config['repeats'])
    if not queries or repeats < 1:
        raise ValueError('Auxiliary evaluation requires validation queries and repeats')
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    totals = []
    batch_size = int(config['batch_size'])
    with (output / 'per_query.jsonl').open('w') as handle:
        for start in range(0, len(queries), batch_size):
            batch = queries[start:start + batch_size]
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                                enabled=device.type == 'cuda'):
                context, full, valid, mask, labels = data.batch(
                    batch, encoder, device, need_full=model.arm == 'QVAE')
                # A deterministic model has no Monte Carlo error to repeat.
                for repeat in range(repeats if model.arm == 'QVAE' else 1):
                    generator = torch.Generator(device=device.type)
                    generator.manual_seed(stable_seed(config['seed'], 'aux_evaluation', start, repeat))
                    scores = conditional_scores(model, context, full, valid, mask, labels,
                                                sample_count, generator)
                    for index, query in enumerate(batch):
                        uid = query['record']['uid']
                        sites = int(scores['sites'][index].item())
                        item = {'sequence_id': uid,
                                'component': data.split[uid]['domain_cluster50'],
                                'panel': query['panel']['panel'], 'repeat': repeat,
                                'weight': float(query['weight']), 'sites': sites,
                                'aux_kind': query['record']['kind']}
                        for name, values in scores.items():
                            if name == 'sites' or values is None:
                                continue
                            value = float(values[index].item())
                            item[name + '_per_residue' if name.endswith('nll') else name] = (
                                value / sites if name.endswith('nll') else value)
                        totals.append(item)
                        handle.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(json.dumps({'evaluated_queries': min(start + batch_size, len(queries)),
                              'total_queries': len(queries),
                              'seconds': time.monotonic() - started}), flush=True)
    weight_sum = sum(item['weight'] for item in totals)
    summary = {
        'arm': model.arm, 'seed': int(payload['seed']),
        'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256_file(checkpoint),
        'validation_sequences': len(data.validation), 'queries': len(queries),
        'components': len({item['component'] for item in totals}),
        'sample_count': sample_count if model.arm == 'QVAE' else None,
        'repeats': repeats if model.arm == 'QVAE' else 1,
        'aggregation': 'equal domain groups; mean per-target-residue score over fixed masks',
        'estimator': ('posterior IS and direct-prior Monte Carlo; finite-sample log bias remains'
                      if model.arm == 'QVAE' else 'exact site-factorized probability'),
        'inputs': {name: {'path': str(path), 'sha256': sha256_file(path)}
                   for name, path in [('config', config_path), ('domains', data.sources['domains']),
                                      ('masks', data.sources['masks']),
                                      ('split', data.sources['split'])]},
        'seconds': time.monotonic() - started,
    }
    for name in totals[0]:
        if name.endswith('_per_residue') or name in {'kl', 'ess', 'max_weight'}:
            summary[name] = sum(item['weight'] * item[name] for item in totals) / weight_sum
    if model.arm == 'QVAE':
        by_repeat = []
        for repeat in range(repeats):
            rows = [item for item in totals if item['repeat'] == repeat]
            norm = sum(item['weight'] for item in rows)
            by_repeat.append(sum(item['weight'] * item['posterior_joint_nll_per_residue']
                                 for item in rows) / norm)
        summary['posterior_joint_nll_repeats'] = by_repeat
        summary['posterior_joint_nll_mc_se'] = (
            float(torch.tensor(by_repeat, dtype=torch.float64).std(unbiased=True).item()) /
            math.sqrt(repeats) if repeats > 1 else None)
        summary['ess_below_10_percent_fraction'] = (
            sum(item['weight'] for item in totals if item['ess'] < 0.1 * sample_count) / weight_sum)
    write_json(output / 'evaluation.json', summary)
    return summary


def summarize_auxiliary(config_path, training_dir, generation_dir, evaluation_dir):
    """Summarize the completed paired study without new sampling or training."""
    from .data import Target, load_yaml

    config = load_yaml(config_path)
    wt = Target.load(config['paths']['target']).sequence
    training_dir, generation_dir, evaluation_dir = map(
        Path, (training_dir, generation_dir, evaluation_dir))
    destination = evaluation_dir / 'comparison.json'
    if destination.exists():
        raise FileExistsError(destination)
    inputs = [Path(config_path), training_dir / 'training_summary.json']
    grouped = {}
    metrics = ('posterior_joint_nll_per_residue', 'prior_joint_nll_per_residue',
               'marginal_nll_per_residue')
    for arm in ('QVAE', 'deterministic'):
        path = evaluation_dir / arm / 'per_query.jsonl'
        inputs.append(path)
        queries = {}
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = (row['sequence_id'], row['component'], row['panel'])
            item = queries.setdefault(key, {'weight': row['weight'], 'values': []})
            item['values'].append([row[name] for name in metrics])
        grouped[arm] = {key: {'weight': item['weight'],
                             'scores': np.mean(item['values'], axis=0)}
                        for key, item in queries.items()}
    if grouped['QVAE'].keys() != grouped['deterministic'].keys():
        raise ValueError('The two arms do not score identical validation queries')
    components = sorted({key[1] for key in grouped['QVAE']})
    differences = {name: [] for name in ('det_minus_posterior_is', 'det_minus_prior_mc',
                                        'own_marginal_minus_shared_prior')}
    for component in components:
        keys = [key for key in grouped['QVAE'] if key[1] == component]
        weights = np.array([grouped['QVAE'][key]['weight'] for key in keys])
        q = np.array([grouped['QVAE'][key]['scores'] for key in keys])
        d = np.array([grouped['deterministic'][key]['scores'][1] for key in keys])
        for name, values in zip(differences, (d - q[:, 0], d - q[:, 1], q[:, 2] - q[:, 1])):
            differences[name].append(float(np.average(values, weights=weights)))
    rng = np.random.default_rng(8802)
    resamples = rng.integers(len(components), size=(2000, len(components)))
    comparisons = {}
    for name, values in differences.items():
        values = np.asarray(values)
        comparisons[name] = {'mean': float(values.mean()),
                             'paired_group_bootstrap_95ci':
                                 np.quantile(values[resamples].mean(1), [.025, .975]).tolist(),
                             'groups_positive': int((values > 0).sum()),
                             'group_differences': values.tolist()}
    generation_metrics, datasets = {}, {}
    for source in ('aux', 'wt_kvp40'):
        for arm in ('QVAE', 'deterministic'):
            name = f'{source}_{arm}'
            path = generation_dir / name / 'candidates.csv'
            inputs.append(path)
            with path.open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            datasets[name] = rows
            hammings, anchor_changes = [], 0
            event_counts = Counter()
            for row in rows:
                sequence = row['kvp40_sequence']
                if len(sequence) != len(wt) or not set(sequence) <= set('ACDEFGHIKLMNPQRSTVWY'):
                    raise ValueError(f'Invalid full-length candidate: {row["candidate_id"]}')
                import hashlib
                if hashlib.sha256(sequence.encode()).hexdigest() != row['kvp40_sequence_sha256']:
                    raise ValueError(f'Candidate SHA differs: {row["candidate_id"]}')
                allowed = set(json.loads(row['wt_positions']))
                changed = {i + 1 for i, (a, b) in enumerate(zip(wt, sequence)) if a != b}
                if not changed <= allowed:
                    raise ValueError('A generated full sequence changed an unmasked WT position')
                domain_mask = set(json.loads(row['mask_positions']))
                if any(a != b and i not in domain_mask for i, (a, b) in
                       enumerate(zip(row['source_sequence'], row['sequence']))):
                    raise ValueError('A generated domain changed an unmasked source position')
                if source == 'wt_kvp40' and sequence[3:181] != row['sequence']:
                    raise ValueError('The WT-domain generation was embedded incorrectly')
                hammings.append(len(changed))
                anchor_changes += bool(changed & {14, 17, 135, 138})
                for position, alt in ((18, 'I'), (13, 'Y'), (22, 'Q'), (134, 'G'),
                                      (130, 'R'), (140, 'M')):
                    if position in changed and sequence[position - 1] == alt:
                        event_counts[f'{wt[position - 1]}{position}{alt}'] += 1
                event_counts['F13Y+A134G'] += sequence[12] == 'Y' and sequence[133] == 'G'
            unique = len({row['kvp40_sequence_sha256'] for row in rows})
            generation_metrics[name] = {
                'attempts': len(rows), 'unique_full_sequences': unique,
                'duplicate_attempts_beyond_first': len(rows) - unique,
                'wt_returns': hammings.count(0), 'mean_Hamming': float(np.mean(hammings)),
                'Hamming_range': [min(hammings), max(hammings)],
                'anchor_changed_sequences': anchor_changes, 'event_counts': dict(event_counts)}
        q, d = datasets[f'{source}_QVAE'], datasets[f'{source}_deterministic']
        fields = ('background_id', 'panel', 'draw', 'mask_positions', 'uniform_seed')
        if len(q) != len(d) or any(tuple(a[k] for k in fields) != tuple(b[k] for k in fields)
                                   for a, b in zip(q, d)):
            raise ValueError('The two arms did not use identical generation conditions')
    # Representative questions select existing WT-conditioned sequences only.
    anchors = (14, 17, 135, 138)
    unchanged = lambda seq: all(seq[i - 1] == wt[i - 1] for i in anchors)
    questions = (
        ('低编辑参照', unchanged),
        ('L18I 邻域变化', lambda seq: unchanged(seq) and seq[17] == 'I'),
        ('K130R 邻域变化', lambda seq: unchanged(seq) and seq[129] == 'R'),
        ('F13Y+A134G 组合', lambda seq: seq[12] == 'Y' and seq[133] == 'G'),
        ('T135S 锚点变化', lambda seq: seq[134] == 'S' and
         all(seq[i - 1] == wt[i - 1] for i in (14, 17, 138))),
    )
    representatives, fasta = [], {}
    for arm in ('QVAE', 'deterministic'):
        selected = {}
        pool = datasets[f'wt_kvp40_{arm}']
        for question, predicate in questions:
            eligible = [row for row in pool if predicate(row['kvp40_sequence'])]
            if not eligible:
                continue
            row = min(eligible, key=lambda r: (sum(a != b for a, b in
                       zip(wt, r['kvp40_sequence'])), r['candidate_id']))
            selected.setdefault(row['candidate_id'], {'row': row, 'questions': []})['questions'].append(question)
        for number, item in enumerate(selected.values(), 1):
            row = item['row']
            rid = f'{arm}_R{number:02d}'
            sequence = row['kvp40_sequence']
            digest = row['kvp40_sequence_sha256']
            duplicate_of = fasta.get(digest, {}).get('id', '')
            fasta.setdefault(digest, {'id': rid, 'sequence': sequence})
            edits = [f'{a}{i + 1}{b}' for i, (a, b) in enumerate(zip(wt, sequence)) if a != b]
            representatives.append({'representative_id': rid, 'arm': arm,
                                    'candidate_id': row['candidate_id'],
                                    'research_question': ';'.join(item['questions']),
                                    'Hamming': len(edits), 'edits': ';'.join(edits),
                                    'duplicate_of': duplicate_of, 'panel': row['panel'],
                                    'draw': row['draw'], 'uniform_seed': row['uniform_seed'],
                                    'sequence_sha256': digest, 'sequence': sequence})
    with (generation_dir / 'representative_candidates.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(representatives[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(representatives)
    with (generation_dir / 'representative_candidates.fasta').open('w') as handle:
        for item in fasta.values():
            handle.write(f'>{item["id"]}\n{item["sequence"]}\n')
    result = {'seed': int(config['training']['seed']), 'queries': len(grouped['QVAE']),
              'components': components, 'comparisons': comparisons,
              'bootstrap': {'unit': 'paired 50% domain group', 'replicates': 2000, 'seed': 8802,
                            'limits': 'conditional on the trained seed and averaged MC scores; excludes training-seed variance and common finite-K log bias'},
              'generation': generation_metrics, 'representative_records': len(representatives),
              'representative_unique_sequences': len(fasta),
              'inputs': {str(path): sha256_file(path) for path in inputs},
              'implementation_sha256': sha256_file(__file__)}
    write_json(destination, result)
    return result


_TRANSFER_REGIONS = ('global', 'NMNAT', 'Nudix', 'linker', 'random')


def _panel_region(panel):
    for region in _TRANSFER_REGIONS:
        if panel.startswith(region + '_'):
            return region
    raise ValueError(f'Unknown panel region: {panel}')


def _bootstrap(values, reps=2000, seed=8802):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(values), size=(reps, len(values)))
    means = values[draws].mean(1)
    return {'groups': len(values), 'mean': float(values.mean()),
            'ci95': np.quantile(means, [0.025, 0.975]).tolist()}


def _read_transfer_run(run):
    rows = [json.loads(line) for line in open(run['per_query'])]
    component = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    repeats = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0]))
    for row in rows:
        region = _panel_region(row['panel'])
        weight = float(row['weight'])
        component[region]['prior_joint_nll_per_residue'][row['component']].append(
            row['prior_joint_nll_per_residue'])
        component[region]['posterior_joint_nll_per_residue'][row['component']].append(
            row['posterior_joint_nll_per_residue'])
        bucket = repeats[region][int(row['repeat'])]
        bucket[0] += weight
        bucket[1] += weight * row['prior_joint_nll_per_residue']
        bucket[2] += weight * row['posterior_joint_nll_per_residue']
    summary = json.load(open(run['evaluation']))
    regions = {}
    for region in _TRANSFER_REGIONS:
        prior_repeats = [bucket[1] / bucket[0] for _, bucket in sorted(repeats[region].items())]
        post_repeats = [bucket[2] / bucket[0] for _, bucket in sorted(repeats[region].items())]
        prior_se = float(np.std(prior_repeats, ddof=1) / math.sqrt(len(prior_repeats)))
        post_se = float(np.std(post_repeats, ddof=1) / math.sqrt(len(post_repeats)))
        regions[region] = {
            'rows': sum(len(v) for v in component[region]['prior_joint_nll_per_residue'].values()),
            'components': len(component[region]['prior_joint_nll_per_residue']),
            'prior_joint_nll_per_residue': float(np.mean([
                np.mean(values) for values in
                component[region]['prior_joint_nll_per_residue'].values()])),
            'posterior_joint_nll_per_residue': float(np.mean([
                np.mean(values) for values in
                component[region]['posterior_joint_nll_per_residue'].values()])),
            'prior_joint_nll_mc_se': prior_se,
            'posterior_joint_nll_mc_se': post_se,
            'repeats': len(prior_repeats),
        }
    repeat_aggregates = {
        region: {'prior': [bucket[1] / bucket[0] for _, bucket in sorted(repeats[region].items())],
                 'posterior': [bucket[2] / bucket[0] for _, bucket in sorted(repeats[region].items())]}
        for region in _TRANSFER_REGIONS}
    return {'components': {
        region: {metric: {name: float(np.mean(values)) for name, values in by_name.items()}
                 for metric, by_name in by_metric.items()}
        for region, by_metric in component.items()},
        'regions': regions, 'repeat_aggregates': repeat_aggregates,
        'evaluation': summary}


def _paired_transfer(data, pairs):
    result = {'per_pair': {}, 'component_differences': {}, 'overall': {}, 'ci_scope': (
        'component bootstrap over the common groups given the fixed training seeds; '
        'seed variability is reported separately')}
    for metric in ('prior_joint_nll_per_residue', 'posterior_joint_nll_per_residue'):
        result['per_pair'][metric] = {}
        result['component_differences'][metric] = {}
        for random_name, aux_name in pairs:
            key = f'{random_name}|{aux_name}'
            result['per_pair'][metric][key] = {}
            result['component_differences'][metric][key] = {}
            for region in _TRANSFER_REGIONS:
                random = data[random_name]['components'][region][metric]
                aux = data[aux_name]['components'][region][metric]
                common = sorted(set(random) & set(aux))
                differences = {c: float(aux[c] - random[c]) for c in common}
                result['component_differences'][metric][key][region] = differences
                result['per_pair'][metric][key][region] = _bootstrap(list(differences.values()))
        result['overall'][metric] = {}
        for region in _TRANSFER_REGIONS:
            by_component = defaultdict(list)
            for random_name, aux_name in pairs:
                random = data[random_name]['components'][region][metric]
                aux = data[aux_name]['components'][region][metric]
                for component in sorted(set(random) & set(aux)):
                    by_component[component].append(aux[component] - random[component])
            result['overall'][metric][region] = _bootstrap(
                [float(np.mean(values)) for values in by_component.values()])
    return result


def _generation_transfer(manifest):
    from .data import Target
    wt = Target.load(manifest['target']).sequence
    mapping = json.load(open(manifest['pocket_mapping']))
    pocket = set(mapping['pocket_positions'])
    anchors = set(manifest['anchors'])
    runs = {run['name']: run for run in manifest['runs']}
    rows = {name: list(csv.DictReader(open(run['candidates'], newline='')))
            for name, run in runs.items()}
    output = {'runs': {}, 'pairs': {}, 'totals': {}}
    for name, values in rows.items():
        pocket_rows = anchor_rows = nudix_ok = 0
        pocket_edits = anchor_edits = 0
        for row in values:
            edits = json.loads(row['edits'])
            positions = {edit['position'] for edit in edits}
            pocket_hits = positions & pocket
            anchor_hits = positions & anchors
            pocket_rows += bool(pocket_hits)
            anchor_rows += bool(anchor_hits)
            pocket_edits += len(pocket_hits)
            anchor_edits += len(anchor_hits)
            if not positions <= set(range(4, 182)):
                raise ValueError(f'Candidate changed outside NMNAT: {row["candidate_id"]}')
            nudix_ok += 1
        unique = len({row['sequence_sha256'] for row in values})
        output['runs'][name] = {
            'arm': runs[name]['arm'], 'seed': int(runs[name]['seed']),
            'attempts': len(values), 'unique_sequences': unique,
            'duplicate_attempts_beyond_first': len(values) - unique,
            'mean_Hamming': float(np.mean([int(row['Hamming']) for row in values])),
            'pocket_edit_rows': pocket_rows, 'pocket_edits': pocket_edits,
            'anchor_edit_rows': anchor_rows, 'anchor_edits': anchor_edits,
            'nudix_unchanged': nudix_ok,
        }
    for random_name, aux_name in manifest['pairs']:
        random, aux = rows[random_name], rows[aux_name]
        random_by_key = {(row['group'], int(row['draw'])): row for row in random}
        aux_by_key = {(row['group'], int(row['draw'])): row for row in aux}
        if set(random_by_key) != set(aux_by_key):
            raise ValueError('Generation attempts differ between arms')
        for key in random_by_key:
            a, b = random_by_key[key], aux_by_key[key]
            if any(a[field] != b[field] for field in
                   ('group', 'mask_size', 'prior_seed', 'uniform_seed')):
                raise ValueError('Generation masks/uniforms differ between arms')
        matched = [key for key in random_by_key
                   if int(random_by_key[key]['Hamming']) == int(aux_by_key[key]['Hamming'])]
        def stats(pool, source):
            pocket_rows = sum(bool({e['position'] for e in json.loads(row['edits'])} & pocket)
                              for row in (pool[k] for k in source))
            anchor_rows = sum(bool({e['position'] for e in json.loads(row['edits'])} & anchors)
                              for row in (pool[k] for k in source))
            return {'rows': len(source), 'pocket_rows': pocket_rows,
                    'pocket_rate': pocket_rows / len(source) if source else None,
                    'anchor_rows': anchor_rows,
                    'anchor_rate': anchor_rows / len(source) if source else None}
        output['pairs'][f'{random_name}|{aux_name}'] = {
            'pairs': len(random_by_key), 'common_masks_uniforms': True,
            'hamming_equal_pairs': len(matched),
            'raw': {'random': stats(random_by_key, list(random_by_key)),
                    'auxinit': stats(aux_by_key, list(aux_by_key))},
            'hamming_matched': {'random': stats(random_by_key, matched),
                                'auxinit': stats(aux_by_key, matched)},
        }
    arm_stats = {}
    for arm in ('random', 'auxinit'):
        names = [run['name'] for run in manifest['runs'] if run['arm'] == arm]
        per_run = [len({row['sequence_sha256'] for row in rows[name]}) for name in names]
        union = set()
        for name in names:
            union |= {row['sequence_sha256'] for row in rows[name]}
        arm_stats[arm] = {
            'attempts': sum(len(rows[name]) for name in names),
            'per_run_unique_sum': sum(per_run),
            'cross_seed_union': len(union),
            'duplicate_attempts_beyond_first': sum(len(rows[name]) for name in names) - len(union),
        }
    global_union = set()
    for values in rows.values():
        global_union |= {row['sequence_sha256'] for row in values}
    output['totals'] = {
        'attempts': sum(v['attempts'] for v in output['runs'].values()),
        'per_run_unique_sum': sum(v['unique_sequences'] for v in output['runs'].values()),
        'per_run_duplicate_attempts_beyond_first': sum(
            v['duplicate_attempts_beyond_first'] for v in output['runs'].values()),
        'global_union_unique': len(global_union),
        'global_duplicate_attempts_beyond_first': sum(len(v) for v in rows.values()) - len(global_union),
        'arms': arm_stats,
        'nudix_unchanged': sum(v['nudix_unchanged'] for v in output['runs'].values()),
    }
    return output


def _training_curves(manifest, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(8, 5))
    colors = {'random': '#1f77b4', 'auxinit': '#d62728'}
    curve_rows = []
    for run in manifest['runs']:
        values = [json.loads(line) for line in open(run['training_curve'])]
        epochs = [row['epoch'] for row in values]
        scores = [row['prior_joint_nll_per_residue'] for row in values]
        axis.plot(epochs, scores, marker='o', ms=3, lw=1.3, alpha=.85,
                  color=colors[run['arm']], label=run['name'])
        best = min(values, key=lambda row: row['prior_joint_nll_per_residue'])
        axis.scatter([best['epoch']], [best['prior_joint_nll_per_residue']],
                     marker='*', s=90, color=colors[run['arm']], edgecolor='black', zorder=5)
        for row in values:
            curve_rows.append([run['name'], row['epoch'], row['stage'],
                               row['prior_joint_nll_per_residue'],
                               row['posterior_joint_nll_per_residue'], row.get('ess_mean')])
    axis.set_xlabel('epoch')
    axis.set_ylabel('NMNAT selection prior joint NLL / residue')
    axis.set_title('Full NatV transfer: fixed 384-query selection')
    axis.grid(alpha=.25)
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(output / 'training_curves.png', dpi=160)
    with (output / 'training_curves.csv').open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['run', 'epoch', 'stage', 'prior_joint_nll_per_residue',
                         'posterior_joint_nll_per_residue', 'ess_mean'])
        writer.writerows(curve_rows)


def _transfer_representatives(manifest, output):
    runs = {run['name']: run for run in manifest['runs']}
    rows = {name: list(csv.DictReader(open(run['candidates'], newline='')))
            for name, run in runs.items()}
    selected, fasta = [], {}
    for item in manifest['representatives']:
        pool = rows[item['run']]
        matches = [row for row in pool if row['candidate_id'] == item['candidate_id']]
        if len(matches) != 1:
            raise ValueError(f'Representative not unique: {item}')
        row = matches[0]
        if int(row['Hamming']) != int(item['matched_hamming']):
            raise ValueError(f'Representative Hamming differs: {item}')
        sequence = row['sequence']
        digest = row['sequence_sha256']
        if hashlib.sha256(sequence.encode()).hexdigest() != digest:
            raise ValueError(f'Representative SHA differs: {item}')
        representative_id = item['representative_id']
        fasta.setdefault(digest, {'id': representative_id, 'sequence': sequence})
        selected.append({
            'representative_id': representative_id, 'arm': item['arm'],
            'seed': int(runs[item['run']]['seed']), 'candidate_id': row['candidate_id'],
            'research_question': item['research_question'],
            'matched_hamming': int(item['matched_hamming']), 'Hamming': int(row['Hamming']),
            'edits': ';'.join(f'{edit["from"]}{edit["position"]}{edit["to"]}'
                              for edit in json.loads(row['edits'])),
            'sequence_sha256': digest, 'sequence': sequence,
            'group': row['group'], 'draw': row['draw'],
            'uniform_seed': row['uniform_seed'],
        })
    with (output / 'representative_candidates.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(selected)
    with (output / 'representative_candidates.fasta').open('w') as handle:
        for item in fasta.values():
            handle.write(f'>{item["id"]}\n{item["sequence"]}\n')
    return selected


def _transfer_identity(manifest, manifest_path, paired_path, generation_path, curves_path,
                       representatives_path):
    def digest(path):
        return sha256_file(Path(path)) if Path(path).exists() else None
    runs = {}
    for run in manifest['runs']:
        summary = json.load(open(run['training_summary']))
        runs[run['name']] = {
            'arm': run['arm'], 'seed': int(run['seed']),
            'training_summary': {'path': run['training_summary'],
                                 'sha256': digest(run['training_summary'])},
            'evaluation': {'path': run['evaluation'], 'sha256': digest(run['evaluation'])},
            'per_query': {'path': run['per_query'], 'sha256': digest(run['per_query'])},
            'generation': {'path': run['generation'], 'sha256': digest(run['generation'])},
            'candidates': {'path': run['candidates'], 'sha256': digest(run['candidates'])},
            'training_curve': {'path': run['training_curve'], 'sha256': digest(run['training_curve'])},
            'validation_selection': {'path': run['validation_selection'],
                                     'sha256': digest(run['validation_selection'])},
            'checkpoints': summary['checkpoints'],
        }
    structures = {}
    for path in manifest.get('structure_summaries', []):
        structures[path] = {'sha256': digest(path)}
    identity = {
        'manifest': {'path': str(manifest_path), 'sha256': digest(manifest_path)},
        'git_revision': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'summary_implementation_sha256': sha256_file(__file__),
        'inputs': {path: digest(path) for path in manifest.get('inputs', [])},
        'runs': runs,
        'structures': structures,
        'paired_summary': {'path': str(paired_path), 'sha256': digest(paired_path)},
        'generation_summary': {'path': str(generation_path), 'sha256': digest(generation_path)},
        'training_curves': {'path': str(curves_path), 'sha256': digest(curves_path)},
        'representative_candidates': {'path': str(representatives_path),
                                      'sha256': digest(representatives_path)},
    }
    return identity


def summarize_transfer(manifest_path, output):
    """Rebuild transfer summaries from existing per-query and candidate files."""
    manifest = json.load(open(manifest_path))
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Transfer summary output already contains files: {output}')
    output.mkdir(parents=True, exist_ok=True)
    data = {run['name']: _read_transfer_run(run) for run in manifest['runs']}
    paired = _paired_transfer(data, manifest['pairs'])
    paired['runs'] = {run['name']: {
        'regions': data[run['name']]['regions'],
        'repeat_aggregates': data[run['name']]['repeat_aggregates'],
        'global_evaluation': data[run['name']]['evaluation'],
    } for run in manifest['runs']}
    paired['all_mask_overall'] = {}
    for metric in ('prior_joint_nll_per_residue', 'posterior_joint_nll_per_residue'):
        random_values = [data[run['name']]['evaluation'][metric]
                         for run in manifest['runs'] if run['arm'] == 'random']
        aux_values = [data[run['name']]['evaluation'][metric]
                      for run in manifest['runs'] if run['arm'] == 'auxinit']
        per_seed = [a - r for r, a in zip(random_values, aux_values)]
        paired['all_mask_overall'][metric] = {
            'random': random_values, 'auxinit': aux_values,
            'per_seed_delta': per_seed, 'mean_delta': float(np.mean(per_seed)),
            'note': 'equal-weight mean over the six evaluation.json summaries; not the global_* region mask'}

    paired['limits'] = (
        'component-level bootstrap given the trained seeds; not a resampling of training seeds; '
        'MC SE is the observed four-repeat standard error')
    generation = _generation_transfer(manifest)
    _training_curves(manifest, output)
    representatives = _transfer_representatives(manifest, output)
    write_json(output / 'paired_summary.json', paired)
    write_json(output / 'generation_summary.json', generation)
    identity = _transfer_identity(
        manifest, manifest_path, output / 'paired_summary.json',
        output / 'generation_summary.json', output / 'training_curves.png',
        output / 'representative_candidates.csv')
    write_json(output / 'run_identity.json', identity)
    return {'status': 'COMPLETED', 'runs': len(data), 'representatives': len(representatives),
            'output': str(output)}
