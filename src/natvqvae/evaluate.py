"""Validation evaluation for the QVAE checkpoint."""
from __future__ import annotations

import json
import math
import time
import csv
from collections import Counter
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
