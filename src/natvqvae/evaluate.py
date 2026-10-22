"""Validation evaluation for the QVAE checkpoint."""
from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path

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
