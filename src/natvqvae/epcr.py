"""Structure comparison of the two full-length KVP40 ePCR variants."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from .data import load_yaml, sha256_file, write_json
from .design import (_ca_map, _cif_sequence, _confidence, _kabsch, _model_paths,
                     _write_msa)


def _read_wt_dna(path):
    raw = Path(path).read_bytes()
    candidates = re.findall(rb'[acgt]{1000,}', raw)
    if not candidates:
        raise ValueError(f'No DNA sequence in {path}')
    dna = max(candidates, key=len).decode()
    if len(dna) != 1026:
        raise ValueError(f'Unexpected WT DNA length: {len(dna)}')
    return dna


def _translate(dna):
    from Bio.Seq import Seq
    protein = str(Seq(dna).translate())
    if len(protein) != 342 or protein[-1] != '*':
        raise ValueError('WT DNA does not translate to 341 aa plus stop')
    return protein[:-1]


def _read_ab1(path):
    from Bio import SeqIO
    record = next(SeqIO.parse(str(path), 'abi'))
    return str(record.seq).lower(), list(record.letter_annotations['phred_quality'])


def _read_mapping(wt_dna, read, quality):
    from Bio.Align import PairwiseAligner
    from Bio.Seq import Seq
    aligner = PairwiseAligner()
    aligner.mode = 'local'
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -6
    aligner.extend_gap_score = -1
    choices = [('forward', read, quality),
               ('reverse', str(Seq(read).reverse_complement()), quality[::-1])]
    best = None
    for orientation, sequence, scores in choices:
        alignment = aligner.align(wt_dna, sequence)[0]
        if best is None or alignment.score > best[0]:
            best = (alignment.score, orientation, sequence, scores, alignment)
    score, orientation, sequence, scores, alignment = best
    mapping = {}
    for (wt_start, wt_end), (read_start, _) in zip(alignment.aligned[0],
                                                   alignment.aligned[1]):
        for offset in range(wt_end - wt_start):
            mapping[wt_start + offset] = read_start + offset
    return mapping, sequence, scores, orientation, score


def _verify_sanger(config, wt_dna, clones):
    output = {}
    for name, edits in clones.items():
        observations = []
        for suffix in ('F', 'R'):
            path = Path(config['paths']['ab1_dir']) / f'{name}-yicuo-{suffix}.ab1'
            read, quality = _read_ab1(path)
            mapping, sequence, scores, orientation, score = _read_mapping(wt_dna, read, quality)
            for edit in edits:
                position = int(edit['position']) - 1
                if position not in mapping:
                    observations.append({'clone': name, 'read': suffix, 'position': position + 1,
                                         'protein': edit['protein'], 'covered': False})
                    continue
                index = mapping[position]
                observations.append({'clone': name, 'read': suffix, 'position': position + 1,
                                     'protein': edit['protein'], 'covered': True,
                                     'base': sequence[index], 'quality': scores[index],
                                     'orientation': orientation, 'alignment_score': score})
        for edit in edits:
            position = int(edit['position'])
            covered = [item for item in observations
                       if item['position'] == position and item['covered']]
            alt_seen = [item for item in covered
                        if item['base'] == edit['alt'] and item['quality'] >= 20]
            if not alt_seen:
                raise ValueError(f"Sanger does not confirm {name} {edit['protein']} at {position}")
            if any(item['base'] == edit['ref'] for item in covered):
                raise ValueError(f"Sanger shows ref base for {name} {edit['protein']} at {position}")
        output[name] = observations
    return output


def _variant_dna(wt_dna, edits):
    sequence = list(wt_dna)
    for edit in edits:
        position = int(edit['position']) - 1
        if sequence[position] != edit['ref']:
            raise ValueError(f"WT base differs at {position + 1}")
        sequence[position] = edit['alt']
    return ''.join(sequence)


def _apply_variants(config, wt_dna):
    target = load_yaml(config['paths']['target'])['sequence']
    sequences = {'WT': target}
    mutations = {'WT': []}
    for name, edits in config['variants'].items():
        dna = _variant_dna(wt_dna, edits)
        protein = _translate(dna)
        observed = [f'{target[i]}{i + 1}{protein[i]}' for i in range(341)
                    if target[i] != protein[i]]
        expected = [edit['protein'] for edit in edits if edit['protein'][0] != edit['protein'][-1]]
        if observed != expected:
            raise ValueError(f'Protein edits differ from Sanger labels: {name} {observed} vs {expected}')
        sequences[name] = protein
        mutations[name] = edits
    return sequences, mutations


def _yaml_text(name, sequence, msa):
    return (f'version: 1\nsequences:\n  - protein:\n      id: A\n'
            f'      sequence: {sequence}\n      msa: {msa.resolve()}\n')


def build_epcr(config_path, output):
    config = load_yaml(config_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    inputs = output / 'inputs'
    inputs.mkdir(exist_ok=True)
    wt_dna = _read_wt_dna(config['paths']['wt_dna'])
    target = load_yaml(config['paths']['target'])['sequence']
    if _translate(wt_dna) != target:
        raise ValueError('WT DNA does not match config/target.yaml')
    sequences, mutations = _apply_variants(config, wt_dna)
    verification = _verify_sanger(config, wt_dna, config['variants'])
    rows = []
    for name, sequence in sequences.items():
        msa = inputs / f'{name}.a3m'
        _write_msa(config['paths']['msa_source'], msa, name, sequence,
                   int(config['msa_rows']), target)
        yaml_path = inputs / f'{name}.yaml'
        yaml_path.write_text(_yaml_text(name, sequence, msa))
        rows.append({'name': name, 'length': len(sequence),
                     'protein_edits': ';'.join(edit['protein'] for edit in mutations[name]
                                               if edit['protein'][0] != edit['protein'][-1]) or 'none',
                     'dna_edits': ';'.join(f"{edit['position']}{edit['ref']}>{edit['alt']}"
                                           for edit in mutations[name]) or 'none',
                     'sequence': sequence,
                     'sequence_sha256': hashlib.sha256(sequence.encode()).hexdigest()})
    with (output / 'variants.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / 'sanger_verification.json', verification)
    ab1 = sorted(Path(config['paths']['ab1_dir']).glob('*.ab1'))
    manifest = {'config': str(config_path), 'config_sha256': sha256_file(config_path),
                'target_sha256': sha256_file(config['paths']['target']),
                'wt_dna': str(config['paths']['wt_dna']),
                'wt_dna_sha256': sha256_file(config['paths']['wt_dna']),
                'msa_source_sha256': sha256_file(config['paths']['msa_source']),
                'ab1_sha256': {path.name: sha256_file(path) for path in ab1},
                'variants': {row['name']: {'protein_edits': row['protein_edits'],
                                           'dna_edits': row['dna_edits'],
                                           'sequence_sha256': row['sequence_sha256']}
                             for row in rows}}
    write_json(output / 'build.json', manifest)
    return {'status': 'COMPLETED', 'variants': [row['name'] for row in rows],
            'sanger_checked': list(verification)}


def run_epcr_boltz(config_path, output):
    config = load_yaml(config_path)
    output = Path(output)
    boltz = config['boltz']
    env = dict(os.environ)
    env['BOLTZ_CACHE'] = boltz['cache']
    log_dir = output / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name in config['sequences']:
        run_dir = output / name
        paths = sorted((run_dir / f'boltz_results_{name}' / 'predictions' / name).glob('*.cif'))
        if len(paths) == int(boltz['diffusion_samples']):
            records.append({'name': name, 'status': 'SKIPPED_EXISTING', 'output': str(run_dir)})
            continue
        if paths:
            raise RuntimeError(f'Incomplete existing run for {name}: {len(paths)} models')
        command = [boltz['binary'], 'predict', str(output / 'inputs' / f'{name}.yaml'),
                   '--out_dir', str(run_dir), '--accelerator', 'gpu', '--devices', '1',
                   '--diffusion_samples', str(boltz['diffusion_samples']),
                   '--sampling_steps', str(boltz['sampling_steps']),
                   '--recycling_steps', str(boltz['recycling_steps']),
                   '--seed', str(boltz['seed']), '--output_format', boltz['output_format'],
                   '--num_workers', str(boltz['num_workers'])]
        started = time.monotonic()
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        log = log_dir / f'{name}.log'
        log.write_text(result.stdout + result.stderr)
        record = {'name': name, 'command': command, 'returncode': result.returncode,
                  'seconds': time.monotonic() - started, 'log': str(log),
                  'output': str(run_dir),
                  'status': 'COMPLETED' if result.returncode == 0 else 'FAILED'}
        records.append(record)
        if result.returncode != 0:
            raise RuntimeError(f'Boltz failed for {name}; see {log}')
    with (output / 'boltz_runs.jsonl').open('w') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    return {'status': 'COMPLETED', 'runs': len(records)}


def _pair_rows(config, output, sequences):
    models = {name: [_ca_map(path) for path in _model_paths(output, name)]
              for name in config['sequences']}
    rows = []
    for name in config['sequences']:
        if name == 'WT':
            continue
        for region, (start, end) in config['domains'].items():
            indices = list(range(start, end + 1))
            values = []
            for left in models[name]:
                for right in models['WT']:
                    values.append(_kabsch([left[i] for i in indices],
                                          [right[i] for i in indices])[0])
            rows.append({'variant': name, 'region': region,
                         'median_rmsd_A': float(np.median(values)),
                         'min_rmsd_A': float(min(values)), 'max_rmsd_A': float(max(values)),
                         'model_pairs': len(values)})
    return rows


def _self_rows(config, output):
    models = {name: [_ca_map(path) for path in _model_paths(output, name)]
              for name in config['sequences']}
    rows = []
    for name, values in models.items():
        for region, (start, end) in config['domains'].items():
            indices = list(range(start, end + 1))
            pair_values = []
            for left in range(3):
                for right in range(left + 1, 3):
                    pair_values.append(_kabsch([values[left][i] for i in indices],
                                               [values[right][i] for i in indices])[0])
            rows.append({'name': name, 'region': region,
                         'median_rmsd_A': float(np.median(pair_values)),
                         'min_rmsd_A': float(min(pair_values)),
                         'max_rmsd_A': float(max(pair_values)), 'model_pairs': 3})
    return rows


def _mutation_rows(config, output, sequences):
    models = {name: [_ca_map(path) for path in _model_paths(output, name)]
              for name in config['sequences']}
    rows = []
    for name, edits in config['variants'].items():
        for edit in edits:
            label = edit['protein']
            if label[0] == label[-1]:
                continue
            position = int(label[1:-1])
            domain = 'NMNAT' if position <= 181 else 'Nudix'
            start, end = config['domains'][domain]
            indices = list(range(start, end + 1))
            displacements = []
            neighborhoods = []
            for left in models[name]:
                for right in models['WT']:
                    _, rotation, source_center, target_center = _kabsch(
                        [left[i] for i in indices], [right[i] for i in indices])
                    aligned = (right[position] - target_center) @ rotation.T + source_center
                    displacements.append(float(np.linalg.norm(aligned - left[position])))
                    neighborhood = [i for i in range(max(1, position - 5),
                                                    min(341, position + 5) + 1)]
                    diffs = np.array([((right[i] - target_center) @ rotation.T + source_center)
                                      - left[i] for i in neighborhood])
                    neighborhoods.append(float(np.sqrt(np.mean(np.sum(diffs * diffs, axis=1)))))
            rows.append({'variant': name, 'edit': label, 'position': position,
                         'domain': domain, 'ca_displacement_median_A': float(np.median(displacements)),
                         'ca_displacement_range_A': [float(min(displacements)),
                                                     float(max(displacements))],
                         'neighborhood_rmsd_median_A': float(np.median(neighborhoods)),
                         'neighborhood_rmsd_range_A': [float(min(neighborhoods)),
                                                       float(max(neighborhoods))],
                         'model_pairs': len(displacements)})
    return rows



def _self_local_rows(config, output):
    models = {name: [_ca_map(path) for path in _model_paths(output, name)]
              for name in config['sequences']}
    positions = sorted({int(edit['protein'][1:-1])
                        for edits in config['variants'].values() for edit in edits
                        if edit['protein'][0] != edit['protein'][-1]})
    rows = []
    for name, values in models.items():
        for position in positions:
            domain = 'NMNAT' if position <= 181 else 'Nudix'
            start, end = config['domains'][domain]
            indices = list(range(start, end + 1))
            displacements = []
            neighborhoods = []
            for left in range(3):
                for right in range(left + 1, 3):
                    _, rotation, source_center, target_center = _kabsch(
                        [values[left][i] for i in indices],
                        [values[right][i] for i in indices])
                    aligned = (values[right][position] - target_center) @ rotation.T + source_center
                    displacements.append(float(np.linalg.norm(aligned - values[left][position])))
                    neighborhood = [i for i in range(max(1, position - 5),
                                                    min(341, position + 5) + 1)]
                    diffs = np.array([((values[right][i] - target_center) @ rotation.T + source_center)
                                      - values[left][i] for i in neighborhood])
                    neighborhoods.append(float(np.sqrt(np.mean(np.sum(diffs * diffs, axis=1)))))
            rows.append({'name': name, 'position': position, 'domain': domain,
                         'self_ca_displacement_median_A': float(np.median(displacements)),
                         'self_ca_displacement_range_A': [float(min(displacements)),
                                                          float(max(displacements))],
                         'self_neighborhood_rmsd_median_A': float(np.median(neighborhoods)),
                         'self_neighborhood_rmsd_range_A': [float(min(neighborhoods)),
                                                            float(max(neighborhoods))]})
    return rows

def _domain_rows(config, output):
    rows = []
    for name in config['sequences']:
        for path in _model_paths(output, name):
            model = _ca_map(path)
            nmnat = np.array([model[i] for i in range(*[config['domains']['NMNAT'][0],
                                                        config['domains']['NMNAT'][1] + 1])])
            nudix = np.array([model[i] for i in range(*[config['domains']['Nudix'][0],
                                                        config['domains']['Nudix'][1] + 1])])
            rows.append({'name': name, 'model': path.name,
                         'domain_centroid_distance_A': float(np.linalg.norm(nmnat.mean(0) - nudix.mean(0)))})
    return rows


def _confidence_rows(config, output):
    rows = []
    for name in config['sequences']:
        paths = _model_paths(output, name)
        values = []
        for path in paths:
            confidence = path.parent / f'confidence_{path.stem}.json'
            payload = json.load(open(confidence))
            values.append({key: payload.get(key) for key in
                           ('ptm', 'iptm', 'complex_plddt', 'confidence_score')})
        for key in ('ptm', 'iptm', 'complex_plddt', 'confidence_score'):
            vals = [value[key] for value in values if value[key] is not None]
            if vals:
                rows.append({'name': name, 'metric': key, 'median': float(np.median(vals)),
                             'min': float(min(vals)), 'max': float(max(vals))})
    return rows


def _write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def _figure(path, pairs, self_rows, mutations, self_local):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    # variant vs WT, domain RMSD
    labels = [f"{row['variant']}\n{row['region']}" for row in pairs]
    medians = [row['median_rmsd_A'] for row in pairs]
    lows = [row['median_rmsd_A'] - row['min_rmsd_A'] for row in pairs]
    highs = [row['max_rmsd_A'] - row['median_rmsd_A'] for row in pairs]
    axes[0, 0].bar(range(len(pairs)), medians, yerr=[lows, highs], capsize=3, color='#4c78a8')
    axes[0, 0].set_xticks(range(len(pairs)))
    axes[0, 0].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    axes[0, 0].set_ylabel('C-alpha RMSD to WT (Å)')
    axes[0, 0].set_title('Variant vs WT (median, min-max)')
    # within-variant sample dispersion
    regions = ['full', 'NMNAT', 'Nudix']
    width = 0.25
    names = sorted({row['name'] for row in self_rows})
    for offset, name in enumerate(names):
        values = [next(row for row in self_rows if row['name'] == name and row['region'] == region)
                  ['median_rmsd_A'] for region in regions]
        axes[0, 1].bar(np.arange(len(regions)) + (offset - 1) * width, values, width, label=name)
    axes[0, 1].set_xticks(np.arange(len(regions)))
    axes[0, 1].set_xticklabels(regions)
    axes[0, 1].set_ylabel('Within-variant sample RMSD (Å)')
    axes[0, 1].set_title('Sample dispersion (median)')
    axes[0, 1].legend(fontsize=7)
    # local site displacement and neighborhood vs same-position self baseline
    site_rows = [row for row in mutations if row['edit'][0] != row['edit'][-1]]
    x = np.arange(len(site_rows))
    local_labels = [f"{row['variant'].replace('NatV', '')} {row['edit']}" for row in site_rows]
    local = {f"{row['name']}:{row['position']}": row for row in self_local}
    for panel, metric, self_metric, title in (
            (axes[1, 0], 'ca_displacement_median_A', 'self_ca_displacement_median_A',
             'Mutation-site CA displacement'),
            (axes[1, 1], 'neighborhood_rmsd_median_A', 'self_neighborhood_rmsd_median_A',
             'Mutation +/-5 neighborhood RMSD')):
        med = [row[metric] for row in site_rows]
        err_low = [row[metric] - row['ca_displacement_range_A'][0] if metric == 'ca_displacement_median_A'
                   else row[metric] - row['neighborhood_rmsd_range_A'][0] for row in site_rows]
        err_high = [row['ca_displacement_range_A'][1] - row[metric] if metric == 'ca_displacement_median_A'
                    else row['neighborhood_rmsd_range_A'][1] - row[metric] for row in site_rows]
        own = [local[f"{row['variant']}:{row['position']}"][self_metric] for row in site_rows]
        wt = [local[f"WT:{row['position']}"][self_metric] for row in site_rows]
        panel.bar(x, med, yerr=[err_low, err_high], capsize=2, color='#4c78a8', label='variant vs WT')
        panel.scatter(x, own, color='#f58518', marker='o', s=28, label='variant self median', zorder=3)
        panel.scatter(x, wt, color='#54a24b', marker='D', s=22, label='WT self median', zorder=3)
        panel.set_xticks(x)
        panel.set_xticklabels(local_labels, rotation=45, ha='right', fontsize=8)
        panel.set_ylabel('C-alpha RMSD (Å)')
        panel.set_title(title)
        panel.legend(fontsize=7)
    fig.suptitle('CA metrics: descriptive median and min-max; 9 cross-model pairs are not independent samples; no CI', fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path.with_suffix('.png'), dpi=180)
    fig.savefig(path.with_suffix('.svg'))
    plt.close(fig)

def summarize_epcr(config_path, output):
    config = load_yaml(config_path)
    output = Path(output)
    sequences = {row['name']: row['sequence'] for row in
                 csv.DictReader(open(output / 'variants.csv'))}
    for name, sequence in sequences.items():
        for path in _model_paths(output, name):
            if _cif_sequence(path) != sequence:
                raise ValueError(f'Model sequence differs for {name}: {path}')
    pairs = _pair_rows(config, output, sequences)
    self_rows = _self_rows(config, output)
    mutations = _mutation_rows(config, output, sequences)
    self_local = _self_local_rows(config, output)
    domains = _domain_rows(config, output)
    confidence = _confidence_rows(config, output)
    _write_csv(output / 'pairs.csv', pairs)
    _write_csv(output / 'self_rmsd.csv', self_rows)
    _write_csv(output / 'mutation_local.csv', mutations)
    _write_csv(output / 'mutation_local_baseline.csv', self_local)
    _write_csv(output / 'domain_geometry.csv', domains)
    _write_csv(output / 'confidence.csv', confidence)
    figure = output / 'rmsd_summary.svg'
    _figure(figure, pairs, self_rows, mutations, self_local)
    summary = {'status': 'COMPLETED',
               'variants': {name: hashlib.sha256(sequence.encode()).hexdigest()
                            for name, sequence in sequences.items()},
               'pairs': pairs, 'self_rmsd': self_rows,
               'mutation_local': mutations, 'mutation_local_baseline': self_local,
               'domain_geometry': domains,
               'confidence': confidence, 'figure': figure.name,
               'caveat': 'CA-based structural comparison only; no activity or stability inference'}
    write_json(output / 'summary.json', summary)
    return summary
