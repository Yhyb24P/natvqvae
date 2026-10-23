"""Compare WT holo protein conformations with existing BioEmu apo frames."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

from .data import load_yaml, sha256_file, write_json
from .nmnat_af3 import _af3_models
from .nmnat_binding import _structure
from .nmnat_mapping import _kabsch

NMNAT = (4, 181)
NUDIX = (204, 338)
LOOP = (102, 105)
ANCHORS = (14, 17, 133, 135, 138)


def _ca_map(structure):
    return {int(residue.id[1]): residue['CA'].coord.astype(float)
            for residue in structure['A']
            if residue.id[0] == ' ' and 'CA' in residue}


def _rmsd(first, second, positions):
    common = [position for position in positions
              if position in first and position in second]
    if not common:
        return None
    left = np.asarray([first[position] for position in common])
    right = np.asarray([second[position] for position in common])
    return round(float(np.sqrt(np.mean(np.sum(
        (left - right) ** 2, axis=1)))), 3)


def _compare(first, second):
    common = sorted(set(first) & set(second)
                    & set(range(NMNAT[0], NMNAT[1] + 1)))
    if len(common) < 40:
        return None
    source = np.asarray([first[position] for position in common])
    target = np.asarray([second[position] for position in common])
    _, rotation, translation = _kabsch(source, target)
    moved = {position: first[position] @ rotation.T + translation
             for position in first}
    return {
        'alignment_CA_RMSD_A': round(float(np.sqrt(np.mean(np.sum(
            (np.asarray([moved[position] for position in common]) - target) ** 2,
            axis=1)))), 3),
        'nudix_CA_RMSD_A': _rmsd(moved, second, range(NUDIX[0], NUDIX[1] + 1)),
        'loop_CA_RMSD_A': _rmsd(moved, second, range(LOOP[0], LOOP[1] + 1)),
        'anchor_CA_RMSD_A': _rmsd(moved, second, ANCHORS),
        'local_CA_RMSD_A': _rmsd(
            moved, second,
            list(range(LOOP[0], LOOP[1] + 1)) + list(ANCHORS)),
    }


def _bioemu_frames(directory, mode, label):
    import mdtraj

    directory = Path(directory)
    trajectory = mdtraj.load_xtc(str(directory / 'samples.xtc'),
                                top=str(directory / 'topology.cif'))
    indices = trajectory.topology.select('name CA')
    frames = []
    for index in range(trajectory.n_frames):
        frames.append({
            'label': f'{label}_frame{index}', 'mode': mode,
            'ca': {int(trajectory.topology.atom(atom).residue.resSeq):
                   trajectory.xyz[index, atom].astype(float) * 10.0
                   for atom in indices}})
    return frames, trajectory


def _summary(values):
    clean = [value for value in values if value is not None]
    if not clean:
        return {'n': 0, 'median': None, 'min': None, 'max': None}
    return {'n': len(clean),
            'median': round(float(np.median(clean)), 3),
            'min': round(float(min(clean)), 3),
            'max': round(float(max(clean)), 3)}


def _msa_identity(directory, sequence):
    directory = Path(directory)
    candidates = []
    for base in (directory.parent, directory.parent.parent):
        if (base / 'msa').is_dir():
            candidates.extend(sorted((base / 'msa').glob('*.json')))
    for path in candidates:
        try:
            values = json.load(open(path))
        except Exception:
            continue
        queries = values.get('request', {}).get('sequences') or []
        if not queries or queries[0].strip() != sequence.strip():
            continue
        text = values['a3ms'][0]
        return {'source': 'colabfold_cache_matching_current_wt',
                'path': str(path),
                'a3m_sha256': __import__('hashlib').sha256(
                    text.encode()).hexdigest(),                'rows': text.count('\n>') + 1 if text.startswith('>') else None,
                'use_env': values.get('request', {}).get('use_env'),
                'run_time_version': (
                    'not separately recorded; this is an existing cache whose '
                    'full query sequence matches the current WT, not proof '
                    'that the earlier BioEmu runs used this cache version')}
    return None


def _boltz_ca_records(joint_run):
    records = {}
    for path in sorted((Path(joint_run) / 'coordinates').glob('*.cif.gz')):
        name = path.name[:-len('.cif.gz')]
        state, rest = name.split('_seed')
        seed, model = rest.split('_model_')
        records.setdefault(state, []).append({
            'label': f'seed{seed}_model{model}',
            'ca': _ca_map(_structure(path))})
    return records


def analyze_apo(config_path, output):
    config = load_yaml(config_path)
    output = Path(output).resolve()
    af3 = {'wt_atp_nmn': [], 'wt_atp_nmn_mg': []}
    for item in _af3_models(output):
        if item['condition'] in af3:
            af3[item['condition']].append({
                'label': f"seed{item['seed']}_sample{item['sample']}",
                'ca': _ca_map(_structure(item['path']))})
    boltz = {state: records
             for state, records in _boltz_ca_records(config['joint_run']).items()
             if state in ('ATP_NMN', 'ATP_NMN_MG')}
    if not any(af3.values()):
        raise ValueError('No AF3 WT models found')
    rows = []
    identity = {'af3_models': {key: len(value) for key, value in af3.items()},
                'boltz_models': {key: len(value) for key, value in boltz.items()},
                'apo_sets': []}
    for entry in config['apo']['sets']:
        frames, universe = _bioemu_frames(
            entry['directory'], entry['mode'], entry['label'])
        system = (Path(entry['directory']) / 'system.yaml').read_text().strip()
        sequence = re.search(r'^\s*sequence:\s*(\S+)', system, re.M).group(1)
        identity['apo_sets'].append({
            'label': entry['label'], 'mode': entry['mode'],
            'directory': str(Path(entry['directory']).resolve()),
            'topology_sha256': sha256_file(
                Path(entry['directory']) / 'topology.cif'),
            'samples_sha256': sha256_file(
                Path(entry['directory']) / 'samples.xtc'),
            'frames': len(frames), 'atoms': int(universe.n_atoms),
            'command': (Path(entry['directory']) / 'command.txt').read_text().strip()
            if (Path(entry['directory']) / 'command.txt').exists() else '',
            'system': system,
            'msa': _msa_identity(entry['directory'], sequence),
            'boundary': ('independent BioEmu samples, not a physical '
                         'trajectory, equilibrium population, or probability; '
                         'apo only; no ligand or metal inference'),
        })
        sources = {'af3_' + key: value for key, value in af3.items()}
        sources.update({'boltz_' + key: value for key, value in boltz.items()})
        for source, models in sources.items():
            for model in models:
                for frame in frames:
                    result = _compare(model['ca'], frame['ca'])
                    if result is None:
                        continue
                    rows.append({'apo_label': entry['label'],
                                 'apo_mode': entry['mode'],
                                 'source': source,
                                 'model': model['label'],
                                 'apo_frame': frame['label'], **result})
    with (output / 'apo_compatibility.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(
            {key for row in rows for key in row}), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for row in rows:
        key = (row['apo_label'], row['apo_mode'], row['source'])
        summary.setdefault(key, {}).setdefault(
            'local_CA_RMSD_A', []).append(row['local_CA_RMSD_A'])
        summary[key].setdefault(
            'nudix_CA_RMSD_A', []).append(row['nudix_CA_RMSD_A'])
        summary[key].setdefault(
            'alignment_CA_RMSD_A', []).append(row['alignment_CA_RMSD_A'])
    summary_rows = []
    for key, values in sorted(summary.items()):
        entry = {'apo_label': key[0], 'apo_mode': key[1], 'source': key[2]}
        for metric, items in values.items():
            entry[metric] = _summary(items)
        summary_rows.append(entry)
    with (output / 'apo_compatibility_summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]),
                                lineterminator='\n') if summary_rows else None
        if writer:
            writer.writeheader()
            writer.writerows(summary_rows)
    identity['model'] = 'descriptive comparison only'
    write_json(output / 'apo_identity.json', identity)
    return {'pairs': len(rows), 'output': str(output)}
