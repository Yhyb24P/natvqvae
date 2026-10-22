"""Build, run, and summarize the bounded NatV Nudix metal-site states."""
from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from .data import load_yaml, sha256_file, write_json
from .design import _cif_sequence, _kabsch, _source_alignment, _write_msa

DOMAINS = {'NMNAT': (4, 181), 'Nudix': (204, 338)}
SIDECHAIN_O = {'E': ('OE1', 'OE2'), 'Q': ('OE1',), 'D': ('OD1', 'OD2'), 'N': ('OD1',)}
AMIDE_N = {'Q': 'NE2', 'N': 'ND2'}


def _sequences(config):
    rows = list(csv.DictReader(open(config['paths']['candidates'], newline='')))
    sequences = {row['name']: row['sequence'] for row in rows}
    selected = {name: sequences[name] for name in config['sequences']}
    for name, sequence in selected.items():
        if len(sequence) != 341:
            raise ValueError(f'Unexpected sequence length for {name}')
    return selected


def _state_file(config, output, state, sequence, constrained):
    suffix = '' if constrained else '_nocon'
    pattern = config['states'][state]['input'].format(sequence=sequence)
    return Path(output) / 'inputs' / f'{pattern}{suffix}.yaml'


def _yaml_text(config, state, sequence, msa_path, constrained):
    item = config['states'][state]
    lines = ['version: 1', 'sequences:', '  - protein:',
             '      id: A', f'      sequence: {sequence}', f'      msa: {msa_path.resolve()}']
    lines += ['  - ligand:', '      id: B', f"      ccd: {item['ligand']}"]
    for index in range(int(item['ions'])):
        lines += ['  - ligand:', f'      id: {chr(67 + index)}', f"      ccd: {item['ion']}"]
    if constrained:
        contacts = ', '.join(f'[A, {position}]' for position in config['pocket_contacts'])
        lines += ['constraints:', '  - pocket:', '      binder: B', f'      contacts: [{contacts}]']
    return '\n'.join(lines) + '\n'


def build_microenvironment(config_path, output):
    config = load_yaml(config_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    inputs = output / 'inputs'
    inputs.mkdir(exist_ok=True)
    sequences = _sequences(config)
    source_msa = config['paths']['msa_source']
    msa_source_sequence = sequences['WT']
    manifest = {'config': str(config_path), 'config_sha256': sha256_file(config_path),
                'candidates_sha256': sha256_file(config['paths']['candidates']),
                'msa_source_sha256': sha256_file(source_msa),
                'source_structure': config['source']['path'],
                'source_structure_sha256': sha256_file(config['source']['path']),
                'sequences': {name: hashlib_sha(sequence) for name, sequence in sequences.items()},
                'states': {}, 'inputs': []}
    for name, sequence in sequences.items():
        msa = inputs / f'{name}.a3m'
        _write_msa(source_msa, msa, name, sequence, int(config['msa_rows']), msa_source_sequence)
        for state in config['states']:
            for constrained in (True, False):
                path = _state_file(config, output, state, name, constrained)
                path.write_text(_yaml_text(config, state, sequence, msa, constrained))
                manifest['inputs'].append({'path': str(path), 'sha256': sha256_file(path)})
    for state, item in config['states'].items():
        manifest['states'][state] = {'ligand': item['ligand'], 'ion': item['ion'],
                                     'ions': int(item['ions']),
                                     'source': item.get('source', '')}
    write_json(output / 'build.json', manifest)
    return manifest


def hashlib_sha(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


def _model_paths(root):
    return sorted(Path(root).glob('boltz_results_*/predictions/*/*_model_*.cif'))


def _boltz_runs(config, output):
    for run in config['runs']:
        for name in config['sequences']:
            yield (run['state'], run['kind'], name, int(run['seed']),
                   Path(output) / run['pattern'].format(sequence=name))


def run_microenvironment_boltz(config_path, output):
    config = load_yaml(config_path)
    output = Path(output)
    binary = config['boltz']['binary']
    env = dict(os.environ)
    env['BOLTZ_CACHE'] = config['boltz']['cache']
    log_dir = output / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for state, kind, name, seed, run_dir in _boltz_runs(config, output):
        constrained = kind == 'constrained'
        yaml_path = _state_file(config, output, state, name, constrained)
        if not yaml_path.exists():
            raise FileNotFoundError(f'Build inputs first: {yaml_path}')
        expected = int(config['boltz']['diffusion_samples'])
        existing = _model_paths(run_dir)
        if len(existing) == expected:
            records.append({'state': state, 'kind': kind, 'sequence': name, 'seed': seed,
                            'output': str(run_dir), 'status': 'SKIPPED_EXISTING'})
            continue
        if existing:
            raise RuntimeError(f'Incomplete existing run ({len(existing)}/{expected}): {run_dir}')
        command = [binary, 'predict', str(yaml_path), '--out_dir', str(run_dir),
                   '--accelerator', 'gpu', '--devices', '1',
                   '--diffusion_samples', str(config['boltz']['diffusion_samples']),
                   '--sampling_steps', str(config['boltz']['sampling_steps']),
                   '--recycling_steps', str(config['boltz']['recycling_steps']),
                   '--seed', str(seed), '--output_format', config['boltz']['output_format'],
                   '--num_workers', str(config['boltz']['num_workers'])]
        started = time.monotonic()
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        log = log_dir / f'{state}_{kind}_{name}_seed{seed}.log'
        log.write_text(result.stdout + result.stderr)
        record = {'state': state, 'kind': kind, 'sequence': name, 'seed': seed,
                  'output': str(run_dir), 'command': command, 'returncode': result.returncode,
                  'seconds': time.monotonic() - started, 'log': str(log),
                  'status': 'COMPLETED' if result.returncode == 0 else 'FAILED'}
        records.append(record)
        if result.returncode != 0:
            raise RuntimeError(f'Boltz failed: {state} {name} seed {seed}; see {log}')
    with (output / 'boltz_runs.jsonl').open('w') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    return {'status': 'COMPLETED', 'runs': len(records)}


def _metal_atoms(structure):
    atoms = []
    for residue in structure.get_residues():
        if residue.id[0].startswith('H_') and residue.resname in ('MN', 'MG'):
            for atom in residue:
                if atom.element != 'H':
                    atoms.append((residue.get_parent().id, residue.resname, residue.id[1], atom))
    return atoms


def _atom_metric(chain, position, names, metals):
    try:
        residue = chain[position]
    except KeyError:
        return None, None
    values = []
    for atom in residue:
        if atom.name not in names or atom.element == 'H':
            continue
        metal = min(metals, key=lambda item: math.dist(atom.coord, item[3].coord))
        values.append((math.dist(atom.coord, metal[3].coord), f'{metal[1]}:{metal[0]}:{metal[2]}'))
    if not values:
        return None, None
    best = min(values)
    return round(best[0], 2), best[1]


def _ligand_site(structure, ligand_name):
    ligand = next((residue for residue in structure.get_residues()
                   if residue.id[0].startswith('H_') and residue.resname == ligand_name), None)
    if ligand is None:
        return None
    center = np.mean([atom.coord for atom in ligand if atom.element != 'H'], axis=0)
    chain = structure['A']
    result = {}
    for label, (start, end) in DOMAINS.items():
        positions = [chain[i]['CA'].coord for i in range(start, end + 1)
                     if i in chain and 'CA' in chain[i]]
        result[label] = float(min(np.linalg.norm(center - position) for position in positions))
    return result


def _l9_metric(structure, source_mapping):
    model = {int(residue.id[1]): residue['CA'].coord.astype(float)
             for residue in structure['A'] if residue.id[0] == ' ' and 'CA' in residue}
    pairs = [(position + 1, residue['CA'].coord.astype(float))
             for position, residue in source_mapping.items()
             if 204 <= position + 1 <= 338 and 'CA' in residue]
    source_nudix = np.array([coordinate for _, coordinate in pairs])
    model_nudix = np.array([model[position] for position, _ in pairs])
    _, rotation, source_center, target_center = _kabsch(source_nudix, model_nudix)
    l9_positions = sorted(position for position, residue in source_mapping.items()
                          if 302 <= int(residue.id[1]) <= 314)
    source_l9 = np.array([source_mapping[position]['CA'].coord.astype(float)
                          for position in l9_positions])
    model_l9 = np.array([model[position + 1] for position in l9_positions])
    aligned = (source_l9 - source_center) @ rotation + target_center
    return float(np.sqrt(np.mean(np.sum((aligned - model_l9) ** 2, axis=1))))


def _ligand_atom_metric(structure, ligand_name, names, metals):
    ligand = next((residue for residue in structure.get_residues()
                   if residue.id[0].startswith('H_') and residue.resname == ligand_name), None)
    if ligand is None:
        return None, None
    values = []
    for atom in ligand:
        if atom.name not in names or atom.element == 'H':
            continue
        metal = min(metals, key=lambda item: math.dist(atom.coord, item[3].coord))
        values.append((math.dist(atom.coord, metal[3].coord), f'{metal[1]}:{metal[0]}:{metal[2]}'))
    if not values:
        return None, None
    best = min(values)
    return round(best[0], 2), best[1]


def _nad_rmsd(structure, reference, source_mapping):
    model = {int(residue.id[1]): residue['CA'].coord.astype(float)
             for residue in structure['A'] if residue.id[0] == ' ' and 'CA' in residue}
    other = {int(residue.id[1]): residue['CA'].coord.astype(float)
             for residue in reference['A'] if residue.id[0] == ' ' and 'CA' in residue}
    pairs = [(position + 1, residue['CA'].coord.astype(float))
             for position, residue in source_mapping.items()
             if 204 <= position + 1 <= 338 and 'CA' in residue]
    model_nudix = np.array([model[position] for position, _ in pairs])
    reference_nudix = np.array([other[position] for position, _ in pairs])
    _, rotation, reference_center, model_center = _kabsch(reference_nudix, model_nudix)
    model_ligand = next(residue for residue in structure.get_residues()
                        if residue.id[0].startswith('H_') and residue.resname == 'NAD')
    reference_ligand = next(residue for residue in reference.get_residues()
                            if residue.id[0].startswith('H_') and residue.resname == 'NAD')
    model_atoms = {atom.name: atom.coord.astype(float) for atom in model_ligand if atom.element != 'H'}
    reference_atoms = {atom.name: atom.coord.astype(float) for atom in reference_ligand if atom.element != 'H'}
    common = sorted(set(model_atoms) & set(reference_atoms))
    transformed = (np.array([model_atoms[name] for name in common]) - model_center) @ rotation.T + reference_center
    target = np.array([reference_atoms[name] for name in common])
    return float(np.sqrt(np.mean(np.sum((transformed - target) ** 2, axis=1))))


def _validate_model(path, sequence, state, config):
    from Bio.PDB import MMCIFParser
    if _cif_sequence(path) != sequence:
        raise ValueError(f'Model sequence differs from configuration: {path}')
    structure = MMCIFParser(QUIET=True).get_structure(str(path), str(path))[0]
    ligands = [residue for residue in structure.get_residues()
               if residue.id[0].startswith('H_') and residue.resname == config['states'][state]['ligand']]
    ions = [residue for residue in structure.get_residues()
            if residue.id[0].startswith('H_') and residue.resname in ('MN', 'MG')]
    if len(ligands) != 1:
        raise ValueError(f'Expected one {config["states"][state]["ligand"]} in {path}')
    if len(ions) != int(config['states'][state]['ions']) or any(
            residue.resname != config['states'][state]['ion'] for residue in ions):
        raise ValueError(f'Ion composition differs from configuration: {path}')
    return structure


def _confidence(path):
    confidence_path = path.parent / f'confidence_{path.stem}.json'
    if not confidence_path.exists():
        return {}
    values = json.load(open(confidence_path))
    return {key: round(float(values[key]), 3)
            for key in ('ptm', 'ligand_iptm', 'complex_plddt') if key in values}


def _range(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) not in (None, '')]
    if not values:
        return None
    return {'min': round(min(values), 3), 'max': round(max(values), 3),
            'median': round(float(np.median(values)), 3)}


def summarize_microenvironment(config_path, output):
    from Bio.PDB import MMCIFParser
    config = load_yaml(config_path)
    output = Path(output)
    sequences = _sequences(config)
    source = _source_alignment(Path(config['source']['path']), config['source']['chain'],
                               sequences['WT'])[2]
    reference = {}
    for state, relative in config.get('reference_models', {}).items():
        path = output / relative
        if not path.exists():
            raise FileNotFoundError(f'Missing reference model: {path}')
        reference[state] = MMCIFParser(QUIET=True).get_structure(str(path), str(path))[0]
    expected = int(config['boltz']['diffusion_samples'])
    mutations = config.get('mutations', {})
    rows = []
    for run in config['runs']:
        state = run['state']
        kind = run['kind']
        seed = int(run['seed'])
        ligand_name = config['states'][state]['ligand']
        phosphate_names = {'AMP': {'O1P', 'O2P', 'O3P'},
                           'NAD': {'O1A', 'O2A', 'O1N', 'O2N'}}[ligand_name]
        for name in config['sequences']:
            run_dir = output / run['pattern'].format(sequence=name)
            paths = _model_paths(run_dir)
            if len(paths) != expected:
                raise ValueError(f'Incomplete run ({len(paths)}/{expected}): {run_dir}')
            mutation = mutations.get(name, {})
            residue_248 = mutation.get('mut') if mutation.get('position') == 248 else 'E'
            residue_308 = mutation.get('mut') if mutation.get('position') == 308 else 'D'
            amide_248 = AMIDE_N.get(residue_248)
            amide_308 = AMIDE_N.get(residue_308)
            for path in paths:
                structure = _validate_model(path, sequences[name], state, config)
                chain = structure['A']
                metals = _metal_atoms(structure)
                site = _ligand_site(structure, ligand_name)
                e_o = _atom_metric(chain, 248, SIDECHAIN_O[residue_248], metals)
                e_n = _atom_metric(chain, 248, {amide_248}, metals) if amide_248 else (None, None)
                d_o = _atom_metric(chain, 308, SIDECHAIN_O[residue_308], metals)
                d_n = _atom_metric(chain, 308, {amide_308}, metals) if amide_308 else (None, None)
                phosphate = _ligand_atom_metric(structure, ligand_name, phosphate_names, metals)
                rows.append({
                    'state': state, 'kind': kind, 'sequence': name, 'seed': seed,
                    'model': path.name,
                    'model_path': str(path.relative_to(output)),
                    'model_sha256': sha256_file(path),
                    'ligand_site': min(site, key=site.get),
                    'site_nmnat_min_ca_A': round(site['NMNAT'], 1),
                    'site_nudix_min_ca_A': round(site['Nudix'], 1),
                    'E248_residue': residue_248,
                    'E248_O_min_A': e_o[0],
                    'E248_O_metal': e_o[1],
                    'E248_amide_N_min_A': e_n[0],
                    'E248_amide_N_metal': e_n[1],
                    'D308_residue': residue_308,
                    'D308_O_min_A': d_o[0],
                    'D308_O_metal': d_o[1],
                    'D308_amide_N_min_A': d_n[0],
                    'D308_amide_N_metal': d_n[1],
                    'E252_min_A': _atom_metric(chain, 252, SIDECHAIN_O['E'], metals)[0],
                    'D307_OD1_A': _atom_metric(chain, 307, {'OD1'}, metals)[0],
                    'D307_OD2_A': _atom_metric(chain, 307, {'OD2'}, metals)[0],
                    'G232O_min_A': _atom_metric(chain, 232, {'O'}, metals)[0],
                    'l9_ca_rmsd_A': round(_l9_metric(structure, source), 2),
                    'ligand_phosphate_O_min_A': phosphate[0],
                    'ligand_phosphate_O_metal': phosphate[1],
                    'ligand_O2A_min_A': _ligand_atom_metric(structure, ligand_name, {'O2A'}, metals)[0],
                    'ligand_O2N_min_A': _ligand_atom_metric(structure, ligand_name, {'O2N'}, metals)[0],
                    'ligand_O2P_min_A': _ligand_atom_metric(structure, ligand_name, {'O2P'}, metals)[0],
                    'ligand_O3P_min_A': _ligand_atom_metric(structure, ligand_name, {'O3P'}, metals)[0],
                    'nad_rmsd_to_wt_model0_A': round(_nad_rmsd(structure, reference['D4'], source), 2)
                    if state in reference and ligand_name == 'NAD' else None,
                    **_confidence(path),
                })
    if not rows:
        raise ValueError('No model predictions found')
    with (output / 'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    metric_keys = ['E248_O_min_A', 'E248_amide_N_min_A', 'D308_O_min_A', 'D308_amide_N_min_A',
                   'E252_min_A', 'D307_OD1_A', 'D307_OD2_A', 'G232O_min_A', 'l9_ca_rmsd_A',
                   'ligand_phosphate_O_min_A', 'nad_rmsd_to_wt_model0_A',
                   'ptm', 'ligand_iptm', 'complex_plddt']
    ranges = {}
    for state in config['states']:
        for kind in ('constrained', 'unconstrained'):
            for name in config['sequences']:
                subset = [row for row in rows if row['state'] == state
                          and row['kind'] == kind and row['sequence'] == name]
                if subset:
                    ranges[f'{state}:{kind}:{name}'] = {
                        key: value for key in metric_keys
                        if (value := _range(subset, key)) is not None}
    summary = {
        'status': 'COMPLETED',
        'tool': config.get('tool', {}),
        'config_sha256': sha256_file(config_path),
        'candidates_sha256': sha256_file(config['paths']['candidates']),
        'msa_source_sha256': sha256_file(config['paths']['msa_source']),
        'source_structure_sha256': sha256_file(config['source']['path']),
        'runs': config['runs'],
        'model_rows': len(rows),
        'models_by_state': {state: sum(1 for row in rows if row['state'] == state)
                            for state in config['states']},
        'validation': {'models_per_run': expected, 'sequence_checked': True,
                       'ligand_ion_checked': True},
        'metrics_file': 'metrics.csv',
        'models_identity': 'metrics.csv: model_path, model_sha256',
        'ranges': ranges,
        'metrics': ['ligand_site', 'E248_O', 'E248_amide_N', 'D308_O', 'D308_amide_N',
                    'E252', 'D307_OD1', 'D307_OD2', 'G232O', 'l9_ca_rmsd',
                    'ligand_phosphate_O', 'nad_rmsd_to_wt_model0'],
    }
    write_json(output / 'summary.json', summary)
    return summary
