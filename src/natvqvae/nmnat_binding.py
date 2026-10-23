"""Build, run, and summarize NMNAT substrate-binding co-folding inputs."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import Target, load_yaml, sha256_file, write_json
from .design import _write_msa
from .nmnat_mapping import _kabsch, _open_structure


def _sequences(config):
    rows = list(csv.DictReader(open(config['candidates'], newline='')))
    sequences = {row['name']: row['sequence'] for row in rows}
    selected = {name: sequences[name] for name in config['sequences']}
    for name, sequence in selected.items():
        if len(sequence) != 341:
            raise ValueError(f'Unexpected sequence length for {name}')
    return selected


def _input_path(config, output, state, mode, name, seed):
    return Path(output) / 'inputs' / f'seed{seed}' / f'{state}_{mode}_{name}_seed{seed}.yaml'


def _yaml_text(config, state, sequence, msa_path, mode):
    item = config['states'][state]
    lines = ['version: 1', 'sequences:', '  - protein:',
             '      id: A', f'      sequence: {sequence}', f'      msa: {msa_path.resolve()}']
    lines += ['  - ligand:', '      id: B', f"      ccd: {item['ligand']}"]
    for index, ion in enumerate(item.get('ions', [])):
        lines += ['  - ligand:', f'      id: {chr(67 + index)}', f'      ccd: {ion}']
    if mode == 'anchor':
        contacts = ', '.join(f'[A, {position}]' for position in config['constraints'][state])
        lines += ['constraints:', '  - pocket:', '      binder: B', f'      contacts: [{contacts}]']
        pocket = config.get('pocket', {})
        lines += [f"      max_distance: {float(pocket.get('max_distance', 6.0))}",
                  f"      force: {str(bool(pocket.get('force', False))).lower()}"]
    return '\n'.join(lines) + '\n'


def build_nmnat_binding(config_path, output):
    config = load_yaml(config_path)
    target = Target.load(config['target'])
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'NMNAT binding output already exists: {output}')
    output.mkdir(parents=True, exist_ok=True)
    inputs = output / 'inputs'
    inputs.mkdir()
    sequences = _sequences(config)
    if sequences['WT'] != target.sequence:
        raise ValueError('Candidate WT differs from target')
    msa_source = Path(config['msa_source']).resolve()
    manifest = {
        'config': str(Path(config_path).resolve()),
        'config_sha256': sha256_file(config_path),
        'config_values': config,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'code_sha256': sha256_file(__file__),
        'target': config['target'],
        'target_sha256': sha256_file(config['target']),
        'candidates': config['candidates'],
        'candidates_sha256': sha256_file(config['candidates']),
        'msa_source': str(msa_source),
        'msa_source_sha256': sha256_file(msa_source),
        'msa_files': [],
        'msa_rows': int(config['msa_rows']),
        'sequences': {name: hashlib.sha256(sequence.encode()).hexdigest()
                      for name, sequence in sequences.items()},
        'states': config['states'], 'constraints': config['constraints'],
        'modes': config['modes'], 'seeds': config['seeds'],
        'inputs': [],
    }
    for name, sequence in sequences.items():
        msa = inputs / f'{name}.a3m'
        _write_msa(msa_source, msa, name, sequence, int(config['msa_rows']),
                   target.sequence)
        manifest['msa_files'].append({'path': str(msa.resolve()),
                                      'sha256': sha256_file(msa)})
        for state in config['states']:
            for mode in config['modes'][state]:
                for seed in config['seeds']:
                    path = _input_path(config, output, state, mode, name, seed)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(_yaml_text(config, state, sequence, msa, mode))
                    manifest['inputs'].append({
                        'state': state, 'mode': mode, 'sequence': name,
                        'seed': int(seed), 'path': str(path),
                        'sha256': sha256_file(path),
                    })
    write_json(output / 'build.json', manifest)
    return manifest


def _model_paths(run_dir):
    return sorted(Path(run_dir).glob('boltz_results_*/predictions/*/*_model_*.cif'))


def _stage_checkpoint(source, directory, expected_sha256):
    """Stage only this checkpoint off slow WSL 9p; verify bytes before use."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / Path(source).name
    if destination.exists():
        if sha256_file(destination) != expected_sha256:
            raise ValueError(f'Existing staged checkpoint has wrong SHA: {destination}')
        return destination
    temporary = destination.with_suffix('.partial')
    size = Path(source).stat().st_size
    output_fd = os.open(temporary, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.ftruncate(output_fd, size)
        chunk = (size + 3) // 4
        def copy_section(index):
            start, end = index * chunk, min((index + 1) * chunk, size)
            with open(source, 'rb', buffering=0) as handle:
                handle.seek(start)
                while start < end:
                    block = handle.read(min(8 << 20, end - start))
                    if not block:
                        raise IOError('Checkpoint source ended unexpectedly')
                    written = os.pwrite(output_fd, block, start)
                    if written != len(block):
                        raise IOError('Incomplete staged checkpoint write')
                    start += written
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(copy_section, range(4)))
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        os.close(output_fd)
    if sha256_file(temporary) != expected_sha256:
        temporary.unlink()
        raise ValueError('Staged checkpoint differs from expected SHA')
    temporary.replace(destination)
    return destination


def run_nmnat_binding_boltz(config_path, output, wait_for_pid=None,
                           checkpoint_stage_dir=None):
    config = load_yaml(config_path)
    output = Path(output)
    build = json.load(open(output / 'build.json'))
    if sha256_file(config_path) != build['config_sha256']:
        raise ValueError('Run config differs from built inputs')
    for item in build['inputs'] + build['msa_files']:
        if sha256_file(item['path']) != item['sha256']:
            raise ValueError(f"Built input changed: {item['path']}")
    binary = config['boltz']['binary']
    env = dict(os.environ)
    env['BOLTZ_CACHE'] = config['boltz']['cache']
    logs = output / 'logs'
    logs.mkdir(exist_ok=True)
    write_json(output / 'run_status.json', {'status': 'CHECKING_WEIGHTS',
                                           'pid': os.getpid()})
    checkpoint = config['boltz']['checkpoint']
    if checkpoint_stage_dir:
        expected_sha = config['boltz']['checkpoint_sha256']
        write_json(output / 'run_status.json', {'status': 'STAGING_WEIGHTS',
                                               'pid': os.getpid(),
                                               'source': checkpoint,
                                               'stage_dir': checkpoint_stage_dir})
        checkpoint = str(_stage_checkpoint(checkpoint, checkpoint_stage_dir, expected_sha))
    identity = {
        'config_sha256': sha256_file(config_path),
        'build_sha256': sha256_file(output / 'build.json'),
        'code_sha256': sha256_file(__file__),
        'checkpoint_source': config['boltz']['checkpoint'],
        'checkpoint': checkpoint,
        'checkpoint_sha256': sha256_file(checkpoint),
        'boltz': config['boltz'],
    }
    expected_sha = config['boltz'].get('checkpoint_sha256')
    if expected_sha and identity['checkpoint_sha256'] != expected_sha:
        raise ValueError('Actual checkpoint does not match configured SHA')
    versions = subprocess.run(
        [str(Path(binary).with_name('python')), '-c',
         'import importlib.metadata,json; print(json.dumps({k:importlib.metadata.version(k) for k in ("boltz","torch")}))'],
        capture_output=True, text=True, check=True)
    write_json(output / 'tool_versions.json', json.loads(versions.stdout))
    identity_path = output / 'run_identity.json'
    if identity_path.exists() and json.load(open(identity_path)) != identity:
        raise ValueError('Existing run identity differs; refusing to mix runs')
    write_json(identity_path, identity)
    # Resource coordination only; this does not change scientific inputs.
    if wait_for_pid is not None:
        process_path = Path('/proc') / str(int(wait_for_pid))
        while process_path.exists():
            write_json(output / 'run_status.json', {'status': 'WAITING_FOR_PROCESS',
                                                   'pid': os.getpid(),
                                                   'wait_for_pid': int(wait_for_pid)})
            time.sleep(5)
    records = []
    for seed in build['seeds']:
        items = [item for item in build['inputs'] if item['seed'] == seed]
        input_dir = output / 'inputs' / f'seed{seed}'
        run_dir = output / 'runs' / f'seed{seed}'
        expected = int(config['boltz']['diffusion_samples'])
        existing = _model_paths(run_dir)
        if len(existing) == expected * len(items):
            records.extend({**item, 'output': str(run_dir), 'status': 'SKIPPED_EXISTING'}
                           for item in items)
            continue
        if existing:
            raise RuntimeError(f'Incomplete existing seed; retain results and inspect: {run_dir}')
        command = [binary, 'predict', str(input_dir), '--out_dir', str(run_dir),
                   '--checkpoint', checkpoint,
                   '--accelerator', 'gpu', '--devices', '1',
                   '--diffusion_samples', str(expected),
                   '--sampling_steps', str(config['boltz']['sampling_steps']),
                   '--recycling_steps', str(config['boltz']['recycling_steps']),
                   '--seed', str(seed), '--output_format', config['boltz']['output_format'],
                   '--num_workers', str(config['boltz']['num_workers'])]
        started = time.monotonic()
        log = logs / f'seed{seed}.log'
        write_json(output / 'run_status.json', {'status': 'RUNNING', 'seed': seed,
                                               'pid': os.getpid(), 'command': command,
                                               'log': str(log)})
        try:
            with log.open('a', buffering=1) as handle:
                result = subprocess.run(command, env=env, stdout=handle,
                                        stderr=subprocess.STDOUT,
                                        timeout=int(config['boltz'].get('timeout_seconds', 7200)))
        except subprocess.TimeoutExpired:
            write_json(output / 'run_status.json', {'status': 'TIMEOUT', 'seed': seed,
                                                   'log': str(log)})
            raise
        record = {'output': str(run_dir), 'command': command,
                  'returncode': result.returncode,
                  'seconds': round(time.monotonic() - started, 1),
                  'log': str(log),
                  'status': 'COMPLETED' if result.returncode == 0 else 'FAILED'}
        for item in items:
            predictions = [p for p in _model_paths(run_dir)
                           if p.parent.name == Path(item['path']).stem]
            records.append({**item, **record, 'model_count': len(predictions),
                            'status': 'COMPLETED' if result.returncode == 0 and
                            len(predictions) == expected else 'FAILED'})
        with (output / 'boltz_runs.jsonl').open('w') as handle:
            for row in records:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        if any(row['status'] == 'FAILED' for row in records):
            write_json(output / 'run_status.json', {'status': 'FAILED', 'seed': seed,
                                                   'log': str(log)})
            raise RuntimeError(f'Boltz seed {seed} incomplete; see {log}')
    with (output / 'boltz_runs.jsonl').open('w') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    write_json(output / 'run_status.json', {'status': 'COMPLETED', 'runs': len(records)})
    return {'status': 'COMPLETED', 'runs': len(records)}


def _structure(path):
    return _open_structure(path)[0]


def _protein_atoms(structure):
    return [(residue, atom) for residue in structure['A']
            if residue.id[0] == ' ' and 1 <= int(residue.id[1]) <= 341
            for atom in residue if atom.element != 'H']


def _ligand_atoms(structure, name):
    residue = next((residue for residue in structure.get_residues()
                    if residue.id[0].startswith('H_') and residue.resname == name), None)
    if residue is None:
        raise ValueError(f'Ligand {name} not found')
    return residue, [(atom.name, atom.element, atom.coord.astype(float))
                     for atom in residue if atom.element != 'H']


def _metal_atoms(structure):
    return [atom.coord.astype(float) for residue in structure.get_residues()
            if residue.id[0].startswith('H_') and residue.resname in ('MG', 'MN')
            for atom in residue if atom.element != 'H']


def _min_pair(points_a, points_b):
    best = None
    for label_a, a in points_a:
        for label_b, b in points_b:
            distance = float(np.linalg.norm(a - b))
            if best is None or distance < best[0]:
                best = (distance, label_a, label_b)
    return best


def _contacts(structure, ligand_name, cutoff, clash_cutoff):
    _, ligand = _ligand_atoms(structure, ligand_name)
    protein = _protein_atoms(structure)
    rows = []
    for atom_name, _, coord in ligand:
        for residue, atom in protein:
            distance = float(np.linalg.norm(coord - atom.coord))
            if distance <= cutoff:
                rows.append({
                    'ligand_atom': atom_name, 'protein_resnum': int(residue.id[1]),
                    'protein_resname': residue.resname,
                    'protein_atom': atom.name, 'distance_A': round(distance, 2),
                    'severe': distance < clash_cutoff,
                })
    return rows


def _anchor_metrics(structure, ligand_name, anchors, cutoff):
    _, ligand = _ligand_atoms(structure, ligand_name)
    ligand_points = [(name, coord) for name, _, coord in ligand]
    output = []
    for position in anchors:
        residue = next((residue for residue in structure['A']
                        if residue.id[0] == ' ' and int(residue.id[1]) == position), None)
        if residue is None:
            output.append({'position': position, 'sidechain_min_A': None,
                           'ligand_atom': '', 'protein_atom': '', 'severe': False})
            continue
        sidechain = [(atom.name, atom.coord.astype(float)) for atom in residue
                     if atom.name not in {'N', 'CA', 'C', 'O', 'OXT'} and atom.element != 'H']
        best = _min_pair([(name, coord) for name, coord in sidechain], ligand_points)
        output.append({
            'position': position,
            'sidechain_min_A': round(best[0], 2) if best else None,
            'ligand_atom': best[2] if best else '',
            'protein_atom': best[1] if best else '',
            'severe': bool(best and best[0] < cutoff),
        })
    return output


def _loop_metrics(structure, ligand_name, loop, reference_ca):
    _, ligand = _ligand_atoms(structure, ligand_name)
    ligand_points = [(name, coord) for name, _, coord in ligand]
    protein = _protein_atoms(structure)
    loop_residues = [residue for residue in structure['A']
                     if residue.id[0] == ' ' and loop[0] <= int(residue.id[1]) <= loop[1]]
    loop_points = [(atom.name, atom.coord.astype(float))
                   for residue in loop_residues for atom in residue if atom.element != 'H']
    best = _min_pair(loop_points, ligand_points)
    contact_count = sum(1 for _, coord in loop_points
                        for _, lcoord in ligand_points
                        if float(np.linalg.norm(coord - lcoord)) <= 4.0)
    model_ca = []
    for residue in structure['A']:
        if residue.id[0] == ' ' and 4 <= int(residue.id[1]) <= 181 and 'CA' in residue:
            model_ca.append((int(residue.id[1]), residue['CA'].coord.astype(float)))
    loop_ca = []
    for residue in loop_residues:
        if 'CA' in residue:
            loop_ca.append(residue['CA'].coord.astype(float))
    displacement = None
    if reference_ca:
        source = np.asarray([coord for _, coord in model_ca], dtype=float)
        target = np.asarray([reference_ca[position] for position, _ in model_ca], dtype=float)
        if len(source) == len(target) and len(source) >= 20:
            _, rotation, translation = _kabsch(source, target)
            aligned = loop_ca @ rotation.T + translation
            reference_loop = [reference_ca[residue.id[1]] for residue in loop_residues
                              if residue.id[1] in reference_ca]
            if len(aligned) == len(reference_loop):
                displacement = float(np.sqrt(np.mean(
                    np.sum((aligned - np.asarray(reference_loop)) ** 2, axis=1))))
    return {
        'loop_min_A': round(best[0], 2) if best else None,
        'loop_contact_atom_pairs_4A': contact_count,
        'loop_ca_displacement_A': round(displacement, 2) if displacement is not None else None,
    }


def _metal_metrics(structure, ligand_name, cutoff):
    _, ligand = _ligand_atoms(structure, ligand_name)
    ligand_oxygen = [(name, coord) for name, element, coord in ligand
                     if element == 'O']
    metals = _metal_atoms(structure)
    protein = _protein_atoms(structure)
    if not metals:
        return {'metal_count': 0, 'metal_ligand_O_min_A': None,
                'metal_protein_min_A': None, 'protein_atoms_within_cutoff': None}
    metal = metals[0]
    ligand_best = min((float(np.linalg.norm(metal - coord)), name)
                      for name, coord in ligand_oxygen)
    protein_best = min((float(np.linalg.norm(metal - atom.coord)),
                        f'{residue.resname}{residue.id[1]}:{atom.name}')
                       for residue, atom in protein)
    within = sum(1 for _, atom in protein if np.linalg.norm(metal - atom.coord) <= cutoff)
    return {'metal_count': len(metals),
            'metal_ligand_O_min_A': round(ligand_best[0], 2),
            'metal_ligand_O_atom': ligand_best[1],
            'metal_protein_min_A': round(protein_best[0], 2),
            'metal_protein_atom': protein_best[1],
            'protein_atoms_within_cutoff': within}


def _confidence(path):
    confidence_path = path.parent / f'confidence_{path.stem}.json'
    if not confidence_path.exists():
        return {}
    values = json.load(open(confidence_path))
    return {key: values[key] for key in ('ptm', 'ligand_iptm', 'complex_plddt')
            if key in values}


def _reference_ca(path):
    if not path:
        return {}
    structure = _structure(path)
    return {int(residue.id[1]): residue['CA'].coord.astype(float)
            for residue in structure['A']
            if residue.id[0] == ' ' and 4 <= int(residue.id[1]) <= 181 and 'CA' in residue}


def _domain(position):
    if position <= 3:
        return 'N_terminal'
    if position <= 181:
        return 'NMNAT'
    if position <= 203:
        return 'linker'
    if position <= 338:
        return 'Nudix'
    return 'C_terminal'


def _ligand_domain(structure, ligand_name):
    residue, atoms = _ligand_atoms(structure, ligand_name)
    center = np.mean([coord for _, _, coord in atoms], axis=0)
    best = min((float(np.linalg.norm(center - atom.coord)), int(res.id[1]))
               for res, atom in _protein_atoms(structure))
    return _domain(best[1]), best[1]


def _heavy_atom_contacts(structure, ligand_name, cutoff):
    _, ligand = _ligand_atoms(structure, ligand_name)
    rows = []
    for atom_name, element, coord in ligand:
        for residue, atom in _protein_atoms(structure):
            distance = float(np.linalg.norm(coord - atom.coord))
            if distance <= cutoff:
                rows.append({
                    'ligand_atom': atom_name, 'ligand_element': element,
                    'protein_resnum': int(residue.id[1]),
                    'protein_resname': residue.resname,
                    'protein_atom': atom.name, 'protein_element': atom.element.strip(),
                    'distance_A': round(distance, 2),
                    'domain': _domain(int(residue.id[1])),
                    'severe': distance < 2.0,
                })
    return rows


PHOSPHATE_GROUPS = {
    'PA': ('O1A', 'O2A', 'O3A', "O5'"),
    'PB': ('O1B', 'O2B', 'O3B', 'O3A'),
    'PG': ('O1G', 'O2G', 'O3G', 'O3B'),
}
ANCHOR_ATOMS = {14: ('ND1', 'NE2'), 17: ('ND1', 'NE2'), 135: ('OG1',),
                138: ('NE', 'NH1', 'NH2')}


def _anchor_phosphate_distances(structure, ligand_name):
    if ligand_name != 'ATP':
        return []
    residue, ligand = _ligand_atoms(structure, ligand_name)
    lookup = {name: coord for name, _, coord in ligand}
    rows = []
    for position, atom_names in ANCHOR_ATOMS.items():
        target = next((res for res in structure['A']
                       if res.id[0] == ' ' and int(res.id[1]) == position), None)
        if target is None:
            continue
        for atom_name in atom_names:
            if atom_name not in target:
                continue
            coord = target[atom_name].coord.astype(float)
            for group, names in PHOSPHATE_GROUPS.items():
                for ligand_atom in names:
                    if ligand_atom not in lookup:
                        continue
                    rows.append({
                        'anchor_resnum': position, 'anchor_resname': target.resname,
                        'anchor_atom': atom_name, 'phosphate_group': group,
                        'ligand_atom': ligand_atom,
                        'distance_A': round(float(np.linalg.norm(coord - lookup[ligand_atom])), 2),
                    })
    return rows


def _metal_neighbors(structure, cutoff):
    metals = [residue for residue in structure.get_residues()
              if residue.id[0].startswith('H_') and residue.resname in ('MG', 'MN')]
    rows = []
    for metal in metals:
        coord = metal['MG' if metal.resname == 'MG' else 'MN'].coord.astype(float)
        for residue, atom in _protein_atoms(structure):
            distance = float(np.linalg.norm(coord - atom.coord))
            if distance <= cutoff:
                rows.append({
                    'metal_resnum': int(metal.id[1]), 'metal_resname': metal.resname,
                    'neighbor_type': 'protein',
                    'neighbor_resnum': int(residue.id[1]),
                    'neighbor_resname': residue.resname,
                    'neighbor_atom': atom.name, 'neighbor_element': atom.element.strip(),
                    'distance_A': round(distance, 2),
                    'domain': _domain(int(residue.id[1])),
                })
        for ligand in structure.get_residues():
            if not ligand.id[0].startswith('H_') or ligand.resname in ('MG', 'MN'):
                continue
            for atom in ligand:
                if atom.element == 'H':
                    continue
                distance = float(np.linalg.norm(coord - atom.coord))
                if distance <= cutoff:
                    rows.append({
                        'metal_resnum': int(metal.id[1]), 'metal_resname': metal.resname,
                        'neighbor_type': 'nucleotide',
                        'neighbor_resnum': int(ligand.id[1]),
                        'neighbor_resname': ligand.resname,
                        'neighbor_atom': atom.name,
                        'neighbor_element': atom.element.strip(),
                        'distance_A': round(distance, 2),
                        'domain': 'ligand',
                    })
    return rows


def _ligand_bond_check(structure, ligand_name, cache):
    import pickle
    ccd = Path(cache) / 'mols' / f'{ligand_name}.pkl'
    if not ccd.exists():
        return {'bond_graph': 'unavailable', 'bonds_checked': 0,
                'min_bond_A': None, 'max_bond_A': None, 'outliers': []}
    mol = pickle.load(open(ccd, 'rb'))
    names = {atom.GetIdx(): atom.GetPropsAsDict().get('name') for atom in mol.GetAtoms()}
    _, ligand = _ligand_atoms(structure, ligand_name)
    coords = {name: coord for name, _, coord in ligand}
    values = []
    outliers = []
    for bond in mol.GetBonds():
        a, b = names[bond.GetBeginAtomIdx()], names[bond.GetEndAtomIdx()]
        if a not in coords or b not in coords:
            continue
        distance = float(np.linalg.norm(coords[a] - coords[b]))
        values.append(distance)
        if distance < 0.9 or distance > 2.0:
            outliers.append(f'{a}-{b}:{distance:.2f}')
    return {'bond_graph': str(ccd), 'bonds_checked': len(values),
            'min_bond_A': round(min(values), 2) if values else None,
            'max_bond_A': round(max(values), 2) if values else None,
            'outliers': outliers}


def summarize_nmnat_binding(config_path, output):
    config = load_yaml(config_path)
    output = Path(output)
    reference = _reference_ca(config.get('reference_apo'))
    records = [json.loads(line) for line in open(output / 'boltz_runs.jsonl')]
    rows = []
    for record in records:
        if record.get('status') not in {'COMPLETED', 'SKIPPED_EXISTING'}:
            continue
        state, mode, name, seed = (record['state'], record['mode'],
                                   record['sequence'], record['seed'])
        ligand_name = config['states'][state]['ligand']
        for path in _model_paths(record['output']):
            if path.parent.name != Path(record['path']).stem:
                continue
            structure = _structure(path)
            contacts = _contacts(structure, ligand_name,
                                 float(config['analysis']['contact_cutoff_A']),
                                 float(config['analysis']['clash_cutoff_A']))
            anchors = _anchor_metrics(structure, ligand_name,
                                      [int(x) for x in config['analysis']['anchors']],
                                      float(config['analysis']['clash_cutoff_A']))
            loop = _loop_metrics(structure, ligand_name,
                                 [int(x) for x in config['analysis']['atp_loop']],
                                 reference)
            metal = _metal_metrics(structure, ligand_name,
                                   float(config['analysis']['metal_cutoff_A']))
            ligand_residue, ligand_atoms = _ligand_atoms(structure, ligand_name)
            rows.append({
                'state': state, 'mode': mode, 'sequence': name, 'seed': seed,
                'model': path.name,
                'ligand': ligand_name, 'ligand_atoms': len(ligand_atoms),
                'contacts_4A': len(contacts),
                'severe_contacts': sum(1 for row in contacts if row['severe']),
                'min_contact_A': min((row['distance_A'] for row in contacts), default=None),
                'nearest_contact': ';'.join(
                    f"{row['protein_resname']}{row['protein_resnum']}:{row['protein_atom']}"
                    for row in sorted(contacts, key=lambda row: row['distance_A'])[:5]),
                **metal, **loop,
                'anchor_contacts': ';'.join(
                    f"{item['position']}:{item['sidechain_min_A']}:{item['protein_atom']}"
                    f":{item['ligand_atom']}" for item in anchors),
                'anchor_severe': sum(1 for item in anchors if item['severe']),
                **_confidence(path),
            })
    fields = ['state', 'mode', 'sequence', 'seed', 'model', 'ligand', 'ligand_atoms',
              'contacts_4A', 'severe_contacts', 'min_contact_A', 'nearest_contact',
              'metal_count', 'metal_ligand_O_min_A', 'metal_ligand_O_atom',
              'metal_protein_min_A', 'metal_protein_atom', 'protein_atoms_within_cutoff',
              'loop_min_A', 'loop_contact_atom_pairs_4A', 'loop_ca_displacement_A',
              'anchor_contacts', 'anchor_severe', 'ptm', 'ligand_iptm',
              'complex_plddt']
    with (output / 'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore',
                                lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    summary = {'models': len(rows), 'states': {}, 'source': str(output),
               'notes': ['co-folded models are hypotheses, not experimental complexes',
                         'contacts are distance flags, not bonds or affinities',
                         'no explicit water or protonation analysis']}
    for state in config['states']:
        for mode in config['modes'][state]:
            subset = [row for row in rows if row['state'] == state and row['mode'] == mode]
            if not subset:
                continue
            summary['states'][f'{state}_{mode}'] = {
                'n_models': len(subset),
                'contacts_4A': [row['contacts_4A'] for row in subset],
                'severe_contacts': [row['severe_contacts'] for row in subset],
                'anchor_severe': [row['anchor_severe'] for row in subset],
                'metal_ligand_O_min_A': [row['metal_ligand_O_min_A'] for row in subset],
                'metal_protein_min_A': [row['metal_protein_min_A'] for row in subset],
                'loop_min_A': [row['loop_min_A'] for row in subset],
                'loop_ca_displacement_A': [row['loop_ca_displacement_A'] for row in subset],
                'ptm': [row.get('ptm') for row in subset],
            }
    write_json(output / 'summary.json', summary)
    write_binding_details(config_path, output, reference)
    write_ligand_pose_consistency(config_path, output)
    if set(config['states']) == {'ATP_ONLY'}:
        summarize_atp_variants(output)
    else:
        plot_binding_details(output)
    return {'status': 'COMPLETED', 'models': len(rows), 'output': str(output)}


def _metal_global_min(structure, ligand_name):
    metals = [residue for residue in structure.get_residues()
              if residue.id[0].startswith('H_') and residue.resname in ('MG', 'MN')]
    if not metals:
        return {'metal_count': 0, 'global_min_A': None, 'global_atom': ''}
    _, ligand = _ligand_atoms(structure, ligand_name)
    oxygens = [(name, coord) for name, element, coord in ligand if element == 'O']
    best = min((float(np.linalg.norm(metal['MG' if metal.resname == 'MG' else 'MN'].coord - coord)),
                name) for metal in metals for name, coord in oxygens)
    return {'metal_count': len(metals), 'global_min_A': round(best[0], 2),
            'global_atom': best[1]}


def write_binding_details(config_path, output, reference):
    from datetime import datetime, timezone
    config = load_yaml(config_path)
    output = Path(output)
    records = [json.loads(line) for line in open(output / 'boltz_runs.jsonl')]
    run_identity = json.load(open(output / 'run_identity.json')) if (output / 'run_identity.json').exists() else {}
    cache = config['boltz']['cache']
    model_rows, heavy_rows, anchor_rows, metal_rows, bond_rows = [], [], [], [], []
    model_files = []
    for record in records:
        if record.get('status') not in {'COMPLETED', 'SKIPPED_EXISTING'}:
            continue
        state, mode, name, seed = (record['state'], record['mode'],
                                   record['sequence'], record['seed'])
        ligand_name = config['states'][state]['ligand']
        for path in _model_paths(record['output']):
            if path.parent.name != Path(record['path']).stem:
                continue
            structure = _structure(path)
            model_files.append({'path': str(path), 'sha256': sha256_file(path)})
            contacts = _heavy_atom_contacts(structure, ligand_name,
                                            float(config['analysis']['contact_cutoff_A']))
            anchor = _anchor_phosphate_distances(structure, ligand_name)
            metal = _metal_neighbors(structure, float(config['analysis']['metal_cutoff_A']))
            bond = _ligand_bond_check(structure, ligand_name, cache)
            loop = _loop_metrics(structure, ligand_name,
                                 [int(x) for x in config['analysis']['atp_loop']], reference)
            ligand_domain, ligand_domain_res = _ligand_domain(structure, ligand_name)
            anchor_min = {}
            anchor_group = {}
            for row in anchor:
                position = int(row['anchor_resnum'])
                if position not in anchor_min or row['distance_A'] < anchor_min[position]:
                    anchor_min[position] = row['distance_A']
                    anchor_group[position] = f"{row['phosphate_group']}:{row['ligand_atom']}"
            dom = {}
            for row in contacts:
                dom[row['domain']] = dom.get(row['domain'], 0) + 1
            nucleotide = [row for row in metal if row['neighbor_type'] == 'nucleotide']
            protein = [row for row in metal if row['neighbor_type'] == 'protein']
            nearest_protein = min(protein, key=lambda row: row['distance_A']) if protein else None
            global_metal = _metal_global_min(structure, ligand_name)
            base = {'state': state, 'mode': mode, 'sequence': name, 'seed': seed,
                    'model': path.name, 'ligand': ligand_name}
            for row in contacts:
                heavy_rows.append({**base, **row})
            for row in anchor:
                anchor_rows.append({**base, **row})
            for row in metal:
                metal_rows.append({**base, **row})
            bond_rows.append({**base, 'bond_graph': bond['bond_graph'],
                              'bonds_checked': bond['bonds_checked'],
                              'bond_min_A': bond['min_bond_A'], 'bond_max_A': bond['max_bond_A'],
                              'bond_outliers': ';'.join(bond['outliers'])})
            model_rows.append({
                **base, 'ligand_domain': ligand_domain,
                'ligand_domain_resnum': ligand_domain_res,
                'contacts_4A': len(contacts),
                'severe_contacts': sum(1 for row in contacts if row['severe']),
                'min_contact_A': min((row['distance_A'] for row in contacts), default=None),
                'contacts_NMNAT': dom.get('NMNAT', 0), 'contacts_linker': dom.get('linker', 0),
                'contacts_Nudix': dom.get('Nudix', 0),
                'contacts_terminal': dom.get('N_terminal', 0) + dom.get('C_terminal', 0),
                'H14_min_phosphate_A': anchor_min.get(14), 'H14_min_group': anchor_group.get(14),
                'H17_min_phosphate_A': anchor_min.get(17), 'H17_min_group': anchor_group.get(17),
                'T135_min_phosphate_A': anchor_min.get(135), 'T135_min_group': anchor_group.get(135),
                'R138_min_phosphate_A': anchor_min.get(138), 'R138_min_group': anchor_group.get(138),
                'metal_count': len({(row['metal_resnum'], row['metal_resname']) for row in metal}),
                'metal_ligand_O_min_A': min((row['distance_A'] for row in nucleotide), default=None),
                'metal_neighbors_total': len(metal),
                'metal_neighbors_protein': len(protein),
                'metal_neighbors_nucleotide': len(nucleotide),
                'metal_nearest_protein': (
                    f"{nearest_protein['neighbor_resname']}{nearest_protein['neighbor_resnum']}:"
                    f"{nearest_protein['neighbor_atom']}:{nearest_protein['distance_A']}"
                    if nearest_protein else ''),
                'metal_ligand_O_global_min_A': global_metal['global_min_A'],
                'metal_ligand_O_global_atom': global_metal['global_atom'],
                **loop, 'bond_bonds_checked': bond['bonds_checked'],
                'bond_min_A': bond['min_bond_A'], 'bond_max_A': bond['max_bond_A'],
                'bond_outliers': ';'.join(bond['outliers']),
                **_confidence(path),
            })
    heavy_fields = ['state', 'mode', 'sequence', 'seed', 'model', 'ligand', 'ligand_atom',
                    'ligand_element', 'protein_resnum', 'protein_resname', 'protein_atom',
                    'protein_element', 'distance_A', 'domain', 'severe']
    anchor_fields = ['state', 'mode', 'sequence', 'seed', 'model', 'ligand',
                     'anchor_resnum', 'anchor_resname', 'anchor_atom', 'phosphate_group',
                     'ligand_atom', 'distance_A']
    metal_fields = ['state', 'mode', 'sequence', 'seed', 'model', 'metal_resnum',
                    'metal_resname', 'neighbor_type', 'neighbor_resnum', 'neighbor_resname',
                    'neighbor_atom', 'neighbor_element', 'distance_A', 'domain']
    bond_fields = ['state', 'mode', 'sequence', 'seed', 'model', 'ligand',
                   'bond_graph', 'bonds_checked', 'bond_min_A', 'bond_max_A', 'bond_outliers']
    model_fields = ['state', 'mode', 'sequence', 'seed', 'model', 'ligand',
                    'ligand_domain', 'ligand_domain_resnum', 'contacts_4A',
                    'severe_contacts', 'min_contact_A', 'contacts_NMNAT',
                    'contacts_linker', 'contacts_Nudix', 'contacts_terminal',
                    'H14_min_phosphate_A', 'H14_min_group', 'H17_min_phosphate_A',
                    'H17_min_group', 'T135_min_phosphate_A', 'T135_min_group',
                    'R138_min_phosphate_A', 'R138_min_group', 'metal_count',
                    'metal_ligand_O_min_A', 'metal_ligand_O_global_min_A',
                    'metal_ligand_O_global_atom', 'metal_neighbors_total',
                    'metal_neighbors_protein', 'metal_neighbors_nucleotide',
                    'metal_nearest_protein', 'loop_min_A',
                    'loop_contact_atom_pairs_4A', 'loop_ca_displacement_A',
                    'bond_bonds_checked', 'bond_min_A', 'bond_max_A', 'bond_outliers',
                    'ptm', 'ligand_iptm', 'complex_plddt']
    for filename, fields, rows in (
            ('heavy_atom_contacts.csv', heavy_fields, heavy_rows),
            ('anchor_phosphate_distances.csv', anchor_fields, anchor_rows),
            ('metal_neighbors.csv', metal_fields, metal_rows),
            ('ligand_bond_check.csv', bond_fields, bond_rows),
            ('detailed_metrics.csv', model_fields, model_rows)):
        with (output / filename).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore',
                                    lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    ccd_pickles = {}
    ccd_dir = Path(config['boltz']['cache']) / 'mols'
    ligands = {item['ligand'] for item in config['states'].values()}
    ligands.update(ion for item in config['states'].values() for ion in item.get('ions', []))
    for ligand in sorted(ligands):
        path = ccd_dir / f'{ligand}.pkl'
        if path.exists():
            ccd_pickles[ligand] = {'path': str(path), 'sha256': sha256_file(path)}
    identity = {
        'config_sha256': sha256_file(config_path),
        'build_sha256': sha256_file(output / 'build.json'),
        'run_identity': run_identity,
        'source_sha256': sha256_file(__file__),
        'reference_apo_sha256': sha256_file(config['reference_apo']),
        'ccd_pickle_sha256': ccd_pickles,
        'preliminary_runs': _preliminary_identity(
            load_yaml(Path(config_path).with_name('nmnat_preliminary.yaml'))
            if Path(config_path).with_name('nmnat_preliminary.yaml').exists() else config),
        'models': model_files,
        'created_utc': datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / 'analysis_identity.json', identity)
    return {'models': len(model_rows), 'heavy_contacts': len(heavy_rows),
            'anchor_rows': len(anchor_rows), 'metal_rows': len(metal_rows)}


def plot_binding_details(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    output = Path(output)
    with (output / 'detailed_metrics.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    atp = [row for row in rows if row['state'] in {'ATP_MG', 'ATP_ONLY'}]
    nmn = [row for row in rows if row['state'] == 'NMN']
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    positions = {row['model']: index for index, row in enumerate(atp)}
    labels = [f"{row['seed'][-4:]}-{row['mode'][0]}-m{row['model'].split('_model_')[-1].split('.')[0]}"
              for row in atp]
    ax = axes[0, 0]
    counts = [int(row['metal_neighbors_nucleotide'] or 0) for row in atp]
    colors = ['tab:orange' if row['mode'] == 'anchor' else 'tab:blue' for row in atp]
    ax.bar(range(len(atp)), counts, color=colors)
    ax.set_title('Mg neighbors within 3.2 A (ATP models)')
    ax.set_ylabel('nucleotide atom neighbors')
    ax.set_xticks(range(len(atp)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)

    ax = axes[0, 1]
    for mode, color in (('unconstrained', 'tab:blue'), ('anchor', 'tab:orange')):
        subset = [row for row in atp if row['mode'] == mode]
        xs = [positions[row['model']] for row in subset]
        values = [float(row['metal_ligand_O_global_min_A'])
                  if row['metal_ligand_O_global_min_A'] else None for row in subset]
        ax.scatter(xs, values, color=color, label=mode, s=30)
    ax.axhline(3.2, color='grey', linestyle='--', linewidth=0.8)
    ax.set_ylabel('Mg-ATP O global min (A)')
    ax.set_title('Mg placement (global distance)')
    ax.set_xticks(range(len(atp)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    anchors = ['H14_min_phosphate_A', 'H17_min_phosphate_A',
               'T135_min_phosphate_A', 'R138_min_phosphate_A']
    labels = ['H14', 'H17', 'T135', 'R138']
    for index, (key, label) in enumerate(zip(anchors, labels)):
        for mode, color, offset in (('unconstrained', 'tab:blue', -0.08),
                                    ('anchor', 'tab:orange', 0.08)):
            values = [float(row[key]) for row in atp
                      if row['mode'] == mode and row[key]]
            ax.scatter([index + offset] * len(values), values, color=color, s=16,
                       alpha=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel('anchor atom-phosphate O min (A)')
    ax.set_title('Anchor-phosphate proximity (ATP models)')

    ax = axes[1, 1]
    for state, color in (('ATP', 'tab:green'), ('ATP-only', 'tab:cyan'), ('NMN', 'tab:purple')):
        if state == 'ATP':
            subset = [row for row in atp if row['state'] == 'ATP_MG']
        elif state == 'ATP-only':
            subset = [row for row in atp if row['state'] == 'ATP_ONLY']
        else:
            subset = nmn
        values = [float(row['loop_ca_displacement_A']) for row in subset
                  if row['loop_ca_displacement_A']]
        if values:
            ax.scatter([state] * len(values), values, color=color, s=20)
    ax.set_ylabel('loop 102-105 CA displacement vs WT0 (A)')
    ax.set_title('Y102-D105 loop: WT0 reference only')
    fig.suptitle('WT substrate co-folding: ATP/Mg and NMN (27 models)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = output / 'binding_analysis.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_ligand_pose_consistency(config_path, output):
    config = load_yaml(config_path)
    output = Path(output)
    records = [json.loads(line) for line in open(output / 'boltz_runs.jsonl')]
    by_state = {}
    for record in records:
        if record.get('status') not in {'COMPLETED', 'SKIPPED_EXISTING'}:
            continue
        state = record['state']
        paths = [path for path in _model_paths(record['output'])
                 if path.parent.name == Path(record['path']).stem]
        for path in paths:
            by_state.setdefault(state, []).append((record, path))
    rows = []
    for state, items in sorted(by_state.items()):
        reference_record, reference_path = items[0]
        reference = _structure(reference_path)
        reference_ligand = {name: coord for name, _, coord in
                            _ligand_atoms(reference, config['states'][state]['ligand'])[1]}
        reference_ca = {int(res.id[1]): res['CA'].coord.astype(float)
                        for res in reference['A']
                        if res.id[0] == ' ' and 4 <= int(res.id[1]) <= 181}
        for record, path in items:
            structure = _structure(path)
            ca = {int(res.id[1]): res['CA'].coord.astype(float)
                  for res in structure['A']
                  if res.id[0] == ' ' and 4 <= int(res.id[1]) <= 181}
            keys = sorted(set(ca) & set(reference_ca))
            source = np.asarray([ca[key] for key in keys])
            target = np.asarray([reference_ca[key] for key in keys])
            _rmsd, rotation, translation = _kabsch(source, target)
            ligand = {name: coord for name, _, coord in
                      _ligand_atoms(structure, config['states'][state]['ligand'])[1]}
            common = sorted(set(ligand) & set(reference_ligand))
            moved = np.asarray([ligand[name] @ rotation.T + translation for name in common])
            ref = np.asarray([reference_ligand[name] for name in common])
            pose_rmsd = float(np.sqrt(np.mean(np.sum((moved - ref) ** 2, axis=1))))
            rows.append({
                'state': state, 'mode': record['mode'], 'sequence': record['sequence'],
                'seed': record['seed'], 'model': path.name,
                'reference_model': reference_path.name,
                'ligand': config['states'][state]['ligand'],
                'common_atoms': len(common),
                'pose_rmsd_A': round(pose_rmsd, 3),
            })
    fields = ['state', 'mode', 'sequence', 'seed', 'model', 'reference_model',
              'ligand', 'common_atoms', 'pose_rmsd_A']
    with (output / 'ligand_pose_consistency.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    return {'rows': len(rows)}


def _coordinate_geometry(path, ligand):
    structure = _structure(path)
    ca = {int(residue.id[1]): residue['CA'].coord.astype(float)
          for residue in structure['A']
          if residue.id[0] == ' ' and 4 <= int(residue.id[1]) <= 181
          and 'CA' in residue}
    loop = {position: ca[position] for position in range(102, 106) if position in ca}
    ligand_coords = {name: coord for name, _, coord in _ligand_atoms(structure, ligand)[1]}
    return ca, loop, ligand_coords


def _pair_metrics(geometry_a, geometry_b):
    ca_a, loop_a, ligand_a = geometry_a
    ca_b, loop_b, ligand_b = geometry_b
    keys = sorted(set(ca_a) & set(ca_b))
    _rmsd, rotation, translation = _kabsch(
        np.asarray([ca_a[key] for key in keys]),
        np.asarray([ca_b[key] for key in keys]))
    common = sorted(set(ligand_a) & set(ligand_b))
    moved = np.asarray([ligand_a[name] @ rotation.T + translation for name in common])
    reference = np.asarray([ligand_b[name] for name in common])
    atp_rmsd = float(np.sqrt(np.mean(np.sum((moved - reference) ** 2, axis=1))))
    loop_keys = sorted(set(loop_a) & set(loop_b))
    moved_loop = np.asarray([loop_a[key] @ rotation.T + translation for key in loop_keys])
    loop_reference = np.asarray([loop_b[key] for key in loop_keys])
    loop_rmsd = float(np.sqrt(np.mean(np.sum((moved_loop - loop_reference) ** 2, axis=1))))
    return atp_rmsd, loop_rmsd, len(common), len(loop_keys)


def _distribution(values, digits):
    return (round(float(np.median(values)), digits), round(min(values), digits),
            round(max(values), digits))


def _preliminary_identity(config):
    records = []
    for item in config.get('preliminary_runs', []):
        path = Path(item['path'])
        record = {'label': item['label'], 'kind': item['kind'], 'path': str(path),
                  'actual_samples': int(item['actual_samples']),
                  'note': item.get('note', '')}
        for key in ('config', 'build', 'run_identity', 'analysis_identity'):
            candidate = item.get(key)
            if candidate:
                file_path = Path(candidate)
                if file_path.exists():
                    record[f'{key}_sha256'] = sha256_file(file_path)
        inputs = []
        for candidate in item.get('inputs', []):
            file_path = Path(candidate)
            if not file_path.is_absolute():
                file_path = path / file_path
            if file_path.exists():
                inputs.append({'path': str(file_path), 'sha256': sha256_file(file_path)})
        record['input_sha256'] = inputs
        results = []
        for file_path in sorted(path.glob(item.get('result_glob', '**/*.cif'))):
            results.append({'path': str(file_path), 'sha256': sha256_file(file_path)})
        record['result_sha256'] = results
        records.append(record)
    return records


def summarize_atp_variants(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = Path(output)
    with (output / 'detailed_metrics.csv').open(newline='') as handle:
        metrics = list(csv.DictReader(handle))
    with (output / 'ligand_pose_consistency.csv').open(newline='') as handle:
        poses = {(row['seed'], row['model']): float(row['pose_rmsd_A'])
                 for row in csv.DictReader(handle)}
    numeric = ['contacts_NMNAT', 'severe_contacts', 'H14_min_phosphate_A',
               'H17_min_phosphate_A', 'T135_min_phosphate_A', 'R138_min_phosphate_A',
               'loop_ca_displacement_A']
    summary_rows = []
    grouped = {}
    for row in metrics:
        key = (int(row['seed']), row['sequence'])
        grouped.setdefault(key, []).append(row)
    for (seed, sequence), group in sorted(grouped.items()):
        entry = {'seed': seed, 'sequence': sequence, 'n_models': len(group)}
        for key in numeric:
            values = [float(row[key]) for row in group if row[key] not in ('', None)]
            entry[f'{key}_median'] = round(float(np.median(values)), 2) if values else None
            entry[f'{key}_min'] = round(min(values), 2) if values else None
            entry[f'{key}_max'] = round(max(values), 2) if values else None
        pose_values = [poses[(row['seed'], row['model'])] for row in group
                       if (row['seed'], row['model']) in poses]
        entry['pose_rmsd_median'] = round(float(np.median(pose_values)), 3) if pose_values else None
        entry['pose_rmsd_min'] = round(min(pose_values), 3) if pose_values else None
        entry['pose_rmsd_max'] = round(max(pose_values), 3) if pose_values else None
        summary_rows.append(entry)
    fields = ['seed', 'sequence', 'n_models']
    for key in numeric + ['pose_rmsd']:
        fields += [f'{key}_median', f'{key}_min', f'{key}_max']
    with (output / 'atp_variant_summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(summary_rows)
    by_seed = {}
    for row in summary_rows:
        by_seed.setdefault(int(row['seed']), {})[row['sequence']] = row
    identity = json.loads((output / 'analysis_identity.json').read_text())
    model_paths = {Path(item['path']).name: item['path'] for item in identity['models']}
    geometry_cache = {}

    def geometry(model_name):
        if model_name not in geometry_cache:
            geometry_cache[model_name] = _coordinate_geometry(model_paths[model_name], 'ATP')
        return geometry_cache[model_name]

    models_by_group = {key: sorted(row['model'] for row in group)
                       for key, group in grouped.items()}
    pair_rows = []
    for seed in sorted({key[0] for key in grouped}):
        wt_models = models_by_group[(seed, 'WT')]
        for left, right in combinations(wt_models, 2):
            atp_rmsd, loop_rmsd, n_atp, n_loop = _pair_metrics(geometry(left), geometry(right))
            pair_rows.append({'seed': seed, 'sequence': 'WT', 'comparison': 'WT_self',
                              'model_a': left, 'model_b': right,
                              'n_atp_atoms': n_atp, 'n_loop_ca': n_loop,
                              'atp_rmsd_A': round(atp_rmsd, 3),
                              'loop_ca_rmsd_A': round(loop_rmsd, 3)})
        for sequence in ('A1', 'A2', 'A3'):
            variant_models = models_by_group[(seed, sequence)]
            for variant_model in variant_models:
                for wt_model in wt_models:
                    atp_rmsd, loop_rmsd, n_atp, n_loop = _pair_metrics(
                        geometry(variant_model), geometry(wt_model))
                    pair_rows.append({'seed': seed, 'sequence': sequence,
                                      'comparison': f'{sequence}_vs_WT',
                                      'model_a': variant_model, 'model_b': wt_model,
                                      'n_atp_atoms': n_atp, 'n_loop_ca': n_loop,
                                      'atp_rmsd_A': round(atp_rmsd, 3),
                                      'loop_ca_rmsd_A': round(loop_rmsd, 3)})
            for left, right in combinations(variant_models, 2):
                atp_rmsd, loop_rmsd, n_atp, n_loop = _pair_metrics(geometry(left), geometry(right))
                pair_rows.append({'seed': seed, 'sequence': sequence,
                                  'comparison': f'{sequence}_self',
                                  'model_a': left, 'model_b': right,
                                  'n_atp_atoms': n_atp, 'n_loop_ca': n_loop,
                                  'atp_rmsd_A': round(atp_rmsd, 3),
                                  'loop_ca_rmsd_A': round(loop_rmsd, 3)})
    pair_fields = ['seed', 'sequence', 'comparison', 'model_a', 'model_b',
                   'n_atp_atoms', 'n_loop_ca', 'atp_rmsd_A', 'loop_ca_rmsd_A']
    with (output / 'atp_variant_pairs.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=pair_fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(pair_rows)

    pair_groups = {}
    for row in pair_rows:
        pair_groups.setdefault((row['seed'], row['sequence'], row['comparison']), []).append(row)
    pair_summary = []
    for (seed, sequence, comparison_name), group in sorted(pair_groups.items()):
        atp_median, atp_min, atp_max = _distribution([row['atp_rmsd_A'] for row in group], 3)
        loop_median, loop_min, loop_max = _distribution(
            [row['loop_ca_rmsd_A'] for row in group], 3)
        pair_summary.append({'seed': seed, 'sequence': sequence,
                             'comparison': comparison_name, 'n_pairs': len(group),
                             'atp_rmsd_median_A': atp_median, 'atp_rmsd_min_A': atp_min,
                             'atp_rmsd_max_A': atp_max,
                             'loop_ca_rmsd_median_A': loop_median,
                             'loop_ca_rmsd_min_A': loop_min,
                             'loop_ca_rmsd_max_A': loop_max})
    pair_summary_fields = ['seed', 'sequence', 'comparison', 'n_pairs',
                           'atp_rmsd_median_A', 'atp_rmsd_min_A', 'atp_rmsd_max_A',
                           'loop_ca_rmsd_median_A', 'loop_ca_rmsd_min_A',
                           'loop_ca_rmsd_max_A']
    with (output / 'atp_variant_pair_summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=pair_summary_fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(pair_summary)

    anchor_names = ['H14', 'H17', 'T135', 'R138']
    anchor_rows = []
    for seed in sorted({key[0] for key in grouped}):
        wt_group = grouped[(seed, 'WT')]
        for sequence in ('A1', 'A2', 'A3'):
            variant_group = grouped[(seed, sequence)]
            for anchor in anchor_names:
                key = f'{anchor}_min_phosphate_A'
                wt_values = [float(row[key]) for row in wt_group]
                variant_values = [float(row[key]) for row in variant_group]
                deltas = [variant - wt for variant in variant_values for wt in wt_values]
                wt_median, wt_min, wt_max = _distribution(wt_values, 2)
                variant_median, variant_min, variant_max = _distribution(variant_values, 2)
                delta_median, delta_min, delta_max = _distribution(deltas, 2)
                anchor_rows.append({'seed': seed, 'sequence': sequence, 'anchor': anchor,
                                    'n_cross': len(deltas),
                                    'wt_median_A': wt_median, 'wt_min_A': wt_min,
                                    'wt_max_A': wt_max,
                                    'variant_median_A': variant_median,
                                    'variant_min_A': variant_min,
                                    'variant_max_A': variant_max,
                                    'delta_median_A': delta_median,
                                    'delta_min_A': delta_min,
                                    'delta_max_A': delta_max})
    anchor_fields = ['seed', 'sequence', 'anchor', 'n_cross',
                     'wt_median_A', 'wt_min_A', 'wt_max_A',
                     'variant_median_A', 'variant_min_A', 'variant_max_A',
                     'delta_median_A', 'delta_min_A', 'delta_max_A']
    with (output / 'atp_variant_anchors.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=anchor_fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(anchor_rows)

    by_pair_summary = {(row['seed'], row['sequence'], row['comparison']): row
                       for row in pair_summary}
    comparison = []
    for seed in sorted({key[0] for key in grouped}):
        wt = by_seed[seed]['WT']
        wt_self = by_pair_summary[(seed, 'WT', 'WT_self')]
        for sequence in ('A1', 'A2', 'A3'):
            item = by_seed[seed][sequence]
            cross = by_pair_summary[(seed, sequence, f'{sequence}_vs_WT')]
            self_pair = by_pair_summary[(seed, sequence, f'{sequence}_self')]
            comparison.append({
                'seed': seed, 'sequence': sequence, 'n_cross_pairs': cross['n_pairs'],
                'atp_rmsd_median_A': cross['atp_rmsd_median_A'],
                'atp_rmsd_min_A': cross['atp_rmsd_min_A'],
                'atp_rmsd_max_A': cross['atp_rmsd_max_A'],
                'loop_ca_rmsd_median_A': cross['loop_ca_rmsd_median_A'],
                'loop_ca_rmsd_min_A': cross['loop_ca_rmsd_min_A'],
                'loop_ca_rmsd_max_A': cross['loop_ca_rmsd_max_A'],
                'wt_self_atp_median_A': wt_self['atp_rmsd_median_A'],
                'wt_self_atp_min_A': wt_self['atp_rmsd_min_A'],
                'wt_self_atp_max_A': wt_self['atp_rmsd_max_A'],
                'wt_self_loop_median_A': wt_self['loop_ca_rmsd_median_A'],
                'wt_self_loop_min_A': wt_self['loop_ca_rmsd_min_A'],
                'wt_self_loop_max_A': wt_self['loop_ca_rmsd_max_A'],
                'atp_rmsd_minus_wt_self_median_A': round(
                    cross['atp_rmsd_median_A'] - wt_self['atp_rmsd_median_A'], 3),
                'loop_ca_rmsd_minus_wt_self_median_A': round(
                    cross['loop_ca_rmsd_median_A'] - wt_self['loop_ca_rmsd_median_A'], 3),
                'self_atp_median_A': self_pair['atp_rmsd_median_A'],
                'self_loop_median_A': self_pair['loop_ca_rmsd_median_A'],
                'contacts_delta_median': round((item['contacts_NMNAT_median'] or 0) -
                                               (wt['contacts_NMNAT_median'] or 0), 2),
            })
    comparison_fields = ['seed', 'sequence', 'n_cross_pairs',
                         'atp_rmsd_median_A', 'atp_rmsd_min_A', 'atp_rmsd_max_A',
                         'loop_ca_rmsd_median_A', 'loop_ca_rmsd_min_A', 'loop_ca_rmsd_max_A',
                         'wt_self_atp_median_A', 'wt_self_atp_min_A', 'wt_self_atp_max_A',
                         'wt_self_loop_median_A', 'wt_self_loop_min_A', 'wt_self_loop_max_A',
                         'atp_rmsd_minus_wt_self_median_A',
                         'loop_ca_rmsd_minus_wt_self_median_A',
                         'self_atp_median_A', 'self_loop_median_A', 'contacts_delta_median']
    with (output / 'atp_variant_vs_wt.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(comparison)

    sequences = ['WT', 'A1', 'A2', 'A3']
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.ravel()
    panels = [('contacts_NMNAT', 'ATP contacts in NMNAT (4 A)'),
              ('H14_min_phosphate_A', 'H14-phosphate O (A)'),
              ('H17_min_phosphate_A', 'H17-phosphate O (A)'),
              ('T135_min_phosphate_A', 'T135-phosphate O (A)'),
              ('R138_min_phosphate_A', 'R138-phosphate O (A)')]
    seeds = sorted({row['seed'] for row in summary_rows})
    seed_colors = dict(zip(seeds, ('tab:blue', 'tab:purple', 'tab:olive')))
    for ax, (key, title) in zip(axes, panels):
        for seed in seeds:
            for index, sequence in enumerate(sequences):
                group = [row for row in metrics
                         if int(row['seed']) == seed and row['sequence'] == sequence]
                values = [float(row[key]) for row in group if row[key] not in ('', None)]
                jitter = (seed - 20261026) * 0.04
                ax.scatter([index + jitter] * len(values), values, s=24,
                           color=seed_colors[seed],
                           label=f'seed {seed}' if index == 0 else None)
        ax.set_xticks(range(len(sequences)))
        ax.set_xticklabels(sequences)
        ax.set_title(title, fontsize=9)
        ax.grid(axis='y', alpha=0.2)
    axes[0].legend(fontsize=7)
    ax = axes[5]
    categories = [('WT_self', 'WT'), ('A1_vs_WT', 'A1'), ('A2_vs_WT', 'A2'),
                  ('A3_vs_WT', 'A3')]
    for index, (comparison_name, label) in enumerate(categories):
        group = [row for row in pair_rows if row['comparison'] == comparison_name]
        atp = [row['atp_rmsd_A'] for row in group]
        loop = [row['loop_ca_rmsd_A'] for row in group]
        ax.scatter([index - 0.12] * len(atp), atp, s=24, marker='o', color='tab:blue')
        ax.scatter([index + 0.12] * len(loop), loop, s=24, marker='^', color='tab:orange')
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([label for _, label in categories])
    ax.set_title('paired RMSD after NMNAT 4-181 CA fit (A)', fontsize=9)
    ax.set_xlabel('model pairs pooled across seeds; not independent', fontsize=8)
    ax.scatter([], [], marker='o', color='tab:blue', label='ATP 31 atoms')
    ax.scatter([], [], marker='^', color='tab:orange', label='loop 102-105 CA')
    ax.legend(fontsize=7, title='per model pair', title_fontsize=7)
    ax.grid(axis='y', alpha=0.2)
    fig.suptitle('WT/A1/A2/A3 ATP-only co-folding (36 models)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = output / 'atp_variant_analysis.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
