"""Build and run the bounded NatV dual-domain A/B factorial design."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from .data import Target, load_yaml, sha256_file, write_json
from .model import AA

ANCHORS = {14, 17, 135, 138}


def _read_candidate(pool_root, run, candidate_id):
    path = Path(pool_root) / run / 'candidates.csv'
    matches = [row for row in csv.DictReader(open(path, newline=''))
               if row['candidate_id'] == candidate_id]
    if len(matches) != 1:
        raise ValueError(f'Candidate not unique in {run}: {candidate_id}')
    row = matches[0]
    edits = [(int(edit['position']), edit['from'], edit['to'])
             for edit in json.loads(row['edits'])]
    sequence = row['sequence']
    if len(sequence) != 341 or not set(sequence) <= set(AA):
        raise ValueError(f'Invalid candidate sequence: {candidate_id}')
    if int(row['Hamming']) != len(edits):
        raise ValueError(f'Hamming differs from edits: {candidate_id}')
    return row, sequence, edits


def _apply(sequence, edits):
    chars = list(sequence)
    for position, before, after in edits:
        if chars[position - 1] != before:
            raise ValueError(f'Edit {before}{position}{after} does not match {chars[position - 1]}')
        chars[position - 1] = after
    return ''.join(chars)


def _validate_a3m(path, name, sequence, rows):
    records = _read_a3m(path)
    if len(records) != rows:
        raise ValueError(f'A3M row count differs: {path}')
    if records[0][0] != name or records[0][1] != sequence:
        raise ValueError(f'A3M query differs: {path}')
    if any(_aligned_length(row[1]) != 341 for row in records):
        raise ValueError(f'A3M aligned length differs: {path}')


def _edits_text(edits):
    return ';'.join(f'{before}{position}{after}' for position, before, after in edits)


def _read_a3m(path):
    records = []
    header = None
    chunks = []
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\r\n')
        if line.startswith('>'):
            if header is not None:
                records.append((header, ''.join(chunks)))
            header = line[1:]
            chunks = []
        elif line:
            chunks.append(line.strip())
    if header is not None:
        records.append((header, ''.join(chunks)))
    return records


def _aligned_length(sequence):
    return len(''.join(residue for residue in sequence if not residue.islower()))


def _write_msa(source, destination, name, sequence, rows, wt):
    records = _read_a3m(source)
    if len(records) < rows or records[0][1].upper() != wt:
        raise ValueError('MSA source does not start with WT or has too few rows')
    homologs = records[1:rows]
    if len(sequence) != 341 or any(_aligned_length(row[1]) != 341 for row in homologs):
        raise ValueError('A3M aligned lengths differ from 341')
    with destination.open('w') as handle:
        handle.write(f'>{name}\n{sequence}\n')
        for header, homolog in homologs:
            handle.write(f'>{header}\n{homolog}\n')


def build_design(config_path, output):
    config = load_yaml(config_path)
    target = Target.load(config['target'])
    wt = target.sequence
    if len(wt) != 341:
        raise ValueError('Target is not 341 aa')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Design candidate output already exists: {output}')
    output.mkdir(parents=True, exist_ok=True)
    msa_dir = output / 'msa'
    yaml_dir = output / 'boltz_inputs'
    msa_dir.mkdir()
    yaml_dir.mkdir()
    msa_source = Path(config['msa_source']).resolve()
    candidates = [{
        'name': 'WT', 'group': 'WT', 'parent_a': '', 'parent_b': '',
        'source_run': '', 'source_candidate': '', 'sequence': wt,
        'edits': [], 'anchor_changed': False, 'nudix_changed': False,
    }]
    a_sequences = {}
    for item in config['a']:
        row, sequence, edits = _read_candidate(config['pool_root'], item['run'],
                                               item['candidate_id'])
        if any(position in ANCHORS for position, _, _ in edits):
            raise ValueError(f'Selected A changes an NMNAT anchor: {item["id"]}')
        if any(position > 181 for position, _, _ in edits):
            raise ValueError(f'Selected A changes Nudix: {item["id"]}')
        a_sequences[item['id']] = sequence
        candidates.append({
            'name': item['id'], 'group': 'A', 'parent_a': item['id'], 'parent_b': '',
            'source_run': item['run'], 'source_candidate': item['candidate_id'],
            'reason': item['reason'], 'sequence': sequence, 'edits': edits,
            'anchor_changed': False, 'nudix_changed': False,
        })
    b_sequences = {}
    for item in config['b']:
        position, before, after = int(item['position']), item['wt'], item['alt']
        if wt[position - 1] != before:
            raise ValueError(f'B WT residue differs at {position}')
        b_sequences[item['id']] = (position, before, after)
        candidates.append({
            'name': item['id'], 'group': 'B', 'parent_a': '', 'parent_b': item['id'],
            'source_run': '', 'source_candidate': '', 'reason': item['mechanism'],
            'sequence': _apply(wt, [(position, before, after)]),
            'edits': [(position, before, after)],
            'anchor_changed': False, 'nudix_changed': True,
        })
    for a_item in config['a']:
        a_name = a_item['id']
        a_sequence = a_sequences[a_name]
        for b_item in config['b']:
            b_name = b_item['id']
            position, before, after = b_sequences[b_name]
            if a_sequence[position - 1] != before:
                raise ValueError(f'A parent already differs at B site: {a_name}/{b_name}')
            candidates.append({
                'name': f'{a_name}{b_name}', 'group': 'A+B',
                'parent_a': a_name, 'parent_b': b_name,
                'source_run': a_item['run'], 'source_candidate': a_item['candidate_id'],
                'reason': f'{a_item["reason"]} + {b_item["mechanism"]}',
                'sequence': _apply(a_sequence, [(position, before, after)]),
                'edits': _read_candidate(config['pool_root'], a_item['run'],
                                         a_item['candidate_id'])[2] +
                         [(position, before, after)],
                'anchor_changed': False, 'nudix_changed': True,
            })
    seen = set()
    for candidate in candidates:
        sequence = candidate['sequence']
        if len(sequence) != 341 or not set(sequence) <= set(AA):
            raise ValueError(f'Invalid design sequence: {candidate["name"]}')
        digest = hashlib.sha256(sequence.encode()).hexdigest()
        if digest in seen:
            raise ValueError(f'Duplicate design sequence: {candidate["name"]}')
        seen.add(digest)
        candidate['sequence_sha256'] = digest
        candidate['Hamming'] = len(candidate['edits'])
        candidate['edits'] = _edits_text(candidate['edits'])
        msa_path = msa_dir / f'{candidate["name"]}.a3m'
        _write_msa(msa_source, msa_path, candidate['name'], sequence,
                   int(config['msa_rows']), wt)
        _validate_a3m(msa_path, candidate['name'], sequence, int(config['msa_rows']))
        yaml_path = yaml_dir / f'{candidate["name"]}.yaml'
        yaml_path.write_text(
            'version: 1\nsequences:\n'
            '  - protein:\n'
            '      id: A\n'
            f'      sequence: {sequence}\n'
            f'      msa: {(msa_dir / (candidate["name"] + ".a3m")).resolve()}\n')
    with (output / 'candidates.fasta').open('w') as handle:
        for candidate in candidates:
            handle.write(f'>{candidate["name"]}\n{candidate["sequence"]}\n')
    fields = ['name', 'group', 'parent_a', 'parent_b', 'source_run', 'source_candidate',
              'reason', 'Hamming', 'edits', 'anchor_changed', 'nudix_changed',
              'sequence_sha256', 'sequence']
    with (output / 'candidates.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore',
                                lineterminator='\n')
        writer.writeheader()
        writer.writerows(candidates)
    manifest = {
        'task': config['task'], 'config': str(config_path),
        'config_sha256': sha256_file(config_path), 'target': config['target'],
        'target_sha256': sha256_file(config['target']),
        'msa_source': str(msa_source), 'msa_source_sha256': sha256_file(msa_source),
        'msa_rows': int(config['msa_rows']), 'candidates': len(candidates),
        'candidate_names': [candidate['name'] for candidate in candidates],
        'boltz': config['boltz'], 'implementation_sha256': sha256_file(__file__),
    }
    write_json(output / 'design.json', manifest)
    return {'status': 'COMPLETED', 'candidates': len(candidates), 'output': str(output)}


def run_boltz(config_path, output):
    config = load_yaml(config_path)
    candidate_dir = Path(config['candidates_output'])
    design = json.load(open(candidate_dir / 'design.json'))
    yaml_dir = candidate_dir / 'boltz_inputs'
    names = design['candidate_names']
    if names[0] != 'WT':
        raise ValueError('WT must be the first formal prediction')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Structure output already exists: {output}')
    output.mkdir(parents=True, exist_ok=True)
    logs = output / 'logs'
    logs.mkdir()
    executable = Path(config['boltz']['executable'])
    if not executable.exists():
        raise FileNotFoundError(executable)
    env = os.environ.copy()
    env['BOLTZ_CACHE'] = str(config['boltz']['cache'])
    summary = {'status': 'RUNNING', 'runs': [], 'boltz': config['boltz'],
               'candidate_manifest_sha256': sha256_file(candidate_dir / 'design.json')}
    failures = 0
    for name in names:
        yaml_path = yaml_dir / f'{name}.yaml'
        destination = output / name
        if not yaml_path.exists():
            raise FileNotFoundError(yaml_path)
        command = [
            str(executable), 'predict', str(yaml_path),
            '--out_dir', str(destination),
            '--accelerator', 'gpu', '--devices', '1',
            '--diffusion_samples', str(config['boltz']['diffusion_samples']),
            '--sampling_steps', str(config['boltz']['sampling_steps']),
            '--recycling_steps', str(config['boltz']['recycling_steps']),
            '--seed', str(config['boltz']['seed']),
            '--output_format', str(config['boltz']['output_format']),
            '--num_workers', str(config['boltz']['num_workers']),
        ]
        started = time.monotonic()
        try:
            result = subprocess.run(command, env=env, capture_output=True, text=True,
                                    timeout=int(config['boltz']['timeout_seconds']))
            seconds = time.monotonic() - started
            (logs / f'{name}.stdout').write_text(result.stdout)
            (logs / f'{name}.stderr').write_text(result.stderr)
            item = {'name': name, 'command': command, 'seconds': seconds,
                    'returncode': result.returncode,
                    'output_exists': destination.exists(), 'error': None}
            if result.returncode != 0:
                item['error'] = result.stderr[-4000:]
        except Exception as error:
            seconds = time.monotonic() - started
            item = {'name': name, 'command': command, 'seconds': seconds,
                    'returncode': None, 'output_exists': destination.exists(),
                    'error': f'{type(error).__name__}: {error}'}
        summary['runs'].append(item)
        (output / 'boltz_runs.jsonl').open('a').write(
            json.dumps(item, ensure_ascii=False) + '\n')
        write_json(output / 'boltz_run_summary.json', summary)
        if item['error'] or not item['output_exists']:
            failures += 1
            if name == 'WT':
                summary['status'] = 'FAILED_WT'
                write_json(output / 'boltz_run_summary.json', summary)
                return summary
    summary['status'] = 'COMPLETED' if failures == 0 else 'COMPLETED_WITH_FAILURES'
    summary['failures'] = failures
    write_json(output / 'boltz_run_summary.json', summary)
    return summary


def _model_paths(root, name):
    paths = sorted((Path(root) / name / f'boltz_results_{name}' / 'predictions' / name).glob('*.cif'))
    if len(paths) != 3:
        raise FileNotFoundError(f'Expected 3 Boltz models for {name}, found {len(paths)}')
    return paths


def _ca_map(path):
    from Bio.PDB import MMCIFParser
    structure = MMCIFParser(QUIET=True).get_structure(str(path), str(path))[0]
    chain = structure['A']
    output = {}
    for residue in chain:
        if residue.id[0] == ' ' and 'CA' in residue:
            output[int(residue.id[1])] = residue['CA'].coord.astype(float)
    if len(output) != 341:
        raise ValueError(f'Unexpected model length: {path}')
    return output


def _kabsch(source, target):
    import numpy as np
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    source_center = source.mean(0)
    target_center = target.mean(0)
    source = source - source_center
    target = target - target_center
    covariance = target.T @ source
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    aligned = target @ rotation.T
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - source) ** 2, axis=1))))
    return rmsd, rotation, source_center, target_center


def _region_indices(region):
    if region == 'full':
        return list(range(1, 342))
    if region == 'NMNAT':
        return list(range(4, 182))
    if region == 'Nudix':
        return list(range(204, 339))
    raise ValueError(region)


def _pairwise_rmsd(design, root):
    names = design['candidate_names']
    models = {name: [_ca_map(path) for path in _model_paths(root, name)] for name in names}
    pairs = []
    # candidate manifest table does not carry groups; derive from design names
    for name in names:
        if name == 'WT':
            continue
        pairs.append((name, 'WT'))
    for a in ('A1', 'A2', 'A3'):
        for b in ('B1', 'B2'):
            pairs.append((f'{a}{b}', a))
            pairs.append((f'{a}{b}', b))
    rows = []
    for left, right in pairs:
        for region in ('full', 'NMNAT', 'Nudix'):
            indices = _region_indices(region)
            values = []
            for lm in models[left]:
                for rm in models[right]:
                    values.append(_kabsch([lm[i] for i in indices],
                                          [rm[i] for i in indices])[0])
            rows.append({'left': left, 'right': right, 'region': region,
                         'median_rmsd_A': float(__import__('numpy').median(values)),
                         'min_rmsd_A': float(min(values)), 'max_rmsd_A': float(max(values)),
                         'model_pairs': len(values)})
    return rows


def _self_rmsd(design, root):
    models = {name: [_ca_map(path) for path in _model_paths(root, name)]
              for name in design['candidate_names']}
    rows = []
    for name, values in models.items():
        for region in ('full', 'NMNAT', 'Nudix'):
            indices = _region_indices(region)
            pair_values = []
            for left in range(len(values)):
                for right in range(left + 1, len(values)):
                    pair_values.append(_kabsch([values[left][i] for i in indices],
                                               [values[right][i] for i in indices])[0])
            import numpy as np
            rows.append({'name': name, 'region': region,
                         'median_rmsd_A': float(np.median(pair_values)),
                         'min_rmsd_A': float(min(pair_values)),
                         'max_rmsd_A': float(max(pair_values)),
                         'model_pairs': len(pair_values)})
    return rows


def _source_alignment(path, chain_id, wt):
    from Bio.PDB import MMCIFParser
    from Bio.Align import PairwiseAligner, substitution_matrices
    from Bio.Data.PDBData import protein_letters_3to1
    structure = MMCIFParser(QUIET=True).get_structure(str(path), str(path))[0]
    chain = structure[chain_id]
    residues = [residue for residue in chain if residue.id[0] == ' ']
    sequence = ''.join(protein_letters_3to1.get(residue.resname, 'X')
                       for residue in residues)
    aligner = PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load('BLOSUM62')
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    aligner.end_gap_score = 0
    alignment = aligner.align(wt, sequence)[0]
    mapping = {}
    for (w0, w1), (s0, _) in zip(alignment.aligned[0], alignment.aligned[1]):
        for offset in range(w1 - w0):
            mapping[w0 + offset] = residues[s0 + offset]
    return structure, chain, mapping


def _assembly_atoms(structure, metadata, ref_chain):
    import numpy as np
    ids = metadata.get('_pdbx_struct_oper_list.id', ['1'])
    ops = {}
    for index, op_id in enumerate(ids):
        matrix = np.array([[float(metadata[f'_pdbx_struct_oper_list.matrix[{i}][{j}]'][index])
                            for j in (1, 2, 3)] for i in (1, 2, 3)])
        vector = np.array([float(metadata[f'_pdbx_struct_oper_list.vector[{i}]'][index])
                           for i in (1, 2, 3)])
        ops[str(op_id)] = (matrix, vector)
    expression = metadata.get('_pdbx_struct_assembly_gen.oper_expression', ['1'])[0]
    gen_ops = [token.strip() for token in expression.replace('(', '').replace(')', '').split(',')]
    asym = set(metadata.get('_pdbx_struct_assembly_gen.asym_id_list', [''])[0].split(','))
    copies = []
    reference_id = None
    for chain in structure:
        if chain.id not in asym:
            continue
        for op_id in gen_ops:
            if op_id not in ops:
                continue
            matrix, vector = ops[op_id]
            coords = [atom.coord.astype(float) @ matrix.T + vector
                      for residue in chain if residue.id[0] == ' '
                      for atom in residue if atom.element != 'H']
            if not coords:
                continue
            copy_id = f'{chain.id}:{op_id}'
            copies.append({'id': copy_id, 'chain': chain.id, 'op': op_id,
                           'coords': np.asarray(coords)})
            if chain.id == ref_chain and op_id == '1':
                reference_id = copy_id
    return copies, reference_id


def _source_site_map(config, wt):
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    import math
    result = {}
    for name, item in config['source_structures'].items():
        path = Path(item['path'])
        structure, chain, mapping = _source_alignment(path, item['ref_chain'], wt)
        metadata = MMCIF2Dict(str(path))
        assembly = {
            'id': metadata.get('_pdbx_struct_assembly.id'),
            'details': metadata.get('_pdbx_struct_assembly.oligomeric_details'),
            'count': metadata.get('_pdbx_struct_assembly.oligomeric_count'),
            'oper_expression': metadata.get('_pdbx_struct_assembly_gen.oper_expression'),
        }
        fields = ['_atom_site.group_PDB', '_atom_site.label_entity_id',
                  '_atom_site.label_asym_id', '_atom_site.auth_asym_id',
                  '_atom_site.label_seq_id', '_atom_site.auth_seq_id',
                  '_atom_site.label_comp_id']
        atom_rows = list(zip(*[metadata.get(field, []) for field in fields]))
        sites = {}
        for position in (248, 252, 308):
            residue = mapping.get(position - 1)
            if residue is None:
                sites[str(position)] = {'status': 'unmapped'}
                continue
            label = {'GLU': 'GLU', 'ASP': 'ASP'}.get(residue.resname, residue.resname)
            atom_row = next((row for row in atom_rows
                             if row[0] == 'ATOM' and row[3] == item['ref_chain']
                             and row[5] == str(residue.id[1]) and row[6] == label), None)
            ligand_contacts = []
            for ligand in chain:
                if ligand.id[0] == ' ' or ligand.resname not in item['ligands']:
                    continue
                best = min(math.dist(a1.coord, a2.coord) for a1 in residue
                           if a1.element != 'H' for a2 in ligand if a2.element != 'H')
                if best <= 4.5:
                    ligand_contacts.append({'ligand': ligand.resname, 'number': ligand.id[1],
                                            'min_distance_A': round(best, 2)})
            copies, reference_id = _assembly_atoms(structure, metadata, item['ref_chain'])
            other = []
            for copy in copies:
                if copy['id'] == reference_id:
                    continue
                best = min(math.dist(a1.coord, a2) for a1 in residue
                           if a1.element != 'H' for a2 in copy['coords'])
                other.append({'assembly_copy': copy['id'],
                              'min_distance_A': round(best, 2)})
            sites[str(position)] = {
                'status': 'mapped', 'wt_residue': wt[position - 1],
                'source_residue': residue.resname, 'source_auth_seq_id': residue.id[1],
                'atom_site': {'entity': atom_row[1], 'label_asym': atom_row[2],
                              'auth_asym': atom_row[3], 'label_seq': atom_row[4],
                              'auth_seq': atom_row[5], 'component': atom_row[6]}
                if atom_row else None,
                'ligand_contacts_4.5A': ligand_contacts,
                'other_chain_min_A': other,
            }
        result[name] = {'path': str(path), 'ref_chain': item['ref_chain'],
                        'assembly': assembly, 'sites': sites}
    return result


def _model_atoms(path, positions):
    from Bio.PDB import MMCIFParser
    structure = MMCIFParser(QUIET=True).get_structure(str(path), str(path))[0]
    chain = structure['A']
    output = {}
    for residue in chain:
        if residue.id[0] == ' ' and int(residue.id[1]) in positions:
            output[int(residue.id[1])] = [atom.coord.astype(float) for atom in residue
                                          if atom.element != 'H']
    return output


def _interface_overlay(design, config, root, wt, source_name):
    import numpy as np
    import math
    names = design['candidate_names']
    item = config['source_structures'][source_name]
    structure, chain, mapping = _source_alignment(item['path'], item['ref_chain'], wt)
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    metadata = MMCIF2Dict(str(item['path']))
    source_ca = {index + 1: residue['CA'].coord.astype(float)
                 for index, residue in mapping.items() if 'CA' in residue}
    copies, reference_id = _assembly_atoms(structure, metadata, item['ref_chain'])
    other_atoms = [coord for copy in copies if copy['id'] != reference_id
                   for coord in copy['coords']]
    ligand_atoms = [atom.coord.astype(float) for residue in chain
                    if residue.id[0] != ' ' and residue.resname in item['ligands']
                    for atom in residue if atom.element != 'H']
    output = {}
    for name in names:
        output[name] = []
        for path in _model_paths(root, name):
            candidate_ca = _ca_map(path)
            fits = {}
            transform = None
            for region in ('full', 'NMNAT', 'Nudix'):
                indices = [position for position in _region_indices(region)
                           if position in source_ca and position in candidate_ca]
                source_points = np.array([source_ca[position] for position in indices])
                candidate_points = np.array([candidate_ca[position] for position in indices])
                rmsd, rotation, source_center, candidate_center = _kabsch(
                    source_points, candidate_points)
                fits[region] = round(rmsd, 2)
                if region == 'Nudix':
                    transform = (rotation, source_center, candidate_center)
            atoms = _model_atoms(path, (248, 308))
            distances = {}
            for position in (248, 308):
                if position not in atoms or transform is None:
                    distances[str(position)] = None
                    continue
                rotation, source_center, candidate_center = transform
                points = (np.array(atoms[position]) - candidate_center) @ rotation.T + source_center
                distances[str(position)] = {
                    'other_chain_min_A': round(float(np.min(np.linalg.norm(
                        points[:, None, :] - np.array(other_atoms)[None, :, :], axis=2))), 2)
                        if other_atoms else None,
                    'ligand_min_A': round(float(np.min(np.linalg.norm(
                        points[:, None, :] - np.array(ligand_atoms)[None, :, :], axis=2))), 2)
                        if ligand_atoms else None,
                }
            output[name].append({'fit_rmsd_A': fits, 'distances': distances})
    return output


def _confidence(paths):
    directories = {path.parent for path in paths}
    if len(directories) != 1:
        raise ValueError('Models for one candidate are not in one directory')
    values = [json.load(open(item)) for item in
              sorted(next(iter(directories)).glob('confidence_*.json'))]
    if len(values) != 3:
        raise ValueError('Expected 3 confidence files per candidate')
    import numpy as np
    return {
        'complex_plddt': {'median': float(np.median([v['complex_plddt'] for v in values])),
                          'min': float(min(v['complex_plddt'] for v in values)),
                          'max': float(max(v['complex_plddt'] for v in values))},
        'ptm': {'median': float(np.median([v['ptm'] for v in values])),
                'min': float(min(v['ptm'] for v in values)),
                'max': float(max(v['ptm'] for v in values))},
        'confidence_score': {'median': float(np.median([v['confidence_score'] for v in values])),
                             'min': float(min(v['confidence_score'] for v in values)),
                             'max': float(max(v['confidence_score'] for v in values))},
    }


def _cif_sequence(path):
    from Bio.PDB import MMCIFParser
    from Bio.Data.PDBData import protein_letters_3to1
    structure = MMCIFParser(QUIET=True).get_structure(str(path), str(path))[0]
    sequence = ''.join(protein_letters_3to1.get(residue.resname, 'X')
                       for residue in structure['A'] if residue.id[0] == ' ')
    if len(sequence) != 341 or 'X' in sequence:
        raise ValueError(f'Invalid Boltz structure sequence: {path}')
    return sequence


def _anchor_local_metrics(design, root):
    import numpy as np
    models = {name: [_ca_map(path) for path in _model_paths(root, name)]
              for name in design['candidate_names']}
    anchors = (14, 17, 135, 138)
    neighborhoods = {'10-24': list(range(10, 25)), '130-141': list(range(130, 142))}
    comparisons = [('B1', 'WT'), ('B2', 'WT')]
    for a in ('A1', 'A2', 'A3'):
        for b in ('B1', 'B2'):
            comparisons.append((f'{a}{b}', a))
    comparisons += [(name, name) for name in ('WT', 'A1', 'A2', 'A3', 'B1', 'B2',
                                              'A1B1', 'A1B2', 'A2B1', 'A2B2',
                                              'A3B1', 'A3B2')]
    output = {}
    for left, right in comparisons:
        values = {'anchor_displacements_A': {str(p): [] for p in anchors},
                  'neighborhood_rmsd_A': {key: [] for key in neighborhoods}}
        model_pairs = [(lm, rm) for lm in models[left] for rm in models[right]
                       if not (left == right and lm is rm)]
        for lm, rm in model_pairs:
            fit_indices = list(range(4, 182))
            source = np.array([lm[i] for i in fit_indices])
            target = np.array([rm[i] for i in fit_indices])
            _, rotation, source_center, target_center = _kabsch(source, target)
            aligned = {i: (rm[i] - target_center) @ rotation.T + source_center
                       for i in rm}
            for position in anchors:
                values['anchor_displacements_A'][str(position)].append(
                    float(np.linalg.norm(aligned[position] - lm[position])))
            for key, positions in neighborhoods.items():
                diffs = np.array([aligned[i] - lm[i] for i in positions])
                values['neighborhood_rmsd_A'][key].append(
                    float(np.sqrt(np.mean(np.sum(diffs * diffs, axis=1)))))
        summary = {'left': left, 'right': right, 'model_pairs': len(model_pairs)}
        for position, vals in values['anchor_displacements_A'].items():
            summary[f'anchor_{position}_median_A'] = float(np.median(vals))
            summary[f'anchor_{position}_range_A'] = [float(min(vals)), float(max(vals))]
        for key, vals in values['neighborhood_rmsd_A'].items():
            summary[f'neighborhood_{key}_median_A'] = float(np.median(vals))
            summary[f'neighborhood_{key}_range_A'] = [float(min(vals)), float(max(vals))]
        output[f'{left}|{right}'] = summary
    return output


def summarize_design(config_path, output):
    import numpy as np
    config = load_yaml(config_path)
    candidate_dir = Path(config['candidates_output'])
    design = json.load(open(candidate_dir / 'design.json'))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = Target.load(config['target'])
    run_summary = json.load(open(output / 'boltz_run_summary.json'))
    rows = _pairwise_rmsd(design, output)
    self_rows = _self_rmsd(design, output)
    with (output / 'structure_pairs.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    site_map = _source_site_map(config, target.sequence)
    overlay = {source: _interface_overlay(design, config, output, target.sequence, source)
               for source in config['source_structures']}
    confidence = {name: _confidence(_model_paths(output, name)) for name in design['candidate_names']}
    anchor_metrics = _anchor_local_metrics(design, output)
    candidates = {row['name']: row for row in csv.DictReader(open(candidate_dir / 'candidates.csv', newline=''))}
    sequence_checks = {}
    for name in design['candidate_names']:
        expected = candidates[name]['sequence']
        sequence_checks[name] = []
        for path in _model_paths(output, name):
            sequence = _cif_sequence(path)
            sequence_checks[name].append({
                'model': path.name, 'sequence_sha256': hashlib.sha256(sequence.encode()).hexdigest(),
                'expected_sequence_sha256': candidates[name]['sequence_sha256'],
                'sequence_matches': sequence == expected,
                'structure_sha256': sha256_file(path),
            })
        if not all(item['sequence_matches'] for item in sequence_checks[name]):
            raise ValueError(f'Boltz structure sequence differs: {name}')
    boltz_python = Path(config['boltz']['executable']).parent / 'python'
    versions = subprocess.check_output(
        [str(boltz_python), '-c',
         "import importlib.metadata as m, torch; print(m.version('boltz')); print(torch.__version__)"],
        text=True).splitlines()
    checkpoint = Path(config['boltz']['checkpoint'])
    tool_identity = {
        'boltz_version': versions[0], 'torch_version': versions[1],
        'executable': str(config['boltz']['executable']),
        'cache': str(config['boltz']['cache']),
        'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256_file(checkpoint),
        'summary_implementation_sha256': sha256_file(__file__),
    }
    summary = {
        'status': 'COMPLETED',
        'candidate_manifest_sha256': sha256_file(candidate_dir / 'design.json'),
        'boltz_run_summary': run_summary,
        'structure_pairs': rows,
        'self_rmsd': self_rows,
        'experimental_site_map': site_map,
        'interface_overlays': overlay,
        'confidence': confidence,
        'structure_sequence_checks': sequence_checks,
        'anchor_local_metrics': anchor_metrics,
        'tool_identity': tool_identity,
        'notes': {
            'structure_samples': 'three diffusion samples per sequence; descriptive only',
            'overlay': 'candidate monomer fitted to each homolog reference chain by Nudix-domain CA; template constraint, not a predicted complex',
            'ligands': 'APR/NAD/AMP/MN/MG are experimental homolog ligands; not KVP40 affinity or activity measurements',
            'assembly': '2QJO hexamer; 2QJT/2R5W dimer; KVP40 assembly unknown',
        },
    }
    write_json(output / 'structure_summary.json', summary)
    return {'status': 'COMPLETED', 'pairs': len(rows), 'output': str(output)}
