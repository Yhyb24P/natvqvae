"""Map state-specific NMNAT contacts onto matched full-length KVP40 WT models."""
from __future__ import annotations

import csv
import gzip
import hashlib
import os
import subprocess
import sys
import tempfile
from glob import glob
from pathlib import Path

import numpy as np

from .data import Target, load_yaml, sha256_file, write_json


def _open_structure(path):
    from Bio.PDB import MMCIFParser
    parser = MMCIFParser(QUIET=True)
    structure_id = Path(path).name.split('.')[0]
    if not str(path).endswith('.gz'):
        return parser.get_structure(structure_id, str(path))
    text = gzip.open(path, 'rt').read()
    temporary = tempfile.NamedTemporaryFile('w', suffix='.cif', delete=False)
    try:
        temporary.write(text)
        temporary.close()
        return parser.get_structure(structure_id, temporary.name)
    finally:
        if not temporary.closed:
            temporary.close()
        os.unlink(temporary.name)


def _protein_residues(chain, start, end):
    return [residue for residue in chain
            if residue.id[0] == ' ' and start <= int(residue.id[1]) <= end]


def _ligand_residues(chain, names):
    return [residue for residue in chain
            if residue.id[0] != ' ' and residue.resname.strip() in names]


def _heavy_atoms(residue):
    return [atom for atom in residue if atom.element.strip() != 'H']


def _sequence(residues):
    from Bio.Data.PDBData import protein_letters_3to1
    return ''.join(protein_letters_3to1.get(residue.resname, 'X') for residue in residues)


def _write_protein_pdb(structure, chain_id, start, end, path):
    from Bio.PDB import PDBIO, Select

    class SelectProtein(Select):
        def accept_chain(self, chain):
            return chain.id == chain_id

        def accept_residue(self, residue):
            return residue.id[0] == ' ' and start <= int(residue.id[1]) <= end

        def accept_atom(self, atom):
            return atom.element.strip() != 'H'

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(path), SelectProtein())


def _run_tmalign(executable, source_pdb, target_pdb):
    result = subprocess.run([str(executable), str(source_pdb), str(target_pdb)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'TMalign failed: {result.stderr.strip()}')
    return result.stdout


def _parse_tmalign(output):
    lines = output.splitlines()
    start = next((index for index, line in enumerate(lines)
                  if line.startswith('(":" denotes')), None)
    if start is None:
        raise ValueError('TMalign output lacks an alignment block')
    block = []
    for line in lines[start + 1:]:
        if line.startswith('#Total'):
            break
        if line.strip():
            block.append(line.rstrip())
    if len(block) % 3:
        raise ValueError('Unexpected TMalign alignment block')
    source_seq = ''.join(block[index] for index in range(0, len(block), 3))
    match = ''.join(block[index + 1] for index in range(0, len(block), 3))
    target_seq = ''.join(block[index + 2] for index in range(0, len(block), 3))
    scores = []
    for line in lines:
        if line.startswith('TM-score='):
            scores.append(float(line.split('=')[1].split()[0]))
    aligned_length = None
    rmsd = None
    for line in lines:
        if line.startswith('Aligned length='):
            aligned_length = int(line.split('=')[1].split(',')[0])
            rmsd = float(line.split('RMSD=')[1].split(',')[0])
    return {
        'source_seq': source_seq,
        'target_seq': target_seq,
        'match': match,
        'tm_scores': scores,
        'aligned_length': aligned_length,
        'rmsd_A': rmsd,
    }


def _kabsch(source, target):
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source0 = source - source_center
    target0 = target - target_center
    u, _, vt = np.linalg.svd(source0.T @ target0)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - source_center @ rotation.T
    aligned = source @ rotation.T + translation
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    return rmsd, rotation, translation


def _apply_transform(points, rotation, translation):
    points = np.asarray(points, dtype=float)
    return points @ rotation.T + translation


def _contact_type(ligand_resname, protein_resname, ligand_atom, protein_atom, distance):
    metals = {'MG', 'MN', 'ZN', 'CA', 'NA', 'K'}
    if ligand_resname in metals or protein_resname in metals:
        return 'metal'
    elements = {ligand_atom.element.strip(), protein_atom.element.strip()}
    if elements <= {'N', 'O', 'S'} and distance <= 3.6:
        return 'polar'
    if elements <= {'C'}:
        return 'vdw'
    return 'other'


def _atom_row(atom):
    return {
        'occupancy': atom.occupancy,
        'altloc': atom.altloc,
        'b_factor': atom.bfactor,
    }


def _source_contacts(state, source):
    structure = _open_structure(source['path'])
    model = structure[0]
    protein_chain = model[source['protein_chain']]
    start, end = source['protein_range']
    protein = _protein_residues(protein_chain, start, end)
    cutoff = float(source.get('contact_cutoff_A', 4.0))
    metal_cutoff = float(source.get('metal_cutoff_A', cutoff))
    rows = []
    contact_id = 0
    for ligand_chain_id in source['ligand_chains']:
        chain = model[ligand_chain_id]
        for ligand in _ligand_residues(chain, set(source['ligands'])):
            ligand_cutoff = metal_cutoff if ligand.resname.strip() in {'MG', 'MN', 'ZN'} else cutoff
            for ligand_atom in _heavy_atoms(ligand):
                for residue in protein:
                    best = None
                    for protein_atom in _heavy_atoms(residue):
                        distance = float(np.linalg.norm(ligand_atom.coord - protein_atom.coord))
                        if best is None or distance < best[0]:
                            best = (distance, protein_atom)
                    if best is not None and best[0] <= ligand_cutoff:
                        distance, protein_atom = best
                        contact_id += 1
                        row = {
                            'contact_id': contact_id,
                            'state': state,
                            'pdb_id': source['pdb_id'],
                            'protein_chain': source['protein_chain'],
                            'protein_resnum': int(residue.id[1]),
                            'protein_resname': residue.resname.strip(),
                            'protein_atom': protein_atom.name,
                            'ligand_chain': ligand_chain_id,
                            'ligand_resname': ligand.resname.strip(),
                            'ligand_resnum': int(ligand.id[1]),
                            'ligand_atom': ligand_atom.name,
                            'distance_A': round(distance, 2),
                            'contact_type': _contact_type(
                                ligand.resname.strip(), residue.resname.strip(),
                                ligand_atom, protein_atom, distance),
                        }
                        row.update({f'ligand_{key}': value for key, value in
                                    _atom_row(ligand_atom).items()})
                        rows.append(row)
    return structure, model, protein, rows


def _alignment_pairs(source_residues, target_residues, parsed):
    source_seq = parsed['source_seq']
    target_seq = parsed['target_seq']
    match = parsed['match']
    if len(source_seq.replace('-', '')) != len(source_residues):
        raise ValueError('TMalign source alignment length differs from the PDB chain')
    if len(target_seq.replace('-', '')) != len(target_residues):
        raise ValueError('TMalign target alignment length differs from the PDB chain')
    pairs = []
    i = j = 0
    for source_char, match_char, target_char in zip(source_seq, match, target_seq):
        if source_char != '-' and target_char != '-':
            pairs.append((source_residues[i], target_residues[j], match_char))
        if source_char != '-':
            i += 1
        if target_char != '-':
            j += 1
    return pairs


def _mapping_from_pairs(pairs, rotation, translation):
    mapping = {}
    for source_residue, target_residue, match_char in pairs:
        source_num = int(source_residue.id[1])
        target_num = int(target_residue.id[1])
        if source_num in mapping:
            raise ValueError(f'Duplicate source residue number in alignment: {source_num}')
        ca_distance = None
        if 'CA' in source_residue and 'CA' in target_residue:
            source_ca = _apply_transform([source_residue['CA'].coord], rotation, translation)[0]
            ca_distance = float(np.linalg.norm(source_ca - target_residue['CA'].coord))
        confidence = 'low'
        if match_char == ':' and ca_distance is not None and ca_distance <= 2.0:
            confidence = 'high'
        elif match_char == ':' and ca_distance is not None and ca_distance <= 5.0:
            confidence = 'medium'
        mapping[source_num] = {
            'target_resnum': target_num,
            'target_resname': target_residue.resname.strip(),
            'match': match_char,
            'ca_distance_A': ca_distance,
            'confidence': confidence,
        }
    return mapping


def _fit_from_pairs(pairs, min_fit_pairs):
    selected = [(source, target) for source, target, match in pairs
                if match == ':' and 'CA' in source and 'CA' in target]
    if len(selected) < min_fit_pairs:
        selected = [(source, target) for source, target, _ in pairs
                    if 'CA' in source and 'CA' in target]
    if len(selected) < min_fit_pairs:
        raise ValueError(f'Too few aligned CA pairs for a stable fit: {len(selected)}')
    source = np.asarray([residue['CA'].coord for residue, _ in selected], dtype=float)
    target = np.asarray([residue['CA'].coord for _, residue in selected], dtype=float)
    rmsd, rotation, translation = _kabsch(source, target)
    return {
        'fit_pairs': len(selected),
        'fit_rmsd_A': round(rmsd, 2),
        'rotation': rotation.tolist(),
        'translation': translation.tolist(),
    }


def _wt_residue(model, number):
    for residue in model['A']:
        if residue.id[0] == ' ' and int(residue.id[1]) == number:
            return residue
    return None


def _wt_min_distance(points, model, residues=None):
    if residues is None:
        residues = [residue for residue in model['A'] if residue.id[0] == ' ']
    best = None
    for point in np.atleast_2d(points):
        for residue in residues:
            for atom in _heavy_atoms(residue):
                distance = float(np.linalg.norm(point - atom.coord))
                if best is None or distance < best[0]:
                    best = (distance, residue, atom)
    return best


def _state_ligands(structure, source):
    model = structure[0]
    ligands = []
    for chain_id in source['ligand_chains']:
        for residue in _ligand_residues(model[chain_id], set(source['ligands'])):
            ligands.append({
                'chain': chain_id,
                'resname': residue.resname.strip(),
                'resnum': int(residue.id[1]),
                'atoms': _heavy_atoms(residue),
            })
    return ligands


def _format_residue(residue, atom):
    return f'{residue.resname.strip()}{int(residue.id[1])}:{atom.name}'


def _write_csv(path, rows, fields):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in fields})


def build_nmnat_mapping(config_path, output):
    config = load_yaml(config_path)
    target = Target.load(config['target'])
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f'NMNAT mapping output already exists: {output}. '
            'Use a new --output run directory; existing results are not overwritten.')
    output.mkdir(parents=True, exist_ok=True)

    wt_paths = sorted(Path(path) for path in glob(config['wt']['glob']))
    if len(wt_paths) != int(config['wt']['expected_models']):
        raise FileNotFoundError('Matched WT model count differs from configuration')
    tmalign = Path(config['alignment']['tmalign'])
    if not tmalign.exists():
        raise FileNotFoundError(f'TMalign executable not found: {tmalign}')
    min_fit_pairs = int(config['alignment']['min_fit_pairs'])
    all_contacts = []
    mapping_rows = []
    state_rows = []
    transform_rows = []
    variant_rows = []
    interface_rows = []
    state_cache = {}

    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        for state, source in config['sources'].items():
            structure, model, protein, contacts = _source_contacts(state, source)
            state_cache[state] = {
                'structure': structure,
                'model': model,
                'protein': protein,
                'ligands': _state_ligands(structure, source),
                'source': source,
            }
            all_contacts.extend(contacts)
            source_pdb = temporary / f'{state}_source.pdb'
            _write_protein_pdb(structure, source['protein_chain'],
                               source['protein_range'][0], source['protein_range'][1],
                               source_pdb)
            for wt_path in wt_paths:
                wt_structure = _open_structure(wt_path)
                wt_model = wt_structure[0]
                target_residues = _protein_residues(wt_model['A'], 4, 181)
                target_pdb = temporary / f'{state}_{wt_path.stem}_target.pdb'
                _write_protein_pdb(wt_structure, 'A', 4, 181, target_pdb)
                raw = _run_tmalign(tmalign, source_pdb, target_pdb)
                parsed = _parse_tmalign(raw)
                pairs = _alignment_pairs(protein, target_residues, parsed)
                fit = _fit_from_pairs(pairs, min_fit_pairs)
                rotation = np.asarray(fit['rotation'], dtype=float)
                translation = np.asarray(fit['translation'], dtype=float)
                mapping = _mapping_from_pairs(pairs, rotation, translation)
                transform_rows.append({
                    'state': state,
                    'wt_model': wt_path.name,
                    'tm_scores': parsed['tm_scores'],
                    'aligned_length': parsed['aligned_length'],
                    'tmalign_rmsd_A': parsed['rmsd_A'],
                    'fit_pairs': fit['fit_pairs'],
                    'fit_rmsd_A': fit['fit_rmsd_A'],
                    'rotation': fit['rotation'],
                    'translation': fit['translation'],
                })
                transformed_ligands = []
                for ligand in state_cache[state]['ligands']:
                    atoms = []
                    for atom in ligand['atoms']:
                        point = _apply_transform([atom.coord], rotation, translation)[0]
                        atoms.append({'name': atom.name, 'coord': point})
                    transformed_ligands.append({
                        'chain': ligand['chain'],
                        'resname': ligand['resname'],
                        'resnum': ligand['resnum'],
                        'atoms': atoms,
                    })
                all_ligand_points = np.asarray(
                    [atom['coord'] for ligand in transformed_ligands for atom in ligand['atoms']])
                ligand_protein = _wt_min_distance(all_ligand_points, wt_model)
                severe = ligand_protein[0] < float(config['analysis']['clash_cutoff_A'])
                close = ligand_protein[0] < float(config['analysis']['close_cutoff_A'])
                ligand_pair = None
                if len(transformed_ligands) >= 2:
                    pair_values = []
                    for left in range(len(transformed_ligands)):
                        for right in range(left + 1, len(transformed_ligands)):
                            if {transformed_ligands[left]['resname'],
                                    transformed_ligands[right]['resname']} & {'MG', 'MN'}:
                                continue
                            left_atoms = transformed_ligands[left]['atoms']
                            right_atoms = transformed_ligands[right]['atoms']
                            if not left_atoms or not right_atoms:
                                continue
                            left_points = np.asarray([atom['coord'] for atom in left_atoms])
                            right_points = np.asarray([atom['coord'] for atom in right_atoms])
                            distance = float(np.min(np.linalg.norm(
                                left_points[:, None, :] - right_points[None, :, :], axis=2)))
                            pair_values.append((distance, transformed_ligands[left]['resname'],
                                                transformed_ligands[right]['resname']))
                    if pair_values:
                        ligand_pair = min(pair_values)
                interface_status = None
                interface_protein_min = None
                interface_basic_distance = None
                interface_basic = None
                if 'interface_chain' in source:
                    interface_chain = model[source['interface_chain']]
                    interface_residue = None
                    for residue in interface_chain:
                        if residue.id[0] == ' ' and int(residue.id[1]) == int(source['interface_residue']):
                            interface_residue = residue
                            break
                    if interface_residue is not None and 'NZ' in interface_residue:
                        interface_point = _apply_transform(
                            [interface_residue['NZ'].coord], rotation, translation)[0]
                        interface_protein_min = float(_wt_min_distance(
                            np.asarray([interface_point]), wt_model)[0])
                        probe = float(config['analysis']['interface_probe_A'])
                        best_basic = None
                        for residue in wt_model['A']:
                            if residue.id[0] != ' ' or residue.resname.strip() not in {'LYS', 'ARG'}:
                                continue
                            for atom in _heavy_atoms(residue):
                                if atom.name not in {'NZ', 'NE', 'NH1', 'NH2'}:
                                    continue
                                distance = float(np.linalg.norm(interface_point - atom.coord))
                                if best_basic is None or distance < best_basic[0]:
                                    best_basic = (distance, residue, atom)
                        if best_basic is not None:
                            interface_basic_distance = best_basic[0]
                            interface_basic = _format_residue(best_basic[1], best_basic[2])
                        if interface_protein_min < float(config['analysis']['clash_cutoff_A']):
                            interface_status = 'overlap'
                        elif (interface_basic_distance is not None
                              and 2.5 <= interface_basic_distance <= probe):
                            interface_status = 'weak_candidate'
                        else:
                            interface_status = 'no_candidate'
                        source_ligand_chain = model[source['ligand_chains'][0]]
                        source_ligands = _ligand_residues(
                            source_ligand_chain, {source['interface_ligand']})
                        source_distance = None
                        if source_ligands and source['interface_ligand_atom'] in source_ligands[0]:
                            source_distance = float(np.linalg.norm(
                                interface_residue['NZ'].coord
                                - source_ligands[0][source['interface_ligand_atom']].coord))
                        interface_rows.append({
                            'state': state,
                            'wt_model': wt_path.name,
                            'source_chain': source['interface_chain'],
                            'source_residue': f'LYS{int(interface_residue.id[1])}',
                            'source_atom': 'NZ',
                            'ligand_chain': source['ligand_chains'][0],
                            'ligand_resname': source['interface_ligand'],
                            'ligand_atom': source['interface_ligand_atom'],
                            'source_distance_A': round(source_distance, 2)
                            if source_distance is not None else '',
                            'protein_min_A': round(interface_protein_min, 2),
                            'nearest_basic': interface_basic or '',
                            'basic_distance_A': round(interface_basic_distance, 2)
                            if interface_basic_distance is not None else '',
                            'status': interface_status,
                        })
                metal_min = None
                metal_atp_o_min = None
                metal_nearest = None
                if any(ligand['resname'] in {'MG', 'MN'} for ligand in transformed_ligands):
                    metal_points = np.asarray([
                        atom['coord'] for ligand in transformed_ligands
                        if ligand['resname'] in {'MG', 'MN'} for atom in ligand['atoms']])
                    if metal_points.size:
                        nearest = _wt_min_distance(metal_points, wt_model)
                        metal_min = round(nearest[0], 2)
                        metal_nearest = _format_residue(nearest[1], nearest[2])
                        atp_oxygens = [
                            atom['coord'] for ligand in transformed_ligands
                            if ligand['resname'] == 'ATP' for atom in ligand['atoms']
                            if atom['name'].startswith('O')]
                        if atp_oxygens:
                            distances = [float(np.linalg.norm(point - oxygen))
                                         for point in metal_points for oxygen in atp_oxygens]
                            metal_atp_o_min = round(min(distances), 2)
                state_rows.append({
                    'state': state,
                    'wt_model': wt_path.name,
                    'source_contacts': len(contacts),
                    'mapped_contacts': sum(1 for row in contacts
                                           if int(row['protein_resnum']) in mapping),
                    'high_confidence': sum(1 for row in contacts
                                           if mapping.get(int(row['protein_resnum']),
                                                          {}).get('confidence') == 'high'),
                    'ligand_protein_min_A': round(ligand_protein[0], 2),
                    'ligand_protein_nearest': _format_residue(ligand_protein[1], ligand_protein[2]),
                    'severe_clash': severe,
                    'close_clash': close,
                    'ligand_ligand_min_A': round(ligand_pair[0], 2) if ligand_pair else '',
                    'ligand_ligand_pair': f'{ligand_pair[1]}-{ligand_pair[2]}' if ligand_pair else '',
                    'interface_status': interface_status or '',
                    'interface_nearest_basic': interface_basic or '',
                    'interface_basic_distance_A': round(interface_basic_distance, 2)
                    if interface_basic_distance is not None else '',
                    'interface_protein_min_A': round(interface_protein_min, 2)
                    if interface_protein_min is not None else '',
                    'metal_min_A': metal_min if metal_min is not None else '',
                    'metal_atp_o_min_A': metal_atp_o_min if metal_atp_o_min is not None else '',
                    'metal_nearest': metal_nearest or '',
                })
                ligand_atoms = {
                    (ligand['chain'], ligand['resnum'], atom.name): atom
                    for ligand in state_cache[state]['ligands']
                    for atom in ligand['atoms']
                }
                recovery_cutoff = float(source.get('contact_cutoff_A', 4.0))
                for row in contacts:
                    info = mapping.get(int(row['protein_resnum']))
                    ligand_atom = ligand_atoms[
                        (row['ligand_chain'], row['ligand_resnum'], row['ligand_atom'])]
                    point = _apply_transform([ligand_atom.coord], rotation, translation)[0]
                    nearest = _wt_min_distance(np.asarray([point]), wt_model)
                    mapped_residue = _wt_residue(wt_model, info['target_resnum']) if info else None
                    mapped_distance = None
                    if mapped_residue is not None:
                        mapped_distance = min(
                            float(np.linalg.norm(point - atom.coord))
                            for atom in _heavy_atoms(mapped_residue))
                    mapping_rows.append({
                        'contact_id': row['contact_id'],
                        'state': state,
                        'pdb_id': row['pdb_id'],
                        'wt_model': wt_path.name,
                        'source_protein_resnum': row['protein_resnum'],
                        'source_protein_resname': row['protein_resname'],
                        'source_protein_atom': row['protein_atom'],
                        'source_ligand_resname': row['ligand_resname'],
                        'source_ligand_atom': row['ligand_atom'],
                        'source_distance_A': row['distance_A'],
                        'mapping_status': 'mapped' if info else 'unmapped',
                        'kvp40_resnum': info['target_resnum'] if info else '',
                        'kvp40_resname': info['target_resname'] if info else '',
                        'mapping_match': info['match'] if info else '',
                        'mapping_ca_distance_A': round(info['ca_distance_A'], 2)
                        if info and info['ca_distance_A'] is not None else '',
                        'mapping_confidence': info['confidence'] if info else 'unmapped',
                        'transformed_ligand_to_mapped_residue_A': round(mapped_distance, 2)
                        if mapped_distance is not None else '',
                        'contact_recovered': mapped_distance is not None
                        and mapped_distance <= recovery_cutoff,
                        'wt_nearest_residue': _format_residue(nearest[1], nearest[2]),
                        'wt_nearest_distance_A': round(nearest[0], 2),
                        'clash_status': 'severe' if nearest[0] < float(config['analysis']['clash_cutoff_A'])
                        else ('close' if nearest[0] < float(config['analysis']['close_cutoff_A']) else 'none'),
                    })
                for variant, item in config['variants'].items():
                    for position in item['positions']:
                        residue = _wt_residue(wt_model, int(position))
                        if residue is None:
                            continue
                        variant_points = np.asarray([atom.coord for atom in _heavy_atoms(residue)])
                        best_ligand = None
                        for ligand in transformed_ligands:
                            for atom in ligand['atoms']:
                                distance = float(np.min(np.linalg.norm(
                                    variant_points - atom['coord'], axis=1)))
                                if best_ligand is None or distance < best_ligand[0]:
                                    best_ligand = (distance, ligand['resname'], atom['name'])
                        nearest_contact = None
                        for source_num, info in mapping.items():
                            contact_residue = _wt_residue(wt_model, info['target_resnum'])
                            if contact_residue is None:
                                continue
                            contact_points = np.asarray(
                                [atom.coord for atom in _heavy_atoms(contact_residue)])
                            distance = float(np.min(np.linalg.norm(
                                variant_points[:, None, :] - contact_points[None, :, :], axis=2)))
                            if nearest_contact is None or distance < nearest_contact[0]:
                                nearest_contact = (distance, info['target_resnum'],
                                                   info['target_resname'])
                        direct_cutoff = float(config['analysis']['variant_direct_A'])
                        motif_ranges = config['analysis'].get('variant_motif_ranges', [])
                        if best_ligand and best_ligand[0] <= direct_cutoff:
                            selection = 'direct_candidate'
                        elif any(start <= int(position) <= end for start, end in motif_ranges):
                            selection = 'motif_candidate'
                        else:
                            selection = 'not_selected'
                        variant_rows.append({
                            'variant': variant,
                            'position': position,
                            'wt_resname': residue.resname.strip(),
                            'state': state,
                            'wt_model': wt_path.name,
                            'nearest_ligand_resname': best_ligand[1] if best_ligand else '',
                            'nearest_ligand_atom': best_ligand[2] if best_ligand else '',
                            'min_distance_A': round(best_ligand[0], 2) if best_ligand else '',
                            'nearest_mapped_contact': f'{nearest_contact[2]}{nearest_contact[1]}'
                            if nearest_contact else '',
                            'selection': selection,
                        })

    contact_fields = ['contact_id', 'state', 'pdb_id', 'protein_chain', 'protein_resnum',
                      'protein_resname', 'protein_atom', 'ligand_chain', 'ligand_resname',
                      'ligand_resnum', 'ligand_atom', 'distance_A', 'contact_type',
                      'ligand_occupancy', 'ligand_altloc', 'ligand_b_factor']
    mapping_fields = ['contact_id', 'state', 'pdb_id', 'wt_model', 'source_protein_resnum',
                      'source_protein_resname', 'source_protein_atom', 'source_ligand_resname',
                      'source_ligand_atom', 'source_distance_A', 'mapping_status',
                      'kvp40_resnum', 'kvp40_resname', 'mapping_match', 'mapping_ca_distance_A',
                      'mapping_confidence', 'transformed_ligand_to_mapped_residue_A',
                      'contact_recovered', 'wt_nearest_residue', 'wt_nearest_distance_A',
                      'clash_status']
    state_fields = ['state', 'wt_model', 'source_contacts', 'mapped_contacts', 'high_confidence',
                    'ligand_protein_min_A', 'ligand_protein_nearest', 'severe_clash', 'close_clash',
                    'ligand_ligand_min_A', 'ligand_ligand_pair', 'interface_status',
                    'interface_nearest_basic', 'interface_basic_distance_A',
                    'interface_protein_min_A', 'metal_min_A', 'metal_atp_o_min_A', 'metal_nearest']
    variant_fields = ['variant', 'position', 'wt_resname', 'state', 'wt_model',
                      'nearest_ligand_resname', 'nearest_ligand_atom', 'min_distance_A',
                      'nearest_mapped_contact', 'selection']
    interface_fields = ['state', 'wt_model', 'source_chain', 'source_residue', 'source_atom',
                        'ligand_chain', 'ligand_resname', 'ligand_atom', 'source_distance_A',
                        'protein_min_A', 'nearest_basic', 'basic_distance_A', 'status']
    _write_csv(output / 'experimental_contacts.csv', all_contacts, contact_fields)
    _write_csv(output / 'mapping.csv', mapping_rows, mapping_fields)
    _write_csv(output / 'state_summary.csv', state_rows, state_fields)
    _write_csv(output / 'variant_selection.csv', variant_rows, variant_fields)
    _write_csv(output / 'interface.csv', interface_rows, interface_fields)
    write_json(output / 'transforms.json', transform_rows)
    state_summary = {}
    for state in config['sources']:
        state_model_rows = [row for row in state_rows if row['state'] == state]
        state_summary[state] = {
            'models': [row['wt_model'] for row in state_model_rows],
            'source_contacts': state_model_rows[0]['source_contacts'],
            'mapped_contacts': [row['mapped_contacts'] for row in state_model_rows],
            'high_confidence': [row['high_confidence'] for row in state_model_rows],
            'recovered_contacts': [
                sum(1 for row in mapping_rows
                    if row['state'] == state and row['wt_model'] == model_row['wt_model']
                    and row['contact_recovered'])
                for model_row in state_model_rows
            ],
            'ligand_protein_min_A': [row['ligand_protein_min_A'] for row in state_model_rows],
            'severe_clash_models': sum(bool(row['severe_clash']) for row in state_model_rows),
            'close_clash_models': sum(bool(row['close_clash']) for row in state_model_rows),
            'ligand_ligand_min_A': [row['ligand_ligand_min_A'] for row in state_model_rows
                                    if row['ligand_ligand_min_A'] != ''],
            'interface_status': [row['interface_status'] for row in state_model_rows
                                 if row['interface_status']],
            'metal_atp_o_min_A': [row['metal_atp_o_min_A'] for row in state_model_rows
                                  if row['metal_atp_o_min_A'] != ''],
            'note': 'Rigid ligand transfer onto apo predictions; not a predicted complex',
        }
    write_json(output / 'state_summary.json', state_summary)

    source_manifest = []
    for state, source in config['sources'].items():
        source_manifest.append({
            'state': state,
            'pdb_id': source['pdb_id'],
            'path': source['path'],
            'sha256': sha256_file(source['path']),
            'assembly_id': source['assembly_id'],
            'protein_chain': source['protein_chain'],
            'protein_range': source['protein_range'],
            'ligand_chains': source['ligand_chains'],
            'ligands': source['ligands'],
        })
    wt_manifest = []
    for path in wt_paths:
        wt_manifest.append({
            'path': str(path),
            'sha256': sha256_file(path),
            'sequence_sha256': hashlib.sha256(
                _sequence(_protein_residues(_open_structure(path)[0]['A'], 1, 341)).encode()).hexdigest(),
        })
    variant_manifest = []
    for variant, item in config['variants'].items():
        for path in sorted(Path(path) for path in glob(item['glob'])):
            variant_manifest.append({
                'variant': variant,
                'path': str(path),
                'sha256': sha256_file(path),
            })
    tmalign_version = next(
        (line.strip() for line in subprocess.run(
            [str(tmalign), '-v'], capture_output=True, text=True).stdout.splitlines()
         if 'Version' in line), 'unknown')
    provenance = {
        'config': str(Path(config_path).resolve()),
        'config_sha256': sha256_file(config_path),
        'target_sequence_sha256': hashlib.sha256(target.sequence.encode()).hexdigest(),
        'target_length': len(target.sequence),
        'wt_models': wt_manifest,
        'wt_identity': config['wt'],
        'variant_models': variant_manifest,
        'sources': source_manifest,
        'negative_controls': config.get('negative_controls', {}),
        'tmalign': {
            'path': str(tmalign),
            'version': tmalign_version,
        },
        'software': {
            'python': sys.version.split()[0],
            'numpy': np.__version__,
            'biopython': __import__('Bio').__version__,
        },
        'analysis': config['analysis'],
    }
    write_json(output / 'provenance.json', provenance)
    return output
