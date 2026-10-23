"""Map experimental NMN contacts across homologs without treating them as one pocket."""
from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from .data import load_yaml, sha256_file, write_json
from .nmnat_mapping import (_alignment_pairs, _fit_from_pairs, _mapping_from_pairs,
                            _open_structure, _parse_tmalign, _protein_residues,
                            _run_tmalign, _write_protein_pdb)

BACKBONE = {'N', 'CA', 'C', 'O', 'OXT'}
PHOSPHATE = {'P', 'O1P', 'O2P', 'O3P'}
PHOSPHATE_BRIDGE = {'O5R'}
RIBOSE = {'C5R', 'C4R', 'O4R', 'C3R', 'O3R', 'C2R', 'O2R', 'C1R'}
NICOTINAMIDE = {'N1', 'C2', 'C3', 'C7', 'O7', 'N7', 'C4', 'C5', 'C6'}
RING_ATOMS = {
    'PHE': {'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'},
    'TYR': {'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'},
    'TRP': {'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'},
    'HIS': {'CG', 'ND1', 'CD2', 'CE1', 'NE2'},
}
HYDROXYL = {'SER': {'OG'}, 'THR': {'OG1'}, 'TYR': {'OH'}}
CARBOXYL = {'ASP': {'OD1', 'OD2'}, 'GLU': {'OE1', 'OE2'}}
AMIDE = {'ASN': {'OD1', 'ND2'}, 'GLN': {'OE1', 'NE2'}}
CATIONIC = {'LYS': {'NZ'}, 'ARG': {'NE', 'NH1', 'NH2'}}
IMIDAZOLE = {'HIS': {'ND1', 'NE2'}}
SULFUR = {'MET': {'SD'}, 'MSE': {'SD', 'SE'}, 'CYS': {'SG'}}


def _ligand_group(name):
    if name in PHOSPHATE:
        return 'phosphate'
    if name in PHOSPHATE_BRIDGE:
        return 'phosphate_bridge'
    if name in RIBOSE:
        return 'ribose'
    if name in NICOTINAMIDE:
        return 'nicotinamide'
    return 'other'


def _protein_group(resname, atom_name):
    if atom_name in BACKBONE:
        return 'backbone'
    if resname == 'MSE' and atom_name == 'SE':
        return 'selenium'
    parent = 'MET' if resname == 'MSE' else resname
    for group, table in (('hydroxyl', HYDROXYL), ('carboxyl', CARBOXYL),
                         ('amide', AMIDE), ('cationic', CATIONIC),
                         ('imidazole', IMIDAZOLE), ('sulfur', SULFUR)):
        if atom_name in table.get(parent, set()):
            return group
    if parent in RING_ATOMS and atom_name in RING_ATOMS[parent]:
        return 'aromatic_ring'
    return 'aliphatic'


def _atom_function(resname, atom_name):
    if resname == 'MSE' and atom_name == 'SE':
        return 'selenium_not_equivalent'
    parent = 'MET' if resname == 'MSE' else resname
    group = _protein_group(parent, atom_name)
    if parent == 'TYR' and atom_name == 'OH':
        return 'tyrosine_hydroxyl'
    if parent == 'TRP' and atom_name == 'NE1':
        return 'tryptophan_ring_NH'
    if parent == 'HIS' and atom_name in IMIDAZOLE['HIS']:
        return 'histidine_imidazole'
    return group


def _compatibility(source_resname, source_atom, target_resname, target_atoms,
                   modified_source=False):
    source_group = _protein_group(source_resname, source_atom)
    target_groups = {_protein_group(target_resname, atom) for atom in target_atoms}
    if modified_source:
        return 'modified_sidechain_not_equivalent', source_group, ';'.join(sorted(target_groups))
    if target_resname == 'GLY' and source_group != 'backbone':
        return 'target_lacks_sidechain', source_group, 'backbone'
    if source_group == 'backbone':
        return ('backbone_same_name' if source_atom in target_atoms
                else 'backbone_no_same_name', source_group, ';'.join(sorted(target_groups)))
    if source_resname == target_resname and source_atom in target_atoms:
        return 'same_residue_atom', source_group, ';'.join(sorted(target_groups))
    if source_group == 'aromatic_ring' and 'aromatic_ring' in target_groups:
        return ('aromatic_same_atom_name' if source_atom in target_atoms
                else 'aromatic_ring_role', source_group, ';'.join(sorted(target_groups)))
    if source_group in target_groups:
        if source_atom in target_atoms:
            return 'same_group_atom_name', source_group, ';'.join(sorted(target_groups))
        return 'role_equivalent', source_group, ';'.join(sorted(target_groups))
    return 'no_role_equivalent', source_group, ';'.join(sorted(target_groups))


def _mse_as_met_structure(structure, chain_id, start, end):
    from Bio.PDB import Atom, Chain, Model, Residue, Structure
    output = Structure.Structure('mse_as_met')
    model = Model.Model(0)
    output.add(model)
    chain = Chain.Chain(chain_id)
    model.add(chain)
    for residue in structure[chain_id]:
        if not start <= int(residue.id[1]) <= end:
            continue
        if residue.id[0] == ' ':
            resname = residue.resname
            modified = False
        elif residue.resname == 'MSE':
            resname = 'MET'
            modified = True
        else:
            continue
        new = Residue.Residue((' ', int(residue.id[1]), ' '), resname, '')
        for atom in residue:
            if atom.element == 'H':
                continue
            name = 'SD' if modified and atom.name == 'SE' else atom.name
            element = 'S' if modified and atom.element == 'SE' else atom.element
            new.add(Atom.Atom(name, atom.coord, atom.get_bfactor(), atom.get_occupancy(),
                              atom.get_altloc(), name, atom.get_serial_number(), element=element))
        chain.add(new)
    return output


def _source_residues(structure, chain_id, start, end):
    residues = []
    for residue in structure[chain_id]:
        if not start <= int(residue.id[1]) <= end:
            continue
        if residue.id[0] == ' ' or residue.resname == 'MSE':
            residues.append(residue)
    return residues


def _residue_map(reference_protein, source_protein, source_pdb, reference_pdb,
                 tmalign, min_fit_pairs):
    parsed = _parse_tmalign(_run_tmalign(tmalign, source_pdb, reference_pdb))
    pairs = _alignment_pairs(source_protein, reference_protein, parsed)
    fit = _fit_from_pairs(pairs, min_fit_pairs)
    mapping = _mapping_from_pairs(pairs, np.asarray(fit['rotation']),
                                  np.asarray(fit['translation']))
    return mapping, parsed, fit


def analyze_nmn_sources(config_path, output):
    config = load_yaml(config_path)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'NMN source output already exists: {output}')
    output.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(prefix='natv_nmn_sources_')
    tmpdir = Path(temporary.name)
    reference = _open_structure(config['reference'])
    reference_protein = _protein_residues(reference[0]['A'], 4, 181)
    reference_pdb = tmpdir / 'reference.pdb'
    _write_protein_pdb(reference, 'A', 4, 181, reference_pdb)
    reference_atoms = {int(residue.id[1]): residue for residue in reference_protein}
    tmalign = config['alignment']['tmalign']
    min_fit_pairs = int(config['alignment']['min_fit_pairs'])
    rows = []
    provenance = {'config': str(Path(config_path).resolve()),
                  'config_sha256': sha256_file(config_path),
                  'reference': config['reference'],
                  'reference_sha256': sha256_file(config['reference']),
                  'implementation_sha256': sha256_file(__file__),
                  'tmalign': config['alignment']['tmalign'],
                  'sources': {}, 'ligand': 'NMN',
                  'contact_cutoff_A': config['contact_cutoff_A'],
                  'method': 'TMalign source chain to WT0 NMNAT 4-181; NMN contacts <= cutoff; '
                            'chemical groups and atom/role compatibility recorded'}
    for pdb_id, item in config['sources'].items():
        path = Path(item['path'])
        structure = _open_structure(path)
        source_residues = _source_residues(structure[0], item['chain'],
                                           item['protein_range'][0], item['protein_range'][1])
        if item.get('mse_as_met'):
            work = _mse_as_met_structure(structure[0], item['chain'],
                                         item['protein_range'][0], item['protein_range'][1])
            source_pdb = tmpdir / f'{pdb_id}_mse_as_met.pdb'
            _write_protein_pdb(work, item['chain'], item['protein_range'][0],
                               item['protein_range'][1], source_pdb)
            work_protein = _protein_residues(work[0][item['chain']],
                                             item['protein_range'][0], item['protein_range'][1])
        else:
            source_pdb = tmpdir / f'{pdb_id}_source.pdb'
            _write_protein_pdb(structure, item['chain'], item['protein_range'][0],
                               item['protein_range'][1], source_pdb)
            work_protein = source_residues
        work_protein = [residue for residue in work_protein if 'CA' in residue]
        mapping, parsed, fit = _residue_map(reference_protein, work_protein, source_pdb,
                                            reference_pdb, tmalign, min_fit_pairs)
        provenance['sources'][pdb_id] = {
            'path': str(path), 'sha256': sha256_file(path),
            'family': item['family'], 'chain': item['chain'],
            'protein_range': item['protein_range'],
            'note': item.get('note', ''),
            'tm_scores': parsed['tm_scores'], 'tmalign_rmsd_A': parsed['rmsd_A'],
            'fit_rmsd_A': fit['fit_rmsd_A'], 'fit_pairs': fit['fit_pairs'],
            'mapped_residues': len(mapping),
        }
        chain = structure[0][item['chain']]
        ligands = [residue for residue in chain
                   if residue.id[0].startswith('H_') and residue.resname == 'NMN']
        if len(ligands) != 1:
            raise ValueError(f'Expected one NMN in {pdb_id}, got {len(ligands)}')
        ligand = ligands[0]
        for residue in source_residues:
            modified = residue.resname == 'MSE'
            for source_atom in residue:
                if source_atom.element == 'H':
                    continue
                for ligand_atom in ligand:
                    if ligand_atom.element == 'H':
                        continue
                    distance = float(np.linalg.norm(source_atom.coord - ligand_atom.coord))
                    if distance > float(config['contact_cutoff_A']):
                        continue
                    info = mapping.get(int(residue.id[1]))
                    target = reference_atoms.get(int(info['target_resnum'])) if info else None
                    target_atoms = [atom.name for atom in target] if target is not None else []
                    target_functions = ';'.join(sorted({
                        _atom_function(target.resname, atom.name) for atom in target}
                    )) if target is not None else ''
                    compatibility = 'unmapped'
                    target_group = ''
                    is_selenium = modified and (source_atom.name.upper() == 'SE' or
                                                source_atom.element.strip().upper() == 'SE')
                    if target is not None and is_selenium:
                        compatibility = 'modified_selenium_not_equivalent'
                        target_group = 'selenium'
                    elif target is not None:
                        compatibility, source_group, target_group = _compatibility(
                            residue.resname, source_atom.name, target.resname, target_atoms,
                            modified_source=False)
                    else:
                        source_group = _protein_group(residue.resname, source_atom.name)
                    rows.append({
                        'pdb_id': pdb_id, 'family': item['family'],
                        'source_resnum': int(residue.id[1]), 'source_resname': residue.resname,
                        'source_atom': source_atom.name,
                        'source_group': _protein_group(residue.resname, source_atom.name),
                        'source_function': _atom_function(residue.resname, source_atom.name),
                        'ligand_atom': ligand_atom.name,
                        'ligand_group': _ligand_group(ligand_atom.name),
                        'distance_A': round(distance, 2),
                        'target_resnum': info['target_resnum'] if info else '',
                        'target_resname': info['target_resname'] if info else '',
                        'target_atoms': ';'.join(target_atoms),
                        'target_functions': target_functions,
                        'target_group': target_group,
                        'mapping_confidence': info['confidence'] if info else 'unmapped',
                        'mapping_ca_distance_A': round(info['ca_distance_A'], 2)
                        if info and info['ca_distance_A'] is not None else '',
                        'compatibility': compatibility,
                        'modified_residue': modified,
                        'source_note': item.get('note', ''),
                    })
    fields = ['pdb_id', 'family', 'source_resnum', 'source_resname', 'source_atom',
              'source_group', 'source_function', 'ligand_atom', 'ligand_group',
              'distance_A', 'target_resnum', 'target_resname', 'target_atoms',
              'target_functions', 'target_group', 'mapping_confidence',
              'mapping_ca_distance_A', 'compatibility', 'modified_residue',
              'source_note']
    with (output / 'source_contacts.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    groups = {}
    for row in rows:
        if not row['target_resnum']:
            continue
        key = (row['target_resnum'], row['target_resname'], row['ligand_group'])
        group = groups.setdefault(key, {
            'families': set(), 'sources': set(), 'rows': 0,
            'min_distance_A': row['distance_A'], 'source_groups': set(),
            'compatibilities': set(), 'target_atoms': set(),
        })
        group['families'].add(row['family'])
        group['sources'].add(row['pdb_id'])
        group['rows'] += 1
        group['min_distance_A'] = min(group['min_distance_A'], row['distance_A'])
        group['source_groups'].add(row['source_group'])
        group['compatibilities'].add(row['compatibility'])
        if row['target_atoms']:
            group['target_atoms'].add(row['target_atoms'])
    summary = []
    for (target_resnum, target_resname, ligand_group), group in sorted(groups.items()):
        summary.append({
            'target_resnum': target_resnum, 'target_resname': target_resname,
            'ligand_group': ligand_group,
            'n_families': len(group['families']),
            'families': ';'.join(sorted(group['families'])),
            'sources': ';'.join(sorted(group['sources'])),
            'contact_rows': group['rows'], 'min_distance_A': group['min_distance_A'],
            'source_groups': ';'.join(sorted(group['source_groups'])),
            'target_atoms': ';'.join(sorted(group['target_atoms'])),
            'compatibilities': ';'.join(sorted(group['compatibilities'])),
        })
    with (output / 'target_group_summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            'target_resnum', 'target_resname', 'ligand_group', 'n_families',
            'families', 'sources', 'contact_rows', 'min_distance_A',
            'source_groups', 'target_atoms', 'compatibilities'], lineterminator='\n')
        writer.writeheader()
        writer.writerows(summary)
    write_json(output / 'provenance.json', provenance)
    temporary.cleanup()
    return {'status': 'COMPLETED', 'contacts': len(rows), 'groups': len(summary),
            'output': str(output)}
