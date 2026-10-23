"""Prepare and run explicit-solvent WT NMNAT joint-substrate systems."""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from .data import load_yaml, sha256_file, write_json

SMILES = {
    'ATP': 'Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])[O-])[C@@H](O)[C@H]1O',
    'NMN': 'NC(=O)c1ccc[n+](c1)[C@@H]1O[C@H](COP(=O)([O-])[O-])[C@@H](O)[C@H]1O',
}
CHARGE = {'ATP': -4, 'NMN': -1}
ATOM_COUNT = {'ATP': 43, 'NMN': 36}
SOLVATE_BUFFER_A = 12.0  # tleap solvateBox uses angstroms; 12 A = 1.2 nm


def _run(command, cwd, log):
    with Path(log).open('w') as handle:
        handle.write('$ ' + ' '.join(str(item) for item in command) + '\n')
        handle.flush()
        result = subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}): {command}; see {log}')
    return result


def _ligand_graph_mapping(ccd, ligand):
    import networkx as nx
    from rdkit import Chem

    param = Chem.MolFromSmiles(SMILES[ligand])
    graph_c, graph_p = nx.Graph(), nx.Graph()
    for mol, graph in ((ccd, graph_c), (param, graph_p)):
        for atom in mol.GetAtoms():
            graph.add_node(atom.GetIdx(), elem=atom.GetSymbol())
        for bond in mol.GetBonds():
            graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                           order=round(bond.GetBondTypeAsDouble(), 1))
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        graph_c, graph_p, node_match=lambda a, b: a['elem'] == b['elem'],
        edge_match=lambda a, b: a['order'] == b['order'])
    if not matcher.is_isomorphic():
        raise ValueError(f'Ligand graph mismatch: {ligand}')
    names = {atom.GetIdx(): atom.GetPropsAsDict().get('name') for atom in ccd.GetAtoms()}
    return {param_index: names[ccd_index]
            for ccd_index, param_index in matcher.mapping.items()}


def _correct_charge(path, ligand):
    path = Path(path)
    lines = path.read_text().splitlines()
    start = lines.index('@<TRIPOS>ATOM') + 1
    end = lines.index('@<TRIPOS>BOND')
    atoms = [index for index in range(start, end) if lines[index].strip()]
    charges = {index: float(lines[index].split()[8]) for index in atoms}
    residual = CHARGE[ligand] - sum(charges.values())
    if abs(residual) > 0.01:
        raise ValueError(f'{ligand} charge residual too large: {residual}')
    max_change = 0.0
    if abs(residual) > 1e-12:
        target = next(index for index in atoms
                      if lines[index].split()[5].startswith('p'))
        fields = lines[target].split()
        fields[8] = f'{float(fields[8]) + residual:.8f}'
        max_change = abs(float(fields[8]) - charges[target])
        lines[target] = ' '.join(fields)
        path.write_text('\n'.join(lines) + '\n')
    corrected = sum(float(lines[index].split()[8]) for index in atoms)
    if abs(corrected - CHARGE[ligand]) > 1e-6:
        raise ValueError(f'{ligand} corrected charge is {corrected}')
    return residual, max_change


def _mol2_elements(path):
    lines = Path(path).read_text().splitlines()
    start = lines.index('@<TRIPOS>ATOM') + 1
    end = lines.index('@<TRIPOS>BOND')
    aliases = {'n': 'N', 'c': 'C', 'o': 'O', 'p': 'P', 'h': 'H',
               's': 'S', 'f': 'F', 'b': 'B', 'l': 'L', 'r': 'R'}
    return [aliases.get(lines[index].split()[5][0].lower(), '?')
            for index in range(start, end) if lines[index].strip()]


def _mol2_atom_serials(path):
    lines = Path(path).read_text().splitlines()
    start = lines.index('@<TRIPOS>ATOM') + 1
    end = lines.index('@<TRIPOS>BOND')
    return [lines[index].split()[0] for index in range(start, end)
            if lines[index].strip()]


def _mol2_bonds(path):
    lines = Path(path).read_text().splitlines()
    start = lines.index('@<TRIPOS>BOND') + 1
    end = next((index for index in range(start, len(lines))
                if lines[index].startswith('@<TRIPOS>')), len(lines))
    return {tuple(sorted((fields[1], fields[2])))
            for fields in (lines[index].split() for index in range(start, end)
                           if lines[index].strip())
            if len(fields) >= 4}


def _check_parameter_order(ligand, path):
    from rdkit import Chem

    molecule = Chem.AddHs(Chem.MolFromSmiles(SMILES[ligand]))
    expected = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    observed = _mol2_elements(path)
    if observed != expected:
        raise ValueError(f'{ligand} acpype atom order differs from SMILES order')
    serials = _mol2_atom_serials(path)
    if len(serials) != len(expected) or len(set(serials)) != len(serials):
        raise ValueError(f'{ligand} acpype atom serials are inconsistent')
    position = {serial: index for index, serial in enumerate(serials)}
    observed_bonds = {tuple(sorted((position[first], position[second])))
                      for first, second in _mol2_bonds(path)}
    expected_bonds = {tuple(sorted((bond.GetBeginAtomIdx(),
                                    bond.GetEndAtomIdx())))
                      for bond in molecule.GetBonds()}
    if observed_bonds != expected_bonds:
        raise ValueError(f'{ligand} acpype bond order differs from SMILES order')


def _parameterize(config, output):
    import pickle
    from rdkit import Chem
    from rdkit.Chem import AllChem

    parameter_root = output / 'parameters'
    parameter_root.mkdir()
    records = {}
    for ligand in ('ATP', 'NMN'):
        molecule = Chem.AddHs(Chem.MolFromSmiles(SMILES[ligand]))
        AllChem.EmbedMolecule(molecule, randomSeed=7)
        source = parameter_root / f'{ligand}.sdf'
        writer = Chem.SDWriter(str(source))
        writer.write(molecule)
        writer.close()
        _run([config['acpype'], '-i', str(source), '-b', ligand,
              '-c', 'bcc', '-a', 'gaff2', '-n', str(CHARGE[ligand])],
             parameter_root, parameter_root / f'{ligand}_acpype.log')
        mol2 = parameter_root / f'{ligand}.acpype' / f'{ligand}_bcc_gaff2.mol2'
        frcmod = parameter_root / f'{ligand}.acpype' / f'{ligand}_AC.frcmod'
        if not mol2.exists() or not frcmod.exists():
            raise FileNotFoundError(f'Missing acpype output for {ligand}')
        _check_parameter_order(ligand, mol2)
        residual, max_change = _correct_charge(mol2, ligand)
        records[ligand] = {
            'mol2': str(mol2.resolve()), 'frcmod': str(frcmod.resolve()),
            'charge': CHARGE[ligand], 'charge_residual': residual,
            'max_atom_charge_change': max_change,
            'acpype_log': str((parameter_root / f'{ligand}_acpype.log').resolve()),
        }
    return records


def _structure(path):
    from Bio.PDB import MMCIFParser
    return MMCIFParser(QUIET=True).get_structure('x', str(path))[0]


def _protein_pdb(structure, path):
    from Bio.PDB import PDBIO, Select

    class SelectProtein(Select):
        def accept_chain(self, chain):
            return chain.id == 'A'

        def accept_residue(self, residue):
            return residue.id[0] == ' ' and 1 <= residue.id[1] <= 341

        def accept_atom(self, atom):
            return atom.element != 'H'

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(path), SelectProtein())


def _ligand_mol2(structure, ligand, template, output, ccd_cache):
    import pickle
    from rdkit import Chem

    ccd = Chem.RemoveHs(pickle.load(open(
        str(Path(ccd_cache) / f'{ligand}.pkl'), 'rb')))
    mapping = _ligand_graph_mapping(ccd, ligand)
    residue = next((row for row in structure.get_residues()
                    if row.resname == ligand), None)
    if residue is None:
        raise ValueError(f'Missing {ligand}')
    coords = {atom.name: atom.coord.astype(float) for atom in residue}
    heavy = Chem.MolFromSmiles(SMILES[ligand])
    conformer = Chem.Conformer(heavy.GetNumAtoms())
    for index in range(heavy.GetNumAtoms()):
        conformer.SetAtomPosition(index, coords[mapping[index]])
    heavy.RemoveAllConformers()
    heavy.AddConformer(conformer)
    molecule = Chem.AddHs(heavy, addCoords=True)
    if molecule.GetNumAtoms() != ATOM_COUNT[ligand]:
        raise ValueError(f'Unexpected {ligand} atom count: {molecule.GetNumAtoms()}')
    lines = Path(template).read_text().splitlines()
    lines[lines.index('@<TRIPOS>MOLECULE') + 1] = ligand
    start = lines.index('@<TRIPOS>ATOM') + 1
    end = lines.index('@<TRIPOS>BOND')
    atoms = [index for index in range(start, end) if lines[index].strip()]
    if len(atoms) != molecule.GetNumAtoms():
        raise ValueError(f'Unexpected {ligand} template atom count: {len(atoms)}')
    conformer = molecule.GetConformer()
    for position, index in enumerate(atoms):
        fields = lines[index].split()
        point = conformer.GetAtomPosition(position)
        fields[2] = f'{point.x:.4f}'
        fields[3] = f'{point.y:.4f}'
        fields[4] = f'{point.z:.4f}'
        fields[7] = ligand
        lines[index] = ' '.join(fields)
    substructure = lines.index('@<TRIPOS>SUBSTRUCTURE') + 1
    fields = lines[substructure].split()
    fields[1] = ligand
    lines[substructure] = ' '.join(fields)
    Path(output).write_text('\n'.join(lines) + '\n')


def _mg_pdb(structure, path):
    residue = next((row for row in structure.get_residues()
                    if row.resname == 'MG'), None)
    if residue is None:
        return False
    atom = next((row for row in residue if row.element == 'MG'), None)
    if atom is None:
        return False
    Path(path).write_text(
        'HETATM    1 MG   MG  A   1    '
        f'{atom.coord[0]:8.3f}{atom.coord[1]:8.3f}{atom.coord[2]:8.3f}'
        '  1.00  0.00          MG\nEND\n')
    return True


def _tleap(config, model_dir, parameters, has_mg):
    lines = [
        'source leaprc.protein.ff14SB',
        'source leaprc.gaff2',
        'source leaprc.water.tip3p',
        f"loadAmberParams {parameters['ATP']['frcmod']}",
        f"loadAmberParams {parameters['NMN']['frcmod']}",
        'atp = loadMol2 ATP_model.mol2',
        'nmn = loadMol2 NMN_model.mol2',
        'prot = loadPdb protein.pdb',
    ]
    if has_mg:
        lines += ['mg = loadPdb mg.pdb', 'com = combine { prot atp nmn mg }']
    else:
        lines += ['com = combine { prot atp nmn }']
    lines += [f'solvateBox com TIP3PBOX {SOLVATE_BUFFER_A}', 'addIons com Na+ 0',
              'saveAmberParm com com.prmtop com.rst7', 'quit']
    script = model_dir / 'tleap.in'
    script.write_text('\n'.join(lines) + '\n')
    _run([config['tleap'], '-f', 'tleap.in'], model_dir, model_dir / 'tleap.log')
    text = (model_dir / 'tleap.log').read_text()
    if 'Errors = 0' not in text:
        raise RuntimeError(f'tleap errors: {model_dir / "tleap.log"}')


def _export_gromacs(model_dir, config, has_mg):
    import parmed as pmd

    structure = pmd.load_file(str(model_dir / 'com.prmtop'),
                              xyz=str(model_dir / 'com.rst7'))
    structure.save(str(model_dir / 'com_gmx.top'), format='gromacs')
    structure.save(str(model_dir / 'com_gmx.gro'), format='gro')
    charge = float(sum(atom.charge for atom in structure.atoms))
    return charge, _structure_metrics(structure, has_mg)


def _minimum_image(first, second, box):
    left = np.array([[atom.xx, atom.xy, atom.xz] for atom in first])
    right = np.array([[atom.xx, atom.xy, atom.xz] for atom in second])
    best = (float('inf'), None, None)
    for atom, point in zip(first, left):
        delta = right - point
        delta -= box * np.round(delta / box)
        distance = np.linalg.norm(delta, axis=1)
        index = int(distance.argmin())
        if distance[index] < best[0]:
            best = (float(distance[index]), atom, second[index])
    return best


def _structure_metrics(structure, has_mg):
    box = np.asarray(structure.box[:3], dtype=float)
    if abs(float(structure.box[3]) - 90.0) > 1e-3:
        raise ValueError('Only orthorhombic boxes are expected')
    metrics = {}
    ligands = {}
    for ligand in ('ATP', 'NMN'):
        atoms = [atom for atom in structure.atoms if atom.residue.name == ligand]
        if len(atoms) != ATOM_COUNT[ligand]:
            raise ValueError(f'{ligand} atom count in topology: {len(atoms)}')
        ligand_charge = float(sum(atom.charge for atom in atoms))
        if abs(ligand_charge - CHARGE[ligand]) > 1e-3:
            raise ValueError(f'{ligand} topology charge: {ligand_charge}')
        ligands[ligand] = [atom for atom in atoms if atom.element != 1]
        metrics[f'{ligand.lower()}_topology_charge'] = round(ligand_charge, 6)
        metrics[f'{ligand.lower()}_atom_count'] = len(atoms)
    protein = [atom for atom in structure.atoms
               if atom.residue.name not in ('ATP', 'NMN', 'MG', 'Na+', 'WAT')]
    residues = {atom.residue.idx for atom in protein}
    if len(residues) != 341:
        raise ValueError(f'Protein residue count in topology: {len(residues)}')
    heavy_protein = [atom for atom in protein if atom.element != 1]
    cross = _minimum_image(ligands['ATP'], ligands['NMN'], box)
    metrics.update({
        'box_nm': ';'.join(f'{value / 10:.4f}' for value in box),
        'protein_residues': len(residues),
        'protein_charge': round(float(sum(atom.charge for atom in protein)), 4),
        'cross_min_A': round(cross[0], 3),
        'cross_min_atoms': f'ATP:{cross[1].name};NMN:{cross[2].name}',
        'atp_protein_min_A': round(_minimum_image(
            ligands['ATP'], heavy_protein, box)[0], 3),
        'nmn_protein_min_A': round(_minimum_image(
            ligands['NMN'], heavy_protein, box)[0], 3),
    })
    metals = [atom for atom in structure.atoms if atom.residue.name == 'MG']
    if has_mg:
        if len(metals) != 1:
            raise ValueError(f'Expected one MG, found {len(metals)}')
        metal = metals[0]
        metrics.update({
            'mg_charge': round(float(metal.charge), 3),
            'mg_sigma_A': round(float(metal.sigma), 6),
            'mg_epsilon_kJ_mol': round(float(metal.epsilon) * 4.184, 6),
            'mg_atp_min_A': round(_minimum_image([metal], ligands['ATP'], box)[0], 3),
            'mg_nmn_min_A': round(_minimum_image([metal], ligands['NMN'], box)[0], 3),
        })
    elif metals:
        raise ValueError(f'Unexpected MG in no-Mg system: {len(metals)}')
    return metrics


def build_nmnat_md(config_path, output):
    import networkx  # noqa: F401  (checked here so the dependency is explicit)
    import parmed  # noqa: F401
    import rdkit  # noqa: F401

    started_at = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    code_sha256 = sha256_file(Path(__file__))
    config = load_yaml(config_path)
    config_sha256 = sha256_file(config_path)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'NMNAT MD output already exists: {output}')
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, output / 'config_used.yaml')
    source = Path(config['joint_run']) / 'analysis_identity.json'
    models = json.load(open(source))['models']
    if config.get('models'):
        selected = set(config['models'])
        models = [model for model in models if Path(model['path']).stem in selected]
    parameters = _parameterize(config, output)
    rows = []
    for model in models:
        path = Path(model['path'])
        name = path.stem
        state = 'ATP_NMN_MG' if '_MG_' in name else 'ATP_NMN'
        seed = int(re.search(r'seed(\d+)', name).group(1))
        model_dir = output / 'models' / filter_name(name)
        model_dir.mkdir(parents=True)
        structure = _structure(path)
        _protein_pdb(structure, model_dir / 'protein.pdb')
        for ligand in ('ATP', 'NMN'):
            _ligand_mol2(structure, ligand, parameters[ligand]['mol2'],
                         model_dir / f'{ligand}_model.mol2', config['ccd_cache'])
        has_mg = _mg_pdb(structure, model_dir / 'mg.pdb')
        _tleap(config, model_dir, parameters, has_mg)
        charge, metrics = _export_gromacs(model_dir, config, has_mg)
        rows.append({
            'state': state, 'seed': seed, 'model': path.name,
            'cif': str(path), 'cif_sha256': model['sha256'],
            'directory': str(model_dir), 'has_mg': has_mg,
            'system_charge': charge, **metrics, 'status': 'BUILT',
        })
    fields = _system_fields(rows)
    with (output / 'systems.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / 'build.json', {
        'config': str(Path(config_path).resolve()),
        'config_sha256': config_sha256,
        'code': str(Path(__file__).resolve()),
        'code_sha256': code_sha256,
        'started_at': started_at,
        'joint_run': config['joint_run'],
        'models': rows, 'parameters': parameters,
    })
    if config.get('initial_manifest', True):
        _write_initial_manifest(output, config)
    return {'systems': len(rows)}


def filter_name(name):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', name)


def _em_mdp(path, config):
    path.write_text('\n'.join([
        'integrator = steep', f"nsteps = {int(config['em_max_steps'])}",
        f"emtol = {float(config['em_tol'])}",
        'emstep = 0.01', 'cutoff-scheme = Verlet', 'pbc = xyz',
        'coulombtype = PME', 'rcoulomb = 1.0',
        'vdwtype = cutoff', 'rvdw = 1.0', 'nstlist = 10',
    ]) + '\n')


EM_FIELDS = ('potential_energy', 'maximum_force', 'em_steps', 'em_status',
             'em_converged', 'status', 'error')


def run_nmnat_md_em(config_path, output):
    started_at = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    config = load_yaml(config_path)
    output = Path(output)
    shutil.copyfile(config_path, output / 'em_config_used.yaml')
    write_json(output / 'em_identity.json', {
        'started_at': started_at,
        'code': str(Path(__file__).resolve()),
        'code_sha256': sha256_file(Path(__file__)),
        'config': str(Path(config_path).resolve()),
        'config_sha256': sha256_file(config_path),
        'systems_sha256': sha256_file(output / 'systems.csv'),
    })
    rows = list(csv.DictReader(open(output / 'systems.csv')))
    fields = _system_fields(rows)
    for field in EM_FIELDS:
        if field not in fields:
            fields.append(field)
    failed = 0
    for row in rows:
        model_dir = Path(row['directory'])
        existing = _existing_em_result(model_dir, config)
        if existing is not None:
            row.pop('error', None)
            row.update(existing)
            _write_systems(output, rows, fields)
            continue
        try:
            _em_mdp(model_dir / 'em.mdp', config)
            _run([config['grompp'], 'grompp', '-f', 'em.mdp', '-c', 'com_gmx.gro',
                  '-p', 'com_gmx.top', '-o', 'em.tpr'], model_dir,
                 model_dir / 'grompp_em.log')
            _run([config['mdrun'], 'mdrun', '-s', 'em.tpr', '-deffnm', 'em',
                  '-nb', 'cpu', '-nt', str(config['em_threads'])],
                 model_dir, model_dir / 'mdrun_em.log')
            if not (model_dir / 'em.gro').exists():
                raise RuntimeError('EM did not produce em.gro')
            row.pop('error', None)
            row.update(_em_result((model_dir / 'em.log').read_text(),
                                  config, completed=True))
        except Exception as error:
            row['status'] = 'FAILED'
            row['em_status'] = 'FAILED'
            row['em_converged'] = 'False'
            row['error'] = f'{type(error).__name__}: {error}'
            failed += 1
        _write_systems(output, rows, fields)
    return {'em': len(rows), 'failed': failed}


def _existing_em_result(model_dir, config):
    if not (model_dir / 'em.gro').exists() or not (model_dir / 'em.log').exists():
        return None
    result = _em_result((model_dir / 'em.log').read_text(), config, completed=True)
    return None if result['em_status'] == 'FAILED' else result


def _em_result(log, config, completed):
    energy = re.search(r'Potential Energy\s*=\s*([-\d.eE+]+)', log)
    force = re.search(r'Maximum force\s*=\s*([-\d.eE+]+)', log)
    steps = re.search(r'in (\d+) steps', log)
    finished = completed and 'Finished mdrun' in log
    result = {
        'potential_energy': energy.group(1) if energy else '',
        'maximum_force': force.group(1) if force else '',
        'em_steps': steps.group(1) if steps else '',
        'em_status': 'FAILED',
        'em_converged': 'False',
    }
    if not finished or not (energy and force and steps):
        result['status'] = 'FAILED'
        return result
    fmax = float(force.group(1))
    count = int(steps.group(1))
    max_steps = int(config['em_max_steps'])
    if fmax <= float(config['em_tol']):
        status = 'FMAX_CONVERGED'
    elif 'maximum number of steps' in log or count >= max_steps:
        status = 'MAX_STEPS'
    else:
        status = 'PRECISION_STALLED'
    result.update({
        'em_status': status,
        'em_converged': str(status == 'FMAX_CONVERGED'),
        'status': status,
    })
    return result


def _system_fields(rows):
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    return fields


def _write_systems(output, rows, fields=None):
    with (output / 'systems.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or _system_fields(rows),
                                lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def _write_initial_manifest(output, config):
    rows = list(csv.DictReader(open(output / 'systems.csv')))
    metrics = {row['model']: row for row in
               csv.DictReader(open(Path(config['joint_run']) / 'joint_metrics.csv'))}
    selected = []
    seen = {}
    for row in rows:
        name = row['model']
        metric = metrics.get(name)
        if metric is None:
            raise KeyError(f'No joint metrics for {name}')
        if row['has_mg'] == 'False':
            group = 'noMg'
        else:
            domain = metric.get('metal_protein_domain')
            if domain not in ('NMNAT', 'Nudix'):
                raise ValueError(f'Unexpected metal domain for {name}: {domain}')
            group = f'Mg_{domain}'
        key = (group, int(row['seed']))
        index = int(re.search(r'model_(\d+)', name).group(1))
        if key not in seen or index < seen[key][0]:
            seen[key] = (index, row)
    expected = {(group, seed) for group in ('noMg', 'Mg_NMNAT', 'Mg_Nudix')
                for seed in (20261026, 20261027, 20261028)}
    if set(seen) != expected:
        raise ValueError(f'Initial-state groups incomplete: {sorted(seen)}')
    for (group, seed), (_, row) in sorted(seen.items()):
        selected.append({**row, 'group': group})
    if len(selected) != 9:
        raise ValueError(f'Expected 9 initial states, got {len(selected)}')
    with (output / 'md_initial_states.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=_system_fields(selected),
                                lineterminator='\n')
        writer.writeheader()
        writer.writerows(selected)


def _add_position_restraints(model_dir):
    model_dir = Path(model_dir)
    top = model_dir / 'com_gmx.top'
    text = top.read_text()
    if 'posre.itp' not in text:
        lines = text.splitlines()
        atoms = lines.index('[ atoms ]')
        insert = next(index for index in range(atoms + 1, len(lines))
                      if lines[index].startswith('['))
        lines[insert:insert] = ['#ifdef POSRES', '#include "posre.itp"',
                                '#endif', '']
        top.write_text('\n'.join(lines) + '\n')
    posre = model_dir / 'posre.itp'
    if not posre.exists():
        import parmed as pmd
        structure = pmd.load_file(str(model_dir / 'com.prmtop'))
        rows = ['[ position_restraints ]', '; atom type fx fy fz']
        for index, atom in enumerate(structure.atoms):
            if atom.residue.name in ('ATP', 'NMN', 'MG', 'Na+', 'WAT'):
                continue
            if atom.element == 1:
                continue
            rows.append(f'{index + 1:6d} 1 1000 1000 1000')
        posre.write_text('\n'.join(rows) + '\n')
    return posre


def _stage_steps(stage, config):
    dt = float(config.get('md_dt_fs', 2.0))
    duration = {'nvt': float(config.get('md_nvt_ps', 100)),
                'npt': float(config.get('md_npt_ps', 100)),
                'production': float(config.get('md_production_ps', 1000))}[stage]
    return int(duration * 1000 / dt)


def _md_mdp(path, stage, config, seed=None, restraints=False):
    temperature = float(config.get('md_temperature', 298.15))
    dt = float(config.get('md_dt_fs', 2.0))
    production = stage == 'production'
    steps = _stage_steps(stage, config)
    nstxout = max(100, min(5000, steps // 10))
    lines = [
        'integrator = md', f'dt = {dt / 1000:.5f}',
        f'nsteps = {steps}',
        'comm-mode = Linear', 'comm-grps = System', 'nstcomm = 100',
        'nstxout = 0', 'nstvout = 0', 'nstfout = 0',
        f'nstxout-compressed = {nstxout}', 'compressed-x-grps = System',
        f"nstenergy = {5000 if production else 500}",
        f"nstlog = {5000 if production else 500}",
        'continuation = ' + ('yes' if stage != 'nvt' else 'no'),
        'constraint-algorithm = lincs', 'constraints = h-bonds',
        'lincs-iter = 1', 'lincs-order = 4',
        'cutoff-scheme = Verlet', 'nstlist = 10', 'pbc = xyz',
        'coulombtype = PME', 'rcoulomb = 1.0',
        'vdwtype = cutoff', 'rvdw = 1.0', 'DispCorr = EnerPres',
        'verlet-buffer-tolerance = 0.005', 'pme-order = 4',
        'tcoupl = V-rescale', 'tc-grps = System', 'tau-t = 0.1',
        f'ref-t = {temperature}', f'ld-seed = {int(seed)}',
    ]
    if restraints:
        lines.append('define = -DPOSRES')
    if stage == 'nvt':
        lines += ['pcoupl = no', 'gen-vel = yes',
                  f'gen-temp = {temperature}', f'gen-seed = {int(seed)}']
    else:
        lines += ['pcoupl = C-rescale', 'tau-p = 2.0', 'ref-p = 1.0',
                  'compressibility = 4.5e-5', 'refcoord-scaling = com',
                  'gen-vel = no']
    Path(path).write_text('\n'.join(lines) + '\n')


def _stage_result(log, stage, config):
    matches = re.findall(r'Step\s+Time\s*\n\s*(\d+)\s+([\d.]+)', log)
    expected = _stage_steps(stage, config)
    finished = 'Finished mdrun' in log
    step = int(matches[-1][0]) if matches else None
    time_ps = float(matches[-1][1]) if matches else None
    status = ('DONE' if finished and step is not None and step >= expected
              else 'INCOMPLETE')
    return {'status': status, 'finished': str(finished),
            'step': step, 'time_ps': time_ps, 'expected_steps': expected}


def _stage_done(run_dir, stage, config):
    run_dir = Path(run_dir)
    log = run_dir / f'{stage}.log'
    if not (run_dir / f'{stage}.gro').exists():
        return False
    if not (run_dir / f'{stage}.cpt').exists():
        return False
    if stage == 'production' and not (run_dir / 'production.xtc').exists():
        return False
    return log.exists() and _stage_result(
        log.read_text(), stage, config)['status'] == 'DONE'


def _md_stage(model_dir, run_dir, stage, config, input_gro, restraints,
              use_gpu, previous_cpt=None):
    model_dir = Path(model_dir).resolve()
    run_dir = Path(run_dir).resolve()
    input_gro = Path(input_gro).resolve()
    tpr = run_dir / f'{stage}.tpr'
    command = [config['grompp'], 'grompp', '-f', str(run_dir / f'{stage}.mdp'),
               '-c', str(input_gro), '-p', str(model_dir / 'com_gmx.top'),
               '-o', str(tpr)]
    if restraints:
        command += ['-r', str(model_dir / 'em.gro')]
    if previous_cpt is not None:
        command += ['-t', str(Path(previous_cpt).resolve())]
    _run(command, model_dir, run_dir / f'grompp_{stage}.log')
    command = [config['mdrun'], 'mdrun', '-s', str(tpr), '-deffnm', stage,
               '-ntmpi', '1', '-ntomp', str(config.get('md_threads', 8))]
    command += (['-nb', 'gpu', '-pme', 'gpu', '-bonded', 'gpu']
                if use_gpu else ['-nb', 'cpu'])
    _run(command, run_dir, run_dir / f'mdrun_{stage}.log')


def _gpu_state(config):
    try:
        apps = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
             '--format=csv,noheader,nounits'], text=True).strip()
        gpu = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'], text=True).strip()
    except Exception:
        return None
    rows = []
    for line in gpu.splitlines():
        utilization, used, total = (float(item) for item in line.split(','))
        rows.append({'utilization': utilization, 'used_mb': used,
                     'total_mb': total})
    return {'apps': [line for line in apps.splitlines() if line.strip()],
            'gpus': rows}


def _gpu_idle(config):
    state = _gpu_state(config)
    if state is None or state['apps'] or not state['gpus']:
        return False
    max_fraction = float(config.get('md_gpu_max_mem_fraction', 0.1))
    return all(row['used_mb'] / row['total_mb'] <= max_fraction
               for row in state['gpus'])


def _wait_for_idle_gpu(config, output):
    poll = float(config.get('md_gpu_poll_seconds', 60))
    deadline = time.time() + float(config.get('md_gpu_wait_hours', 12)) * 3600
    log = Path(output) / 'gpu_wait.log'
    while time.time() < deadline:
        state = _gpu_state(config)
        with log.open('a') as handle:
            handle.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} '
                         f'{json.dumps(state, ensure_ascii=False)}\n')
        if _gpu_idle(config):
            return True
        time.sleep(poll)
    return False


def _check_trajectory(config, run_dir):
    run_dir = Path(run_dir)
    _run([config['grompp'], 'check', '-f', 'production.xtc'], run_dir,
         run_dir / 'check_production.log')
    return 'OK'


def _write_md_runs(output, rows):
    with (Path(output) / 'md_runs.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=_system_fields(rows),
                                lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def run_nmnat_md_pilot(config_path, output, wait_gpu=False):
    started_at = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    config = load_yaml(config_path)
    output = Path(output).resolve()
    shutil.copyfile(config_path, output / 'md_config_used.yaml')
    states = list(csv.DictReader(open(output / 'md_initial_states.csv')))
    if len(states) != 9:
        raise ValueError(f'Expected 9 initial states, got {len(states)}')
    systems = {row['model']: row for row in
               csv.DictReader(open(output / 'systems.csv'))}
    velocity_seeds = [int(seed) for seed in
                      config.get('md_velocity_seeds', [20261024, 20261025])]
    runs = len(states) * len(velocity_seeds)
    production_ns = runs * float(config.get('md_production_ps', 1000)) / 1000
    equilibration_ns = runs * (
        float(config.get('md_nvt_ps', 100)) +
        float(config.get('md_npt_ps', 100))) / 1000
    write_json(output / 'md_identity.json', {
        'started_at': started_at,
        'code': str(Path(__file__).resolve()),
        'code_sha256': sha256_file(Path(__file__)),
        'config': str(Path(config_path).resolve()),
        'config_sha256': sha256_file(config_path),
        'manifest_sha256': sha256_file(output / 'md_initial_states.csv'),
        'velocity_seeds': velocity_seeds,
        'planned': {'runs': runs, 'production_ns': production_ns,
                    'equilibration_ns': equilibration_ns,
                    'total_ns': production_ns + equilibration_ns},
        'settings': {key: config.get(key) for key in (
            'md_temperature', 'md_dt_fs', 'md_nvt_ps', 'md_npt_ps',
            'md_production_ps', 'md_threads', 'md_use_gpu',
            'md_require_gpu')},
        'stages': {'nvt': 'protein heavy-atom position restraints',
                   'npt': 'protein heavy-atom position restraints',
                   'production': 'no restraints'},
    })
    use_gpu = bool(config.get('md_use_gpu', True))
    resume = (output / 'md_runs.csv').exists()
    if use_gpu:
        idle = True if resume else (
            _wait_for_idle_gpu(config, output) if wait_gpu else _gpu_idle(config))
        if not idle:
            if config.get('md_require_gpu', True):
                return {'runs': 0, 'status': 'GPU_BUSY'}
            use_gpu = False
    rows = []
    for state in states:
        model_dir = Path(state['directory']).resolve()
        system = systems.get(state['model'], {})
        if system.get('status') not in ('FMAX_CONVERGED', 'EM_CONVERGED'):
            for seed in velocity_seeds:
                rows.append({
                    'group': state['group'], 'model': state['model'],
                    'em_status': system.get('status', 'MISSING'),
                    'velocity_seed': seed, 'directory': '',
                    'status': 'NOT_RUN_EM',
                    'error': f"EM status {system.get('status', 'MISSING')}",
                })
            _write_md_runs(output, rows)
            continue
        _add_position_restraints(model_dir)
        for seed in velocity_seeds:
            run_dir = (output / 'runs' / filter_name(state['group'])
                       / filter_name(Path(state['model']).stem)
                       / f'velocity_{seed}')
            run_dir.mkdir(parents=True, exist_ok=True)
            row = {'group': state['group'], 'model': state['model'],
                   'em_status': system.get('status', ''),
                   'velocity_seed': seed, 'directory': str(run_dir),
                   'status': 'RUNNING'}
            try:
                for stage, source, restraints, previous in (
                        ('nvt', model_dir / 'em.gro', True, None),
                        ('npt', run_dir / 'nvt.gro', True, run_dir / 'nvt.cpt'),
                        ('production', run_dir / 'npt.gro', False,
                         run_dir / 'npt.cpt')):
                    log = run_dir / f'{stage}.log'
                    if _stage_done(run_dir, stage, config):
                        result = _stage_result(log.read_text(), stage, config)
                        row[stage] = 'EXISTS'
                    else:
                        _md_mdp(run_dir / f'{stage}.mdp', stage, config, seed,
                                restraints)
                        _md_stage(model_dir, run_dir, stage, config, source,
                                  restraints, use_gpu, previous)
                        result = _stage_result(log.read_text(), stage, config)
                        row[stage] = result['status']
                    row[f'{stage}_steps'] = result['step']
                    row[f'{stage}_time_ps'] = result['time_ps']
                    if result['status'] != 'DONE':
                        raise RuntimeError(
                            f'{stage} incomplete: step {result["step"]} '
                            f'of {result["expected_steps"]}')
                    if stage == 'production':
                        row['trajectory_check'] = _check_trajectory(config, run_dir)
                row['status'] = 'DONE'
                row['production_ns'] = float(row.get('production_time_ps') or 0) / 1000
            except Exception as error:
                row['status'] = 'FAILED'
                row['error'] = f'{type(error).__name__}: {error}'
            rows.append(row)
            _write_md_runs(output, rows)
    return {'runs': len(rows),
            'done': sum(1 for row in rows if row['status'] == 'DONE')}


def write_atom_mapping(config_path, output):
    import pickle
    from rdkit import Chem

    output = Path(output).resolve()
    config = load_yaml(config_path)
    build = json.load(open(output / 'build.json'))
    rows = list(csv.DictReader(open(output / 'systems.csv')))
    atomic_number = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 15: 'P'}
    result = {
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'code': str(Path(__file__).resolve()),
        'code_sha256': sha256_file(Path(__file__)),
        'config': str(Path(config_path).resolve()),
        'config_sha256': sha256_file(config_path),
        'note': 'Use topology_index, not GROMACS atom names, for geometry.',
        'ligands': {}, 'models': {},
    }
    for ligand in ('ATP', 'NMN'):
        mol2 = Path(build['parameters'][ligand]['mol2'])
        frcmod = Path(build['parameters'][ligand]['frcmod'])
        ccd_path = Path(config['ccd_cache']) / f'{ligand}.pkl'
        ccd = Chem.RemoveHs(pickle.load(open(ccd_path, 'rb')))
        mapping = _ligand_graph_mapping(ccd, ligand)
        result['ligands'][ligand] = {
            'smiles': SMILES[ligand],
            'canonical_smiles': Chem.MolToSmiles(
                Chem.MolFromSmiles(SMILES[ligand])),
            'charge': CHARGE[ligand],
            'atom_count': len(_mol2_elements(mol2)),
            'ccd': str(ccd_path),
            'ccd_sha256': sha256_file(ccd_path),
            'mol2': str(mol2),
            'mol2_sha256': sha256_file(mol2),
            'frcmod': str(frcmod),
            'frcmod_sha256': sha256_file(frcmod),
            'parameter_index_to_ccd': {
                str(index): name for index, name in sorted(mapping.items())},
        }
    for row in rows:
        model_dir = Path(row['directory'])
        structure = __import__('parmed').load_file(
            str(model_dir / 'com.prmtop'))
        model = {'cif_sha256': row['cif_sha256'], 'ligands': {}}
        for ligand in ('ATP', 'NMN'):
            residue = next(item for item in structure.residues
                           if item.name == ligand)
            atoms = list(residue.atoms)
            expected = _mol2_elements(
                build['parameters'][ligand]['mol2'])
            observed = [atomic_number.get(atom.element, '?')
                        for atom in atoms]
            if observed != expected:
                raise ValueError(f'{ligand} topology order mismatch: '
                                 f'{Path(row["model"]).name}')
            mapping = result['ligands'][ligand]['parameter_index_to_ccd']
            model['ligands'][ligand] = {
                'residue_index': residue.idx + 1,
                'atoms': [{
                    'topology_index': atom.idx + 1,
                    'parameter_index': index,
                    'acpype_atom_name': atom.name,
                    'ccd_name': mapping.get(str(index), 'H'),
                    'element': element,
                    'charge': round(float(atom.charge), 8),
                } for index, (atom, element) in enumerate(zip(atoms, observed))],
            }
        result['models'][Path(row['model']).name] = model
    write_json(output / 'atom_mapping.json', result)
    return {'models': len(result['models']),
            'ligands': len(result['ligands'])}
