"""Build and run the AF3 cross-model check for the WT ATP/NMN joint pose."""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .data import load_yaml, sha256_file, write_json
from .nmnat_binding import (_cross_from_atoms, _domain, _ligand_atoms,
                            _ligand_domain, _metal_atoms, _min_pair,
                            _protein_atoms, _reaction_from_atoms, _structure)
from .nmnat_mapping import _kabsch

AF3_LIGANDS = ('ATP', 'NMN', 'MG')
AA3 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'MSE': 'M',
}


def _chain_sequence(cif, chain_id):
    from Bio.PDB import MMCIFParser

    structure = MMCIFParser(QUIET=True).get_structure(str(cif), str(cif))
    chain = structure[0][chain_id]
    return ''.join(AA3.get(res.resname, 'X') for res in chain
                   if res.id[0] == ' ')


def _canonical_sequence(cif):
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict

    header = MMCIF2Dict(str(cif))
    text = header['_entity_poly.pdbx_seq_one_letter_code_can']
    if isinstance(text, list):
        text = text[0]
    return ''.join(text.split())


def _ligand_entries(codes, first_chain='B'):
    entries = []
    start = ord(first_chain) - 64
    for index, code in enumerate(codes):
        entries.append({'ligand': {
            'id': [chr(64 + start + index)], 'ccdCodes': [code]}})
    return entries


def _protein_entry(sequence, msa=None, msa_text=None):
    protein = {'id': ['A'], 'sequence': sequence, 'templates': []}
    if msa is not None:
        protein['unpairedMsaPath'] = str(Path(msa).resolve())
        protein['pairedMsa'] = ''
    elif msa_text is not None:
        protein['unpairedMsa'] = msa_text
        protein['pairedMsa'] = ''
    return {'protein': protein}


def _fold_input(name, sequence, codes, msa=None, msa_text=None):
    return {
        'name': name,
        'modelSeeds': [1],
        'sequences': [_protein_entry(sequence, msa, msa_text)]
                      + _ligand_entries(codes),
        'dialect': 'alphafold3',
        'version': 1,
    }


def build_af3_inputs(config_path, output):
    config = load_yaml(config_path)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {output}')
    inputs = output / 'inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    target = load_yaml(config['target'])
    sequence = target['sequence']
    wt_msa = Path(config['wt_msa']).resolve()
    rows = []
    for name, codes in config['conditions'].items():
        payload = _fold_input(name, sequence, codes, wt_msa)
        write_json(inputs / f'{name}.json', payload)
        rows.append({'name': name, 'kind': 'wt', 'sequence': sequence,
                     'sequence_length': len(sequence), 'msa': str(wt_msa),
                     'msa_rows': sum(1 for line in open(wt_msa)
                                     if line.startswith('>')),
                     'ligands': ';'.join(codes), 'seeds': config['af3']['num_seeds'],
                     'samples': config['af3'].get('num_diffusion_samples', 5),
                     'run_data_pipeline': 'false'})
    reference = config['reference']
    ref_cif = Path(reference['cif']).resolve()
    ref_sequence = _chain_sequence(ref_cif, reference['chain'])
    ref_canonical = _canonical_sequence(ref_cif)
    if not ref_canonical.startswith(ref_sequence):
        raise ValueError('Observed reference chain is not a prefix of canonical')
    payload = _fold_input(reference['name'], ref_sequence,
                          reference['ligands'],
                          msa_text=f'>query\n{ref_sequence}\n')
    write_json(inputs / f"{reference['name']}.json", payload)
    rows.append({'name': reference['name'], 'kind': 'reference',
                 'sequence': ref_sequence, 'sequence_length': len(ref_sequence),
                 'msa': 'query-only', 'msa_rows': 1,
                 'ligands': ';'.join(reference['ligands']),
                 'seeds': config['af3']['num_seeds'],
                 'samples': config['af3'].get('num_diffusion_samples', 5),
                 'run_data_pipeline': 'false'})
    with (inputs / 'manifest.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]),
                                lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / 'build.json', {
        'code': str(Path(__file__).resolve()),
        'code_sha256': sha256_file(Path(__file__)),
        'config': str(Path(config_path).resolve()),
        'config_sha256': sha256_file(config_path),
        'target': str(Path(config['target']).resolve()),
        'target_sha256': sha256_file(config['target']),
        'wt_msa': str(wt_msa), 'wt_msa_sha256': sha256_file(wt_msa),
        'reference_cif': str(ref_cif),
        'reference_cif_sha256': sha256_file(ref_cif),
        'reference_sequence_length': len(ref_sequence),
        'reference_canonical_length': len(ref_canonical),
        'reference_omitted_terminal_aa': ref_canonical[len(ref_sequence):],
        'reference_boundary': ('single chain A; observed residues 1-'
                               f'{len(ref_sequence)} of canonical '
                               f'{len(ref_canonical)}; native hexamer assembly '
                               'ignored'),
        'af3': config['af3'],
    })
    return {'jobs': len(rows), 'output': str(output)}


def _gpu_total_gb():
    try:
        text = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            text=True).strip().splitlines()
        return min(float(value) for value in text) / 1024
    except Exception:
        return 0.0


def _gpu_free_gb():
    try:
        text = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
            text=True).strip().splitlines()
        return min(float(value) for value in text) / 1024
    except Exception:
        return 0.0


def _gpu_ready(config):
    required = max(float(config['af3']['min_free_gb']),
                   float(config['af3']['memory_fraction']) * _gpu_total_gb())
    return _gpu_free_gb() >= required


def _gpu_compute_apps():
    try:
        text = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid,used_memory',
             '--format=csv,noheader,nounits'], text=True).strip()
    except Exception:
        return []
    return [line for line in text.splitlines() if line.strip()]


def _wait_for_free_gpu(config, output):
    required = max(float(config['af3']['min_free_gb']),
                   float(config['af3']['memory_fraction']) * _gpu_total_gb())
    poll = 60
    log = Path(output) / 'gpu_wait.log'
    while True:
        free = _gpu_free_gb()
        apps = _gpu_compute_apps()
        with log.open('a') as handle:
            handle.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} '
                         f'free_gb={free:.1f} required_gb={required:.1f} '
                         f'apps={apps}\n')
        if free >= required:
            return True
        time.sleep(poll)


def _run(command, cwd, log, env):
    with open(log, 'w') as handle:
        subprocess.run(command, cwd=cwd, env=env, check=True,
                       stdout=handle, stderr=subprocess.STDOUT)


def _job_models(job_dir):
    return sorted(Path(job_dir).glob('**/seed-*_sample-*/*_model.cif'))


def run_af3(config_path, output, wait_gpu=False):
    config = load_yaml(config_path)
    output = Path(output).resolve()
    manifest = list(csv.DictReader(open(output / 'inputs' / 'manifest.csv')))
    if not manifest:
        raise ValueError('No AF3 inputs found')
    if not _gpu_ready(config):
        if not wait_gpu:
            return {'runs': 0, 'status': 'GPU_BUSY'}
        _wait_for_free_gpu(config, output)
    af3 = config['af3']
    logs = output / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in manifest:
        job_dir = output / 'runs' / item['name']
        job_dir.mkdir(parents=True, exist_ok=True)
        expected = int(item['seeds']) * int(item['samples'])
        models = _job_models(job_dir)
        if len(models) >= expected:
            rows.append({**item, 'status': 'EXISTS', 'expected': expected,
                         'models': len(models)})
            continue
        if models:
            raise ValueError(
                f"Partial AF3 output in {job_dir}: {len(models)}/{expected}")
        command = [af3['python'], af3['script'],
                   '--json_path', str(output / 'inputs' / f"{item['name']}.json"),
                   '--output_dir', str(job_dir),
                   '--model_dir', af3['model_dir'],
                   '--db_dir', af3['db_dir'],
                   '--buckets', str(af3['buckets']),
                   '--num_recycles', str(af3['num_recycles']),
                   '--num_diffusion_samples', str(item['samples']),
                   '--jax_compilation_cache_dir', af3['jax_cache'],
                   '--run_data_pipeline', item['run_data_pipeline'],
                   '--run_inference', 'true']
        if int(item['seeds']) > 1:
            command += ['--num_seeds', str(item['seeds'])]
        env = dict(os.environ)
        env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        env['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(af3['memory_fraction'])
        row = {**item, 'status': 'RUNNING', 'expected': expected}
        try:
            _run(command, output, logs / f"{item['name']}.log", env)
            found = len(_job_models(job_dir))
            row['models'] = found
            row['status'] = 'DONE' if found >= expected else 'INCOMPLETE'
        except Exception as error:
            row['status'] = 'FAILED'
            row['error'] = f'{type(error).__name__}: {error}'
        rows.append(row)
    with (output / 'runs.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(
            {key for row in rows for key in row}), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    return {'runs': len(rows),
            'done': sum(row['status'] in ('DONE', 'EXISTS') for row in rows)}


def _af3_confidence(path):
    name = path.name
    for suffix in ('_model.cif.gz', '_model.cif'):
        if name.endswith(suffix):
            prefix = name[:-len(suffix)]
            break
    else:
        return {}
    summary = path.parent / f'{prefix}_summary_confidences.json'
    if not summary.exists():
        return {}
    values = json.load(open(summary))
    return {key: values.get(key) for key in
            ('ptm', 'iptm', 'ranking_score', 'has_clash')}


def _run_model_rows(output):
    rows = []
    runs = Path(output) / 'runs'
    if not runs.is_dir():
        return rows
    for job_dir in sorted(runs.iterdir()):
        if not job_dir.is_dir():
            continue
        for path in sorted(job_dir.glob('**/seed-*_sample-*/*_model.cif')):
            sample_dir = path.parent.name
            seed, sample = sample_dir.replace('seed-', '').split('_sample-')
            rows.append({'condition': job_dir.name,
                         'seed': int(seed), 'sample': int(sample),
                         'path': path})
    return rows


def _archive_model_rows(output):
    rows = []
    archive = Path(output) / 'cifs'
    if not archive.is_dir():
        return rows
    for condition_dir in sorted(archive.iterdir()):
        if not condition_dir.is_dir():
            continue
        for path in sorted(condition_dir.glob('*_seed-*_sample-*_model.cif.gz')):
            match = re.search(r'seed-(\d+)_sample-(\d+)_model\.cif\.gz$',
                              path.name)
            if match is None:
                continue
            rows.append({'condition': condition_dir.name,
                         'seed': int(match.group(1)),
                         'sample': int(match.group(2)),
                         'path': path})
    return rows


def _af3_models(output):
    rows = _run_model_rows(output)
    seen = {(row['condition'], row['seed'], row['sample']) for row in rows}
    rows.extend(row for row in _archive_model_rows(output)
                if (row['condition'], row['seed'], row['sample']) not in seen)
    return rows


def _phosphate_oxygens(structure, ligand):
    _, atoms = _ligand_atoms(structure, ligand)
    lookup = {name: coord for name, element, coord in atoms}
    if ligand == 'ATP':
        names = [f'O{index}{letter}' for letter in 'ABG' for index in (1, 2, 3)]
    else:
        names = ['O1P', 'O2P', 'O3P']
    return [(name, lookup[name]) for name in names if name in lookup]


def _has_ligand(structure, name):
    return any(residue.id[0].startswith('H_') and residue.resname == name
               for residue in structure.get_residues())


def _model_metrics(item):
    structure = _structure(item['path'])
    protein = _protein_atoms(structure)
    _, atp_atoms = _ligand_atoms(structure, 'ATP')
    has_nmn = _has_ligand(structure, 'NMN')
    nmn_atoms = _ligand_atoms(structure, 'NMN')[1] if has_nmn else []
    atp_domain, atp_resnum = _ligand_domain(structure, 'ATP')
    nmn_domain, nmn_resnum = (
        _ligand_domain(structure, 'NMN') if has_nmn else (None, None))
    reaction, candidates = _reaction_from_atoms(atp_atoms, nmn_atoms)
    cross = ({'cross_min_A': None, 'cross_nearest': '', 'cross_severe': None}
             if not has_nmn else _cross_from_atoms(atp_atoms, nmn_atoms, 2.0)[0])
    phosphate = (_min_pair(_phosphate_oxygens(structure, 'ATP'),
                           _phosphate_oxygens(structure, 'NMN'))
                 if has_nmn else None)
    contacts = sum(1 for _, _, coord in atp_atoms + nmn_atoms
                   for _, atom in protein
                   if float(np.linalg.norm(coord - atom.coord)) <= 4.0)
    severe = sum(1 for _, _, coord in atp_atoms + nmn_atoms
                 for _, atom in protein
                 if float(np.linalg.norm(coord - atom.coord)) < 2.0)
    row = {
        **item, 'file': item['path'].name,
        'protein_residues': sum(1 for residue in structure['A']
                                if residue.id[0] == ' '),
        'atp_domain': atp_domain, 'atp_domain_resnum': atp_resnum,
        'nmn_domain': nmn_domain, 'nmn_domain_resnum': nmn_resnum,
        'both_in_nmnat': ((atp_domain == 'NMNAT' and nmn_domain == 'NMNAT')
                          if has_nmn else None),
        'cross_min_A': cross['cross_min_A'],
        'cross_nearest': cross['cross_nearest'],
        'cross_severe': cross['cross_severe'],
        'phosphate_O_O_min_A': round(phosphate[0], 3) if phosphate else None,
        'protein_contacts_4A': contacts,
        'protein_severe': severe,
        **reaction,
    }
    for candidate in candidates:
        atom = candidate['candidate_atom']
        row[f'PA_{atom}_A'] = candidate['PA_Onuc_A']
        row[f'O3A_PA_{atom}_deg'] = candidate['O3A_PA_Onuc_deg']
        row[f'P_{atom}_A'] = candidate['P_Onuc_A']
    row.update(_af3_confidence(item['path']))
    row['atp_chain_intact'] = bool(
        (row.get('atp_PA_O3A_A') or 0) and (row.get('atp_O3A_PB_A') or 0)
        and 1.4 <= row['atp_PA_O3A_A'] <= 1.9
        and 1.4 <= row['atp_O3A_PB_A'] <= 1.9
        and 2.4 <= (row.get('atp_PA_PB_A') or 0) <= 3.4)

    metals = _metal_atoms(structure)
    row['metal_count'] = len(metals)
    for key in ('metal_domain', 'metal_ATP_O_min_A', 'metal_NMN_O_min_A',
                'metal_protein_min_A', 'metal_nearby_ATP_O',
                'metal_nearby_NMN_O', 'metal_nearby_protein_O',
                'metal_nearby_protein_N', 'metal_shared_near_neighbor',
                'metal_O2B_A', 'metal_O3A_A', 'metal_O2G_A', 'metal_O3B_A'):
        row[key] = None
    if metals:
        metal = metals[0]
        nearest = min((float(np.linalg.norm(metal - atom.coord)),
                       int(residue.id[1])) for residue, atom in protein)
        row['metal_domain'] = _domain(nearest[1])
        atp_o = [(name, coord) for name, element, coord in atp_atoms
                 if element == 'O']
        nmn_o = [(name, coord) for name, element, coord in nmn_atoms
                 if element == 'O']
        atp_min = min((float(np.linalg.norm(metal - coord)), name)
                      for name, coord in atp_o)
        nmn_min = (min((float(np.linalg.norm(metal - coord)), name)
                       for name, coord in nmn_o) if nmn_o else None)
        protein_min = min(float(np.linalg.norm(metal - atom.coord))
                          for _, atom in protein)
        row['metal_ATP_O_min_A'] = round(atp_min[0], 3)
        row['metal_NMN_O_min_A'] = round(nmn_min[0], 3) if nmn_min else None
        row['metal_protein_min_A'] = round(protein_min, 3)
        row['metal_nearby_ATP_O'] = sum(
            1 for _, coord in atp_o
            if float(np.linalg.norm(metal - coord)) <= 2.8)
        row['metal_nearby_NMN_O'] = (
            sum(1 for _, coord in nmn_o
                if float(np.linalg.norm(metal - coord)) <= 2.8)
            if nmn_o else None)
        row['metal_nearby_protein_O'] = sum(
            1 for _, atom in protein
            if atom.element == 'O' and float(np.linalg.norm(metal - atom.coord)) <= 2.8)
        row['metal_nearby_protein_N'] = sum(
            1 for _, atom in protein
            if atom.element == 'N' and float(np.linalg.norm(metal - atom.coord)) <= 2.8)
        for atom in ('O2B', 'O3A', 'O2G', 'O3B'):
            lookup = {name: coord for name, _, coord in atp_atoms}
            row[f'metal_{atom}_A'] = (round(float(np.linalg.norm(metal - lookup[atom])), 3)
                                      if atom in lookup else None)
        row['metal_shared_near_neighbor'] = (bool(atp_min[0] <= 2.8 and nmn_min[0] <= 2.8)
                                            if has_nmn else None)
    return row, structure, candidates


def _phosphate_groups(structure):
    _, atp_atoms = _ligand_atoms(structure, 'ATP')
    atp = {name: coord for name, _, coord in atp_atoms}
    groups = {}
    if all(name in atp for name in ('PA', 'O1A', 'O2A')):
        groups['atp_PA_terminal'] = (atp['PA'], [atp['O1A'], atp['O2A']])
    if all(name in atp for name in ('PG', 'O1G', 'O2G', 'O3G')):
        groups['atp_PG_terminal'] = (atp['PG'], [atp['O1G'], atp['O2G'], atp['O3G']])
    if _has_ligand(structure, 'NMN'):
        _, nmn_atoms = _ligand_atoms(structure, 'NMN')
        nmn = {name: coord for name, _, coord in nmn_atoms}
        if all(name in nmn for name in ('P', 'O1P', 'O2P', 'O3P')):
            groups['nmn_phosphate'] = (
                nmn['P'], [nmn['O1P'], nmn['O2P'], nmn['O3P']])
    return groups


def _ca_atoms(structure, reference=False):
    if reference:
        selection = [residue for residue in structure['A']
                     if residue.id[0] == ' ' and 'CA' in residue]
    else:
        selection = [residue for residue in structure['A']
                     if residue.id[0] == ' ' and 4 <= int(residue.id[1]) <= 181
                     and 'CA' in residue]
    return {int(residue.id[1]): residue['CA'].coord.astype(float)
            for residue in selection}


def _symmetry_rmsd(group_a, group_b):
    from itertools import permutations

    center_a, oxygen_a = group_a
    center_b, oxygen_b = group_b
    best = None
    for order in permutations(range(len(oxygen_b))):
        left = np.asarray([center_a] + oxygen_a)
        right = np.asarray([center_b] + [oxygen_b[index] for index in order])
        value = float(np.sqrt(np.mean(np.sum((left - right) ** 2, axis=1))))
        if best is None or value < best:
            best = value
    return best


def _scaffold_atoms(structure, ligand):
    _, atoms = _ligand_atoms(structure, ligand)
    return {name: coord for name, element, coord in atoms
            if element not in ('P', 'O', 'H')}


def _ligand_symmetry_rmsd(first, second, ligand, rotation, translation):
    from itertools import permutations, product

    _, atoms_a = _ligand_atoms(first, ligand)
    _, atoms_b = _ligand_atoms(second, ligand)
    lookup_a = {name: coord for name, _, coord in atoms_a}
    lookup_b = {name: coord for name, _, coord in atoms_b}
    shared = sorted(set(lookup_a) & set(lookup_b))
    if not shared:
        return None
    groups = ([('O1A', 'O2A'), ('O1B', 'O2B'),
               ('O1G', 'O2G', 'O3G')] if ligand == 'ATP'
              else [('O1P', 'O2P', 'O3P')])
    choices = []
    for group in groups:
        present = [name for name in group if name in lookup_b]
        if len(present) < 2:
            continue
        choices.append([list(order) for order in permutations(present)])
    best = None
    for order in product(*choices):
        mapping = {name: name for name in shared}
        for group, permutation in zip(groups, order):
            present = [name for name in group if name in lookup_b]
            for name in present:
                mapping[name] = permutation[present.index(name)]
        moved = np.asarray([lookup_a[name] for name in shared]) @ rotation.T \
            + translation
        target = np.asarray([lookup_b[mapping[name]] for name in shared])
        value = float(np.sqrt(np.mean(np.sum((moved - target) ** 2, axis=1))))
        if best is None or value < best:
            best = value
    return round(best, 3) if best is not None else None


def _pose_record(structure, label, reference=False):
    return {'label': label, 'structure': structure,
            'ca': _ca_atoms(structure, reference=reference),
            'groups': _phosphate_groups(structure),
            'atp_scaffold': _scaffold_atoms(structure, 'ATP'),
            'nmn_scaffold': (_scaffold_atoms(structure, 'NMN')
                             if _has_ligand(structure, 'NMN') else {})}


def _pose_compare(first, second):
    common = sorted(set(first['ca']) & set(second['ca']))
    if len(common) < 20:
        return None
    source = np.asarray([first['ca'][key] for key in common])
    target = np.asarray([second['ca'][key] for key in common])
    rmsd, rotation, translation = _kabsch(source, target)
    output = {'protein_CA_RMSD_A': round(rmsd, 3)}
    for name, ligand in (('atp_scaffold', 'ATP'), ('nmn_scaffold', 'NMN')):
        shared = sorted(set(first[name]) & set(second[name]))
        if len(shared) < 5:
            output[f'{name.replace("_scaffold", "_CN_scaffold")}_RMSD_A'] = None
        else:
            moved = np.asarray([first[name][key] for key in shared]) @ rotation.T \
                + translation
            target_atoms = np.asarray([second[name][key] for key in shared])
            output[f'{name.replace("_scaffold", "_CN_scaffold")}_RMSD_A'] = round(float(np.sqrt(np.mean(
                np.sum((moved - target_atoms) ** 2, axis=1)))), 3)
        if first.get('structure') is not None and second.get('structure') is not None:
            try:
                output[f'{ligand.lower()}_heavy_symmetry_RMSD_A'] = (
                    _ligand_symmetry_rmsd(
                        first['structure'], second['structure'], ligand,
                        rotation, translation))
            except ValueError:
                output[f'{ligand.lower()}_heavy_symmetry_RMSD_A'] = None
    for group, value in first['groups'].items():
        if group not in second['groups']:
            output[group] = None
            continue
        moved = np.asarray([value[0]] + value[1]) @ rotation.T + translation
        output[group] = round(_symmetry_rmsd(
            (moved[0], list(moved[1:])), second['groups'][group]), 3)
    return output


def _pose_pairs(records):
    pairs = []
    for condition, members in records.items():
        for index, first in enumerate(members):
            for second in members[index + 1:]:
                result = _pose_compare(first, second)
                if result is None:
                    continue
                pairs.append({'condition': condition,
                              'model_a': first['label'],
                              'model_b': second['label'], **result})
    return pairs


def _cross_pose_pairs(first_records, second_records, label):
    pairs = []
    same_set = first_records is second_records
    for index, first in enumerate(first_records):
        remaining = (second_records[index + 1:] if same_set else second_records)
        for second in remaining:
            result = _pose_compare(first, second)
            if result is None:
                continue
            pairs.append({'comparison': label,
                          'model_a': first['label'],
                          'model_b': second['label'], **result})
    return pairs


def _boltz_records(joint_run):
    records = {}
    coordinates = Path(joint_run) / 'coordinates'
    for path in sorted(coordinates.glob('*.cif.gz')):
        name = path.name[:-len('.cif.gz')]
        state, rest = name.split('_seed')
        seed, model = rest.split('_model_')
        structure = _structure(path)
        records.setdefault(state, []).append(
            _pose_record(structure, f'seed{seed}_model{model}'))
    return records


def _crystal_mg_distances(structure):
    atp = {name: coord for name, _, coord in _ligand_atoms(structure, 'ATP')[1]}
    metals = _metal_atoms(structure)
    if not metals:
        return {}
    metal = metals[0]
    output = {}
    for atom in ('O2B', 'O3A', 'O2G', 'O3B'):
        if atom in atp:
            output[f'mg_{atom}_A'] = round(
                float(np.linalg.norm(metal - atp[atom])), 3)
    return output


def _reference_reconstruction(cif, models):
    crystal = _structure(cif)
    crystal_record = _pose_record(crystal, 'crystal_1f9a_chainA',
                                  reference=True)
    crystal_ca = crystal_record['ca']
    crystal_atp = {name: coord for name, _, coord in
                   _ligand_atoms(crystal, 'ATP')[1]}
    crystal_mg = _crystal_mg_distances(crystal)
    rows = [{'model': 'crystal_1f9a_chainA', 'protein_CA_RMSD_A': 0.0,
             'atp_heavy_RMSD_A': 0.0, 'atp_heavy_symmetry_RMSD_A': 0.0,
             'mg_displacement_A': 0.0, 'atp_CN_scaffold_RMSD_A': 0.0, 'atp_PA_terminal': 0.0,
             'atp_PG_terminal': 0.0, **crystal_mg}]
    for item in models:
        structure = _structure(item['path'])
        record = _pose_record(structure, f"seed{item['seed']}_sample{item['sample']}",
                              reference=True)
        result = _pose_compare(record, crystal_record)
        if result is None:
            continue
        common = sorted(set(record['ca']) & set(crystal_ca))
        source = np.asarray([record['ca'][key] for key in common])
        target = np.asarray([crystal_ca[key] for key in common])
        _, rotation, translation = _kabsch(source, target)
        atp = _ligand_atoms(structure, 'ATP')[1]
        shared = sorted(set(name for name, _, _ in atp) & set(crystal_atp))
        moved = np.asarray([coord for name, _, coord in atp
                            if name in crystal_atp]) @ rotation.T + translation
        reference = np.asarray([crystal_atp[name] for name, _, _ in atp
                                if name in crystal_atp])
        atp_rmsd = (float(np.sqrt(np.mean(np.sum(
            (moved - reference) ** 2, axis=1)))) if shared else None)
        metal = _metal_atoms(structure)
        displacement = None
        if metal and crystal['A']:
            crystal_metals = [residue for residue in crystal['A']
                              if residue.id[0].startswith('H_')
                              and residue.resname == 'MG']
            if crystal_metals:
                crystal_mg_coord = next(
                    atom.coord.astype(float) for atom in crystal_metals[0]
                    if atom.element != 'H')
                moved_mg = metal[0] @ rotation.T + translation
                displacement = float(np.linalg.norm(moved_mg - crystal_mg_coord))
        rows.append({
            'model': record['label'],
            'protein_CA_RMSD_A': result['protein_CA_RMSD_A'],
            'atp_heavy_RMSD_A': round(atp_rmsd, 3) if atp_rmsd else None,
            'atp_heavy_symmetry_RMSD_A': result.get('atp_heavy_symmetry_RMSD_A'),
            'atp_CN_scaffold_RMSD_A': result.get('atp_CN_scaffold_RMSD_A'),
            'atp_PA_terminal': result.get('atp_PA_terminal'),
            'atp_PG_terminal': result.get('atp_PG_terminal'),
            'mg_displacement_A': round(displacement, 3)
            if displacement is not None else None,
            **_crystal_mg_distances(structure),
        })
    return rows


def _distribution(values, digits=3):
    clean = [value for value in values if value is not None]
    if not clean:
        return {'n': 0, 'median': None, 'min': None, 'max': None}
    return {'n': len(clean),
            'median': round(float(np.median(clean)), digits),
            'min': round(float(min(clean)), digits),
            'max': round(float(max(clean)), digits)}


def _boltz_comparison(config):
    joint = Path(config['joint_run']).resolve()
    metrics_path = joint / 'joint_metrics.csv'
    candidates_path = joint / 'joint_reaction_candidates.csv'
    output = []
    if not metrics_path.exists():
        return output
    import statistics
    groups = {}
    for row in csv.DictReader(open(metrics_path)):
        groups.setdefault(row['state'], []).append(row)
    candidate_groups = {}
    if candidates_path.exists():
        for row in csv.DictReader(open(candidates_path)):
            candidate_groups.setdefault(row['state'], []).append(row)
    for state, rows in groups.items():
        output.append({
            'state': state,
            'models': len(rows),
            'both_in_nmnat': sum(row['both_in_nmnat'] == 'True' for row in rows),
            'cross_min_median_A': round(statistics.median(
                float(row['cross_min_A']) for row in rows), 3),
            'lowest_PA_Onuc_median_A': round(statistics.median(
                float(row['lowest_distance_PA_Onuc_A']) for row in rows), 3),
            'lowest_angle_median_deg': round(statistics.median(
                float(row['lowest_distance_angle_deg']) for row in rows), 1),
            'metal_shared_near_neighbor_2p8_models': sum(
                1 for row in rows
                if row.get('metal_ATP_O_min_A') and row.get('metal_NMN_O_min_A')
                and float(row['metal_ATP_O_min_A']) <= 2.8
                and float(row['metal_NMN_O_min_A']) <= 2.8),
        })
    return output


def analyze_af3(config_path, output):
    config = load_yaml(config_path)
    output = Path(output).resolve()
    models = _af3_models(output)
    if not models:
        raise ValueError('No AF3 model coordinates found')
    rows = []
    records = {}
    reference_models = []
    candidates = []
    for model in models:
        row, structure, reaction = _model_metrics(model)
        rows.append(row)
        reference = row['condition'].startswith('ref_')
        label = f"seed{row['seed']}_sample{row['sample']}"
        records.setdefault(row['condition'], []).append(
            _pose_record(structure, label, reference=reference))
        if reference:
            reference_models.append(model)
        for item in reaction:
            candidates.append({'condition': row['condition'],
                               'seed': row['seed'], 'sample': row['sample'],
                               'model': row['file'], **item})
    pairs = _pose_pairs(records)
    boltz_records = _boltz_records(config['joint_run'])
    cross_pairs = []
    for af3_state, boltz_state in (('wt_atp_nmn', 'ATP_NMN'),
                                   ('wt_atp_nmn_mg', 'ATP_NMN_MG')):
        if af3_state in records and boltz_state in boltz_records:
            cross_pairs.extend(_cross_pose_pairs(
                records[af3_state], boltz_records[boltz_state],
                f'{af3_state}_vs_{boltz_state}'))
            cross_pairs.extend(_cross_pose_pairs(
                records[af3_state], records[af3_state],
                f'{af3_state}_within'))
        if boltz_state in boltz_records:
            cross_pairs.extend(_cross_pose_pairs(
                boltz_records[boltz_state], boltz_records[boltz_state],
                f'{boltz_state}_within'))
    with (output / 'models.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(
            {key for row in rows for key in row}), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    with (output / 'reaction_candidates.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(
            {key for row in candidates for key in row}), lineterminator='\n')
        writer.writeheader()
        writer.writerows(candidates)
    with (output / 'pose_pairs.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(
            {key for row in pairs for key in row}),
            lineterminator='\n') if pairs else None
        if writer:
            writer.writeheader()
            writer.writerows(pairs)
    with (output / 'cross_model_pose_pairs.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(
            {key for row in cross_pairs for key in row}),
            lineterminator='\n') if cross_pairs else None
        if writer:
            writer.writeheader()
            writer.writerows(cross_pairs)
    pose_summary = {}
    metrics = ('protein_CA_RMSD_A', 'atp_CN_scaffold_RMSD_A',
               'nmn_CN_scaffold_RMSD_A', 'atp_heavy_symmetry_RMSD_A',
               'nmn_heavy_symmetry_RMSD_A', 'atp_PA_terminal',
               'atp_PG_terminal', 'nmn_phosphate')
    for pair in pairs:
        for metric in metrics:
            if pair.get(metric) is not None:
                pose_summary.setdefault((pair['condition'], metric), []).append(
                    pair[metric])
    pose_rows = [{'comparison': key[0], 'metric': key[1],
                  **_distribution(values)}
                 for key, values in sorted(pose_summary.items())]
    with (output / 'pose_summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pose_rows[0]),
                                lineterminator='\n') if pose_rows else None
        if writer:
            writer.writeheader()
            writer.writerows(pose_rows)
    cross_summary = {}
    for pair in cross_pairs:
        for metric in metrics:
            if pair.get(metric) is not None:
                cross_summary.setdefault((pair['comparison'], metric), []).append(
                    pair[metric])
    cross_rows = [{'comparison': key[0], 'metric': key[1],
                   **_distribution(values)}
                  for key, values in sorted(cross_summary.items())]
    with (output / 'cross_model_pose_summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cross_rows[0]),
                                lineterminator='\n') if cross_rows else None
        if writer:
            writer.writeheader()
            writer.writerows(cross_rows)
    reference_rows = []
    if reference_models:
        reference_rows = _reference_reconstruction(
            config['reference']['cif'], reference_models)
        with (output / 'reference_reconstruction.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(
                {key for row in reference_rows for key in row}),
                lineterminator='\n')
            writer.writeheader()
            writer.writerows(reference_rows)

    summary = {'models': len(rows), 'conditions': {}}
    for condition in sorted({row['condition'] for row in rows}):
        subset = [row for row in rows if row['condition'] == condition]
        with_nmn = [row for row in subset if row['both_in_nmnat'] is not None]
        with_metal = [row for row in subset
                      if row['metal_shared_near_neighbor'] is not None]
        entry = {
            'models': len(subset),
            'nMN_models': len(with_nmn),
            'both_in_nmnat': (sum(bool(row['both_in_nmnat']) for row in with_nmn)
                              if with_nmn else None),
            'cross_min_A': _distribution([row['cross_min_A'] for row in subset]),
            'phosphate_O_O_min_A': _distribution(
                [row['phosphate_O_O_min_A'] for row in subset]),
            'protein_contacts_4A': _distribution(
                [row['protein_contacts_4A'] for row in subset], 0),
            'protein_severe': sum(row['protein_severe'] for row in subset),
            'atp_chain_intact': sum(bool(row['atp_chain_intact']) for row in subset),
            'ptm': _distribution([row.get('ptm') for row in subset]),
            'iptm': _distribution([row.get('iptm') for row in subset]),
            'has_clash': sum(bool(row.get('has_clash')) for row in subset),
        }
        for atom in ('O1P', 'O2P', 'O3P'):
            entry[f'PA_{atom}_A'] = _distribution([
                row.get(f'PA_{atom}_A') for row in subset
                if f'PA_{atom}_A' in row])
            entry[f'angle_{atom}_deg'] = _distribution([
                row.get(f'O3A_PA_{atom}_deg') for row in subset
                if f'O3A_PA_{atom}_deg' in row], 1)
        entry['lowest_PA_Onuc_A'] = _distribution(
            [row.get('lowest_distance_PA_Onuc_A') for row in subset])
        entry['lowest_angle_deg'] = _distribution(
            [row.get('lowest_distance_angle_deg') for row in subset], 1)
        entry['metal_count_models'] = sum(
            (row['metal_count'] or 0) > 0 for row in subset)
        entry['metal_nmnat'] = sum(
            row.get('metal_domain') == 'NMNAT' for row in subset)
        entry['metal_shared_near_neighbor'] = (
            sum(bool(row['metal_shared_near_neighbor']) for row in with_metal)
            if with_metal else None)
        entry['metal_shared_near_neighbor_models'] = len(with_metal)
        entry['metal_ATP_O_min_A'] = _distribution(
            [row.get('metal_ATP_O_min_A') for row in subset])
        entry['metal_NMN_O_min_A'] = _distribution(
            [row.get('metal_NMN_O_min_A') for row in subset])
        entry['metal_nearby_protein_O'] = _distribution(
            [row.get('metal_nearby_protein_O') for row in subset], 0)
        entry['metal_nearby_protein_N'] = _distribution(
            [row.get('metal_nearby_protein_N') for row in subset], 0)
        summary['conditions'][condition] = entry
    summary['boltz_reference'] = _boltz_comparison(config)
    summary['pose_summary'] = {
        f'{key[0]}:{key[1]}': _distribution(values)
        for key, values in sorted(pose_summary.items())}
    summary['cross_model_pose_summary'] = {
        f'{key[0]}:{key[1]}': _distribution(values)
        for key, values in sorted(cross_summary.items())}
    summary['reference_reconstruction'] = reference_rows
    confidence_models = sum(row.get('ptm') is not None for row in rows)
    summary['confidence'] = {
        'models_with_summary_confidences': confidence_models,
        'models': len(rows),
        'note': ('archived CIF-only reruns have no pTM/ipTM; those fields '
                 'are omitted rather than treated as zero'),
    }
    summary['boundary'] = (
        'AF3 models are independent diffusion samples, not experimental '
        'replicates or equilibrium populations. CIF output contains no water, '
        'so 2.8 A contacts are shared near-neighbour counts, not validated '
        'coordination shells or metal bridging. The 1F9A reference is a '
        'single-chain technical calibration against the observed crystal '
        'chain A and omits the unresolved terminal QTDK. AF3 and Boltz models '
        'are not paired by seed or sample, and no best pose is selected.')
    write_json(output / 'summary.json', summary)
    build_record = (json.load(open(output / 'build.json'))
                    if (output / 'build.json').exists() else {})
    script = Path(config['af3']['script'])
    identity = {
        'build': {
            'code_sha256': build_record.get('code_sha256'),
            'config_sha256': build_record.get('config_sha256'),
            'target_sha256': build_record.get('target_sha256'),
            'wt_msa_sha256': build_record.get('wt_msa_sha256'),
            'reference_cif_sha256': build_record.get('reference_cif_sha256'),
        },
        'prediction': {
            'weights': config['af3'].get('weights'),
            'weights_sha256': config['af3'].get('weights_sha256'),
            'script': str(script),
            'script_sha256_at_analysis': (
                sha256_file(script) if script.exists() else None),
            'seeds': sorted({model['seed'] for model in models}),
            'samples': sorted({model['sample'] for model in models}),
            'models': [{'condition': model['condition'],
                        'seed': model['seed'], 'sample': model['sample'],
                        'file': model['path'].name,
                        'sha256': sha256_file(model['path'])}
                       for model in models],
            'note': ('build, actual prediction inputs/coordinates, and final '
                     'analysis are recorded separately. A later analysis-code '
                     'change does not imply that the completed predictions '
                     'need to be rerun.'),
        },
        'analysis': {
            'code': str(Path(__file__).resolve()),
            'code_sha256': sha256_file(Path(__file__).resolve()),
            'config_sha256': sha256_file(Path(config_path).resolve()),
            'python': sys.executable,
        },
    }
    write_json(output / 'identity.json', identity)
    return {'models': len(rows), 'pairs': len(pairs),
            'cross_pairs': len(cross_pairs),
            'summary': str(output / 'summary.json')}
