"""Full-length candidate structure sampling with a common frozen family MSA."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from .data import Target, load_fasta, load_yaml, sha256_file, write_json


def _trajectory_metrics(trajectory, reference, sequence, wt, same_target=False):
    """Describe all candidate/WT frame pairs; coordinates are nm, output is Å."""
    import mdtraj as md

    ca = trajectory.topology.select('name CA')
    ref_ca = reference.topology.select('name CA')
    if len(ca) != len(sequence) or len(ref_ca) != len(wt):
        raise ValueError('Missing or additional C-alpha atoms in predicted full-length structure')
    if trajectory.topology.to_fasta() != [sequence] or reference.topology.to_fasta() != [wt]:
        raise ValueError('Coordinate sequence does not match the intended candidate/WT')
    ranges = {'full': np.arange(len(wt)), 'NMNAT': np.arange(3, 181),
              'Nudix': np.arange(203, 338)}
    values = {name + '_rmsd_A': [] for name in ranges}
    values['I145_neighborhood_rmsd_A'] = []
    for position in (14, 17, 135, 138):
        values[f'anchor_{position}_CA_displacement_A'] = []
    pair_rows = []
    for ref_frame in range(reference.n_frames):
        fits = {}
        for name, positions in ranges.items():
            aligned = trajectory[:].superpose(reference, frame=ref_frame,
                                               atom_indices=ca[positions],
                                               ref_atom_indices=ref_ca[positions],
                                               parallel=False)
            delta = aligned.xyz[:, ca] - reference.xyz[ref_frame, ref_ca]
            fits[name] = np.sum(delta * delta, axis=-1)
        for frame in range(trajectory.n_frames):
            if same_target and frame == ref_frame:
                continue
            row = {'frame': frame, 'WT_frame': ref_frame}
            for name, positions in ranges.items():
                row[name + '_rmsd_A'] = float(np.sqrt(fits[name][frame, positions].mean()) * 10)
            row['I145_neighborhood_rmsd_A'] = float(
                np.sqrt(fits['NMNAT'][frame, 140:149].mean()) * 10)
            for position in (14, 17, 135, 138):
                row[f'anchor_{position}_CA_displacement_A'] = float(
                    np.sqrt(fits['NMNAT'][frame, position - 1]) * 10)
            for key in values:
                values[key].append(row[key])
            pair_rows.append(row)
    summary = {key: {'median': float(np.median(value)), 'min': float(np.min(value)),
                     'max': float(np.max(value))} for key, value in values.items()}
    summary['frame_pairs'] = len(pair_rows)
    summary['frame_pair_note'] = 'Descriptive pairs, not independent observations or a confidence interval'
    centers_a = trajectory.xyz[:, ca[ranges['NMNAT']]].mean(axis=1)
    centers_b = trajectory.xyz[:, ca[ranges['Nudix']]].mean(axis=1)
    distances = np.linalg.norm(centers_a - centers_b, axis=-1) * 10
    summary['domain_centroid_distance_A'] = distances.tolist()
    secondary = md.compute_dssp(trajectory, simplified=True)
    for name in ('NMNAT', 'Nudix'):
        letters = secondary[:, ranges[name]]
        summary[name + '_helix_fraction'] = (letters == 'H').mean(axis=1).tolist()
        summary[name + '_strand_fraction'] = (letters == 'E').mean(axis=1).tolist()
    for label, calculator, atom_column in (
            ('phi', md.compute_phi, 2), ('psi', md.compute_psi, 1)):
        indices, angles = calculator(trajectory)
        column = next((i for i, atoms in enumerate(indices)
                       if trajectory.topology.atom(int(atoms[atom_column])).residue.index == 144), None)
        summary[f'I145_{label}_degrees'] = (
            np.rad2deg(angles[:, column]).tolist() if column is not None else None)
    return summary, pair_rows


def sample_candidate_structures(config_path, output):
    """Run installed BioEmu2 on WT and unique candidates; no MSA/model downloads."""
    import torch
    from Bio import SeqIO
    from bioemu2 import (EDMSampler, EmuSession, Protein, System, check_physicality,
                         load_a3m_msa, load_model, save_frames)
    from bioemu2.rng import seed_random_generators

    config = load_yaml(config_path)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Structure run already contains files: {output}')
    checkpoint = Path(config['checkpoint'])
    if sha256_file(checkpoint) != config['checkpoint_sha256']:
        raise ValueError('BioEmu2 checkpoint identity differs from configuration')
    wt = Target.load(config['target']).sequence
    candidates = load_fasta(config['candidates'])
    sequences = {'WT': wt}
    seen = {wt}
    for name, sequence in candidates.items():
        if len(sequence) != len(wt) or not set(sequence) <= set('ACDEFGHIKLMNPQRSTVWY'):
            raise ValueError(f'Invalid full-length candidate: {name}')
        if sequence not in seen:
            sequences[name] = sequence
            seen.add(sequence)
    cache = json.loads(Path(config['msa_cache']).read_text())
    if cache['request']['sequences'] != [wt] or len(cache['a3ms']) != 1:
        raise ValueError('Frozen MSA cache is not for the single-chain KVP40 WT')
    msa_records = list(SeqIO.parse(io.StringIO(cache['a3ms'][0]), 'fasta-pearson'))
    if str(msa_records[0].seq) != wt:
        raise ValueError('The frozen MSA query does not exactly match WT')
    homologs = msa_records[1:int(config['msa_rows'])]
    homolog_text = ''.join(f'>{record.id}\n{record.seq}\n' for record in homologs)
    output.mkdir(parents=True)
    started = time.monotonic()
    torch.set_num_threads(int(config['cpu_threads']))
    model = load_model(checkpoint_path=checkpoint, device=config['device'],
                       precision=config['precision'], matmul_precision='highest')
    trajectories, targets = {}, {}
    for name, sequence in sequences.items():
        destination = output / name
        destination.mkdir()
        msa_path = destination / 'input.a3m'
        msa_path.write_text(f'>{name}\n{sequence}\n' + homolog_text)
        system = System([Protein(sequence=sequence, asym_ids=['A'])])
        msa = load_a3m_msa(system, {'A': msa_path})
        seed_random_generators(int(config['seed']))
        session = EmuSession(system, model=model, mode='structure', msa=msa)
        target_start = time.monotonic()
        record = {'sequence': sequence, 'sequence_sha256': hashlib.sha256(sequence.encode()).hexdigest(),
                  'msa_sha256': sha256_file(msa_path), 'provided_msa_rows': 1 + len(homologs)}
        try:
            trajectory = session.sample(
                num_samples=int(config['samples']), batch_size=int(config['batch_size']),
                sampler=EDMSampler(num_steps=int(config['sampler_steps']), step_scale=1.5),
                seed=int(config['seed']), progress=True)
            if trajectory.topology.to_fasta() != [sequence]:
                raise ValueError('Predicted coordinate sequence differs from intended target')
            physicality = check_physicality(trajectory, system=system)
            save_frames(trajectory, destination, physicality=physicality)
            statuses = json.loads((destination / 'physicality.json').read_text())
            record.update({'samples': trajectory.n_frames,
                           'physicality_passed': sum(item['status'] == 'passed' for item in statuses),
                           'physicality': statuses,
                           'output_sha256': {file: sha256_file(destination / file) for file in
                                             ('topology.cif', 'samples.xtc', 'physicality.json')}})
            trajectories[name] = trajectory
        except Exception as error:
            record['error'] = {'type': type(error).__name__, 'message': str(error)}
            write_json(destination / 'failure.json', record['error'])
        record['seconds'] = time.monotonic() - target_start
        targets[name] = record
        session.cache_clear()
        del session
        torch.cuda.empty_cache()
        write_json(destination / 'sampling.json', record)
        print(json.dumps({'target': name, 'seconds': record['seconds'],
                          'physicality_passed': record.get('physicality_passed'),
                          'error': record.get('error')}), flush=True)
    pair_rows = []
    if 'WT' in trajectories:
        for name, trajectory in trajectories.items():
            metrics, rows = _trajectory_metrics(trajectory, trajectories['WT'], sequences[name], wt,
                                                same_target=name == 'WT')
            targets[name]['metrics'] = metrics
            pair_rows.extend({'target': name, **row} for row in rows)
    if pair_rows:
        with (output / 'frame_comparisons.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(pair_rows)
    result = {
        'targets': targets, 'config': config, 'seconds': time.monotonic() - started,
        'checkpoint_sha256': config['checkpoint_sha256'],
        'inputs': {str(path): sha256_file(path) for path in
                   (config_path, config['target'], config['candidates'],
                    config['candidate_records'], config['msa_cache'])},
        'homolog_rows_sha256': hashlib.sha256(homolog_text.encode()).hexdigest(),
        'implementation_sha256': sha256_file(__file__),
        'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'coordinate_units': 'MDTraj nm; all reported distances converted to angstroms',
        'mode': 'BioEmu2 structure predictions; no ligand, no experimental WT, no pLDDT/PAE',
        'comparison': 'All predicted frames retained, including physicality failures; paired-frame ranges are descriptive',
    }
    write_json(output / 'structure_summary.json', result)
    return result
