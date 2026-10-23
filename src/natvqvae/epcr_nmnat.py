"""Compare NMNAT substrate environments across the ePCR WT/NatV10/NatV89 models."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from glob import glob
from pathlib import Path

import numpy as np

from .data import Target, load_yaml, sha256_file, write_json
from .nmnat_mapping import (
    _alignment_pairs,
    _apply_transform,
    _atom_region,
    _atom_role,
    _audit_status,
    _distance_flag,
    _fit_from_pairs,
    _format_residue,
    _heavy_atoms,
    _kabsch,
    _mapping_from_pairs,
    _open_structure,
    _parse_tmalign,
    _protein_residues,
    _run_tmalign,
    _sequence,
    _source_contacts,
    _state_ligands,
    _summary_status,
    _target_atom_candidates,
    _write_protein_pdb,
)


def _write_csv(path, rows, fields):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in fields})


def _model_paths(config):
    models = {}
    for clone, item in config['models'].items():
        paths = sorted(Path(path) for path in glob(item['glob']))
        if len(paths) != 3:
            raise FileNotFoundError(f'Expected 3 models for {clone}, found {len(paths)}')
        models[clone] = paths
    reference = models[config['reference']['clone']][int(config['reference']['model_index'])]
    return models, reference


def _atom_dict(atom, transform=None):
    coord = atom.coord if transform is None else _apply_transform([atom.coord], *transform)[0]
    return {'name': atom.name, 'element': atom.element.strip(), 'coord': coord}


def _model_frame(path, reference_ca, reference_full_ca, identity=False):
    structure = _open_structure(path)
    protein = _protein_residues(structure[0]['A'], 1, 341)
    if len(protein) != 341:
        raise ValueError(f'Model is not 341 aa: {path}')
    source_ca = np.asarray([residue['CA'].coord for residue in protein
                            if 4 <= int(residue.id[1]) <= 181], dtype=float)
    target_ca = np.asarray([reference_ca[int(residue.id[1])] for residue in protein
                            if 4 <= int(residue.id[1]) <= 181], dtype=float)
    if identity:
        rmsd, rotation, translation = 0.0, np.eye(3), np.zeros(3)
    else:
        rmsd, rotation, translation = _kabsch(source_ca, target_ca)
    residues = {}
    for residue in protein:
        position = int(residue.id[1])
        residues[position] = {
            'resname': residue.resname.strip(),
            'atoms': {atom.name: _atom_dict(atom, (rotation, translation))
                      for atom in _heavy_atoms(residue)},
        }
    ca_disp = {
        position: float(np.linalg.norm(
            residues[position]['atoms']['CA']['coord'] - reference_full_ca[position]))
        for position in reference_full_ca
    }
    return {
        'path': path,
        'protein': residues,
        'sequence': _sequence(protein),
        'fit_rmsd_A': float(rmsd),
        'rotation': rotation.tolist(),
        'translation': translation.tolist(),
        'ca_displacement_A': ca_disp,
    }


def _source_state(state, source, reference_structure, reference_protein, temporary, tmalign,
                  min_fit_pairs, mapping_config):
    structure, model, protein, contacts = _source_contacts(state, source)
    source_pdb = temporary / f'{state}_source.pdb'
    reference_pdb = temporary / f'{state}_reference.pdb'
    _write_protein_pdb(structure, source['protein_chain'],
                       source['protein_range'][0], source['protein_range'][1], source_pdb)
    _write_protein_pdb(reference_structure, 'A', 4, 181, reference_pdb)
    parsed = _parse_tmalign(_run_tmalign(tmalign, source_pdb, reference_pdb))
    pairs = _alignment_pairs(protein, reference_protein, parsed)
    fit = _fit_from_pairs(pairs, min_fit_pairs)
    rotation = np.asarray(fit['rotation'], dtype=float)
    translation = np.asarray(fit['translation'], dtype=float)
    mapping = _mapping_from_pairs(pairs, rotation, translation)
    ligands = _state_ligands(structure, source)
    transformed = []
    for ligand in ligands:
        atoms = []
        for atom in ligand['atoms']:
            point = _apply_transform([atom.coord], rotation, translation)[0]
            atoms.append({'name': atom.name, 'element': atom.element.strip(),
                          'coord': point})
        transformed.append({
            'chain': ligand['chain'], 'resname': ligand['resname'],
            'resnum': ligand['resnum'], 'atoms': atoms,
        })
    lookup = {
        (ligand['chain'], ligand['resnum'], atom['name']): atom['coord']
        for ligand in transformed for atom in ligand['atoms']
    }
    return {
        'state': state,
        'source': source,
        'contacts': contacts,
        'ligands': transformed,
        'ligand_lookup': lookup,
        'mapping': mapping,
        'fit_rmsd_A': float(fit['fit_rmsd_A']),
        'fit_pairs': fit['fit_pairs'],
        'tm_scores': parsed['tm_scores'],
        'tmalign_rmsd_A': parsed['rmsd_A'],
    }


def _target_residue(frame, position):
    return frame['protein'].get(position)


def _global_nearest(point, frame):
    best = None
    for position, residue in frame['protein'].items():
        for atom_name, atom in residue['atoms'].items():
            distance = float(np.linalg.norm(point - atom['coord']))
            if best is None or distance < best[0]:
                best = (distance, position, residue['resname'], atom_name)
    return best


def _write_ligand_pdb(ligands, path):
    lines = []
    serial = 1
    for ligand_index, ligand in enumerate(ligands, start=1):
        for atom in ligand['atoms']:
            element = atom['element'] or 'C'
            name = atom['name'][:4]
            resname = ligand['resname'][:3]
            x, y, z = atom['coord']
            lines.append(
                f'HETATM{serial:5d} {name:<4s} {resname:>3s} L{ligand_index:4d}    '
                f'{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}')
            serial += 1
    lines.append('END')
    Path(path).write_text('\n'.join(lines) + '\n')


def _write_frame_pdb(frame, path):
    lines = []
    serial = 1
    for position in sorted(frame['protein']):
        residue = frame['protein'][position]
        for name, atom in residue['atoms'].items():
            element = atom['element'] or 'C'
            x, y, z = atom['coord']
            lines.append(
                f'ATOM  {serial:5d} {name:<4s} {residue["resname"]:>3s} A{position:4d}    '
                f'{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}')
            serial += 1
    lines.append('END')
    Path(path).write_text('\n'.join(lines) + '\n')


def _render_structure(reference_frame, models, states, config, coordinates, figures):
    reference_pdb = coordinates / 'reference_WT0.pdb'
    _write_frame_pdb(reference_frame, reference_pdb)
    ligand_objects = {}
    for state, state_data in states.items():
        path = coordinates / f'ligands_{state}.pdb'
        _write_ligand_pdb(state_data['ligands'], path)
        ligand_objects[state] = path
    model_objects = {}
    for clone, frames in models.items():
        for index, frame in enumerate(frames):
            path = coordinates / f'{clone}_model_{index}.pdb'
            _write_frame_pdb(frame, path)
            model_objects[(clone, index)] = path

    executable = Path(config['plot']['pymol'])
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        state_paths = {}
        lines = [f'load {reference_pdb.resolve()}, wt']
        for state, path in ligand_objects.items():
            lines.append(f'load {path.resolve()}, lig_{state}')
        lines += [
            'hide everything',
            'show cartoon, wt',
            'set cartoon_transparency, 0.15, wt',
            'color grey80, wt',
            'select nmnat, wt and resi 4-181',
            'color palecyan, nmnat',
            'select mut10, wt and (resi 83 or resi 162)',
            'show spheres, mut10',
            'color orange, mut10',
            'select mut89, wt and (resi 21 or resi 137 or resi 156)',
            'show spheres, mut89',
            'color magenta, mut89',
            'set sphere_scale, 0.45',
            'set stick_radius, 0.15',
            'bg_color white',
            'orient (wt and resi 4-181) or lig_1F9A or lig_1NUS or lig_2QJO',
        ]
        colors = {'1F9A': 'yellow', '1NUS': 'green', '2QJO': 'blue'}
        for state in ligand_objects:
            state_paths[state] = temporary / f'state_{state}.png'
            lines.append(f'disable lig_{state}')
        for state, path in ligand_objects.items():
            lines += [
                f'enable lig_{state}',
                f'color {colors.get(state, "grey50")}, lig_{state}',
                f'show sticks, lig_{state}',
                f'png {state_paths[state].resolve()}',
                f'disable lig_{state}',
            ]
        lines.append('quit')
        script = figures / 'render_structure.pml'
        script.write_text('\n'.join(lines) + '\n')
        result = subprocess.run([str(executable), '-cq', str(script)],
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f'PyMOL state rendering failed: {result.stderr[-1000:]}')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        titles = {'1F9A': '1F9A ATP/Mg reference pose',
                  '1NUS': '1NUS NMN/APC reference pose',
                  '2QJO': '2QJO NAD/POP reference pose'}
        for ax, state in zip(axes, ligand_objects):
            ax.imshow(plt.imread(state_paths[state]))
            ax.set_title(titles.get(state, state))
            ax.axis('off')
        fig.suptitle('NMNAT reference ligand poses transferred to WT model 0; '
                     'each panel contains one state only', fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        structure_png = figures / 'epcr_nmnat_structure.png'
        fig.savefig(structure_png, dpi=200)
        plt.close(fig)

    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        lines = [f'load {reference_pdb.resolve()}, wt']
        for (clone, index), path in model_objects.items():
            lines.append(f'load {path.resolve()}, {clone}_{index}')
        lines += [
            'hide everything',
            'set stick_radius, 0.18',
            'set stick_transparency, 0.15',
            'bg_color white',
        ]
        panels = (
            ('D83V', 'NatV10', 83, ('orange', 'tv_orange', 'wheat')),
            ('M162V', 'NatV10', 162, ('orange', 'tv_orange', 'wheat')),
            ('I21N', 'NatV89', 21, ('magenta', 'hotpink', 'violet')),
            ('F137Y', 'NatV89', 137, ('magenta', 'hotpink', 'violet')),
            ('I156T', 'NatV89', 156, ('magenta', 'hotpink', 'violet')),
        )
        panel_paths = []
        for mutation, clone, position, colors in panels:
            output = temporary / f'sidechain_{position}.png'
            panel_paths.append((mutation, output))
            lines += [
                'hide everything',
                'show cartoon, wt',
                'set cartoon_transparency, 0.6, wt',
                'color grey80, wt',
                f'show sticks, wt and resi {position}',
                f'color grey50, wt and resi {position}',
            ]
            for index, color in enumerate(colors):
                lines += [
                    f'show sticks, {clone}_{index} and resi {position}',
                    f'color {color}, {clone}_{index} and resi {position}',
                ]
            lines += [
                f'orient (wt and resi {position})',
                f'zoom (wt and resi {position}), 8',
                f'png {output.resolve()}',
            ]
        lines.append('quit')
        script = figures / 'render_sidechains.pml'
        script.write_text('\n'.join(lines) + '\n')
        result = subprocess.run([str(executable), '-cq', str(script)],
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f'PyMOL sidechain rendering failed: {result.stderr[-1000:]}')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(panels), figsize=(25, 5.5))
        for ax, (mutation, path) in zip(axes, panel_paths):
            ax.imshow(plt.imread(path))
            ax.set_title(mutation, fontsize=11)
            ax.axis('off')
        fig.suptitle('Mutation-site sidechains in the common WT0 frame', fontsize=12)
        fig.text(0.5, 0.01, 'grey = WT model 0; orange shades = 3 NatV10 models; '
                 'magenta shades = 3 NatV89 models', ha='center', fontsize=9)
        fig.tight_layout(rect=[0, 0.04, 1, 0.95])
        sidechain_png = figures / 'epcr_nmnat_sidechains.png'
        fig.savefig(sidechain_png, dpi=200)
        plt.close(fig)
    return structure_png, sidechain_png


def _plot_differences(environment_summary, figures):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    clones = ['WT', 'NatV10', 'NatV89']
    ca_positions = [21, 83, 137, 156, 162, 14, 17, 135, 138]
    ca_labels = ['I21', 'D83', 'F137', 'I156', 'M162', 'H14', 'H17', 'T135', 'R138']
    ligand_positions = [21, 83, 137, 156, 162]
    ligand_labels = ['I21', 'D83', 'F137', 'I156', 'M162']
    states = ['1F9A', '1NUS', '2QJO']
    ca = np.full((len(clones), len(ca_positions)), np.nan)
    for i, clone in enumerate(clones):
        for j, position in enumerate(ca_positions):
            rows = [row for row in environment_summary
                    if row['clone'] == clone and int(row['position']) == position
                    and row['state'] == '1F9A']
            if rows:
                ca[i, j] = float(rows[0]['ca_median_A'])
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    image = axes[0].imshow(ca, cmap='viridis', aspect='auto')
    axes[0].set_xticks(range(len(ca_labels)))
    axes[0].set_xticklabels(ca_labels, rotation=45, ha='right')
    axes[0].set_yticks(range(len(clones)))
    axes[0].set_yticklabels(clones)
    axes[0].set_title('CA displacement to WT0 (A; NMNAT 4-181 fit)')
    for i in range(ca.shape[0]):
        for j in range(ca.shape[1]):
            if np.isfinite(ca[i, j]):
                axes[0].text(j, i, f'{ca[i, j]:.2f}', ha='center', va='center',
                             color='white', fontsize=8)
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    for ax, state in zip(axes[1:], states):
        matrix = np.full((len(clones), len(ligand_positions)), np.nan)
        for i, clone in enumerate(clones):
            for j, position in enumerate(ligand_positions):
                rows = [row for row in environment_summary
                        if row['clone'] == clone and int(row['position']) == position
                        and row['state'] == state
                        and row['nearest_ligand_median_A'] != '']
                if rows:
                    matrix[i, j] = float(rows[0]['nearest_ligand_median_A'])
        image = ax.imshow(matrix, cmap='viridis', aspect='auto', vmin=0, vmax=22)
        ax.set_xticks(range(len(ligand_labels)))
        ax.set_xticklabels(ligand_labels, rotation=45, ha='right')
        ax.set_yticks(range(len(clones)))
        ax.set_yticklabels(clones)
        ax.set_title(f'{state} nearest sidechain-ligand distance (A)')
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center',
                            color='white' if matrix[i, j] < 15 else 'black', fontsize=8)
    fig.colorbar(image, ax=axes[-1], fraction=0.046, pad=0.04)
    fig.text(0.5, 0.01, 'Each ligand panel is a separate rigidly transferred state; '
             'not a simultaneous binding pose and not a docking score.', ha='center',
             fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    png = figures / 'epcr_nmnat_differences.png'
    svg = figures / 'epcr_nmnat_differences.svg'
    fig.savefig(png, dpi=200)
    fig.savefig(svg)
    plt.close(fig)
    return png


def build_epcr_nmnat_environment(config_path, output):
    config = load_yaml(config_path)
    mapping_config = load_yaml(config['nmnat_config'])
    target = Target.load(config['target'])
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f'ePCR NMNAT environment output already exists: {output}. '
            'Use a new --output run directory; existing results are not overwritten.')
    output.mkdir(parents=True, exist_ok=True)
    figures = output / 'figures'
    coordinates = output / 'coordinates'
    figures.mkdir()
    coordinates.mkdir()

    paths, reference_path = _model_paths(config)
    reference_structure = _open_structure(reference_path)
    reference_full_protein = _protein_residues(reference_structure[0]['A'], 1, 341)
    reference_protein = _protein_residues(reference_structure[0]['A'], 4, 181)
    reference_ca = {int(residue.id[1]): residue['CA'].coord.astype(float)
                    for residue in reference_protein}
    reference_full_ca = {int(residue.id[1]): residue['CA'].coord.astype(float)
                         for residue in reference_full_protein}
    models = {}
    model_meta = []
    for clone, clone_paths in paths.items():
        frames = []
        for index, path in enumerate(clone_paths):
            frame = _model_frame(path, reference_ca, reference_full_ca,
                                 identity=(path == reference_path))
            frame['name'] = path.name
            frame['index'] = index
            frames.append(frame)
            model_meta.append({
                'clone': clone, 'model': path.name, 'path': str(path),
                'sha256': sha256_file(path), 'sequence_sha256':
                hashlib.sha256(frame['sequence'].encode()).hexdigest(),
                'fit_rmsd_A': round(frame['fit_rmsd_A'], 3),
            })
        if frames[0]['sequence'] != frames[1]['sequence'] or frames[0]['sequence'] != frames[2]['sequence']:
            raise ValueError(f'Models within clone differ in sequence: {clone}')
        models[clone] = frames
    reference_frame = next(
        frame for clone, frames in models.items() for frame in frames
        if frame['path'] == reference_path)
    if reference_frame['sequence'] != target.sequence:
        raise ValueError('Reference model sequence differs from target')
    for clone, items in config['mutations'].items():
        for item in items:
            position = int(item['position'])
            if reference_frame['sequence'][position - 1] != item['from']:
                raise ValueError(f'WT does not contain {item["from"]} at {position}')
            for frame in models[clone]:
                if frame['sequence'][position - 1] != item['to']:
                    raise ValueError(f'{clone} does not contain {item["to"]} at {position}')

    tmalign = Path(mapping_config['alignment']['tmalign'])
    min_fit_pairs = int(mapping_config['alignment']['min_fit_pairs'])
    clash_cutoff = float(mapping_config['analysis']['clash_cutoff_A'])
    close_cutoff = float(mapping_config['analysis']['close_cutoff_A'])
    states = {}
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        for state, source in mapping_config['sources'].items():
            states[state] = _source_state(
                state, source, reference_structure, reference_protein,
                temporary, tmalign, min_fit_pairs, mapping_config)

    # Contact rows: one ligand pose per state, common across all nine models.
    rows = []
    state_cache = {}
    for state, state_data in states.items():
        source_structure = _open_structure(state_data['source']['path'])
        source_protein = _protein_residues(
            source_structure[0][state_data['source']['protein_chain']],
            state_data['source']['protein_range'][0],
            state_data['source']['protein_range'][1])
        residue_lookup = {int(residue.id[1]): residue for residue in source_protein}
        state_cache[state] = {'residue_lookup': residue_lookup}
        for contact in state_data['contacts']:
            source_residue = residue_lookup[int(contact['protein_resnum'])]
            source_atom = source_residue[contact['protein_atom']]
            source_element = source_atom.element.strip()
            info = state_data['mapping'].get(int(contact['protein_resnum']))
            point = state_data['ligand_lookup'][
                (contact['ligand_chain'], contact['ligand_resnum'], contact['ligand_atom'])]
            for clone, frames in models.items():
                for frame in frames:
                    target_atom = None
                    target_distance = None
                    target_atom_match = 'unmapped'
                    chemistry = 'unmapped'
                    target_resname = ''
                    target_resnum = info['target_resnum'] if info else ''
                    target_position = info['target_resnum'] if info else None
                    target_region = ''
                    target_element = ''
                    target_role = ''
                    source_role = _atom_role(contact['protein_resname'], source_atom.name,
                                             source_element)
                    if info is not None:
                        target_residue = _target_residue(frame, target_position)
                        if target_residue is None:
                            target_atom_match = 'target_residue_missing'
                            chemistry = 'missing'
                        else:
                            target_resname = target_residue['resname']
                            names, target_atom_match, chemistry = _target_atom_candidates(
                                contact['protein_resname'], source_atom.name,
                                source_element, target_resname)
                            candidates = [target_residue['atoms'][name] for name in names
                                          if name in target_residue['atoms']]
                            if candidates:
                                target_atom = min(
                                    candidates,
                                    key=lambda atom: float(np.linalg.norm(point - atom['coord'])))
                                target_distance = float(np.linalg.norm(point - target_atom['coord']))
                                target_region = _atom_region(target_atom['name'])
                                target_element = target_atom['element']
                                target_role = _atom_role(target_resname, target_atom['name'],
                                                         target_element)
                            elif target_atom_match == 'same_name':
                                target_atom_match = 'target_atom_missing'
                                chemistry = 'missing'
                    global_best = _global_nearest(point, frame)
                    cutoff = (mapping_config['sources'][state].get('metal_cutoff_A', 4.0)
                              if contact['ligand_resname'] in {'MG', 'MN'} else
                              float(mapping_config['sources'][state].get('contact_cutoff_A', 4.0)))
                    local_flag = _distance_flag(target_distance, clash_cutoff, close_cutoff)
                    global_flag = _distance_flag(global_best[0], clash_cutoff, close_cutoff)
                    mapping_confidence = info['confidence'] if info else 'unmapped'
                    status = _audit_status(mapping_confidence, target_distance, cutoff,
                                           local_flag, global_flag, chemistry)
                    if target_atom_match in {'role_mismatch', 'backbone_proline_mismatch'}:
                        status = 'role_mismatch'
                    rows.append({
                        'state': state,
                        'contact_id': contact['contact_id'],
                        'source_resnum': contact['protein_resnum'],
                        'source_resname': contact['protein_resname'],
                        'source_atom': contact['protein_atom'],
                        'ligand_resname': contact['ligand_resname'],
                        'ligand_atom': contact['ligand_atom'],
                        'source_distance_A': contact['distance_A'],
                        'clone': clone,
                        'model': frame['name'],
                        'mapping_status': 'mapped' if info else 'unmapped',
                        'kvp40_resnum': target_resnum,
                        'kvp40_resname': target_resname,
                        'mapping_confidence': mapping_confidence,
                        'target_atom': target_atom['name'] if target_atom else '',
                        'target_atom_match': target_atom_match,
                        'target_region': target_region,
                        'target_distance_A': round(target_distance, 2)
                        if target_distance is not None else '',
                        'local_flag': local_flag,
                        'global_nearest': f'{global_best[2]}{global_best[1]}:{global_best[3]}',
                        'global_distance_A': round(global_best[0], 2),
                        'global_flag': global_flag,
                        'chemistry_class': chemistry,
                        'source_role': source_role,
                        'target_role': target_role,
                        'contact_status': status,
                    })
    contact_fields = ['state', 'contact_id', 'source_resnum', 'source_resname', 'source_atom',
                      'ligand_resname', 'ligand_atom', 'source_distance_A', 'clone', 'model',
                      'mapping_status', 'kvp40_resnum', 'kvp40_resname', 'mapping_confidence',
                      'target_atom', 'target_atom_match', 'target_region', 'target_distance_A',
                      'local_flag', 'global_nearest', 'global_distance_A', 'global_flag',
                      'chemistry_class', 'source_role', 'target_role', 'contact_status']
    _write_csv(output / 'contact_rows.csv', rows, contact_fields)

    # Contact summary: clone versus WT self-range.
    grouped = {}
    for row in rows:
        grouped.setdefault((row['state'], row['contact_id'], row['clone']), []).append(row)
    summary_rows = []
    for (state, contact_id, clone), group in grouped.items():
        statuses = [row['contact_status'] for row in group]
        counts = {status: statuses.count(status) for status in
                  ('supported_candidate', 'geometric_only', 'clash', 'not_recovered',
                   'uncertain_mapping', 'unmapped', 'role_mismatch')}
        distance_values = [float(row['target_distance_A']) for row in group
                           if row['target_distance_A'] != '']
        wt_group = grouped.get((state, contact_id, 'WT'), [])
        wt_values = [float(row['target_distance_A']) for row in wt_group
                     if row['target_distance_A'] != '']
        wt_min = min(wt_values) if wt_values else ''
        wt_max = max(wt_values) if wt_values else ''
        outside = [row['model'] for row in group
                   if row['target_distance_A'] != '' and wt_min != ''
                   and (float(row['target_distance_A']) < wt_min
                        or float(row['target_distance_A']) > wt_max)]
        first = group[0]
        summary_rows.append({
            'state': state,
            'contact_id': contact_id,
            'source_resnum': first['source_resnum'],
            'source_resname': first['source_resname'],
            'source_atom': first['source_atom'],
            'ligand_resname': first['ligand_resname'],
            'ligand_atom': first['ligand_atom'],
            'clone': clone,
            'n_models': len(group),
            'n_supported': counts['supported_candidate'],
            'n_role_mismatch': counts['role_mismatch'],
            'n_not_recovered': counts['not_recovered'],
            'n_clash': counts['clash'],
            'n_unmapped': counts['unmapped'],
            'status_summary': _summary_status(counts, len(group)),
            'target_distance_median_A': round(float(np.median(distance_values)), 2)
            if distance_values else '',
            'target_distance_min_A': round(min(distance_values), 2) if distance_values else '',
            'target_distance_max_A': round(max(distance_values), 2) if distance_values else '',
            'wt_median_A': round(float(np.median(wt_values)), 2) if wt_values else '',
            'wt_min_A': round(wt_min, 2) if wt_min != '' else '',
            'wt_max_A': round(wt_max, 2) if wt_max != '' else '',
            'clone_vs_wt_median_A': round(float(np.median(distance_values))
                                          - float(np.median(wt_values)), 2)
            if distance_values and wt_values else '',
            'models_outside_wt_range': ';'.join(outside),
            'n_models_outside_wt_range': len(outside),
            'chemistry_classes': ';'.join(sorted({row['chemistry_class'] for row in group})),
            'target_atom_matches': ';'.join(sorted({row['target_atom_match'] for row in group})),
        })
    contact_summary_fields = ['state', 'contact_id', 'source_resnum', 'source_resname',
                              'source_atom', 'ligand_resname', 'ligand_atom', 'clone',
                              'n_models', 'n_supported', 'n_role_mismatch', 'n_not_recovered',
                              'n_clash', 'n_unmapped', 'status_summary',
                              'target_distance_median_A', 'target_distance_min_A',
                              'target_distance_max_A', 'wt_median_A', 'wt_min_A', 'wt_max_A',
                              'clone_vs_wt_median_A', 'models_outside_wt_range',
                              'n_models_outside_wt_range', 'chemistry_classes',
                              'target_atom_matches']
    _write_csv(output / 'contact_summary.csv', summary_rows, contact_summary_fields)

    # Mutation and anchor local environments.
    union_mutations = {}
    for clone, items in config['mutations'].items():
        for item in items:
            union_mutations[int(item['position'])] = item
    anchors = [int(position) for position in config['analysis']['anchor_residues']]
    positions = sorted(set(union_mutations) | set(anchors))
    env_rows = []
    for clone, frames in models.items():
        clone_mutations = {int(item['position']): item
                           for item in config['mutations'].get(clone, [])}
        for frame in frames:
            for position in positions:
                residue = frame['protein'].get(position)
                if residue is None:
                    continue
                sidechain = {name: atom for name, atom in residue['atoms'].items()
                             if name not in {'N', 'CA', 'C', 'O', 'OXT'}}
                sidechain_atoms = sorted(sidechain)
                focus_points = list(sidechain.values()) or [residue['atoms']['CA']]
                item = union_mutations.get(position)
                mutation_focus_names = item.get('focus_atoms', []) if item else []
                mutation_focus_atoms = {
                    name: residue['atoms'][name]
                    for name in mutation_focus_names
                    if name in residue['atoms']
                }
                others = [
                    (other_position, other)
                    for other_position, other in frame['protein'].items()
                    if other_position != position
                ]
                local = []
                for other_position, other in others:
                    for atom in other['atoms'].values():
                        distance = min(float(np.linalg.norm(atom['coord'] - focus['coord']))
                                       for focus in focus_points)
                        if distance <= float(config['analysis']['mutation_environment_A']):
                            local.append((distance, other_position, other['resname'], atom['name']))
                local.sort()
                nearest_protein = local[0] if local else None
                state_values = []
                if position <= 181:
                    for state, state_data in states.items():
                        best = None
                        focus_best = None
                        for ligand in state_data['ligands']:
                            for ligand_atom in ligand['atoms']:
                                distance = min(
                                    float(np.linalg.norm(ligand_atom['coord'] - focus['coord']))
                                    for focus in focus_points)
                                if best is None or distance < best[0]:
                                    best = (distance, ligand['resname'], ligand_atom['name'])
                                for focus in mutation_focus_atoms.values():
                                    focus_distance = float(np.linalg.norm(
                                        ligand_atom['coord'] - focus['coord']))
                                    if focus_best is None or focus_distance < focus_best[0]:
                                        focus_best = (focus_distance, ligand['resname'],
                                                      ligand_atom['name'])
                        state_values.append((state, best, focus_best))
                else:
                    state_values.append(('none', None, None))
                for state, best, focus_best in state_values:
                    env_rows.append({
                        'category': 'anchor' if position in anchors else 'mutation',
                        'clone': clone,
                        'model': frame['name'],
                        'position': position,
                        'ref_resname': target.sequence[position - 1],
                        'model_resname': residue['resname'],
                        'domain': ('NMNAT' if position <= 181 else 'Nudix'),
                        'mutation': clone_mutations[position]['name']
                        if position in clone_mutations else '',
                        'is_clone_mutation': position in clone_mutations,
                        'state': state,
                        'ca_displacement_A': round(frame['ca_displacement_A'][position], 2),
                        'sidechain_atoms': ';'.join(sidechain_atoms),
                        'local_contact_count_8A': len({entry[1] for entry in local}),
                        'nearest_protein_residue': f'{nearest_protein[2]}{nearest_protein[1]}'
                        if nearest_protein else '',
                        'nearest_protein_atom': nearest_protein[3] if nearest_protein else '',
                        'nearest_protein_distance_A': round(nearest_protein[0], 2)
                        if nearest_protein else '',
                        'nearest_ligand_resname': best[1] if best else '',
                        'nearest_ligand_atom': best[2] if best else '',
                        'nearest_ligand_distance_A': round(best[0], 2) if best else '',
                        'focus_atoms': ';'.join(sorted(mutation_focus_atoms)),
                        'focus_nearest_ligand_resname': focus_best[1] if focus_best else '',
                        'focus_nearest_ligand_atom': focus_best[2] if focus_best else '',
                        'focus_nearest_ligand_distance_A': round(focus_best[0], 2)
                        if focus_best else '',
                        'wt_ca_min_A': round(min(models['WT'][i]['ca_displacement_A'][position]
                                                 for i in range(3)), 2),
                        'wt_ca_max_A': round(max(models['WT'][i]['ca_displacement_A'][position]
                                                 for i in range(3)), 2),
                        'outside_wt_ca_range': not (
                            min(models['WT'][i]['ca_displacement_A'][position] for i in range(3))
                            <= frame['ca_displacement_A'][position]
                            <= max(models['WT'][i]['ca_displacement_A'][position] for i in range(3))),
                    })
    env_fields = ['category', 'clone', 'model', 'position', 'ref_resname', 'model_resname',
                  'domain', 'mutation', 'is_clone_mutation', 'state', 'ca_displacement_A',
                  'sidechain_atoms',
                  'local_contact_count_8A', 'nearest_protein_residue', 'nearest_protein_atom',
                  'nearest_protein_distance_A', 'nearest_ligand_resname', 'nearest_ligand_atom',
                  'nearest_ligand_distance_A', 'focus_atoms',
                  'focus_nearest_ligand_resname', 'focus_nearest_ligand_atom',
                  'focus_nearest_ligand_distance_A', 'wt_ca_min_A', 'wt_ca_max_A',
                  'outside_wt_ca_range']
    _write_csv(output / 'position_environment.csv', env_rows, env_fields)
    env_grouped = {}
    for row in env_rows:
        env_grouped.setdefault((row['category'], row['position'], row['clone'], row['state']),
                               []).append(row)
    env_summary = []
    for (category, position, clone, state), group in env_grouped.items():
        ca_values = [float(row['ca_displacement_A']) for row in group]
        ligand_values = [float(row['nearest_ligand_distance_A']) for row in group
                         if row['nearest_ligand_distance_A'] != '']
        focus_values = [float(row['focus_nearest_ligand_distance_A']) for row in group
                        if row['focus_nearest_ligand_distance_A'] != '']
        first = group[0]
        env_summary.append({
            'category': category,
            'position': position,
            'clone': clone,
            'state': state,
            'ref_resname': first['ref_resname'],
            'model_resnames': ';'.join(sorted({row['model_resname'] for row in group})),
            'domain': first['domain'],
            'mutation': first['mutation'],
            'is_clone_mutation': first['is_clone_mutation'],
            'n_models': len(group),
            'ca_median_A': round(float(np.median(ca_values)), 2),
            'ca_min_A': round(min(ca_values), 2),
            'ca_max_A': round(max(ca_values), 2),
            'wt_ca_min_A': first['wt_ca_min_A'],
            'wt_ca_max_A': first['wt_ca_max_A'],
            'models_outside_wt_ca_range': sum(bool(row['outside_wt_ca_range'])
                                              for row in group),
            'nearest_ligand_median_A': round(float(np.median(ligand_values)), 2)
            if ligand_values else '',
            'nearest_ligand_min_A': round(min(ligand_values), 2) if ligand_values else '',
            'nearest_ligand_max_A': round(max(ligand_values), 2) if ligand_values else '',
            'focus_atoms': ';'.join(sorted({atom for row in group
                                            for atom in row['focus_atoms'].split(';')
                                            if atom})),
            'focus_nearest_ligand_median_A': round(float(np.median(focus_values)), 2)
            if focus_values else '',
            'focus_nearest_ligand_min_A': round(min(focus_values), 2) if focus_values else '',
            'focus_nearest_ligand_max_A': round(max(focus_values), 2) if focus_values else '',
            'sidechain_atoms': ';'.join(sorted({atom for row in group
                                                for atom in row['sidechain_atoms'].split(';')
                                                if atom})),
        })
    env_summary_fields = ['category', 'position', 'clone', 'state', 'ref_resname',
                          'model_resnames', 'domain', 'mutation', 'is_clone_mutation',
                          'n_models', 'ca_median_A',
                          'ca_min_A', 'ca_max_A', 'wt_ca_min_A', 'wt_ca_max_A',
                          'models_outside_wt_ca_range', 'nearest_ligand_median_A',
                          'nearest_ligand_min_A', 'nearest_ligand_max_A', 'focus_atoms',
                          'focus_nearest_ligand_median_A', 'focus_nearest_ligand_min_A',
                          'focus_nearest_ligand_max_A', 'sidechain_atoms']
    _write_csv(output / 'position_environment_summary.csv', env_summary, env_summary_fields)

    figures_out = {}
    if config.get('plot', {}).get('enabled', True):
        structure_png, sidechain_png = _render_structure(
            reference_frame, models, states, config, coordinates, figures)
        figures_out['structure'] = str(structure_png)
        figures_out['sidechains'] = str(sidechain_png)
        figures_out['differences'] = str(_plot_differences(env_summary, figures))

    provenance = {
        'config': str(Path(config_path).resolve()),
        'config_sha256': sha256_file(config_path),
        'nmnat_config': str(Path(config['nmnat_config']).resolve()),
        'nmnat_config_sha256': sha256_file(config['nmnat_config']),
        'target_sequence_sha256': hashlib.sha256(target.sequence.encode()).hexdigest(),
        'reference': {'clone': config['reference']['clone'],
                      'model': reference_path.name},
        'model_identity': config.get('model_identity', {}),
        'models': model_meta,
        'state_sources': {
            state: {
                'pdb_id': data['source']['pdb_id'],
                'path': data['source']['path'],
                'sha256': sha256_file(data['source']['path']),
                'tm_scores': data['tm_scores'],
                'tmalign_rmsd_A': data['tmalign_rmsd_A'],
                'fit_rmsd_A': data['fit_rmsd_A'],
                'fit_pairs': data['fit_pairs'],
            }
            for state, data in states.items()
        },
        'common_frame': 'WT model 0 NMNAT 4-181; each model aligned by CA 4-181; source ligands transformed once to WT0',
        'source_code': {
            'epcr_nmnat': {'path': str(Path(__file__).resolve()),
                           'sha256': sha256_file(Path(__file__).resolve())},
            'nmnat_mapping': {'path': str(Path(__file__).with_name('nmnat_mapping.py')),
                              'sha256': sha256_file(Path(__file__).with_name('nmnat_mapping.py'))},
        },
        'invocation': {'argv': list(sys.argv), 'cwd': str(Path.cwd())},
        'software': {
            'python': sys.version.split()[0],
            'numpy': np.__version__,
        },
        'analysis': config['analysis'],
        'figures': figures_out,
    }
    write_json(output / 'provenance.json', provenance)
    return output
