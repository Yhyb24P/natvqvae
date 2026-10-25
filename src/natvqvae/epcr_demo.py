"""Render an independent static comparison of the ePCR WT/NatV10/NatV89 clones."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from .data import load_yaml, sha256_file, write_json
from .nmnat_mapping import _protein_residues

CLONES = ('WT', 'NatV10', 'NatV89')
DOMAINS = ('NMNAT', 'linker', 'Nudix', 'other')
OUTPUTS = ('clone_overview', 'clone_overlay')


def _pdb_chain(path):
    from Bio.PDB import PDBParser

    structure = PDBParser(QUIET=True).get_structure(Path(path).stem, str(path))
    chains = list(structure.get_chains())
    if len(chains) != 1:
        raise ValueError(f'Expected one chain in {path}, found {len(chains)}')
    return chains[0]


def _protein_sequence(path):
    from Bio.Data.PDBData import protein_letters_3to1

    residues = _protein_residues(_pdb_chain(path), 1, 341)
    if len(residues) != 341:
        raise ValueError(f'{path} is not a full-length 341 aa model')
    return ''.join(protein_letters_3to1.get(residue.resname, 'X') for residue in residues), residues


def _ca_coordinates(residues, start, end):
    return {int(residue.id[1]): np.asarray(residue['CA'].coord, dtype=float)
            for residue in residues if start <= int(residue.id[1]) <= end}


def _load_variants(path):
    rows = {}
    with Path(path).open(newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows[row['name']] = row
    if set(rows) != set(CLONES):
        raise ValueError(f'Expected variants for {CLONES}')
    for clone, row in rows.items():
        sequence = row['sequence']
        if int(row['length']) != 341 or len(sequence) != 341:
            raise ValueError(f'{clone} in variants.csv is not 341 aa')
        if hashlib.sha256(sequence.encode()).hexdigest() != row['sequence_sha256']:
            raise ValueError(f'{clone} sequence SHA differs from variants.csv')
    return rows


def _load_demo(config_path):
    config_path = Path(config_path)
    config = load_yaml(config_path)
    if set(config['models']) != set(CLONES):
        raise ValueError(f'Expected models for {CLONES}')
    reference = config['reference']
    if reference.get('clone') != 'WT' or int(reference.get('model', -1)) != 0:
        raise ValueError('Expected reference WT model_0')
    for key in ('source_report', 'source_variants'):
        if not reference.get(key):
            raise ValueError(f'reference.{key} is required')
        if not Path(reference[key]).is_file():
            raise FileNotFoundError(reference[key])
    expected_domains = {'NMNAT': [4, 181], 'linker': [182, 203], 'Nudix': [204, 338]}
    if {name: [int(value) for value in bounds]
            for name, bounds in config['domains'].items()} != expected_domains:
        raise ValueError('Domains must be exactly NMNAT 4-181, linker 182-203, Nudix 204-338')
    if set(config['colors']['domains']) != set(DOMAINS):
        raise ValueError('Domain colors are incomplete')
    if set(config['colors']['clones']) != set(CLONES):
        raise ValueError('Clone colors are incomplete')
    if not Path(config['plot']['pymol']).is_file():
        raise FileNotFoundError(config['plot']['pymol'])
    for name in ('width', 'height', 'dpi'):
        if int(config['plot'][name]) <= 0:
            raise ValueError(f'plot.{name} must be positive')

    variants = _load_variants(reference['source_variants'])
    models = {}
    sequences = {}
    reference_ca = None
    for clone in CLONES:
        path = Path(config['models'][clone])
        if not path.is_file():
            raise FileNotFoundError(path)
        sequence, residues = _protein_sequence(path)
        expected = variants[clone]
        if sequence != expected['sequence']:
            raise ValueError(f'{clone} PDB sequence differs from variants.csv')
        if hashlib.sha256(sequence.encode()).hexdigest() != expected['sequence_sha256']:
            raise ValueError(f'{clone} PDB sequence SHA differs from variants.csv')
        ca = _ca_coordinates(residues, 4, 181)
        if reference_ca is None:
            reference_ca = ca
        model_rmsd = float(np.sqrt(np.mean([
            np.sum((ca[position] - reference_ca[position]) ** 2)
            for position in sorted(reference_ca)
        ])))
        sequences[clone] = sequence
        models[clone] = {
            'path': str(path.resolve()),
            'sha256': sha256_file(path),
            'sequence_sha256': expected['sequence_sha256'],
            'alignment_rmsd_A': model_rmsd,
        }
        if clone == reference['clone'] and model_rmsd != 0.0:
            raise ValueError('Reference model is not the alignment origin')

    mutations = config['mutations']
    if set(mutations) != {'NatV10', 'NatV89'}:
        raise ValueError('Expected mutations for NatV10 and NatV89 only')
    for clone, items in mutations.items():
        expected = {(int(item['position']), item['from'], item['to']) for item in items}
        actual = {(position, wt, alt)
                  for position, (wt, alt) in enumerate(zip(sequences['WT'], sequences[clone]), 1)
                  if wt != alt}
        if actual != expected:
            raise ValueError(f'{clone} mutation list differs from WT-to-clone sequence changes')
        for item in items:
            if item['from'] == item['to']:
                raise ValueError(f'{clone} has a non-changing edit at {item["position"]}')
    return config_path.resolve(), config, models


def _set_color_lines(config):
    lines = []
    for name in DOMAINS:
        rgb = config['colors']['domains'][name]
        lines.append(f'set_color demo_{name.lower()}, [{rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f}]')
    for clone in CLONES:
        rgb = config['colors']['clones'][clone]
        lines.append(f'set_color demo_{clone.lower()}, [{rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f}]')
    return lines


def _domain_color_lines(config, obj):
    lines = [f'color demo_other, {obj}']
    for name in DOMAINS[:3]:
        start, end = config['domains'][name]
        lines.append(f'color demo_{name.lower()}, {obj} and resi {start}-{end}')
    return lines


def _common_lines(config):
    width = int(config['plot']['width'])
    height = int(config['plot']['height'])
    return [
        'hide everything',
        'show cartoon, all',
        *_set_color_lines(config),
        'bg_color white',
        'set ray_opaque_background, 1',
        'set antialias, 2',
        'set ray_shadows, 0',
        'set cartoon_fancy_helices, 1',
        'set sphere_scale, 0.65',
        f'viewport {width}, {height}',
    ]


def _panel_script(config, raw_paths):
    lines = []
    for clone in CLONES:
        lines.append(f'load "{Path(config["models"][clone]).resolve()}", demo_{clone.lower()}')
    lines += _common_lines(config)
    for clone in CLONES:
        lines += _domain_color_lines(config, f'demo_{clone.lower()}')
    lines += ['orient all']
    for index, clone in enumerate(CLONES):
        obj = f'demo_{clone.lower()}'
        for other in CLONES:
            if other != clone:
                lines.append(f'disable demo_{other.lower()}')
        lines.append(f'enable {obj}')
        if clone != 'WT':
            positions = '+'.join(str(int(item['position'])) for item in config['mutations'][clone])
            lines += [
                f'show spheres, {obj} and name CA and resi {positions}',
                f'color demo_{clone.lower()}, {obj} and name CA and resi {positions}',
            ]
        width, height = int(config['plot']['width']), int(config['plot']['height'])
        lines += [f'ray {width}, {height}', f'png {raw_paths[clone]}']
    lines += ['quit']
    return '\n'.join(lines) + '\n'


def _overlay_script(config, raw_path):
    lines = []
    for clone in CLONES:
        lines.append(f'load "{Path(config["models"][clone]).resolve()}", demo_{clone.lower()}')
    lines += _common_lines(config)
    for clone in CLONES:
        lines.append(f'color demo_{clone.lower()}, demo_{clone.lower()}')
    lines += [
        'set cartoon_transparency, 0.60, demo_wt',
        'set cartoon_transparency, 0.05, demo_natv10',
        'set cartoon_transparency, 0.05, demo_natv89',
        'orient all',
    ]
    width, height = int(config['plot']['width']), int(config['plot']['height'])
    lines += [f'ray {width}, {height}', f'png {raw_path}', 'quit']
    return '\n'.join(lines) + '\n'


def _marker_script(config, mask_paths):
    lines = []
    for clone in CLONES:
        lines.append(f'load "{Path(config["models"][clone]).resolve()}", demo_{clone.lower()}')
    lines += _common_lines(config)
    lines += ['orient all', 'set antialias, 0', 'set_color demo_black, [0.0, 0.0, 0.0]']
    for (clone, position), path in mask_paths.items():
        obj = f'demo_{clone.lower()}'
        for other in CLONES:
            lines.append(f'{"enable" if other == clone else "disable"} demo_{other.lower()}')
        lines += [
            'hide everything',
            f'show spheres, {obj} and name CA and resi {position}',
            f'color demo_black, {obj} and name CA and resi {position}',
            'set sphere_scale, 0.80',
            f'ray {int(config["plot"]["width"])}, {int(config["plot"]["height"])}',
            f'png {path}',
        ]
    lines += ['quit']
    return '\n'.join(lines) + '\n'


def _mask_centroids(mask_paths, config):
    from PIL import Image

    found = {clone: [] for clone in ('NatV10', 'NatV89')}
    for clone, items in config['mutations'].items():
        for item in items:
            path = mask_paths[(clone, int(item['position']))]
            pixels = np.asarray(Image.open(path).convert('RGB'))
            mask = np.any(pixels < 245, axis=2)
            if not mask.any():
                raise RuntimeError(f'Marker mask is empty: {path}')
            y, x = np.nonzero(mask)
            found[clone].append((item['name'], float(x.mean()), float(y.mean())))
        found[clone].sort(key=lambda item: item[2])
    return found



def _run_pymol(executable, script, workdir):
    script_path = Path(workdir) / script.name
    script_path.write_text(script.read_text())
    result = subprocess.run([str(executable), '-cq', str(script_path)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'PyMOL rendering failed: {result.stderr[-1000:]}')


def _pymol_version(executable):
    result = subprocess.run([str(executable), '-cq', '-d', 'print(cmd.get_version()[0])'],
                            capture_output=True, text=True)
    match = re.search(r'(\d+\.\d+\.\d+)', result.stdout)
    return match.group(1) if match else None


def _edits(config, clone):
    if clone == 'WT':
        return 'reference'
    return '; '.join(item['name'] for item in config['mutations'][clone])


def _label_y_positions(values, low, high, min_gap):
    positions = []
    for value in values:
        position = max(float(value), low)
        if positions:
            position = max(position, positions[-1] + min_gap)
        positions.append(position)
    if positions and positions[-1] > high:
        shift = positions[-1] - high
        positions = [position - shift for position in positions]
        if positions[0] < low:
            positions = list(np.linspace(low, high, len(values)))
    return positions


def _annotate_markers(ax, clone, centroids, config, width, height):
    items = centroids.get(clone, [])
    if not items:
        return
    color = config['colors']['clones'][clone]
    groups = (
        ([item for item in items if item[1] < width / 2], width * 0.035, 'left'),
        ([item for item in items if item[1] >= width / 2], width * 0.965, 'right'),
    )
    for subset, text_x, align in groups:
        if not subset:
            continue
        ys = _label_y_positions([item[2] for item in subset],
                                height * 0.12, height * 0.88, height * 0.075)
        for (name, x, y), text_y in zip(subset, ys):
            ax.annotate(
                name, xy=(x, y), xytext=(text_x, text_y), textcoords='data',
                ha=align, va='center', fontsize=10.5, fontweight='bold', color=color,
                arrowprops=dict(arrowstyle='-', color=color, lw=0.9, shrinkA=3, shrinkB=3),
                bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='none',
                          alpha=0.85),
                zorder=7,
            )
            ax.plot([x], [y], marker='o', markersize=4, color=color,
                    markeredgecolor='white', markeredgewidth=0.8, zorder=6)


def _compose_overview(raw_paths, config, output, centroids):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.6))
    for ax, clone in zip(axes, CLONES):
        image = plt.imread(raw_paths[clone])
        ax.imshow(image)
        height, width = image.shape[:2]
        _annotate_markers(ax, clone, centroids, config, width, height)
        ax.set_title(f'{clone}\n{_edits(config, clone)}', fontsize=11)
        ax.axis('off')
    handles = [Patch(facecolor=config['colors']['domains'][name], label=name)
               for name in DOMAINS]
    handles += [Patch(facecolor=config['colors']['clones'][clone], label=f'{clone} marker')
                for clone in ('NatV10', 'NatV89')]
    fig.suptitle('KVP40 NatV WT / NatV10 / NatV89 predicted structures', fontsize=16)
    fig.text(0.5, 0.925,
             'model_0; WT model_0 frame after NMNAT 4–181 Cα alignment; '
             'domain colors; labels/leaders project the true CA positions',
             ha='center', fontsize=10)
    fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.07))
    fig.text(0.5, 0.018,
             'Predicted monomers; no activity, stability, or expression inference. '
             'Nudix placement can shift with the NMNAT alignment.',
             ha='center', fontsize=9)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.17, wspace=0.02)
    fig.savefig(output, dpi=int(config['plot']['dpi']), facecolor='white')
    plt.close(fig)


def _write_mutation_sites(config, models, output):
    rows = []
    for clone in ('NatV10', 'NatV89'):
        residues = _protein_residues(_pdb_chain(models[clone]['path']), 1, 341)
        ca = _ca_coordinates(residues, 1, 341)
        for item in config['mutations'][clone]:
            position = int(item['position'])
            domain = next(name for name, (start, end) in config['domains'].items()
                          if start <= position <= end)
            rows.append({
                'clone': clone,
                'edit': item['name'],
                'position': position,
                'from': item['from'],
                'to': item['to'],
                'domain': domain,
                'ca_x': f'{ca[position][0]:.3f}',
                'ca_y': f'{ca[position][1]:.3f}',
                'ca_z': f'{ca[position][2]:.3f}',
                'source_pdb': models[clone]['path'],
            })
    path = output / 'mutation_sites.csv'
    fields = ('clone', 'edit', 'position', 'from', 'to', 'domain',
              'ca_x', 'ca_y', 'ca_z', 'source_pdb')
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    return path


def _compose_overlay(raw_path, config, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(plt.imread(raw_path))
    ax.axis('off')
    handles = [Patch(facecolor=config['colors']['clones'][clone], label=clone)
               for clone in CLONES]
    fig.suptitle('Full-length overlay after NMNAT 4–181 Cα alignment', fontsize=16)
    fig.text(0.5, 0.925,
             'WT model_0 reference; WT grey, NatV10 orange, NatV89 magenta; '
             'predicted monomers',
             ha='center', fontsize=10)
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.07))
    fig.text(0.5, 0.018,
             'Nudix placement can shift with the NMNAT alignment; this is not an '
             'independent per-domain RMSD or a functional comparison.',
             ha='center', fontsize=9)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.15)
    fig.savefig(output, dpi=int(config['plot']['dpi']), facecolor='white')
    plt.close(fig)


def render_epcr_clone_demo(config_path, output):
    config_path, config, models = _load_demo(config_path)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Clone demo output already contains files: {output}')
    output.mkdir(parents=True, exist_ok=True)

    render_input = {
        'task': config.get('task', 'kvp40_epcr_clone_demo'),
        'config': str(config_path),
        'config_sha256': sha256_file(config_path),
        'reference': config['reference'],
        'sources': {
            'variants': {
                'path': str(Path(config['reference']['source_variants']).resolve()),
                'sha256': sha256_file(config['reference']['source_variants']),
            },
            'report': str(Path(config['reference']['source_report']).resolve()),
        },
        'models': models,
        'domains': config['domains'],
        'mutations': config['mutations'],
        'plot': {key: config['plot'][key] for key in ('pymol', 'width', 'height', 'dpi')},
    }
    render_input_path = output / 'render_input.json'
    write_json(render_input_path, render_input)

    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        raw_paths = {clone: temporary / f'{clone}.png' for clone in CLONES}
        panel_script = temporary / 'clone_panels.pml'
        panel_script.write_text(_panel_script(config, raw_paths))
        _run_pymol(config['plot']['pymol'], panel_script, temporary)

        mask_paths = {
            (clone, int(item['position'])): temporary / f'mask_{clone}_{item["position"]}.png'
            for clone, items in config['mutations'].items() for item in items
        }
        marker_script = temporary / 'clone_markers.pml'
        marker_script.write_text(_marker_script(config, mask_paths))
        _run_pymol(config['plot']['pymol'], marker_script, temporary)
        centroids = _mask_centroids(mask_paths, config)
        if [len(centroids['NatV10']), len(centroids['NatV89'])] != [4, 5]:
            raise RuntimeError('Unexpected marker mask count')

        overlay_script = temporary / 'clone_overlay.pml'
        overlay_raw = temporary / 'overlay.png'
        overlay_script.write_text(_overlay_script(config, overlay_raw))
        _run_pymol(config['plot']['pymol'], overlay_script, temporary)
        expected_raw = [*raw_paths.values(), *mask_paths.values(), overlay_raw]
        missing = [str(path) for path in expected_raw if not path.exists()]
        if missing:
            raise RuntimeError(f'PyMOL did not produce expected images: {missing}')
        overview_path = output / 'clone_overview.png'
        overlay_path = output / 'clone_overlay.png'
        mutation_sites_path = _write_mutation_sites(config, models, output)
        _compose_overview(raw_paths, config, overview_path, centroids)
        _compose_overlay(overlay_raw, config, overlay_path)

    identity = {
        'status': 'COMPLETED',
        'mode': 'static',
        'render_input_sha256': sha256_file(render_input_path),
        'script_sha256': sha256_file(Path(__file__)),
        'source_variants': {
            'path': str(Path(config['reference']['source_variants']).resolve()),
            'sha256': sha256_file(config['reference']['source_variants']),
        },
        'pymol': {'executable': str(Path(config['plot']['pymol']).resolve()),
                  'version': _pymol_version(config['plot']['pymol'])},
        'outputs': {
            'clone_overview': {'path': str(overview_path), 'sha256': sha256_file(overview_path)},
            'clone_overlay': {'path': str(overlay_path), 'sha256': sha256_file(overlay_path)},
        },
        'auxiliary': {
            'mutation_sites': {'path': str(mutation_sites_path),
                               'sha256': sha256_file(mutation_sites_path)},
        },
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'limits': 'Predicted full-length monomers; NMNAT 4-181 alignment; no activity, '
                  'stability, expression, or ranking claim.',
    }
    identity_path = output / 'render_identity.json'
    write_json(identity_path, identity)
    return {'status': 'COMPLETED', 'output': str(output),
            'render_identity': str(identity_path)}


def validate_clone_demo_identity(render_input_path, script_path, identity_path, expected_outputs):
    data = json.loads(Path(identity_path).read_text())
    if data.get('status') != 'COMPLETED' or data.get('mode') != 'static':
        raise ValueError('Clone demo identity is not a completed static render')
    if data.get('render_input_sha256') != sha256_file(render_input_path):
        raise ValueError('Clone demo identity does not match render input')
    if data.get('script_sha256') != sha256_file(script_path):
        raise ValueError('Clone demo identity does not match render script')
    outputs = data.get('outputs', {})
    if set(outputs) != set(expected_outputs):
        raise ValueError('Clone demo identity output set differs')
    for key in expected_outputs:
        path = Path(outputs[key]['path'])
        if not path.is_file():
            raise FileNotFoundError(path)
        if outputs[key]['sha256'] != sha256_file(path):
            raise ValueError(f'Clone demo output hash differs: {key}')
    return data
