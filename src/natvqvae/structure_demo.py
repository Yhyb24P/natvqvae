"""Summarize existing design structures for a reproducible, non-ranking demo."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from .data import load_yaml, sha256_file, write_json

CANDIDATE_NAMES = (
    'WT', 'A1', 'A2', 'A3', 'B1', 'B2',
    'A1B1', 'A1B2', 'A2B1', 'A2B2', 'A3B1', 'A3B2',
)
REGIONS = ('full', 'NMNAT', 'Nudix')


def _git_head():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).resolve().parents[2], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _read_candidates(path):
    rows = list(csv.DictReader(open(path, newline='')))
    if len(rows) != len(CANDIDATE_NAMES):
        raise ValueError(f'Expected {len(CANDIDATE_NAMES)} design candidates, found {len(rows)}')
    by_name = {row['name']: row for row in rows}
    if set(by_name) != set(CANDIDATE_NAMES):
        raise ValueError('Design candidate names differ from the expected 12')
    for name, row in by_name.items():
        if len(row['sequence']) != 341:
            raise ValueError(f'Candidate {name} is not 341 aa')
        if hashlib.sha256(row['sequence'].encode()).hexdigest() != row['sequence_sha256']:
            raise ValueError(f'Candidate {name} sequence SHA differs')
    return by_name


def _read_pairs(path):
    rows = list(csv.DictReader(open(path, newline='')))
    expected = {'left', 'right', 'region', 'median_rmsd_A', 'min_rmsd_A',
                'max_rmsd_A', 'model_pairs'}
    if not rows or set(rows[0]) != expected:
        raise ValueError('Unexpected structure_pairs.csv columns')
    if len(rows) != 69:
        raise ValueError(f'Expected 69 comparison rows, found {len(rows)}')
    by_pair = {}
    for row in rows:
        key = (row['left'], row['right'], row['region'])
        if key in by_pair:
            raise ValueError(f'Duplicate structure comparison: {key}')
        if row['region'] not in REGIONS:
            raise ValueError(f'Unknown region: {row["region"]}')
        if int(row['model_pairs']) != 9:
            raise ValueError(f'Expected 9 model pairs: {key}')
        by_pair[key] = {
            'left': row['left'], 'right': row['right'], 'region': row['region'],
            'median': float(row['median_rmsd_A']),
            'min': float(row['min_rmsd_A']),
            'max': float(row['max_rmsd_A']),
            'model_pairs': int(row['model_pairs']),
        }
    comparisons = {(key[0], key[1]) for key in by_pair}
    if len(comparisons) != 23:
        raise ValueError(f'Expected 23 candidate comparisons, found {len(comparisons)}')
    return rows, by_pair, comparisons


def _read_summary(path):
    data = json.load(open(path))
    self_rows = data.get('self_rmsd', [])
    if len(self_rows) != 36:
        raise ValueError(f'Expected 36 self-RMSD rows, found {len(self_rows)}')
    self_by = {}
    for row in self_rows:
        key = (row['name'], row['region'])
        if key in self_by:
            raise ValueError(f'Duplicate self-RMSD row: {key}')
        if int(row['model_pairs']) != 3:
            raise ValueError(f'Expected 3 within-model pairs: {key}')
        self_by[key] = {
            'median': float(row['median_rmsd_A']),
            'min': float(row['min_rmsd_A']),
            'max': float(row['max_rmsd_A']),
            'model_pairs': int(row['model_pairs']),
        }
    if set(name for name, _ in self_by) != set(CANDIDATE_NAMES):
        raise ValueError('Self-RMSD candidates differ from the expected 12')
    confidence = data.get('confidence', {})
    if set(confidence) != set(CANDIDATE_NAMES):
        raise ValueError('Confidence candidates differ from the expected 12')
    for name, item in confidence.items():
        for key in ('complex_plddt', 'ptm'):
            value = float(item[key]['median'])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f'{name} {key} median is outside 0-1')
    return data, self_by, confidence


def _primary_comparison(candidate, candidate_row):
    if candidate == 'WT':
        return None
    if candidate_row['group'] in ('A', 'B'):
        return 'WT'
    parent = candidate_row.get('parent_a', '')
    if not parent:
        raise ValueError(f'A+B candidate {candidate} has no parent_a')
    return parent


def _comparison_values(pair_by, left, right):
    if right is None:
        return {region: None for region in REGIONS}
    values = {}
    for region in REGIONS:
        key = (left, right, region)
        if key not in pair_by:
            raise ValueError(f'Missing comparison: {key}')
        values[region] = pair_by[key]
    return values


def _fmt(value):
    return 'NA' if value is None else f'{value:.6f}'


def _candidate_table(candidates, pair_by, self_by, confidence):
    table = []
    for name in CANDIDATE_NAMES:
        row = candidates[name]
        parent = _primary_comparison(name, row)
        comparison = _comparison_values(pair_by, name, parent)
        self_values = {region: self_by[(name, region)] for region in REGIONS}
        conf = confidence[name]
        item = {
            'candidate': name,
            'group': row['group'],
            'parent': parent or '',
            'primary_comparison': 'self' if parent is None else parent,
            'comparison_full_median_A': _fmt(comparison['full']['median'] if comparison['full'] else None),
            'comparison_full_min_A': _fmt(comparison['full']['min'] if comparison['full'] else None),
            'comparison_full_max_A': _fmt(comparison['full']['max'] if comparison['full'] else None),
            'comparison_full_model_pairs': comparison['full']['model_pairs'] if comparison['full'] else 'NA',
            'comparison_nmnat_median_A': _fmt(comparison['NMNAT']['median'] if comparison['NMNAT'] else None),
            'comparison_nmnat_min_A': _fmt(comparison['NMNAT']['min'] if comparison['NMNAT'] else None),
            'comparison_nmnat_max_A': _fmt(comparison['NMNAT']['max'] if comparison['NMNAT'] else None),
            'comparison_nmnat_model_pairs': comparison['NMNAT']['model_pairs'] if comparison['NMNAT'] else 'NA',
            'comparison_nudix_median_A': _fmt(comparison['Nudix']['median'] if comparison['Nudix'] else None),
            'comparison_nudix_min_A': _fmt(comparison['Nudix']['min'] if comparison['Nudix'] else None),
            'comparison_nudix_max_A': _fmt(comparison['Nudix']['max'] if comparison['Nudix'] else None),
            'comparison_nudix_model_pairs': comparison['Nudix']['model_pairs'] if comparison['Nudix'] else 'NA',
            'self_full_median_A': _fmt(self_values['full']['median']),
            'self_full_min_A': _fmt(self_values['full']['min']),
            'self_full_max_A': _fmt(self_values['full']['max']),
            'self_full_model_pairs': self_values['full']['model_pairs'],
            'self_nmnat_median_A': _fmt(self_values['NMNAT']['median']),
            'self_nmnat_min_A': _fmt(self_values['NMNAT']['min']),
            'self_nmnat_max_A': _fmt(self_values['NMNAT']['max']),
            'self_nmnat_model_pairs': self_values['NMNAT']['model_pairs'],
            'self_nudix_median_A': _fmt(self_values['Nudix']['median']),
            'self_nudix_min_A': _fmt(self_values['Nudix']['min']),
            'self_nudix_max_A': _fmt(self_values['Nudix']['max']),
            'self_nudix_model_pairs': self_values['Nudix']['model_pairs'],
            'plddt_median_0_1': f"{float(conf['complex_plddt']['median']):.6f}",
            'ptm_median_0_1': f"{float(conf['ptm']['median']):.6f}",
            'notes': _candidate_note(name),
        }
        table.append(item)
    return table


def _candidate_note(name):
    if name == 'WT':
        return 'reference; self-RMSD from 3 model pairs'
    if name == 'A2':
        return 'NMNAT self-dispersion is model uncertainty, not flexibility or elimination evidence'
    if name == 'A3':
        return 'A134G rigid template transfer has severe clashes; proximity is not a real contact'
    if name.startswith('A') and 'B' in name:
        return 'A+B vs parent A; WT and B-parent comparisons are in comparison_summary.csv'
    if name in ('B1', 'B2'):
        return 'metal-environment hypothesis; not proof of reduced hydrolysis'
    return 'design-site structure comparison; no activity or ranking evidence'


def _write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_comparison_summary(path, pair_rows, candidate_rows):
    fields = list(pair_rows[0]) + ['is_primary', 'comparison_scope']
    output = []
    for row in pair_rows:
        left, right = row['left'], row['right']
        if right == 'WT' and candidate_rows[left]['group'] in ('A', 'B'):
            primary, scope = True, 'candidate_vs_WT'
        elif left in ('A1B1', 'A1B2', 'A2B1', 'A2B2', 'A3B1', 'A3B2') and right == candidate_rows[left]['parent_a']:
            primary, scope = True, 'A_plus_B_vs_parent_A'
        elif left in ('A1B1', 'A1B2', 'A2B1', 'A2B2', 'A3B1', 'A3B2') and right == 'WT':
            primary, scope = False, 'A_plus_B_vs_WT'
        else:
            primary, scope = False, 'A_plus_B_vs_B_parent'
        output.append({**row, 'is_primary': primary, 'comparison_scope': scope})
    _write_csv(path, fields, output)


def _write_mutation_map(path, mutations):
    fields = ['position', 'wt', 'alt', 'domain', 'candidates', 'hypothesis', 'caveat']
    rows = [{**item, 'candidates': ';'.join(item['candidates'])} for item in mutations]
    _write_csv(path, fields, rows)


def _write_reference_sites(path, references):
    fields = ['position', 'wt', 'label']
    _write_csv(path, fields, references)


def _parent_comparison(path, pair_by):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    groups = {
        'A1 parent': ('A1B1', 'A1B2'),
        'A2 parent': ('A2B1', 'A2B2'),
        'A3 parent': ('A3B1', 'A3B2'),
    }
    region_colors = {'full': '#666666', 'NMNAT': '#2b62d9', 'Nudix': '#1f9d55'}
    labels = [name for variants in groups.values() for name in variants]
    x = list(range(len(labels)))
    width = 0.25
    figure, axis = plt.subplots(figsize=(9.5, 4.8))
    all_values = []
    for offset, region in zip((-width, 0.0, width), REGIONS):
        values = []
        for variants in groups.values():
            parent = variants[0][:2]
            for name in variants:
                values.append(pair_by[(name, parent, region)]['median'])
        all_values.extend(values)
        axis.bar([value + offset for value in x], values, width=width,
                 label=region, color=region_colors[region])
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylabel('median C-alpha RMSD (A)')
    axis.set_title('A+B vs parent A: median C-alpha RMSD over 9 model pairs')
    axis.set_ylim(0, max(all_values) * 1.18)
    axis.grid(axis='y', alpha=0.25)
    axis.legend(frameon=False, ncol=3, loc='upper left')
    figure.text(0.5, 0.01, 'descriptive; 9 model pairs are not independent seeds; min-max is not a CI',
                ha='center', va='bottom', fontsize=7, color='#444444')
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _ca_coords(path, lo, hi):
    import gemmi

    structure = gemmi.read_structure(str(path))
    coords = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if lo <= residue.seqid.num <= hi:
                    atom = residue.find_atom('CA', '*')
                    if atom is not None:
                        coords[residue.seqid.num] = np.array(
                            [atom.pos.x, atom.pos.y, atom.pos.z])
        break
    return coords


def _alignment_transform(candidate, reference):
    keys = sorted(set(candidate) & set(reference))
    if len(keys) < 100:
        raise ValueError('Too few common C-alpha residues for alignment')
    p = np.array([candidate[key] for key in keys])
    q = np.array([reference[key] for key in keys])
    pc, qc = p.mean(0), q.mean(0)
    covariance = (p - pc).T @ (q - qc)
    u, _, vt = np.linalg.svd(covariance)
    determinant = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, determinant]) @ u.T
    translation = qc - rotation @ pc
    matrix = []
    for row in range(3):
        matrix.append([float(rotation[row, 0]), float(rotation[row, 1]),
                       float(rotation[row, 2]), float(translation[row] * 0.1)])
    matrix.append([0.0, 0.0, 0.0, 1.0])
    return matrix


def _alignment_transforms(cifs):
    reference = _ca_coords(cifs['WT'], 4, 181)
    transforms = {}
    for name in ('A1', 'A2', 'A3'):
        candidate = _ca_coords(cifs[name], 4, 181)
        transforms[name] = _alignment_transform(candidate, reference)
    return transforms


def _joint_ligands(path):
    import gemmi

    structure = gemmi.read_structure(str(path))
    ligands = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.het_flag != 'H' or residue.name in ('HOH', 'WAT', 'DOD'):
                    continue
                positions = [atom.pos for atom in residue]
                if not positions:
                    continue
                center = [sum(getattr(pos, axis) for pos in positions) / len(positions)
                          for axis in ('x', 'y', 'z')]
                ligands.append({'name': residue.name, 'chain': chain.name,
                                'center_blender_units': [round(value * 0.1, 4) for value in center]})
    return ligands


def _resolve_cifs(config):
    root = Path(config['design_root']).resolve()
    model = int(config['model_index'])
    paths = {}
    for name in CANDIDATE_NAMES:
        path = root / name / f'boltz_results_{name}' / 'predictions' / name / f'{name}_model_{model}.cif'
        if not path.exists():
            raise FileNotFoundError(path)
        paths[name] = path
    joint = Path(config['joint']['cif']).resolve()
    if not joint.exists():
        raise FileNotFoundError(joint)
    return paths, joint


def summarize_structure_demo(config_path, output):
    config = load_yaml(config_path)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Demo output already contains files: {output}')
    output.mkdir(parents=True, exist_ok=True)

    candidates = _read_candidates(config['candidates'])
    pair_rows, pair_by, comparisons = _read_pairs(config['structure_pairs'])
    summary_source, self_by, confidence = _read_summary(config['structure_summary'])
    cifs, joint_cif = _resolve_cifs(config)

    candidate_rows = _candidate_table(candidates, pair_by, self_by, confidence)
    candidate_fields = list(candidate_rows[0])
    _write_csv(output / 'candidate_structure_summary.csv', candidate_fields, candidate_rows)
    _write_comparison_summary(output / 'comparison_summary.csv', pair_rows, candidates)
    _write_mutation_map(output / 'mutation_map.csv', config['mutations'])
    _write_reference_sites(output / 'reference_sites.csv', config.get('references', []))
    _parent_comparison(output / 'parent_comparison.png', pair_by)

    render_input = {
        'wt_cif': str(cifs['WT']),
        'candidate_cifs': {name: str(path) for name, path in cifs.items()},
        'candidate_edits': {name: (row['edits'] or 'WT') for name, row in candidates.items()},
        'align_transforms': _alignment_transforms(cifs),
        'model_index': int(config['model_index']),
        'domains': config['domains'],
        'colors': config['colors'],
        'marker_colors': config['marker_colors'],
        'mutations': config['mutations'],
        'references': config.get('references', []),
        'joint': {**config['joint'], 'cif': str(joint_cif),
                  'ligands': _joint_ligands(joint_cif)},
        'render': config['render'],
        'outputs': {
            'wt_overview': str(output / 'wt_overview.png'),
            'a_class_panels': str(output / 'a_class_panels.png'),
            'nudix_site': str(output / 'nudix_site.png'),
            'nmnat_inset': str(output / 'nmnat_inset.png'),
            'animation': str(output / 'demo.mp4'),
            'blend': str(output / 'demo.blend'),
            'identity': str(output / 'render_identity.json'),
        },
    }
    write_json(output / 'render_input.json', render_input)

    input_paths = [Path(config['candidates']), Path(config['structure_pairs']),
                   Path(config['structure_summary']), joint_cif, *cifs.values()]
    summary = {
        'status': 'COMPLETED',
        'task': config['task'],
        'model_index': int(config['model_index']),
        'counts': {
            'candidates': len(CANDIDATE_NAMES),
            'structure_models': len(CANDIDATE_NAMES) * 3,
            'comparisons': len(comparisons),
            'comparison_rows': len(pair_rows),
            'self_rmsd_rows': len(summary_source['self_rmsd']),
        },
        'metric_definitions': {
            'comparison_rmsd': 'C-alpha RMSD after least-squares alignment; median/min/max over 9 model pairs (3 candidate samples x 3 reference samples).',
            'self_rmsd': 'Median/min/max over 3 within-candidate model pairs; descriptive only.',
            'model_pairs': 'Model pairs are not independent seeds, replicates, or a confidence interval.',
            'confidence': 'complex_plddt and ptm are 0-1 Boltz-2 confidence medians; not activity, stability, or candidate ranking.',
            'range': 'min-max is the observed pair range, not a confidence interval.',
        },
        'candidate_summary': candidate_rows,
        'comparisons': [
            {'left': left, 'right': right,
             'primary': (right == 'WT' and candidates[left]['group'] in ('A', 'B'))
                        or (left in ('A1B1', 'A1B2', 'A2B1', 'A2B2', 'A3B1', 'A3B2')
                            and right == candidates[left]['parent_a'])}
            for left, right in sorted(comparisons)
        ],
        'mutation_map': str(output / 'mutation_map.csv'),
        'reference_sites': str(output / 'reference_sites.csv'),
        'parent_comparison': str(output / 'parent_comparison.png'),
        'render_input': str(output / 'render_input.json'),
        'inputs': {str(path): sha256_file(path) for path in input_paths},
        'source_structure_summary': str(Path(config['structure_summary'])),
        'source_structure_summary_sha256': sha256_file(config['structure_summary']),
        'git_head': _git_head(),
        'module_sha256': sha256_file(__file__),
        'config_sha256': sha256_file(config_path),
        'notes': {
            'structures': 'Predicted Boltz-2 monomers; no ligands in design models; no new predictions.',
            'A2': 'NMNAT self-dispersion is model uncertainty, not flexibility or elimination evidence.',
            'A3': 'A134G rigid template transfer has severe clashes; proximity is not a real contact.',
            'B1_B2': 'Metal-environment hypotheses; not proof of reduced hydrolysis or NAD+ net accumulation.',
            'NMNAT_joint': config['joint']['source'],
        },
    }
    write_json(output / 'summary.json', summary)
    return output




def validate_render_identity(render_input_path, script_path, identity_path, expected_outputs):
    """Validate that an identity belongs to this input, script and output set."""
    identity_path = Path(identity_path)
    if not identity_path.exists():
        raise FileNotFoundError(identity_path)
    data = json.loads(identity_path.read_text())
    if data.get('status') != 'COMPLETED':
        raise ValueError('Render identity is not COMPLETED')
    if data.get('render_input_sha256') != sha256_file(render_input_path):
        raise ValueError('Render identity does not match render_input.json')
    if data.get('script_sha256') != sha256_file(script_path):
        raise ValueError('Render identity does not match blender_demo.py')
    outputs = data.get('outputs', {})
    if 'identity' in outputs:
        raise ValueError('Render identity must not include itself in outputs')
    for key in expected_outputs:
        if key not in outputs:
            raise ValueError(f'Render identity is missing output: {key}')
        path = Path(outputs[key]['path'])
        if not path.exists():
            raise FileNotFoundError(path)
        if outputs[key].get('sha256') != sha256_file(path):
            raise ValueError(f'Render output hash differs: {key}')
    if data.get('mode') == 'stills_only' and 'animation' in outputs:
        raise ValueError('A stills-only identity must not claim an animation')
    if data.get('mode') != 'stills_only' and 'animation' not in outputs:
        raise ValueError('A full render identity must include the animation')
    return data


def render_structure_demo(config_path, output):
    config = load_yaml(config_path)
    output = Path(output)
    render_input = output / 'render_input.json'
    if not render_input.exists():
        raise FileNotFoundError(f'Run structure-demo-summary first: {render_input}')
    executable = Path(config['blender']['executable'])
    if not executable.exists():
        raise FileNotFoundError(executable)
    module = Path(__file__).with_name('blender_demo.py')
    identity = output / 'render_identity.json'
    identity.unlink(missing_ok=True)
    subprocess.run(
        [str(executable), '-b', '--python-exit-code', '1', '--python', str(module), '--',
         '--config', str(render_input), '--output', str(output)],
        check=True,
    )
    expected = {'wt_overview', 'a_class_panels', 'nudix_site', 'nmnat_inset',
                'blend', 'blend_session', 'animation'}
    data = validate_render_identity(render_input, module, identity, expected)
    return {'status': 'COMPLETED', 'output': str(output),
            'render_identity': str(identity), 'mode': data.get('mode')}
