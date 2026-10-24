import csv
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from natvqvae.structure_demo import (CANDIDATE_NAMES, render_structure_demo,
                                     summarize_structure_demo,
                                     validate_render_identity)

ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _extract(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(gzip.decompress(Path(source).read_bytes()))


def _portable_config(tmp_path):
    config = yaml.safe_load((ROOT / 'config/structure_demo.yaml').read_text())
    design_root = tmp_path / 'design'
    for name in CANDIDATE_NAMES:
        source = (ROOT / 'data/results/evaluation/natv_dual_domain_design/coordinates'
                  / f'{name}_{name}_model_0.cif.gz')
        destination = (design_root / name / f'boltz_results_{name}' / 'predictions'
                       / name / f'{name}_model_0.cif')
        _extract(source, destination)
    joint = tmp_path / 'joint_model_0.cif'
    _extract(ROOT / 'data/results/evaluation/kvp40_nmnat_joint/coordinates'
             / 'ATP_NMN_MG_seed20261026_model_0.cif.gz', joint)
    config['design_root'] = str(design_root)
    config['joint']['cif'] = str(joint)
    for key in ('candidates', 'structure_pairs', 'structure_summary'):
        config[key] = str(ROOT / config[key])
    path = tmp_path / 'structure_demo.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def test_structure_demo_summary_contract(tmp_path):
    output = tmp_path / 'demo'
    config = _portable_config(tmp_path)
    summarize_structure_demo(config, output)
    candidates = list(csv.DictReader(open(output / 'candidate_structure_summary.csv')))
    assert len(candidates) == 12
    wt = next(row for row in candidates if row['candidate'] == 'WT')
    assert wt['comparison_full_median_A'] == 'NA'
    assert wt['self_full_model_pairs'] == '3'
    comparisons = list(csv.DictReader(open(output / 'comparison_summary.csv')))
    assert len(comparisons) == 69
    summary = json.load(open(output / 'summary.json'))
    assert summary['counts'] == {'candidates': 12, 'structure_models': 36,
                                 'comparisons': 23, 'comparison_rows': 69,
                                 'self_rmsd_rows': 36}
    assert sum(item['primary'] for item in summary['comparisons']) == 11
    render_input = json.load(open(output / 'render_input.json'))
    assert set(render_input['align_transforms']) == {'A1', 'A2', 'A3'}
    assert all(str(tmp_path) in path for path in render_input['candidate_cifs'].values())
    assert str(tmp_path) in render_input['joint']['cif']


def test_identity_validation_rejects_self_reference_and_stale_input(tmp_path):
    render_input = tmp_path / 'render_input.json'
    render_input.write_text('{}')
    script = tmp_path / 'script.py'
    script.write_text('print(1)\n')
    output = tmp_path / 'still.png'
    output.write_bytes(b'still')
    identity = tmp_path / 'identity.json'
    data = {
        'status': 'COMPLETED', 'mode': 'stills_only',
        'render_input_sha256': _sha(render_input), 'script_sha256': _sha(script),
        'outputs': {'still': {'path': str(output), 'sha256': _sha(output)}},
    }
    identity.write_text(json.dumps(data))
    assert validate_render_identity(render_input, script, identity, {'still'})['mode'] == 'stills_only'

    bad = dict(data)
    bad['outputs'] = {**data['outputs'],
                      'identity': {'path': str(identity), 'sha256': _sha(identity)}}
    identity.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        validate_render_identity(render_input, script, identity, {'still'})

    bad = dict(data)
    bad['render_input_sha256'] = '0' * 64
    identity.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        validate_render_identity(render_input, script, identity, {'still'})

    output.unlink()
    identity.write_text(json.dumps(data))
    with pytest.raises(FileNotFoundError):
        validate_render_identity(render_input, script, identity, {'still'})


def _fake_config(tmp_path, body):
    executable = tmp_path / 'fake_blender'
    executable.write_text('#!/bin/sh\n' + body + '\n')
    executable.chmod(0o755)
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'blender': {'executable': str(executable)}}))
    return config


def test_render_failure_removes_stale_identity(tmp_path):
    config = _fake_config(tmp_path, 'exit 1')
    output = tmp_path / 'demo'
    output.mkdir()
    (output / 'render_input.json').write_text('{}')
    (output / 'render_identity.json').write_text('{"status": "COMPLETED"}')
    with pytest.raises(subprocess.CalledProcessError):
        render_structure_demo(config, output)
    assert not (output / 'render_identity.json').exists()


def test_render_without_identity_is_not_success(tmp_path):
    config = _fake_config(tmp_path, 'exit 0')
    output = tmp_path / 'demo'
    output.mkdir()
    (output / 'render_input.json').write_text('{}')
    (output / 'render_identity.json').write_text('{"status": "COMPLETED"}')
    with pytest.raises(FileNotFoundError):
        render_structure_demo(config, output)
    assert not (output / 'render_identity.json').exists()
