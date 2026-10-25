import hashlib
import json
from pathlib import Path

import pytest

from natvqvae.epcr_demo import _load_demo, validate_clone_demo_identity

ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_clone_demo_config_matches_models():
    config_path, config, models = _load_demo(ROOT / 'config/epcr_clone_demo.yaml')
    assert config_path.is_file()
    assert set(models) == {'WT', 'NatV10', 'NatV89'}
    assert models['WT']['alignment_rmsd_A'] == 0.0
    assert 0.0 < models['NatV10']['alignment_rmsd_A'] < 2.0
    assert 0.0 < models['NatV89']['alignment_rmsd_A'] < 2.0
    assert [item['name'] for item in config['mutations']['NatV10']] == [
        'D83V', 'M162V', 'L317P', 'M323K']
    assert [item['name'] for item in config['mutations']['NatV89']] == [
        'I21N', 'F137Y', 'I156T', 'P301S', 'V320E']


def test_clone_demo_identity_validation(tmp_path):
    render_input = tmp_path / 'render_input.json'
    render_input.write_text('{}')
    script = tmp_path / 'epcr_demo.py'
    script.write_text('print(1)\n')
    overview = tmp_path / 'clone_overview.png'
    overlay = tmp_path / 'clone_overlay.png'
    overview.write_bytes(b'overview')
    overlay.write_bytes(b'overlay')
    identity = tmp_path / 'render_identity.json'
    data = {
        'status': 'COMPLETED', 'mode': 'static',
        'render_input_sha256': _sha(render_input), 'script_sha256': _sha(script),
        'outputs': {
            'clone_overview': {'path': str(overview), 'sha256': _sha(overview)},
            'clone_overlay': {'path': str(overlay), 'sha256': _sha(overlay)},
        },
    }
    identity.write_text(json.dumps(data))
    assert validate_clone_demo_identity(
        render_input, script, identity, {'clone_overview', 'clone_overlay'})['mode'] == 'static'

    bad = dict(data)
    bad['outputs'] = {**data['outputs'],
                      'identity': {'path': str(identity), 'sha256': _sha(identity)}}
    identity.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        validate_clone_demo_identity(
            render_input, script, identity, {'clone_overview', 'clone_overlay'})

    bad = dict(data)
    bad['render_input_sha256'] = '0' * 64
    identity.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        validate_clone_demo_identity(
            render_input, script, identity, {'clone_overview', 'clone_overlay'})

    overlay.unlink()
    identity.write_text(json.dumps(data))
    with pytest.raises(FileNotFoundError):
        validate_clone_demo_identity(
            render_input, script, identity, {'clone_overview', 'clone_overlay'})
