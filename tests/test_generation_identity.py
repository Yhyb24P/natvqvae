import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest
import numpy as np

from natvqvae import generate as generation_module
from natvqvae.data import Record, sha256_file, stable_seed
from natvqvae.model import AA, ConditionalModel, sample_letters


@pytest.mark.parametrize('arm', ['QVAE', 'deterministic'])
def test_generation_identity_records_actual_overrides_without_changing_draws(tmp_path, monkeypatch, arm):
    inputs = {}
    for name in ('config', 'sources', 'target', 'population', 'training', 'validation', 'test'):
        path = tmp_path / name
        path.write_text(name)
        inputs[name] = str(path)
    esm = tmp_path / 'esm'
    esm.mkdir()
    for name in ('config.json', 'model.safetensors', 'vocab.txt'):
        (esm / name).write_text(name)
    wt = (AA * 18)[:341]
    setting = {'seed': 8401, 'masks_per_condition': 1, 'draws_per_mask': 64,
               'temperature': 1.0, 'regions_1based': {'NMNAT': [4, 181]},
               'mask_sizes': {'NMNAT': [8]}}
    sources = {'population': inputs['population'], 'esm_model': str(esm),
               'splits': {key: inputs[key] for key in ('training', 'validation', 'test')}}
    data = SimpleNamespace(study={'generation': setting}, sources=sources,
                           target=SimpleNamespace(sequence=wt))
    monkeypatch.setattr(generation_module, 'PopulationData', lambda *args: data)

    encoder_loads = []

    class TinyEncoder:
        def __init__(self, *args, **kwargs):
            encoder_loads.append(1)

        def __call__(self, sequences, masked_positions):
            length = len(sequences[0])
            valid = torch.ones(len(sequences), length, dtype=torch.bool)
            mask = torch.zeros_like(valid)
            for index, positions in enumerate(masked_positions):
                mask[index, positions] = True
            return torch.zeros(len(sequences), length, 4), valid, mask

    monkeypatch.setattr(generation_module, 'FrozenEncoder', TinyEncoder)
    torch.manual_seed(32)
    model = ConditionalModel(arm, hidden=4, factors=2, states=2, feature_dim=4)
    checkpoint = tmp_path / 'best.pt'
    torch.save({'state_dict': model.state_dict(), 'arm': arm, 'seed': 9201, 'step': 7,
                'model_config': {'hidden': 4, 'factors': 2, 'states': 2, 'feature_dim': 4}}, checkpoint)
    outputs = [tmp_path / 'first', tmp_path / 'second']
    rows = [generation_module.generate(inputs['config'], inputs['sources'], inputs['target'],
                                       checkpoint, output, device='cpu',
                                       group_filter='NMNAT_m08_r00', draws=2)
            for output in outputs]
    assert rows[0] == rows[1]
    identity = json.loads((outputs[0] / 'run_identity.json').read_text())
    assert identity['generation']['draws_per_mask'] == 2
    assert setting['draws_per_mask'] == 64
    assert identity['draws_override'] == 2 and identity['group_filter'] == 'NMNAT_m08_r00'
    assert identity['target_sha256'] == hashlib.sha256(wt.encode()).hexdigest()
    assert identity['checkpoint_seed'] == 9201 and identity['checkpoint_step'] == 7
    assert identity['input_sha256'][str(checkpoint)] == sha256_file(checkpoint)
    assert identity['checkpoint_sha256'] == sha256_file(checkpoint)
    assert identity['masks_sha256'] == sha256_file(outputs[0] / 'masks.json')
    assert identity['esm_files_sha256']['model.safetensors'] == sha256_file(esm / 'model.safetensors')
    assert identity['code_sha256']['generate.py'] == sha256_file(Path(generation_module.__file__))
    assert len(identity['git_revision']) == 40
    expected_prior = stable_seed(8401, 9201, 'NMNAT_m08_r00', 'latent') if arm == 'QVAE' else None
    assert identity['checkpoint_arm'] == arm
    masks = json.loads((outputs[0] / 'masks.json').read_text())[0]['positions']
    assert all(row['arm'] == arm and len(row['sequence']) == len(wt) and
               all(aa == wt[index] for index, aa in enumerate(row['sequence']) if index not in masks)
               for row in rows[0])
    if arm == 'deterministic':
        assert all(row['log_prior_z'] is None and row['z_codes'] == 'null' for row in rows[0])
    assert all(row['prior_seed'] == expected_prior and
               row['uniform_seed'] == stable_seed(8401, 9201, 'NMNAT_m08_r00', 'residues') for row in rows[0])
    setting['mask_sizes']['NMNAT'] = [8, 16]
    mixed = generation_module.generate(inputs['config'], inputs['sources'], inputs['target'],
                                       checkpoint, tmp_path / 'mixed', device='cpu', draws=2)
    assert len(mixed) == 4
    assert {row['mask_size'] for row in mixed} == {8, 16}
    assert all(row['Hamming'] <= row['mask_size'] for row in mixed)
    if arm == 'QVAE':
        marginal_runs = [generation_module.generate(
            inputs['config'], inputs['sources'], inputs['target'], checkpoint,
            tmp_path / f'marginal_{repeat}', device='cpu', draws=2, distribution='site_marginal')
            for repeat in range(2)]
        assert marginal_runs[0] == marginal_runs[1]
        for row, joint in zip(marginal_runs[0], mixed):
            codes = json.loads(row['z_codes'])
            assert len(codes) == row['mask_size'] and all(len(code) == 2 for code in codes)
            assert row['uniform_seed'] == joint['uniform_seed']
            assert row['prior_seed'] != joint['prior_seed']
            assert row['distribution'] == 'site_marginal'
            assert row['Hamming'] <= row['mask_size']
            assert row['candidate_id'] != joint['candidate_id']
        row = marginal_runs[0][0]
        positions = json.loads((tmp_path / 'marginal_0/masks.json').read_text())[0]['positions']
        valid = torch.ones(1, len(wt), dtype=torch.bool)
        mask = torch.zeros_like(valid)
        mask[0, positions] = True
        codes = json.loads(row['z_codes'])
        with torch.no_grad():
            context = model.encode_context(torch.zeros(1, len(wt), 4), valid, mask)
            targets = context[:, positions]
            # Independently decode each code and select only its intended site.
            reference = torch.stack([model.decode_codes(targets, torch.tensor([[code]]))[0, 0, site]
                                     for site, code in enumerate(codes)]).numpy()
        uniforms = np.random.default_rng(row['uniform_seed']).random((2, len(positions)))[0]
        expected_letters = sample_letters(reference, uniforms)
        assert [row['sequence'][position] for position in positions] == list(expected_letters)
    else:
        with pytest.raises(ValueError, match='already factorized'):
            generation_module.generate(inputs['config'], inputs['sources'], inputs['target'],
                                       checkpoint, tmp_path / 'rejected', device='cpu',
                                       distribution='site_marginal')
    from natvqvae import test_data as test_module
    restored = tmp_path / 'restored'
    restored.mkdir()
    for name in ('test.csv', 'test.fasta', 'manifest.json', 'historical_use.json'):
        (restored / name).write_text(name)
    sequence = wt[:220]
    uid = hashlib.sha256(sequence.encode()).hexdigest()
    record = Record(uid, sequence, 'heldout_group', len(sequence), 10, 160, 180, 210)
    monkeypatch.setattr(test_module, 'load_recovered_test', lambda *args: [record])
    setting['draws_per_mask'] = 2
    manifest = generation_module.prepare_generation_backgrounds(
        inputs['config'], inputs['sources'], inputs['target'], restored, tmp_path / 'backgrounds')
    assert len(manifest['backgrounds']) == 1
    assert manifest['prediction_scores_used'] is False
    background = generation_module.generate(
        inputs['config'], inputs['sources'], inputs['target'], checkpoint, tmp_path / 'heldout',
        device='cpu', draws=2, background_id=uid, test_data=restored)
    fixed_masks = {row['id']: row['positions'] for row in manifest['backgrounds'][0]['masks']}
    for row in background:
        assert len(row['sequence']) == 220 and row['candidate_id'].startswith(uid)
        positions = fixed_masks[row['group']]
        assert min(positions) >= 9 and max(positions) < 160
        assert all(aa == sequence[i] for i, aa in enumerate(row['sequence']) if i not in positions)
    metadata = json.loads((tmp_path / 'heldout/run_identity.json').read_text())
    assert metadata['target_sha256'] == uid
    assert metadata['background_component'] == 'heldout_group'
    assert metadata['generation']['seed'] == stable_seed(8401, 'background', uid)
    distributions = ['joint', 'site_marginal'] if arm == 'QVAE' else ['joint']
    for distribution in distributions:
        loads_before = len(encoder_loads)
        batch_output = tmp_path / f'batch_{distribution}'
        result = generation_module.generate(
            inputs['config'], inputs['sources'], inputs['target'], checkpoint, batch_output,
            device='cpu', draws=2, distribution=distribution, test_data=restored,
            backgrounds_manifest=tmp_path / 'backgrounds/backgrounds.json')
        assert len(encoder_loads) == loads_before + 1
        assert result['backgrounds'] == 2 and result['attempts'] == 8
        for name, selected_id in [('KVP40', None), (uid, uid)]:
            single = tmp_path / f'single_{distribution}_{name}'
            generation_module.generate(
                inputs['config'], inputs['sources'], inputs['target'], checkpoint, single,
                device='cpu', draws=2, distribution=distribution, background_id=selected_id,
                test_data=restored if selected_id else None)
            for filename in ('candidates.csv', 'candidates.fasta', 'masks.json'):
                assert (batch_output / name / filename).read_bytes() == (single / filename).read_bytes()
            metadata = json.loads((batch_output / name / 'run_identity.json').read_text())
            assert metadata['checkpoint_sha256'] == sha256_file(checkpoint)
        with pytest.raises(FileExistsError):
            generation_module.generate(
                inputs['config'], inputs['sources'], inputs['target'], checkpoint, batch_output,
                test_data=restored, backgrounds_manifest=tmp_path / 'backgrounds/backgrounds.json')
    with pytest.raises(ValueError, match='settings differ'):
        generation_module.generate(
            inputs['config'], inputs['sources'], inputs['target'], checkpoint, tmp_path / 'bad_batch',
            draws=3, test_data=restored, backgrounds_manifest=tmp_path / 'backgrounds/backgrounds.json')
    (restored / 'test.fasta').write_text('modified')
    with pytest.raises(ValueError, match='input hashes differ'):
        generation_module.generate(
            inputs['config'], inputs['sources'], inputs['target'], checkpoint, tmp_path / 'changed_batch',
            test_data=restored, backgrounds_manifest=tmp_path / 'backgrounds/backgrounds.json')
