from pathlib import Path

import pytest
import torch

from natvqvae.model import build_model, load_checkpoint

CHECKPOINT = Path('data/weights/qvae/seed8301/best.pt')


@pytest.mark.skipif(not CHECKPOINT.exists(), reason='current QVAE checkpoint not present')
def test_current_qvae_checkpoint_loads():
    payload = load_checkpoint(CHECKPOINT, device='cpu')
    model = build_model(payload, device='cpu')
    assert payload['arm'] == 'QVAE'
    assert payload['seed'] == 8301
    assert payload['model_config'] == {'hidden': 256, 'factors': 16, 'states': 8, 'feature_dim': 1280}
    assert len(payload['state_dict']) == 79
    assert sum(parameter.numel() for parameter in model.parameters()) == 5_333_268
