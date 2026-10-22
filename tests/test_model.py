import torch

from natvqvae.model import (ConditionalModel, categorical_samples, load_checkpoint,
                            build_model, sample_letters, target_tensors)


def test_model_forward_and_latent_shapes():
    torch.manual_seed(0)
    model = ConditionalModel('QVAE', hidden=16, factors=4, states=4, feature_dim=32).eval()
    features = torch.randn(2, 5, 32)
    valid = torch.ones(2, 5, dtype=torch.bool)
    mask = torch.zeros_like(valid)
    mask[0, [1, 3]] = True
    mask[1, [0, 4]] = True
    with torch.no_grad():
        context = model.encode_context(features, valid, mask)
        prior = model.prior(context, valid, mask)
        posterior = model.posterior(features, context, valid, mask)
        targets, _, _, _ = target_tensors(context, mask, torch.zeros(2, 5, dtype=torch.long))
        codes = categorical_samples(prior, 3, torch.Generator().manual_seed(1))
        logits = model.decode_codes(targets, codes)
    assert prior.shape == (2, 4, 4)
    assert posterior.shape == (2, 4, 4)
    assert logits.shape == (3, 2, 2, 20)


def test_sampling_is_reproducible():
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    a = categorical_samples(logits, 4, torch.Generator().manual_seed(7))
    b = categorical_samples(logits, 4, torch.Generator().manual_seed(7))
    assert torch.equal(a, b)
    letters = sample_letters(torch.zeros(2, 3, 20).numpy(), [[0.1, 0.5, 0.9], [0.2, 0.6, 0.8]])
    assert letters.shape == (2, 3)


def test_checkpoint_roundtrip(tmp_path):
    model = ConditionalModel('QVAE', hidden=16, factors=4, states=4, feature_dim=32).eval()
    path = tmp_path / 'checkpoint.pt'
    torch.save({'state_dict': model.state_dict(), 'arm': 'QVAE', 'seed': 1,
                'model_config': {'hidden': 16, 'factors': 4, 'states': 4, 'feature_dim': 32},
                'step': 1}, path)
    payload = load_checkpoint(path, device='cpu')
    loaded = build_model(payload, device='cpu')
    assert loaded.arm == 'QVAE'
    assert set(loaded.state_dict()) == set(model.state_dict())
