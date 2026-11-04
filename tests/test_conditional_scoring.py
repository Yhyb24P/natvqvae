import pytest
import torch

from natvqvae.evaluate import conditional_scores, evaluate
from natvqvae.esm_baseline import evaluate_esm
from natvqvae.model import ConditionalModel


def test_saved_site_scores_are_marginals_and_preserve_joint_scores():
    torch.manual_seed(31)
    model = ConditionalModel('QVAE', hidden=16, factors=2, states=2, feature_dim=32).eval()
    features = torch.randn(2, 6, 32)
    full = torch.randn_like(features)
    valid = torch.ones(2, 6, dtype=torch.bool)
    mask = torch.zeros_like(valid)
    mask[0, [1, 4]] = True
    mask[1, [0, 2, 5]] = True
    labels = torch.randint(0, 20, (2, 6))
    base = conditional_scores(model, features, full, valid, mask, labels, 8,
                              torch.Generator().manual_seed(13))
    saved = conditional_scores(model, features, full, valid, mask, labels, 8,
                               torch.Generator().manual_seed(13), return_site_log_probs=True)
    for name, value in base.items():
        assert torch.equal(value, saved[name])
    for i, count in enumerate([2, 3]):
        assert torch.allclose(-saved['prior_site_marginal_log_probs'][i, :count].sum(),
                              saved['marginal_nll'][i])
    # There is no additive per-site decomposition asserted for prior_joint_nll.
    assert saved['prior_site_marginal_log_probs'].shape == (2, 3)


@pytest.mark.parametrize('split,test_data', [('test', None), ('validation', 'test_data')])
def test_test_scoring_requires_explicit_and_exclusive_inputs(split, test_data):
    with pytest.raises(ValueError, match='[Tt]est'):
        evaluate('missing', 'missing', 'missing', 'missing', 'missing',
                 split=split, test_data=test_data)
    with pytest.raises(ValueError, match='[Tt]est'):
        evaluate_esm('missing', 'missing', 'missing', 'missing',
                     split=split, test_data=test_data)
