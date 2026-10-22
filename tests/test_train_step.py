import torch

from natvqvae.model import ConditionalModel
from natvqvae.train import _train_step


class _BatchData:
    def __init__(self):
        self.context = torch.randn(2, 6, 32)
        self.full = torch.randn(2, 6, 32)
        self.valid = torch.ones(2, 6, dtype=torch.bool)
        self.mask = torch.zeros_like(self.valid)
        self.mask[0, [1, 4]] = True
        self.mask[1, [0, 5]] = True
        self.labels = torch.zeros(2, 6, dtype=torch.long)

    def batch(self, queries, encoder, device, need_full):
        return self.context, self.full if need_full else None, self.valid, self.mask, self.labels


def test_qvae_train_step_is_finite():
    model = ConditionalModel('QVAE', hidden=16, factors=4, states=4, feature_dim=32)
    data = _BatchData()
    weights = torch.ones(2)
    loss, values = _train_step(model, data, [{}, {}], weights, 0.5,
                               torch.Generator().manual_seed(1), 2, None, torch.device('cpu'))
    assert torch.isfinite(loss)
    assert values['target_sites'] == 4
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
