"""Categorical NatV model, frozen ESM encoder, and checkpoint loading."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

AA = 'ACDEFGHIKLMNPQRSTVWY'


def sample_letters(logits, uniforms, temperature=1.0):
    scores = np.asarray(logits, dtype=np.float64)
    if temperature <= 0 or scores.shape[-1] != 20 or not np.isfinite(scores).all():
        raise ValueError('Invalid logits or temperature')
    scores = scores / temperature
    u = np.asarray(uniforms)
    if u.shape != scores.shape[:-1] or ((u < 0) | (u >= 1)).any():
        raise ValueError('Invalid uniforms')
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    p = weights / weights.sum(-1, keepdims=True)
    indices = (u[..., None] >= np.cumsum(p, axis=-1)).sum(-1).clip(max=19)
    return np.asarray(list(AA))[indices]


def categorical_kl(q_logits, p_logits):
    logq = q_logits.float().log_softmax(-1)
    logp = p_logits.float().log_softmax(-1)
    return (logq.exp() * (logq - logp)).sum((-1, -2))


def categorical_samples(logits, count, generator):
    prob = logits.float().softmax(-1)
    u = torch.rand((count,) + prob.shape[:-1], device=logits.device, generator=generator)
    return (u[..., None] >= prob.cumsum(-1)).sum(-1).clamp_max(prob.shape[-1] - 1)


def categorical_log_prob(logits, codes):
    logp = logits.float().log_softmax(-1)
    expanded = logp.unsqueeze(0).expand(len(codes), *logp.shape)
    return expanded.gather(-1, codes[..., None]).squeeze(-1).sum(-1)


def rloo_objective(reconstruction, logq, kl, beta, weights):
    if len(reconstruction) < 2 or reconstruction.shape != logq.shape:
        raise ValueError('RLOO requires at least two independent samples')
    detached = reconstruction.detach()
    baseline = (detached.sum(0, keepdim=True) - detached) / (len(detached) - 1)
    score = ((detached - baseline) * logq).mean(0)
    numeric = reconstruction.mean(0) + beta * kl
    return ((numeric + score - score.detach()) * weights).mean()


class ConditionalModel(nn.Module):
    """QVAE with a full-sequence posterior and a context/mask-conditioned prior."""

    def __init__(self, arm, hidden=256, factors=16, states=8, feature_dim=1280):
        super().__init__()
        if arm not in {'QVAE', 'deterministic'}:
            raise ValueError('Unknown arm')
        self.arm, self.hidden, self.factors, self.states = arm, hidden, factors, states
        self.context_projection = nn.Linear(feature_dim, hidden)
        self.mask_embedding = nn.Embedding(2, hidden)

        def transformer():
            layer = nn.TransformerEncoderLayer(hidden, 4, 4 * hidden, dropout=0.0, batch_first=True)
            return nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)

        self.context_encoder = transformer()
        self.condition_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, factors * states))
        self.decoder_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(2)])
        self.decoder_film = nn.ModuleList([nn.Linear(factors * states, 2 * hidden) for _ in range(2)])
        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Linear(4 * hidden, hidden))
            for _ in range(2)])
        self.output = nn.Linear(hidden, 20)
        if arm == 'QVAE':
            self.full_projection = nn.Linear(feature_dim, hidden)
            self.posterior_encoder = transformer()
            self.posterior_head = nn.Sequential(
                nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, factors * states))

    def _position(self, length, device):
        phase = torch.arange(length, device=device)[:, None] * torch.exp(
            torch.arange(0, self.hidden, 2, device=device) * (-math.log(10000) / self.hidden))
        position = torch.zeros(length, self.hidden, device=device)
        position[:, 0::2], position[:, 1::2] = phase.sin(), phase.cos()
        return position

    @staticmethod
    def _check_mask(valid, mask):
        if mask.shape != valid.shape or (mask & ~valid).any() or not mask.any(1).all():
            raise ValueError('Invalid target mask')
        if not (valid & ~mask).any(1).all():
            raise ValueError('No visible context')

    @staticmethod
    def _pool(h, valid, mask):
        visible = valid & ~mask
        a = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        b = (h * visible[..., None]).sum(1) / visible.sum(1, keepdim=True)
        return torch.cat((a, b), -1)

    def encode_context(self, masked_features, valid, mask):
        self._check_mask(valid, mask)
        if masked_features.requires_grad:
            raise ValueError('Frozen features must not require gradients')
        h = self.context_projection(masked_features.float().masked_fill(~valid[..., None], 0))
        h = h + self._position(h.shape[1], h.device) + self.mask_embedding(mask.long())
        return self.context_encoder(h, src_key_padding_mask=~valid)

    def prior(self, context, valid, mask):
        if self.arm != 'QVAE':
            raise ValueError('Deterministic model has no latent prior')
        return self.condition_head(self._pool(context, valid, mask)).reshape(-1, self.factors, self.states)

    def posterior(self, full_features, context, valid, mask):
        if self.arm != 'QVAE' or full_features is None or full_features.requires_grad:
            raise ValueError('Posterior requires frozen full-sequence features')
        h = self.full_projection(full_features.float().masked_fill(~valid[..., None], 0)) + context
        h = self.posterior_encoder(h, src_key_padding_mask=~valid)
        return self.posterior_head(self._pool(h, valid, mask)).reshape(-1, self.factors, self.states)

    def deterministic_context(self, context, valid, mask):
        if self.arm != 'deterministic':
            raise ValueError('QVAE requires explicit categorical codes')
        return F.gelu(self.condition_head(self._pool(context, valid, mask)))

    def decode(self, targets, latent):
        h = targets.unsqueeze(0)
        for norm, film, block in zip(self.decoder_norms, self.decoder_film, self.decoder_blocks):
            scale, shift = film(latent).chunk(2, -1)
            modulated = norm(h) * (1 + 0.1 * scale.tanh()[..., None, :]) + shift[..., None, :]
            h = h + block(modulated)
        return self.output(h)

    def decode_codes(self, targets, codes):
        latent = F.one_hot(codes, self.states).float().flatten(-2)
        return self.decode(targets, latent)

    def sample_prior(self, context, valid, mask, count, generator):
        return categorical_samples(self.prior(context, valid, mask), count, generator)


def target_tensors(context, mask, labels):
    lengths = mask.sum(1)
    width = int(lengths.max())
    indices = torch.zeros(len(context), width, device=context.device, dtype=torch.long)
    for i in range(len(context)):
        sites = mask[i].nonzero().flatten()
        indices[i, :len(sites)] = sites
    use = torch.arange(width, device=context.device)[None] < lengths[:, None]
    targets = context.gather(1, indices[..., None].expand(-1, -1, context.shape[-1]))
    y = labels.gather(1, indices)
    return targets, y, use, indices


def reconstruction_nll(logits, labels, use):
    logp = logits.float().log_softmax(-1)
    target = labels[None, ..., None].expand(len(logits), -1, -1, 1)
    return -(logp.gather(-1, target).squeeze(-1) * use).sum(-1)


class FrozenEncoder:
    """Frozen ESM-2 masked encoder used by both training and generation."""

    def __init__(self, path, device='cuda'):
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.model = AutoModelForMaskedLM.from_pretrained(
            str(path), local_files_only=True, use_safetensors=True).to(self.device).eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, sequences, positions):
        tokens = self.tokenizer(sequences, padding=True, truncation=False, return_tensors='pt').to(self.device)
        width = max(map(len, sequences))
        sizes = torch.tensor(list(map(len, sequences)), device=self.device)
        if tokens.input_ids.shape[1] != width + 2 or not torch.equal(tokens.attention_mask.sum(1), sizes + 2):
            raise ValueError('Tokenizer changed residue coordinates')
        mask = torch.zeros(len(sequences), width, device=self.device, dtype=torch.bool)
        for i, (seq, sites) in enumerate(zip(sequences, positions)):
            if len(set(sites)) != len(sites) or (sites and (min(sites) < 0 or max(sites) >= len(seq))):
                raise ValueError('Invalid frozen-encoder mask')
            mask[i, sites] = True
        tokens.input_ids[:, 1:width + 1].masked_fill_(mask, self.tokenizer.mask_token_id)
        if self.device.type == 'cuda':
            with torch.autocast('cuda', dtype=torch.bfloat16):
                h = self.model.esm(**tokens).last_hidden_state[:, 1:width + 1]
        else:
            h = self.model.esm(**tokens).last_hidden_state[:, 1:width + 1]
        valid = torch.arange(width, device=self.device)[None] < sizes[:, None]
        return h, valid, mask

    @torch.no_grad()
    def amino_acid_log_probs(self, sequences, positions):
        """Use the frozen MLM head on the same masked inputs, normalized over AA."""
        h, valid, mask = self(sequences, positions)
        token_ids = self.tokenizer.convert_tokens_to_ids(list(AA))
        if (len(set(token_ids)) != len(AA) or
                self.tokenizer.convert_ids_to_tokens(token_ids) != list(AA)):
            raise ValueError('ESM vocabulary does not map the 20 amino acids uniquely')
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=self.device.type == 'cuda'):
            logits = self.model.lm_head(h)
        return logits[..., token_ids].float().log_softmax(-1), valid, mask


def load_checkpoint(path, device='cpu'):
    payload = torch.load(path, map_location=device, weights_only=True)
    if 'state_dict' not in payload or 'arm' not in payload:
        raise ValueError('Unsupported checkpoint format')
    return payload


def build_model(payload, device='cpu'):
    config = payload.get('model_config', {})
    allowed = {key: config[key] for key in ('hidden', 'factors', 'states', 'feature_dim') if key in config}
    model = ConditionalModel(payload['arm'], **allowed).to(device).eval()
    model.load_state_dict(payload['state_dict'], strict=True)
    return model
