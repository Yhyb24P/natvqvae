"""Reuse the verified WT-column QVAE and an independently trained deterministic control."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import torch
from torch import nn

from .intake import REPO_ROOT, sha256_file
from .sequence import wt_sequence

AA = 'ACDEFGHIKLMNPQRSTVWY'
ALPHABET = AA + '-X'
GAP, UNKNOWN, MASK = 20, 21, 22


def legacy_implementation(source: Path):
    receipt = json.loads((REPO_ROOT / 'discovery/natv_model_inheritance/source_checks.json').read_text())
    for relative in ['src/nad_qvae/design/sequence_generator.py', 'src/nad_qvae/design/sequence_corpus.py']:
        if sha256_file(source / relative) != receipt['inputs'][relative]['sha256']:
            raise ValueError(f'Legacy source differs: {relative}')
    # Read-only imports must not leave bytecode in the external repository.
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source / 'src'))
    try:
        module = importlib.import_module('nad_qvae.design.sequence_generator')
        if Path(module.__file__).resolve() != (source / 'src/nad_qvae/design/sequence_generator.py').resolve():
            raise ValueError('Another nad_qvae installation was imported')
        return module.SequenceGenerator, module.losses
    finally:
        sys.path.pop(0)
        sys.dont_write_bytecode = previous


class AlignedGenerator(nn.Module):
    """Same parameters and decoder; the control infers a deterministic WT-only context.

    The control is optimized independently. It never encodes target tokens and
    has no sampled latent variable or KL term. The VAE's posterior is used only
    for variational training/scoring; generation uses the explicit uniform prior.
    """
    def __init__(self, source, config, wt_embeddings, arm):
        super().__init__()
        if arm not in {'categorical_vae', 'no_latent'}:
            raise ValueError(arm)
        implementation, self._loss = legacy_implementation(Path(source))
        self.network = implementation(config, wt_embeddings, 'categorical_vae')
        self.arm = arm
        self.register_buffer('wt_tokens', torch.tensor([ALPHABET.index(a) for a in wt_sequence()]))

    def forward(self, corrupted, temperature=.8):
        if self.arm == 'categorical_vae':
            return self.network(corrupted, temperature)
        context, _ = self.network.encode(self.wt_tokens[None].expand(len(corrupted), -1))
        deterministic = context.softmax(-1)
        logits = self.network.decode(deterministic)
        return logits, logits.new_zeros(len(corrupted)), context

    def objective(self, logits, targets, kl, beta):
        return self._loss(logits, targets, targets < GAP, kl, 'categorical_vae', beta)

    def prior_logits(self, count, generator):
        if self.training:
            raise ValueError('Prior generation requires eval mode')
        if self.arm == 'categorical_vae':
            return self.network.decode(self.network.sample_prior(count, generator))
        return self(self.wt_tokens[None].expand(count, -1))[0]
