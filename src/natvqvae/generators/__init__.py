"""Generator seam. No implementation is provided in v0.

All NatV sequence models available to this project are masked-recovery models: they
recover residues under a mask, conditioned on a frozen PLM. A candidate produced by
such a model is a *recovery sample under a specified mask* and must be reported with
the mask set, temperature, model identity and checkpoint hash.
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ..sequence import Variant


@runtime_checkable
class Generator(Protocol):
    name: str
    model_identity: dict

    def propose(
        self,
        mask_positions: Sequence[int],
        n: int,
        temperature: float,
        design_class: str,
    ) -> list[Variant]:
        """Return n proposals under the given mask set.

        Implementations must:
        - reject mask positions outside the declared design class's editable set;
        - record model identity, checkpoint hash, temperature and RNG seed;
        - never emit indels, truncations or edits outside the mask.
        """
        ...
