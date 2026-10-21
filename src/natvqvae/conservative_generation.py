"""WT-conservative generation: editable masks, unconstrained draws and candidate audit."""
from __future__ import annotations

import numpy as np

from .conservative_editing import AA, edit_distance
from .sequence import wt_sequence

PROTECTED_WT_POSITIONS_1BASED = (1, 14, 17, 135, 138)


def protected_positions_0based(positions=PROTECTED_WT_POSITIONS_1BASED):
    return tuple(int(p) - 1 for p in positions)


def eligible_positions(protected=PROTECTED_WT_POSITIONS_1BASED):
    protected_zero = set(protected_positions_0based(protected))
    return tuple(p for p in range(len(wt_sequence())) if p not in protected_zero)


def stage_schedule(sites, sweeps=2):
    """Return the all-sites mask followed by alternating halves for each sweep."""
    sites = [int(p) for p in sites]
    if not sites or len(set(sites)) != len(sites):
        raise ValueError('Invalid mask sites')
    half = len(sites) // 2
    if half == 0:
        raise ValueError('At least two sites are required for staged updates')
    stages = [sites]
    for sweep in range(2 * sweeps):
        stages.append(sites[:half] if sweep % 2 == 0 else sites[half:])
    return stages


def draw_letters(logits, uniforms, temperature=1.0):
    """Draw all 20 amino acids; the reference residue is not excluded."""
    scores = np.asarray(logits, dtype=np.float64) / temperature
    if scores.ndim != 2 or scores.shape[1] != len(AA) or not np.isfinite(scores).all():
        raise ValueError('Invalid logits')
    if len(uniforms) != len(scores) or not all(0 <= u < 1 for u in uniforms):
        raise ValueError('Invalid random draws')
    probabilities = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    cdf = probabilities.cumsum(axis=1)
    cdf[:, -1] = 1.0
    indices = (np.asarray(uniforms)[:, None] >= cdf).sum(axis=1)
    return ''.join(AA[int(i)] for i in indices)


def candidate_audit(wt, sequence, mask_positions, ever_changed_positions,
                    protected=PROTECTED_WT_POSITIONS_1BASED):
    """Describe actual edits relative to WT; mask positions may remain or revert to WT."""
    mask = sorted({int(p) for p in mask_positions})
    if len(mask) != len(mask_positions):
        raise ValueError('Duplicate mask position')
    protected_zero = protected_positions_0based(protected)
    length_ok = len(sequence) == len(wt)
    actual = tuple(p for p, (a, b) in enumerate(zip(sequence, wt)) if a != b)
    actual_set = set(actual)
    reverted = tuple(sorted(set(int(p) for p in ever_changed_positions) - actual_set))
    distance = edit_distance(sequence, wt)
    identity = 1 - distance / max(len(sequence), len(wt))
    subset = actual_set <= set(mask)
    protected_intact = all(p < len(sequence) and sequence[p] == wt[p] for p in protected_zero)
    return {
        'length': len(sequence),
        'distance': distance,
        'WT_global_identity': identity,
        'actual_positions_0based': actual,
        'actual_count': len(actual),
        'mask_retained_count': len(mask) - len(actual),
        'reversion_positions_0based': reverted,
        'reversion_count': len(reverted),
        'actual_subset_of_mask': subset,
        'protected_intact': protected_intact,
        'identity_ok': identity >= 0.9,
        'wt_reference': distance == 0,
        'constraint_pass': length_ok and subset and protected_intact and identity >= 0.9,
    }
