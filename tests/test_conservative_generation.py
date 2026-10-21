import numpy as np
import pytest

pytest.importorskip('Bio')

from natvqvae.conservative_editing import AA, constrained_letters
from natvqvae.conservative_generation import (
    PROTECTED_WT_POSITIONS_1BASED,
    candidate_audit,
    draw_letters,
    eligible_positions,
    protected_positions_0based,
    stage_schedule,
)
from natvqvae.sequence import wt_sequence


def test_draw_letters_can_keep_reference():
    wt = wt_sequence()
    reference = wt[0]
    logits = np.full((1, 20), -50.0)
    logits[0, AA.index(reference)] = 50.0
    assert draw_letters(logits, [0.5]) == reference


def test_draw_letters_uses_full_alphabet_cdf():
    logits = np.zeros((4, 20))
    assert draw_letters(logits, [0.0, 0.05, 0.5, 0.999999]) == 'ACMY'
    assert draw_letters(logits, [0.0, 0.05, 0.5, 0.999999]) == 'ACMY'


def test_draw_letters_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        draw_letters(np.zeros((2, 19)), [0.0, 0.0])
    with pytest.raises(ValueError):
        draw_letters(np.array([[np.nan] * 20]), [0.0])
    with pytest.raises(ValueError):
        draw_letters(np.zeros((1, 20)), [1.0])


def test_stage_schedule_alternates_halves():
    assert stage_schedule([10, 20, 30, 40]) == [
        [10, 20, 30, 40], [10, 20], [30, 40], [10, 20], [30, 40]]
    stages = stage_schedule(list(range(34)))
    assert [len(stage) for stage in stages] == [34, 17, 17, 17, 17]
    assert sorted(stages[1] + stages[2]) == list(range(34))


def test_eligible_positions_exclude_protected():
    eligible = eligible_positions()
    assert len(eligible) == 341 - len(PROTECTED_WT_POSITIONS_1BASED)
    assert not set(eligible) & set(protected_positions_0based())


def _spaced_substitutions(wt, count):
    eligible = set(eligible_positions())
    positions = [p for p in range(4, 341, 9) if p in eligible][:count]
    result = list(wt)
    for position in positions:
        nearby = set(wt[max(0, position - 2):position + 3]) | {wt[position]}
        result[position] = next(a for a in AA if a not in nearby)
    return ''.join(result), positions


def test_candidate_audit_wt_reference():
    wt = wt_sequence()
    audit = candidate_audit(wt, wt, [0, 4, 8], [])
    assert audit['distance'] == 0 and audit['wt_reference']
    assert audit['mask_retained_count'] == 3
    assert audit['constraint_pass'] and audit['protected_intact']


def test_candidate_audit_edits_and_reversion():
    wt = wt_sequence()
    sequence, positions = _spaced_substitutions(wt, 34)
    audit = candidate_audit(wt, sequence, positions, positions[:10])
    assert audit['actual_count'] == 34 and audit['reversion_count'] == 0
    assert audit['WT_global_identity'] == pytest.approx(1 - 34 / 341)
    assert audit['constraint_pass']
    reverted = list(sequence)
    for p in positions[:4]:
        reverted[p] = wt[p]
    audit = candidate_audit(wt, ''.join(reverted), positions, positions[:10])
    assert audit['actual_count'] == 30 and audit['reversion_count'] == 4
    assert audit['mask_retained_count'] == 4 and audit['constraint_pass']


def test_candidate_audit_rejects_outside_mask_and_identity():
    wt = wt_sequence()
    sequence, positions = _spaced_substitutions(wt, 35)
    assert not candidate_audit(wt, sequence, positions[1:3], [])['constraint_pass']
    audit = candidate_audit(wt, sequence, positions, positions)
    assert audit['actual_count'] == 35 and not audit['identity_ok']
    assert not audit['constraint_pass']


def test_candidate_audit_protected_position():
    wt = wt_sequence()
    position = protected_positions_0based()[1]
    sequence = list(wt)
    sequence[position] = next(a for a in AA if a != wt[position])
    assert not candidate_audit(wt, ''.join(sequence), [position], [])['protected_intact']


def test_historical_constrained_letters_still_excludes_reference():
    logits = np.zeros((1, 20))
    letter = constrained_letters(logits, [AA[0]], [0.0])
    assert letter == AA[1]
