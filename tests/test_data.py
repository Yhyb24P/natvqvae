import json
from pathlib import Path

import numpy as np

from natvqvae.data import PopulationData, Target, generation_masks, mask_panels, stable_seed


def test_target_and_splits():
    data = PopulationData('config/sources.yaml', 'config/study.yaml', 'config/target.yaml')
    assert data.target.sequence == data.target.sequence.upper()
    assert len(data.target.sequence) == 341
    assert len(data.population) == 1871
    assert len(data.train) == 1644
    assert len(data.validation) == 227
    assert len(data.test_rows) == 231
    assert len(set(record.component for record in data.train)) == 47
    assert len(set(record.component for record in data.validation)) == 44
    assert not (set(record.component for record in data.train) &
                set(record.component for record in data.validation))


def test_masks_are_deterministic_and_weighted():
    data = PopulationData('config/sources.yaml', 'config/study.yaml', 'config/target.yaml')
    record = data.train[0]
    first = mask_panels(record, 7401)
    second = mask_panels(record, 7401)
    assert first == second
    assert len(first) == 27
    assert np.isclose(sum(panel['weight'] for panel in first), 1.0)
    assert all(set(panel['positions']) <= set(range(record.length)) for panel in first)
    assert all(len(set(panel['positions'])) == len(panel['positions']) for panel in first)


def test_generation_reference_masks():
    reference = json.loads(Path('tests/generation_reference.json').read_text())
    data = PopulationData('config/sources.yaml', 'config/study.yaml', 'config/target.yaml')
    groups = {group['id']: group for group in generation_masks(data.study['generation'], data.target.sequence)}
    assert len(groups) == 192
    for case in reference['cases']:
        group = groups[case['group']['id']]
        assert group['positions'] == case['group']['positions']
        assert stable_seed(8401, 8301, group['id'], 'latent') == case['prior_seed']
        assert stable_seed(8401, 8301, group['id'], 'residues') == case['uniform_seed']
