import pytest

from natvqvae.fragment_comparison import _six_cell_contrasts

SEEDS = [9201, 9202, 9203]
CONDITIONS = ['random', 'self', 'external']


def _tables(components):
    qbase = {'random': 1.0, 'self': 2.0, 'external': 3.0}
    dbase = {'random': 10.0, 'self': 20.0, 'external': 30.0}
    tables = {'QVAE': [], 'deterministic': []}
    for index, component in enumerate(components):
        for condition in CONDITIONS:
            for seed in SEEDS:
                for metric, offset in (('prior_joint_nll_per_residue', 0.0),
                                       ('marginal_nll_per_residue', 0.5)):
                    for arm, base, scale in (('QVAE', qbase, 1.0),
                                             ('deterministic', dbase, 10.0)):
                        tables[arm].append({
                            'condition': condition, 'downstream_seed': str(seed),
                            'region': 'NMNAT', 'identity_stratum': 'all_identities',
                            'target_sites': '8', 'component': component, 'metric': metric,
                            'nll': str(base[condition] + scale * index + offset),
                            'sequences': '1', 'queries': '1'})
        for arm in tables:
            tables[arm].append({
                'condition': 'frozen_esm2', 'downstream_seed': '',
                'region': 'NMNAT', 'identity_stratum': 'all_identities',
                'target_sites': '8', 'component': component,
                'metric': 'factorized_nll_per_residue',
                'nll': str(100.0 + index), 'sequences': '1', 'queries': '1'})
    return tables


def test_six_cell_contrast_signs_and_means():
    summaries, groups = _six_cell_contrasts(_tables(['c1', 'c2']), SEEDS,
                                            bootstrap_repeats=10, bootstrap_seed=7)
    means = {(row['comparison'], row['scope']): row['mean']
             for row in summaries if row['identity_stratum'] == 'all_identities'}
    assert means['QVAE:external_minus_self', 'mean_of_pipeline_seeds'] == pytest.approx(1.0)
    assert means['deterministic:external_minus_self', 'mean_of_pipeline_seeds'] == pytest.approx(10.0)
    assert means['QVAE:self_minus_random', 'mean_of_pipeline_seeds'] == pytest.approx(1.0)
    assert means['QVAE:external_minus_random', 'mean_of_pipeline_seeds'] == pytest.approx(2.0)
    assert means['deterministic:self_minus_random', 'mean_of_pipeline_seeds'] == pytest.approx(10.0)
    assert means['deterministic:external_minus_random', 'mean_of_pipeline_seeds'] == pytest.approx(20.0)
    assert means['external_minus_self:QVAE_minus_deterministic', 'mean_of_pipeline_seeds'] == pytest.approx(-9.0)
    assert means['external:joint_minus_site_marginal', 'mean_of_pipeline_seeds'] == pytest.approx(-0.5)
    assert means['QVAE:random_minus_frozen_esm2', 'mean_of_pipeline_seeds'] == pytest.approx(-99.0)
    assert len(groups) == 19 * 2 * 3


def test_single_component_stratum_has_no_interval():
    summaries, _ = _six_cell_contrasts(_tables(['c1']), SEEDS,
                                       bootstrap_repeats=10, bootstrap_seed=7)
    primary = [row for row in summaries if row['comparison'] == 'QVAE:external_minus_self'
               and row['scope'] == 'mean_of_pipeline_seeds' and row['identity_stratum'] == 'all_identities']
    assert len(primary) == 1
    assert primary[0]['groups'] == 1
    assert primary[0]['ci95'] is None


def test_six_cell_outputs_preserve_paired_estimates(tmp_path):
    import csv
    import json
    from natvqvae.fragment_comparison import _write_six_cell_outputs, plot_six_cells
    from natvqvae.data import sha256_file

    tables = _tables(['c1', 'c2'])
    tables = {arm: [{**row, 'region': region, 'target_sites': 'all_sizes'}
                    for row in rows for region in ('NMNAT', 'Nudix', 'global')]
              for arm, rows in tables.items()}
    estimates, _ = _six_cell_contrasts(tables, SEEDS, bootstrap_repeats=10, bootstrap_seed=7)
    result = {'status': 'COMPLETED', 'split': 'test', 'pipeline_seed_pairs': [{'downstream': seed} for seed in SEEDS],
              'comparisons': estimates, 'limits': ['Synthetic test fixture.']}
    source = tmp_path / 'summary.json'
    source.write_text(json.dumps(result))
    digest = sha256_file(source)
    identity = plot_six_cells(tmp_path)
    assert identity['summary_sha256'] == sha256_file(source) == digest
    assert not identity['prediction_scoring_performed']
    assert identity['outputs_sha256']['prediction_effects.png'] == sha256_file(tmp_path / 'prediction_effects.png')
    for extension in ('pdf', 'png'):
        assert (tmp_path / f'prediction_effects.{extension}').stat().st_size > 1000
    with (tmp_path / 'prediction_effects.csv').open() as handle:
        displayed = list(csv.DictReader(handle))
    assert len(displayed) == len(estimates)
    primary = [row for row in displayed if row['comparison'] == 'QVAE:external_minus_self'
               and row['region'] == 'NMNAT' and row['scope'] == 'mean_of_pipeline_seeds']
    assert len(primary) == 1 and float(primary[0]['mean']) == pytest.approx(1.0)
    assert 'global 不是全部掩码' in (tmp_path / 'REPORT.md').read_text()
    result['comparisons'] = estimates[1:]
    with pytest.raises(ValueError, match='Incomplete six-cell figure'):
        _write_six_cell_outputs(result, tmp_path)
