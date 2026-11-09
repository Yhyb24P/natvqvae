from pathlib import Path

import pytest

from natvqvae.cli import _result_output, build_parser


def test_cli_has_formal_commands():
    parser = build_parser()
    for command in ('doctor', 'preprocess', 'train', 'generate', 'evaluate', 'evaluate-batch',
                    'aux-train', 'aux-generate', 'aux-evaluate', 'candidate-structures',
                    'generation-backgrounds', 'six-cell-compare', 'six-cell-plot', 'latent-precision'):
        with pytest.raises(SystemExit):
            parser.parse_args([command, '--help'])


def test_results_are_organized_by_purpose():
    expected = Path('data/results/generation/wt_qvae').resolve()
    assert _result_output('data/results/generation/wt_qvae', 'generation') == expected
    with pytest.raises(ValueError):
        _result_output('data/raw/variants', 'generation')
    with pytest.raises(ValueError):
        _result_output('data/results/training/run1', 'generation')
    with pytest.raises(ValueError):
        _result_output('data/results/generation', 'generation')



def test_generate_accepts_heldout_background():
    args = build_parser().parse_args([
        'generate', '--config', 'config/study.yaml', '--sources', 'sources.yaml',
        '--target', 'target.yaml', '--checkpoint', 'model.pt',
        '--output', 'data/results/generation/heldout_demo',
        '--background-id', 'seq1', '--test-data', 'data/results/preprocessing/test'])
    assert args.background_id == 'seq1'
    assert args.test_data == 'data/results/preprocessing/test'
    assert args.distribution == 'joint'


def test_generate_accepts_backgrounds_manifest():
    args = build_parser().parse_args([
        'generate', '--config', 'config/study.yaml', '--sources', 'sources.yaml',
        '--target', 'target.yaml', '--checkpoint', 'model.pt',
        '--output', 'data/results/generation/batch_demo',
        '--test-data', 'data/results/preprocessing/test',
        '--backgrounds-manifest', 'data/results/preprocessing/backgrounds.json'])
    assert args.backgrounds_manifest == 'data/results/preprocessing/backgrounds.json'
    assert args.background_id is None


def test_evaluate_batch_routes_each_output(monkeypatch, tmp_path):
    import natvqvae.evaluate as evaluate_module

    runs = tmp_path / 'runs.yaml'
    runs.write_text(
        '- checkpoint: first.pt\n  output: data/results/evaluation/batch/first\n'
        '- checkpoint: second.pt\n  output: data/results/evaluation/batch/second\n')
    captured = {}

    def fake_evaluate_many(config, sources, target, listed, device, split, limit, test_data):
        captured.update({'listed': listed, 'split': split, 'test_data': test_data})
        return 'ok'

    monkeypatch.setattr(evaluate_module, 'evaluate_many', fake_evaluate_many, raising=False)
    args = build_parser().parse_args([
        'evaluate-batch', '--config', 'config/transfer.yaml', '--sources', 'sources.yaml',
        '--target', 'target.yaml', '--runs', str(runs), '--split', 'test',
        '--test-data', 'data/results/preprocessing/test'])
    assert args.func(args) == 'ok'
    assert captured['split'] == 'test'
    assert [row['checkpoint'] for row in captured['listed']] == ['first.pt', 'second.pt']
    assert all(str(row['output']).startswith(str(Path('data/results/evaluation').resolve()))
               for row in captured['listed'])


def test_generation_backgrounds_writes_to_preprocessing():
    args = build_parser().parse_args([
        'generation-backgrounds', '--config', 'config/study.yaml',
        '--sources', 'sources.yaml', '--target', 'target.yaml',
        '--test-data', 'data/results/preprocessing/test',
        '--output', 'data/results/preprocessing/backgrounds'])
    assert args.output == 'data/results/preprocessing/backgrounds'



def test_six_cell_compare_routes_to_evaluation():
    args = build_parser().parse_args([
        'six-cell-compare', '--config', 'config/fragment_six_cell_comparison.yaml',
        '--output', 'data/results/evaluation/natv_six_cell_demo', '--split', 'test'])
    assert args.output == 'data/results/evaluation/natv_six_cell_demo'
    assert args.split == 'test'


def test_six_cell_plot_routes_to_evaluation():
    args = build_parser().parse_args([
        'six-cell-plot', '--comparison', 'data/results/evaluation/natv_six_cell/test'])
    assert args.comparison == 'data/results/evaluation/natv_six_cell/test'


def test_latent_precision_routes_to_evaluation():
    args = build_parser().parse_args([
        'latent-precision', '--comparison-dir', 'data/results/evaluation/natv_fragment_transfer',
        '--output', 'data/results/evaluation/natv_latent_precision'])
    assert args.comparison_dir == 'data/results/evaluation/natv_fragment_transfer'
    assert args.output == 'data/results/evaluation/natv_latent_precision'


def test_fragment_compare_accepts_neighbors_dir():
    args = build_parser().parse_args([
        'fragment-compare', '--evaluation-dir', 'data/results/evaluation/natv_transfer',
        '--esm-dir', 'data/results/evaluation/esm2_natv_validation',
        '--output', 'data/results/evaluation/natv_fragment_validation',
        '--neighbors-dir', 'data/results/preprocessing/natv_sequence_neighbors'])
    assert args.neighbors_dir == 'data/results/preprocessing/natv_sequence_neighbors'
