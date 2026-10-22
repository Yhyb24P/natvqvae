from pathlib import Path

import pytest

from natvqvae.cli import _result_output, build_parser


def test_cli_has_formal_commands():
    parser = build_parser()
    for command in ('doctor', 'preprocess', 'train', 'generate', 'evaluate',
                    'aux-train', 'aux-generate', 'aux-evaluate', 'candidate-structures'):
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
