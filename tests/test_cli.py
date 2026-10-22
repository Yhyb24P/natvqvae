from pathlib import Path

import pytest

from natvqvae.cli import _external_output, build_parser


def test_cli_has_formal_commands():
    parser = build_parser()
    for command in ('doctor', 'train', 'generate', 'evaluate'):
        with pytest.raises(SystemExit):
            parser.parse_args([command, '--help'])


def test_output_must_be_outside_data(tmp_path):
    assert _external_output(tmp_path / 'run') == (tmp_path / 'run').resolve()
    with pytest.raises(ValueError):
        _external_output(Path('data/results/run'))
