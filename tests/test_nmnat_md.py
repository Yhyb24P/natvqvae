from pathlib import Path

import numpy as np
import yaml

from natvqvae.nmnat_md import (SOLVATE_BUFFER_A, _correct_charge, _em_result,
                               _existing_em_result, _minimum_image, _tleap,
                               filter_name, run_nmnat_md_em)
import natvqvae.nmnat_md as md


def _toy_mol2(path, charge):
    path.write_text(
        '@<TRIPOS>MOLECULE\nMOL\n2 1 0 0 0\nSMALL\nNO_CHARGES\n'
        '\n@<TRIPOS>ATOM\n'
        f'1 P1 0.0 0.0 0.0 p5 1 MOL {charge:.6f}\n'
        '2 C1 1.0 0.0 0.0 c3 1 MOL 0.000000\n'
        '\n@<TRIPOS>BOND\n1 1 2 1\n')


def test_charge_correction_is_written_to_mol2(tmp_path):
    path = tmp_path / 'ligand.mol2'
    _toy_mol2(path, -3.998)
    residual, max_change = _correct_charge(path, 'ATP')
    assert abs(residual + 0.002) < 1e-9
    lines = path.read_text().splitlines()
    start = lines.index('@<TRIPOS>ATOM') + 1
    total = sum(float(lines[index].split()[8]) for index in range(start, start + 2))
    assert abs(total + 4.0) < 1e-8
    assert abs(max_change - 0.002) < 1e-8


def test_solvate_buffer_is_in_angstroms(tmp_path, monkeypatch):
    assert SOLVATE_BUFFER_A == 12.0

    def fake_run(command, cwd, log):
        Path(log).write_text('Errors = 0\n')

    monkeypatch.setattr(md, '_run', fake_run)
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'protein.pdb').write_text('END\n')
    (model / 'ATP_model.mol2').write_text('')
    (model / 'NMN_model.mol2').write_text('')
    params = {'ATP': {'frcmod': 'atp.frcmod'}, 'NMN': {'frcmod': 'nmn.frcmod'}}
    _tleap({'tleap': 'tleap'}, model, params, False)
    assert 'solvateBox com TIP3PBOX 12.0' in (model / 'tleap.in').read_text()


def test_filter_name_removes_path_separators():
    assert filter_name('a/b c.cif') == 'a_b_c.cif'


class _Atom:
    def __init__(self, x, y, z, name):
        self.xx, self.xy, self.xz, self.name = x, y, z, name


def test_minimum_image_wraps_across_box_boundary():
    first = [_Atom(0.5, 5.0, 5.0, 'a')]
    second = [_Atom(9.5, 5.0, 5.0, 'b')]
    distance, left, right = _minimum_image(first, second, np.array([10.0, 10.0, 10.0]))
    assert abs(distance - 1.0) < 1e-9
    assert left.name == 'a' and right.name == 'b'


def _em_log(fmax, steps, finished=True, max_steps=False):
    lines = []
    if max_steps:
        lines.append('Energy minimization reached the maximum number of steps '
                     'before the forces reached the requested precision Fmax < 500.')
    lines += [
        f'Steepest Descents did not converge in {steps} steps.',
        'Potential Energy  = -1.0000000e+05',
        f'Maximum force     =  {fmax:.6f} on atom 1',
    ]
    if finished:
        lines.append('Finished mdrun on rank 0 Sat Oct 24 03:31:01 2026')
    return '\n'.join(lines) + '\n'


CONFIG = {'em_tol': 500, 'em_max_steps': 20000}


def test_em_requires_numeric_fmax_not_status_string():
    converged = _em_result(_em_log(400, 123), CONFIG, completed=True)
    max_steps = _em_result(_em_log(36216, 20001, max_steps=True), CONFIG,
                           completed=True)
    stalled = _em_result(_em_log(8800, 1000), CONFIG, completed=True)
    unfinished = _em_result(_em_log(400, 123, finished=False), CONFIG,
                            completed=True)
    assert converged['em_status'] == 'FMAX_CONVERGED'
    assert converged['em_converged'] == 'True'
    assert max_steps['em_status'] == 'MAX_STEPS'
    assert max_steps['em_converged'] == 'False'
    assert stalled['em_status'] == 'PRECISION_STALLED'
    assert stalled['em_converged'] == 'False'
    assert unfinished['em_status'] == 'FAILED'


def test_tpr_alone_does_not_count_as_completed_em(tmp_path):
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'em.tpr').write_text('')
    assert _existing_em_result(model, CONFIG) is None


def test_later_row_failure_keeps_error_column(tmp_path, monkeypatch):
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    (first / 'em.gro').write_text('')
    (first / 'em.log').write_text(_em_log(400, 123))
    (second / 'com_gmx.gro').write_text('')
    (second / 'com_gmx.top').write_text('')
    output = tmp_path / 'out'
    output.mkdir()
    (output / 'systems.csv').write_text(
        'model,directory,status\n'
        f'a,{first},BUILT\n'
        f'b,{second},BUILT\n')
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({
        'grompp': 'gmx', 'mdrun': 'gmx', 'em_threads': 1,
        'em_max_steps': 20000, 'em_tol': 500,
    }))

    def fail(command, cwd, log):
        raise RuntimeError('grompp failed')

    monkeypatch.setattr(md, '_run', fail)
    result = run_nmnat_md_em(config, output)
    rows = list(__import__('csv').DictReader(open(output / 'systems.csv')))
    assert result['failed'] == 1
    assert rows[0]['status'] == 'FMAX_CONVERGED'
    assert rows[0]['error'] == ''
    assert rows[1]['status'] == 'FAILED'
    assert 'grompp failed' in rows[1]['error']


def test_md_mdp_stage_settings(tmp_path):
    config = {'md_temperature': 298.15, 'md_dt_fs': 2,
              'md_nvt_ps': 100, 'md_npt_ps': 100, 'md_production_ps': 1000}
    for stage, steps in (('nvt', 50000), ('npt', 50000),
                         ('production', 500000)):
        path = tmp_path / f'{stage}.mdp'
        md._md_mdp(path, stage, config, 123, stage != 'production')
        text = path.read_text()
        assert f'nsteps = {steps}' in text
        assert ('gen-vel = yes' in text) == (stage == 'nvt')
        assert ('pcoupl = C-rescale' in text) == (stage != 'nvt')
        assert ('define = -DPOSRES' in text) == (stage != 'production')
        assert 'ld-seed = 123' in text


def test_stage_done_requires_target_steps(tmp_path):
    stage = 'nvt'
    config = {'md_dt_fs': 2, 'md_nvt_ps': 100}
    (tmp_path / f'{stage}.gro').write_text('')
    (tmp_path / f'{stage}.cpt').write_text('')
    log = tmp_path / f'{stage}.log'
    log.write_text('Step           Time\n'
                   '   50000      100.00000\n'
                   'Finished mdrun on rank 0\n')
    assert md._stage_done(tmp_path, stage, config)
    log.write_text('Step           Time\n'
                   '    5000       10.00000\n'
                   'Finished mdrun on rank 0\n')
    assert not md._stage_done(tmp_path, stage, config)
    (tmp_path / f'{stage}.cpt').unlink()
    log.write_text('Step           Time\n'
                   '   50000      100.00000\n'
                   'Finished mdrun on rank 0\n')
    assert not md._stage_done(tmp_path, stage, config)


def test_production_stage_requires_trajectory(tmp_path):
    stage = 'production'
    config = {'md_dt_fs': 2, 'md_production_ps': 1000}
    (tmp_path / f'{stage}.gro').write_text('')
    (tmp_path / f'{stage}.cpt').write_text('')
    (tmp_path / f'{stage}.log').write_text(
        'Step           Time\n'
        '  500000     1000.00000\n'
        'Finished mdrun on rank 0\n')
    assert not md._stage_done(tmp_path, stage, config)
    (tmp_path / f'{stage}.xtc').write_text('')
    assert md._stage_done(tmp_path, stage, config)


def test_md_stage_uses_gmx_subcommands(tmp_path, monkeypatch):
    calls = []

    def fake_run(command, cwd, log):
        calls.append(command)

    monkeypatch.setattr(md, '_run', fake_run)
    config = {'grompp': 'gmx', 'mdrun': 'gmx', 'md_threads': 1}
    model = tmp_path / 'model'
    model.mkdir()
    md._md_stage(model, tmp_path / 'run', 'nvt', config,
                 model / 'em.gro', True, False)
    assert calls[0][1] == 'grompp'
    assert calls[1][1] == 'mdrun'


def test_gpu_idle_uses_compute_apps(monkeypatch):
    import subprocess
    config = {'md_gpu_max_mem_fraction': 0.1}

    def no_apps(command, text=True):
        if any(str(item).startswith('--query-compute-apps')
               for item in command):
            return ''
        return '12, 100, 32000\n'

    monkeypatch.setattr(subprocess, 'check_output', no_apps)
    assert md._gpu_idle(config)

    def with_app(monkeypatch):
        def output(command, text=True):
            if any(str(item).startswith('--query-compute-apps')
                   for item in command):
                return '1234, other-service, 26000\n'
            return '97, 26591, 32607\n'
        monkeypatch.setattr(subprocess, 'check_output', output)
        assert not md._gpu_idle(config)

    with_app(monkeypatch)
