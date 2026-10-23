"""Analyze the WT NMNAT joint-substrate MD pilot without name guessing."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import sha256_file, write_json

CANDIDATES = ('O1P', 'O2P', 'O3P')
ATP_PHOSPHATE_O = ('O1A', 'O2A', 'O3A', 'O1B', 'O2B', 'O3B',
                   'O1G', 'O2G', 'O3G')
NMN_PHOSPHATE_O = CANDIDATES
MG_CUTOFF = 2.8


def _kabsch(reference, mobile):
    left = reference - reference.mean(axis=0)
    right = mobile - mobile.mean(axis=0)
    u, _, vt = np.linalg.svd(left.T @ right)
    sign = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, sign]) @ vt
    return rotation, reference.mean(axis=0), mobile.mean(axis=0)


def _rmsd(reference, mobile):
    rotation, ref_mean, mob_mean = _kabsch(reference, mobile)
    fitted = (reference - ref_mean) @ rotation + mob_mean
    return float(np.sqrt(np.mean(np.sum((fitted - mobile) ** 2, axis=1))))


def _mic(delta, box):
    return delta - box * np.round(delta / box)


def _na_phosphate_summary(na_atp, na_nmn, cutoff=4.0):
    na_min = float(min(na_atp.min(), na_nmn.min()))
    within = int(((na_atp <= cutoff).any(axis=1)
                  | (na_nmn <= cutoff).any(axis=1)).sum())
    return na_min, within


def _ccd_index(mapping, model, ligand, names):
    atoms = mapping['models'][model]['ligands'][ligand]['atoms']
    found = {}
    for atom in atoms:
        if atom['ccd_name'] in names:
            found[atom['ccd_name']] = atom['topology_index'] - 1
    missing = set(names) - set(found)
    if missing:
        raise KeyError(f'{ligand} missing CCD atoms: {sorted(missing)}')
    return found


def _unwrap(atoms):
    atoms.unwrap(compound='fragments')


def _same_image(group, anchor, box):
    delta = group.center_of_mass() - anchor.center_of_mass()
    group.positions += _mic(delta, box[:3]) - delta


def _run_frames(run_dir, model_dir, mapping, initial_group, model, velocity_seed):
    import MDAnalysis as mda
    from MDAnalysis.lib import distances

    run_dir = Path(run_dir)
    model_dir = Path(model_dir)
    tpr = run_dir / 'production.tpr'
    xtc = run_dir / 'production.xtc'
    mobile = mda.Universe(str(tpr), str(xtc))
    reference = mda.Universe(str(tpr), str(model_dir / 'em.gro'))
    if len(mobile.atoms) != len(reference.atoms):
        raise ValueError(f'Topology mismatch in {run_dir}')
    protein = mobile.select_atoms('protein and not name H*')
    ca = mobile.select_atoms('protein and name CA and resid 4:181')
    ref_ca = reference.select_atoms('protein and name CA and resid 4:181')
    atp_heavy = mobile.select_atoms(
        'resname ATP and not name H*')
    nmn_heavy = mobile.select_atoms(
        'resname NMN and not name H*')
    ref_atp = reference.select_atoms('resname ATP and not name H*')
    ref_nmn = reference.select_atoms('resname NMN and not name H*')
    water_o = mobile.select_atoms('resname WAT and name O')
    protein_o = mobile.select_atoms('protein and name O*')
    protein_n = mobile.select_atoms('protein and name N*')
    mg_atoms = mobile.select_atoms('resname MG')
    na_atoms = mobile.select_atoms('resname Na+')
    atp_o = atp_heavy.select_atoms('name O*')
    nmn_o = nmn_heavy.select_atoms('name O*')

    atp = _ccd_index(mapping, model, 'ATP', ('PA', 'O3A'))
    nmn = _ccd_index(mapping, model, 'NMN', ('P',) + CANDIDATES)
    atp_phosphate = _ccd_index(mapping, model, 'ATP', ATP_PHOSPHATE_O)
    nmn_phosphate = _ccd_index(mapping, model, 'NMN', NMN_PHOSPHATE_O)
    atp_phosphate_group = mobile.atoms[
        [atp_phosphate[name] for name in ATP_PHOSPHATE_O]]
    nmn_phosphate_group = mobile.atoms[
        [nmn_phosphate[name] for name in NMN_PHOSPHATE_O]]
    mg_index = mg_atoms.indices[0] if len(mg_atoms) else None
    ref_mg = reference.atoms[mg_index] if mg_index is not None else None
    ccd_by_index = {}
    for ligand in ('ATP', 'NMN'):
        for atom in mapping['models'][model]['ligands'][ligand]['atoms']:
            ccd_by_index[atom['topology_index'] - 1] = atom['ccd_name']
    _unwrap(mobile.select_atoms('protein or resname ATP NMN MG'))
    _unwrap(reference.select_atoms('protein or resname ATP NMN MG'))
    box0 = np.asarray(mobile.dimensions, dtype=float)
    ref_protein = reference.select_atoms('protein')
    ref_box = np.asarray(reference.dimensions, dtype=float)
    for ligand_group in (mobile.select_atoms('resname ATP'),
                         mobile.select_atoms('resname NMN')):
        _same_image(ligand_group, protein, box0)
    for ligand_group in (reference.select_atoms('resname ATP'),
                         reference.select_atoms('resname NMN')):
        _same_image(ligand_group, ref_protein, ref_box)
    if mg_index is not None:
        _same_image(mobile.atoms[[mg_index]], protein, box0)
        _same_image(reference.atoms[[mg_index]], ref_protein, ref_box)

    rows = []
    mg_ligand_frames = {}
    frames_seen = 0
    for frame in mobile.trajectory:
        frames_seen += 1
        _unwrap(mobile.select_atoms('protein or resname ATP NMN MG'))
        box = np.asarray(mobile.dimensions, dtype=float)
        box3 = box[:3]
        for ligand_group in (mobile.select_atoms('resname ATP'),
                             mobile.select_atoms('resname NMN')):
            _same_image(ligand_group, protein, box)
        if mg_index is not None:
            _same_image(mobile.atoms[[mg_index]], protein, box)
        ca_xyz = ca.positions.copy()
        rotation, ref_mean, mob_mean = _kabsch(ref_ca.positions, ca_xyz)
        atp_xyz = atp_heavy.positions.copy()
        nmn_xyz = nmn_heavy.positions.copy()
        atp_pose = float(np.sqrt(np.mean(np.sum(
            ((ref_atp.positions - ref_mean) @ rotation + mob_mean
             - atp_xyz) ** 2, axis=1))))
        nmn_pose = float(np.sqrt(np.mean(np.sum(
            ((ref_nmn.positions - ref_mean) @ rotation + mob_mean
             - nmn_xyz) ** 2, axis=1))))
        atp_internal = _rmsd(ref_atp.positions, atp_xyz)
        nmn_internal = _rmsd(ref_nmn.positions, nmn_xyz)
        cross = distances.distance_array(atp_xyz, nmn_xyz, box=box).min()
        phosphate = distances.distance_array(
            atp_phosphate_group.positions, nmn_phosphate_group.positions,
            box=box).min()
        atp_protein = distances.distance_array(
            atp_xyz, protein.positions, box=box).min()
        nmn_protein = distances.distance_array(
            nmn_xyz, protein.positions, box=box).min()
        if len(na_atoms):
            na_atp = distances.distance_array(
                na_atoms.positions, atp_phosphate_group.positions, box=box)
            na_nmn = distances.distance_array(
                na_atoms.positions, nmn_phosphate_group.positions, box=box)
            na_min, na_within = _na_phosphate_summary(na_atp, na_nmn)
        else:
            na_min = None
            na_within = 0
        pa = mobile.atoms[atp['PA']].position
        o3a = mobile.atoms[atp['O3A']].position
        mg_values = {}
        if mg_index is not None:
            mg = mobile.atoms[mg_index].position
            mg_delta = _mic(mg - ((ref_mg.position - ref_mean)
                                  @ rotation + mob_mean), box3)
            mg_values['mg_displacement_A'] = float(np.linalg.norm(mg_delta))
            oxygen = mobile.select_atoms('name O*')
            d = distances.distance_array(mg[None, :], oxygen.positions,
                                         box=box)[0]
            close = oxygen[d <= MG_CUTOFF]
            close_d = d[d <= MG_CUTOFF]
            categories = []
            for atom in close:
                name = atom.residue.resname
                if name == 'WAT':
                    category = 'water'
                elif name == 'ATP':
                    category = 'atp'
                elif name == 'NMN':
                    category = 'nmn'
                elif name in ('Na+', 'MG'):
                    category = 'other'
                else:
                    category = 'protein'
                categories.append(category)
            mg_values['mg_o_coord'] = len(close)
            for category in ('water', 'atp', 'nmn', 'protein', 'other'):
                mg_values[f'mg_{category}_o'] = categories.count(category)
            mg_values['mg_nearest_o_A'] = (float(close_d.min())
                                           if len(close_d) else None)
            n_close = distances.capped_distance(
                mg[None, :], protein_n.positions, MG_CUTOFF, box=box)[0]
            mg_values['mg_protein_n'] = len(n_close)
            mg_values['mg_atp_min_A'] = float(distances.distance_array(
                mg[None, :], atp_o.positions, box=box).min())
            mg_values['mg_nmn_min_A'] = float(distances.distance_array(
                mg[None, :], nmn_o.positions, box=box).min())
            mg_values['mg_bridging'] = int(mg_values['mg_atp_o'] > 0
                                           and mg_values['mg_nmn_o'] > 0)
            for atom in close:
                resname = atom.residue.resname
                if resname in ('ATP', 'NMN'):
                    key = f'{resname}:{ccd_by_index[atom.index]}'
                elif resname in ('WAT', 'Na+', 'MG'):
                    continue
                else:
                    key = f'{resname}{atom.residue.resid} {atom.name}'
                mg_ligand_frames[key] = mg_ligand_frames.get(key, 0) + 1
        for candidate in CANDIDATES:
            onuc = mobile.atoms[nmn[candidate]].position
            v1 = _mic(o3a - pa, box3)
            v2 = _mic(onuc - pa, box3)
            angle = float(np.degrees(np.arccos(np.clip(
                np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)),
                -1.0, 1.0))))
            row = {
                'group': initial_group, 'model': model,
                'velocity_seed': velocity_seed,
                'time_ps': float(frame.time), 'candidate': candidate,
                'pa_onuc_A': float(np.linalg.norm(v2)),
                'pa_o3a_onuc_deg': angle,
                'atp_pose_rmsd_A': atp_pose, 'nmn_pose_rmsd_A': nmn_pose,
                'atp_internal_rmsd_A': atp_internal,
                'nmn_internal_rmsd_A': nmn_internal,
                'cross_min_A': float(cross),
                'phosphate_min_A': float(phosphate),
                'atp_protein_min_A': float(atp_protein),
                'nmn_protein_min_A': float(nmn_protein),
                'na_phosphate_min_A': na_min,
                'na_within_4': na_within,
            }
            row.update(mg_values)
            rows.append(row)
    return rows, mg_ligand_frames, frames_seen, mg_index is not None


def _analysis_counts(frames):
    trajectories = {(row['model'], row['velocity_seed']) for row in frames}
    physical = {(row['model'], row['velocity_seed'], row['time_ps'])
                for row in frames}
    return len(trajectories), len(physical), len(frames)


def _mg_ligand_table(rows, trajectories=None):
    frame = pd.DataFrame(
        rows, columns=['group', 'model', 'velocity_seed',
                       'ligand_atom', 'fraction'])
    if trajectories is None:
        trajectories = (frame[['group', 'model', 'velocity_seed']]
                        .drop_duplicates() if len(frame) else [])
    traj = pd.DataFrame(trajectories,
                        columns=['group', 'model', 'velocity_seed'])
    if not len(traj) or not len(frame):
        return pd.DataFrame()
    index = pd.MultiIndex.from_frame(traj)
    wide = frame.pivot_table(
        index=['group', 'model', 'velocity_seed'], columns='ligand_atom',
        values='fraction', aggfunc='max')
    wide = wide.reindex(index).fillna(0.0)
    table = []
    for group, block in wide.groupby(level='group'):
        total = len(block)
        for key in block.columns:
            values = block[key]
            persistent = int((values >= 0.5).sum())
            if not persistent:
                continue
            table.append({
                'group': group, 'ligand_atom': key,
                'persistent': persistent, 'trajectories': total,
                'median_fraction': float(values.median())})
    table = pd.DataFrame(table)
    if len(table):
        table = table.sort_values(
            ['group', 'persistent', 'median_fraction'],
            ascending=[True, False, False]).reset_index(drop=True)
    return table


def _write_report(output, analysis, summary, trajectories, physical_frames,
                  candidate_rows, mg_ligands, mg_trajectories):
    runs = list(csv.DictReader(open(Path(output) / 'md_runs.csv')))
    try:
        planned = int(json.load(open(Path(output) / 'md_identity.json'))
                      ['planned']['runs'])
    except Exception:
        planned = len(runs)
    done = [row for row in runs if row.get('status') == 'DONE']
    failed = [row for row in runs if row.get('status') == 'FAILED']
    not_run = [row for row in runs if row.get('status') == 'NOT_RUN_EM']
    lines = [
        '# WT NMNAT ATP/NMN 显式水 MD 初始响应', '',
        '本报告只描述 1 ns production 的局部初始响应；不是平衡、Mg 交换、'
        '占有率、反应速率或催化活性证据。初态组在模拟前固定，不按结果换组。',
        '',
        '## 完成度', '',
        f'- 计划轨迹：{planned}；已记录：{len(runs)}；DONE：{len(done)}；'
        f'FAILED：{len(failed)}；NOT_RUN_EM：{len(not_run)}。',
        f'- 已分析轨迹：{trajectories}；物理帧：{physical_frames}；'
        f'候选行：{candidate_rows}；候选汇总行：{len(summary)}。',
        f'- 每轨迹为 100 ps NVT + 100 ps NPT + 1 ns production；'
        f'观察段合计 {trajectories} ns，'
        f'准备段合计 {trajectories * 0.2:.1f} ns。',
        '',
    ]
    if len(summary):
        group = summary.groupby(['group', 'candidate']).agg(
            trajectories=('velocity_seed', 'size'),
            distance=('pa_onuc_median_A', 'median'),
            angle=('angle_median_deg', 'median'),
            cross=('cross_median_A', 'median'),
            cross_fraction=('cross_lt_2_8_fraction', 'median'),
            phosphate=('phosphate_min_median_A', 'median'),
            phosphate_fraction=('phosphate_lt_2_8_fraction', 'median'),
            na=('na_phosphate_median_A', 'median'),
            atp_pose=('atp_pose_median_A', 'median'),
            nmn_pose=('nmn_pose_median_A', 'median'),
            mg_water=('mg_water_o_median', 'median'),
            mg_water_min=('mg_water_o_median', 'min'),
            mg_water_max=('mg_water_o_median', 'max'),
            mg_water_fraction=('mg_water_o_gt0_fraction', 'median'),
            mg_bridging=('mg_bridging_fraction', 'median'),
        ).reset_index()
        lines += ['## 按初态组与 NMN 候选汇总（中位数）', '',
                  '| 组 | 候选 | 轨迹 | PA–Onuc (Å) | O3A–PA–Onuc (°) | '
                  '跨配体最小 (Å) | <2.8 Å 帧比例 | 磷酸 O–O 最小 (Å) | '
                  '磷酸 <2.8 Å 帧比例 | Na⁺–磷酸最小 (Å) | ATP pose RMSD (Å) | '
                  'NMN pose RMSD (Å) | Mg 水 O 中位 | Mg 水 O 范围 | '
                  '有水 O 帧比例 | Mg 共配位帧比例 |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for row in group.itertuples():
            def fmt(value):
                return '' if pd.isna(value) else f'{value:.3f}'
            water_range = ('' if pd.isna(row.mg_water_min)
                           else f'{row.mg_water_min:.0f}–{row.mg_water_max:.0f}')
            lines.append(
                f'| {row.group} | {row.candidate} | {row.trajectories} | '
                f'{fmt(row.distance)} | {fmt(row.angle)} | {fmt(row.cross)} | '
                f'{fmt(row.cross_fraction)} | {fmt(row.phosphate)} | '
                f'{fmt(row.phosphate_fraction)} | {fmt(row.na)} | '
                f'{fmt(row.atp_pose)} | '
                f'{fmt(row.nmn_pose)} | {fmt(row.mg_water)} | '
                f'{water_range} | {fmt(row.mg_water_fraction)} | '
                f'{fmt(row.mg_bridging)} |')
        lines += ['', '表中为组×候选的轨迹中位数；`md_summary.csv` 保留逐轨迹'
                  '数值，组中位数不表示各轨迹一致。']
        if len(mg_ligands):
            frequent = _mg_ligand_table(
                mg_ligands, mg_trajectories).head(30)
            if len(frequent):
                lines += ['', '## Mg 第一壳中反复出现的配体原子', '',
                          '按初态组分别统计；分母为该组已分析 Mg 轨迹数，'
                          '未出现该原子的轨迹按 0 计。仅列出至少一条轨迹中'
                          ' ≥50% 帧出现在 2.8 Å 内的蛋白/核苷酸氧；'
                          '这是距离接触统计，不是配位化学判定。',
                          '', '| 组 | 配体原子 | 持续轨迹数 | 轨迹总数 | '
                          '中位帧比例 |', '|---|---|---:|---:|---:|']
                for row in frequent.itertuples():
                    lines.append(f'| {row.group} | {row.ligand_atom} | '
                                 f'{row.persistent} | {row.trajectories} | '
                                 f'{row.median_fraction:.3f} |')
        lines += ['', '## 失败与未运行', '']
        if failed:
            lines += ['| 模型 | 速度种子 | 状态 | 原因 |', '|---|---:|---|---|']
            for row in failed:
                lines.append(f"| {row['model']} | {row['velocity_seed']} | "
                             f"FAILED | {row.get('error', '')} |")
        else:
            lines.append('- 无 FAILED 轨迹。')
        if not_run:
            lines += ['', '未达到 EM 标准的初态按原计划保留为 NOT_RUN_EM：']
            for row in not_run:
                lines.append(f"- {row['model']} seed {row['velocity_seed']}: "
                             f"{row.get('error', '')}")
    else:
        lines += ['## 结果', '', '当前没有完成的轨迹可汇总。']
    lines += ['', '## 解释边界', '',
              '- 1 ns 只描述局部初始响应，不能推断平衡、配位交换或占有率。',
              '- 一个 Mg 只放在预测位置；Na+ 中和不等于 12 mM MgCl2 缓冲液。',
              '- 初始组别固定；不同初态/种子的差异不单独归因于 Mg 或底物。',
              '- 同一初态的两个速度种子不是独立重复；不同初态组来自不同联合模型，'
              '组间比较不是配对实验。',
              '- 2.8 Å 是短接触筛查值，不是通用范德华硬阈值或反应判据。',
              '- 磷酸 O–O 最小距离只描述两个磷酸端的最近接触，不计算静电'
              '相互作用能，也不判定质子化或配位状态。',
              '- Na⁺–磷酸距离只描述直接接触；Na⁺ 仅用于中和体系净电荷'
              '（Mg 体系 5 个、noMg 体系 7 个），未模拟体相离子强度，'
              '不能据此判断静电屏蔽。',
              '- Mg 配位统计区分水 O、ATP/NMN O、蛋白 O 与蛋白 N；'
              'K104 NZ 不当作氧供体。',
              '- Mg 共配位帧比例只表示同一帧内 Mg 同时接触 ATP 与 NMN 的氧，'
              '不证明桥接配位合理或区分内外球，也不代表金属占有率。',
              '- RMSD 参照是 EM 结构，不是原始 Boltz 模型。', '']
    (Path(analysis) / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def analyze_nmnat_md(config_path, output):
    output = Path(output).resolve()
    mapping = json.load(open(output / 'atom_mapping.json'))
    runs = list(csv.DictReader(open(output / 'md_runs.csv')))
    systems = {row['model']: row['directory'] for row in
               csv.DictReader(open(output / 'systems.csv'))}
    frames = []
    summaries = []
    mg_ligand_rows = []
    mg_trajectories = []
    for run in runs:
        if run.get('status') != 'DONE':
            continue
        run_dir = Path(run['directory'])
        if not (run_dir / 'production.xtc').exists():
            continue
        rows, ligand_frames, frames_seen, has_mg = _run_frames(
            run_dir, systems[run['model']], mapping,
            run['group'], run['model'], int(run['velocity_seed']))
        frames.extend(rows)
        if has_mg:
            mg_trajectories.append({
                'group': run['group'], 'model': run['model'],
                'velocity_seed': run['velocity_seed']})
        for key, count in ligand_frames.items():
            mg_ligand_rows.append({
                'group': run['group'], 'model': run['model'],
                'velocity_seed': run['velocity_seed'], 'ligand_atom': key,
                'frames': count, 'total_frames': frames_seen,
                'fraction': count / frames_seen})
        frame = pd.DataFrame(rows)
        for candidate, part in frame.groupby('candidate'):
            summaries.append({
                'group': run['group'], 'model': run['model'],
                'velocity_seed': run['velocity_seed'], 'candidate': candidate,
                'frames': len(part),
                'time_min_ps': part['time_ps'].min(),
                'time_max_ps': part['time_ps'].max(),
                'pa_onuc_median_A': part['pa_onuc_A'].median(),
                'pa_onuc_min_A': part['pa_onuc_A'].min(),
                'angle_median_deg': part['pa_o3a_onuc_deg'].median(),
                'angle_max_deg': part['pa_o3a_onuc_deg'].max(),
                'cross_median_A': part['cross_min_A'].median(),
                'cross_lt_2_8_fraction': float((part['cross_min_A'] < 2.8).mean()),
                'phosphate_min_median_A': part['phosphate_min_A'].median(),
                'phosphate_min_min_A': part['phosphate_min_A'].min(),
                'phosphate_lt_2_8_fraction': float(
                    (part['phosphate_min_A'] < 2.8).mean()),
                'na_phosphate_median_A': part['na_phosphate_min_A'].median(),
                'na_phosphate_frame_min_A': part['na_phosphate_min_A'].min(),
                'na_within_4_median': part['na_within_4'].median(),
                'atp_pose_median_A': part['atp_pose_rmsd_A'].median(),
                'nmn_pose_median_A': part['nmn_pose_rmsd_A'].median(),
                'atp_internal_median_A': part['atp_internal_rmsd_A'].median(),
                'nmn_internal_median_A': part['nmn_internal_rmsd_A'].median(),
                'atp_pose_lt_3_fraction': float(
                    (part['atp_pose_rmsd_A'] < 3.0).mean()),
                'nmn_pose_lt_3_fraction': float(
                    (part['nmn_pose_rmsd_A'] < 3.0).mean()),
                'rmsd_reference': 'EM',
                'mg_o_coord_median': part['mg_o_coord'].median()
                if 'mg_o_coord' in part else None,
                'mg_water_o_median': part['mg_water_o'].median()
                if 'mg_water_o' in part else None,
                'mg_ligand_o_median': (part['mg_atp_o'] + part['mg_nmn_o']).median()
                if 'mg_atp_o' in part else None,
                'mg_protein_o_median': part['mg_protein_o'].median()
                if 'mg_protein_o' in part else None,
                'mg_protein_n_median': part['mg_protein_n'].median()
                if 'mg_protein_n' in part else None,
                'mg_water_o_gt0_fraction': float(
                    (part['mg_water_o'] > 0).mean())
                if 'mg_water_o' in part else None,
                'mg_bridging_fraction': float(part['mg_bridging'].mean())
                if 'mg_bridging' in part else None,
            })
    trajectory_count, physical_frames, candidate_rows = _analysis_counts(frames)
    analysis = output / 'analysis'
    analysis.mkdir(exist_ok=True)
    pd.DataFrame(frames).to_csv(analysis / 'md_frames.csv', index=False)
    ligands = pd.DataFrame(mg_ligand_rows)
    if len(ligands):
        ligands.to_csv(analysis / 'md_mg_ligands.csv', index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(analysis / 'md_summary.csv', index=False)
    if len(summary):
        group = summary.groupby(['group', 'candidate']).agg(
            trajectories=('velocity_seed', 'size'),
            pa_onuc_median_A=('pa_onuc_median_A', 'median'),
            angle_median_deg=('angle_median_deg', 'median'),
            cross_median_A=('cross_median_A', 'median'),
            phosphate_min_median_A=('phosphate_min_median_A', 'median'),
            na_phosphate_median_A=('na_phosphate_median_A', 'median'),
            mg_water_o_median=('mg_water_o_median', 'median'),
            mg_bridging_fraction=('mg_bridging_fraction', 'median'),
        ).reset_index()
        group.to_csv(analysis / 'md_group_summary.csv', index=False)
    _write_report(output, analysis, summary, trajectory_count,
                  physical_frames, candidate_rows, ligands, mg_trajectories)
    write_json(analysis / 'md_analysis_identity.json', {
        'code': str(Path(__file__).resolve()),
        'code_sha256': sha256_file(Path(__file__)),
        'config': str(Path(config_path).resolve()),
        'config_sha256': sha256_file(config_path),
        'md_runs_sha256': sha256_file(output / 'md_runs.csv'),
        'atom_mapping_sha256': sha256_file(output / 'atom_mapping.json'),
        'trajectories': trajectory_count,
        'physical_frames': physical_frames,
        'candidate_rows': candidate_rows,
        'candidate_summaries': len(summary),
        'mg_ligand_rows': len(ligands),
        'rmsd_reference': 'EM structure (not the original Boltz model)',
        'limits': ['1 ns is a local initial-response window, not equilibrium',
                   'no ligand dissociation/occupancy or catalytic activity claim',
                   'one Mg at its predicted position; no 12 mM buffer claim',
                   'initial groups are fixed from the pre-simulation manifest'],
    })
    return {'trajectories': trajectory_count,
            'physical_frames': physical_frames,
            'candidate_rows': candidate_rows,
            'candidate_summaries': len(summary),
            'mg_ligand_rows': len(ligands)}
