"""Build a unified, provenance-tracked NatV candidate delivery package."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .data import load_yaml, sha256_file, write_json
from .delivery_coordinates import convert_coordinate as _convert_coordinate

DESIGN_NAMES = ('WT', 'A1', 'A2', 'A3', 'B1', 'B2',
                'A1B1', 'A1B2', 'A2B1', 'A2B2', 'A3B1', 'A3B2')
CLONE_NAMES = ('NatV10', 'NatV89')
CATEGORIES = ('shared_reference', 'computational_design', 'epcr_clone')
FIELDS = ('candidate_id', 'source_category', 'group', 'full_protein_edits',
          'parent_a', 'parent_b', 'source_run', 'source_table_record',
          'generation_candidate_id', 'dna_edits', 'protein_length',
          'sequence_sha256', 'sequence', 'notes')
REPO_URL = 'https://github.com/Yhyb24P/natvqvae'


def _read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def _edit_set(text):
    text = (text or '').strip()
    if text in ('', 'none'):
        return set()
    result = set()
    for item in text.split(';'):
        match = re.fullmatch(r'\s*([A-Z])(\d+)([A-Z])\s*', item)
        if match is None:
            raise ValueError(f'Invalid protein edit: {item!r}')
        result.add((int(match.group(2)), match.group(1), match.group(3)))
    return result


def _sequence_edits(wt, sequence, label):
    if len(sequence) != 341:
        raise ValueError(f'{label} is not 341 aa')
    return {(position, wt[position - 1], sequence[position - 1])
            for position in range(1, 342) if wt[position - 1] != sequence[position - 1]}


def _edits_text(edits):
    if not edits:
        return 'none'
    return ';'.join(f'{wt}{position}{alt}' for position, wt, alt in sorted(edits))


def _read_fasta(path):
    entries = {}
    name = None
    sequence = []
    for line in Path(path).read_text().splitlines():
        if line.startswith('>'):
            if name is not None:
                entries[name] = ''.join(sequence)
            name = line[1:].strip()
            sequence = []
        elif line.strip():
            sequence.append(line.strip())
    if name is not None:
        entries[name] = ''.join(sequence)
    return entries


def _copy_item(item, output):
    source = Path(item['source'])
    package_path = Path(item['package'])
    if not source.is_file():
        raise FileNotFoundError(source)
    if package_path.is_absolute() or '..' in package_path.parts:
        raise ValueError(f'Invalid package path: {package_path}')
    destination = output / package_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    record = {
        'package': package_path.as_posix(),
        'source': str(source),
        'source_sha256': sha256_file(source),
        'sha256': sha256_file(destination),
    }
    if item.get('description'):
        record['description'] = item['description']
    return record


def _github_url(commit, path):
    return f'{REPO_URL}/blob/{commit}/{path}' if commit else f'{REPO_URL}/blob/main/{path}'


METRIC_FIELDS = (
    'candidate_id', 'source', 'prediction_batch', 'comparison_reference',
    'plddt_median_0_1', 'ptm_median_0_1',
    'rmsd_vs_wt_full_median_A', 'rmsd_vs_wt_full_min_A', 'rmsd_vs_wt_full_max_A',
    'rmsd_vs_wt_full_model_pairs',
    'rmsd_vs_wt_nmnat_median_A', 'rmsd_vs_wt_nmnat_min_A', 'rmsd_vs_wt_nmnat_max_A',
    'rmsd_vs_wt_nmnat_model_pairs',
    'rmsd_vs_wt_nudix_median_A', 'rmsd_vs_wt_nudix_min_A', 'rmsd_vs_wt_nudix_max_A',
    'rmsd_vs_wt_nudix_model_pairs',
    'rmsd_vs_parent_a_full_median_A', 'rmsd_vs_parent_a_full_min_A',
    'rmsd_vs_parent_a_full_max_A', 'rmsd_vs_parent_a_full_model_pairs',
    'rmsd_vs_parent_a_nmnat_median_A', 'rmsd_vs_parent_a_nmnat_min_A',
    'rmsd_vs_parent_a_nmnat_max_A', 'rmsd_vs_parent_a_nmnat_model_pairs',
    'rmsd_vs_parent_a_nudix_median_A', 'rmsd_vs_parent_a_nudix_min_A',
    'rmsd_vs_parent_a_nudix_max_A', 'rmsd_vs_parent_a_nudix_model_pairs',
    'self_full_median_A', 'self_full_min_A', 'self_full_max_A', 'self_full_model_pairs',
    'self_nmnat_median_A', 'self_nmnat_min_A', 'self_nmnat_max_A',
    'self_nmnat_model_pairs',
    'self_nudix_median_A', 'self_nudix_min_A', 'self_nudix_max_A',
    'self_nudix_model_pairs',
    'domain_centroid_distance_median_A', 'domain_centroid_distance_wt_reference_A',
    'local_site_count', 'local_sites_above_baseline',
    'local_sites_above_baseline_list', 'max_local_ca_displacement_A',
    'max_local_site', 'baseline_ca_displacement_at_max_local_site_A',
    'local_metrics_status',
)
LOCAL_FIELDS = (
    'candidate_id', 'edit', 'position', 'domain', 'prediction_batch',
    'ca_displacement_median_A', 'ca_displacement_min_A', 'ca_displacement_max_A',
    'neighborhood_rmsd_median_A', 'neighborhood_rmsd_min_A',
    'neighborhood_rmsd_max_A',
    'self_baseline_ca_displacement_median_A',
    'self_baseline_ca_displacement_min_A',
    'self_baseline_ca_displacement_max_A',
    'self_baseline_neighborhood_rmsd_median_A',
    'displacement_range_disjoint_from_baseline', 'disjoint_criterion', 'model_pairs',
)
NOT_APPLICABLE = '不适用'
ABOVE_BASELINE_CRITERION = ('位移范围与自身基准不重叠：跨序列位移最小值大于同序列同位置'
                            '自身基准最大值；仅为描述性筛选，不是显著性检验')
NOT_COMPUTED = '未计算'


def _fmt(value, digits=4):
    if value is None or value == '':
        return ''
    try:
        return f'{float(value):.{digits}f}'
    except (TypeError, ValueError):
        return str(value)


def _metrics_tables(config, sources, batch_by_candidate):
    design_rows = _read_csv(sources['design_metrics']['path'])
    design_by_name = {row['candidate']: row for row in design_rows}
    comparison_rows = _read_csv(sources['design_comparisons']['path'])
    comparison_by_key = {(row['left'], row['right'], row['region']): row
                         for row in comparison_rows}
    clone_summary = json.loads(Path(sources['clone_summary']['path']).read_text())

    clone_pairs = {(item['variant'], item['region']): item
                   for item in clone_summary['pairs']}
    clone_self = {(item['name'], item['region']): item
                  for item in clone_summary['self_rmsd']}
    clone_confidence = {(item['name'], item['metric']): item
                        for item in clone_summary['confidence']}
    clone_geometry = {}
    for item in clone_summary['domain_geometry']:
        clone_geometry.setdefault(item['name'], []).append(
            item['domain_centroid_distance_A'])
    local_rows = {(item['variant'], item['position']): item
                  for item in clone_summary['mutation_local']}
    baseline_rows = {(item['name'], item['position']): item
                     for item in clone_summary['mutation_local_baseline']}

    def _comparison_columns(row, left, right, prefix):
        for region in ('full', 'NMNAT', 'Nudix'):
            key_region = region.lower()
            item = comparison_by_key.get((left, right, region))
            if item is None:
                for suffix in ('median_A', 'min_A', 'max_A'):
                    row[f'{prefix}_{key_region}_{suffix}'] = ''
                row[f'{prefix}_{key_region}_model_pairs'] = ''
                continue
            row[f'{prefix}_{key_region}_median_A'] = _fmt(item['median_rmsd_A'])
            row[f'{prefix}_{key_region}_min_A'] = _fmt(item['min_rmsd_A'])
            row[f'{prefix}_{key_region}_max_A'] = _fmt(item['max_rmsd_A'])
            row[f'{prefix}_{key_region}_model_pairs'] = item['model_pairs']

    def _not_applicable(row, prefix):
        for region in ('full', 'nmnat', 'nudix'):
            for suffix in ('median_A', 'min_A', 'max_A', 'model_pairs'):
                row[f'{prefix}_{region}_{suffix}'] = NOT_APPLICABLE

    source_labels = {'WT': '野生型', 'A': 'QVAE 生成（A 类）', 'B': '机制假说设计（B 类）',
                     'AB': 'A+B 组合', 'ePCR': '易错 PCR 克隆'}
    rows = []
    local_table = []
    for name in ('WT', *DESIGN_NAMES[1:], *CLONE_NAMES):
        batch = batch_by_candidate[name]
        if name in design_by_name:
            design = design_by_name[name]
            group = design['group']
            if group in ('AB', 'A+B'):
                group = 'AB'
                parent_a = name[:2]
            row = {
                'candidate_id': name,
                'source': source_labels[group],
                'prediction_batch': batch,
                'comparison_reference': '自身（WT 为基准）' if name == 'WT' else 'WT',
                'plddt_median_0_1': _fmt(design['plddt_median_0_1']),
                'ptm_median_0_1': _fmt(design['ptm_median_0_1']),
                'domain_centroid_distance_median_A': NOT_COMPUTED,
                'domain_centroid_distance_wt_reference_A': NOT_COMPUTED,
                'local_metrics_status': NOT_COMPUTED,
            }
            if name == 'WT':
                for region in ('full', 'nmnat', 'nudix'):
                    for statistic in ('median', 'min', 'max'):
                        row[f'rmsd_vs_wt_{region}_{statistic}_A'] = _fmt(
                            design[f'comparison_{region}_{statistic}_A'])
                    row[f'rmsd_vs_wt_{region}_model_pairs'] = design[
                        f'comparison_{region}_model_pairs']
            elif group == 'AB':
                _comparison_columns(row, name, 'WT', 'rmsd_vs_wt')
                _comparison_columns(row, name, parent_a, 'rmsd_vs_parent_a')
            else:
                _comparison_columns(row, name, 'WT', 'rmsd_vs_wt')
                _not_applicable(row, 'rmsd_vs_parent_a')
            for region in ('full', 'nmnat', 'nudix'):
                for statistic in ('median', 'min', 'max'):
                    row[f'self_{region}_{statistic}_A'] = _fmt(
                        design[f'self_{region}_{statistic}_A'])
                row[f'self_{region}_model_pairs'] = design[f'self_{region}_model_pairs']
            for key in ('local_site_count', 'local_sites_above_baseline',
                        'local_sites_above_baseline_list', 'max_local_ca_displacement_A',
                        'max_local_site',
                        'baseline_ca_displacement_at_max_local_site_A'):
                row[key] = NOT_COMPUTED if key in ('local_site_count',
                                                   'local_sites_above_baseline',
                                                   'local_sites_above_baseline_list') else ''
        else:
            confidence_ptm = clone_confidence.get((name, 'ptm'))
            confidence_plddt = clone_confidence.get((name, 'complex_plddt'))
            row = {
                'candidate_id': name,
                'source': source_labels['ePCR'],
                'prediction_batch': batch,
                'comparison_reference': 'WT',
                'plddt_median_0_1': _fmt(confidence_plddt['median']) if confidence_plddt else '',
                'ptm_median_0_1': _fmt(confidence_ptm['median']) if confidence_ptm else '',
            }
            for region in ('full', 'NMNAT', 'Nudix'):
                pair = clone_pairs.get((name, region))
                self_row = clone_self.get((name, region))
                key_region = region.lower()
                if pair is not None:
                    row[f'rmsd_vs_wt_{key_region}_median_A'] = _fmt(pair['median_rmsd_A'])
                    row[f'rmsd_vs_wt_{key_region}_min_A'] = _fmt(pair['min_rmsd_A'])
                    row[f'rmsd_vs_wt_{key_region}_max_A'] = _fmt(pair['max_rmsd_A'])
                    row[f'rmsd_vs_wt_{key_region}_model_pairs'] = pair['model_pairs']
                if self_row is not None:
                    row[f'self_{key_region}_median_A'] = _fmt(self_row['median_rmsd_A'])
                    row[f'self_{key_region}_min_A'] = _fmt(self_row['min_rmsd_A'])
                    row[f'self_{key_region}_max_A'] = _fmt(self_row['max_rmsd_A'])
                    row[f'self_{key_region}_model_pairs'] = self_row['model_pairs']
            _not_applicable(row, 'rmsd_vs_parent_a')
            distances = clone_geometry.get(name, [])
            wt_distances = clone_geometry.get('WT', [])
            row['domain_centroid_distance_median_A'] = (
                _fmt(sorted(distances)[len(distances) // 2]) if distances else '')
            row['domain_centroid_distance_wt_reference_A'] = (
                _fmt(sorted(wt_distances)[len(wt_distances) // 2]) if wt_distances else '')

            sites = [item for (variant, _), item in local_rows.items() if variant == name]
            above = []
            maximum = None
            for site in sites:
                position = site['position']
                baseline = baseline_rows.get((name, position))
                baseline_median = (baseline['self_ca_displacement_median_A']
                                   if baseline else None)
                baseline_max = (baseline['self_ca_displacement_range_A'][1]
                                if baseline else None)
                displacement = site['ca_displacement_median_A']
                displacement_min = site['ca_displacement_range_A'][0]
                is_above = baseline_max is not None and displacement_min > baseline_max
                if is_above:
                    above.append(site['edit'])
                if maximum is None or displacement > maximum[0]:
                    maximum = (displacement, site['edit'], baseline_median)
                local_table.append({
                    'candidate_id': name,
                    'edit': site['edit'],
                    'position': position,
                    'domain': site['domain'],
                    'prediction_batch': batch,
                    'ca_displacement_median_A': _fmt(displacement),
                    'ca_displacement_min_A': _fmt(displacement_min),
                    'ca_displacement_max_A': _fmt(site['ca_displacement_range_A'][1]),
                    'neighborhood_rmsd_median_A': _fmt(site['neighborhood_rmsd_median_A']),
                    'neighborhood_rmsd_min_A': _fmt(site['neighborhood_rmsd_range_A'][0]),
                    'neighborhood_rmsd_max_A': _fmt(site['neighborhood_rmsd_range_A'][1]),
                    'self_baseline_ca_displacement_median_A': _fmt(baseline_median),
                    'self_baseline_ca_displacement_min_A': _fmt(
                        baseline['self_ca_displacement_range_A'][0]) if baseline else '',
                    'self_baseline_ca_displacement_max_A': _fmt(baseline_max),
                    'self_baseline_neighborhood_rmsd_median_A': _fmt(
                        baseline['self_neighborhood_rmsd_median_A']) if baseline else '',
                    'displacement_range_disjoint_from_baseline': '是' if is_above else '否',
                    'disjoint_criterion': ABOVE_BASELINE_CRITERION,
                    'model_pairs': site['model_pairs'],
                })
            row['local_site_count'] = len(sites)
            row['local_sites_above_baseline'] = len(above)
            row['local_sites_above_baseline_list'] = ';'.join(sorted(above))
            row['max_local_ca_displacement_A'] = _fmt(maximum[0]) if maximum else ''
            row['max_local_site'] = maximum[1] if maximum else ''
            row['baseline_ca_displacement_at_max_local_site_A'] = (
                _fmt(maximum[2]) if maximum and maximum[2] is not None else '')
            row['local_metrics_status'] = '已计算'
        rows.append({field: row.get(field, '') for field in METRIC_FIELDS})
    local_table.sort(key=lambda item: (item['candidate_id'], item['position']))
    return rows, local_table


def _write_metrics(rows, local_table, output):
    summary_path = output / 'candidate_metrics_summary.csv'
    with summary_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(METRIC_FIELDS),
                                lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    local_path = output / 'mutation_local_metrics.csv'
    with local_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LOCAL_FIELDS),
                                lineterminator='\n')
        writer.writeheader()
        writer.writerows(local_table)
    return summary_path, local_path


def _readme(rows, attachments, commit):
    lines = [
        '# NatV 蛋白序列与结构资料',
        '',
        '本资料包包括野生型（WT）、11 条设计变体和 NatV10、NatV89 两条易错 PCR 克隆，'
        '共 14 条不同的蛋白序列，每条均为 341 个氨基酸。WT 是两组研究的共同序列对照。',
        '',
        '## 蛋白来源与氨基酸替换',
        '',
        '编号从 WT 的第 1 个氨基酸开始。例如，L18I 表示第 18 位亮氨酸被异亮氨酸替换。',
        '',
        '| 名称 | 来源 | 完整氨基酸替换 | 比较对象 |',
        '|---|---|---|---|',
    ]
    for row in rows:
        group = row['group']
        if row['candidate_id'] == 'WT':
            origin, comparison = '野生型', '—'
        elif row['source_category'] == 'epcr_clone':
            origin, comparison = '易错 PCR 克隆', 'WT'
        elif group == 'A':
            origin, comparison = 'QVAE 生成序列', 'WT'
        elif group == 'B':
            origin, comparison = '依据同源酶机制设计', 'WT'
        else:
            origin = 'A 与 B 的组合'
            comparison = f"WT、{row['parent_a']}、{row['parent_b']}"
        edits = '无' if row['full_protein_edits'] == 'none' else row['full_protein_edits']
        lines.append(f"| {row['candidate_id']} | {origin} | {edits} | {comparison} |")
    lines += [
        '',
        'A1–A3 的生成记录、B1/B2 的设计依据以及两条克隆的 Sanger 测序核查记录保存在 '
        '`evidence/`。A+B 是相应 A、B 变体的组合，并非另一次 QVAE 生成的结果。'
        'NatV89 另有 DNA 同义变化 561c>t（对应 I187I），不改变蛋白序列，单独记录在 CSV 的 DNA 变化栏。',
        '',
        '## 文件怎么看',
        '',
        '| 文件或文件夹 | 内容和用途 |',
        '|---|---|',
        '| `candidates.csv` | 14 条完整序列、所有氨基酸替换、来源、亲本和序列校验值（SHA-256） |',
        '| `candidates.fasta` | 同一组 14 条序列，可导入序列分析软件 |',
        '| `candidate_metrics_summary.csv` | 14 条候选的置信度、与 WT 的全长/分域 RMSD、'
        'A+B 另列对亲本 A 的 RMSD、自身预测离散、结构域质心距离（克隆含 ePCR 批次 WT 参照）'
        '和局部差异汇总，含来源类别与预测批次 |',
        '| `mutation_local_metrics.csv` | NatV10/NatV89 各位点的 Cα 位移、邻域 RMSD、'
        '同序列自身离散基准，以及“位移范围与自身基准是否不重叠”的描述性筛选 |',
        '| `coordinates/` | 每条蛋白 3 个预测结构，共 42 个 PDB 文件 |',
        '| `references/` | 易错 PCR 研究中另外预测的 3 个 WT 结构，供复算该批比较结果 |',
        '| `evidence/` | 原始序列表、生成和设计记录、Sanger 测序核查记录 |',
        '| `attachments/` | 全长结构图、突变位点图和 RMSD 结果 |',
        '| `manifest.json` | 文件来源、模型编号和校验值，用于确认文件与原资料一致 |',
        '',
        'PDB 文件可用 PyMOL 或 ChimeraX 打开。文件名中的 `model_0`、`model_1`、`model_2` '
        '是同一条件下的 3 个预测样本，不是实验重复。所有 PDB 均保留原预测坐标，没有重新拟合或松弛。',
        '',
        '42 个候选结构包括设计集合的 36 个结构和 NatV10/NatV89 的 6 个结构。'
        '`references/` 中的 3 个 WT 来自易错 PCR 研究的另一批预测；'
        '它们与设计集合的 WT 序列相同，但坐标不同。复算时应使用原比较所对应的 WT。',
        '',
        'RMSD 表示相应区域的 Cα 原子经刚体叠合后的位置差异，单位为 Å。'
        '跨序列比较使用 3×3＝9 个模型对；这些模型对不是 9 次独立实验。'
        '设计集合与易错 PCR 克隆分别按各自预测批次比较，具体数值见附件。',
        '',
        '## 研究这些变体要回答什么',
        '',
        '- A1–A3：NMNAT 域的氨基酸替换是否改变 NAD⁺ 合成能力？',
        '- B1/B2：Nudix 域的 E248Q、D308N 是否减少 NAD⁺ 水解？',
        '- A+B：两类替换同时存在时，是否保留合成能力，并增加指定反应条件下的 NAD⁺ 积累？',
        '',
        '**目前没有任何经验证的活性排序。**结构接近不能证明活性保持或增强。'
        '完整 NatV 同时合成和水解 NAD⁺，因此 NAD⁺ 净积累不能直接当作 NMNAT 的独立合成速率。'
        '现有 A570 终点吸光记录也不足以判断这些蛋白的活性高低。',
        '',
        '## 图和测序证据',
        '',
        '| 文件 | 内容 |',
        '|---|---|',
    ]
    for item in attachments:
        lines.append(f"| [`{Path(item['package']).name}`]({item['package']}) "
                     f"| {item.get('description', '结构图或比较结果')} |")
    lines += [
        '',
        '结构图和 PDB 均来自预测。NMNAT 配体局部图另取自 WT 的 ATP/NMN 联合预测，'
        '不属于上述 42 个无配体单体结构。',
        '',
        '`evidence/clone_sanger_verification.json` 保存测序位点与读段质量的核查结果；'
        '`clone_variants.csv` 保存克隆序列和替换，`clone_build.json` 保存序列重建与计算记录。'
        '原始 `.ab1` 测序峰图未放入本包，仍保存在原始实验材料中。'
        '其余文件记录 A 类序列的生成来源及 B 类设计依据；这些设计依据仍是假说，尚未证明能提高活性。',
        '',
        '## 需要详细方法时查阅',
        '',
        f'- [设计与结构比较]({_github_url(commit, "data/results/evaluation/natv_dual_domain_design/REPORT.md")})',
        f'- [NatV10/NatV89 全长结构比较]({_github_url(commit, "data/results/evaluation/kvp40_epcr_full_length_structures/REPORT.md")})',
        f'- [NatV10/NatV89 的 NMNAT 底物环境]({_github_url(commit, "data/results/evaluation/kvp40_epcr_nmnat_environment/REPORT.md")})',
        f'- [功能比较的说明]({_github_url("main", "data/results/evaluation/natv_dual_domain_design/demo/candidate_handoff.md")})',
    ]
    return '\n'.join(lines) + '\n'


def _write_fasta(rows, path):
    lines = []
    for row in rows:
        lines.append(f">{row['candidate_id']}|{row['source_category']}|{row['group']}")
        sequence = row['sequence']
        lines.extend(sequence[index:index + 60] for index in range(0, len(sequence), 60))
    Path(path).write_text('\n'.join(lines) + '\n')


def _git_head():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root,
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def build_candidate_delivery(config_path, output):
    config_path = Path(config_path)
    config = load_yaml(config_path)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Delivery output already contains files: {output}')
    output.mkdir(parents=True, exist_ok=True)

    sources = {}
    for key, value in config['sources'].items():
        if key == 'clone_source_run':
            continue
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(path)
        sources[key] = {'path': str(path), 'sha256': sha256_file(path)}

    design_rows = _read_csv(sources['design_candidates']['path'])
    design_names = [row['name'] for row in design_rows]
    if design_names != list(DESIGN_NAMES):
        raise ValueError(f'Unexpected design candidate names/order: {design_names}')
    design_by_name = {row['name']: row for row in design_rows}
    design_fasta = _read_fasta(sources['design_fasta']['path'])
    if set(design_fasta) != set(DESIGN_NAMES):
        raise ValueError('Design FASTA names differ from candidates.csv')
    for name, sequence in design_fasta.items():
        if sequence != design_by_name[name]['sequence']:
            raise ValueError(f'Design FASTA sequence differs: {name}')

    clone_rows = _read_csv(sources['clone_variants']['path'])
    clone_by_name = {row['name']: row for row in clone_rows}
    if set(clone_by_name) != {'WT', *CLONE_NAMES}:
        raise ValueError('Clone variants must contain WT/NatV10/NatV89 only')
    wt_sequence = design_by_name['WT']['sequence']
    if wt_sequence != clone_by_name['WT']['sequence']:
        raise ValueError('Shared WT sequences differ between sources')
    if design_by_name['WT']['sequence_sha256'] != clone_by_name['WT']['sequence_sha256']:
        raise ValueError('Shared WT SHA differs between sources')

    rows = [{
        'candidate_id': 'WT',
        'source_category': 'shared_reference',
        'group': 'WT',
        'full_protein_edits': 'none',
        'parent_a': '',
        'parent_b': '',
        'source_run': '',
        'source_table_record': 'candidates.csv:WT; variants.csv:WT',
        'generation_candidate_id': '',
        'dna_edits': '',
        'protein_length': '341',
        'sequence_sha256': design_by_name['WT']['sequence_sha256'],
        'sequence': wt_sequence,
        'notes': 'shared reference for both groups',
    }]

    for name in DESIGN_NAMES[1:]:
        source = design_by_name[name]
        sequence = source['sequence']
        if len(sequence) != 341:
            raise ValueError(f'{name} design sequence is not 341 aa')
        if hashlib.sha256(sequence.encode()).hexdigest() != source['sequence_sha256']:
            raise ValueError(f'{name} sequence SHA does not match source value')
        expected = _edit_set(source['edits'])
        actual = _sequence_edits(wt_sequence, sequence, name)
        if actual != expected:
            raise ValueError(f'{name} edits differ from WT-to-sequence changes')
        group = source['group']
        if group in ('A', 'B'):
            parent_a, parent_b = 'WT', ''
            output_group = group
        elif group in ('AB', 'A+B'):
            match = re.fullmatch(r'(A\d)(B\d)', name)
            if match is None:
                raise ValueError(f'Invalid A+B name: {name}')
            parent_a, parent_b = match.groups()
            output_group = 'AB'
        else:
            raise ValueError(f'Unexpected design group: {name} {group}')
        if parent_a != 'WT' and parent_a not in design_by_name:
            raise ValueError(f'Missing parent A: {parent_a}')
        if parent_b and parent_b not in design_by_name:
            raise ValueError(f'Missing parent B: {parent_b}')
        note = ''
        if output_group == 'AB':
            note = ('combination; source_run and generation_candidate_id inherited '
                    'from parent A, not an independent QVAE generation')
        rows.append({
            'candidate_id': name,
            'source_category': 'computational_design',
            'group': output_group,
            'full_protein_edits': _edits_text(actual),
            'parent_a': parent_a,
            'parent_b': parent_b,
            'source_run': source['source_run'],
            'source_table_record': f'candidates.csv:{name}',
            'generation_candidate_id': source['source_candidate'],
            'dna_edits': '',
            'protein_length': '341',
            'sequence_sha256': source['sequence_sha256'],
            'sequence': sequence,
            'notes': note,
        })

    for name in CLONE_NAMES:
        source = clone_by_name[name]
        sequence = source['sequence']
        if len(sequence) != 341:
            raise ValueError(f'{name} clone sequence is not 341 aa')
        if hashlib.sha256(sequence.encode()).hexdigest() != source['sequence_sha256']:
            raise ValueError(f'{name} sequence SHA does not match source value')
        actual = _sequence_edits(wt_sequence, sequence, name)
        if actual != _edit_set(source['protein_edits']):
            raise ValueError(f'{name} protein edits differ from WT-to-sequence changes')
        note = 'independent ePCR clone; WT is comparison reference only'
        if name == 'NatV89':
            note += '; DNA 561c>t is synonymous (I187I), not a protein edit'
        rows.append({
            'candidate_id': name,
            'source_category': 'epcr_clone',
            'group': 'ePCR',
            'full_protein_edits': _edits_text(actual),
            'parent_a': '',
            'parent_b': '',
            'source_run': config['sources']['clone_source_run'],
            'source_table_record': f'variants.csv:{name}',
            'generation_candidate_id': '',
            'dna_edits': source['dna_edits'],
            'protein_length': '341',
            'sequence_sha256': source['sequence_sha256'],
            'sequence': sequence,
            'notes': note,
        })

    if len(rows) != 14:
        raise ValueError(f'Expected 14 unique proteins, got {len(rows)}')
    if len({row['sequence'] for row in rows}) != 14:
        raise ValueError('Delivery contains duplicated protein sequences')
    if len({row['sequence_sha256'] for row in rows}) != 14:
        raise ValueError('Delivery contains duplicated sequence SHA values')
    for row in rows:
        if hashlib.sha256(row['sequence'].encode()).hexdigest() != row['sequence_sha256']:
            raise ValueError(f"{row['candidate_id']} sequence SHA does not match its sequence")

    coordinate_entries = config['coordinates']
    if {item['candidate'] for item in coordinate_entries} != {row['candidate_id'] for row in rows}:
        raise ValueError('Coordinate entries must cover exactly the 14 unified candidates')
    coordinates = []
    for item in coordinate_entries:
        candidate = item['candidate']
        expected = next(row['sequence'] for row in rows if row['candidate_id'] == candidate)
        for model_index in (0, 1, 2):
            source = item['source_pattern'].format(model=model_index)
            coordinates.append(_convert_coordinate(
                candidate, model_index, source, item['batch'], output, expected, 'coordinates'))
    if len(coordinates) != 42:
        raise ValueError(f'Expected 42 candidate coordinates, got {len(coordinates)}')

    reference_coordinates = []
    for item in config.get('reference_coordinates', []):
        name = item['name']
        expected = wt_sequence
        for model_index in (0, 1, 2):
            source = item['source_pattern'].format(model=model_index)
            record = _convert_coordinate(name, model_index, source, item['batch'],
                                         output, expected, 'references')
            record['note'] = item.get('description', '')
            reference_coordinates.append(record)

    candidates_path = output / 'candidates.csv'
    with candidates_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    fasta_path = output / 'candidates.fasta'
    _write_fasta(rows, fasta_path)

    batch_by_candidate = {item['candidate']: item['batch'] for item in coordinate_entries}
    metric_rows, local_table = _metrics_tables(config, sources, batch_by_candidate)
    if len(metric_rows) != 14:
        raise ValueError(f'Expected 14 metric rows, got {len(metric_rows)}')
    metrics_path, local_metrics_path = _write_metrics(metric_rows, local_table, output)

    copied = []
    for item in [*config.get('evidence', []), *config.get('attachments', [])]:
        copied.append(_copy_item(item, output))
    evidence = [item for item in copied if item['package'].startswith('evidence/')]
    attachments = [item for item in copied if item['package'].startswith('attachments/')]
    if len(evidence) != len(config.get('evidence', [])):
        raise ValueError('Evidence copy failed')
    if len(attachments) != len(config.get('attachments', [])):
        raise ValueError('Attachment copy failed')

    commit = _git_head()
    readme_path = output / 'README.md'
    readme_path.write_text(_readme(rows, attachments, commit))

    files = {}
    for path in sorted(output.rglob('*')):
        if path.is_file() and path.name != 'manifest.json':
            files[path.relative_to(output).as_posix()] = sha256_file(path)
    manifest = {
        'package': 'natv_candidate_delivery',
        'description': 'Unified handoff snapshot of 12 design candidates and 2 ePCR clones '
                       'with one shared WT reference; no new predictions or functional claims.',
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'source_data_commit': commit,
        'source_data_commit_note': 'commit of the tracked source files listed in sources/evidence; '
                                   'not the packaging code version',
        'packaging': {
            'module': 'src/natvqvae/delivery.py',
            'module_sha256': sha256_file(Path(__file__)),
            'coordinate_module': 'src/natvqvae/delivery_coordinates.py',
            'coordinate_module_sha256': sha256_file(
                Path(__file__).with_name('delivery_coordinates.py')),
            'config': str(config_path),
            'config_sha256': sha256_file(config_path),
            'command': 'natvqvae delivery-build --config '
                       f'{config_path} --output {output}',
        },
        'counts': {
            'unique_proteins': len(rows),
            'shared_reference': sum(row['source_category'] == 'shared_reference' for row in rows),
            'computational_design': sum(row['source_category'] == 'computational_design' for row in rows),
            'epcr_clone': sum(row['source_category'] == 'epcr_clone' for row in rows),
            'coordinates': len(coordinates),
            'reference_coordinates': len(reference_coordinates),
            'total_coordinate_files': len(coordinates) + len(reference_coordinates),
        },
        'schema': list(FIELDS),
        'sources': sources,
        'metrics_sources': {
            'design_metrics': sources['design_metrics'],
            'design_comparisons': sources['design_comparisons'],
            'clone_summary': sources['clone_summary'],
            'note': 'candidate_metrics_summary.csv and mutation_local_metrics.csv are '
                    'compiled from these existing result files; no new computation. '
                    'A+B rows carry two independent comparisons: rmsd_vs_wt_* is the '
                    'A+B-versus-WT supplementary comparison and rmsd_vs_parent_a_* is '
                    'the A+B-versus-parent-A primary comparison; A/B candidates have '
                    'rmsd_vs_parent_a_* marked 不适用. '
                    'Local per-site displacement exists only for the ePCR clones; '
                    'design candidates are marked 未计算. '
                    'displacement_range_disjoint_from_baseline is a descriptive screen '
                    'using the criterion: the minimum cross-sequence displacement is '
                    'larger than the maximum same-clone self-baseline displacement at '
                    'that position. It is not a significance test; the 9 cross-sequence '
                    'model pairs and 3 self model pairs are statistically dependent, and '
                    'the baseline is the variant\'s own sample dispersion, not a common '
                    'WT-plus-variant upper bound. '
                    'domain_centroid_distance_wt_reference_A is the ePCR-batch WT median '
                    '(31.494 A, range 31.300-31.625 A over 3 models) for the clone rows.',
        },
        'derivation': {
            'WT': 'candidates.csv:WT, cross-checked against variants.csv:WT',
            'computational_design': '11 non-WT design rows; protein edits and sequence SHA '
                                    'recomputed and verified against source',
            'epcr_clone': 'NatV10/NatV89 from variants.csv; protein edits and sequence SHA '
                          'recomputed and verified against source; DNA edits retained separately',
            'coordinates': 'three diffusion samples (model_0/1/2) per candidate, converted from '
                           'tracked .cif.gz archives; converted PDB sequence verified against '
                           'the unified CSV; 14 candidates x 3 samples = 42 candidate coordinates',
            'reference_coordinates': 'ePCR-batch WT samples kept as extra comparison reference; '
                                     'same sequence as unified WT but a different prediction batch',
        },
        'evidence': evidence,
        'attachments': attachments,
        'coordinates': coordinates,
        'reference_coordinates': reference_coordinates,
        'files': files,
        'notes': [
            'A+B source_run/generation_candidate_id are inherited from parent A; '
            'not independent QVAE generation.',
            'B1/B2 are mechanism-hypothesis designs and have empty source_run/generation_candidate_id.',
            'Raw AB1 traces remain in raw wet-lab material; only tracked verification records are included.',
            'Structure figures and coordinates are predictions, not activity, stability, expression, '
            'or ranking evidence.',
            'The 12-design and ePCR model batches have different WT references/denominators and are '
            'not combined into one ranking.',
            'Three diffusion samples per candidate are not independent seeds or replicates.',
            'No experimentally validated activity or stability ranking exists.',
        ],
    }
    write_json(output / 'manifest.json', manifest)
    return {'status': 'COMPLETED', 'output': str(output),
            'manifest': str(output / 'manifest.json'), 'rows': len(rows)}
