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


def _readme(rows, attachments, evidence, coordinates, sources, commit):
    by_id = {row['candidate_id']: row for row in rows}
    lines = [
        '# NatV 候选交接包（14 条唯一蛋白）',
        '',
        '本包把计算设计候选和湿实验 ePCR 克隆合并为一份可独立交接的快照：共同 WT 1 条、'
        '计算设计 11 条、ePCR 克隆 2 条，合计 14 条不同的 341 aa 蛋白序列。'
        '完整序列、编辑、亲本、来源和 SHA 见 `candidates.csv` / `candidates.fasta`；'
        '本页只列身份和编辑，不复制长序列。',
        '',
        '## 14 条对象',
        '',
        '| ID | 来源类别 | 完整蛋白编辑 | 比较亲本 | 预测批次 |',
        '|---|---|---|---|---|',
    ]
    for row in rows:
        parents = ' + '.join(value for value in (row['parent_a'], row['parent_b']) if value)
        if row['source_category'] == 'shared_reference':
            parents = '两组共同参照'
        elif row['source_category'] == 'epcr_clone':
            parents = '独立克隆；WT 仅作参照'
        batch = next((item['batch'] for item in coordinates
                      if item['candidate'] == row['candidate_id']), '')
        lines.append(f"| {row['candidate_id']} | {row['source_category']} | "
                     f"{row['full_protein_edits']} | {parents or '—'} | {batch} |")
    lines += [
        '',
        'WT 只保留一行，作为设计和 ePCR 两组比较的共同参照。A1–A3 来自 QVAE 生成，'
        'B1/B2 为机制假说设计（不由 QVAE 生成，`source_run` 为空），A+B 为组合；'
        'A+B 的 `source_run`/`generation_candidate_id` 继承对应 A，不代表 A+B 自身由 QVAE 直接生成。'
        'NatV10/NatV89 是独立 ePCR 克隆，不是 B1/B2。',
        '',
        '## 可查看的预测结构（coordinates/ 与 references/）',
        '',
        '每条候选 3 个固定 diffusion samples 的全长 apo 预测结构，'
        '`coordinates/<ID>_model_{0,1,2}.pdb`，共 14 × 3 = 42 个 PDB，'
        '由已跟踪的 `.cif.gz` 归档转换而来，未重新预测。'
        '`references/epcr_WT_model_{0,1,2}.pdb` 另存 ePCR 批次的 WT 3 个样本：'
        '序列与统一 WT 相同，但属于不同预测批次，仅作额外比较参照，'
        '不计入 42 个候选坐标，也不替代设计批次的 WT。全包坐标合计 45 个。',
        '',
        '- 12 条设计（含 WT）来自设计批次；NatV10/NatV89 来自 ePCR 批次；两批的 WT 预测坐标不同，'
        '不要互相替代，也不要拼接成同一构象。',
        '- 这些是**代表性预测单体**：无配体、不是实验结构、不是候选平均构象，也不能当作活性证据。',
        '- 3 个 diffusion samples 不是独立种子或独立重复；包内图片固定展示 model_0，'
        '但 3 个样本都可在 `coordinates/` 中查看，并支持按批次复算原有 RMSD。',
        '- 查看示例：`pymol coordinates/*.pdb`；叠合比较可用 viewer 的 CA 叠合命令，'
        '本包不为此重跑渲染或生成新模型。',
        '',
        '## 先看什么',
        '',
        '1. `candidates.csv` / `candidates.fasta`：14 条序列、编辑、亲本关系和 SHA。',
        '2. `coordinates/`：42 个可打开的全长预测 PDB（14 候选 × 3 samples），'
        '另 `references/` 有 3 个 ePCR 批次 WT 参照。',
        '3. `attachments/`：必要的结构图、位点表和 RMSD 表。',
        '4. `evidence/`：原始设计表、Sanger 核查记录和生成来源快照。',
        '5. `manifest.json`：逐文件 SHA-256、来源路径和派生关系。',
        '',
        '## 三个问题',
        '',
        '- A 类组合是否改变合成相关表现？不能把完整 NatV 的净 NAD⁺ 增加直接称为独立 vNMNAT 活性。',
        '- B1/B2 是否改变 NAD⁺ 水解？结构距离或模型置信度不能替代水解释放读数。',
        '- A 与 B 是否兼容并改善目标表现？应在选定条件下比较 WT/A/B/A+B 的净积累，并结合可获得的合成与水解信息。',
        '',
        '**目前没有任何经验证的活性、稳定性或 NAD⁺ 净积累排序**；本包只提供序列身份、'
        '亲本关系和预测结构证据。具体方法、对照和待湿实验确认事项见仓库中的 '
        f'[candidate_handoff.md]({_github_url(commit, "data/results/evaluation/natv_dual_domain_design/demo/candidate_handoff.md")})。',
        '',
        '## 附件',
        '',
        '| 文件 | 用途 |',
        '|---|---|',
    ]
    for item in attachments:
        lines.append(f"| [`{Path(item['package']).name}`]({item['package']}) "
                     f"| {item.get('description', '结构附件')} |")
    lines += [
        '',
        '说明：12 条设计与 NatV10/NatV89 的模型批次、WT 参照和分母不同，不能把两组 RMSD 拼成同一条件排名。',
        '',
        '## 证据来源与边界',
        '',
        '- `evidence/design_candidates_original.csv`：统一表合并前的原始 12 设计表；'
        '`evidence/clone_variants.csv`、`evidence/clone_sanger_verification.json`、'
        '`evidence/clone_build.json`：ePCR 克隆的序列、Sanger 核查、序列重建与运行记录。'
        '原始 `.ab1` 峰图仍在原始湿实验材料中，不包含在本包内。',
        '- `evidence/design.json`、`evidence/design.yaml`、'
        '`evidence/random_seed8301_generation.json`、`evidence/random_seed8303_generation.json`：'
        '生成/设计来源快照，用于追溯候选 ID；其中机制措辞是历史设计记录/假说，不表示已验证增强。',
        '- 结构图与坐标来自 Boltz-2 预测，不是实验结构；pLDDT/pTM、Cα RMSD 和局部差异'
        '不能证明活性、稳定性或表达。',
        '- 完整 NatV 的 NAD⁺ 净积累不是独立 vNMNAT 活性；A570 的历史终点吸光信号不能用于排序或机制归因。',
        '- 本包不含 DNA/构建、表达纯化、检测 SOP、对照/重复/判定标准或功能结果；这些需湿实验团队确认。',
        '',
        '## 详细报告（仓库追溯）',
        '',
        f'- [设计报告]({_github_url(commit, "data/results/evaluation/natv_dual_domain_design/REPORT.md")})',
        f'- [ePCR 全长结构比较]({_github_url(commit, "data/results/evaluation/kvp40_epcr_full_length_structures/REPORT.md")})',
        f'- [ePCR NMNAT 环境比较]({_github_url(commit, "data/results/evaluation/kvp40_epcr_nmnat_environment/REPORT.md")})',
        f'- [RESEARCH_PLAN.md]({_github_url(commit, "docs/RESEARCH_PLAN.md")})',
        '',
        '交付时记录 Git commit；序列、身份文件和报告以该 commit 为准。',
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
    readme_path.write_text(_readme(rows, attachments, evidence, coordinates, sources, commit))

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
