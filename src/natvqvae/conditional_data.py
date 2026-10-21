"""正式人口、任务混合掩码及可校验的冻结特征缓存。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import torch

from .conditional_generation import AA, FrozenEncoder
from .intake import sha256_file

ROOT = Path(__file__).resolve().parents[2]
REGIONS = {'global': (4, 338), 'NMNAT': (4, 181), 'Nudix': (204, 338), 'linker': (182, 203)}
SIZES = {'global': [8, 16, 32], 'NMNAT': [8, 16, 32], 'Nudix': [8, 16, 32], 'linker': [4, 8, 16]}


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def stable_seed(*parts):
    return int(hashlib.sha256(':'.join(map(str, parts)).encode()).hexdigest()[:16], 16) % (2**63-1)


def status(run, phase, **values):
    item = {'phase': phase, **values}
    write_json(run/'status.json', item)
    print(json.dumps(item, ensure_ascii=False), flush=True)


def region_intervals(row):
    # CDD 坐标是 1-based inclusive，返回 Python 半开区间。
    return {'global': (int(row.nmnat_cdd_from)-1, int(row.nudix_cdd_to)),
            'NMNAT': (int(row.nmnat_cdd_from)-1, int(row.nmnat_cdd_to)),
            'Nudix': (int(row.nudix_cdd_from)-1, int(row.nudix_cdd_to)),
            'linker': (int(row.nmnat_cdd_to), int(row.nudix_cdd_from)-1)}


def mask_panels(row, seed):
    """27 个独立面板，权重严格对应 25% 广域 + 75% 任务条件。"""
    result = []
    for fraction in [.08, .15, .30]:
        name = f'random_{fraction:.2f}'
        rng = np.random.default_rng(stable_seed(seed, row.sequence_id, name))
        sites = sorted(map(int, rng.choice(int(row.length), max(1, round(row.length*fraction)), replace=False)))
        result.append({'panel': name, 'region': 'random', 'size_rule': 'fraction', 'positions': sites, 'weight': 1/12})
    for region, (low, high) in region_intervals(row).items():
        if low < 0 or high > row.length or high <= low:
            raise ValueError('Invalid CDD interval')
        wt_length = REGIONS[region][1]-REGIONS[region][0]+1
        for count in SIZES[region]:
            for rule in ['count', 'density']:
                name = f'{region}_{count}_{rule}'
                size = min(count, high-low) if rule == 'count' else min(high-low, max(1, round(count*(high-low)/wt_length)))
                rng = np.random.default_rng(stable_seed(seed, row.sequence_id, name))
                sites = sorted(map(int, rng.choice(np.arange(low, high), size, replace=False)))
                result.append({'panel': name, 'region': region, 'size_rule': rule, 'positions': sites, 'weight': 1/32})
    if len(result) != 27 or not np.isclose(sum(p['weight'] for p in result), 1):
        raise ValueError('Invalid mixture weights')
    return result


def verify_inputs(run, plan):
    manifest = json.loads((ROOT/'data/manifest.json').read_text())
    source = Path(os.environ.get('NATV_SOURCE_ROOT', manifest['source_root_default'])).resolve()
    entries = {e['name']: e for e in manifest['entries']}
    role_path = source/entries['family_roles']['path']
    if sha256_file(role_path) != entries['family_roles']['sha256']:
        raise ValueError('Source role identity differs')
    # 只读取训练和开发验证的身份字段，不读取测试目标序列。
    formal = pd.read_parquet(role_path, columns=['sequence_id', 'training_role_v2', 'component_v2'],
                             filters=[('training_role_v2', 'in', ['training', 'validation'])])
    rows = pd.read_csv(ROOT/'data/processed/samples.csv').sort_values('sequence_id').reset_index(drop=True)
    if set(rows.training_role_v2) != {'training', 'validation'} or rows.sequence_id.duplicated().any():
        raise ValueError('Unexpected population')
    for role, count, fasta_name in [('training', 1644, 'training.fasta'), ('validation', 227, 'validation.fasta')]:
        part = rows[rows.training_role_v2 == role]
        if len(part) != count or set(part.sequence_id) != set(formal[formal.training_role_v2 == role].sequence_id):
            raise ValueError('Training/validation role mismatch')
        from Bio import SeqIO
        fasta = {r.id: str(r.seq) for r in SeqIO.parse(ROOT/'data/processed'/fasta_name, 'fasta')}
        if fasta != part.set_index('sequence_id').sequence.to_dict():
            raise ValueError('FASTA identity differs')
    if not rows.component_v2.eq(rows.sequence_id.map(formal.set_index('sequence_id').component_v2)).all():
        raise ValueError('Component identity differs')
    if set(rows[rows.training_role_v2=='training'].component_v2) & set(rows[rows.training_role_v2=='validation'].component_v2):
        raise ValueError('Component leakage')
    for row in rows.itertuples():
        if hashlib.sha256(row.sequence.encode()).hexdigest() != row.sequence_id or len(row.sequence)!=row.length or not set(row.sequence)<=set(AA):
            raise ValueError('Sequence identity differs')
        region_intervals(row)
    backbone = source/'nad_models/esm2'
    expected = plan['backbone_sha256']
    hashes = {}
    for name, value in expected.items():
        path = backbone/name; actual = sha256_file(path)
        if actual != value:
            raise ValueError('Frozen ESM identity differs: '+name)
        hashes[str(path.resolve())] = actual
    paths = [ROOT/'data/manifest.json', role_path] + [ROOT/'data/processed'/name for name in ['training.fasta', 'validation.fasta', 'samples.csv']]
    hashes.update({str(p.resolve()): sha256_file(p) for p in paths})
    raw = ROOT/'data/results'/run.name/'raw'; raw.mkdir(parents=True, exist_ok=True)
    rows.to_csv(raw/'population.csv', index=False)
    write_json(run/'inputs/data_identity.json', {'source_root': str(source), 'sha256': hashes,
        'population': {'training': 1644, 'validation': 227}, 'components': {'training': 47, 'validation': 44},
        'test_targets_loaded': False, 'legacy_NatV_weights_loaded': False})
    return rows, backbone


def prepare_features(run, plan):
    status(run, 'VERIFYING_INPUTS')
    rows, backbone = verify_inputs(run, plan)
    raw = ROOT/'data/results'/run.name/'raw'
    cache = ROOT/'data/interim'/run.name; cache.mkdir(parents=True, exist_ok=True)
    queries = []; offset = 0
    for i, row in enumerate(rows.itertuples()):
        panels = [{'panel': 'full', 'positions': [], 'weight': 0, 'region': 'full', 'size_rule': 'full'}] + mask_panels(row, plan['feature_seed'])
        for panel in panels:
            queries.append({'row': i, 'offset': offset, 'length': int(row.length), **panel})
            offset += int(row.length)
    spec_path = raw/'masks.json'; write_json(spec_path, queries)
    cache_path = cache/'features.npy'; progress_path = cache/'progress.json'
    completed = json.loads(progress_path.read_text())['completed'] if progress_path.exists() else 0
    if cache_path.exists():
        mmap = np.lib.format.open_memmap(cache_path, mode='r+')
        if mmap.shape != (offset, 1280):
            raise ValueError('Feature cache shape differs')
    else:
        if completed:
            raise ValueError('Feature cache missing')
        mmap = np.lib.format.open_memmap(cache_path, mode='w+', dtype=np.uint16, shape=(offset, 1280))
    receipt_path = raw/'feature_manifest.json'
    if completed == len(queries) and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if sha256_file(cache_path) != receipt['sha256'] or sha256_file(spec_path) != receipt['masks_sha256']:
            raise ValueError('Completed cache changed')
        return rows, queries, cache_path
    source_paths = [Path(__file__), ROOT/'src/natvqvae/conditional_generation.py', run/'plan.json']
    write_json(run/'inputs/feature_identity.json', {'git_revision': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        'sha256': {str(p.relative_to(ROOT)): sha256_file(p) for p in source_paths},
        'feature_dtype': 'bfloat16_stored_as_uint16', 'inherited_NatV_features': False})
    encoder = FrozenEncoder(backbone); started = time.monotonic()
    batch_size = plan['feature_batch_size']
    for start in range(completed, len(queries), batch_size):
        batch = queries[start:start+batch_size]
        h, _, _ = encoder([rows.iloc[q['row']].sequence for q in batch], [q['positions'] for q in batch])
        values = h.to(torch.bfloat16).cpu().view(torch.uint16).numpy()
        for i, q in enumerate(batch):
            mmap[q['offset']:q['offset']+q['length']] = values[i, :q['length']]
        completed = start+len(batch)
        if completed % (batch_size*16) == 0 or completed == len(queries):
            mmap.flush(); write_json(progress_path, {'completed': completed, 'total': len(queries), 'masks_sha256': sha256_file(spec_path)})
            status(run, 'PREPARING_FROZEN_FEATURES', completed=completed, total=len(queries), seconds=time.monotonic()-started)
    mmap.flush(); del encoder; torch.cuda.empty_cache()
    write_json(receipt_path, {'path': str(cache_path.relative_to(ROOT)), 'sha256': sha256_file(cache_path),
        'masks_sha256': sha256_file(spec_path), 'queries': len(queries), 'shape': [offset,1280], 'storage_dtype': 'uint16_bfloat16_bits',
        'preparation_seconds': time.monotonic()-started})
    return rows, queries, cache_path


class FeatureData:
    def __init__(self, run):
        raw = ROOT/'data/results'/run.name/'raw'
        self.rows = pd.read_csv(raw/'population.csv')
        self.queries = json.loads((raw/'masks.json').read_text())
        receipt = json.loads((raw/'feature_manifest.json').read_text())
        self.features = np.load(ROOT/receipt['path'], mmap_mode='r')
        self.full = {}; self.by_row = {}
        for q in self.queries:
            if q['panel']=='full': self.full[q['row']] = q
            else: self.by_row.setdefault(q['row'], []).append(q)
        counts = self.rows[self.rows.training_role_v2=='training'].component_v2.value_counts()
        norm = len(self.rows[self.rows.training_role_v2=='training'])/len(counts)
        self.weights = {i: norm/counts[row.component_v2] for i,row in self.rows.iterrows() if row.training_role_v2=='training'}
        self.validation = [q for q in self.queries if q['panel']!='full' and self.rows.iloc[q['row']].training_role_v2=='validation']
        self.validation.sort(key=lambda q:(q['panel'], q['length'], q['row']))

    def array(self, q):
        bits = np.array(self.features[q['offset']:q['offset']+q['length']], copy=True)
        return torch.from_numpy(bits).view(torch.bfloat16)

    def batch(self, queries, device='cuda', need_full=True):
        width = max(q['length'] for q in queries); n = len(queries)
        context = torch.zeros(n,width,1280,dtype=torch.bfloat16)
        full = torch.zeros_like(context) if need_full else None
        valid = torch.zeros(n,width,dtype=torch.bool); mask = torch.zeros_like(valid)
        labels = torch.zeros(n,width,dtype=torch.long)
        for i,q in enumerate(queries):
            size=q['length']; context[i,:size] = self.array(q)
            if need_full: full[i,:size] = self.array(self.full[q['row']])
            valid[i,:size]=True; mask[i,q['positions']]=True
            labels[i,:size]=torch.tensor([AA.index(a) for a in self.rows.iloc[q['row']].sequence])
        return context.to(device), full.to(device) if full is not None else None, valid.to(device), mask.to(device), labels.to(device)

    def training_queries(self, epoch, seed):
        rng = np.random.default_rng(stable_seed(seed, 'training_order', epoch)); queries=[]
        for i in self.weights:
            choices=self.by_row[i]; index=int(rng.choice(len(choices), p=[q['weight'] for q in choices]))
            queries.append(choices[index])
        return [queries[i] for i in rng.permutation(len(queries))]
