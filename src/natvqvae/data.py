"""Target, raw population, exact splits, masks, and on-the-fly frozen features."""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from .model import AA

REGIONS = {'global': (4, 338), 'NMNAT': (4, 181), 'Nudix': (204, 338), 'linker': (182, 203)}
SIZES = {'global': [8, 16, 32], 'NMNAT': [8, 16, 32], 'Nudix': [8, 16, 32], 'linker': [4, 8, 16]}


def load_yaml(path):
    with open(path, encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def stable_seed(*parts):
    return int(hashlib.sha256(':'.join(map(str, parts)).encode()).hexdigest()[:16], 16) % (2**63 - 1)


def load_fasta(path):
    records = {}
    name = None
    chunks = []
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if name is not None:
                    records[name] = ''.join(chunks).upper()
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if name is not None:
        records[name] = ''.join(chunks).upper()
    return records


@dataclass(frozen=True)
class Target:
    name: str
    sequence: str
    domains: dict
    seed_domains: dict
    anchors: dict

    @classmethod
    def load(cls, path):
        value = load_yaml(path)
        sequence = str(value['sequence']).strip().upper()
        if len(sequence) != int(value['length']) or not set(sequence) <= set(AA):
            raise ValueError('Invalid target sequence')
        return cls(value['name'], sequence, value['domains'], value['seed_domains'], value['anchors'])


@dataclass(frozen=True)
class Record:
    sequence_id: str
    sequence: str
    component: str
    length: int
    nmnat_from: int
    nmnat_to: int
    nudix_from: int
    nudix_to: int

    def intervals(self):
        return {
            'global': (self.nmnat_from - 1, self.nudix_to),
            'NMNAT': (self.nmnat_from - 1, self.nmnat_to),
            'Nudix': (self.nudix_from - 1, self.nudix_to),
            'linker': (self.nmnat_to, self.nudix_from - 1),
        }


def _read_rows(path, require_cdd=True):
    rows = []
    with open(path, newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            if require_cdd:
                rows.append({
                    'sequence_id': row['sequence_id'],
                    'component': row['component_v2'],
                    'length': int(row['length']),
                    'nmnat_from': int(row['nmnat_cdd_from']),
                    'nmnat_to': int(row['nmnat_cdd_to']),
                    'nudix_from': int(row['nudix_cdd_from']),
                    'nudix_to': int(row['nudix_cdd_to']),
                })
            else:
                rows.append({'sequence_id': row['sequence_id'], 'component': row['component_v2']})
    return rows


def mask_panels(record, seed):
    """27 independent panels, 25% broad + 75% task-conditioned."""
    result = []
    for fraction in [0.08, 0.15, 0.30]:
        name = f'random_{fraction:.2f}'
        rng = np.random.default_rng(stable_seed(seed, record.sequence_id, name))
        sites = sorted(map(int, rng.choice(record.length, max(1, round(record.length * fraction)), replace=False)))
        result.append({'panel': name, 'region': 'random', 'size_rule': 'fraction',
                       'positions': sites, 'weight': 1 / 12})
    for region, (low, high) in record.intervals().items():
        if low < 0 or high > record.length or high <= low:
            raise ValueError('Invalid CDD interval')
        wt_length = REGIONS[region][1] - REGIONS[region][0] + 1
        for count in SIZES[region]:
            for rule in ['count', 'density']:
                name = f'{region}_{count}_{rule}'
                size = min(count, high - low) if rule == 'count' else min(
                    high - low, max(1, round(count * (high - low) / wt_length)))
                rng = np.random.default_rng(stable_seed(seed, record.sequence_id, name))
                sites = sorted(map(int, rng.choice(np.arange(low, high), size, replace=False)))
                result.append({'panel': name, 'region': region, 'size_rule': rule,
                               'positions': sites, 'weight': 1 / 32})
    if len(result) != 27 or not np.isclose(sum(panel['weight'] for panel in result), 1):
        raise ValueError('Invalid mixture weights')
    return result


def generation_masks(generation, target_sequence):
    groups = []
    for region, (low, high) in generation['regions_1based'].items():
        for size in generation['mask_sizes'][region]:
            for repeat in range(int(generation['masks_per_condition'])):
                gid = f'{region}_m{size:02d}_r{repeat:02d}'
                rng = np.random.default_rng(stable_seed(generation['seed'], gid, 'positions'))
                positions = sorted(map(int, rng.choice(np.arange(low - 1, high), size, replace=False)))
                groups.append({'id': gid, 'region': region, 'mask_size': size, 'positions': positions})
    if any(len(group['positions']) != group['mask_size'] for group in groups):
        raise ValueError('Generation mask size differs')
    return groups


class PopulationData:
    def __init__(self, sources_path, study_path, target_path):
        sources = load_yaml(sources_path)
        self.sources = sources
        self.study = load_yaml(study_path)
        self.target = Target.load(target_path)
        self.population = load_fasta(sources['population'])
        self.test_rows = _read_rows(sources['splits']['test'], require_cdd=False)
        self.train_rows = _read_rows(sources['splits']['training'])
        self.validation_rows = _read_rows(sources['splits']['validation'])
        if len(self.train_rows) != 1644 or len(self.validation_rows) != 227:
            raise ValueError('Unexpected split sizes')
        if set(row['sequence_id'] for row in self.train_rows) & set(
                row['sequence_id'] for row in self.validation_rows):
            raise ValueError('Training/validation overlap')
        if set(row['component'] for row in self.train_rows) & set(
                row['component'] for row in self.validation_rows):
            raise ValueError('Training/validation component leakage')
        self.train = self._records(self.train_rows)
        self.validation = self._records(self.validation_rows)
        self.by_id = {record.sequence_id: record for record in self.train + self.validation}
        components = {}
        for row in self.train_rows:
            components[row['component']] = components.get(row['component'], 0) + 1
        norm = len(self.train_rows) / len(components)
        self.weights = {record.sequence_id: norm / components[record.component]
                        for record in self.train}

    def _records(self, rows):
        output = []
        for row in rows:
            sequence = self.population.get(row['sequence_id'])
            if sequence is None:
                raise ValueError(f"Missing population sequence: {row['sequence_id']}")
            if len(sequence) != row['length']:
                raise ValueError(f"Length differs: {row['sequence_id']}")
            output.append(Record(row['sequence_id'], sequence, row['component'], row['length'],
                                 row['nmnat_from'], row['nmnat_to'], row['nudix_from'], row['nudix_to']))
        return output

    def training_queries(self, epoch, seed):
        rng = np.random.default_rng(stable_seed(seed, 'training_order', epoch))
        queries = []
        panels = {record.sequence_id: mask_panels(record, self.study['training']['mask_seed'])
                  for record in self.train}
        for record in self.train:
            choices = panels[record.sequence_id]
            index = int(rng.choice(len(choices), p=[panel['weight'] for panel in choices]))
            queries.append({'record': record, 'panel': choices[index]})
        return [queries[i] for i in rng.permutation(len(queries))]

    def validation_queries(self):
        queries = []
        for record in self.validation:
            for panel in mask_panels(record, self.study['training']['mask_seed']):
                queries.append({'record': record, 'panel': panel})
        return queries

    def batch(self, queries, encoder, device, need_full):
        sequences = [q['record'].sequence for q in queries]
        context, valid, mask = encoder(sequences, [q['panel']['positions'] for q in queries])
        full = None
        if need_full:
            full, _, _ = encoder(sequences, [[] for _ in queries])
        width = context.shape[1]
        labels = torch.zeros(len(queries), width, dtype=torch.long, device=device)
        for index, query in enumerate(queries):
            labels[index, :len(query['record'].sequence)] = torch.tensor(
                [AA.index(residue) for residue in query['record'].sequence], device=device)
        return context.to(device), None if full is None else full.to(device), valid.to(device), mask.to(device), labels

    def encode_wt(self, encoder, positions, device):
        h, valid, mask = encoder([self.target.sequence], [positions])
        return h.to(device), valid.to(device), mask.to(device)
