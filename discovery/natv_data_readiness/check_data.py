"""核对本地既有数据并导出任务无关的训练/验证输入；不拟合模型。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from natvqvae.intake import load_manifest, source_root, verify_all, sha256_file


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_fasta(path):
    result = {}
    key = None
    for line in path.read_text().splitlines():
        if line.startswith('>'):
            key = line[1:].split()[0]
            require(key not in result, f'FASTA duplicate ID: {key}')
            result[key] = ''
        elif line.strip():
            require(key is not None, 'FASTA sequence before header')
            result[key] += line.strip()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--out', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest = load_manifest()
    source = (args.root or source_root(manifest)).resolve()
    out = args.out.resolve()
    require(not out.is_relative_to(source), 'Output must be outside read-only source')
    results = verify_all(manifest, source)
    require(all(r.ok for r in results), str([r.detail for r in results if not r.ok]))
    paths = {r.name: r.path for r in results}
    load = lambda name: json.loads(paths[name].read_text())
    roles = pd.read_parquet(paths['family_roles'])
    core = load('core_split')
    require(sha256_file(paths['family_roles']) == core['core_manifest_sha256'], 'Role identity')
    require(len(roles) == 2430 and roles.sequence_id.is_unique, 'Population identity')
    require(Counter(roles.training_role_v2) == Counter(training=1644, validation=227, test=231,
            annotation_conflict_quarantine=249, quarantined_old_sealed_component=75,
            excluded_noncanonical_or_domain=4), 'Role counts')
    population = read_fasta(paths['family_population'])
    require(set(population) == set(roles.sequence_id), 'FASTA membership')
    for row in roles.itertuples():
        require(population[row.sequence_id] == row.sequence, 'FASTA / role sequence differs')
        require(hashlib.sha256(row.sequence.encode()).hexdigest() == row.sequence_id, 'Sequence hash')
        require(len(row.sequence) == row.length, 'Length metadata')
    selected = roles[roles.training_role_v2.isin(['training', 'validation'])].copy()
    require(not selected.source_annotation_conflict.any() and selected.family_eligible_v2.all(),
            'Conflict or ineligible training/validation sample')
    require(set(roles.loc[roles.core_training_eligible_v2 & (roles.training_role_v2 == 'training'), 'sequence_id']) ==
            set(selected.loc[selected.training_role_v2 == 'training', 'sequence_id']), 'Training eligibility')
    split_ids = {}
    split_components = {}
    for role in ['training', 'validation', 'test', 'quarantined_old_sealed_component']:
        frame = roles[roles.training_role_v2 == role]
        split_ids[role] = set(frame.sequence_id)
        split_components[role] = set(frame.component_v2)
        require(set(core['splits'][role]) == split_ids[role], f'Core split membership: {role}')
    for i, a in enumerate(split_components):
        for b in list(split_components)[i+1:]:
            require(not split_components[a] & split_components[b], f'Component leakage {a}/{b}')
    require(len(split_components['training']) == 47, 'Training component count')
    allowed = pd.read_parquet(paths['allowed_samples'])
    historical = pd.read_csv(paths['training_samples'])
    for table, expected_ids in [(allowed, set(selected.sequence_id)), (historical, split_ids['training'])]:
        require(table.sequence_id.is_unique and set(table.sequence_id) == expected_ids,
                'Historical allowed sample membership')
        for row in table.itertuples():
            ref = roles.loc[roles.sequence_id == row.sequence_id].iloc[0]
            require(row.sequence == ref.sequence and row.training_role_v2 == ref.training_role_v2
                    and row.component_v2 == ref.component_v2, 'Historical sample drift')
    train_fasta = read_fasta(paths['train_sequences'])
    require(set(train_fasta) == split_ids['training'], 'Retrieval FASTA includes other roles')
    require(all(population[k] == v for k, v in train_fasta.items()), 'Retrieval sequence drift')
    plan = load('masked_input_plan')
    require(sha256_file(paths['train_sequences']) == plan['train_fasta_sha256'], 'Plan FASTA hash')
    require(sha256_file(paths['allowed_samples']) == plan['allowed_samples_sha256'], 'Plan allowed hash')
    require(sha256_file(paths['training_samples']) == plan['training_samples_sha256'], 'Plan samples hash')
    require(len(plan['retrieval_database_ids']) == 1644 and
            set(plan['retrieval_database_ids']) == split_ids['training'], 'Plan retrieval boundary')
    queries = {q['query_id']: q for q in plan['queries']}
    require(len(queries) == len(plan['queries']) == 13097, 'Duplicate/missing query')
    for q in queries.values():
        require(q['role'] in ['training', 'validation'] and
                q['sequence_id'] in split_ids[q['role']], 'Query role leakage')
        sequence = population[q['sequence_id']]
        require(q['current_sequence'] == sequence and q['length'] == len(sequence), 'Query sequence')
        positions = q['positions']
        require(len(set(positions)) == len(positions) and
                all(0 <= i < len(sequence) for i in positions), 'Mask positions')
        require(q['masked_sequence'] == ''.join('X' if i in positions else aa
                for i, aa in enumerate(sequence)), 'Mask does not match query')
    query_counts = Counter(q['sequence_id'] for q in queries.values())
    require(set(query_counts) == set(selected.sequence_id) and set(query_counts.values()) == {7},
            'Expected seven panels per allowed sequence')
    feature_stats = {}
    evolution_paths = {e['query_id']: source / e['path'] for e in load('evolution_features')}
    for name in ['evolution_features', 'masked_contexts']:
        entries = load(name)
        require(len(entries) == len(queries) and {e['query_id'] for e in entries} == set(queries),
                f'{name}: query membership')
        total_bytes = 0
        for i, entry in enumerate(entries):
            file = (source / entry['path']).resolve()
            require(file.is_relative_to(source), 'Feature outside source root')
            require(sha256_file(file) == entry['sha256'], f'Feature missing or hash differs: {file}')
            total_bytes += file.stat().st_size
            q = queries[entry['query_id']]
            length = q['length']
            if name == 'evolution_features':
                with np.load(file, allow_pickle=False) as bundle:
                    value = bundle['evolution']
                    require(value.shape == (length, 21) and np.isfinite(value).all(), 'Evolution array')
            else:
                payload = torch.load(file, map_location='cpu', weights_only=True)
                require(all(payload[k] == q[k] for k in
                            ['query_id', 'sequence_id', 'role', 'length', 'panel']), 'Context identity')
                enc = payload['encoded']
                require(enc['masked_before_encoding'] and enc['all_encoder_parameters_frozen'],
                        'Encoder masking/frozen flag')
                require(enc['representations'].shape == (length, 1280) and
                        enc['logits'].shape == (length, 33) and
                        payload['evolution'].shape == (length, 21) and payload['labels'].shape == (length,),
                        'Context dimensions')
                require(all(torch.isfinite(value).all().item() for value in
                            [enc['representations'], enc['logits'], payload['evolution']]), 'Non-finite context')
                require(set(torch.where(enc['target_mask'])[0].tolist()) == set(q['positions']),
                        'Context target mask')
                offset = int(enc['prepend_bos'])
                require(set(torch.where(enc['tokens'] == enc['mask_token_id'])[0].tolist()) ==
                        {position + offset for position in q['positions']}, 'Actual encoded mask tokens')
                with np.load(evolution_paths[entry['query_id']], allow_pickle=False) as evolution:
                    require(np.array_equal(payload['evolution'].numpy(), evolution['evolution']),
                            'Context / evolution feature mismatch')
            if (i+1) % 2000 == 0:
                print(f'{name}: {i+1}/{len(entries)} verified', flush=True)
        feature_stats[name] = {'files': len(entries), 'bytes': total_bytes, 'status': 'PASS'}
    # Verify existing structural annotation sources without treating them as training features.
    basis_path = ROOT / 'discovery/natv_pairing_audit/structural_interpretation_basis.json'
    basis = json.loads(basis_path.read_text())
    for relative, expected in basis['sources'].items():
        require(sha256_file(basis_path.parent / relative) == expected, f'Structural source: {relative}')
    out.mkdir(parents=True, exist_ok=True)
    artifacts = []
    def record(path):
        artifacts.append({'path': str(path.relative_to(out)), 'sha256': sha256_file(path),
                          'bytes': path.stat().st_size})
    for role in ['training', 'validation']:
        frame = selected[selected.training_role_v2 == role].sort_values('sequence_id')
        path = out / f'{role}.fasta'
        path.write_text(''.join(f'>{r.sequence_id}\n{r.sequence}\n' for r in frame.itertuples()))
        record(path)
    fields = ['sequence_id', 'training_role_v2', 'component_v2', 'split_v2', 'length',
              'core_training_eligible_v2', 'source_annotation_conflict']
    path = out / 'roles.csv'
    roles[fields].sort_values('sequence_id').to_csv(path, index=False)
    record(path)
    fields += ['sequence', 'nmnat_cdd_from', 'nmnat_cdd_to', 'nudix_cdd_from', 'nudix_cdd_to']
    path = out / 'samples.csv'
    selected[fields].sort_values('sequence_id').to_csv(path, index=False)
    record(path)
    summary = {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'script_sha256': sha256_file(Path(__file__)),
        'input_manifest_sha256': sha256_file(ROOT / 'inputs/manifest.json'),
        'source_root': str(source),
        'inputs': [{'name': r.name, 'path': str(r.path), 'sha256': sha256_file(r.path)} for r in results],
        'roles': dict(Counter(roles.training_role_v2)),
        'components': {k: len(v) for k, v in split_components.items()},
        'training_lengths': [int(selected[selected.training_role_v2 == 'training'].length.min()),
                             int(selected[selected.training_role_v2 == 'training'].length.max())],
        'query_roles': dict(Counter(q['role'] for q in queries.values())),
        'feature_checks': feature_stats,
        'retrieval_fasta': 'training.fasta',
        'component_overlap': 0,
        'structure_basis_sha256': sha256_file(basis_path),
        'artifacts': artifacts,
        'readiness': {'sequence_inputs': 'PASS', 'historical_masked_features': 'PASS',
                      'new_generation_task_features': 'UNKNOWN', 'experimental_activity_labels': 'UNKNOWN'},
        'environment': {'python': sys.version.split()[0], 'pandas': pd.__version__,
                        'numpy': np.__version__, 'torch': torch.__version__},
    }
    (out / 'data_checks.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary['readiness'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
