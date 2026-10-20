"""核对旧模型身份、权重可加载性及数据角色重叠，不执行模型 forward。"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

# Importing the read-only source must not create __pycache__ there.
sys.dont_write_bytecode = True
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from natvqvae.intake import load_manifest, source_root, verify_entry


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--out', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    manifest = load_manifest()
    source = (args.root or source_root(manifest)).resolve()
    out = args.out.resolve()
    if out.is_relative_to(source):
        raise ValueError('Output lies in read-only source')
    files = {}
    def bind(relative, expected=None):
        path = source / relative
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError(f'Source hash mismatch: {relative}')
        files[relative] = {'sha256': actual, 'bytes': path.stat().st_size}
        return path
    def read(relative):
        return json.loads(bind(relative).read_text())
    seq_run = 'experiments/run_reports/natv_sequence_design_dev_v1/'
    seq_data = 'data/processed/releases/natv_sequence_design_dev_v1/'
    h2_run = 'experiments/run_reports/natv_h2_architecture_generation_v1_attempt1/'
    seq_freeze = read(seq_run + 'training/checkpoint_freeze.json')
    for relative, expected in seq_freeze['code_sha256'].items():
        bind(relative, expected)
    bind('config/natv/sequence_design_dev_v1.yaml', seq_freeze['config_sha256'])
    condition_prov = read(seq_run + 'condition/provenance.json')
    bind(seq_run + 'condition/wt_condition.pt', condition_prov['cache_sha256'])
    frozen_identity = read(seq_run + 'training/frozen_identity.json')
    old_rows = [json.loads(line) for line in bind(seq_data + 'sequence_manifest.jsonl',
                    frozen_identity['corpus_sha256']).read_text().splitlines()]
    bind(seq_data + 'split_manifest.json', frozen_identity['split_sha256'])
    h2_freeze = read(h2_run + 'formal_source_freeze.json')
    for relative, expected in h2_freeze['source_hashes'].items():
        bind(relative, expected)
    bind('config/natv/h2_architecture_generation_comparison_v1.json', h2_freeze['contract_sha256'])
    bind('src/nad_qvae/core/model.py')
    sys.path.insert(0, str(source / 'src'))
    from nad_qvae.design.sequence_generator import SequenceGenerator
    from nad_qvae.design.h2_model_v1 import H2RecoveryModel
    torch.set_num_threads(1)
    receipts = []
    for run in seq_freeze['runs']:
        relative = seq_run + f"training/{run['arm']}_seed{run['seed']}/model_best.pt"
        saved = torch.load(bind(relative, run['checkpoint_sha256']), map_location='cpu', weights_only=True)
        model = SequenceGenerator(saved['model_config'], saved['state_dict']['wt_embeddings'], saved['arm'])
        model.load_state_dict(saved['state_dict'], strict=True)
        if not all(torch.isfinite(p).all().item() for p in model.parameters()):
            raise ValueError('Nonfinite checkpoint')
        receipts.append({'model': run['arm'], 'seed': run['seed'], 'path': relative,
                         'strict_load': 'PASS', 'output_positions': model.length})
    h2_checkpoints = read(h2_run + 'generation/checkpoint_freeze.json')
    for key, entry in h2_checkpoints.items():
        saved = torch.load(bind(entry['path'], entry['sha256']), map_location='cpu', weights_only=True)
        model = H2RecoveryModel(aa_token_ids=saved['aa_token_ids'], architecture=saved['architecture'],
                                hidden_dim=128, latent_dim=32, latent_states=8, modality='1')
        model.load_state_dict(saved['state_dict'], strict=True)
        if not all(torch.isfinite(p).all().item() for p in model.parameters()):
            raise ValueError('Nonfinite H2 checkpoint')
        receipts.append({'model': 'H2/' + saved['architecture'], 'seed': saved['seed'],
                         'path': entry['path'], 'strict_load': 'PASS'})
    roles_entry = next(e for e in manifest['entries'] if e['name'] == 'family_roles')
    verified = verify_entry(roles_entry, source)
    if not verified.ok:
        raise ValueError(verified.detail)
    bind(roles_entry['path'], roles_entry['sha256'])
    current = pd.read_parquet(verified.path, columns=['sequence_id', 'training_role_v2'])
    old = pd.DataFrame([{'sequence_id': r['sequence_id'], 'old_split': r['split']} for r in old_rows])
    joined = current.merge(old, validate='one_to_one')
    if len(joined) != len(current) or len(old) != len(current):
        raise ValueError('Historical population identity differs')
    table = pd.crosstab(joined.training_role_v2, joined.old_split)
    for relative in [seq_run + 'REPORT.md', seq_run + 'scientific_interpretation.json',
                     seq_run + 'evaluation/health.json', h2_run + 'PHASE_SUMMARY.md']:
        bind(relative)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / 'split_overlap.csv')
    output = {
        'source_root': str(source),
        'source_git_revision': subprocess.check_output(['git','rev-parse','HEAD'], cwd=source, text=True).strip(),
        'script_sha256': sha(Path(__file__)),
        'inputs': files,
        'checkpoint_loads': receipts,
        'split_overlap_sha256': sha(out / 'split_overlap.csv'),
        'old_training_in_current_validation': int(table.loc['validation', 'training']),
        'old_training_in_current_test': int(table.loc['test', 'training']),
        'implementation_reuse': 'SequenceGenerator preferred for WT-conditioned aligned-column prior generation',
        'historical_weights_as_current_formal_model': 'NOT_ADMITTED',
        'new_task_generation_advantage': 'UNKNOWN',
    }
    (out / 'source_checks.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    print(f'{len(receipts)} checkpoints strict-loaded; old training overlaps current validation/test: '
          f"{output['old_training_in_current_validation']}/{output['old_training_in_current_test']}")


if __name__ == '__main__':
    main()
