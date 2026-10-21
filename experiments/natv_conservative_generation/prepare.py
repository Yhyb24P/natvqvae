"""Pin the read-only QVAE sources and verify the formal generation inputs."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.dont_write_bytecode = True

from natvqvae.intake import load_manifest, sha256_file, source_root, verify_entry
from natvqvae.sequence import wt_sequence

RUN = Path(__file__).parent


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def fasta_records(path):
    records = {}
    current = None
    for line in path.read_text().splitlines():
        if line.startswith('>'):
            current = line[1:].split()[0]
            records[current] = ''
        else:
            records[current] += line.strip()
    return records


def main():
    plan = json.loads((RUN / 'plan.json').read_text())
    source = source_root(load_manifest()).resolve()
    out = RUN / 'inputs'
    out.mkdir(exist_ok=True)
    pinned = {}

    def pin(path, expected=None):
        path = Path(path)
        digest = sha256_file(path)
        if expected and digest != expected:
            raise ValueError('Input differs: ' + str(path))
        pinned[str(path)] = {'sha256': digest, 'bytes': path.stat().st_size}
        return path

    for path in sorted((source / 'src' / 'nad_qvae').rglob('*.py')):
        pin(path)
    for path in sorted((source / 'nad_models' / 'esm2').rglob('*')):
        if path.is_file():
            pin(path)

    freeze_path = source / 'experiments/run_reports/natv_h2_architecture_generation_v1_attempt1/generation/checkpoint_freeze.json'
    freeze = json.loads(freeze_path.read_text())
    categorical = {}
    for seed in plan['seeds']:
        entry = freeze[f'categorical/{seed}']
        pin(source / entry['path'], entry['sha256'])
        categorical[str(seed)] = {'path': entry['path'], 'sha256': entry['sha256']}
    pin(freeze_path)

    release = source / 'data/processed/releases/natv_formal_training_data_v1'
    for name in ['input_plan.json', 'training_samples.csv', 'train_only.fasta', 'allowed_samples.parquet']:
        pin(release / name)
    for path in sorted((release / 'retrieval_db').glob('train*')):
        if path.is_file():
            pin(path)

    paths = {}
    for entry in load_manifest()['entries']:
        if entry['name'] in ['allowed_samples', 'core_split', 'masked_input_plan', 'train_sequences']:
            result = verify_entry(entry, source)
            if not result.ok:
                raise ValueError(result.detail)
            paths[entry['name']] = pin(result.path, entry['sha256'])

    samples = pd.read_parquet(paths['allowed_samples'])
    train = samples[samples.training_role_v2 == 'training'].sort_values('sequence_id')
    core = json.loads(paths['core_split'].read_text())
    masked_plan = json.loads(paths['masked_input_plan'].read_text())
    if len(train) != 1644 or train.component_v2.nunique() != 47:
        raise ValueError('Training identity changed')
    if set(train.sequence_id) != set(core['splits']['training']):
        raise ValueError('Training role and split differ')
    if set(masked_plan['retrieval_database_ids']) != set(train.sequence_id):
        raise ValueError('Retrieval role identity changed')

    binary = Path(sys.executable).with_name('mmseqs')
    export = out / 'retrieval_database.fasta'
    subprocess.run([str(binary), 'convert2fasta', str(release / 'retrieval_db/train'), str(export), '-v', '0'], check=True)
    records = fasta_records(export)
    if records != dict(zip(train.sequence_id, train.sequence)):
        raise ValueError('MMseqs database does not equal formal train')

    train_fasta = pin(ROOT / 'discovery/natv_data_readiness/training.fasta')
    validation_fasta = pin(ROOT / 'discovery/natv_data_readiness/validation.fasta')
    if len(fasta_records(train_fasta)) != 1644 or len(fasta_records(validation_fasta)) != 227:
        raise ValueError('Exported role FASTA differs from formal counts')
    for name in ['src/natvqvae/conservative_generation.py', 'experiments/natv_conservative_generation/prepare.py']:
        pin(ROOT / name)

    wt = wt_sequence()
    regions = json.loads((ROOT / 'contracts/regions.json').read_text())
    wt_digest = hashlib.sha256(wt.encode()).hexdigest()
    if len(wt) != 341 or wt_digest != regions['target']['sequence_sha256']:
        raise ValueError('WT identity changed')

    write(out / 'source_checks.json', {
        'source_root': str(source),
        'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip(),
        'mmseqs_version': subprocess.check_output([str(binary), 'version'], text=True).strip(),
        'files': pinned,
        'checkpoint_freeze': {'path': str(freeze_path), 'sha256': sha256_file(freeze_path), 'categorical': categorical},
        'training_sequences': len(train),
        'training_components': int(train.component_v2.nunique()),
        'validation_sequences': len(fasta_records(validation_fasta)),
        'retrieval_records_verified': len(records),
        'wt_sha256': wt_digest,
        'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    })
    print(json.dumps({'source': str(source), 'files': len(pinned), 'checkpoints': len(categorical),
                      'training_sequences': len(train), 'retrieval_records': len(records)}), flush=True)


if __name__ == '__main__':
    main()
