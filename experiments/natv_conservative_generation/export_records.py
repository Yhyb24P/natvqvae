"""Merge per-group trajectories and edit scripts into committed records."""
from pathlib import Path
import gzip
import hashlib
import json

RUN = Path(__file__).parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def merge(name, expected):
    destination = RUN / f'{name}.jsonl.gz'
    count = 0
    with gzip.open(destination, 'wt') as sink:
        for group in json.loads((RUN / 'sampling_groups.json').read_text()):
            for seed in json.loads((RUN / 'plan.json').read_text())['seeds']:
                source = RUN / 'generation' / f'checkpoint_{seed}' / group['group_id'] / f'{name}.jsonl.gz'
                with gzip.open(source, 'rt') as handle:
                    for line in handle:
                        json.loads(line)
                        sink.write(line)
                        count += 1
    if count != expected:
        raise ValueError(f'{name} count differs: {count}')
    return {'rows': count, 'sha256': sha256(destination)}


def main():
    result = {
        'trajectory.jsonl.gz': merge('trajectory', 12000),
        'edit_scripts.jsonl.gz': merge('edit_scripts', 2400),
    }
    (RUN / 'export_checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
