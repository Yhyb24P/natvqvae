"""Command line entry point for data checks, training, generation, and evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__
from .auxiliary import generate_aux, train_aux
from .data import PopulationData, load_yaml, sha256_file
from .evaluate import evaluate, evaluate_auxiliary, summarize_auxiliary
from .generate import generate
from .preprocessing import preprocess
from .train import train


def _result_output(value, category):
    path = Path(value).resolve()
    root = (Path(__file__).resolve().parents[2] / 'data' / 'results' / category).resolve()
    if root not in path.parents:
        raise ValueError(f'Output must be a run directory under data/results/{category}/')
    return path


def _doctor(args):
    data = PopulationData(args.sources, args.config, args.target)
    study = load_yaml(args.config)
    sources = load_yaml(args.sources)
    checkpoints = sorted(Path(sources['checkpoint_dir']).glob('*/best.pt'))
    print(f'natvqvae {__version__}')
    print(f"target {data.target.name}: {len(data.target.sequence)} aa")
    print(f"population {sources['population']}: {len(data.population)} sequences")
    print(f"training {len(data.train)} sequences / {len(set(r.component for r in data.train))} components")
    print(f"validation {len(data.validation)} sequences / {len(set(r.component for r in data.validation))} components")
    print(f"test ids {len(data.test_rows)} (not loaded as sequences)")
    print(f"ESM model {sources['esm_model']}")
    print(f"checkpoints {len(checkpoints)}")
    for path in checkpoints:
        print(f'  {path}')
    for key in ('population', 'esm_model'):
        path = Path(sources[key])
        print(f"  [{'OK' if path.exists() else 'MISSING'}] {key}: {path}")
    if not Path(sources['esm_model']).exists() or not checkpoints:
        return 1
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog='natvqvae', description='KVP40 NatV QVAE generation')
    parser.add_argument('--version', action='version', version=f'natvqvae {__version__}')
    sub = parser.add_subparsers(dest='command', required=True)

    doctor = sub.add_parser('doctor', help='check target, raw inputs, splits and checkpoints')
    doctor.add_argument('--config', default='config/study.yaml')
    doctor.add_argument('--sources', default='config/sources.yaml')
    doctor.add_argument('--target', default='config/target.yaml')
    doctor.set_defaults(func=_doctor)

    common = {
        'config': dict(default='config/study.yaml'),
        'sources': dict(default='config/sources.yaml'),
        'target': dict(default='config/target.yaml'),
        'device': dict(default=None),
    }
    prep = sub.add_parser('preprocess', help='recover and stratify the old PF01467-only corpus')
    prep.add_argument('--config', default='config/pf01467.yaml')
    prep.set_defaults(func=lambda args: preprocess(args.config))

    summary = sub.add_parser('aux-summary', help='compare completed auxiliary runs and select WT candidates')
    summary.add_argument('--config', default='config/aux_qvae.yaml')
    summary.add_argument('--training', required=True)
    summary.add_argument('--generation', required=True)
    summary.add_argument('--evaluation', required=True)
    summary.set_defaults(func=lambda args: summarize_auxiliary(
        args.config, _result_output(args.training, 'training'),
        _result_output(args.generation, 'generation'), _result_output(args.evaluation, 'evaluation')))

    trainer = sub.add_parser('train', help='train a QVAE from raw population and split config')
    trainer.add_argument('--config', **common['config'])
    trainer.add_argument('--sources', **common['sources'])
    trainer.add_argument('--target', **common['target'])
    trainer.add_argument('--device', **common['device'])
    trainer.add_argument('--output', required=True)
    trainer.add_argument('--init-checkpoint', default=None)
    trainer.add_argument('--seed', type=int, default=None)
    trainer.set_defaults(func=lambda args: train(
        args.config, args.sources, args.target, _result_output(args.output, 'training'),
        args.device, args.init_checkpoint, args.seed))

    generator = sub.add_parser('generate', help='sample full NatV variants from a checkpoint')
    generator.add_argument('--config', **common['config'])
    generator.add_argument('--sources', **common['sources'])
    generator.add_argument('--target', **common['target'])
    generator.add_argument('--device', **common['device'])
    generator.add_argument('--checkpoint', required=True)
    generator.add_argument('--output', required=True)
    generator.add_argument('--group', default=None)
    generator.add_argument('--draws', type=int, default=None)
    generator.set_defaults(func=lambda args: generate(
        args.config, args.sources, args.target, args.checkpoint, _result_output(args.output, 'generation'),
        args.device, args.group, args.draws))

    evaluator = sub.add_parser('evaluate', help='evaluate a checkpoint on validation masks')
    evaluator.add_argument('--config', **common['config'])
    evaluator.add_argument('--sources', **common['sources'])
    evaluator.add_argument('--target', **common['target'])
    evaluator.add_argument('--device', **common['device'])
    evaluator.add_argument('--checkpoint', required=True)
    evaluator.add_argument('--output', required=True)
    evaluator.add_argument('--split', default='validation')
    evaluator.add_argument('--limit', type=int, default=None)
    evaluator.set_defaults(func=lambda args: evaluate(
        args.config, args.sources, args.target, args.checkpoint, _result_output(args.output, 'evaluation'),
        args.device, args.split, args.limit))

    aux_trainer = sub.add_parser('aux-train', help='train QVAE and deterministic arms on auxiliary domains')
    aux_trainer.add_argument('--config', default='config/aux_qvae.yaml')
    aux_trainer.add_argument('--device', default=None)
    aux_trainer.add_argument('--output', required=True)
    aux_trainer.set_defaults(func=lambda args: train_aux(
        args.config, _result_output(args.output, 'training'), args.device))

    aux_generator = sub.add_parser('aux-generate', help='sample auxiliary-domain variants from a QVAE checkpoint')
    aux_generator.add_argument('--config', default='config/aux_qvae.yaml')
    aux_generator.add_argument('--device', default=None)
    aux_generator.add_argument('--checkpoint', required=True)
    aux_generator.add_argument('--output', required=True)
    aux_generator.add_argument('--limit', type=int, default=None)
    aux_generator.add_argument('--draws', type=int, default=None)
    aux_generator.add_argument('--source', choices=('aux', 'wt_kvp40'), default='aux')
    aux_generator.set_defaults(func=lambda args: generate_aux(
        args.config, args.checkpoint, _result_output(args.output, 'generation'),
        args.device, args.limit, args.draws, args.source))

    aux_evaluator = sub.add_parser('aux-evaluate', help='evaluate an auxiliary checkpoint on held-out domains')
    aux_evaluator.add_argument('--config', default='config/aux_qvae.yaml')
    aux_evaluator.add_argument('--device', default=None)
    aux_evaluator.add_argument('--checkpoint', required=True)
    aux_evaluator.add_argument('--output', required=True)
    aux_evaluator.set_defaults(func=lambda args: evaluate_auxiliary(
        args.config, args.checkpoint, _result_output(args.output, 'evaluation'), args.device))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = args.func(args)
    if isinstance(result, Path):
        print(json.dumps({'status': 'COMPLETED', 'output': str(result)}, ensure_ascii=False))
        return 0
    if isinstance(result, dict):
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if isinstance(result, int):
        return result
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
