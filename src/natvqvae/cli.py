"""Command line entry point for data checks, training, generation, and evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__
from .auxiliary import generate_aux, train_aux
from .data import PopulationData, load_yaml, sha256_file
from .design import build_design, run_boltz, summarize_design
from .epcr import build_epcr, run_epcr_boltz, summarize_epcr
from .epcr_nmnat import build_epcr_nmnat_environment
from .evaluate import evaluate, evaluate_auxiliary, summarize_auxiliary, summarize_transfer
from .generate import generate
from .microenvironment import (build_microenvironment, run_microenvironment_boltz,
                               summarize_microenvironment)
from .nmnat_binding import (build_nmnat_binding, run_nmnat_binding_boltz,
                            summarize_joint_binding, summarize_nmnat_binding,
                            write_fixed_pose_reference,
                            write_single_substrate_pose_comparison)
from .nmnat_af3 import analyze_af3, build_af3_inputs, run_af3
from .nmnat_apo import analyze_apo
from .nmnat_mapping import audit_atp_contacts, build_nmnat_mapping
from .nmnat_md import (build_nmnat_md, run_nmnat_md_em, run_nmnat_md_pilot,
                       write_atom_mapping)
from .nmnat_nmn import analyze_nmn_sources
from .preprocessing import preprocess
from .train import train


def _candidate_structures(config, output):
    from .structure import sample_candidate_structures
    return sample_candidate_structures(config, output)


def _analyze_nmnat_md(config, output):
    from .nmnat_md_analysis import analyze_nmnat_md
    return analyze_nmnat_md(config, output)


def _structure_demo_summary(config, output):
    from .structure_demo import summarize_structure_demo
    return summarize_structure_demo(config, output)


def _structure_demo_render(config, output):
    from .structure_demo import render_structure_demo
    return render_structure_demo(config, output)


def _epcr_clone_demo(config, output):
    from .epcr_demo import render_epcr_clone_demo
    return render_epcr_clone_demo(config, _result_output(output, 'evaluation'))


def _candidate_delivery(config, output):
    from .delivery import build_candidate_delivery
    return build_candidate_delivery(config, _result_output(output, 'evaluation'))


def _fragment_overlap(config):
    from .fragment_overlap import check_fragment_overlap
    study = load_yaml(config)
    return check_fragment_overlap(config, _result_output(study['output'], 'preprocessing'))


def _generation_backgrounds(args):
    from .generate import prepare_generation_backgrounds
    return prepare_generation_backgrounds(args.config, args.sources, args.target,
                                          args.test_data, _result_output(args.output, 'preprocessing'))


def _split_relations(args):
    from .split_relations import analyze_split_relations
    return analyze_split_relations(args.config)


def _fragment_containment(args):
    from .split_relations import analyze_fragment_containment
    return analyze_fragment_containment(args.config)


def _sequence_neighbors(args):
    from .split_relations import analyze_sequence_neighbors
    return analyze_sequence_neighbors(args.config, reuse_alignments=args.reuse_alignments)


def _fragment_pretraining(config):
    from .fragment_pretraining import build_fragment_pretraining
    study = load_yaml(config)
    return build_fragment_pretraining(config, _result_output(study['output'], 'preprocessing'))


def _fragment_plot(args):
    from .fragment_pretraining import plot_fragment_matching
    return plot_fragment_matching(_result_output(args.directory, 'preprocessing'))


def _test_recover(args):
    from .test_data import recover_test_data
    study = load_yaml(args.config)
    return recover_test_data(args.config,
                             _result_output(args.output or study['output'], 'preprocessing'))


def _test_history(args):
    from .test_data import inspect_test_history
    return inspect_test_history(args.config, _result_output(args.recovered_dir, 'preprocessing'))


def _fragment_compare(args):
    from .fragment_comparison import compare_fragment_transfer
    return compare_fragment_transfer(args.config, args.evaluation_dir, args.esm_dir,
                                     _result_output(args.output, 'evaluation'),
                                     args.split, args.test_data, args.neighbors_dir)


def _generation_compare(args):
    from .generation_comparison import compare_generation
    return compare_generation(args.config, _result_output(args.output, 'evaluation'))


def _six_cell_compare(args):
    from .fragment_comparison import compare_six_cells
    return compare_six_cells(args.config, _result_output(args.output, 'evaluation'), split=args.split)


def _six_cell_plot(args):
    from .fragment_comparison import plot_six_cells
    return plot_six_cells(_result_output(args.comparison, 'evaluation'))


def _head_mixture(args):
    from .head_mixture import run_head_mixture
    return run_head_mixture(args.config, _result_output(args.output, 'evaluation'),
                            getattr(args, 'fit', None))


def _head_probabilities(args):
    from .head_probabilities import export_head_probabilities
    return _head_run(export_head_probabilities, args, split=args.split, device=args.device)


def _head_calibration(args):
    from .head_mixture import run_head_calibration
    return _head_run(run_head_calibration, args, fitted_path=getattr(args, 'fit', None))


def _head_complementarity(args):
    from .head_complementarity import run_head_complementarity
    return _head_run(run_head_complementarity, args, split=args.split)


def _head_run(function, args, **kwargs):
    from .data import write_json
    output = _result_output(args.output, 'evaluation')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    try:
        return function(args.config, output, **kwargs)
    except BaseException as error:
        output.mkdir(parents=True, exist_ok=True)
        failure = {'status': 'FAILED', 'error_type': type(error).__name__, 'error': str(error),
                   'retry': 'Use a new empty output directory; existing results are never overwritten.'}
        write_json(output / 'failure.json', failure)
        identity = output / 'identity.json'
        if identity.exists():
            write_json(identity, json.loads(identity.read_text()) | failure)
        raise


def _latent_precision(args):
    from .latent_precision import summarize_latent_precision
    return summarize_latent_precision(args.comparison_dir,
                                      _result_output(args.output, 'evaluation'))


def _fragment_generation_summary(args):
    from .fragment_comparison import summarize_fragment_generation
    return summarize_fragment_generation(args.config, args.generation_dir,
                                          _result_output(args.output, 'evaluation'))


def _esm_evaluate(args):
    from .esm_baseline import evaluate_esm
    return evaluate_esm(args.config, args.sources, args.target,
                        _result_output(args.output, 'evaluation'), args.device, args.split, args.test_data)


def _esm_compare(args):
    from .esm_baseline import compare_esm_baseline
    return compare_esm_baseline(args.config, args.esm_dir,
                                _result_output(args.output, 'evaluation'))


def _esm_plot(args):
    from .esm_baseline import plot_esm_comparison
    return plot_esm_comparison(_result_output(args.comparison, 'evaluation'))


def _result_output(value, category):
    path = Path(value).resolve()
    root = (Path(__file__).resolve().parents[2] / 'data' / 'results' / category).resolve()
    if root not in path.parents:
        raise ValueError(f'Output must be a run directory under data/results/{category}/')
    return path


def _evaluate_batch(args):
    from .evaluate import evaluate_many

    runs = load_yaml(args.runs)
    if not isinstance(runs, list) or not runs:
        raise ValueError('Evaluation batch file must contain a non-empty list of runs')
    routed = []
    for item in runs:
        if set(item) != {'checkpoint', 'output'}:
            raise ValueError('Each evaluation batch run needs checkpoint and output')
        routed.append({'checkpoint': item['checkpoint'],
                       'output': str(_result_output(item['output'], 'evaluation'))})
    return evaluate_many(args.config, args.sources, args.target, routed, args.device,
                         args.split, args.limit, args.test_data)


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
    esm = sub.add_parser('esm-evaluate', help='score the frozen ESM-2 head on identical validation masks')
    for name, options in common.items():
        esm.add_argument('--' + name, **options)
    esm.add_argument('--output', required=True)
    esm.add_argument('--split', choices=('validation', 'test'), default='validation')
    esm.add_argument('--test-data', default=None)
    esm.set_defaults(func=_esm_evaluate)
    esm_compare = sub.add_parser('esm-compare', help='compare ESM-2 and NatV scores on paired validation tasks')
    esm_compare.add_argument('--config', default='data/results/evaluation/natv_transfer/manifest.json')
    esm_compare.add_argument('--esm-dir', required=True)
    esm_compare.add_argument('--output', required=True)
    esm_compare.set_defaults(func=_esm_compare)
    esm_plot = sub.add_parser('esm-plot', help='export a figure from existing ESM comparison results')
    esm_plot.add_argument('--comparison', required=True)
    esm_plot.set_defaults(func=_esm_plot)
    prep = sub.add_parser('preprocess', help='recover and stratify the old PF01467-only corpus')
    prep.add_argument('--config', default='config/pf01467.yaml')
    prep.set_defaults(func=lambda args: preprocess(args.config))

    overlap = sub.add_parser('fragment-overlap', help='check auxiliary fragments against NatV splits')
    overlap.add_argument('--config', default='config/fragment_overlap.yaml')
    overlap.set_defaults(func=lambda args: _fragment_overlap(args.config))

    split_relations = sub.add_parser(
        'split-relations', help='restore component_v2 rules and report cross-split sequence relations')
    split_relations.add_argument('--config', default='config/natv_split_relations.yaml')
    split_relations.set_defaults(func=_split_relations)

    containment = sub.add_parser(
        'fragment-containment', help='report exact short-fragment containment across auxiliary and NatV splits')
    containment.add_argument('--config', default='config/natv_fragment_containment.yaml')
    containment.set_defaults(func=_fragment_containment)

    sequence_neighbors = sub.add_parser(
        'sequence-neighbors', help='report cross-split nearest neighbours for full and domain sequences')
    sequence_neighbors.add_argument('--config', default='config/natv_sequence_neighbors.yaml')
    sequence_neighbors.add_argument('--reuse-alignments', action='store_true')
    sequence_neighbors.set_defaults(func=_sequence_neighbors)

    paired_fragments = sub.add_parser('fragment-pretraining', help='prepare matched external and NatV training fragments')
    paired_fragments.add_argument('--config', default='config/fragment_pretraining.yaml')
    paired_fragments.set_defaults(func=lambda args: _fragment_pretraining(args.config))

    fragment_plot = sub.add_parser('fragment-plot', help='plot the prepared fragment-length and alignment matching')
    fragment_plot.add_argument('--directory', required=True)
    fragment_plot.set_defaults(func=_fragment_plot)

    test_recover = sub.add_parser('test-recover', help='recover test sequences and CDD annotations without scoring')
    test_recover.add_argument('--config', default='config/natv_test_recovery.yaml')
    test_recover.add_argument('--output', default=None)
    test_recover.set_defaults(func=_test_recover)

    test_history = sub.add_parser('test-history', help='verify historical test membership without reading score values')
    test_history.add_argument('--config', default='config/natv_test_history.yaml')
    test_history.add_argument('--recovered-dir', required=True)
    test_history.set_defaults(func=_test_history)

    fragment_compare = sub.add_parser('fragment-compare', help='compare completed random/self/external prediction scores')
    fragment_compare.add_argument('--config', default='config/fragment_transfer_comparison.yaml')
    fragment_compare.add_argument('--evaluation-dir', required=True)
    fragment_compare.add_argument('--esm-dir', required=True)
    fragment_compare.add_argument('--output', required=True)
    fragment_compare.add_argument('--split', choices=('validation', 'test'), default='validation')
    fragment_compare.add_argument('--test-data', default=None)
    fragment_compare.add_argument('--neighbors-dir', default=None)
    fragment_compare.set_defaults(func=_fragment_compare)

    six_cell = sub.add_parser('six-cell-compare', help='summarize the model-arm by pretraining-source six-cell comparison')
    six_cell.add_argument('--config', default='config/fragment_six_cell_comparison.yaml')
    six_cell.add_argument('--output', required=True)
    six_cell.add_argument('--split', choices=('validation', 'test'), default='validation')
    six_cell.set_defaults(func=_six_cell_compare)

    six_cell_plot = sub.add_parser('six-cell-plot', help='re-export six-cell figures from completed comparisons')
    six_cell_plot.add_argument('--comparison', required=True)
    six_cell_plot.set_defaults(func=_six_cell_plot)

    explain = sub.add_parser('head-complementarity', help='decompose existing frozen head-mixture scores')
    explain.add_argument('--config', default='config/head_complementarity.yaml')
    explain.add_argument('--output', required=True)
    explain.add_argument('--split', choices=('validation', 'test'), default='test')
    explain.set_defaults(func=_head_complementarity)

    for command in ('fit', 'evaluate'):
        calibration = sub.add_parser(f'head-calibration-{command}', help='temperature and probability mixture control')
        calibration.add_argument('--config', default='config/natv_head_calibration.yaml')
        calibration.add_argument('--output', required=True)
        if command == 'evaluate':
            calibration.add_argument('--fit', required=True)
        calibration.set_defaults(func=_head_calibration)

    mixture_fit = sub.add_parser('head-mixture-fit', help='fit ESM/task probability mixtures on validation only')
    mixture_fit.add_argument('--config', default='config/natv_head_mixture.yaml')
    mixture_fit.add_argument('--output', required=True)
    mixture_fit.set_defaults(func=_head_mixture)

    head_probs = sub.add_parser('head-probabilities', help='export full ESM and deterministic site probabilities')
    head_probs.add_argument('--config', default='config/natv_head_mixture.yaml')
    head_probs.add_argument('--output', required=True)
    head_probs.add_argument('--split', choices=('validation', 'test'), default='validation')
    head_probs.add_argument('--device', default='cuda')
    head_probs.set_defaults(func=_head_probabilities)

    mixture_test = sub.add_parser('head-mixture-evaluate', help='apply frozen validation mixture weights to held-out scores')
    mixture_test.add_argument('--config', default='config/natv_head_mixture.yaml')
    mixture_test.add_argument('--fit', required=True)
    mixture_test.add_argument('--output', required=True)
    mixture_test.set_defaults(func=_head_mixture)

    generation_compare = sub.add_parser('generation-compare', help='compare fixed-background six-cell generation runs')
    generation_compare.add_argument('--config', default='config/fragment_generation_comparison.yaml')
    generation_compare.add_argument('--output', required=True)
    generation_compare.set_defaults(func=_generation_compare)

    latent_precision = sub.add_parser(
        'latent-precision', help='summarize scoring repeats of the latent-marginal comparison')
    latent_precision.add_argument('--comparison-dir', required=True)
    latent_precision.add_argument('--output', required=True)
    latent_precision.set_defaults(func=_latent_precision)

    fragment_generation = sub.add_parser('fragment-generation-summary', help='compare nine completed KVP40 generation runs')
    fragment_generation.add_argument('--config', default='config/fragment_transfer_comparison.yaml')
    fragment_generation.add_argument('--generation-dir', required=True)
    fragment_generation.add_argument('--output', required=True)
    fragment_generation.set_defaults(func=_fragment_generation_summary)

    summary = sub.add_parser('aux-summary', help='compare completed auxiliary runs and select WT candidates')
    summary.add_argument('--config', default='config/aux_qvae.yaml')
    summary.add_argument('--training', required=True)
    summary.add_argument('--generation', required=True)
    summary.add_argument('--evaluation', required=True)
    summary.set_defaults(func=lambda args: summarize_auxiliary(
        args.config, _result_output(args.training, 'training'),
        _result_output(args.generation, 'generation'), _result_output(args.evaluation, 'evaluation')))

    transfer_summary = sub.add_parser(
        'transfer-summary', help='rebuild full NatV transfer summaries from existing outputs')
    transfer_summary.add_argument('--manifest', required=True)
    transfer_summary.add_argument('--output', required=True)
    transfer_summary.set_defaults(func=lambda args: summarize_transfer(
        args.manifest, _result_output(args.output, 'evaluation')))

    trainer = sub.add_parser('train', help='train a QVAE or deterministic model from raw population and split config')
    trainer.add_argument('--config', **common['config'])
    trainer.add_argument('--sources', **common['sources'])
    trainer.add_argument('--target', **common['target'])
    trainer.add_argument('--device', **common['device'])
    trainer.add_argument('--output', required=True)
    trainer.add_argument('--init-checkpoint', default=None)
    trainer.add_argument('--seed', type=int, default=None)
    trainer.add_argument('--arm', choices=('QVAE', 'deterministic'), default='QVAE')
    trainer.set_defaults(func=lambda args: train(
        args.config, args.sources, args.target, _result_output(args.output, 'training'),
        args.device, args.init_checkpoint, args.seed, args.arm))

    generator = sub.add_parser('generate', help='sample full NatV variants from a checkpoint')
    generator.add_argument('--config', **common['config'])
    generator.add_argument('--sources', **common['sources'])
    generator.add_argument('--target', **common['target'])
    generator.add_argument('--device', **common['device'])
    generator.add_argument('--checkpoint', required=True)
    generator.add_argument('--output', required=True)
    generator.add_argument('--group', default=None)
    generator.add_argument('--draws', type=int, default=None)
    generator.add_argument('--distribution', choices=('joint', 'site_marginal'), default='joint')
    generator.add_argument('--background-id', default=None)
    generator.add_argument('--test-data', default=None)
    generator.add_argument('--backgrounds-manifest', default=None)
    generator.set_defaults(func=lambda args: generate(
        args.config, args.sources, args.target, args.checkpoint, _result_output(args.output, 'generation'),
        args.device, args.group, args.draws, args.distribution, args.background_id, args.test_data,
        args.backgrounds_manifest))

    backgrounds = sub.add_parser('generation-backgrounds', help='fix one held-out background per component')
    backgrounds.add_argument('--config', **common['config'])
    backgrounds.add_argument('--sources', **common['sources'])
    backgrounds.add_argument('--target', **common['target'])
    backgrounds.add_argument('--test-data', required=True)
    backgrounds.add_argument('--output', required=True)
    backgrounds.set_defaults(func=_generation_backgrounds)

    evaluator = sub.add_parser('evaluate', help='evaluate a checkpoint on validation or explicit recovered test data')
    evaluator.add_argument('--config', **common['config'])
    evaluator.add_argument('--sources', **common['sources'])
    evaluator.add_argument('--target', **common['target'])
    evaluator.add_argument('--device', **common['device'])
    evaluator.add_argument('--checkpoint', required=True)
    evaluator.add_argument('--output', required=True)
    evaluator.add_argument('--split', choices=('validation', 'test'), default='validation')
    evaluator.add_argument('--test-data', default=None)
    evaluator.add_argument('--limit', type=int, default=None)
    evaluator.set_defaults(func=lambda args: evaluate(
        args.config, args.sources, args.target, args.checkpoint, _result_output(args.output, 'evaluation'),
        args.device, args.split, args.limit, args.test_data))

    evaluator_batch = sub.add_parser('evaluate-batch', help='evaluate several checkpoints with one encoder load')
    evaluator_batch.add_argument('--config', **common['config'])
    evaluator_batch.add_argument('--sources', **common['sources'])
    evaluator_batch.add_argument('--target', **common['target'])
    evaluator_batch.add_argument('--device', **common['device'])
    evaluator_batch.add_argument('--runs', required=True)
    evaluator_batch.add_argument('--split', choices=('validation', 'test'), default='validation')
    evaluator_batch.add_argument('--test-data', default=None)
    evaluator_batch.add_argument('--limit', type=int, default=None)
    evaluator_batch.set_defaults(func=_evaluate_batch)

    aux_trainer = sub.add_parser('aux-train', help='train QVAE and deterministic arms on auxiliary domains')
    aux_trainer.add_argument('--config', default='config/aux_qvae.yaml')
    aux_trainer.add_argument('--device', default=None)
    aux_trainer.add_argument('--seed', type=int, default=None)
    aux_trainer.add_argument('--output', required=True)
    aux_trainer.set_defaults(func=lambda args: train_aux(
        args.config, _result_output(args.output, 'training'), args.device, args.seed))

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

    structures = sub.add_parser('candidate-structures',
                                help='sample full-length structures for selected candidates')
    structures.add_argument('--config', default='config/candidate_structures.yaml')
    structures.add_argument('--output', required=True)
    structures.set_defaults(func=lambda args: _candidate_structures(
        args.config, _result_output(args.output, 'evaluation')))

    design_build = sub.add_parser('design-build',
                                  help='build the bounded NatV A/B/A+B design inputs')
    design_build.add_argument('--config', default='config/design.yaml')
    design_build.add_argument('--output', required=True)
    design_build.set_defaults(func=lambda args: build_design(
        args.config, _result_output(args.output, 'generation')))

    design_boltz = sub.add_parser('design-boltz',
                                  help='run formal Boltz-2 predictions for the design')
    design_boltz.add_argument('--config', default='config/design.yaml')
    design_boltz.add_argument('--output', required=True)
    design_boltz.set_defaults(func=lambda args: run_boltz(
        args.config, _result_output(args.output, 'evaluation')))

    design_summary = sub.add_parser('design-summarize',
                                    help='compare design structures with parent designs and templates')
    design_summary.add_argument('--config', default='config/design.yaml')
    design_summary.add_argument('--output', required=True)
    design_summary.set_defaults(func=lambda args: summarize_design(
        args.config, _result_output(args.output, 'evaluation')))

    microenv_build = sub.add_parser('microenv-build',
                                    help='build Nudix D2/D4 inputs for WT/B1/B2')
    microenv_build.add_argument('--config', default='config/microenvironment.yaml')
    microenv_build.add_argument('--output', required=True)
    microenv_build.set_defaults(func=lambda args: build_microenvironment(
        args.config, _result_output(args.output, 'evaluation')))

    microenv_boltz = sub.add_parser('microenv-boltz',
                                    help='run the recorded Boltz-2 micro-environment states')
    microenv_boltz.add_argument('--config', default='config/microenvironment.yaml')
    microenv_boltz.add_argument('--output', required=True)
    microenv_boltz.set_defaults(func=lambda args: run_microenvironment_boltz(
        args.config, _result_output(args.output, 'evaluation')))

    microenv_summary = sub.add_parser('microenv-summarize',
                                      help='summarize Nudix metal-site metrics from existing models')
    microenv_summary.add_argument('--config', default='config/microenvironment.yaml')
    microenv_summary.add_argument('--output', required=True)
    microenv_summary.set_defaults(func=lambda args: summarize_microenvironment(
        args.config, _result_output(args.output, 'evaluation')))

    epcr_build = sub.add_parser('epcr-build',
                                help='build full-length NatV10/NatV89 inputs from Sanger calls')
    epcr_build.add_argument('--config', default='config/epcr_full_length.yaml')
    epcr_build.add_argument('--output', required=True)
    epcr_build.set_defaults(func=lambda args: build_epcr(
        args.config, _result_output(args.output, 'evaluation')))

    epcr_boltz = sub.add_parser('epcr-boltz',
                                help='run ligand-free full-length structures for WT/NatV10/NatV89')
    epcr_boltz.add_argument('--config', default='config/epcr_full_length.yaml')
    epcr_boltz.add_argument('--output', required=True)
    epcr_boltz.set_defaults(func=lambda args: run_epcr_boltz(
        args.config, _result_output(args.output, 'evaluation')))

    epcr_summary = sub.add_parser('epcr-summarize',
                                  help='compare WT/NatV10/NatV89 full-length structures')
    epcr_summary.add_argument('--config', default='config/epcr_full_length.yaml')
    epcr_summary.add_argument('--output', required=True)
    epcr_summary.set_defaults(func=lambda args: summarize_epcr(
        args.config, _result_output(args.output, 'evaluation')))

    nmnat_map = sub.add_parser(
        'nmnat-map', help='map state-specific NMNAT contacts onto matched full-length WT models')
    nmnat_map.add_argument('--config', default='config/nmnat_mapping.yaml')
    nmnat_map.add_argument('--output', required=True)
    nmnat_map.set_defaults(func=lambda args: build_nmnat_mapping(
        args.config, _result_output(args.output, 'evaluation')))

    nmnat_audit = sub.add_parser(
        'nmnat-audit', help='atom/group-level audit of 1F9A ATP/Mg contacts on matched WT models')
    nmnat_audit.add_argument('--config', default='config/nmnat_mapping.yaml')
    nmnat_audit.add_argument('--output', required=True)
    nmnat_audit.set_defaults(func=lambda args: audit_atp_contacts(
        args.config, _result_output(args.output, 'evaluation')))

    epcr_nmnat = sub.add_parser(
        'epcr-nmnat', help='compare NMNAT substrate environments across ePCR WT/NatV10/NatV89')
    epcr_nmnat.add_argument('--config', default='config/epcr_nmnat_environment.yaml')
    epcr_nmnat.add_argument('--output', required=True)
    epcr_nmnat.set_defaults(func=lambda args: build_epcr_nmnat_environment(
        args.config, _result_output(args.output, 'evaluation')))

    epcr_demo = sub.add_parser(
        'epcr-demo', help='render independent WT/NatV10/NatV89 clone comparison')
    epcr_demo.add_argument('--config', default='config/epcr_clone_demo.yaml')
    epcr_demo.add_argument('--output', required=True)
    epcr_demo.set_defaults(func=lambda args: _epcr_clone_demo(args.config, args.output))

    delivery_build = sub.add_parser(
        'delivery-build', help='build unified 14-protein NatV handoff package')
    delivery_build.add_argument('--config', default='config/natv_candidate_delivery.yaml')
    delivery_build.add_argument('--output', required=True)
    delivery_build.set_defaults(func=lambda args: _candidate_delivery(args.config, args.output))

    nmnat_binding_build = sub.add_parser(
        'nmnat-binding-build', help='build NMNAT substrate co-folding inputs')
    nmnat_binding_build.add_argument('--config', default='config/nmnat_binding.yaml')
    nmnat_binding_build.add_argument('--output', required=True)
    nmnat_binding_build.set_defaults(func=lambda args: build_nmnat_binding(
        args.config, _result_output(args.output, 'evaluation')))

    nmnat_binding_run = sub.add_parser(
        'nmnat-binding-run', help='run NMNAT substrate co-folding with Boltz-2')
    nmnat_binding_run.add_argument('--config', default='config/nmnat_binding.yaml')
    nmnat_binding_run.add_argument('--output', required=True)
    nmnat_binding_run.add_argument('--wait-for-pid', type=int, default=None,
                                  help='wait for an existing coordinated compute process to exit')
    nmnat_binding_run.add_argument('--checkpoint-stage-dir', default=None,
                                  help='verified temporary checkpoint copy for slow WSL storage')
    nmnat_binding_run.set_defaults(func=lambda args: run_nmnat_binding_boltz(
        args.config, _result_output(args.output, 'evaluation'), args.wait_for_pid,
        args.checkpoint_stage_dir))

    nmnat_binding_summary = sub.add_parser(
        'nmnat-binding-summarize', help='summarize NMNAT substrate co-folding models')
    nmnat_binding_summary.add_argument('--config', default='config/nmnat_binding.yaml')
    nmnat_binding_summary.add_argument('--output', required=True)
    nmnat_binding_summary.set_defaults(func=lambda args: summarize_nmnat_binding(
        args.config, _result_output(args.output, 'evaluation')))

    nmnat_joint_reference = sub.add_parser(
        'nmnat-joint-reference',
        help='combine fixed ATP-only and NMN-only poses in a common NMNAT frame')
    nmnat_joint_reference.add_argument('--config', default='config/nmnat_joint.yaml')
    nmnat_joint_reference.add_argument('--output', required=True)
    nmnat_joint_reference.set_defaults(func=lambda args: write_fixed_pose_reference(
        args.config, _result_output(args.output, 'evaluation')))

    nmnat_joint_summary = sub.add_parser(
        'nmnat-joint-summarize', help='summarize ATP+NMN joint co-folding models')
    nmnat_joint_summary.add_argument('--config', default='config/nmnat_joint.yaml')
    nmnat_joint_summary.add_argument('--output', required=True)
    nmnat_joint_summary.set_defaults(func=lambda args: summarize_joint_binding(
        args.config, _result_output(args.output, 'evaluation')))

    nmnat_joint_poses = sub.add_parser(
        'nmnat-joint-poses',
        help='compare joint ATP/NMN poses with same-seed single-substrate models')
    nmnat_joint_poses.add_argument('--config', default='config/nmnat_joint.yaml')
    nmnat_joint_poses.add_argument('--output', required=True)
    nmnat_joint_poses.set_defaults(func=lambda args: write_single_substrate_pose_comparison(
        args.config, _result_output(args.output, 'evaluation')))

    nmnat_md_build = sub.add_parser(
        'nmnat-md-build',
        help='build explicit-solvent WT ATP/NMN GROMACS systems')
    nmnat_md_build.add_argument('--config', default='config/nmnat_md.yaml')
    nmnat_md_build.add_argument('--output', required=True)
    nmnat_md_build.set_defaults(func=lambda args: build_nmnat_md(
        args.config, _result_output(args.output, 'evaluation')))

    nmnat_md_em = sub.add_parser(
        'nmnat-md-em', help='energy-minimize the prepared WT ATP/NMN systems')
    nmnat_md_em.add_argument('--config', default='config/nmnat_md.yaml')
    nmnat_md_em.add_argument('--output', required=True)
    nmnat_md_em.set_defaults(func=lambda args: run_nmnat_md_em(
        args.config, _result_output(args.output, 'evaluation')))

    nmnat_md_run = sub.add_parser(
        'nmnat-md-run', help='run the WT ATP/NMN NVT-NPT-production pilot')
    nmnat_md_run.add_argument('--config', default='config/nmnat_md.yaml')
    nmnat_md_run.add_argument('--output', required=True)
    nmnat_md_run.add_argument('--wait-gpu', action='store_true')
    nmnat_md_run.set_defaults(func=lambda args: run_nmnat_md_pilot(
        args.config, _result_output(args.output, 'evaluation'), args.wait_gpu))

    nmnat_md_map = sub.add_parser(
        'nmnat-md-map', help='write topology-index to CCD-name mapping metadata')
    nmnat_md_map.add_argument('--config', default='config/nmnat_md.yaml')
    nmnat_md_map.add_argument('--output', required=True)
    nmnat_md_map.set_defaults(func=lambda args: write_atom_mapping(
        args.config, _result_output(args.output, 'evaluation')))

    nmnat_md_analyze = sub.add_parser(
        'nmnat-md-analyze', help='analyze completed WT ATP/NMN MD trajectories')
    nmnat_md_analyze.add_argument('--config', default='config/nmnat_md.yaml')
    nmnat_md_analyze.add_argument('--output', required=True)
    nmnat_md_analyze.set_defaults(func=lambda args: _analyze_nmnat_md(
        args.config, _result_output(args.output, 'evaluation')))

    nmnat_nmn_sources = sub.add_parser(
        'nmnat-nmn-sources', help='map experimental NMN contacts by chemical group and source family')
    nmnat_nmn_sources.add_argument('--config', default='config/nmnat_nmn_sources.yaml')
    nmnat_nmn_sources.add_argument('--output', required=True)
    nmnat_nmn_sources.set_defaults(func=lambda args: analyze_nmn_sources(
        args.config, _result_output(args.output, 'evaluation')))

    nmnat_af3_build = sub.add_parser(
        'nmnat-af3-build', help='build AF3 inputs for the WT ATP/NMN joint pose check')
    nmnat_af3_build.add_argument('--config', default='config/nmnat_af3.yaml')
    nmnat_af3_build.add_argument('--output', required=True)
    nmnat_af3_build.set_defaults(func=lambda args: build_af3_inputs(
        args.config, _result_output(args.output, 'evaluation')))

    nmnat_af3_run = sub.add_parser(
        'nmnat-af3-run', help='run the AF3 WT ATP/NMN and 1F9A reference jobs')
    nmnat_af3_run.add_argument('--config', default='config/nmnat_af3.yaml')
    nmnat_af3_run.add_argument('--output', required=True)
    nmnat_af3_run.add_argument('--wait-gpu', action='store_true')
    nmnat_af3_run.set_defaults(func=lambda args: run_af3(
        args.config, _result_output(args.output, 'evaluation'), args.wait_gpu))

    nmnat_af3_analyze = sub.add_parser(
        'nmnat-af3-analyze', help='analyze AF3 WT and 1F9A reference models')
    nmnat_af3_analyze.add_argument('--config', default='config/nmnat_af3.yaml')
    nmnat_af3_analyze.add_argument('--output', required=True)
    nmnat_af3_analyze.set_defaults(func=lambda args: analyze_af3(
        args.config, _result_output(args.output, 'evaluation')))

    nmnat_af3_apo = sub.add_parser(
        'nmnat-af3-apo', help='compare WT holo models with existing BioEmu apo frames')
    nmnat_af3_apo.add_argument('--config', default='config/nmnat_af3.yaml')
    nmnat_af3_apo.add_argument('--output', required=True)
    nmnat_af3_apo.set_defaults(func=lambda args: analyze_apo(
        args.config, _result_output(args.output, 'evaluation')))

    demo_summary = sub.add_parser(
        'structure-demo-summary',
        help='summarize existing design structures and write render inputs')
    demo_summary.add_argument('--config', default='config/structure_demo.yaml')
    demo_summary.add_argument('--output', required=True)
    demo_summary.set_defaults(func=lambda args: _structure_demo_summary(
        args.config, _result_output(args.output, 'evaluation')))

    demo_render = sub.add_parser(
        'structure-demo-render', help='render the Blender structure demo')
    demo_render.add_argument('--config', default='config/structure_demo.yaml')
    demo_render.add_argument('--output', required=True)
    demo_render.set_defaults(func=lambda args: _structure_demo_render(
        args.config, _result_output(args.output, 'evaluation')))
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
