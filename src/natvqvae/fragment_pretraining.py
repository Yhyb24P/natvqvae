"""Prepare full external fragments and length-matched NatV training fragments."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path

import Bio
import numpy as np

from .data import Target, load_yaml, sha256_file, stable_seed, write_json
from .fragment_overlap import _fasta, _rows, _write_csv, _write_fasta
from .model import AA
from .preprocessing import stable_alignment


def _sha(sequence):
    return hashlib.sha256(sequence.encode()).hexdigest()


def _input_records(config):
    paths = config['inputs']
    directory = Path(paths['relation_directory'])
    summary = json.loads((directory / 'summary.json').read_text())
    if (summary['status'] != 'complete_all_targets' or
            summary['auxiliary_relations']['retained_pool']['export_status'] !=
            'complete_relation_screened_pool_not_a_training_split'):
        raise ValueError('Relation-screened pool is incomplete')
    files = [directory / name for name in ['retained_aux_domains.fasta', 'retained_aux_masks.json',
                                          'retained_auxiliary.csv']]
    for path in files:
        if sha256_file(path) != summary['result_sha256'][path.name]:
            raise ValueError(f'Retained input SHA mismatch: {path}')
    target = Target.load(paths['target'])
    wt_start, wt_end = map(int, target.domains['NMNAT'])
    sequences = _fasta(files[0])
    masks = json.loads(files[1].read_text())
    auxiliary = {}
    for row in _rows(files[2]):
        uid = row['aux_uid']
        if uid in auxiliary or row['status'] != 'retained' or not row['original_domain_cluster50']:
            raise ValueError(f'Invalid retained auxiliary row: {uid}')
        sequence = sequences[uid]
        if _sha(sequence) != row['sequence_sha256'] or len(sequence) != int(row['length']):
            raise ValueError(f'Auxiliary sequence identity mismatch: {uid}')
        pairs = [(int(q), int(w)) for q, w in masks[uid]['pairs']]
        if (len({q for q, _ in pairs}) != len(pairs) or len({w for _, w in pairs}) != len(pairs)
                or any(not 0 <= q < len(sequence) or not wt_start <= w <= wt_end for q, w in pairs)):
            raise ValueError(f'Auxiliary mapping is not one-to-one or is out of bounds: {uid}')
        mapping = {w: q for q, w in pairs if sequence[q] in AA}
        if list(mapping[w] for w in sorted(mapping)) != sorted(mapping.values()):
            raise ValueError(f'Non-monotonic auxiliary mapping: {uid}')
        auxiliary[uid] = {**row, 'sequence': sequence, 'mapping': mapping}
    if set(auxiliary) != set(sequences) or set(auxiliary) != set(masks):
        raise ValueError('Retained FASTA, masks, and rows contain different auxiliary IDs')
    population = _fasta(paths['population'])
    training = []
    seen = set()
    for row in sorted(_rows(paths['training']), key=lambda value: value['sequence_id']):
        uid = row['sequence_id']
        if uid in seen:
            raise ValueError(f'Duplicate NatV training sequence ID: {uid}')
        seen.add(uid)
        sequence = population[uid]
        start, end = int(row['nmnat_cdd_from']) - 1, int(row['nmnat_cdd_to'])
        if (_sha(sequence) != uid or len(sequence) != int(row['length'])
                or not 0 <= start < end <= len(sequence)):
            raise ValueError(f'NatV training identity or CDD interval mismatch: {uid}')
        domain = sequence[start:end]
        alignment = stable_alignment(target.sequence[wt_start - 1:wt_end], domain,
                                     int(config['selection']['max_optimal_alignments']))
        mapping = {w + wt_start: q for w, q in alignment['stable'].items() if domain[q] in AA}
        training.append({'sequence_id': uid, 'component': row['component_v2'], 'full_length': len(sequence),
                         'domain': domain, 'domain_start': start, 'domain_end': end, 'mapping': mapping,
                         'alignment_n_optimal': alignment['n_optimal'], 'alignment_identity': alignment['identity']})
    files += [directory / 'summary.json', Path(paths['population']), Path(paths['training']), Path(paths['target'])]
    return auxiliary, training, wt_end, files


def extend_within_domain(core_start, core_end, domain_length, desired_length):
    """Keep the common-column interval, then extend symmetrically within NMNAT."""
    if not 0 <= core_start < core_end <= domain_length:
        raise ValueError('Invalid common-column interval')
    extra = max(0, desired_length - (core_end - core_start))
    start = max(0, core_start - extra // 2)
    end = min(domain_length, core_end + extra - extra // 2)
    remaining = max(0, desired_length - (end - start))
    left = min(start, remaining)
    start -= left
    end += min(domain_length - end, remaining - left)
    return start, end


def _matching_options(auxiliary, training, wt_end, selection):
    positions = np.full((len(training), wt_end + 1), -1, dtype=int)
    lengths = np.array([len(row['domain']) for row in training])
    for index, row in enumerate(training):
        for wt, query in row['mapping'].items():
            positions[index, wt] = query
    options, unmatched = {}, []
    for uid, record in sorted(auxiliary.items()):
        columns = sorted(record['mapping'])
        if len(columns) < 2:
            unmatched.append({'aux_uid': uid, 'reason': 'fewer_than_two_standard_mapped_columns',
                              'max_shared_fraction': 0, 'coverage_eligible_sources': 0})
            continue
        query_positions = positions[:, columns]
        valid = query_positions >= 0
        counts = valid.sum(axis=1)
        fractions = counts / len(columns)
        core_starts = np.where(valid, query_positions, np.iinfo(np.int32).max).min(axis=1)
        core_ends = query_positions.max(axis=1) + 1
        core_lengths = core_ends - core_starts
        desired = len(record['sequence'])
        extra = np.maximum(0, desired - core_lengths)
        starts = np.maximum(0, core_starts - extra // 2)
        ends = np.minimum(lengths, core_ends + extra - extra // 2)
        remaining = np.maximum(0, desired - (ends - starts))
        left = np.minimum(starts, remaining)
        starts -= left
        ends += np.minimum(lengths - ends, remaining - left)
        coverage = (counts >= 2) & (fractions >= float(selection['min_shared_fraction']))
        eligible = np.where(coverage & (np.abs(ends - starts - desired) <=
                                        float(selection['relative_length_tolerance']) * desired))[0]
        if not len(eligible):
            unmatched.append({'aux_uid': uid, 'reason': 'no_source_with_required_coverage_and_length',
                              'max_shared_fraction': float(fractions.max()) if len(fractions) else 0,
                              'coverage_eligible_sources': int(coverage.sum())})
            continue
        options[uid] = {'indices': eligible, 'starts': starts, 'ends': ends,
                        'core_starts': core_starts, 'core_ends': core_ends}
    return options, unmatched


def pair_fragments(auxiliary, training, wt_end, selection):
    """Choose sources by cumulative pair weight, without any prediction score."""
    options, unmatched = _matching_options(auxiliary, training, wt_end, selection)
    group_sizes = Counter(auxiliary[uid]['original_domain_cluster50'] for uid in options)
    weights = {uid: len(options) / (len(group_sizes) * group_sizes[auxiliary[uid]['original_domain_cluster50']])
               for uid in options}
    component_weight, protein_weight = Counter(), Counter()
    records = []
    seed = int(selection['seed'])
    for uid in sorted(options, key=lambda value: (stable_seed(seed, value), value)):
        choice = options[uid]
        components = {training[i]['component'] for i in choice['indices']}
        component = min(components, key=lambda value: (component_weight[value], stable_seed(seed, uid, value), value))
        candidates = [i for i in choice['indices'] if training[i]['component'] == component]
        index = min(candidates, key=lambda i: (protein_weight[training[i]['sequence_id']],
                                              stable_seed(seed, uid, training[i]['sequence_id']),
                                              training[i]['sequence_id']))
        source, external = training[index], auxiliary[uid]
        start, end = int(choice['starts'][index]), int(choice['ends'][index])
        core_start, core_end = int(choice['core_starts'][index]), int(choice['core_ends'][index])
        if (start, end) != extend_within_domain(core_start, core_end, len(source['domain']), len(external['sequence'])):
            raise ValueError('Vectorized crop differs from the specified extension rule')
        common = sorted(set(external['mapping']) & set(source['mapping']))
        a_positions = [external['mapping'][wt] for wt in common]
        s_positions = [source['mapping'][wt] - start for wt in common]
        s_sequence = source['domain'][start:end]
        weight = weights[uid]
        component_weight[component] += weight
        protein_weight[source['sequence_id']] += weight
        a_sha, s_sha = _sha(external['sequence']), _sha(s_sequence)
        record = {'pair_id': 'pair:' + a_sha, 'a_sequence_id': uid, 's_sequence_id': 'self:' + s_sha,
                  'a_sequence_sha256': a_sha, 's_sequence_sha256': s_sha,
                  'a_original_group': external['original_domain_cluster50'],
                  'a_relation_group': external.get('group', ''), 'a_original_split': external['original_split'],
                  's_source_sequence_id': source['sequence_id'], 's_component': component,
                  's_source_length': source['full_length'], 's_cdd_start_1based': source['domain_start'] + 1,
                  's_cdd_end_1based': source['domain_end'], 's_domain_start_0based': start, 's_domain_end_0based': end,
                  's_source_start_1based': source['domain_start'] + start + 1,
                  's_source_end_1based': source['domain_start'] + end,
                  's_core_domain_start_0based': core_start, 's_core_domain_end_0based': core_end,
                  's_extended_left': core_start - start, 's_extended_right': end - core_end,
                  'a_length': len(external['sequence']), 's_length': len(s_sequence),
                  'length_difference_s_minus_a': len(s_sequence) - len(external['sequence']),
                  'a_standard_mapped_count': len(external['mapping']), 'common_count': len(common),
                  'common_fraction_of_a_mapping': len(common) / len(external['mapping']),
                  'eligible_s_proteins': len(choice['indices']), 'eligible_s_components': len(components),
                  'weight': weight, 'a_sequence': external['sequence'], 's_sequence': s_sequence,
                  'common_wt_columns': common, 'a_common_positions': a_positions, 's_common_positions': s_positions}
        if any(not 0 <= position < len(s_sequence) for position in s_positions):
            raise ValueError('Common S position outside cropped fragment')
        records.append(record)
    return sorted(records, key=lambda row: row['pair_id']), unmatched


def paired_panels(record, sizes, seed):
    columns = record['common_wt_columns']
    output = {key: record[key] for key in ['common_wt_columns', 'a_common_positions', 's_common_positions']}
    output['panels'] = []
    for size in sizes:
        actual = min(int(size), len(columns) - 1)
        if actual < 1:
            raise ValueError('A panel must have at least one masked and one visible common column')
        rng = np.random.default_rng(stable_seed(seed, record['pair_id'], int(size)))
        indices = sorted(map(int, rng.choice(len(columns), actual, replace=False)))
        output['panels'].append({'panel': f'mask_{size}', 'nominal_size': int(size), 'actual_size': actual,
                                 'wt_columns': [columns[i] for i in indices],
                                 'a_positions': [record['a_common_positions'][i] for i in indices],
                                 's_positions': [record['s_common_positions'][i] for i in indices]})
    return output


def _quantiles(values):
    return dict(zip(['min', 'q25', 'median', 'q75', 'max'], np.quantile(values, [0, .25, .5, .75, 1]).tolist())) if values else {}


def plot_fragment_matching(directory):
    """Describe already selected pairs; do not rerun matching or alter inputs."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    directory = Path(directory)
    manifest = directory / 'pair_manifest.csv'
    summary_path = directory / 'summary.json'
    summary = json.loads(summary_path.read_text())
    if sha256_file(manifest) != summary['result_sha256']['pair_manifest.csv']:
        raise ValueError('Fragment-pair manifest differs from the recorded preparation')
    rows = _rows(manifest)
    if len(rows) != summary['matched_pairs'] or not rows:
        raise ValueError('Incomplete fragment-pair manifest')
    outputs = [directory / ('fragment_matching.' + suffix) for suffix in ('pdf', 'png')]
    identity_path = directory / 'figure_identity.json'
    if any(path.exists() for path in [*outputs, identity_path]):
        raise FileExistsError('Fragment matching figure already exists')
    a = np.array([int(row['a_length']) for row in rows])
    s = np.array([int(row['s_length']) for row in rows])
    fraction = np.array([float(row['common_fraction_of_a_mapping']) for row in rows])
    tolerance = float(summary['selection']['relative_length_tolerance'])
    threshold = float(summary['selection']['min_shared_fraction'])
    with plt.rc_context({'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.9), layout='constrained')
        ax = axes[0]
        extent = float(max(a.max(), s.max())) * 1.04
        x = np.array([0., extent])
        ax.fill_between(x, (1 - tolerance) * x, (1 + tolerance) * x,
                        color='#d9e8f0', label=f'Allowed difference: {100*tolerance:g}% of external length')
        ax.plot(x, x, color='0.35', linewidth=1, linestyle='--')
        ax.scatter(a, s, s=9, alpha=.25, color='#0072b2', edgecolors='none', rasterized=True)
        ax.set(xlim=(0, extent), ylim=(0, extent), xlabel='External fragment length (residues)',
               ylabel='NatV training fragment length (residues)', title='A  Paired fragment lengths')
        ax.text(.04, .95, f'{len(rows):,} pairs', transform=ax.transAxes, va='top')
        ax.legend(loc='lower right', fontsize=7, frameon=False)
        ax = axes[1]
        ax.hist(fraction, bins=np.linspace(threshold, 1, 11), color='#009e73', edgecolor='white')
        ax.set(xlim=(threshold-.01, 1.01), xlabel='Fraction of external mapped columns shared',
               ylabel='Number of fragment pairs', title='B  Common reliable alignment columns')
        ax.set_xticks(np.linspace(threshold, 1, 5))
        fig.savefig(outputs[0], metadata={'Title': 'Matched auxiliary sequence fragments'})
        fig.savefig(outputs[1], dpi=300)
        plt.close(fig)
    identity = {'input_sha256': {str(path): sha256_file(path) for path in [manifest, summary_path]},
                'implementation_sha256': sha256_file(__file__),
                'result_sha256': {path.name: sha256_file(path) for path in outputs},
                'pairs': len(rows), 'self_source_proteins': summary['s_selected_unique_proteins'],
                'note': 'Unweighted descriptive counts of prepared pairs, not independent protein observations.'}
    write_json(identity_path, identity)
    return identity


def build_fragment_pretraining(config_path, output=None):
    config = load_yaml(config_path)
    output = Path(output or config['output'])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output already contains files: {output}')
    selection = config['selection']
    if not 0 < float(selection['min_shared_fraction']) <= 1 or not 0 <= float(selection['relative_length_tolerance']) < 1:
        raise ValueError('Invalid matching thresholds')
    if config['training_design']['auxiliary_validation'] or int(config['training_design']['updates']) != 750:
        raise ValueError('This preparation specifies 750 fixed auxiliary updates without auxiliary validation')
    sizes = config['masking']['sizes']
    if not sizes or len(set(sizes)) != len(sizes) or any(int(size) < 1 for size in sizes):
        raise ValueError('Mask sizes must be distinct positive integers')
    auxiliary, training, wt_end, files = _input_records(config)
    records, unmatched = pair_fragments(auxiliary, training, wt_end, selection)
    if not records:
        raise ValueError('No eligible fragment pairs')
    output.mkdir(parents=True, exist_ok=True)
    hidden = {'a_sequence', 's_sequence', 'common_wt_columns', 'a_common_positions', 's_common_positions'}
    manifest = [{key: value for key, value in record.items() if key not in hidden} for record in records]
    _write_csv(output / 'pair_manifest.csv', manifest, list(manifest[0]))
    _write_csv(output / 'unmatched_auxiliary.csv', unmatched,
               ['aux_uid', 'reason', 'max_shared_fraction', 'coverage_eligible_sources'])
    for arm in ['a', 's']:
        entries = {row[f'{arm}_sequence_id']: {'sequence': row[f'{arm}_sequence']} for row in records}
        _write_fasta(output / f'{arm}_fragments.fasta', entries)
    panels = {row['pair_id']: paired_panels(row, config['masking']['sizes'], config['masking']['seed']) for row in records}
    write_json(output / 'pair_panels.json', panels)
    mapping_rows = [{key: row[key] for key in ['sequence_id', 'component', 'full_length', 'domain_start', 'domain_end',
                                             'alignment_n_optimal', 'alignment_identity']} |
                    {'mapped_standard_columns': len(row['mapping']),
                     'wt_to_domain_positions_json': json.dumps(sorted(row['mapping'].items()))} for row in training]
    _write_csv(output / 'natv_training_mappings.csv', mapping_rows, list(mapping_rows[0]))
    components, proteins, a_groups = Counter(), Counter(), Counter()
    for row in records:
        components[row['s_component']] += row['weight']
        proteins[row['s_source_sequence_id']] += row['weight']
        a_groups[row['a_original_group']] += row['weight']
    source_uses = Counter(row['s_source_sequence_id'] for row in records)
    group_uses = Counter(row['s_component'] for row in records)
    _write_csv(output / 's_source_weights.csv', [
        {'s_source_sequence_id': uid, 's_component': next(row['s_component'] for row in records if row['s_source_sequence_id'] == uid),
         'pairs': source_uses[uid], 'total_pair_weight': weight} for uid, weight in sorted(proteins.items())],
        ['s_source_sequence_id', 's_component', 'pairs', 'total_pair_weight'])
    _write_csv(output / 's_component_weights.csv', [
        {'s_component': component, 'pairs': group_uses[component], 'total_pair_weight': weight}
        for component, weight in sorted(components.items())], ['s_component', 'pairs', 'total_pair_weight'])
    summary = {'status': 'prepared_not_trained', 'input_sha256': {str(path): sha256_file(path) for path in [Path(config_path), *files]},
               'implementation_sha256': {str(path): sha256_file(path) for path in [Path(__file__),
                   *(Path(__file__).with_name(name) for name in ['preprocessing.py', 'fragment_overlap.py', 'data.py'])]},
               'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
               'git_status': subprocess.check_output(['git', 'status', '--porcelain'], text=True).splitlines(),
               'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'biopython': Bio.__version__},
               'selection': selection, 'masking': config['masking'], 'training_design': config['training_design'],
               'external_pool_sequences': len(auxiliary), 'matched_pairs': len(records), 'unmatched_external_sequences': len(unmatched),
               'a_original_groups': len(a_groups), 's_training_proteins_available': len(training),
               's_training_proteins_without_stable_mapping': sum(not row['mapping'] for row in training),
               's_selected_unique_proteins': len(proteins), 's_selected_components': len(components),
               's_maximum_pairs_from_one_protein': max(source_uses.values()),
               'a_unique_fragment_sequences': len({row['a_sequence_sha256'] for row in records}),
               's_unique_fragment_sequences': len({row['s_sequence_sha256'] for row in records}),
               'matching_quantiles': {field: _quantiles([row[field] for row in records]) for field in [
                   'a_length', 's_length', 'length_difference_s_minus_a', 'common_count', 'common_fraction_of_a_mapping',
                   'eligible_s_proteins', 'eligible_s_components']},
               's_extension_quantiles': _quantiles([row['s_extended_left'] + row['s_extended_right'] for row in records]),
               'a_group_total_weight_quantiles': _quantiles(list(a_groups.values())),
               's_component_total_weight_quantiles': _quantiles(list(components.values())),
               'actual_mask_sizes': {str(size): dict(Counter(panel['actual_size'] for item in panels.values()
                                                            for panel in item['panels'] if panel['nominal_size'] == size))
                                     for size in config['masking']['sizes']},
               'matching_definition': 'A remains complete. S spans the common mapped WT columns, extends equally at both ends when possible, then uses remaining NMNAT CDD space if one end reaches a boundary. Absolute length difference divided by A length must not exceed the relative tolerance; there is no additional absolute-residue allowance.',
               'weight_definition': 'Each pair shares weight N_pairs/(N_original_A_groups * pairs_in_original_A_group) in both arms. S selection balances cumulative pair weight across eligible NatV components, then source proteins; S weights are reported and are not renormalized.',
               'limitations': ['Additional S context is from the same training NMNAT domain; non-target residues need not have one-to-one A correspondence.',
                               'Repeated fragments are not independent source proteins. Source-protein and component counts are reported separately.',
                               'Equal updates and batch size do not imply exactly equal FLOPs. No model fitting or test scoring occurs in this preparation.'],
               'result_sha256': {path.name: sha256_file(path) for path in output.iterdir() if path.is_file()}}
    write_json(output / 'summary.json', summary)
    return summary
