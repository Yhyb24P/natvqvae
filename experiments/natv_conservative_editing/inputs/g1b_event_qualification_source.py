"""Qualification of the 161 candidate InDel event definitions, read-only.

The contract says these definitions are pre-screen evidence: 161 of them keep at least two shared homology
components after the scoring scheme changes, which is not the same as 161 qualified edit sites. This pass
adds the three things the screening step never measured - event length, how reliable the anchors are, and
whether the site has any structural correspondence on KVP40 - and then reports what candidate set, if any,
survives. No fit, no generation, no folding, nothing off-host.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd
import gemmi
from Bio import Align
from Bio.Align import substitution_matrices

from nad_qvae.design.chem_position_plan_v1 import (align_positions, load_plan, load_wt,
                                                   pocket_neighbourhood)
from nad_qvae.design.vnext_edits import edit_script, indel_events
from nad_qvae.paths import data_root

ROOT = Path(__file__).resolve().parents[3]
MSA = ROOT / 'results/natv_g0_local_family_msa_v1/pool.aln.fasta'
CIF_DIRECTORY = 'experiments/run_reports/natv_structure_source_resolution_v2/experimental_candidates'
OUT = ROOT / 'results/natv_g1_event_qualification_v1'
AA = 'ACDEFGHIKLMNPQRSTVWY'
# declared perturbation: the same alignment family, a different matrix and heavier gaps
PERTURBATION = {'matrix': 'BLOSUM45', 'open_gap': -12.0, 'extend_gap': -0.8}
DECLARED = {'matrix': 'BLOSUM62', 'open_gap': -10.0, 'extend_gap': -0.5}


def aligned_events(reference, sequence, scheme):
    """Reference-relative insertion/deletion events, slid to their leftmost equivalent placement."""
    if scheme is DECLARED:
        operations = edit_script(reference, sequence)
    else:
        aligner = Align.PairwiseAligner()
        aligner.mode = 'global'
        aligner.substitution_matrix = substitution_matrices.load(scheme['matrix'])
        aligner.open_gap_score, aligner.extend_gap_score = scheme['open_gap'], scheme['extend_gap']
        indices = aligner.align(reference, sequence)[0].indices
        operations, output, last_ref = [], 0, -1
        for wi, si in indices.T:
            if wi < 0:
                anchor, side = max(1, last_ref + 1), ('BEFORE' if last_ref < 0 else 'AFTER')
                if (operations and operations[-1]['op_type'] == 'INSERT'
                        and operations[-1]['wt_anchor'] == anchor
                        and operations[-1]['insertion_side'] == side):
                    operations[-1]['target_seq'] += sequence[si]
                else:
                    operations.append({'wt_anchor': anchor, 'op_type': 'INSERT', 'source_aa': '',
                                       'target_seq': sequence[si], 'insertion_side': side,
                                       'current_interval': [output, output + 1]})
                output += 1
            else:
                last_ref = int(wi)
                target = sequence[si] if si >= 0 else ''
                kind = 'DELETE' if si < 0 else 'KEEP' if reference[wi] == target else 'SUB'
                operations.append({'wt_anchor': int(wi) + 1, 'op_type': kind, 'source_aa': reference[wi],
                                   'target_seq': target, 'insertion_side': None,
                                   'current_interval': [output, output + int(si >= 0)]})
                output += int(si >= 0)
    events = []
    for event in indel_events(operations):
        anchor, string = int(event['wt_anchor']), event['sequence']
        while string and anchor > (1 if event['type'] == 'INSERT' else 2) \
                and string[-1] == reference[anchor - (1 if event['type'] == 'INSERT' else 2)]:
            string = string[-1] + string[:-1] if event['type'] == 'INSERT' else reference[anchor - 2] + string[:-1]
            anchor -= 1
        events.append({'type': event['type'], 'anchor': anchor, 'length': len(string), 'context': string})
    return events


def msa_frame():
    """The target-anchored column frame plus per-column pool statistics used for anchor reliability."""
    rows = [block.split('\n') for block in MSA.read_text().split('>')[1:]]
    sequences = {head.strip(): ''.join(body).strip() for head, *body in rows}
    target = np.array(list(sequences['TARGET']))
    column_of, position = {}, 0
    for index, letter in enumerate(target):
        if letter != '-':
            position += 1
            column_of[position] = index
    body = np.array([list(seq) for name, seq in sequences.items() if name != 'TARGET'])
    non_gap = (body != '-').mean(0)
    entropy = np.zeros(body.shape[1])
    for column in range(body.shape[1]):
        letters = [row[column] for row in body if row[column] != '-']
        if len(letters) > 1:
            counts = np.array([letters.count(letter) for letter in set(letters)], float) / len(letters)
            entropy[column] = float(-(counts * np.log(counts)).sum() / np.log(20))
    return column_of, non_gap, entropy, len(body)


def secondary_structure(pdb, chain):
    path = data_root() / CIF_DIRECTORY / f'{pdb}.cif'
    if not path.is_file():
        return []
    block = gemmi.cif.read(str(path)).sole_block()

    def column(tag):
        return [str(value) for value in block.find_values(f'_struct_conf.{tag}')]
    start, end, kind = column('beg_auth_seq_id'), column('end_auth_seq_id'), column('conf_type_id')
    begin_chain, end_chain = column('beg_auth_asym_id'), column('end_auth_asym_id')
    return [(int(start[i]), int(end[i]), kind[i]) for i in range(len(start))
            if begin_chain[i] == chain == end_chain[i]]


def run(root: Path = ROOT) -> dict:
    plan = load_plan()
    wt, _ = load_wt()
    release = data_root(root) / 'data/processed/releases/natv_vnext_family_v2'
    train = pd.read_parquet(release / 'family_training_roles_v2.parquet')
    train = train[train['training_role_v2'] == 'training']
    column_of, column_non_gap, column_entropy, pool_size = msa_frame()
    pockets = {position: pocket for pocket in plan['primary_pockets']
               for position in pocket_neighbourhood(pocket, plan)}
    ss_elements = {'2QJO_B': secondary_structure('2QJO', 'B'), '2QJT_A': secondary_structure('2QJT', 'A')}

    support = collections.defaultdict(lambda: {'components': set(), 'sequences': []})
    perturbed_support = collections.defaultdict(set)
    for row in train.itertuples():
        for scheme, sink in ((DECLARED, support), (PERTURBATION, perturbed_support)):
            for event in aligned_events(wt, row.sequence, scheme):
                if event['anchor'] <= 2 or event['anchor'] >= len(wt) - 1:
                    continue
                key = (event['type'], event['anchor'], event['length'])
                if sink is support:
                    sink[key]['components'].add(row.component)
                    sink[key]['sequences'].append(row.sequence_id)
                else:
                    sink[key].add(row.component)

    rows = []
    for key, record in support.items():
        event_type, anchor, length = key
        if len(record['components']) < 2:
            continue
        shared = record['components'] & perturbed_support.get(key, set())
        low_column = column_of.get(anchor)
        high_column = column_of.get(min(len(wt), anchor + max(1, length)))
        if low_column is None or high_column is None:
            anchor_quality = 'UNMAPPED_TO_MSA_COLUMN'
            flank_gap_free = None
        else:
            left = column_non_gap[max(0, low_column - 3):low_column]
            right = column_non_gap[high_column + 1:high_column + 4]
            flank_gap_free = float(np.mean(np.concatenate([left, right]))) if len(left) + len(right) else None
            anchor_quality = ('ANCHORS_CONSERVED' if flank_gap_free is not None and flank_gap_free >= 0.9 else
                              'ANCHORS_VARIABLE' if flank_gap_free is not None else 'NO_FLANK_COLUMNS')
        rows.append({'type': event_type, 'anchor_wt_1based': anchor, 'length': length,
                     'supporting_sequences': len(record['sequences']),
                     'components_supporting': len(record['components']),
                     'components_shared_with_perturbation': len(shared),
                     'stable_across_scoring': bool(len(shared) >= 2),
                     'in_g0_pocket_neighbourhood': anchor in pockets,
                     'pocket': pockets.get(anchor),
                     'anchor_flank_gap_free_fraction': flank_gap_free,
                     'anchor_quality': anchor_quality,
                     'region': next((family for family, window in plan['domain_windows_wt'].items()
                                     if window[0] <= anchor <= window[1]), 'interdomain_or_termini'),
                     'secondary_structure_of_2QJO_at_anchor':
                         [element for element in ss_elements['2QJO_B'] if element[0] <= anchor <= element[1]]
                         or None})

    for row in rows:
        elements = row.pop('secondary_structure_of_2QJO_at_anchor', None) or []
        row['ss_class_2QJO'] = (elements[0][2] if elements else 'NO_ELEMENT_AT_ANCHOR')

    stable = [row for row in rows if row['stable_across_scoring']]

    def histogram(field):
        return dict(sorted(collections.Counter(row[field] for row in stable).items(),
                           key=lambda item: str(item[0])))

    qualified = [row for row in stable
                 if row['anchor_quality'] == 'ANCHORS_CONSERVED' and row['length'] <= 6
                 and row['components_supporting'] >= 3]
    document = {
        'schema': 'natv-g1-event-qualification-v1', 'date': '2026-10-18',
        'authorization': 'read-only event qualification authorized by the owner 2026-10-18; generation '
                         'and folding flags stay false',
        'population': {'core_train_sequences': int(len(train)),
                       'components': int(train['component'].nunique()),
                       'msa_pool_sequences': pool_size,
                       'declared_scheme': DECLARED, 'perturbation_scheme': PERTURBATION},
        'ladder': {'multi_component_definitions': len(rows), 'stable_across_scoring': len(stable),
                   'anchor_quality': histogram('anchor_quality'), 'lengths': histogram('length'),
                   'regions': histogram('region'),
                   'secondary_structure_class_of_the_homolog_at_anchor': histogram('ss_class_2QJO'),
                   'anchor_quality_cross_ss': {f'{a} | {b}': n for (a, b), n in collections.Counter(
                       (row['anchor_quality'], row['ss_class_2QJO']) for row in stable).items()},
                   'anchor_quality_cross_short': {f'{a} | length<=6={b}': n for (a, b), n in collections.Counter(
                       (row['anchor_quality'], row['length'] <= 6) for row in stable).items()},
                   'in_pocket_cross_anchor_quality': {f'in_pocket={a} | {b}': n for (a, b), n in collections.Counter(
                       (row['in_g0_pocket_neighbourhood'], row['anchor_quality']) for row in stable).items()},
                   'inside_a_frozen_g0_pocket_neighbourhood':
                       int(sum(1 for row in stable if row['in_g0_pocket_neighbourhood']))},
        'qualified_candidates': {'rule': 'stable across scoring AND conserved flanking columns AND length '
                                         '<= 6 residues AND at least three supporting components',
                                 'count': len(qualified), 'rows': sorted(
                                     qualified, key=lambda row: (-row['components_supporting'],
                                                                 row['anchor_wt_1based']))[:40]},
        'secondary_structure_sources': {key: len(value) for key, value in ss_elements.items()},
        'reading_rules': {'real localized insertion/deletion material': 'a non-empty qualified set inside '
                                     'conserved anchors',
                          'domain-call or alignment residue': 'stable definitions concentrate in variable '
                                     'anchors, long lengths, or the inter-domain region',
                          'insufficient evidence': 'an empty qualified set with most definitions unresolved '
                                     'on anchors'},
        'evidence_bound': 'component support and scoring stability are comparison properties of the frozen '
                          'alignments, not independent evolutionary events; nothing here is an InDel rate '
                          'and nothing here licenses generation'}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'event_qualification.json').write_text(json.dumps(document, indent=1, sort_keys=True, default=str) + '\n')
    return document


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    document = run()
    print(json.dumps({'ladder': document['ladder'],
                      'qualified': document['qualified_candidates']['count'],
                      'top': document['qualified_candidates']['rows'][:6]}, indent=1, sort_keys=True, default=str))
