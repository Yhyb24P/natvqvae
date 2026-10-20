"""WT-relative conditional edits, with explicit coordinate and distance accounting."""
from __future__ import annotations
from dataclasses import dataclass
import math
from pathlib import Path
import sys

import numpy as np
from Bio import Align
from Bio.Align import substitution_matrices

from .sequence import AMINO_ACIDS, wt_sequence

AA='ACDEFGHIKLMNPQRSTVWY'


def edit_distance(first: str, second: str) -> int:
    """Unit-cost global Levenshtein distance, including every terminal residue."""
    previous=list(range(len(second)+1))
    for i,a in enumerate(first,1):
        current=[i]
        for j,b in enumerate(second,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(a!=b)))
        previous=current
    return previous[-1]


def wt_identity(sequence: str) -> float:
    return 1-edit_distance(sequence,wt_sequence())/max(len(sequence),len(wt_sequence()))


@dataclass(frozen=True)
class EditScaffold:
    sequence: str
    wt_to_current: tuple[int|None,...]
    event: dict|None


def scaffold(event: dict|None=None) -> EditScaffold:
    """Apply one sourced event; insertion anchors follow their preceding WT residue."""
    wt=wt_sequence();mapping=list(range(len(wt)))
    if event is None:return EditScaffold(wt,tuple(mapping),None)
    anchor=int(event['anchor']);length=int(event['length']);kind=event['type']
    if not 1<=length<=3 or not 1<=anchor<=len(wt):raise ValueError('Event outside pilot range')
    if kind=='INSERT':
        string=event['sequence']
        if len(string)!=length or not set(string)<=AMINO_ACIDS:raise ValueError('Invalid inserted residues')
        sequence=wt[:anchor]+string+wt[anchor:]
        mapping=[i if i<anchor else i+length for i in mapping]
    elif kind=='DELETE':
        low=anchor-1;high=low+length
        if high>len(wt):raise ValueError('Deletion outside reference')
        if event['sequence']!=wt[low:high]:raise ValueError('Deletion reference mismatch')
        sequence=wt[:low]+wt[high:]
        mapping=[i if i<low else None if i<high else i-length for i in mapping]
    else:raise ValueError('Only INSERT and DELETE are scaffold events')
    return EditScaffold(sequence,tuple(mapping),event)


def current_positions(base: EditScaffold, wt_positions):
    positions=[base.wt_to_current[p] for p in wt_positions]
    if any(p is None for p in positions):raise ValueError('A deleted WT position was selected')
    if len(set(positions))!=len(positions):raise ValueError('Duplicate current coordinate')
    return list(map(int,positions))


def constrained_letters(logits, reference_letters, uniforms, temperature=1.):
    """Draw alternatives without rejection; all selected WT sites remain substitutions."""
    scores=np.asarray(logits,dtype=np.float64)/temperature
    if scores.shape!=(len(reference_letters),20) or not np.isfinite(scores).all():raise ValueError('Invalid logits')
    if len(uniforms)!=len(scores) or not all(0<=u<1 for u in uniforms):raise ValueError('Invalid random draws')
    for i,letter in enumerate(reference_letters):scores[i,AA.index(letter)]=-np.inf
    probabilities=np.exp(scores-np.max(scores,axis=1,keepdims=True));probabilities/=probabilities.sum(1,keepdims=True)
    cdf=probabilities.cumsum(1);cdf[:,-1]=1.
    indices=(np.asarray(uniforms)[:,None]>=cdf).sum(1)
    return ''.join(AA[int(i)] for i in indices)


def normalized_events(reference, sequence, scheme=None):
    """G1-b event convention: slide equivalent repeat events to the left.

    Alignment and normalization follow the pinned G1-b source snapshot. This
    reports alignment support, not experimentally independent evolutionary events.
    """
    aligner=Align.PairwiseAligner();aligner.mode='global'
    scheme=scheme or {'matrix':'BLOSUM62','open_gap':-10.,'extend_gap':-.5}
    aligner.substitution_matrix=substitution_matrices.load(scheme['matrix'])
    aligner.open_gap_score=scheme['open_gap'];aligner.extend_gap_score=scheme['extend_gap']
    indices=aligner.align(reference,sequence)[0].indices
    raw=[];last=-1
    for wi,si in indices.T:
        if wi<0:
            anchor=max(1,last+1);kind='INSERT';letter=sequence[si]
            if raw and raw[-1]['type']==kind and raw[-1]['anchor']==anchor:raw[-1]['sequence']+=letter
            else:raw.append({'type':kind,'anchor':anchor,'sequence':letter})
        else:
            last=int(wi)
            if si<0:
                if raw and raw[-1]['type']=='DELETE' and raw[-1]['anchor']+len(raw[-1]['sequence'])==int(wi)+1:
                    raw[-1]['sequence']+=reference[wi]
                else:raw.append({'type':'DELETE','anchor':int(wi)+1,'sequence':reference[wi]})
            else:raw.append({'type':'ALIGNED','anchor':int(wi)+1,'sequence':''})
    events=[]
    for event in raw:
        if event['type']=='ALIGNED':continue
        anchor,string=event['anchor'],event['sequence']
        while string and anchor>(1 if event['type']=='INSERT' else 2) and string[-1]==reference[anchor-(1 if event['type']=='INSERT' else 2)]:
            string=string[-1]+string[:-1] if event['type']=='INSERT' else reference[anchor-2]+string[:-1]
            anchor-=1
        events.append({'type':event['type'],'anchor':anchor,'length':len(string),'sequence':string})
    return events


def legacy_scripts(source: Path):
    """Use the existing lossless KEEP/SUB/INSERT/DELETE script implementation."""
    previous=sys.dont_write_bytecode;sys.dont_write_bytecode=True;sys.path.insert(0,str(source/'src'))
    try:
        from nad_qvae.design.vnext_edits import edit_script,apply_script
        return edit_script,apply_script
    finally:sys.path.pop(0);sys.dont_write_bytecode=previous
