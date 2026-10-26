"""Export existing NatV predictions to viewable PDBs without moving atoms."""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import gemmi

from .data import sha256_file


def _read_protein(path):
    structure = gemmi.read_structure(str(path))
    if len(structure) != 1 or len(structure[0]) != 1:
        raise ValueError(f'Expected one model and one chain: {path}')
    chain = structure[0][0]
    if len(chain) != 341:
        raise ValueError(f'Expected 341 protein residues: {path}')
    for position, residue in enumerate(chain, 1):
        if (residue.seqid.num != position or residue.seqid.icode.strip()
                or not gemmi.find_tabulated_residue(residue.name).is_amino_acid()):
            raise ValueError(f'Expected protein residues numbered 1–341: {path}')
        if sum(atom.name == 'CA' for atom in residue) != 1:
            raise ValueError(f'Expected one C-alpha at residue {position}: {path}')
    sequence = gemmi.one_letter_code([residue.name for residue in chain])
    return sequence, structure


def _atoms(structure):
    atoms = {}
    chain = structure[0][0]
    for residue in chain:
        for atom in residue:
            key = (chain.name, residue.seqid.num, residue.seqid.icode,
                   residue.name, atom.name, atom.altloc, atom.element.name)
            if key in atoms:
                raise ValueError(f'Duplicate atom identity: {key}')
            atoms[key] = atom
    return atoms


def _check_roundtrip(source_structure, written_structure):
    source, written = _atoms(source_structure), _atoms(written_structure)
    if not source or source.keys() != written.keys():
        raise ValueError('PDB conversion changed atom identities or atom count')
    maximum = 0.0
    for key, atom in source.items():
        other = written[key]
        distance = atom.pos.dist(other.pos)
        if not math.isfinite(distance) or distance > 0.00087:
            raise ValueError(f'PDB conversion moved atom beyond 0.001 A rounding: {key}')
        # PDB rounds occupancy and temperature factors to two decimal places.
        if (not math.isclose(atom.occ, other.occ, abs_tol=0.0051)
                or not math.isclose(atom.b_iso, other.b_iso, abs_tol=0.0051)):
            raise ValueError(f'PDB conversion changed occupancy or B-factor: {key}')
        maximum = max(maximum, distance)
    return len(source), maximum


def convert_coordinate(candidate, model_index, source, batch, output,
                       expected_sequence, directory):
    """Convert one archived prediction; verify sequence, atom identity and rounding.

    ``model_index`` identifies a diffusion sample in its original batch, not a
    PDB MODEL number. No fitting, relaxation, atom selection or prediction runs.
    """
    if not re.fullmatch(r'[A-Za-z0-9_]+', candidate):
        raise ValueError(f'Invalid candidate identifier: {candidate!r}')
    if model_index not in (0, 1, 2):
        raise ValueError('Expected diffusion sample index 0, 1 or 2')
    if directory not in ('coordinates', 'references'):
        raise ValueError('Coordinates belong in coordinates/ or references/')
    if not re.fullmatch(r'[ACDEFGHIKLMNPQRSTVWY]{341}', expected_sequence):
        raise ValueError('Expected a complete 341 aa protein sequence')
    source, output = Path(source), Path(output)
    if not source.is_file():
        raise FileNotFoundError(source)
    sequence, structure = _read_protein(source)
    if sequence != expected_sequence:
        raise ValueError(f'{candidate} model {model_index} source sequence differs from unified CSV')
    destination = output / directory / f'{candidate}_model_{model_index}.pdb'
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    structure.setup_entities()
    # Exclusive creation protects an existing result even if another writer races.
    with destination.open('x') as handle:
        handle.write(structure.make_pdb_string())
    try:
        written_sequence, written_structure = _read_protein(destination)
        if written_sequence != expected_sequence:
            raise ValueError(f'{candidate} model {model_index} PDB sequence differs from unified CSV')
        atom_count, maximum = _check_roundtrip(structure, written_structure)
    except Exception:
        destination.unlink()
        raise
    return {
        'candidate': candidate,
        'model_index': model_index,
        'batch': batch,
        'package': destination.relative_to(output).as_posix(),
        'sha256': sha256_file(destination),
        'sequence_sha256': hashlib.sha256(written_sequence.encode()).hexdigest(),
        'source_cif_gz': str(source),
        'source_sha256': sha256_file(source),
        'sequence_matches_unified_csv': True,
        'atom_count': atom_count,
        'maximum_coordinate_rounding_A': maximum,
        'conversion': 'CIF to PDB; original frame and all atoms retained; no fitting or relaxation',
    }
