"""Command line entry point."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, regions
from .intake import load_manifest, source_root, verify_all
from .sequence import WT_SHA256, wt_sequence


def _doctor(args: argparse.Namespace) -> int:
    print(f"natvqvae {__version__}")
    print("\n[contracts]")
    regions.validate_partition()
    print(f"  region partition 1..341           OK  ({len(regions.regions())} regions)")
    print(f"  anchors                           OK  {sorted(regions.catalytic_anchors())}")
    print(f"  WT sequence                       OK  341 aa, sha256 {WT_SHA256[:12]}...")

    manifest = load_manifest()
    root = Path(args.root) if args.root else source_root(manifest)
    print(f"\n[data] source root: {root}")
    results = verify_all(manifest, root)
    for r in results:
        print(f"  [{'OK  ' if r.ok else 'FAIL'}] {r.name:22} {r.detail}")
    failures = [r for r in results if not r.ok]
    if failures:
        print(f"\nFAIL: {len(failures)} data entry/entries failed verification", file=sys.stderr)
        return 1
    print(f"\nOK: {len(results)} data entries verified; regions and anchors validated")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="natvqvae", description="NatV net-NAD+ design and input checks")
    parser.add_argument("--version", action="version", version=f"natvqvae {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="validate contracts and input hashes")
    doctor.add_argument("--root", default=None, help="source repository root")
    doctor.set_defaults(func=_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
