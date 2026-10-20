"""Create read-only symlinks to hash-verified source inputs.

The source repository is never modified. Every entry is verified before a link is
created; a single mismatch aborts without linking anything.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from natvqvae.intake import load_manifest, source_root, verify_all  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None, help="source repository root (overrides manifest/env)")
    ap.add_argument("--check-only", action="store_true", help="verify hashes, create no links")
    ap.add_argument("--force", action="store_true", help="replace existing links that point elsewhere")
    args = ap.parse_args(argv)

    manifest = load_manifest()
    root = Path(args.root) if args.root else source_root(manifest)
    results = verify_all(manifest, root)

    failures = [r for r in results if not r.ok]
    for r in results:
        status = "OK  " if r.ok else "FAIL"
        print(f"[{status}] {r.name:22} {r.detail}")
    if failures:
        print(f"\n{len(failures)} input(s) failed verification; no links were created.", file=sys.stderr)
        return 1
    if args.check_only:
        print(f"\nAll {len(results)} inputs verified (check-only, nothing created).")
        return 0

    data_dir = REPO_ROOT / "inputs" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for entry in manifest["entries"]:
        target = (root / entry["path"]).resolve()
        link = data_dir / entry["link"]
        if link.is_symlink() and link.resolve() == target:
            continue
        if link.exists() or link.is_symlink():
            if not args.force:
                print(f"refusing to replace {link} (use --force)", file=sys.stderr)
                return 1
            link.unlink()
        os.symlink(target, link)
        print(f"linked {entry['link']} -> {target}")
    print(f"\n{len(manifest['entries'])} inputs linked under {data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
