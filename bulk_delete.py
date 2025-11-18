#!/usr/bin/env python3
"""
bulk_delete.py

Delete a list of files/folders (and their contents) relative to a parent directory.
Edit PARENT_DIR and TARGETS below. Supports dry-run, logging, and safety checks.

Usage examples:
  python bulk_delete.py                # interactive (asks for confirmation)
  python bulk_delete.py --yes          # perform deletions without prompt
  python bulk_delete.py --dry-run      # show what would be deleted
  python bulk_delete.py --log /tmp/del.log
"""

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

### CONFIGURE HERE ###
# Parent directory - all targets will be interpreted relative to this.
PARENT_DIR = Path("/home/kaushik/Documents/MELO-GAN")

# List of targets relative to PARENT_DIR OR absolute paths.
# Examples:
#   "build/temp"
#   "cache.db"
#   "/absolute/path/if/you/want"
TARGETS = [
    "data/models/ae",
    "data/models/gan",
    "data/splits/test",
    "data/splits/val",
    "data/splits/train",
    "experiments/ae",
    "experiments/gan",
    "train_latent_tsne_visualization.png",
    "val_latent_tsne_visualization.png",
    "generated_tests"
    
    
    # add or remove entries here
]
########################

def resolve_and_check(target, parent_dir):
    """
    Resolve a target path, and ensure it is within parent_dir (for safety).
    Returns resolved_path (Path), allowed (bool), reason (str or None)
    """
    p = Path(target)
    # if relative, make it relative to parent_dir
    if not p.is_absolute():
        p = parent_dir.joinpath(p)
    try:
        resolved = p.resolve(strict=False)
    except Exception as e:
        # fallback to absolute without resolving symlinks if resolve fails
        resolved = p.absolute()
    # safety: ensure resolved is inside parent_dir (unless user overrides with --force)
    try:
        parent_resolved = parent_dir.resolve(strict=False)
    except Exception:
        parent_resolved = parent_dir.absolute()
    try:
        resolved_relative = resolved.relative_to(parent_resolved)
        return resolved, True, None
    except Exception:
        return resolved, False, f"Resolved path {resolved} is outside parent dir {parent_resolved}"

def remove_path(path: Path):
    """Remove file or directory at path. Raises exceptions on failure."""
    if not path.exists() and not path.is_symlink():
        raise FileNotFoundError(f"{path} does not exist")
    if path.is_dir() and not path.is_symlink():
        # use shutil.rmtree for directories
        shutil.rmtree(path)
    else:
        # files or symlinks
        path.unlink()

def main():
    ap = argparse.ArgumentParser(description="Bulk delete specified files/folders (listed inside script).")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting.")
    ap.add_argument("--yes", "-y", action="store_true", help="Perform deletions without confirmation.")
    ap.add_argument("--log", type=str, default=None, help="Write a deletion log to this file.")
    ap.add_argument("--force", action="store_true", help="Allow deletion of targets outside parent directory (dangerous).")
    ap.add_argument("--parent", type=str, default=None, help="Override PARENT_DIR from command line.")
    args = ap.parse_args()

    parent_dir = Path(args.parent) if args.parent else PARENT_DIR
    log_lines = []
    timestamp = datetime.utcnow().isoformat() + "Z"
    header = f"bulk_delete run at {timestamp} parent_dir={parent_dir}"

    print(header)
    log_lines.append(header)

    to_delete = []
    skipped = []

    for t in TARGETS:
        resolved, allowed, reason = resolve_and_check(t, parent_dir)
        if allowed or args.force:
            to_delete.append((t, resolved))
        else:
            skipped.append((t, resolved, reason))

    if skipped:
        print("\nSkipped (safety) — these targets are outside the parent dir:")
        for orig, resolved, reason in skipped:
            print(f"  - {orig} -> {resolved}  (reason: {reason})")
            log_lines.append(f"SKIP: {orig} -> {resolved}  ({reason})")

    if not to_delete:
        print("\nNothing to delete (no allowed targets).")
        if args.log:
            Path(args.log).write_text("\n".join(log_lines))
        return 0

    print("\nPlanned deletions:")
    for orig, resolved in to_delete:
        exists = resolved.exists() or resolved.is_symlink()
        typ = "dir" if resolved.is_dir() and not resolved.is_symlink() else "file/symlink"
        print(f"  - {orig} -> {resolved}  (exists={exists}, type={typ})")
        log_lines.append(f"PLAN: {orig} -> {resolved}  (exists={exists}, type={typ})")

    if args.dry_run:
        print("\nDry-run mode: no changes made.")
        if args.log:
            Path(args.log).write_text("\n".join(log_lines))
        return 0

    if not args.yes:
        resp = input("\nProceed with deletion of the planned items? Type 'YES' to confirm: ")
        if resp != "YES":
            print("Aborted by user (confirmation failed). No changes made.")
            log_lines.append("ABORTED_BY_USER")
            if args.log:
                Path(args.log).write_text("\n".join(log_lines))
            return 0

    # perform deletions
    successes = []
    failures = []
    for orig, resolved in to_delete:
        try:
            if not (resolved.exists() or resolved.is_symlink()):
                log_lines.append(f"NOT_FOUND: {orig} -> {resolved}")
                print(f"[NOT_FOUND] {resolved}")
                continue
            remove_path(resolved)
            print(f"[DELETED] {resolved}")
            log_lines.append(f"DELETED: {orig} -> {resolved}")
            successes.append(resolved)
        except Exception as e:
            print(f"[ERROR] {resolved} -> {e}")
            log_lines.append(f"ERROR: {orig} -> {resolved}  ({e})")
            failures.append((resolved, str(e)))

    # summary
    print("\nSummary:")
    print(f"  Deleted: {len(successes)}")
    print(f"  Failures: {len(failures)}")
    print(f"  Skipped (safety): {len(skipped)}")
    log_lines.append(f"SUMMARY: deleted={len(successes)} failures={len(failures)} skipped={len(skipped)}")

    if args.log:
        try:
            with open(args.log, "a") as f:
                f.write("\n".join(log_lines) + "\n")
            print(f"\nLog appended to {args.log}")
        except Exception as e:
            print(f"Failed to write log to {args.log}: {e}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
