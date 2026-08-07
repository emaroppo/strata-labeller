#!/usr/bin/env python3
"""Convert a project's v1 datasets to the v2 schema format.

v1 stored classification labels directly:

    [{"path": "a.jpg", "labels": ["cat"], "skipped": false}]

v2 stores canonicalized Label Studio results, so any task type fits, plus
an explicit ``annotated`` flag — an annotation with no content (an image
with no boxes) is a real answer, not an absent one:

    {"version": 2, "samples": [
        {"path": "a.jpg", "annotated": true, "results": [
            {"from_name": "label", "to_name": "image", "type": "choices",
             "value": {"choices": ["cat"]}}]}]}

The control names come from the project's schema, so a custom labeling
config is honoured rather than assumed.

Reading v1 needs no migration — the loader upgrades on the way in. Run this
to convert the files on disk, including the round snapshots, so the format
is uniform.

Usage:
    python scripts/migrate_dataset_v2.py -p <project> --dry-run
    python scripts/migrate_dataset_v2.py -p <project>
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auto_labeller.dataset import is_v1, load_dataset, save_dataset  # noqa: E402
from auto_labeller.project import Project, ProjectError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--project", type=Path, default=None,
                        help="Project name or path (default: the usual resolution)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip the .v1.bak copies (not recommended)")
    args = parser.parse_args()

    try:
        project = Project.load(args.project)
        schema = project.schema
    except ProjectError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    targets = [project.dataset_path]
    targets += sorted(project.rounds_dir.glob("round_*/labeled.json"))

    converted = skipped = 0
    for path in targets:
        if not path.exists():
            print(f"skip     {path.name} (missing)")
            continue
        if not is_v1(path):
            print(f"skip     {path.relative_to(project.root)} (already v2)")
            skipped += 1
            continue

        samples = load_dataset(path, schema)
        annotated = sum(1 for s in samples if s.annotated)
        marker = f"{len(samples)} samples, {annotated} annotated"
        print(f"convert  {path.relative_to(project.root)} ({marker})")
        if args.dry_run:
            converted += 1
            continue

        if not args.no_backup:
            shutil.copy2(path, path.with_suffix(".v1.bak"))
        save_dataset(samples, path)
        # Reload to prove the file we just wrote reads back identically
        check = load_dataset(path)
        assert [(s.path, s.results, s.annotated, s.skipped) for s in check] == [
            (s.path, s.results, s.annotated, s.skipped) for s in samples
        ], f"round trip failed for {path}"
        converted += 1

    verb = "would convert" if args.dry_run else "converted"
    print(f"\n{verb} {converted} file(s); {skipped} already v2")
    if converted and not args.dry_run and not args.no_backup:
        print("v1 copies kept alongside as *.v1.bak")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
