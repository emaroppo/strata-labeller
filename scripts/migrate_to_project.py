#!/usr/bin/env python3
"""Migrate a pre-project layout into a project directory.

The old layout kept paths in config.toml (data/dataset.json, data/raw,
models/, data/rounds/) with sample paths relative to the repository root.
This gathers all of it into projects/<name>/:

    data/dataset.json    -> projects/<name>/dataset.json  (paths made data-root relative)
    data/raw/            -> projects/<name>/data/raw/
    models/              -> projects/<name>/checkpoints/
    data/rounds/         -> projects/<name>/rounds/
    data/task_map_*.json -> projects/<name>/.state/
    (new)                -> projects/<name>/project.toml

Moves within one filesystem are renames, so even a large image tree moves
instantly. The Label Studio image URLs are unchanged by this migration, so
an existing LS project keeps working — restart the container with
AUTO_LABELLER_PROJECT set to the new project root so the mount follows.

Usage:
    python scripts/migrate_to_project.py --name my-project --dry-run
    python scripts/migrate_to_project.py --name my-project
"""

import argparse
import json
import shutil
import sys
import tomllib
from pathlib import Path

MODEL_CLASS_MAP = {
    "MyModel": "MultiLabelClassifier",
    "MyMulticlassModel": "MulticlassClassifier",
    "MyPresenceModel": "PresenceClassifier",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--name", default=None,
                        help="Project name (default: the current directory's name)")
    parser.add_argument("--projects-dir", type=Path, default=Path("projects"),
                        help="Where projects live (default: projects/)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"error: {args.config} not found", file=sys.stderr)
        return 1

    with open(args.config, "rb") as f:
        old = tomllib.load(f)
    paths = old.get("paths", {})
    dataset_path = Path(paths.get("dataset", "data/dataset.json"))
    images_dir = Path(paths.get("images_dir", "data/raw"))
    checkpoints_dir = Path(paths.get("checkpoints_dir", "models"))
    rounds_dir = Path(paths.get("rounds_dir", "data/rounds"))

    name = args.name or Path.cwd().name
    root: Path = args.projects_dir / name
    print(f"project root: {root}")
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)
    old_class = old.get("model", {}).get("class_name", "MyModel")
    model_ref = (
        f"auto_labeller.models.classifier:{MODEL_CLASS_MAP.get(old_class, old_class)}"
    )

    # Sample paths were repo-root relative; they become data-root relative
    strip = f"{images_dir.as_posix()}/"

    def rewrite(json_path: Path, target: Path) -> None:
        if not json_path.exists():
            print(f"skip   {json_path} (missing)")
            return
        with open(json_path) as f:
            data = json.load(f)
        missed = 0
        for item in data:
            if item["path"].startswith(strip):
                item["path"] = item["path"][len(strip):]
            else:
                missed += 1
        note = f" [{missed} paths not under {images_dir}]" if missed else ""
        print(f"rewrite {json_path} -> {target} ({len(data)} samples){note}")
        if not args.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w") as f:
                json.dump(data, f, indent=2)
            if json_path.resolve() != target.resolve():
                json_path.unlink()

    def move(src: Path, dst: Path) -> None:
        if not src.exists():
            print(f"skip   {src} (missing)")
            return
        if dst.exists():
            print(f"skip   {src} -> {dst} (target exists)")
            return
        print(f"move   {src} -> {dst}")
        if not args.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))

    # Classes come from the labels actually used, since the old config had none
    classes: list[str] = []
    if dataset_path.exists():
        with open(dataset_path) as f:
            classes = sorted({c for item in json.load(f) for c in item.get("labels", [])})

    project_toml = root / "project.toml"
    content = (
        f'name = "{name}"\n'
        "\n"
        "[label_config]\n"
        'template = "image_classification"\n'
        f"classes = [{', '.join(f'\"{c}\"' for c in classes)}]\n"
        'choice = "multiple"\n'
        "\n"
        "[data]\n"
        'root = "data/raw"\n'
        "\n"
        "[model]\n"
        f'ref = "{model_ref}"\n'
        "\n"
        "[model.params]\n"
        "num_epochs = 4\n"
        "batch_size = 16\n"
        "lr = 5e-5\n"
        "\n"
        "[label_studio]\n"
        "# project_id = <your existing Label Studio project id>\n"
        'local_files_root = "data"\n'
        'local_files_prefix = "images"\n'
    )
    if project_toml.exists():
        print(f"skip   {project_toml} (exists)")
    else:
        print(f"write  {project_toml} (classes: {', '.join(classes) or 'none found'})")
        if not args.dry_run:
            project_toml.write_text(content)

    rewrite(dataset_path, root / "dataset.json")
    move(images_dir, root / "data" / "raw")
    move(checkpoints_dir, root / "checkpoints")
    move(rounds_dir, root / "rounds")
    # In a dry run nothing moved, so the rounds are still at their old path
    moved_rounds = root / "rounds"
    for round_labeled in sorted(
        (moved_rounds if moved_rounds.exists() else rounds_dir).glob("round_*/labeled.json")
    ):
        rewrite(round_labeled, round_labeled)
    for cache in sorted(dataset_path.parent.glob("task_map_*.json")):
        move(cache, root / ".state" / cache.name)

    # The old data/ directory is left behind empty once everything has moved
    old_data = dataset_path.parent
    if not args.dry_run and old_data.is_dir() and not any(old_data.iterdir()):
        print(f"remove {old_data} (now empty)")
        old_data.rmdir()

    print(f"""
Done. Next:
  1. set [label_studio] project_id in {root / 'project.toml'} to your existing
     Label Studio project id
  2. export AUTO_LABELLER_PROJECT={root} && docker compose up -d
     (restarts Label Studio with the images mounted from the new location)
  3. auto-labeller projects""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
