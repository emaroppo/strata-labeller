import json
from datetime import datetime
from pathlib import Path

from strata.modelling import Model

from .dataset import (
    get_classes,
    load_dataset,
    save_dataset,
    split_labeled_unlabeled,
    train_val_split,
)
from .project import Project


def run_training(
    model: Model,
    project: Project,
    round_num: int | None = None,
) -> dict:
    schema = project.schema
    if getattr(model, "schema_type", schema.type) != schema.type:
        raise ValueError(
            f"Model {type(model).__name__} is written for '{model.schema_type}' "
            f"but this project labels with '{schema.type}'"
        )

    dataset = load_dataset(project.dataset_path, schema)
    labeled, unlabeled = split_labeled_unlabeled(dataset)
    classes = schema.classes or get_classes(labeled, schema)

    if not labeled:
        raise ValueError("No labeled samples found in dataset")

    train_samples, val_samples = train_val_split(labeled, group_key=project.group_key)

    if round_num is None:
        round_num = _next_round_num(project.rounds_dir)

    def to_model_input(samples):
        # Models receive absolute paths and the schema's own target type
        return [
            {
                "path": str(project.sample_file(s.path)),
                "target": schema.decode_target(s.results),
            }
            for s in samples
        ]

    metrics = model.finetune(
        to_model_input(train_samples),
        classes,
        val_samples=to_model_input(val_samples),
    )

    checkpoint_path = project.checkpoints_dir / f"round_{round_num:03d}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint_path)

    round_dir = project.rounds_dir / f"round_{round_num:03d}"
    round_dir.mkdir(parents=True, exist_ok=True)

    round_meta = {
        "round": round_num,
        "timestamp": datetime.now().isoformat(),
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "num_unlabeled": len(unlabeled),
        "num_skipped": sum(1 for s in dataset if s.skipped),
        "schema": schema.type,
        "classes": classes,
        "metrics": metrics,
        "checkpoint": str(checkpoint_path.relative_to(project.root)),
    }
    with open(round_dir / "metadata.json", "w") as f:
        json.dump(round_meta, f, indent=2)

    save_dataset(labeled, round_dir / "labeled.json")

    return round_meta


def _next_round_num(rounds_dir: Path) -> int:
    if not rounds_dir.exists():
        return 1
    existing = [
        d
        for d in rounds_dir.iterdir()
        if d.is_dir() and d.name.startswith("round_")
    ]
    if not existing:
        return 1
    return max(int(d.name.split("_")[1]) for d in existing) + 1
