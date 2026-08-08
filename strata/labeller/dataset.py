import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .schemas import LabelSchema, Result

# Read-only now. The catalog is the store; this reads the format it replaced,
# so a project that predates it — or a dataset.json handed over by someone
# else — still has a way in through `to-catalog`. Nothing writes it.
#
# v1 stored {"path", "labels", "skipped"} in a bare list; v2 stores
# canonicalized Label Studio results so any task type fits, and records
# whether a human has annotated the sample rather than inferring it from
# the annotation being non-empty (an image with no boxes is a real answer)
DATASET_VERSION = 2


@dataclass
class Sample:
    # Relative to the project's data root, so the dataset survives moving
    # the project or repointing [data] root
    path: str
    results: list[Result] = field(default_factory=list)
    annotated: bool = False
    # Reviewed but nothing applies: excluded from training and from the
    # unlabeled pool, so it never reappears in the review queue
    skipped: bool = False

    @property
    def is_labeled(self) -> bool:
        return self.annotated and not self.skipped


def load_dataset(path: Path, schema: LabelSchema | None = None) -> list[Sample]:
    """Load a dataset, upgrading a v1 file on the way in.

    ``schema`` is needed only to upgrade v1 files, whose labels carry no
    control names of their own.
    """
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return [_upgrade_v1(item, schema) for item in data]

    version = data.get("version")
    if version != DATASET_VERSION:
        raise ValueError(
            f"{path} has dataset version {version!r}; this build reads "
            f"version {DATASET_VERSION} and the original v1 list format"
        )
    return [
        Sample(
            path=item["path"],
            results=item.get("results", []),
            annotated=item.get("annotated", False),
            skipped=item.get("skipped", False),
        )
        for item in data.get("samples", [])
    ]


def _upgrade_v1(item: dict, schema: LabelSchema | None) -> Sample:
    labels = item.get("labels") or []
    if labels and schema is None:
        raise ValueError(
            "Upgrading a v1 dataset needs the project's schema to know which "
            "labeling control its labels belong to"
        )
    return Sample(
        path=item["path"],
        results=schema.encode_target(labels) if labels else [],
        annotated=bool(labels),
        skipped=item.get("skipped", False),
    )


def save_dataset(samples: list[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    items = []
    for s in samples:
        item: dict = {"path": s.path}
        if s.results:
            item["results"] = s.results
        if s.annotated:
            item["annotated"] = True
        if s.skipped:
            item["skipped"] = True
        items.append(item)
    with open(path, "w") as f:
        json.dump({"version": DATASET_VERSION, "samples": items}, f, indent=2)


def is_v1(path: Path) -> bool:
    with open(path) as f:
        return isinstance(json.load(f), list)


def get_classes(samples: list[Sample], schema: LabelSchema) -> list[str]:
    return schema.classes_in_use([s.results for s in samples])


def split_labeled_unlabeled(
    samples: list[Sample],
) -> tuple[list[Sample], list[Sample]]:
    """Split into (labeled, unlabeled); skipped samples belong to neither."""
    labeled = [s for s in samples if s.is_labeled]
    unlabeled = [s for s in samples if not s.annotated and not s.skipped]
    return labeled, unlabeled


def train_val_split(
    samples: list[Sample],
    val_ratio: float = 0.2,
    seed: int = 42,
    group_key: Callable[[Sample], str] | None = None,
) -> tuple[list[Sample], list[Sample]]:
    """Split samples into train/val.

    With group_key, whole groups (e.g. all frames of one video) land on the
    same side of the split so near-duplicate frames can't leak into val.
    """
    rng = random.Random(seed)
    if group_key is None:
        shuffled = list(samples)
        rng.shuffle(shuffled)
        split_idx = int(len(shuffled) * (1 - val_ratio))
        return shuffled[:split_idx], shuffled[split_idx:]

    groups: dict[str, list[Sample]] = {}
    for s in samples:
        groups.setdefault(group_key(s), []).append(s)
    keys = sorted(groups)
    rng.shuffle(keys)

    target_val = int(len(samples) * val_ratio)
    train: list[Sample] = []
    val: list[Sample] = []
    for key in keys:
        bucket = val if len(val) < target_val else train
        bucket.extend(groups[key])
    return train, val
