"""A training round, on the catalog.

What ``train.py`` did in one process is now three steps across three
packages: the catalog freezes a dataset version, materialises it, and
modelling trains from the directory. The labeller only sequences them.

The gain is not tidiness. Each round's dataset is written down rather than
assembled on the fly, so a run resolves back to the exact samples and
annotations behind it, and validation membership is inherited from the
previous version instead of being recomputed — which is what stopped a
warm-started model being scored on what it had already trained on.
"""

from dataclasses import dataclass
from pathlib import Path

from strata.catalog import Catalog, CatalogError, Manifest
from strata.modelling import Run, RunStore, TrainRequest, train
from strata.modelling.registry import absolute

from .project import Project


class RoundError(Exception):
    """A round that cannot be run as the project stands."""


@dataclass
class RoundResult:
    """One round, start to finish."""

    run: Run
    manifest: Manifest
    dataset_dir: Path

    @property
    def warm_started(self) -> bool:
        return self.run.parent_run_id is not None


def run_round(
    project: Project,
    catalog: Catalog,
    fresh: bool = False,
    val_ratio: float = 0.2,
) -> RoundResult:
    """Freeze a dataset version, materialise it, and train from it."""
    try:
        label_set_id, _ = catalog.label_set(project.label_set_name)
    except CatalogError as exc:
        raise RoundError(
            f"No label set named {project.label_set_name!r} in the catalog. "
            f"Run 'auto-labeller to-catalog' first, or set [catalog] label_set."
        ) from exc

    labelled = catalog.labelled(label_set_id)
    if not labelled:
        raise RoundError(
            f"Nothing is labelled for {project.label_set_name!r} yet, so there "
            f"is nothing to train on."
        )

    dataset_id = catalog.create_dataset(
        project.dataset_name, label_set_id, val_ratio=val_ratio
    )
    manifest, dataset_dir = _materialise(project, catalog, dataset_id)

    store = RunStore.local(project.runs_dir)
    # Warm start from the newest run over this dataset unless told otherwise.
    # The policy lives here rather than inside modelling, which is handed a
    # parent id or nothing.
    previous = None if fresh else store.latest(project.dataset_name)

    run = train(
        TrainRequest(
            dataset_dir=dataset_dir,
            # Anchored at the project, because a model.py belongs to the job
            # rather than to the dataset it happens to be trained on. An
            # absolute ref also resolves from anywhere, which is what a
            # request has to do once it crosses a wire.
            model=absolute(project.model.ref, project.root),
            params=project.model.params,
            parent_run_id=previous.id if previous else None,
        ),
        store,
    )
    return RoundResult(run=run, manifest=manifest, dataset_dir=dataset_dir)


def _materialise(project: Project, catalog: Catalog, dataset_id: int) -> tuple[Manifest, Path]:
    target = project.datasets_dir / project.dataset_name
    # Versions are read off the manifest rather than counted here, so the
    # directory name and what is inside it cannot disagree
    staging = target / "pending"
    catalog.materialise(dataset_id, staging)
    manifest = Manifest.model_validate_json((staging / "manifest.json").read_text())
    final = target / f"v{manifest.version:03d}"
    if final.exists():
        raise RoundError(f"{final} already exists; refusing to overwrite a dataset version")
    staging.rename(final)
    return manifest, final


def describe(result: RoundResult) -> list[str]:
    manifest = result.manifest
    lines = [
        f"Dataset {manifest.dataset} v{manifest.version}: "
        f"{len(manifest.train)} train, {len(manifest.val)} val",
    ]
    if abs(manifest.val_ratio_achieved - manifest.val_ratio) > 0.02:
        # Grouping can make the target unreachable, and a val figure read
        # without knowing that is misleading
        lines.append(
            f"  validation is {manifest.val_ratio_achieved:.0%}, not the "
            f"{manifest.val_ratio:.0%} asked for — groups are indivisible"
        )
    lines.append(
        f"Run {result.run.id}"
        + (f", continuing run {result.run.parent_run_id}" if result.warm_started else " (cold)")
    )
    for name, value in sorted(result.run.metrics.items()):
        lines.append(f"  {name}: {value}")
    return lines
