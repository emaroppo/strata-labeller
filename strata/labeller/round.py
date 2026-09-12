"""A training round, on the catalog.

What ``train.py`` did in one process is three stages across two packages:
the catalog freezes a dataset version and materialises it, and modelling
trains from the directory. The labeller only sequences them — and the
sequence is the same one an experiment file writes down, because these are
the same stage functions.

The gain is not tidiness. Each round's dataset is written down rather than
assembled on the fly, so a run resolves back to the exact samples and
annotations behind it, and validation membership is inherited from the
previous version instead of being recomputed — which is what stopped a
warm-started model being scored on what it had already trained on.
"""

from dataclasses import dataclass
from pathlib import Path

from strata.catalog import Catalog, CatalogError
from strata.catalog.stages import Context as CatalogContext
from strata.catalog.stages import DatasetRequest, MaterialiseRequest, dataset, materialise
from strata.labels import MANIFEST_NAME, Manifest
from strata.modelling import Run, RunStore
from strata.modelling.registry import absolute
from strata.modelling.stages import Context as ModellingContext
from strata.modelling.stages import TrainStageRequest, train

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
    on_progress=None,
    cache: Path | None = None,
) -> RoundResult:
    """Freeze a dataset version, materialise it, and train from it."""
    context = CatalogContext(
        catalog, project.datasets_dir, cache=cache, on_progress=on_progress
    )
    try:
        frozen = dataset(
            DatasetRequest(
                name=project.dataset_name,
                label_set=project.label_set_name,
                collections=project.collections,
                val_ratio=val_ratio,
            ),
            context,
        )
    except CatalogError as exc:
        raise RoundError(str(exc)) from exc
    built = materialise(
        MaterialiseRequest(
            dataset_id=frozen.dataset_id,
            features=[spec.as_dict() for spec in project.feature_specs],
        ),
        context,
    )

    store = RunStore.local(project.runs_dir)
    # The warm-start policy is the caller's, and this caller's is the
    # default: the newest run over this dataset unless told to start fresh.
    record = train(
        TrainStageRequest(
            dataset_dir=built.directory,
            # Anchored at the project, because a model.py belongs to the job
            # rather than to the dataset it happens to be trained on. An
            # absolute ref also resolves from anywhere, which is what a
            # request has to do once it crosses a wire.
            model=absolute(project.model.ref, project.root),
            params=project.model.params,
            fresh_params=project.model.fresh_params,
            fresh=fresh,
        ),
        ModellingContext(store=store),
    )
    manifest = Manifest.model_validate_json((built.directory / MANIFEST_NAME).read_text())
    return RoundResult(run=store.get(record.run_id), manifest=manifest, dataset_dir=built.directory)


def describe(result: RoundResult) -> list[str]:
    manifest = result.manifest
    lines = [
        f"Dataset {manifest.dataset} v{manifest.version}: "
        f"{len(manifest.train)} train, {len(manifest.val)} val"
        + (f", {len(manifest.holdout)} held out" if manifest.holdout else ""),
    ]
    if (
        manifest.val_ratio is not None
        and manifest.val_ratio_achieved is not None
        and abs(manifest.val_ratio_achieved - manifest.val_ratio) > 0.02
    ):
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
