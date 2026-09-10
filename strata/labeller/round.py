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

import shutil
from dataclasses import dataclass
from pathlib import Path

from strata.catalog import Catalog, CatalogError
from strata.labels import MANIFEST_NAME, Manifest, ManifestFormatError
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
    on_progress=None,
    cache: Path | None = None,
) -> RoundResult:
    """Freeze a dataset version, materialise it, and train from it."""
    try:
        label_set_id, _ = catalog.label_set(project.label_set_name)
    except CatalogError as exc:
        raise RoundError(
            f"No label set named {project.label_set_name!r} in the catalog. "
            f"Run 'auto-labeller ingest' first, or set [catalog] label_set."
        ) from exc

    labelled = catalog.labelled(label_set_id, project.collections)
    if not labelled:
        raise RoundError(
            f"Nothing is labelled for {project.label_set_name!r} in "
            f"{', '.join(project.collections)}, so there is nothing to train on."
        )

    dataset_id = catalog.create_dataset(
        project.dataset_name,
        label_set_id,
        collections=project.collections,
        val_ratio=val_ratio,
    )
    manifest, dataset_dir = _materialise(project, catalog, dataset_id, on_progress, cache)

    store = RunStore.local(project.runs_dir)
    # Warm start from the newest run over this dataset unless told otherwise.
    # The policy lives here rather than inside modelling, which is handed a
    # parent id or nothing.
    previous = None if fresh else store.latest(project.dataset_name, catalog.id)
    # Asked after the parent is known, not before. --fresh is a request and
    # being cold is an outcome; they part company when nothing has trained
    # on this dataset yet, and a cold run then trained for as long as a warm
    # one — which is most of why two of the early baselines were not
    # baselines.
    cold = previous is None

    run = train(
        TrainRequest(
            dataset_dir=dataset_dir,
            # Anchored at the project, because a model.py belongs to the job
            # rather than to the dataset it happens to be trained on. An
            # absolute ref also resolves from anywhere, which is what a
            # request has to do once it crosses a wire.
            model=absolute(project.model_ref, project.root),
            params=project.model.params_for(cold),
            parent_run_id=previous.id if previous else None,
        ),
        store,
    )
    return RoundResult(run=run, manifest=manifest, dataset_dir=dataset_dir)


def _materialise(
    project: Project,
    catalog: Catalog,
    dataset_id: int,
    on_progress=None,
    cache: Path | None = None,
) -> tuple[Manifest, Path]:
    target = project.datasets_dir / project.dataset_name

    # Asked before fetching, not after. A round retried after crashing —
    # which is the ordinary case, since training is the part that runs out of
    # memory — reuses its version, and materialising into staging only to
    # discover the directory already existed meant pulling the whole dataset
    # out of object storage to delete it. Cheap when blobs were local files
    # and a hard link away; minutes and gigabytes once they are not.
    specs = project.feature_specs
    final = target / f"v{catalog.dataset_version(dataset_id):03d}"
    if (final / MANIFEST_NAME).exists():
        # Membership is what makes a version, and the catalog already
        # confirmed it matches — including the answers, since a version's
        # identity covers its annotations. The manifest rather than the
        # directory, because only the manifest proves the rename completed.
        try:
            manifest = Manifest.model_validate_json((final / MANIFEST_NAME).read_text())
        except ManifestFormatError:
            # Written by a release whose layout this one does not read, or
            # before manifests said which layout they were. The directory is
            # only a copy of what the catalog holds, so rebuilding it costs a
            # fetch — and guessing at its fields could cost a round trained
            # on the wrong split.
            manifest = None
        # Features are not part of a version's identity: they change what
        # the model is *told*, not which samples were selected or what was
        # said about them. So a project that adds one keeps its version and
        # needs the directory rebuilt — otherwise the round trains from a
        # manifest written before the feature existed and reports a number
        # for a model that never saw it.
        if manifest is not None and [dict(f) for f in manifest.features] == [
            s.as_dict() for s in specs
        ]:
            _finished(on_progress, manifest)
            return manifest, final
        shutil.rmtree(final)

    staging = target / "pending"
    if staging.exists():
        # An interrupted fetch. Its contents are unknowable — some files
        # written, no manifest — and keeping them would let a partial
        # dataset masquerade as a whole one.
        shutil.rmtree(staging)
    catalog.materialise(
        dataset_id, staging, on_progress=on_progress, cache=cache, features=specs
    )
    manifest = Manifest.model_validate_json((staging / MANIFEST_NAME).read_text())
    staging.rename(final)
    _finished(on_progress, manifest)
    return manifest, final


def _finished(on_progress, manifest: Manifest) -> None:
    """One last tick, whichever way the version was obtained.

    A version already on disk fetches nothing, and a local backend links
    rather than downloads — so a caller watching ticks would never learn
    that materialising was over, and would keep saying so while the GPU ran.
    """
    if on_progress is not None:
        on_progress(len(manifest.samples), len(manifest.samples))


def describe(result: RoundResult) -> list[str]:
    manifest = result.manifest
    lines = [
        f"Dataset {manifest.dataset} v{manifest.version}: "
        f"{len(manifest.train)} train, {len(manifest.val)} val",
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
