"""Carrying `rounds/*/metadata.json` into the run store.

The pre-catalog rounds hold the only record of how a model got where it is —
accuracy per round is the curve, and throwing it away to start a clean run
store would lose the more interesting half of it.

Three things cannot be recovered and are handled rather than guessed:

*Lineage.* Warm starting arrived partway through this project's life, so an
early round was trained from scratch and a later one continued its
predecessor, and nothing on disk says which. Imported runs are therefore
unchained unless ``chain`` is passed, and a chain asserted that way is the
caller's claim rather than something read off the files.

*Dataset versions.* An imported round trained on samples no dataset version
describes, so it records none. The round number is not lost: rounds are
imported in order into an empty store, so a run's id is its round number.

*Checkpoints* are referenced where they already sit rather than copied. They
are the large part of a project, and duplicating gigabytes to change a
filename is a poor trade — at the cost that emptying ``checkpoints/`` leaves
those runs pointing at nothing.
"""

import json
from dataclasses import dataclass, field

from strata.modelling import Run, RunStore

from .project import Project


@dataclass
class ImportReport:
    imported: int = 0
    skipped: list[str] = field(default_factory=list)
    missing_checkpoints: list[str] = field(default_factory=list)
    runs: list[Run] = field(default_factory=list)


def read_rounds(project: Project) -> list[dict]:
    """Every round on disk, oldest first."""
    if not project.rounds_dir.exists():
        return []
    found = []
    for directory in sorted(project.rounds_dir.iterdir()):
        metadata = directory / "metadata.json"
        if directory.is_dir() and metadata.exists():
            found.append(json.loads(metadata.read_text()))
    return sorted(found, key=lambda m: m.get("round", 0))


def import_rounds(
    project: Project,
    store: RunStore | None = None,
    chain: bool = False,
    model_version: str = "1",
) -> ImportReport:
    """Write the project's historical rounds into its run store."""
    store = store or RunStore.local(project.runs_dir)
    report = ImportReport()
    previous: Run | None = None

    for metadata in read_rounds(project):
        number = metadata.get("round")
        classes = metadata.get("classes") or []
        if not classes:
            # Without the class list a checkpoint's output neurons cannot be
            # matched to anything, so the run is not usable for a warm start
            # and recording it would only be decoration
            report.skipped.append(f"round {number}: no class list recorded")
            continue

        checkpoint = metadata.get("checkpoint")
        path = project.root / checkpoint if checkpoint else None
        if path is not None and not path.exists():
            report.missing_checkpoints.append(f"round {number}: {checkpoint}")
            path = None

        run = store.record(
            Run(
                id=0,
                parent_run_id=previous.id if (chain and previous) else None,
                dataset=project.dataset_name,
                dataset_version=None,
                label_set=project.label_set_name,
                model=project.model.ref,
                model_version=model_version,
                params=project.model.params,
                classes=list(classes),
                checkpoint=path,
            ),
            _numeric(metadata.get("metrics") or {}),
        )
        report.imported += 1
        report.runs.append(run)
        previous = run

    return report


def _numeric(metrics: dict) -> dict[str, float]:
    return {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}


def describe(report: ImportReport, chained: bool) -> list[str]:
    lines = [f"Imported {report.imported} round(s)"]
    if report.imported and not chained:
        lines.append(
            "  unchained: nothing on disk says which rounds were warm-started, "
            "so lineage is left unclaimed — pass --chain if you know they were"
        )
    for note in report.skipped:
        lines.append(f"  skipped {note}")
    for note in report.missing_checkpoints:
        lines.append(f"  no checkpoint for {note}; the run is recorded without one")
    return lines
