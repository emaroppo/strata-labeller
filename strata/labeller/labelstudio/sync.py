"""Keeping Label Studio and the catalog in step.

One direction each way, and the catalog wins. Task ids are cached, not stored: a sample
is recognised by its task's blob and type, so the map can be rebuilt by listing tasks.
See ``docs/adr/0029`` and ``docs/adr/0052``.
"""

import json
from dataclasses import dataclass, field

from strata.catalog import AmbiguousSample, Catalog, SampleRow
from strata.contracts import AnyValue, Spans

from ..project import LabellingProject
from ..trim import text_of
from .adapter import Addressing, Task, build_tasks, from_results
from .schemas import LabelSchema


@dataclass
class PushReport:
    pushed: int = 0
    already_present: int = 0


@dataclass
class PullReport:
    annotated: int = 0
    skipped: int = 0
    #: Answered in Label Studio, but by an import rather than by a person
    untouched: int = 0
    #: Spans whose edges were moved off whitespace on the way in
    trimmed: int = 0
    unrecognised: list[str] = field(default_factory=list)
    #: Tasks with no type whose bytes several samples hold. docs/adr/0052
    ambiguous: list[int] = field(default_factory=list)
    undeclared: set[str] = field(default_factory=set)

    @property
    def total(self) -> int:
        return self.annotated + self.skipped


class TaskMapError(Exception):
    """The cached map belongs to a different catalog."""


def task_map_path(project: LabellingProject, ls_project_id: int):
    return project.state_dir / f"tasks_{ls_project_id}.json"


def task_map_catalog(project: LabellingProject, ls_project_id: int) -> str | None:
    """Which catalog the cached map was written against, if it says.

    ``None`` for a map written before it recorded one, which is not the
    same as a map that disagrees. See ``docs/adr/0008``.
    """
    path = task_map_path(project, ls_project_id)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return payload.get("catalog") if isinstance(payload, dict) else None


def load_task_map(
    project: LabellingProject, ls_project_id: int, catalog_id: str | None = None
) -> dict[int, int]:
    """The cached sample id -> task id map, refusing another catalog's.

    A map from before identities were recorded is adopted, and the caller
    says so; ``push --rebuild-map`` settles any doubt. See ``docs/adr/0008``.
    """
    path = task_map_path(project, ls_project_id)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())

    tasks = payload
    if isinstance(payload, dict) and "tasks" in payload:
        stored = payload.get("catalog")
        tasks = payload["tasks"]
        if stored and catalog_id and stored != catalog_id:
            raise TaskMapError(
                f"{path} was written against catalog {stored}, and this "
                f"project now reads {catalog_id}. Sample ids mean different "
                f"things in each, so every task in it points somewhere else. "
                f"Point the project back at {stored}, or delete the file and "
                f"let 'push --rebuild-map' rebuild it from Label Studio."
            )
    # JSON keys are strings; sample ids are not
    return {int(k): v for k, v in tasks.items()}


def save_task_map(
    project: LabellingProject,
    ls_project_id: int,
    mapping: dict[int, int],
    catalog_id: str | None = None,
) -> None:
    path = task_map_path(project, ls_project_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        # Stamped on every write. docs/adr/0008
        "catalog": catalog_id or "",
        "tasks": {str(k): v for k, v in mapping.items()},
    }
    path.write_text(json.dumps(payload))


@dataclass
class RelinkReport:
    """What repointing a project's tasks did, or would do."""

    #: (task id, new data) for each task whose URL or type should change.
    changes: list[tuple[int, dict]] = field(default_factory=list)
    unchanged: int = 0
    #: URLs naming no sample, left alone: rewriting one would lose what the task showed.
    unrecognised: list[str] = field(default_factory=list)
    #: Tasks with no type whose bytes several samples hold, left alone. docs/adr/0052
    ambiguous: list[int] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.changes) + self.unchanged + len(self.unrecognised) + len(self.ambiguous)


def relink(
    tasks: list[dict], catalog: Catalog, addressing: Addressing, data_key: str
) -> RelinkReport:
    """Work out how each task's data should now read; the caller applies it.

    Moves tasks off the mount onto the serving API, re-signs, and gives a typeless task
    its one sample's type. See ``docs/adr/0013``, ``docs/adr/0030`` and ``docs/adr/0052``.
    """
    report = RelinkReport()
    for task in tasks:
        data = task.get("data") or {}
        try:
            sample = addressing.sample_from(data, data_key, catalog)
        except AmbiguousSample:
            report.ambiguous.append(task["id"])
            continue
        if sample is None:
            report.unrecognised.append(data.get(data_key, ""))
            continue

        fresh = {**data, **addressing.data_for(sample, data_key)}
        if fresh == data:
            report.unchanged += 1
            continue
        report.changes.append((task["id"], fresh))
    return report


def rebuild_task_map(
    tasks: list[dict], catalog: Catalog, addressing: Addressing, data_key: str
) -> tuple[dict[int, int], list[str], list[int]]:
    """Recover sample id -> task id from what Label Studio holds.

    Returns the map, the URLs it could not place, and the ids of typeless tasks whose
    bytes several samples hold; neither is matched to a guess. See ``docs/adr/0052``.
    """
    mapping: dict[int, int] = {}
    unrecognised: list[str] = []
    ambiguous: list[int] = []
    for task in tasks:
        data = task.get("data") or {}
        try:
            sample = addressing.sample_from(data, data_key, catalog)
        except AmbiguousSample:
            ambiguous.append(task["id"])
            continue
        if sample is None:
            unrecognised.append(data.get(data_key, ""))
            continue
        mapping[sample.id] = task["id"]
    return mapping, unrecognised, ambiguous


def tasks_to_push(
    samples: list[SampleRow],
    catalog: Catalog,
    label_set_id: int,
    schema: LabelSchema,
    addressing: Addressing,
    existing: dict[int, int],
) -> tuple[list[Task], PushReport]:
    """Tasks for samples Label Studio does not have yet. See ``docs/adr/0032``."""
    wanted = [s for s in samples if s.id not in existing]
    report = PushReport(pushed=len(wanted), already_present=len(samples) - len(wanted))
    return build_tasks(wanted, catalog, label_set_id, schema, addressing), report


def _was_opened(annotation: dict) -> bool:
    """Whether a person has actually been in this annotation.

    Any of three marks will do: ``lead_time``, a draft, or ``updated_by``.
    See ``docs/adr/0028``.
    """
    return bool(
        annotation.get("lead_time")
        or annotation.get("draft_created_at")
        or annotation.get("updated_by")
    )


def pull_annotations(
    exported: list[dict],
    catalog: Catalog,
    label_set_id: int,
    schema: LabelSchema,
    addressing: Addressing,
    declared: list[str],
    reviewed_only: bool = False,
) -> tuple[list[tuple[int, AnyValue | None]], PullReport]:
    """Turn a Label Studio export into catalog writes.

    A task with no annotation is left alone, a cancelled one is a skip, and ``reviewed_only``
    keeps back annotations nobody opened. Spans are trimmed of surrounding whitespace against
    the document. See ``docs/adr/0028`` and ``docs/adr/0029``.
    """
    report = PullReport()
    items: list[tuple[int, AnyValue | None]] = []
    known = set(declared)
    stored = schema.catalog_schema()

    for task in exported:
        data = task.get("data") or {}
        try:
            sample = addressing.sample_from(data, schema.data_key, catalog)
        except AmbiguousSample:
            # Not matched to whichever row comes first. docs/adr/0052
            report.ambiguous.append(task["id"])
            continue
        if sample is None:
            report.unrecognised.append(data.get(schema.data_key, ""))
            continue

        annotations = task.get("annotations") or []
        if not annotations:
            continue

        annotation = annotations[0]
        if reviewed_only and not _was_opened(annotation):
            report.untouched += 1
            continue
        if annotation.get("was_cancelled"):
            items.append((sample.id, None))
            report.skipped += 1
            continue

        value = from_results(schema.canonicalize(annotation.get("result") or []), schema)
        if isinstance(value, Spans) and value.values:
            value, moved = value.trimmed(text_of(catalog, sample))
            report.trimmed += moved
        # Which classes a value asserts is the label type's own question. docs/adr/0039
        report.undeclared |= stored.classes_asserted(value) - known
        items.append((sample.id, value))
        report.annotated += 1

    return items, report
