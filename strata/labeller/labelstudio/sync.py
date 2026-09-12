"""Keeping Label Studio and the catalog in step.

One direction each way, and the catalog wins. Label Studio is where a human
answers questions; the catalog is what remembers. That ordering is what lets
a Label Studio project be deleted and rebuilt without losing anything, and
it is why nothing here treats a task id as worth preserving.

Task ids are cached rather than stored. A sample is recognised by the blob
its task points at, so the map can always be rebuilt by listing tasks — the
cache only saves the listing.
"""

import json
from dataclasses import dataclass, field

from strata.catalog import Catalog, SampleRow
from strata.labels import Choices

from ..project import Project
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
    unrecognised: list[str] = field(default_factory=list)
    undeclared: set[str] = field(default_factory=set)

    @property
    def total(self) -> int:
        return self.annotated + self.skipped


# ----------------------------------------------------------------------
# The task map
# ----------------------------------------------------------------------


class TaskMapError(Exception):
    """The cached map belongs to a different catalog."""


def task_map_path(project: Project, ls_project_id: int):
    return project.state_dir / f"tasks_{ls_project_id}.json"


def task_map_catalog(project: Project, ls_project_id: int) -> str | None:
    """Which catalog the cached map was written against, if it says.

    ``None`` for a map written before it recorded one, which is not the
    same as a map that disagrees — see :func:`load_task_map`.
    """
    path = task_map_path(project, ls_project_id)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return payload.get("catalog") if isinstance(payload, dict) else None


def load_task_map(
    project: Project, ls_project_id: int, catalog_id: str | None = None
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
    project: Project,
    ls_project_id: int,
    mapping: dict[int, int],
    catalog_id: str | None = None,
) -> None:
    path = task_map_path(project, ls_project_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        # Stamped on every write, so a map adopted from before this becomes
        # a guarded one the first time anything touches it
        "catalog": catalog_id or "",
        "tasks": {str(k): v for k, v in mapping.items()},
    }
    path.write_text(json.dumps(payload))


@dataclass
class RelinkReport:
    """What repointing a project's tasks did, or would do."""

    #: (task id, new data) for each task whose URL should change.
    changes: list[tuple[int, dict]] = field(default_factory=list)
    unchanged: int = 0
    #: URLs naming no sample. Left alone rather than guessed at — a task
    #: from before the catalog points at a real image this cannot identify,
    #: and rewriting it would lose the only record of what it showed.
    unrecognised: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.changes) + self.unchanged + len(self.unrecognised)


def relink(
    tasks: list[dict], catalog: Catalog, addressing: Addressing, data_key: str
) -> RelinkReport:
    """Work out how each task's image URL should now read.

    Two jobs, and they are the same operation. It moves tasks from the local
    mount onto the serving API, which is what lets the mount go away. It also
    re-signs: a signed URL expires, so a task that sits in a review queue
    longer than the signature's life stops loading, and running this again
    is the fix.

    Computed here and applied by the caller, so a dry run costs nothing and
    an interrupted apply leaves the rest still describable.
    """
    report = RelinkReport()
    for task in tasks:
        data = task.get("data") or {}
        url = data.get(data_key, "")
        checksum = addressing.checksum_from(url)
        sample = catalog.samples.by_checksum(checksum) if checksum else None
        if sample is None:
            report.unrecognised.append(url)
            continue

        fresh = addressing.url_for(sample)
        if fresh == url:
            report.unchanged += 1
            continue
        report.changes.append((task["id"], {**data, data_key: fresh}))
    return report


def rebuild_task_map(
    tasks: list[dict], catalog: Catalog, addressing: Addressing, data_key: str
) -> tuple[dict[int, int], list[str]]:
    """Recover sample id -> task id from what Label Studio holds.

    Returns the map and the URLs it could not place. A URL that names no
    blob is the ordinary case for a task made before the cutover, and saying
    so beats matching it to the wrong sample.
    """
    mapping: dict[int, int] = {}
    unrecognised: list[str] = []
    for task in tasks:
        url = (task.get("data") or {}).get(data_key, "")
        checksum = addressing.checksum_from(url)
        sample = catalog.samples.by_checksum(checksum) if checksum else None
        if sample is None:
            unrecognised.append(url)
            continue
        mapping[sample.id] = task["id"]
    return mapping, unrecognised


# ----------------------------------------------------------------------
# Out
# ----------------------------------------------------------------------


def tasks_to_push(
    samples: list[SampleRow],
    catalog: Catalog,
    label_set_id: int,
    schema: LabelSchema,
    addressing: Addressing,
    existing: dict[int, int],
) -> tuple[list[Task], PushReport]:
    """Tasks for samples Label Studio does not have yet.

    Skipping what is already there is what makes a push resumable: an
    interrupted one can simply be run again.
    """
    wanted = [s for s in samples if s.id not in existing]
    report = PushReport(pushed=len(wanted), already_present=len(samples) - len(wanted))
    return build_tasks(wanted, catalog, label_set_id, schema, addressing), report


# ----------------------------------------------------------------------
# Back
# ----------------------------------------------------------------------


def _was_opened(annotation: dict) -> bool:
    """Whether a person has actually been in this annotation.

    Any of three marks will do. ``lead_time`` is the seconds Label Studio
    measured; a draft means they started; ``updated_by`` means they saved.
    An annotation that arrived through an import and was never opened
    carries none of them.
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
) -> tuple[list[tuple[int, Choices | None]], PullReport]:
    """Turn a Label Studio export into catalog writes.

    A task with no annotation is left alone rather than recorded as empty:
    nobody has answered it, and writing an empty value would claim they had.
    A task marked cancelled is a skip — reviewed, nothing applicable.

    ``reviewed_only`` keeps back annotations nobody has opened. It matters
    when a project was seeded from somewhere else: those tasks arrive
    already answered, an export writes every answer back as a human one,
    and a partly-reviewed queue then hands back the seed's own guesses
    stamped as ground truth. Label Studio records the time spent on an
    annotation, and an untouched one has none.
    """
    report = PullReport()
    items: list[tuple[int, Choices | None]] = []
    known = set(declared)
    stored = schema.catalog_schema()

    for task in exported:
        url = (task.get("data") or {}).get(schema.data_key, "")
        checksum = addressing.checksum_from(url)
        sample = catalog.samples.by_checksum(checksum) if checksum else None
        if sample is None:
            report.unrecognised.append(url)
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
        # Which classes a value asserts is the label type's own question,
        # and every schema answers it — reading `values` as class names is
        # true only for classification, where they happen to be strings.
        # For spans they are Span objects, which do not even compare.
        report.undeclared |= stored.classes_asserted(value) - known
        items.append((sample.id, value))
        report.annotated += 1

    return items, report
