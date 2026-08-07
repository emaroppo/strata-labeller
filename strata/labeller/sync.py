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

from .adapter import Task, build_tasks, from_results, location_from_url
from .project import Project
from .schemas import LabelSchema


@dataclass
class PushReport:
    pushed: int = 0
    already_present: int = 0


@dataclass
class PullReport:
    annotated: int = 0
    skipped: int = 0
    unrecognised: list[str] = field(default_factory=list)
    undeclared: set[str] = field(default_factory=set)

    @property
    def total(self) -> int:
        return self.annotated + self.skipped


# ----------------------------------------------------------------------
# The task map
# ----------------------------------------------------------------------


def task_map_path(project: Project, ls_project_id: int):
    return project.state_dir / f"tasks_{ls_project_id}.json"


def load_task_map(project: Project, ls_project_id: int) -> dict[int, int]:
    path = task_map_path(project, ls_project_id)
    if not path.exists():
        return {}
    # JSON keys are strings; sample ids are not
    return {int(k): v for k, v in json.loads(path.read_text()).items()}


def save_task_map(project: Project, ls_project_id: int, mapping: dict[int, int]) -> None:
    path = task_map_path(project, ls_project_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({str(k): v for k, v in mapping.items()}))


def rebuild_task_map(
    tasks: list[dict], catalog: Catalog, prefix: str, data_key: str
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
        location = location_from_url(url, prefix)
        sample = catalog.by_location(location) if location else None
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
    prefix: str,
    existing: dict[int, int],
) -> tuple[list[Task], PushReport]:
    """Tasks for samples Label Studio does not have yet.

    Skipping what is already there is what makes a push resumable: an
    interrupted one can simply be run again.
    """
    wanted = [s for s in samples if s.id not in existing]
    report = PushReport(pushed=len(wanted), already_present=len(samples) - len(wanted))
    return build_tasks(wanted, catalog, label_set_id, schema, prefix), report


# ----------------------------------------------------------------------
# Back
# ----------------------------------------------------------------------


def pull_annotations(
    exported: list[dict],
    catalog: Catalog,
    label_set_id: int,
    schema: LabelSchema,
    prefix: str,
    declared: list[str],
) -> tuple[list[tuple[int, Choices | None]], PullReport]:
    """Turn a Label Studio export into catalog writes.

    A task with no annotation is left alone rather than recorded as empty:
    nobody has answered it, and writing an empty value would claim they had.
    A task marked cancelled is a skip — reviewed, nothing applicable.
    """
    report = PullReport()
    items: list[tuple[int, Choices | None]] = []
    known = set(declared)

    for task in exported:
        url = (task.get("data") or {}).get(schema.data_key, "")
        location = location_from_url(url, prefix)
        sample = catalog.by_location(location) if location else None
        if sample is None:
            report.unrecognised.append(url)
            continue

        annotations = task.get("annotations") or []
        if not annotations:
            continue

        annotation = annotations[0]
        if annotation.get("was_cancelled"):
            items.append((sample.id, None))
            report.skipped += 1
            continue

        value = from_results(schema.canonicalize(annotation.get("result") or []), schema)
        report.undeclared |= set(value.values) - known
        items.append((sample.id, value))
        report.annotated += 1

    return items, report
