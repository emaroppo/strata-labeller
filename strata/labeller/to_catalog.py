"""Moving a project's ``dataset.json`` into the catalog.

One direction only, and deliberately so: this is the bridge that gets years
of existing annotation into the catalog, not a synchronisation. Everything
Label Studio shaped stops here — results are decoded through the project's
schema and written as :mod:`strata.labels` values, so nothing downstream
ever sees a control name.

Re-running is safe. Ingest is idempotent on content and annotations are
upserted, so a migration interrupted halfway can simply be run again.
"""

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from strata.catalog import Catalog, CatalogError
from strata.labels import Choices, ClassificationSchema

from .dataset import load_dataset
from .project import Project
from .schemas import ClassificationSchema as LabelStudioClassification


class MigrationError(Exception):
    """A project that cannot be moved into the catalog as it stands."""


@dataclass
class Report:
    """What a migration did, for printing and for asserting on."""

    label_set: str
    label_set_id: int
    classes: list[str] = field(default_factory=list)
    ingested: int = 0
    annotated: int = 0
    skipped: int = 0
    unlabelled: int = 0
    missing: list[str] = field(default_factory=list)

    @property
    def samples(self) -> int:
        return self.ingested


def schema_for(project: Project) -> ClassificationSchema:
    """The project's label schema, with Label Studio left behind.

    Media does not survive the crossing: ``image_classification`` and
    ``text_classification`` were template names, and classifying a
    photograph and classifying a document are the same task. What a sample
    is made of is the catalog's business.
    """
    schema = project.schema
    if not isinstance(schema, LabelStudioClassification):
        raise MigrationError(
            f"Only classification projects can be migrated so far; "
            f"'{schema.type}' has no equivalent in strata.labels yet."
        )
    classes = list(project.label_config.classes)
    if not classes:
        # A project that never pinned its classes inferred them from what was
        # in use. The catalog has no such inference, so this is where the
        # list stops being implicit.
        dataset = load_dataset(project.dataset_path, schema)
        classes = schema.classes_in_use([s.results for s in dataset])
    return ClassificationSchema(classes=classes, multiple=schema.choice != "single")


def group_id_for(project: Project, sample_path: str) -> str | None:
    """Which group a sample belongs to, from ``[data] kind``.

    The one thing that setting was ever needed for, and the last thing it
    does: from here on grouping is a column on the sample, so a catalog can
    hold standalone images and video frames without a setting to choose
    between them.
    """
    if project.data.kind != "frames":
        return None
    return str(PurePosixPath(sample_path).parent)


def migrate(
    project: Project,
    catalog: Catalog,
    label_set: str | None = None,
    dry_run: bool = False,
) -> Report:
    """Move a project's samples and annotations into ``catalog``."""
    schema = schema_for(project)
    name = label_set or project.name
    subtype = "frames" if project.data.kind == "frames" else "plain"
    media = project.schema.media.name

    report = Report(label_set=name, label_set_id=0, classes=list(schema.classes))
    dataset = load_dataset(project.dataset_path, project.schema)

    if dry_run:
        for sample in dataset:
            if not project.sample_file(sample.path).exists():
                report.missing.append(sample.path)
                continue
            report.ingested += 1
            if sample.skipped:
                report.skipped += 1
            elif sample.annotated:
                report.annotated += 1
            else:
                report.unlabelled += 1
        return report

    report.label_set_id = _label_set(catalog, name, schema)

    for sample in dataset:
        path = project.sample_file(sample.path)
        if not path.exists():
            # A dataset can outlive the files it points at; say which rather
            # than failing the whole migration on one moved image
            report.missing.append(sample.path)
            continue
        [sample_id] = catalog.ingest(
            [path],
            media=media,
            subtype=subtype,
            group_id=group_id_for(project, sample.path),
        )
        report.ingested += 1

        if sample.skipped:
            catalog.skip(sample_id, report.label_set_id)
            report.skipped += 1
        elif sample.annotated:
            catalog.annotate(
                sample_id,
                report.label_set_id,
                Choices(values=project.schema.decode_target(sample.results)),
                source="import",
            )
            report.annotated += 1
        else:
            # No row at all: nobody has looked at it, which is not the same
            # as having looked and found nothing
            report.unlabelled += 1

    return report


def _label_set(catalog: Catalog, name: str, schema: ClassificationSchema) -> int:
    try:
        label_set_id, existing = catalog.label_set(name)
    except CatalogError:
        return catalog.create_label_set(name, schema)
    if existing.classes != schema.classes:
        # Re-running after adding a class should widen the label set rather
        # than annotate against a list that no longer matches
        catalog.set_classes(label_set_id, schema)
    return label_set_id


def describe(report: Report, catalog_root: Path) -> str:
    lines = [
        f"Label set '{report.label_set}' with {len(report.classes)} class(es): "
        f"{', '.join(report.classes) or 'none'}",
        f"{report.ingested} sample(s) into {catalog_root}",
        f"  {report.annotated} annotated, {report.skipped} skipped, "
        f"{report.unlabelled} never reviewed",
    ]
    if report.missing:
        shown = ", ".join(report.missing[:5])
        more = f" (+{len(report.missing) - 5} more)" if len(report.missing) > 5 else ""
        lines.append(f"  {len(report.missing)} file(s) missing and left behind: {shown}{more}")
    return "\n".join(lines)
