"""Migrating a project's dataset.json into a catalog.

The bridge that gets existing annotation across, so what matters is that
nothing is lost, nothing is invented, and running it twice is harmless.
"""

import json

import pytest

from strata.catalog import Catalog
from strata.labeller.dataset import Sample, save_dataset
from strata.labeller.to_catalog import (
    MigrationError,
    describe,
    group_id_for,
    migrate,
    schema_for,
)
from strata.labels import Choices


@pytest.fixture
def catalog(tmp_path) -> Catalog:
    return Catalog.local(tmp_path / "catalog")


@pytest.fixture
def populated(project):
    """A project with files on disk and a dataset describing them."""

    def _make(annotated=2, skipped=1, unlabelled=1, folder: str = "", classes=("cat", "dog")):
        samples = []
        total = annotated + skipped + unlabelled
        for i in range(total):
            relative = f"{folder}/img{i:03d}.jpg" if folder else f"img{i:03d}.jpg"
            path = project.data_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"image {i}".encode())
            if i < annotated:
                samples.append(
                    Sample(
                        path=relative,
                        results=project.schema.encode_target([classes[i % len(classes)]]),
                        annotated=True,
                    )
                )
            elif i < annotated + skipped:
                samples.append(Sample(path=relative, skipped=True))
            else:
                samples.append(Sample(path=relative))
        save_dataset(samples, project.dataset_path)
        return samples

    return _make


# ----------------------------------------------------------------------
# Schema conversion
# ----------------------------------------------------------------------


def test_the_schema_loses_its_media(project):
    # image_classification and text_classification were template names; what
    # a sample is made of belongs to the catalog now
    assert schema_for(project).task == "classification"


def test_declared_classes_carry_over(project):
    assert schema_for(project).classes == ["cat", "dog"]


def test_multiple_choice_carries_over(project):
    assert schema_for(project).multiple is True


def test_single_choice_carries_over(make_project):
    single = make_project("single", choice="single")
    assert schema_for(single).multiple is False


def test_classes_are_inferred_when_the_project_never_pinned_them(make_project, populated):
    # A project could leave [label_config] classes empty and infer from use.
    # The catalog has no such inference, so the list stops being implicit here
    loose = make_project("loose", classes=[])
    toml = loose.root / "project.toml"
    toml.write_text(toml.read_text().replace("classes = []", "classes = []"))
    save_dataset(
        [
            Sample(path="a.jpg", results=loose.schema.encode_target(["fox"]), annotated=True),
            Sample(path="b.jpg", results=loose.schema.encode_target(["owl"]), annotated=True),
        ],
        loose.dataset_path,
    )
    assert schema_for(loose).classes == ["fox", "owl"]


def test_a_task_with_no_equivalent_is_refused(make_project):
    boxes = make_project("boxes", template="image_bbox")
    with pytest.raises(MigrationError, match="no equivalent"):
        schema_for(boxes)


# ----------------------------------------------------------------------
# Grouping
# ----------------------------------------------------------------------


def test_plain_images_get_no_group(project):
    assert group_id_for(project, "batch/img001.jpg") is None


def test_frames_group_on_their_folder(make_project):
    frames = make_project("frames")
    toml = frames.root / "project.toml"
    toml.write_text(toml.read_text().replace('kind = "images"', 'kind = "frames"'))
    from strata.labeller.project import Project

    frames = Project.load(frames.root)
    assert group_id_for(frames, "vid1/frame0007.jpg") == "vid1"


# ----------------------------------------------------------------------
# Migration
# ----------------------------------------------------------------------


def test_every_file_reaches_the_catalog(project, catalog, populated):
    populated(annotated=3, skipped=1, unlabelled=2)
    report = migrate(project, catalog)
    assert report.ingested == 6


def test_annotations_survive_the_crossing(project, catalog, populated):
    populated(annotated=2, skipped=0, unlabelled=0)
    report = migrate(project, catalog)
    labelled = catalog.labelled(report.label_set_id)
    assert len(labelled) == 2
    assert catalog.annotation_of(labelled[0].id, report.label_set_id) == Choices(values=["cat"])


def test_skipped_samples_stay_skipped(project, catalog, populated):
    populated(annotated=1, skipped=2, unlabelled=0)
    report = migrate(project, catalog)
    assert report.skipped == 2
    # Reviewed with nothing applicable: not training data, and not in the queue
    assert len(catalog.labelled(report.label_set_id)) == 1
    assert catalog.unlabelled(report.label_set_id) == []


def test_unlabelled_samples_get_no_annotation_row(project, catalog, populated):
    populated(annotated=1, skipped=0, unlabelled=3)
    report = migrate(project, catalog)
    # Nobody has looked at them, which is not the same as having looked and
    # found nothing
    assert len(catalog.unlabelled(report.label_set_id)) == 3


def test_an_empty_annotation_stays_a_real_answer(project, catalog):
    path = project.data_dir / "empty.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"nothing in this one")
    save_dataset([Sample(path="empty.jpg", results=[], annotated=True)], project.dataset_path)

    report = migrate(project, catalog)
    [sample] = catalog.labelled(report.label_set_id)
    assert catalog.annotation_of(sample.id, report.label_set_id) == Choices()


def test_the_class_index_is_built(project, catalog, populated):
    populated(annotated=4, skipped=0, unlabelled=0)
    report = migrate(project, catalog)
    # "every sample labelled X" has to be a join from the moment data lands
    assert len(catalog.with_class(report.label_set_id, "cat")) == 2


def test_migrated_annotations_are_marked_as_imported(project, catalog, populated):
    populated(annotated=1, skipped=0, unlabelled=0)
    migrate(project, catalog)
    from sqlalchemy import select

    from strata.catalog import tables as t

    with catalog.engine.connect() as conn:
        source = conn.execute(select(t.annotation.c.source)).scalar_one()
    assert source == "import"


def test_the_label_set_is_named_for_the_project(project, catalog, populated):
    populated()
    assert migrate(project, catalog).label_set == project.name


def test_the_label_set_name_can_be_overridden(project, catalog, populated):
    populated()
    assert migrate(project, catalog, label_set="presence").label_set == "presence"


def test_a_missing_file_is_reported_not_fatal(project, catalog, populated):
    populated(annotated=2, skipped=0, unlabelled=0)
    (project.data_dir / "img000.jpg").unlink()

    report = migrate(project, catalog)
    # A dataset outlives the files it points at; one moved image must not
    # cost the whole migration
    assert report.missing == ["img000.jpg"]
    assert report.ingested == 1


def test_running_twice_changes_nothing(project, catalog, populated):
    populated(annotated=3, skipped=1, unlabelled=2)
    first = migrate(project, catalog)
    second = migrate(project, catalog)

    assert (second.ingested, second.annotated, second.skipped) == (
        first.ingested,
        first.annotated,
        first.skipped,
    )
    assert len(catalog.labelled(first.label_set_id)) == 3


def test_a_second_run_widens_the_label_set(project, catalog, populated):
    populated(annotated=2, skipped=0, unlabelled=0)
    report = migrate(project, catalog)
    project.add_classes(["bird"])

    from strata.labeller.project import Project

    widened = Project.load(project.root)
    migrate(widened, catalog)
    _, schema = catalog.label_set(report.label_set)
    assert schema.classes == ["cat", "dog", "bird"]


def test_the_project_is_left_alone(project, catalog, populated):
    populated(annotated=2, skipped=1, unlabelled=1)
    before = project.dataset_path.read_text()
    migrate(project, catalog)
    assert project.dataset_path.read_text() == before


# ----------------------------------------------------------------------
# Dry run
# ----------------------------------------------------------------------


def test_a_dry_run_counts_without_a_catalog(project, populated):
    populated(annotated=3, skipped=1, unlabelled=2)
    report = migrate(project, None, dry_run=True)
    assert (report.ingested, report.annotated, report.skipped, report.unlabelled) == (6, 3, 1, 2)


def test_a_dry_run_writes_nothing(project, populated, tmp_path):
    populated()
    migrate(project, None, dry_run=True)
    assert not (tmp_path / "catalog").exists()


def test_a_dry_run_still_reports_missing_files(project, populated):
    populated(annotated=2, skipped=0, unlabelled=0)
    (project.data_dir / "img000.jpg").unlink()
    assert migrate(project, None, dry_run=True).missing == ["img000.jpg"]


def test_the_summary_names_what_happened(project, catalog, populated, tmp_path):
    populated(annotated=2, skipped=1, unlabelled=1)
    text = describe(migrate(project, catalog), tmp_path / "catalog")
    assert "2 annotated" in text
    assert "1 skipped" in text


def test_the_summary_mentions_missing_files(project, catalog, populated, tmp_path):
    populated(annotated=2, skipped=0, unlabelled=0)
    (project.data_dir / "img000.jpg").unlink()
    assert "missing" in describe(migrate(project, catalog), tmp_path / "catalog")


# ----------------------------------------------------------------------
# The v1 format still on disk in old projects
# ----------------------------------------------------------------------


def test_a_v1_dataset_migrates_through_the_upgrade(project, catalog):
    path = project.data_dir / "old.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"an old image")
    project.dataset_path.write_text(json.dumps([{"path": "old.jpg", "labels": ["dog"]}]))

    report = migrate(project, catalog)
    [sample] = catalog.labelled(report.label_set_id)
    assert catalog.annotation_of(sample.id, report.label_set_id) == Choices(values=["dog"])


# ----------------------------------------------------------------------
# Keeping the label set in step with the project
# ----------------------------------------------------------------------


def test_adding_a_class_widens_the_label_set(project, catalog, populated, tmp_path):
    from strata.labeller.cli import _add_to_label_set
    from strata.labeller.config import CatalogConfig, Settings

    populated(annotated=2, skipped=0, unlabelled=0)
    report = migrate(project, catalog)
    settings = Settings(catalog=CatalogConfig(root=str(tmp_path / "catalog")))

    classes = project.add_classes(["bird"])
    _add_to_label_set(project, settings, classes)

    # The labeling config comes from project.toml but an export is validated
    # against the label set; a class in one and not the other lets a reviewer
    # apply a label the catalog then refuses
    _, schema = catalog.label_set(report.label_set)
    assert schema.classes == ["cat", "dog", "bird"]


def test_widening_is_append_only(project, catalog, populated, tmp_path):
    from strata.labeller.cli import _add_to_label_set
    from strata.labeller.config import CatalogConfig, Settings

    populated(annotated=1, skipped=0, unlabelled=0)
    report = migrate(project, catalog)
    settings = Settings(catalog=CatalogConfig(root=str(tmp_path / "catalog")))

    _add_to_label_set(project, settings, project.add_classes(["bird"]))
    _, schema = catalog.label_set(report.label_set)
    # Order is data: a checkpoint maps output neurons to it by position
    assert schema.classes[:2] == ["cat", "dog"]


def test_a_class_already_in_the_label_set_is_a_no_op(project, catalog, populated, tmp_path):
    from strata.labeller.cli import _add_to_label_set
    from strata.labeller.config import CatalogConfig, Settings

    populated(annotated=1, skipped=0, unlabelled=0)
    report = migrate(project, catalog)
    settings = Settings(catalog=CatalogConfig(root=str(tmp_path / "catalog")))

    _add_to_label_set(project, settings, ["cat", "dog"])
    _, schema = catalog.label_set(report.label_set)
    assert schema.classes == ["cat", "dog"]


def test_no_catalog_is_not_an_error(project, tmp_path):
    from strata.labeller.cli import _add_to_label_set
    from strata.labeller.config import CatalogConfig, Settings

    # A project can be used before anything is migrated; to-catalog will
    # create the label set from project.toml when it runs
    settings = Settings(catalog=CatalogConfig(root=str(tmp_path / "nothing-here")))
    _add_to_label_set(project, settings, ["cat", "dog", "bird"])


def test_the_source_path_is_recorded(project, catalog, populated):
    # A blob is addressed by its content, so without this there is no way
    # back from a catalogued sample to the file it was read from
    populated(annotated=2, skipped=0, unlabelled=1)
    report = migrate(project, catalog)
    sources = {
        (s.metadata or {}).get("source_path") for s in catalog.unlabelled(report.label_set_id)
    } | {(s.metadata or {}).get("source_path") for s in catalog.labelled(report.label_set_id)}
    assert sources == {"img000.jpg", "img001.jpg", "img002.jpg"}


def test_re_running_backfills_a_missing_source_path(project, catalog, populated):
    populated(annotated=2, skipped=0, unlabelled=0)
    paths = sorted(project.data_dir.glob("*.jpg"))
    catalog.ingest(paths, media="image")  # as an older build would have
    assert all(s.metadata is None for s in catalog.unlabelled(
        catalog.create_label_set("tmp", __import__(
            "strata.labels", fromlist=["ClassificationSchema"]).ClassificationSchema())))

    report = migrate(project, catalog)
    assert all(
        (s.metadata or {}).get("source_path")
        for s in catalog.labelled(report.label_set_id)
    )


def test_a_frame_keeps_its_folder_in_the_source_path(project, catalog, populated):
    populated(annotated=2, skipped=0, unlabelled=0, folder="vid1")
    report = migrate(project, catalog)
    assert all(
        (s.metadata or {}).get("source_path", "").startswith("vid1/")
        for s in catalog.labelled(report.label_set_id)
    )
