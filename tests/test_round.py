"""A training round on the catalog.

What matters is that the three packages agree: the dataset the catalog froze
is the one the model was handed, versions accumulate rather than overwrite,
and a round continues the previous one unless told not to.
"""

import json
import shutil

import pytest
from toy_model import TOY_SOURCE

from strata.catalog import Catalog, CatalogError
from strata.labeller.round import RoundError, describe, run_round


def stock(project, catalog, labelled: int, total: int):
    """Ingest files and answer some, the way a project fills a catalog."""
    from strata.labels import Choices

    paths = []
    for i in range(total):
        path = project.data_dir / f"img{i:03d}.jpg"
        path.write_bytes(f"image {i}".encode())
        paths.append(path)

    ids = catalog.ingest(
        paths,
        media="image",
        collections=project.collections,
        metadata_for=lambda p: {"source_path": p.name},
    )
    try:
        label_set_id, _ = catalog.label_sets.get(project.label_set_name)
    except CatalogError:
        label_set_id = catalog.label_sets.create(
            project.label_set_name, project.schema.catalog_schema()
        )
    catalog.annotations.annotate_many(
        label_set_id,
        [
            (sample_id, Choices(values=["cat" if i % 2 else "dog"]))
            for i, sample_id in enumerate(ids[:labelled])
        ],
    )
    return label_set_id


@pytest.fixture
def ready(project, tmp_path):
    """A project with a stocked catalog and a model it can reach."""

    def _make(labelled: int = 16, total: int = 20, ref: str = "toy.py:Toy"):
        shutil.copyfile(TOY_SOURCE, project.root / "toy.py")

        toml = project.root / "project.toml"
        toml.write_text(toml.read_text().replace('ref = "multilabel"', f'ref = "{ref}"'))

        from strata.labeller.project import LabellingProject

        reloaded = LabellingProject.load(project.root)
        catalog = Catalog.local(tmp_path / "catalog")
        stock(reloaded, catalog, labelled, total)
        return reloaded, catalog

    return _make


# ----------------------------------------------------------------------
# A round
# ----------------------------------------------------------------------


def test_a_round_trains_and_records_a_run(ready):
    project, catalog = ready()
    result = run_round(project, catalog)
    assert result.run.id
    assert result.run.checkpoint.exists()


def test_the_run_names_the_dataset_it_used(ready):
    project, catalog = ready()
    result = run_round(project, catalog)
    assert (result.run.dataset, result.run.dataset_version) == (project.dataset_name, 1)


def test_only_labelled_samples_are_trained_on(ready):
    project, catalog = ready(labelled=12, total=20)
    result = run_round(project, catalog)
    assert result.run.metrics["n_train"] + result.run.metrics["n_val"] == 12


def test_the_split_the_catalog_decided_is_what_the_model_got(ready):
    project, catalog = ready()
    result = run_round(project, catalog)
    assert result.run.metrics["n_train"] == len(result.manifest.train)
    assert result.run.metrics["n_val"] == len(result.manifest.val)


def test_model_params_reach_the_model(ready):
    project, catalog = ready()
    toml = project.root / "project.toml"
    toml.write_text(toml.read_text().replace("num_epochs = 4", 'note = "from params"'))
    from strata.labeller.project import LabellingProject

    result = run_round(LabellingProject.load(project.root), catalog)
    assert json.loads(result.run.checkpoint.read_text())["note"] == "from params"


def test_a_registered_name_works_as_a_ref(ready, monkeypatch):
    project, _catalog = ready(ref="presence")
    pytest.importorskip("timm", reason="needs the image extra")
    # Only that resolution reaches a real baseline; training one is the
    # conformance suite's job
    assert project.model.ref == "presence"


# ----------------------------------------------------------------------
# Versions
# ----------------------------------------------------------------------


def test_a_round_over_the_same_data_reuses_the_version(ready):
    # Two rounds with nothing labelled between them are two attempts at one
    # selection, and a crashed attempt must not burn a version number
    project, catalog = ready()
    first = run_round(project, catalog)
    second = run_round(project, catalog)
    assert first.manifest.version == second.manifest.version == 1


def test_labelling_more_freezes_a_new_version(ready):
    project, catalog = ready(labelled=12, total=20)
    first = run_round(project, catalog)

    # Everything answered, so membership changes and a new version is due
    stock(project, catalog, labelled=20, total=20)

    assert run_round(project, catalog).manifest.version == first.manifest.version + 1


def test_versions_are_written_side_by_side(ready):
    project, catalog = ready(labelled=12, total=20)
    run_round(project, catalog)

    # Everything answered, so membership changes and a new version is due
    stock(project, catalog, labelled=20, total=20)
    run_round(project, catalog)

    versions = sorted(p.name for p in (project.datasets_dir / project.dataset_name).iterdir())
    assert versions == ["v001", "v002"]


def test_a_version_directory_is_never_overwritten(ready):
    project, catalog = ready()
    result = run_round(project, catalog)
    # The manifest inside is what a run points at; silently replacing it
    # would make an earlier run's lineage a lie
    assert (result.dataset_dir / "manifest.json").exists()
    assert not (project.datasets_dir / project.dataset_name / "pending").exists()


def test_the_split_survives_a_second_round(ready):
    project, catalog = ready(labelled=12, total=20)
    first = run_round(project, catalog)
    before = {s.id: s.split for s in first.manifest.samples}

    # label the rest and go again
    stock(project, catalog, labelled=20, total=20)

    second = run_round(project, catalog)
    after = {s.id: s.split for s in second.manifest.samples}
    assert all(after[i] == before[i] for i in before)


# ----------------------------------------------------------------------
# Warm starting
# ----------------------------------------------------------------------


def test_a_second_round_continues_the_first(ready):
    project, catalog = ready()
    first = run_round(project, catalog)
    second = run_round(project, catalog)
    assert second.run.parent_run_id == first.run.id
    assert second.warm_started


def test_the_first_round_is_cold(ready):
    project, catalog = ready()
    assert not run_round(project, catalog).warm_started


def test_fresh_refuses_to_continue(ready):
    project, catalog = ready()
    run_round(project, catalog)
    assert not run_round(project, catalog, fresh=True).warm_started


# ----------------------------------------------------------------------
# Refusals
# ----------------------------------------------------------------------


def test_a_missing_label_set_says_what_to_run(project, tmp_path):
    catalog = Catalog.local(tmp_path / "catalog")
    with pytest.raises(RoundError, match="ingest"):
        run_round(project, catalog)


def test_nothing_labelled_is_an_error(ready):
    project, catalog = ready(labelled=0, total=5)
    with pytest.raises(RoundError, match="Nothing is labelled"):
        run_round(project, catalog)


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------


def test_the_summary_names_the_dataset_version(ready):
    project, catalog = ready()
    assert "v1" in "\n".join(describe(run_round(project, catalog)))


def test_the_summary_says_when_a_round_is_cold(ready):
    project, catalog = ready()
    assert "cold" in "\n".join(describe(run_round(project, catalog)))


def test_the_summary_names_the_run_it_continued(ready):
    project, catalog = ready()
    first = run_round(project, catalog)
    text = "\n".join(describe(run_round(project, catalog)))
    assert f"continuing run {first.run.id}" in text


def test_an_unreachable_ratio_is_called_out(project, tmp_path):
    """Two videos cannot hold out 20% of themselves.

    A val figure read without knowing that is misleading, so the round says
    what grouping actually allowed.
    """
    from strata.labels import Choices

    shutil.copyfile(TOY_SOURCE, project.root / "toy.py")
    toml = project.root / "project.toml"
    toml.write_text(
        toml.read_text()
        .replace('ref = "multilabel"', 'ref = "toy.py:Toy"')
        .replace('type = "image"', 'type = "frames"')
        # The project asks for it; the catalog enforces no grouping on its own
        .replace('# group_by = "video"', 'group_by = "video"')
    )
    from strata.labeller.project import LabellingProject

    reloaded = LabellingProject.load(project.root)
    catalog = Catalog.local(tmp_path / "catalog")

    # Two folders, ten frames each: a group is indivisible, so the split can
    # only ever be half and half
    ids = []
    for video in range(2):
        paths = []
        for i in range(video * 10, video * 10 + 10):
            path = project.data_dir / f"vid{video}" / f"f{i:03d}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"image {i}".encode())
            paths.append(path)
        ids += catalog.ingest(
            paths,
            media="image",
            subtype="frames",
            metadata={"video": f"vid{video}"},
            collections=reloaded.collections,
            metadata_for=lambda p: {"source_path": p.name},
        )
    label_set_id = catalog.label_sets.create(
        reloaded.label_set_name, reloaded.schema.catalog_schema()
    )
    catalog.annotations.annotate_many(label_set_id, [(i, Choices(values=["cat"])) for i in ids])

    assert "not the 20% asked for" in "\n".join(describe(run_round(reloaded, catalog)))


def with_fresh_params(project, block: str):
    """Replace the project's [model.fresh_params] with this and reload.

    Replaced rather than appended: the scaffold now writes one, and a second
    declaration of the same table is a TOML error rather than an override.
    """
    import re

    from strata.labeller.project import LabellingProject

    toml = project.root / "project.toml"
    text = re.sub(r"\[model\.fresh_params\]\n(?:[^\[]*\n)?", "", toml.read_text())
    toml.write_text(text.rstrip("\n") + f"\n\n[model.fresh_params]\n{block}\n")
    return LabellingProject.load(project.root)


def test_a_cold_round_takes_the_fresh_params(ready):
    project, catalog = ready()
    # A run with nothing to inherit has to learn from scratch; the settings
    # that suit an increment undertrain it
    reloaded = with_fresh_params(project, 'note = "cold"')
    assert run_round(reloaded, catalog, fresh=True).run.params["note"] == "cold"


def test_a_warm_round_ignores_them(ready):
    project, catalog = ready()
    reloaded = with_fresh_params(project, 'note = "cold"')
    # A first round has nothing to inherit, so it is cold whatever was
    # asked for. Only the second is genuinely warm.
    run_round(reloaded, catalog)
    second = run_round(reloaded, catalog)

    assert second.warm_started
    # Not overridden and not defaulted-in: params record what the project
    # asked for, and a warm round asked for nothing extra
    assert "note" not in second.run.params


def test_a_first_round_is_cold_however_it_was_asked_for(ready):
    project, catalog = ready()
    reloaded = with_fresh_params(project, 'note = "cold"')
    # --fresh is a request; being cold is an outcome. They part company when
    # nothing has trained on this dataset yet, and a cold run trained on an
    # increment's settings is undertrained — which is what made two early
    # baselines not baselines.
    first = run_round(reloaded, catalog)

    assert not first.warm_started
    assert first.run.params["note"] == "cold"


def test_fresh_params_only_override_what_they_name(ready):
    project, catalog = ready()
    reloaded = with_fresh_params(project, "num_epochs = 12")
    params = run_round(reloaded, catalog, fresh=True).run.params
    assert params["num_epochs"] == 12
    assert params["batch_size"] == 16


def test_a_retry_does_not_refetch_a_version_it_already_has(project, monkeypatch):
    """The OOM case: training died, the dataset directory survived.

    Materialising into staging and then noticing the version was already
    there cost nothing when blobs were local files a hard link away. Once
    they are tar members in a bucket it is minutes and gigabytes, thrown
    away on arrival.

    The rule itself — when a version on disk may be reused — is tested where
    it lives, in the catalog; this only checks that the stage a round runs
    goes through it.
    """
    from strata.catalog.stages import Context, MaterialiseRequest, materialise
    from strata.labels import MANIFEST_FORMAT, MANIFEST_NAME, ClassificationSchema, Manifest

    version_dir = project.datasets_dir / project.dataset_name / "v002"
    (version_dir / "files").mkdir(parents=True)
    manifest_on_disk = Manifest(
        format=MANIFEST_FORMAT,
        dataset=project.dataset_name,
        version=2,
        catalog_id="20260101T000000-aaaaaaaa",
        label_set=project.label_set_name,
        label_schema=ClassificationSchema(classes=["cat"]),
    )
    (version_dir / MANIFEST_NAME).write_text(manifest_on_disk.model_dump_json())

    class Datasets:
        def named(self, dataset_id):
            from strata.catalog import DatasetRef

            return DatasetRef(project.dataset_name, 2, None)

    class Refuses:
        id = "20260101T000000-aaaaaaaa"
        datasets = Datasets()

        def materialise(self, *args, **kwargs):
            raise AssertionError("refetched a version already on disk")

    built = materialise(MaterialiseRequest(dataset_id=7), Context(Refuses(), project.datasets_dir))
    assert built.directory == version_dir
    assert built.version == 2
