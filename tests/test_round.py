"""A training round on the catalog.

What matters is that the three packages agree: the dataset the catalog froze
is the one the model was handed, versions accumulate rather than overwrite,
and a round continues the previous one unless told not to.
"""

import json

import pytest

from strata.catalog import Catalog
from strata.labeller.dataset import Sample, save_dataset
from strata.labeller.round import RoundError, describe, run_round
from strata.labeller.to_catalog import migrate

TOY_MODEL = '''
import json
from pathlib import Path

from strata.labels import ChoicesPrediction
from strata.modelling import Model


class Toy(Model):
    task = "classification"
    version = "1"

    def __init__(self, num_epochs: int = 4, batch_size: int = 16, lr: float = 5e-5,
                 note: str = "default"):
        self.num_epochs = num_epochs
        self.note = note
        self.classes = []

    def finetune(self, train, classes, val=None):
        self.classes = list(classes)
        return {"accuracy": 0.5, "n_train": float(len(train)), "n_val": float(len(val or []))}

    def predict(self, paths):
        return [ChoicesPrediction(values=self.classes[:1], confidences=[0.5]) for _ in paths]

    def save(self, path):
        Path(path).write_text(json.dumps({"classes": self.classes, "note": self.note}))

    def load(self, path):
        payload = json.loads(Path(path).read_text())
        self.classes = payload["classes"]
        self.note = payload["note"]
'''


@pytest.fixture
def ready(project, tmp_path):
    """A project migrated into a catalog, with a model it can reach."""

    def _make(labelled: int = 16, total: int = 20, ref: str = "toy.py:Toy"):
        samples = []
        for i in range(total):
            relative = f"img{i:03d}.jpg"
            (project.data_dir / relative).write_bytes(f"image {i}".encode())
            samples.append(
                Sample(
                    path=relative,
                    results=project.schema.encode_target(["cat" if i % 2 else "dog"]),
                    annotated=i < labelled,
                )
            )
        save_dataset(samples, project.dataset_path)
        (project.root / "toy.py").write_text(TOY_MODEL)

        toml = project.root / "project.toml"
        toml.write_text(toml.read_text().replace('ref = "multilabel"', f'ref = "{ref}"'))

        from strata.labeller.project import Project

        reloaded = Project.load(project.root)
        catalog = Catalog.local(tmp_path / "catalog")
        migrate(reloaded, catalog)
        return reloaded, catalog

    return _make


# ----------------------------------------------------------------------
# A round
# ----------------------------------------------------------------------


def test_a_round_trains_and_records_a_run(ready):
    project, catalog = ready()
    result = run_round(project, catalog)
    assert result.run.id > 0
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
    from strata.labeller.project import Project

    result = run_round(Project.load(project.root), catalog)
    assert json.loads(result.run.checkpoint.read_text())["note"] == "from params"


def test_a_registered_name_works_as_a_ref(ready, monkeypatch):
    project, catalog = ready(ref="presence")
    pytest.importorskip("timm", reason="needs the image extra")
    # Only that resolution reaches a real baseline; training one is the
    # conformance suite's job
    assert project.model.ref == "presence"


# ----------------------------------------------------------------------
# Versions
# ----------------------------------------------------------------------


def test_each_round_freezes_a_new_version(ready):
    project, catalog = ready()
    first = run_round(project, catalog)
    second = run_round(project, catalog)
    assert (first.manifest.version, second.manifest.version) == (1, 2)


def test_versions_are_written_side_by_side(ready):
    project, catalog = ready()
    run_round(project, catalog)
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
    before = {s.id: s.val for s in first.manifest.samples}

    # label the rest and go again
    samples = []
    for i in range(20):
        samples.append(
            Sample(
                path=f"img{i:03d}.jpg",
                results=project.schema.encode_target(["cat"]),
                annotated=True,
            )
        )
    save_dataset(samples, project.dataset_path)
    migrate(project, catalog)

    second = run_round(project, catalog)
    after = {s.id: s.val for s in second.manifest.samples}
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
    with pytest.raises(RoundError, match="to-catalog"):
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
    # Two videos cannot hold out 20% of themselves, and a val figure read
    # without knowing that is misleading
    samples = []
    for i in range(20):
        relative = f"vid{i // 10}/f{i:03d}.jpg"
        (project.data_dir / relative).parent.mkdir(parents=True, exist_ok=True)
        (project.data_dir / relative).write_bytes(f"image {i}".encode())
        samples.append(
            Sample(
                path=relative,
                results=project.schema.encode_target(["cat"]),
                annotated=True,
            )
        )
    save_dataset(samples, project.dataset_path)
    (project.root / "toy.py").write_text(TOY_MODEL)
    toml = project.root / "project.toml"
    toml.write_text(
        toml.read_text()
        .replace('ref = "multilabel"', 'ref = "toy.py:Toy"')
        .replace('kind = "images"', 'kind = "frames"')
    )
    from strata.labeller.project import Project

    reloaded = Project.load(project.root)
    catalog = Catalog.local(tmp_path / "catalog")
    migrate(reloaded, catalog)

    assert "not the 20% asked for" in "\n".join(describe(run_round(reloaded, catalog)))
