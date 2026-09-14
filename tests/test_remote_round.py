"""A remote round, laptop to host, in one process.

The laptop freezes a dataset and asks; the host checks what it was asked,
materialises and trains. Here the laptop's submission is handed straight to
the host's round function, so what the host builds is exactly what the
laptop's request makes it build — which is where remote rounds used to lose
the project's features.
"""

import pytest

from strata.catalog import Catalog
from strata.labels import MANIFEST_NAME, Choices, Manifest

SPECIES = {"name": "species", "source": "metadata", "ref": "species"}


@pytest.fixture
def featured(project, tmp_path):
    """A project that declares a feature, over a catalog that holds it."""
    from strata.labeller.project import LabellingProject

    toml = project.root / "project.toml"
    toml.write_text(
        toml.read_text()
        + '\n[[data.features]]\nname = "species"\nsource = "metadata"\nref = "species"\n'
    )
    project = LabellingProject.load(project.root)

    catalog = Catalog.local(tmp_path / "catalog")
    paths = []
    for i in range(8):
        path = project.data_dir / f"img{i:03d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"image {i}".encode())
        paths.append(path)
    ids = catalog.ingest(
        paths,
        media="image",
        collections=project.collections,
        metadata_for=lambda p: {"species": f"sp-{p.stem}"},
    )
    label_set = catalog.label_sets.create(
        project.label_set_name, project.schema.catalog_schema()
    )
    catalog.annotations.annotate_many(label_set, [(i, Choices(values=["cat"])) for i in ids])
    return project, catalog


@pytest.fixture
def host(featured, tmp_path, monkeypatch):
    """The laptop's submission, run by the host's round function against the same catalog.

    Training itself is stood in for: it is covered where it lives, and what
    matters here is the directory the host would have trained from.
    """
    import strata.modelling.remote.rounds as rounds
    from strata.modelling import RunStore
    from strata.modelling.remote.client import Trainer
    from strata.modelling.requests import Run

    _, catalog = featured
    seen = {}

    def train(request, store, on_epoch=None):
        seen["dataset_dir"] = request.dataset_dir
        return store.record(
            Run(
                id="",
                dataset="demo",
                label_set="demo",
                model=request.model,
                model_version="1",
                classes=["cat"],
            ),
            {},
        )

    def submit(self, request):
        seen["request"] = request
        rounds.run_round(
            request, catalog, RunStore.local(tmp_path / "host-runs"), tmp_path / "host-datasets"
        )
        return {"id": "job"}

    monkeypatch.setattr(rounds, "run_train", train)
    monkeypatch.setattr(Trainer, "submit", submit)
    monkeypatch.setattr(
        Trainer, "served_catalog", lambda self: {"name": "default", "id": catalog.id}
    )
    finished = {
        "state": "done",
        "result": {
            "run": {
                "id": "r",
                "dataset": "demo",
                "label_set": "demo",
                "model": "multilabel",
                "model_version": "1",
                "classes": ["cat"],
            },
            "metrics": {},
        },
    }
    monkeypatch.setattr(Trainer, "follow", lambda self, job_id, on_state=None: finished)
    return seen


def _remote_round(project, catalog):
    from strata.labeller.cli import train as train_command
    from strata.labeller.config import ModellingConfig, Settings

    settings = Settings(modelling=ModellingConfig(url="http://gpu:8082", token="t"))
    train_command._remote_round(project, catalog, settings, fresh=False, val_ratio=0.25)


def test_a_remote_round_trains_on_the_features_the_project_declares(featured, host):
    project, catalog = featured

    _remote_round(project, catalog)

    manifest = Manifest.model_validate_json(
        (host["dataset_dir"] / MANIFEST_NAME).read_text()
    )
    assert manifest.features == [SPECIES]
    assert {s.features["species"] for s in manifest.samples} == {
        f"sp-img{i:03d}" for i in range(8)
    }


def test_a_host_on_another_catalog_is_refused_before_anything_is_frozen(
    featured, host, monkeypatch, capsys
):
    import typer
    from sqlalchemy import func, select

    from strata.catalog.index import tables as t
    from strata.modelling.remote.client import Trainer

    project, catalog = featured
    monkeypatch.setattr(
        Trainer, "served_catalog", lambda self: {"name": "main", "id": "20250101T000000-cccccccc"}
    )

    with pytest.raises(typer.Exit):
        _remote_round(project, catalog)

    assert "request" not in host
    # No dataset version left behind for a round that never ran
    with catalog.engine.connect() as conn:
        assert conn.execute(select(func.count()).select_from(t.dataset)).scalar() == 0
    # This catalog's index is SQLite, which the host could only read by
    # running here — the one case where repointing the host is not the fix
    assert "SQLite" in capsys.readouterr().out


def test_the_request_says_what_its_dataset_id_means(featured, host):
    project, catalog = featured

    _remote_round(project, catalog)

    request = host["request"]
    ref = catalog.datasets.named(request.dataset_id)
    assert (request.dataset_name, request.dataset_version) == (ref.name, ref.version)
    assert request.annotation_digest == ref.annotation_digest
    assert request.catalog_id == catalog.id
