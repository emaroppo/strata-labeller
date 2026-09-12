"""A whole push, against a Label Studio that lives in a dict.

The part of the loop that has broken most and been tested least. Everything
here goes through the command, because the faults were all in the wiring
between pieces that each worked: a cache and a handler disagreeing about a
type, a run passed as one shape and read as another, an import left behind
by a module that moved.

Predictions are seeded into the cache rather than computed, so no model and
no framework are needed — what is under test is the ranking, the tasks and
the pre-annotations, not inference.
"""

import json

import pytest
from fake_label_studio import FakeLabelStudio
from typer.testing import CliRunner

from strata.labeller.cli import app
from strata.labels import Choices, ChoicesPrediction
from strata.modelling import PredictionCache, RunStore
from strata.modelling.requests import Run

runner = CliRunner()

#: Chosen so the tests can fail. Not in descending order, because with the
#: highest first `confidences[0]` and `max(confidences)` are the same number
#: and nothing can tell the score a reviewer is shown from an arbitrary one.
#: And the strategies disagree on which is worst:
#:
#:   least-confident (1 - max)      vague 0.60 > torn 0.50 > confident 0.05
#:   margin (1 - gap of top two)    torn 0.99 > vague 0.65 > confident 0.07
CONFIDENCES = {
    "confident": [0.02, 0.95],
    "torn": [0.49, 0.50],
    "vague": [0.05, 0.40],
}


@pytest.fixture
def stage(make_project, tmp_path, monkeypatch):
    """A project, a catalog with three unlabelled samples, and a recorded run."""
    from strata.catalog import Catalog
    from strata.labels import ClassificationSchema

    project = make_project("demo", classes=["cat", "dog"])
    catalog_root = tmp_path / "catalog"
    catalog = Catalog.local(catalog_root)

    raw = tmp_path / "raw"
    raw.mkdir()
    paths = []
    for name in CONFIDENCES:
        path = raw / f"{name}.jpg"
        path.write_bytes(name.encode())
        paths.append(path)
    ids = catalog.ingest(
        paths, media="image", collections=["demo"],
        metadata_for=lambda p: {"source_path": str(p)},
    )
    label_set_id = catalog.label_sets.create(
        "demo", ClassificationSchema(classes=["cat", "dog"])
    )

    # A run with a checkpoint on disk, so push has something to predict with
    store = RunStore.local(project.runs_dir)
    run = store.record(
        Run(
            id="", dataset="demo", dataset_version=1, label_set="demo",
            model="multilabel", model_version="1", classes=["cat", "dog"],
        ),
        {"val_accuracy": 0.9},
    )
    store.checkpoint_path(run.id).write_bytes(b"weights")
    with store.engine.begin() as conn:
        from sqlalchemy import update

        from strata.modelling import tables as t

        conn.execute(
            update(t.run).where(t.run.c.id == run.id)
            .values(checkpoint=str(store.checkpoint_path(run.id)))
        )

    # Seeded, so nothing here needs a model
    cache = PredictionCache.local(project.runs_dir)
    by_name = {}
    for sample_id, path in zip(ids, paths, strict=True):
        row = catalog.by_checksum(_checksum(catalog, sample_id))
        cache.put(
            run.id,
            {row.checksum: ChoicesPrediction(
                values=["cat", "dog"], confidences=CONFIDENCES[path.stem]
            )},
        )
        by_name[path.stem] = row

    config = tmp_path / "config.toml"
    config.write_text(
        f'[label_studio]\nurl = "http://ls:8080"\napi_key = "t"\n'
        f'[catalog]\nroot = "{catalog_root}"\n'
    )

    fake = FakeLabelStudio()
    monkeypatch.setattr(
        "strata.labeller.cli._ls_client", lambda settings, project, config_path: fake
    )
    project.save_ls_project_id("http://ls:8080", fake.create_project("demo"))

    return project, config, fake, by_name, catalog, label_set_id


def _checksum(catalog, sample_id):
    from sqlalchemy import select

    from strata.catalog import tables as t

    with catalog.engine.connect() as conn:
        return conn.execute(
            select(t.sample.c.checksum).where(t.sample.c.id == sample_id)
        ).scalar()


def push(project, config, *args):
    return runner.invoke(
        app, ["push", "-p", str(project.root), "--config", str(config), *args]
    )


# ----------------------------------------------------------------------


def test_a_push_creates_tasks_and_attaches_predictions(stage):
    project, config, fake, _, _, _ = stage

    result = push(project, config)

    assert result.exit_code == 0, result.output
    assert len(fake.tasks) == 3
    # The part that had no coverage and broke twice: every task created also
    # gets the model's guess attached
    assert len(fake.predictions) == 3


def test_the_least_confident_comes_first(stage):
    project, config, fake, by_name, _, _ = stage

    push(project, config, "--limit", "1")

    # One task, and it is the sample the model was least sure of — what a
    # human settles fastest
    [task] = fake.tasks.values()
    assert by_name["vague"].checksum in task["data"]["image"]


def test_the_score_shown_is_the_confidence_ranked_on(stage):
    project, config, fake, _, _, _ = stage

    push(project, config, "--limit", "1")

    [[prediction]] = fake.predictions.values()
    # Not the first confidence, which is whichever class happened to come
    # first: the number beside a task and its position have to agree
    assert prediction["score"] == pytest.approx(0.40)


def test_a_second_push_creates_nothing_new(stage):
    project, config, fake, _, _, _ = stage

    push(project, config)
    push(project, config)

    # What makes a push resumable: an interrupted one is just run again
    assert len(fake.tasks) == 3


def test_the_task_map_survives_the_command(stage):
    project, config, fake, _, catalog, _ = stage

    push(project, config)

    saved = json.loads(
        next((project.state_dir).glob("tasks_*.json")).read_text()
    )
    # Keyed on sample id; losing it means every task is re-created next time
    assert len(saved["tasks"]) == 3
    assert set(saved["tasks"].values()) == set(fake.tasks)
    # And which catalog those ids belong to. Without it the map is a set of
    # integers that mean something different in every other catalog.
    assert saved["catalog"] == catalog.id


def test_pushing_without_predictions_still_creates_tasks(stage):
    project, config, fake, _, _, _ = stage

    result = push(project, config, "--no-predictions")

    assert result.exit_code == 0, result.output
    assert len(fake.tasks) == 3
    assert fake.predictions == {}


def test_a_labelled_sample_is_not_pushed_again(stage):
    project, config, fake, by_name, catalog, label_set_id = stage
    catalog.annotations.annotate(by_name["confident"].id, label_set_id, Choices(values=["cat"]))

    push(project, config)

    # Answered is not awaiting review; a queue that re-asks settled questions
    # is worse than an empty one
    assert len(fake.tasks) == 2


def test_the_ranking_strategy_is_selectable(stage):
    """Three uncertainties are implemented; only one was reachable.

    They disagree on purpose. least-confident asks how sure the model was of
    its best guess; margin asks whether it could tell the top two apart — a
    model certain of two classes at once is certain about the wrong
    question.
    """
    project, config, fake, by_name, _, _ = stage

    push(project, config, "--limit", "1", "--strategy", "margin")

    # least-confident would pick 'vague'; margin picks 'torn', whose top two
    # are a hundredth apart. Different answers, or the flag proves nothing.
    [task] = fake.tasks.values()
    assert by_name["torn"].checksum in task["data"]["image"]


def test_an_unknown_strategy_says_what_there_is(stage):
    project, config, fake, _, _, _ = stage

    result = push(project, config, "--strategy", "vibes")

    assert result.exit_code == 1
    assert "least-confident" in result.stdout
    # Refused before anything was created, not halfway through a queue
    assert fake.tasks == {}


def test_a_disputed_sample_is_pushed_first(stage):
    """Ahead of the uncertainty ranking, not within it.

    Two people answered it differently, so one is wrong and only a person
    settles which. Where it lands in an uncertainty ordering depends on the
    model's opinion, which has no bearing on the disagreement — and the
    model is confident about this one.
    """
    from strata.labels import Choices

    project, config, fake, by_name, catalog, label_set_id = stage
    settled = by_name["confident"]
    catalog.annotations.annotate(settled.id, label_set_id, Choices(values=["cat"]))
    catalog.annotations.record_conflict(
        settled.id, label_set_id, Choices(values=["cat"]), Choices(values=["dog"])
    )

    result = push(project, config, "--limit", "1")

    assert "answered two ways" in result.stdout
    [task] = fake.tasks.values()
    assert settled.checksum in task["data"]["image"]
