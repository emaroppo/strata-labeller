"""The whole chain, once: project → catalog → dataset → training.

Every other suite tests one package against a hand-written version of its
neighbour's output, which is what keeps them honest about contracts. This is
the one place they meet, so it is where a contract that two packages read
differently would show up.

It needs no ML framework: the model is nine lines and imports nothing.
"""

import json

from strata.catalog import EVERYTHING, Catalog
from strata.labeller.dataset import Sample, save_dataset
from strata.labeller.to_catalog import migrate
from strata.modelling import PredictRequest, RunStore, TrainRequest, predict, train


def split_of(directory) -> dict[int, bool]:
    manifest = json.loads((directory / "manifest.json").read_text())
    return {s["id"]: s["val"] for s in manifest["samples"]}

TOY_MODEL = '''
import json
from pathlib import Path

from strata.labels import ChoicesPrediction
from strata.modelling import Model


class Toy(Model):
    task = "classification"
    version = "1"

    def __init__(self):
        self.classes = []

    def finetune(self, train, classes, val=None, on_epoch=None):
        self.classes = list(classes)
        return {"accuracy": 0.5, "n_train": float(len(train)), "n_val": float(len(val or []))}

    def predict(self, paths, on_batch=None):
        return [ChoicesPrediction(values=self.classes[:1], confidences=[0.5]) for _ in paths]

    def save(self, path):
        Path(path).write_text(json.dumps(self.classes))

    def load(self, path):
        self.classes = json.loads(Path(path).read_text())
'''


def test_a_project_becomes_a_trained_run(project, tmp_path):
    # 20 files, 16 of them labelled — the shape of a project a few rounds in
    samples = []
    for i in range(20):
        relative = f"img{i:03d}.jpg"
        (project.data_dir / relative).write_bytes(f"image {i}".encode())
        samples.append(
            Sample(
                path=relative,
                results=project.schema.encode_target(["cat" if i % 2 else "dog"]),
                annotated=i < 16,
            )
        )
    save_dataset(samples, project.dataset_path)

    catalog = Catalog.local(tmp_path / "catalog")
    report = migrate(project, catalog)
    assert (report.ingested, report.annotated, report.unlabelled) == (20, 16, 4)

    dataset_id = catalog.create_dataset("demo", report.label_set_id, collections=EVERYTHING)
    materialised = catalog.materialise(dataset_id, tmp_path / "materialised")
    manifest = json.loads((materialised / "manifest.json").read_text())

    # Only the labelled samples; the unreviewed four are the pool, not data
    assert len(manifest["samples"]) == 16
    assert 0 < sum(s["val"] for s in manifest["samples"]) < 16

    (materialised / "toy.py").write_text(TOY_MODEL)
    store = RunStore.local(tmp_path / "runs")
    run = train(TrainRequest(dataset_dir=materialised, model="toy.py:Toy"), store)

    assert run.classes == ["cat", "dog"]
    assert run.checkpoint.exists()
    # The split the catalog decided is the split the model was handed
    assert run.metrics["n_train"] + run.metrics["n_val"] == 16

    queue = catalog.unlabelled(report.label_set_id, EVERYTHING)
    pool = [catalog.blobs.path_for(s.location) for s in queue]
    assert len(predict(PredictRequest(run_id=run.id, paths=pool), store)) == 4


def test_a_second_round_keeps_the_split_and_chains_the_run(project, tmp_path):
    samples = []
    for i in range(20):
        relative = f"img{i:03d}.jpg"
        (project.data_dir / relative).write_bytes(f"image {i}".encode())
        samples.append(
            Sample(
                path=relative,
                results=project.schema.encode_target(["cat"]),
                annotated=i < 12,
            )
        )
    save_dataset(samples, project.dataset_path)

    catalog = Catalog.local(tmp_path / "catalog")
    report = migrate(project, catalog)
    store = RunStore.local(tmp_path / "runs")

    first_dir = catalog.materialise(
        catalog.create_dataset("demo", report.label_set_id, collections=EVERYTHING), tmp_path / "v1"
    )
    (first_dir / "toy.py").write_text(TOY_MODEL)
    first = train(TrainRequest(dataset_dir=first_dir, model="toy.py:Toy"), store)
    before = split_of(first_dir)

    # Label the rest and go round again
    for sample in samples[12:]:
        sample.results = project.schema.encode_target(["dog"])
        sample.annotated = True
    save_dataset(samples, project.dataset_path)
    migrate(project, catalog)

    second_dir = catalog.materialise(
        catalog.create_dataset("demo", report.label_set_id, collections=EVERYTHING), tmp_path / "v2"
    )
    (second_dir / "toy.py").write_text(TOY_MODEL)
    second = train(
        TrainRequest(dataset_dir=second_dir, model="toy.py:Toy", parent_run_id=first.id), store
    )
    after = split_of(second_dir)

    # The reason any of this exists: a warm-started model is never scored on
    # a sample an earlier round trained it on
    assert all(after[i] == before[i] for i in before)
    assert len(after) == 20
    assert [r.id for r in store.chain(second.id)] == [first.id, second.id]
    assert [v for _, _, v in store.history("demo", "accuracy")] == [0.5, 0.5]
