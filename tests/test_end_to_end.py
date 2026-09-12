"""The whole chain, once: project → catalog → dataset → training.

Every other suite tests one package against a hand-written version of its
neighbour's output, which is what keeps them honest about contracts. This is
the one place they meet, so it is where a contract that two packages read
differently would show up.

It needs no ML framework: the model is nine lines and imports nothing.
"""

import json
import shutil

from toy_model import TOY_SOURCE

from strata.catalog import EVERYTHING, Catalog
from strata.modelling import PredictRequest, RunStore, TrainRequest, predict, train


def split_of(directory) -> dict[int, str]:
    manifest = json.loads((directory / "manifest.json").read_text())
    return {s["id"]: s["split"] for s in manifest["samples"]}

def _ingested(project, tmp_path, n: int = 20):
    """A catalog holding n of the project's images, each named for its file."""
    paths = []
    for i in range(n):
        path = project.data_dir / f"img{i:03d}.jpg"
        path.write_bytes(f"image {i}".encode())
        paths.append(path)
    catalog = Catalog.local(tmp_path / "catalog")
    ids = catalog.ingest(paths, media="image", metadata_for=lambda p: {"source_path": p.name})
    return catalog, ids


def test_a_project_becomes_a_trained_run(project, tmp_path):
    # 20 files, 16 of them labelled — the shape of a project a few rounds in
    from strata.labels import Choices

    catalog, ids = _ingested(project, tmp_path)
    label_set_id = catalog.label_sets.create(
        "demo", project.schema.catalog_schema()
    )
    catalog.annotations.annotate_many(
        label_set_id,
        [
            (sample_id, Choices(values=["cat" if i % 2 else "dog"]))
            for i, sample_id in enumerate(ids[:16])
        ],
    )
    # Sixteen answered, four still awaiting review
    assert len(catalog.labelled(label_set_id, EVERYTHING)) == 16
    assert len(catalog.unlabelled(label_set_id, EVERYTHING)) == 4

    dataset_id = catalog.create_dataset("demo", label_set_id, collections=EVERYTHING)
    materialised = catalog.materialise(dataset_id, tmp_path / "materialised")
    manifest = json.loads((materialised / "manifest.json").read_text())

    # Only the labelled samples; the unreviewed four are the pool, not data
    assert len(manifest["samples"]) == 16
    assert 0 < sum(s["split"] == "val" for s in manifest["samples"]) < 16

    shutil.copyfile(TOY_SOURCE, materialised / "toy.py")
    store = RunStore.local(tmp_path / "runs")
    run = train(TrainRequest(dataset_dir=materialised, model="toy.py:Toy"), store)

    assert run.classes == ["cat", "dog"]
    assert run.checkpoint.exists()
    # The split the catalog decided is the split the model was handed
    assert run.metrics["n_train"] + run.metrics["n_val"] == 16

    queue = catalog.unlabelled(label_set_id, EVERYTHING)
    pool = [catalog.blobs.path_for(s.location) for s in queue]
    assert len(predict(PredictRequest(run_id=run.id, paths=pool), store)) == 4


def test_a_second_round_keeps_the_split_and_chains_the_run(project, tmp_path):
    from strata.labels import Choices

    catalog, ids = _ingested(project, tmp_path)
    label_set_id = catalog.label_sets.create("demo", project.schema.catalog_schema())
    catalog.annotations.annotate_many(
        label_set_id, [(i, Choices(values=["cat"])) for i in ids[:12]]
    )
    store = RunStore.local(tmp_path / "runs")

    first_dir = catalog.materialise(
        catalog.create_dataset("demo", label_set_id, collections=EVERYTHING), tmp_path / "v1"
    )
    shutil.copyfile(TOY_SOURCE, first_dir / "toy.py")
    first = train(TrainRequest(dataset_dir=first_dir, model="toy.py:Toy"), store)
    before = split_of(first_dir)

    # Label the rest and go round again
    catalog.annotations.annotate_many(
        label_set_id, [(i, Choices(values=["dog"])) for i in ids[12:]]
    )

    second_dir = catalog.materialise(
        catalog.create_dataset("demo", label_set_id, collections=EVERYTHING), tmp_path / "v2"
    )
    shutil.copyfile(TOY_SOURCE, second_dir / "toy.py")
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
