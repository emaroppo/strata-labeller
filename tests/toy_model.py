"""A model with no ML dependency, for tests that need a round to run.

Copied into a project or a dataset directory as ``toy.py`` and named as a
file reference, ``toy.py:Toy``, so a test exercises the same path a
project's own ``model.py`` takes. ``TOY_SOURCE`` is this file.

It accepts the parameters a project template writes, reports metrics a
test can count against the split, and persists its classes and ``note``
so a checkpoint can be read back.
"""

import json
from pathlib import Path

from strata.labels import ChoicesPrediction
from strata.modelling import Model

TOY_SOURCE = Path(__file__)


class Toy(Model):
    task = "classification"
    version = "1"

    def __init__(
        self, num_epochs: int = 4, batch_size: int = 16, lr: float = 5e-5, note: str = "default"
    ):
        self.num_epochs = num_epochs
        self.note = note
        self.classes: list[str] = []

    def finetune(self, train, classes, val=None, on_epoch=None):
        self.classes = list(classes)
        return {"accuracy": 0.5, "n_train": float(len(train)), "n_val": float(len(val or []))}

    def predict(self, paths, on_batch=None, *, features=None):
        return [ChoicesPrediction(values=self.classes[:1], confidences=[0.5]) for _ in paths]

    def save(self, path):
        Path(path).write_text(json.dumps({"classes": self.classes, "note": self.note}))

    def load(self, path):
        payload = json.loads(Path(path).read_text())
        self.classes = payload["classes"]
        self.note = payload["note"]
