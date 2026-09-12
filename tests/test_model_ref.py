"""The `[model] ref` contract — the seam a project plugs its model into.

Resolution itself lives in strata.modelling and is tested there: file and
module references, registry names, missing files and classes, the extra a
baseline needs. What this covers is the project's side of the seam alone:
a reference resolves against the project directory, ``[model.params]``
arrive as constructor arguments, and a failure reaches the caller as a
``ProjectError`` rather than as modelling's own exception.
"""

import re

import pytest

from strata.labeller.project import Project, ProjectError

TOY_MODEL = '''
from pathlib import Path

from strata.labels import ChoicesPrediction
from strata.modelling import Model


class ToyModel(Model):
    """A model with no ML dependency at all."""

    task = "classification"
    version = "1"

    def __init__(self, num_epochs: int = 3, note: str = "default"):
        self.num_epochs = num_epochs
        self.note = note
        self.classes: list[str] = []

    def finetune(self, train, classes, val=None, on_epoch=None):
        self.classes = classes
        return {"loss": 0.0}

    def predict(self, paths, on_batch=None, *, features=None):
        return [ChoicesPrediction(values=["cat"], confidences=[0.87]) for _ in paths]

    def save(self, path: Path) -> None:
        ...

    def load(self, path: Path) -> None:
        ...
'''


def set_ref(project: Project, ref: str, params: str | None = None) -> Project:
    """Point the project at a different model, and reload it.

    Rewrites whatever ref is currently there, so it can be applied twice to
    the same project.
    """
    toml = project.root / "project.toml"
    text = re.sub(r'^ref = ".*"$', f'ref = "{ref}"', toml.read_text(), count=1, flags=re.M)
    if params is not None:
        text = re.sub(
            r"^\[model\.params\]\n(?:\w+ = .*\n)*",
            f"[model.params]\n{params}",
            text,
            count=1,
            flags=re.M,
        )
    toml.write_text(text)
    return Project.load(project.root)


@pytest.fixture
def toy_project(project) -> Project:
    (project.root / "model.py").write_text(TOY_MODEL)
    return set_ref(project, "model.py:ToyModel", 'num_epochs = 7\nnote = "from params"\n')


def test_a_file_ref_resolves_inside_the_project(toy_project, monkeypatch, tmp_path):
    # Relative to the project root, not to wherever the command was run from
    monkeypatch.chdir(tmp_path)
    assert type(toy_project.load_model()).__name__ == "ToyModel"


def test_model_params_reach_the_constructor(toy_project):
    model = toy_project.load_model()
    assert model.num_epochs == 7
    assert model.note == "from params"


def test_a_model_that_will_not_load_is_a_project_error(project):
    (project.root / "model.py").write_text("import definitely_not_installed\n")
    loaded = set_ref(project, "model.py:Whatever")
    with pytest.raises(ProjectError, match="will not import"):
        loaded.load_model()
