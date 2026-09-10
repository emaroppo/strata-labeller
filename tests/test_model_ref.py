"""The `[model] ref` contract — the seam a project plugs its model into.

Resolution itself lives in strata.modelling and is tested there. What this
covers is the project's side: that every form of ref still reaches a model,
that [model.params] arrive as constructor arguments, and that a project.toml
written before the baselines moved keeps working rather than failing with an
import error that says nothing about what to change.
"""

import importlib
import re
from pathlib import Path

import pytest

from strata.labeller.project import LEGACY_MODEL_REFS, Project, ProjectError

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


# ----------------------------------------------------------------------
# A project's own model.py
# ----------------------------------------------------------------------


@pytest.fixture
def toy_project(project) -> Project:
    (project.root / "model.py").write_text(TOY_MODEL)
    return set_ref(project, "model.py:ToyModel", 'num_epochs = 7\nnote = "from params"\n')


def test_a_project_local_model_loads(toy_project):
    assert type(toy_project.load_model()).__name__ == "ToyModel"


def test_model_params_reach_the_constructor(toy_project):
    model = toy_project.load_model()
    assert model.num_epochs == 7
    assert model.note == "from params"
def test_a_missing_model_file_names_the_path(project):
    loaded = set_ref(project, "nope.py:Missing")
    with pytest.raises(ProjectError, match="missing file"):
        loaded.load_model()


def test_a_missing_class_names_the_module(toy_project):
    loaded = set_ref(toy_project, "model.py:NotHere")
    with pytest.raises(ProjectError, match="No class 'NotHere'"):
        loaded.load_model()


def test_a_bare_name_is_looked_up_in_the_registry(project):
    # No ':' now means a registered short name rather than a malformed ref,
    # which is what lets a request carry "multilabel" over a wire
    pytest.importorskip("timm", reason="presence is an image baseline")
    loaded = set_ref(project, "presence")
    assert type(loaded.load_model()).__name__ == "PresenceClassifier"


def test_an_unregistered_bare_name_says_what_is_available(project):
    loaded = set_ref(project, "not-a-model")
    with pytest.raises(ProjectError, match="available: "):
        loaded.load_model()


# ----------------------------------------------------------------------
# An installed package
# ----------------------------------------------------------------------


def test_an_installed_module_ref_imports(project):
    # Any importable pkg.module:Class works — that is what lets a model
    # maintained in its own package be used without changes here
    loaded = set_ref(project, "collections:OrderedDict")
    assert type(loaded.load_model()).__name__ == "OrderedDict"


# ----------------------------------------------------------------------
# Optional extras
# ----------------------------------------------------------------------


def test_a_baseline_without_its_framework_names_the_extra(project, monkeypatch):
    """Simulated rather than uninstalling torch: the behaviour under test is
    the ImportError -> ProjectError mapping, not torch itself."""
    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name.startswith("strata.modelling.baselines."):
            raise ImportError("No module named 'timm'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    loaded = set_ref(project, "strata.modelling.baselines.classifier:MultiLabelClassifier")
    with pytest.raises(ProjectError, match="needs the 'image' extra"):
        loaded.load_model()


def test_an_unrelated_import_failure_is_not_blamed_on_a_missing_extra(project, monkeypatch):
    def fake_import(name, *args, **kwargs):
        raise ImportError("boom")

    monkeypatch.setattr(importlib, "import_module", fake_import)
    loaded = set_ref(project, "somepkg.models:Thing")
    with pytest.raises(ProjectError, match="will not import"):
        loaded.load_model()


def test_a_broken_project_model_is_reported_as_a_project_error(project):
    (project.root / "model.py").write_text("import definitely_not_installed\n")
    loaded = set_ref(project, "model.py:Whatever")
    with pytest.raises(ProjectError, match="will not import"):
        loaded.load_model()


def test_the_toy_model_declares_a_task_it_matches(toy_project):
    # Training refuses a model written for another task, and the label set
    # says which one it is
    assert type(toy_project.load_model()).task == "classification"


@pytest.mark.parametrize(
    "legacy",
    [
        "auto_labeller.models.classifier:PresenceClassifier",
        "strata.labeller.models.classifier:MultiLabelClassifier",
    ],
)
def test_a_ref_from_before_the_move_still_resolves(project, legacy):
    # The baselines moved twice. A project.toml written before either should
    # not fail with an import error that says nothing about what to change.
    assert legacy in LEGACY_MODEL_REFS
    pytest.importorskip("timm", reason="both refs name an image baseline")
    assert set_ref(project, legacy).load_model() is not None


def test_paths_handed_to_a_model_are_absolute(toy_project):
    assert Path(toy_project.sample_file("a.jpg")).is_absolute()
