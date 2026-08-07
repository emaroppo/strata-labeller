"""The `[model] ref` contract — the seam a project plugs its model into.

Both forms have to keep working: a model.py carried inside the project, and
any installed ``pkg.module:Class``. The frameworks are optional extras, so a
missing one has to arrive as an error naming the extra rather than a stray
ModuleNotFoundError from inside importlib.
"""

import importlib
import re
from pathlib import Path

import pytest

from strata.labeller import models
from strata.labeller.project import Project, ProjectError

TOY_MODEL = '''
from pathlib import Path

from strata.labeller.model import BaseModel
from strata.labeller.schemas import ChoiceOutput


class ToyModel(BaseModel):
    """A model with no ML dependency at all."""

    schema_type = "image_classification"

    def __init__(self, num_epochs: int = 3, note: str = "default"):
        self.num_epochs = num_epochs
        self.note = note
        self.classes: list[str] = []

    def finetune(self, samples, classes, val_samples=None):
        self.classes = classes
        return {"loss": 0.0}

    def predict(self, paths):
        return [ChoiceOutput(labels=["cat"], confidences=[0.87]) for _ in paths]

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


def test_a_plugin_output_reaches_label_studio_results(toy_project):
    """The whole point of the seam: a model that knows nothing about Label
    Studio still produces results Label Studio can read."""
    from strata.labeller.dataset import Sample
    from strata.labeller.predict import run_predictions

    predictions = run_predictions(toy_project.load_model(), [Sample(path="a.jpg")], toy_project)

    assert predictions[0].results == [
        {
            "from_name": "label",
            "to_name": "image",
            "type": "choices",
            "value": {"choices": ["cat"]},
        }
    ]
    assert predictions[0].score == pytest.approx(0.87)
    assert predictions[0].uncertainty == pytest.approx(0.13)


def test_a_missing_model_file_names_the_path(project):
    loaded = set_ref(project, "nope.py:Missing")
    with pytest.raises(ProjectError, match="missing file"):
        loaded.load_model()


def test_a_missing_class_names_the_module(toy_project):
    loaded = set_ref(toy_project, "model.py:NotHere")
    with pytest.raises(ProjectError, match="No class 'NotHere'"):
        loaded.load_model()


def test_a_ref_without_a_class_is_refused(project):
    loaded = set_ref(project, "strata.labeller.models.classifier")
    with pytest.raises(ProjectError, match="must be"):
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


def test_extra_hint_names_the_extra_for_each_baseline():
    assert "image" in models.extra_hint("strata.labeller.models.classifier")
    assert "text" in models.extra_hint("strata.labeller.models.text_classifier")
    # A project's own model brings its own dependencies; we have no advice
    assert models.extra_hint("some.third.party:Model") is None


def test_a_baseline_without_its_framework_names_the_extra(project, monkeypatch):
    """Simulated rather than uninstalling torch: the behaviour under test is
    the ImportError -> ProjectError mapping, not torch itself."""
    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name.startswith("strata.labeller.models."):
            raise ImportError("No module named 'timm'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    loaded = set_ref(project, "strata.labeller.models.classifier:MultiLabelClassifier")
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


def test_baselines_are_not_imported_until_asked_for():
    """Importing the package must not pull in a framework — that is what
    keeps the base install free of torch."""
    import sys

    assert "strata.labeller.models" in sys.modules
    module = sys.modules["strata.labeller.models"]
    assert not hasattr(module, "classifier") or "torch" in sys.modules


def test_the_toy_model_declares_a_schema_it_matches(toy_project):
    # train.py refuses a model written for another task, and the check is a
    # bare string match against the template name
    assert toy_project.load_model().schema_type == toy_project.schema.type


def test_paths_handed_to_a_model_are_absolute(toy_project):
    assert Path(toy_project.sample_file("a.jpg")).is_absolute()
