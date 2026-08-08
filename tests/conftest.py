"""Shared fixtures.

Everything here is framework-free: these tests cover the parts of the
pipeline that hold whatever a project labels and whatever trains it, so
none of them import torch and all of them run on the base install.
"""

import os
from pathlib import Path

import pytest

# Rich colourises when $FORCE_COLOR is set, even writing into captured
# output, and the CLI tests match on substrings that escape codes split
# apart. Popped at import rather than in a fixture because the CLI builds
# its Console when the module is imported, which is before any fixture
# runs — and a developer whose terminal sets FORCE_COLOR would otherwise
# see failures with nothing to do with what they changed.
os.environ.pop("FORCE_COLOR", None)

from strata.labeller.project import PROJECT_ENV_VAR, Project  # noqa: E402


@pytest.fixture
def make_project(tmp_path, monkeypatch):
    """Create a project on disk and return it, loaded.

    Project resolution reads the working directory (a bare name resolves
    under ``projects/``), so tests run from a scratch directory rather than
    the repo.

    It also reads $AUTO_LABELLER_PROJECT, which the compose environment sets.
    Left alone, a developer who has sourced .env sees discovery tests fail
    for a reason that has nothing to do with what they changed.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(PROJECT_ENV_VAR, raising=False)

    def _make(name: str = "demo", *, under_projects: bool = True, **kwargs) -> Project:
        root = (tmp_path / "projects" / name) if under_projects else (tmp_path / name)
        root.mkdir(parents=True)
        kwargs.setdefault("classes", ["cat", "dog"])
        return Project.create(root, name=name, **kwargs)

    return _make


@pytest.fixture
def project(make_project) -> Project:
    return make_project()


@pytest.fixture
def sample_image(project) -> Path:
    """A file in the project's data root. Contents are irrelevant here —
    nothing under test opens it."""
    path = project.data_dir / "img001.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not really a jpeg")
    return path
