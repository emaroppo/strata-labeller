"""Shared fixtures.

Everything here is framework-free: these tests cover the parts of the
pipeline that hold whatever a project labels and whatever trains it, so
none of them import torch and all of them run on the base install.
"""

from pathlib import Path

import pytest

from strata.labeller.project import Project


@pytest.fixture
def make_project(tmp_path, monkeypatch):
    """Create a project on disk and return it, loaded.

    Project resolution reads the working directory (a bare name resolves
    under ``projects/``), so tests run from a scratch directory rather than
    the repo.
    """
    monkeypatch.chdir(tmp_path)

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
