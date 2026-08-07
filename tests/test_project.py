"""The project construct: resolution, validation, paths and class edits."""

import os
from pathlib import Path

import pytest

from auto_labeller.dataset import Sample
from auto_labeller.project import (
    PROJECT_ENV_VAR,
    Project,
    ProjectError,
    list_projects,
)

# ----------------------------------------------------------------------
# Loading and validation
# ----------------------------------------------------------------------


def test_create_then_load_round_trip(project):
    reloaded = Project.load(project.root)
    assert reloaded.name == project.name
    assert reloaded.label_config.classes == ["cat", "dog"]
    assert reloaded.schema.type == "image_classification"


def test_a_missing_project_says_what_is_available(make_project):
    make_project("alpha")
    with pytest.raises(ProjectError, match="Available: alpha"):
        Project.load(Path("nope"))


def test_an_unknown_section_is_refused(project):
    toml = project.root / "project.toml"
    toml.write_text(toml.read_text() + '\n[trainig]\nepochs = 3\n')
    with pytest.raises(ProjectError, match="Unknown section"):
        Project.load(project.root)


def test_an_unknown_key_inside_a_section_is_refused(project):
    toml = project.root / "project.toml"
    toml.write_text(toml.read_text().replace("[data]", "[data]\nrooot = \"typo\""))
    with pytest.raises(ProjectError):
        Project.load(project.root)


def test_choice_must_be_single_or_multiple(project):
    toml = project.root / "project.toml"
    toml.write_text(toml.read_text().replace('choice = "multiple"', 'choice = "maybe"'))
    with pytest.raises(ProjectError, match="must be 'single' or 'multiple'"):
        Project.load(project.root)


def test_a_bad_template_fails_at_load_not_mid_push(project):
    toml = project.root / "project.toml"
    toml.write_text(toml.read_text().replace("image_classification", "image_segmentation"))
    with pytest.raises(ProjectError, match="Unknown template"):
        Project.load(project.root)


def set_kind(project, kind: str) -> Project:
    toml = project.root / "project.toml"
    toml.write_text(toml.read_text().replace('kind = "images"', f'kind = "{kind}"'))
    return Project.load(project.root)


def test_data_kind_defaults_to_images(project):
    assert project.data.kind == "images"


def test_an_unknown_data_kind_is_refused(project):
    with pytest.raises(ProjectError, match="\\[data\\] kind must be one of"):
        set_kind(project, "videos")


def test_independent_images_have_no_group_key(project):
    assert project.group_key is None


def test_frames_are_grouped_by_their_folder(project):
    key = set_kind(project, "frames").group_key
    assert key is not None
    assert key(Sample(path="vid1/frame0007.jpg")) == "vid1"


def test_frames_in_a_nested_folder_group_on_the_containing_one(project):
    key = set_kind(project, "frames").group_key
    assert key(Sample(path="shoot-a/vid1/frame0007.jpg")) == "shoot-a/vid1"


def test_a_custom_project_must_not_declare_classes_twice(make_project):
    project = make_project("custom", template="custom", classes=[])
    toml = project.root / "project.toml"
    toml.write_text(
        toml.read_text().replace('template = "custom"', 'template = "custom"\nclasses = ["cat"]')
    )
    # The XML is authoritative; two sources of truth would drift
    with pytest.raises(ProjectError, match="remove 'classes'"):
        Project.load(project.root)


def test_a_custom_project_reads_its_classes_from_the_xml(make_project):
    project = make_project("custom", template="custom", classes=["cat", "dog"])
    assert project.label_config_path.exists()
    assert project.schema.classes == ["cat", "dog"]


def test_a_custom_project_without_its_config_says_where_it_looked(make_project):
    project = make_project("custom", template="custom", classes=[])
    project.label_config_path.unlink()
    with pytest.raises(ProjectError, match="needs a labeling config at"):
        Project.load(project.root)


# ----------------------------------------------------------------------
# Resolution
# ----------------------------------------------------------------------


def test_a_bare_name_resolves_under_projects(make_project):
    make_project("cats")
    assert Project.load(Path("cats")).name == "cats"


def test_a_path_to_the_toml_resolves_to_its_directory(make_project):
    project = make_project("cats")
    assert Project.load(project.root / "project.toml").name == "cats"


def test_a_single_project_is_found_without_being_named(make_project):
    make_project("only")
    assert Project.load().name == "only"


def test_several_projects_require_a_choice(make_project):
    make_project("alpha")
    make_project("beta")
    with pytest.raises(ProjectError, match="Several projects found"):
        Project.load()


def test_the_environment_variable_selects_a_project(make_project, monkeypatch):
    make_project("alpha")
    make_project("beta")
    monkeypatch.setenv(PROJECT_ENV_VAR, "beta")
    assert Project.load().name == "beta"


def test_list_projects_ignores_directories_without_a_project_file(make_project, tmp_path):
    make_project("real")
    (tmp_path / "projects" / "not-a-project").mkdir()
    assert [p.name for p in list_projects()] == ["real"]


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------


def test_sample_file_and_back(project):
    absolute = project.sample_file("batch2/img001.jpg")
    assert absolute == project.data_dir / "batch2" / "img001.jpg"
    # The dataset stores the relative form, so the pair must be exact inverses
    assert project.relative_sample_path(absolute) == os.path.join("batch2", "img001.jpg")


def test_an_absolute_data_root_is_used_as_is(project, tmp_path):
    elsewhere = tmp_path / "shared-images"
    elsewhere.mkdir()
    toml = project.root / "project.toml"
    toml.write_text(toml.read_text().replace('root = "data/raw"', f'root = "{elsewhere}"'))
    assert Project.load(project.root).data_dir == elsewhere


def test_mount_relative_path_round_trip(project):
    # Label Studio addresses files relative to the directory mounted into
    # its container, which is above the data root
    mount_relative = project.mount_relative_path("batch2/img001.jpg")
    assert mount_relative == "raw/batch2/img001.jpg"
    assert project.sample_path_from_mount(mount_relative) == "batch2/img001.jpg"


def test_a_data_root_outside_the_mount_is_an_error(project, tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    toml = project.root / "project.toml"
    toml.write_text(toml.read_text().replace('root = "data/raw"', f'root = "{outside}"'))
    with pytest.raises(ProjectError, match="must live inside"):
        Project.load(project.root).mount_relative_path("a.jpg")


def test_latest_checkpoint_is_the_highest_round(project):
    assert project.latest_checkpoint() is None
    project.checkpoints_dir.mkdir()
    for name in ("round_001.pt", "round_002.pt", "round_010.pt"):
        (project.checkpoints_dir / name).touch()
    # Zero-padded so lexical order is round order
    assert project.latest_checkpoint().name == "round_010.pt"


# ----------------------------------------------------------------------
# Class edits
# ----------------------------------------------------------------------


def test_add_classes_appends_and_persists(project):
    assert project.add_classes(["bird"]) == ["cat", "dog", "bird"]
    # Append-only: a checkpoint maps output neurons to this list by position
    assert Project.load(project.root).label_config.classes == ["cat", "dog", "bird"]


def test_add_classes_refuses_a_duplicate(project):
    with pytest.raises(ProjectError, match="already exists"):
        project.add_classes(["cat"])


def test_add_classes_refuses_an_empty_name(project):
    with pytest.raises(ProjectError, match="cannot be empty"):
        project.add_classes(["   "])


def test_add_classes_pins_the_inferred_order_when_the_list_is_empty(make_project):
    project = make_project("fresh", classes=[])
    # An order that was only ever inferred from the data becomes explicit
    assert project.add_classes(["bird"], known=["dog", "cat"]) == ["cat", "dog", "bird"]
    assert Project.load(project.root).label_config.classes == ["cat", "dog", "bird"]


def test_add_classes_leaves_the_rest_of_the_file_alone(project):
    before = (project.root / "project.toml").read_text()
    project.add_classes(["bird"])
    after = (project.root / "project.toml").read_text()
    changed = [
        (a, b) for a, b in zip(before.splitlines(), after.splitlines()) if a != b
    ]
    assert changed == [('classes = ["cat", "dog"]', 'classes = ["cat", "dog", "bird"]')]


def test_save_ls_project_id_writes_into_the_existing_section(project):
    project.save_ls_project_id(42)
    assert Project.load(project.root).label_studio.project_id == 42
    # Written once, and updated in place on a second call
    project.save_ls_project_id(43)
    text = (project.root / "project.toml").read_text()
    assert text.count("project_id") == 1
    assert Project.load(project.root).label_studio.project_id == 43


def test_require_ls_project_id_points_at_init(project):
    with pytest.raises(ProjectError, match="auto-labeller init"):
        project.require_ls_project_id()


def test_create_refuses_to_overwrite_an_existing_project(project):
    with pytest.raises(ProjectError, match="already exists"):
        Project.create(project.root, name="demo")
