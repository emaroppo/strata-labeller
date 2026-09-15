"""The project as the labeller sees it: the schema it labels with, and whose queue is whose.

The job itself, resolution, paths and class edits, is tested in
``strata.project``. What is here is the half this tool adds.
"""

import pytest

from strata.labeller.project import LabellingProject, ProjectError


def test_create_then_load_round_trip(project):
    reloaded = LabellingProject.load(project.root)
    assert reloaded.name == project.name
    assert reloaded.label_set.classes == ["cat", "dog"]
    assert reloaded.schema.type == "image_classification"


def test_the_template_follows_the_task_and_the_media(make_project):
    spans = make_project("spans", task="span", sample_type="text", classes=["PER"])
    assert spans.schema.type == "text_span"
    boxes = make_project("boxes", task="bbox")
    assert boxes.schema.type == "image_bbox"


def test_a_task_with_no_template_over_its_media_fails_at_load(make_project):
    # Spans over images: the job is well-formed, the tool has no template for it
    with pytest.raises(ProjectError, match="No labeling template"):
        make_project("odd", task="span")


def test_an_unknown_section_is_refused_by_the_tool(project):
    # The job carries what it does not own; the labeller, which owns the
    # rest, is where a typo is caught
    toml = project.root / "project.toml"
    toml.write_text(toml.read_text() + "\n[trainig]\nepochs = 3\n")
    with pytest.raises(ProjectError, match="Unknown section"):
        LabellingProject.load(project.root)


def test_an_unknown_key_in_the_tools_section_is_refused(project):
    toml = project.root / "project.toml"
    toml.write_text(toml.read_text() + "\n[label_studio]\nproject_id = 3\n")
    with pytest.raises(ProjectError, match=r"Unknown key\(s\) in \[label_studio\]"):
        LabellingProject.load(project.root)


# ----------------------------------------------------------------------
# A labeling config of the project's own
# ----------------------------------------------------------------------


def test_a_custom_project_gets_a_config_to_start_from(make_project):
    project = make_project("custom", custom=True)
    assert project.label_studio.config == "label_config.xml"
    assert project.label_config_path.exists()
    assert project.schema.classes == ["cat", "dog"]


def test_a_custom_config_supplies_control_names_and_the_job_supplies_classes(make_project):
    project = make_project("custom", custom=True)
    xml = project.label_config_path.read_text().replace('name="label"', 'name="verdict"')
    project.label_config_path.write_text(xml)
    reloaded = LabellingProject.load(project.root)
    assert reloaded.schema.from_name == "verdict"
    assert reloaded.schema.classes == ["cat", "dog"]


def test_a_custom_config_may_not_offer_a_class_the_job_does_not_declare(make_project):
    # A reviewer could apply it, and the catalog would refuse the export
    project = make_project("custom", custom=True)
    xml = project.label_config_path.read_text().replace(
        'value="dog"', 'value="dog"/>\n    <Choice value="bird"', 1
    )
    project.label_config_path.write_text(xml)
    with pytest.raises(ProjectError, match="offers bird"):
        LabellingProject.load(project.root)


def test_a_custom_config_must_annotate_the_jobs_task(make_project):
    project = make_project("custom", custom=True)
    toml = project.root / "project.toml"
    toml.write_text(
        toml.read_text()
        .replace('task = "classification"', 'task = "bbox"')
        .replace('choice = "multiple"', "")
    )
    with pytest.raises(ProjectError, match="annotates classification"):
        LabellingProject.load(project.root)


def test_a_custom_project_without_its_config_says_where_it_looked(make_project):
    project = make_project("custom", custom=True)
    project.label_config_path.unlink()
    with pytest.raises(ProjectError, match="does not exist"):
        LabellingProject.load(project.root)


def test_add_classes_extends_a_custom_config_too(make_project):
    project = make_project("custom", custom=True)
    project.add_classes(["bird"])
    assert 'value="bird"' in project.label_config_path.read_text()
    assert LabellingProject.load(project.root).schema.classes == ["cat", "dog", "bird"]


# ----------------------------------------------------------------------
# Whose queue is whose
# ----------------------------------------------------------------------


def test_each_label_studio_keeps_its_own_project(project):
    """A project.toml is copied between machines; a queue is not.

    A Label Studio project id means nothing on another install, so one job
    can have a queue on a desktop and another on a laptop. They reconcile
    through the annotations, never through task ids.
    """
    project.save_ls_project_id("http://desktop:8080", 42)
    project.save_ls_project_id("http://laptop:8080", 7)

    reloaded = LabellingProject.load(project.root)
    assert reloaded.ls_project_id("http://desktop:8080") == 42
    assert reloaded.ls_project_id("http://laptop:8080") == 7


def test_a_trailing_slash_is_the_same_instance(project):
    project.save_ls_project_id("http://desktop:8080", 42)
    assert LabellingProject.load(project.root).ls_project_id("http://desktop:8080/") == 42


def test_an_instance_with_no_queue_points_at_init(project):
    with pytest.raises(ProjectError, match="init"):
        project.require_ls_project_id("http://elsewhere:8080")


def test_the_queue_is_state_not_part_of_the_file(project):
    project.save_ls_project_id("http://desktop:8080", 42)
    assert "42" not in (project.root / "project.toml").read_text()
