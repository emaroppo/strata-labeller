"""A project as the labeller sees it: the job, plus what Label Studio needs of it.

The job is :class:`strata.project.Project`; this adds a ``[label_studio]`` section of the
same file (a labeling config of the project's own) and ``.state/``, the Label Studio
bookkeeping that is not part of a handoff. See ``docs/adr/0016``.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from strata.project import (
    PROJECT_ENV_VAR,
    PROJECT_FILE,
    PROJECTS_DIR,
    Project,
    ProjectError,
    list_projects,
    resolve_under,
    section,
)

from .labelstudio import label_config, schemas
from .labelstudio.schemas import LabelSchema

__all__ = [
    "CUSTOM_LABEL_CONFIG",
    "PROJECTS_DIR",
    "PROJECT_ENV_VAR",
    "PROJECT_FILE",
    "LabelStudioSpec",
    "LabellingProject",
    "ProjectError",
    "list_projects",
]

CUSTOM_LABEL_CONFIG = "label_config.xml"


@dataclass
class LabelStudioSpec:
    """The ``[label_studio]`` section: what this tool keeps in the job's file."""

    #: A labeling config of the project's own, relative to the project root. docs/adr/0016
    config: str = ""


@dataclass
class LabellingProject(Project):
    label_studio: LabelStudioSpec = field(default_factory=LabelStudioSpec)

    def __post_init__(self) -> None:
        unknown = set(self.extensions) - {"label_studio"}
        if unknown:
            raise ProjectError(
                f"Unknown section(s) in {self.root / PROJECT_FILE}: {', '.join(sorted(unknown))}"
            )
        self.label_studio = section(
            LabelStudioSpec, self.extensions.get("label_studio", {}), "label_studio"
        )

    def _validate(self) -> None:
        super()._validate()
        # Built here so a bad config or parameter is an error at load time. docs/adr/0030
        _ = self.schema

    @property
    def label_config_path(self) -> Path:
        return resolve_under(self.root, self.label_studio.config or CUSTOM_LABEL_CONFIG)

    @property
    def schema(self) -> LabelSchema:
        """The schema this project labels with, as Label Studio sees it."""
        spec = self.label_set
        try:
            if self.label_studio.config:
                return self._schema_from_config()
            media = self.sample_type().media
            template = f"{media}_{spec.label_type}"
            if template not in schemas.TEMPLATES:
                raise ProjectError(
                    f'No labeling template for label_type = "{spec.label_type}" '
                    f"over {media} samples "
                    f"(available: {', '.join(sorted(schemas.TEMPLATES))})"
                )
            declared = {
                "choice": spec.choice,
                "multi_label": spec.multi_label,
                "overlapping": spec.overlapping,
            }
            params = {k: v for k, v in declared.items() if v is not None}
            return schemas.from_template(template, spec.classes, **params)
        except schemas.SchemaError as e:
            raise ProjectError(str(e)) from None

    def _schema_from_config(self) -> LabelSchema:
        """A project's own config: control names from the XML, the rest from the job.

        Refuses an XML class the job does not declare; the job may declare more. See
        ``docs/adr/0016``.
        """
        path = self.label_config_path
        if not path.exists():
            raise ProjectError(f"[label_studio] config names {path}, which does not exist")
        schema = schemas.from_label_config(path.read_text())
        if schema.label_type != self.label_set.label_type:
            raise ProjectError(
                f"{path.name} annotates {schema.label_type}; [label_set] label_type is "
                f"{self.label_set.label_type}"
            )
        declared = self.label_set.classes
        undeclared = [c for c in schema.classes if c not in declared] if declared else []
        if undeclared:
            raise ProjectError(
                f"{path.name} offers {', '.join(undeclared)}, which [label_set] classes "
                f"does not declare. Add them with 'strata-labeller class add', or take "
                f"them out of the config."
            )
        if declared:
            schema.classes = list(declared)
        return schema

    def schema_with(self, classes: list[str]) -> LabelSchema:
        """This project's schema, but carrying someone else's class list. See ``docs/adr/0014``."""
        schema = self.schema
        schema.classes = list(classes)
        return schema

    @property
    def state_dir(self) -> Path:
        return self.root / ".state"

    #: Each Label Studio instance's project id, keyed by the instance's URL.
    LS_PROJECTS = "label_studio.json"

    def _ls_projects(self) -> dict:
        path = self.state_dir / self.LS_PROJECTS
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    def ls_project_id(self, ls_url: str) -> int | None:
        """This project's queue on one Label Studio, if it has one.

        Per instance, in the state dir, not the file copied between machines. See
        ``docs/adr/0008``.
        """
        return self._ls_projects().get(ls_url.rstrip("/"))

    def save_ls_project_id(self, ls_url: str, project_id: int) -> None:
        """Record which project on ``ls_url`` belongs to this job."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        known = self._ls_projects()
        known[ls_url.rstrip("/")] = project_id
        (self.state_dir / self.LS_PROJECTS).write_text(json.dumps(known, indent=2))

    def require_ls_project_id(self, ls_url: str) -> int:
        found = self.ls_project_id(ls_url)
        if found is None:
            raise ProjectError(
                f"No Label Studio project on {ls_url} for this job. Run "
                f"'strata-labeller init'; each instance keeps its own queue, so a "
                f"project set up elsewhere does not carry over."
            )
        return found

    def add_classes(self, names: list[str], known: list[str] | None = None) -> list[str]:
        """Append classes to project.toml, and to a config of the project's own."""
        classes = super().add_classes(names, known)
        if self.label_studio.config and self.label_config_path.exists():
            xml = self.label_config_path.read_text()
            for name in names:
                if name not in label_config.get_classes(xml):
                    xml = label_config.add_class(xml, name)
            self.label_config_path.write_text(xml)
        return classes

    @classmethod
    def create(
        cls,
        root: Path,
        name: str | None = None,
        classes: list[str] | None = None,
        label_type: str = "classification",
        choice: str = "multiple",
        sample_type: str = "image",
        custom: bool = False,
    ) -> "LabellingProject":
        """Scaffold the job, and with ``custom`` a labeling config of its own."""
        project = super().create(
            root,
            name=name,
            classes=classes,
            label_type=label_type,
            choice=choice,
            sample_type=sample_type,
        )
        if not custom:
            return project
        config_path = root / CUSTOM_LABEL_CONFIG
        if not config_path.exists():
            # Something valid to start from rather than a dangling reference
            starter = project.schema_with(classes or ["example"])
            config_path.write_text(starter.label_config() + "\n")
        project.write_extension("label_studio", {"config": CUSTOM_LABEL_CONFIG})
        return cls.load(root)
