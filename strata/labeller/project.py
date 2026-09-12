"""The project construct: a self-contained, portable labelling job.

A project is a directory holding everything that belongs to one labelling
job — the data, the label schema, the annotations, the model and its
checkpoints — so it can be picked up and moved somewhere else once the
labelling is done. Machine-level settings (Label Studio URL, API key) stay
outside it, in ``config.toml``; they describe your laptop, not the job.

    projects/my-project/
    ├── project.toml            # this file's schema
    ├── dataset.json            # samples + annotations
    ├── data/raw/…              # the samples: images, documents, …
    ├── model.py                # optional project-local model
    ├── checkpoints/            # round_001.pt, …
    ├── rounds/round_001/       # metadata.json, labeled.json
    └── .state/                 # Label Studio bookkeeping, not part of a handoff

Sample paths in ``dataset.json`` are relative to ``[data] root``, so moving
the files or the project never rewrites the dataset.
"""

import json
import os
import re
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

from strata.modelling import ModelError, resolve
from strata.modelling.model import Model

from . import schemas
from .schemas import LabelSchema

PROJECT_FILE = "project.toml"
PROJECT_ENV_VAR = "AUTO_LABELLER_PROJECT"
# Projects live side by side here, addressable by name: -p cats
PROJECTS_DIR = "projects"

CUSTOM_LABEL_CONFIG = "label_config.xml"


class ProjectError(Exception):
    """Raised for a missing, malformed, or inconsistent project."""


@dataclass
class LabelConfigSpec:
    template: str = "image_classification"
    classes: list[str] = field(default_factory=list)
    # Template-specific parameters (e.g. choice="single"); validated against
    # the schema the template selects
    choice: str | None = None
    # Span-only, and None means "not declared" so that setting either on a
    # template that has no such notion is refused by name rather than
    # ignored. Both describe what the job is, not a preference: a model
    # that cannot learn overlapping spans refuses the label set rather than
    # training on a projection of it.
    multi_label: bool | None = None
    overlapping: bool | None = None
    # For template = "custom": the project's own labeling config
    file: str = CUSTOM_LABEL_CONFIG


@dataclass
class ModelSpec:
    # "<file>.py:Class" resolves inside the project; "pkg.module:Class"
    # falls back to an installed package
    ref: str = "multilabel"
    params: dict = field(default_factory=dict)
    # Merged over params when a round starts cold. A run with nothing to
    # inherit has to learn from scratch, where a warm round is an increment
    # onto something already trained — so the epoch count that suits one
    # badly undertrains the other.
    fresh_params: dict = field(default_factory=dict)

    def params_for(self, fresh: bool) -> dict:
        return {**self.params, **self.fresh_params} if fresh else dict(self.params)


@dataclass
class CatalogSpec:
    """Which catalog, and which label set inside it, this job uses.

    Where each catalog *is* lives in config.toml — that describes the
    machine. Which one this job draws from is part of what the job is, so
    it travels with the project.
    """

    #: Which catalog on this host, by the name it has in config.toml. Empty
    #: means the host's default, which is the only catalog on a host with
    #: one — so a project written before any of this still loads.
    #:
    #: Naming it is a statement about the data, not about a machine: a
    #: project moved to another host expects a catalog of the same name
    #: there, and gets an error rather than someone else's corpus if there
    #: is none.
    name: str = ""
    #: Defaults to the project's own name, which is the label set `ingest`
    #: creates.
    label_set: str = ""
    #: The dataset name versions accumulate under; defaults to the label set.
    dataset: str = ""
    #: Which collections in the catalog this job draws from, as paths:
    #: ["sat_images"] takes everything under it, ["sat_images/2024"] one
    #: batch. Defaults to a collection named after the label set.
    #:
    #: Dropping one declares that data out of scope, training included. To
    #: stop being asked about a batch while keeping what it already
    #: answered, skip the rest of it instead.
    collections: list[str] = field(default_factory=list)


@dataclass
class DataSpec:
    root: str = "data/raw"
    #: Where the corpus arrives, before anything has converted it — a
    #: directory of .eml, of video, of whatever this project started from.
    #: Separate from ``root`` because the two hold different things: one is
    #: the corpus as it came, the other is the corpus as the catalog stores
    #: it, and a conversion that overwrote the first would be a one-way
    #: door with no way back.
    source_root: str = "data/source"
    #: Which conversion to run over it. Empty resolves by what the files
    #: are and what this project ingests, and refuses an ambiguity rather
    #: than picking one.
    preparer: str = ""
    #: A registered sample type: what these files are, which decides which
    #: extensions are allowed, what is recorded about each one, and how they
    #: group. Empty falls back to ``kind``, which said two of those three
    #: things at once and is what this replaces.
    type: str = ""
    # How the samples relate to each other. "images": independent samples.
    # "frames": video frames, one folder per video — near-duplicate frames
    # must not straddle the train/val split, so whole videos move together.
    kind: str = "images"
    #: What this job's model is told about a sample besides its bytes.
    #:
    #: A role rather than a fact about the data: the same annotation is a
    #: target for the project that owns it and a feature for this one, and
    #: only these lines differ. Each names its source explicitly —
    #: ``label_set`` or ``metadata`` — because a bare name would have to
    #: guess between them, and guessing wrong reads a different value.
    #:
    #:     [[data.features]]
    #:     name = "species"
    #:     source = "label_set"
    #:     ref = "plant-species"
    features: list[dict] = field(default_factory=list)




@dataclass
class LabelStudioSpec:
    project_id: int | None = None


@dataclass
class Project:
    root: Path
    name: str
    label_config: LabelConfigSpec = field(default_factory=LabelConfigSpec)
    model: ModelSpec = field(default_factory=ModelSpec)
    data: DataSpec = field(default_factory=DataSpec)
    catalog: CatalogSpec = field(default_factory=CatalogSpec)
    label_studio: LabelStudioSpec = field(default_factory=LabelStudioSpec)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: Path | None = None) -> "Project":
        root = _resolve_root(path)
        toml_path = root / PROJECT_FILE
        if not toml_path.exists():
            available = [p.name for p in list_projects()]
            hint = (
                f" Available: {', '.join(available)}."
                if available
                else f" Create one with 'auto-labeller new {root.name}'."
            )
            raise ProjectError(f"No {PROJECT_FILE} in {root}.{hint}")
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)

        known_sections = {"label_config", "model", "data", "catalog", "label_studio"}
        unknown = set(data) - known_sections - {"name"}
        if unknown:
            raise ProjectError(
                f"Unknown section(s) in {toml_path}: {', '.join(sorted(unknown))}"
            )

        project = cls(
            root=root,
            name=data.get("name", root.name),
            label_config=_section(LabelConfigSpec, data.get("label_config", {}), "label_config"),
            model=_section(ModelSpec, data.get("model", {}), "model"),
            data=_section(DataSpec, data.get("data", {}), "data"),
            catalog=_section(CatalogSpec, data.get("catalog", {}), "catalog"),
            label_studio=_section(LabelStudioSpec, data.get("label_studio", {}), "label_studio"),
        )
        project._validate()
        return project

    def _validate(self) -> None:
        if self.label_config.choice not in {None, "single", "multiple"}:
            raise ProjectError(
                f"[label_config] choice must be 'single' or 'multiple', "
                f"got '{self.label_config.choice}'"
            )

        if self.label_config.template == schemas.CUSTOM_TEMPLATE:
            if self.label_config.classes:
                raise ProjectError(
                    f"[label_config] template = \"{schemas.CUSTOM_TEMPLATE}\" reads its "
                    f"classes from {self.label_config.file}; remove 'classes' from "
                    f"{PROJECT_FILE} so the two cannot drift apart"
                )
        # Building it here turns a bad template or parameter into an error
        # at load time rather than mid-push
        self.schema

    # ------------------------------------------------------------------
    # Label schema
    # ------------------------------------------------------------------

    @property
    def label_config_path(self) -> Path:
        return _resolve(self.root, self.label_config.file)

    def schema_with(self, classes: list[str]) -> LabelSchema:
        """This project's schema, but carrying someone else's class list.

        The label set is what an export is validated against and what a
        checkpoint maps its output neurons to, so it is authoritative for
        which classes exist. project.toml still says what kind of job this
        is — template, media, control names — and seeds the list when the
        label set is first created.

        Keeping the two in step by hand was the alternative, and a
        hand-edited file or a class added in the Label Studio UI silently
        put a reviewer's answer beyond what the catalog would accept.
        """
        schema = self.schema
        schema.classes = list(classes)
        return schema

    @property
    def schema(self) -> LabelSchema:
        """The schema this project labels with.

        For a custom project the XML is authoritative: it is what Label
        Studio annotates against, so control names and classes are read
        from it rather than declared twice.
        """
        spec = self.label_config
        try:
            if spec.template == schemas.CUSTOM_TEMPLATE:
                path = self.label_config_path
                if not path.exists():
                    raise ProjectError(
                        f"template = \"{schemas.CUSTOM_TEMPLATE}\" needs a labeling "
                        f"config at {path}"
                    )
                return schemas.from_label_config(path.read_text())
            declared = {
                "choice": spec.choice,
                "multi_label": spec.multi_label,
                "overlapping": spec.overlapping,
            }
            params = {k: v for k, v in declared.items() if v is not None}
            return schemas.from_template(spec.template, spec.classes, **params)
        except schemas.SchemaError as e:
            raise ProjectError(str(e)) from None

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    @property
    def label_set_name(self) -> str:
        return self.catalog.label_set or self.name

    @property
    def dataset_name(self) -> str:
        return self.catalog.dataset or self.label_set_name

    @property
    def collections(self) -> list[str]:
        """Where this job draws its samples from."""
        return self.catalog.collections or [self.label_set_name]

    @property
    def datasets_dir(self) -> Path:
        """Where materialised dataset versions are written."""
        return self.root / "datasets"

    @property
    def runs_dir(self) -> Path:
        """The model catalog for this project: runs, metrics and checkpoints."""
        return self.root / "runs"

    @property
    def rounds_dir(self) -> Path:
        return self.root / "rounds"

    @property
    def state_dir(self) -> Path:
        return self.root / ".state"

    @property
    def data_dir(self) -> Path:
        """Where this project's samples live, whatever kind of file they are."""
        return _resolve(self.root, self.data.root)

    @property
    def source_dir(self) -> Path:
        """Where this project's corpus arrives, before it is converted."""
        return _resolve(self.root, self.data.source_root)

    @property
    def feature_specs(self) -> list:
        """The declarations, validated. Empty for a project with none."""
        from strata.catalog.features import FeatureSpec

        return [FeatureSpec.from_dict(raw) for raw in self.data.features]

    @property
    def sample_type_name(self) -> str:
        """Which registered type this project's files are."""
        if not self.data.type:
            raise ProjectError(
                "[data] type is not set. It replaces [data] kind, which named "
                "both what a sample was and how it grouped — the reason a "
                "satellite scene had nowhere to go. For a project written "
                "before types:\n"
                "  uv run python packages/labeller/scripts/migrate_project_type.py <project>"
            )
        return self.data.type

    def sample_type(self):
        """The type itself, resolved from what is installed."""
        from strata.catalog.sample_types import SampleTypeError, resolve

        name = self.sample_type_name
        try:
            return resolve(name)()
        except SampleTypeError as e:
            raise ProjectError(f"[data] type: {e}") from None

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def load_model(self) -> Model:
        """Instantiate the project's model with its configured parameters.

        Resolution is ``strata.modelling``'s: a short name goes through the
        registry, and anything with a ``:`` is a direct reference — either a
        ``file.py:Class`` carried by the project, which executes code from
        the project directory, or an installed ``module:Class``.

        A model is the only part of the pipeline needing an ML framework, and
        the frameworks are optional dependencies, so this is where a missing
        one surfaces.
        """
        try:
            model_cls = resolve(self.model.ref, root=self.root)
        except ModelError as exc:
            raise ProjectError(str(exc)) from exc
        return model_cls(**self.model.params)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    #: Where each Label Studio instance's project id is kept, keyed by the
    #: instance's URL.
    LS_PROJECTS = "label_studio.json"

    def _ls_projects(self) -> dict:
        path = self.state_dir / self.LS_PROJECTS
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    def ls_project_id(self, ls_url: str) -> int | None:
        """This project's queue on one Label Studio, if it has one.

        Per instance, in the project's state rather than its file, which is
        copied between machines (``docs/adr/0008``). Falls back to a file
        written before this existed; delete that line once the owning
        machine has run init.
        """
        found = self._ls_projects().get(ls_url.rstrip("/"))
        if found is not None:
            return found
        return self.label_studio.project_id

    def save_ls_project_id(self, ls_url: str, project_id: int) -> None:
        """Record which project on ``ls_url`` belongs to this job."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        known = self._ls_projects()
        known[ls_url.rstrip("/")] = project_id
        (self.state_dir / self.LS_PROJECTS).write_text(json.dumps(known, indent=2))
        self.label_studio.project_id = project_id

    def add_classes(self, names: list[str], known: list[str] | None = None) -> list[str]:
        """Append classes to project.toml and return the new full list.

        Classes are append-only: a checkpoint maps output neurons to this
        list by position, so reordering would silently invalidate every
        checkpoint. When the list is empty the classes in use are written out
        first (``known``), turning an inferred order into a pinned one.
        """
        classes = list(self.label_config.classes) or sorted(known or [])
        for name in names:
            if not name.strip():
                raise ProjectError("Class names cannot be empty")
            if name in classes:
                raise ProjectError(f"Class '{name}' already exists in {PROJECT_FILE}")
            classes.append(name)

        rendered = "[" + ", ".join(f'"{c}"' for c in classes) + "]"
        toml_path = self.root / PROJECT_FILE
        text = toml_path.read_text()
        if re.search(r"^classes\s*=\s*\[.*?\]", text, flags=re.MULTILINE | re.DOTALL):
            text = re.sub(
                r"^classes\s*=\s*\[.*?\]",
                f"classes = {rendered}",
                text,
                count=1,
                flags=re.MULTILINE | re.DOTALL,
            )
        elif re.search(r"^\[label_config\]\s*$", text, flags=re.MULTILINE):
            text = re.sub(
                r"^(\[label_config\]\s*)$",
                rf"\1\nclasses = {rendered}",
                text,
                count=1,
                flags=re.MULTILINE,
            )
        else:
            text = text.rstrip("\n") + f"\n\n[label_config]\nclasses = {rendered}\n"
        toml_path.write_text(text)

        self.label_config.classes = classes
        return classes

    def require_ls_project_id(self, ls_url: str) -> int:
        found = self.ls_project_id(ls_url)
        if found is None:
            raise ProjectError(
                f"No Label Studio project on {ls_url} for this job. Run "
                f"'auto-labeller init' — each instance keeps its own queue, so a "
                f"project set up elsewhere does not carry over."
            )
        return found

    # ------------------------------------------------------------------
    # Scaffolding
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        root: Path,
        name: str | None = None,
        classes: list[str] | None = None,
        choice: str = "multiple",
        template: str = "image_classification",
    ) -> "Project":
        if (root / PROJECT_FILE).exists():
            raise ProjectError(f"{root / PROJECT_FILE} already exists")
        name = name or root.resolve().name
        classes = classes or []
        rendered = "[" + ", ".join(f'"{c}"' for c in classes) + "]"
        (root / "data" / "raw").mkdir(parents=True, exist_ok=True)

        custom = template == schemas.CUSTOM_TEMPLATE
        if custom:
            # The config is authoritative for a custom project, so give it
            # something valid to start from rather than a dangling reference
            config_path = root / CUSTOM_LABEL_CONFIG
            if not config_path.exists():
                starter = schemas.from_template(
                    "image_classification", classes or ["example"]
                )
                config_path.write_text(starter.label_config() + "\n")
        (root / PROJECT_FILE).write_text(
            f'name = "{name}"\n'
            "\n"
            "[label_config]\n"
            f'template = "{template}"\n'
            + (
                f"# classes come from {CUSTOM_LABEL_CONFIG}\n"
                if custom
                else f"classes = {rendered}\n"
            )
            + (
                f'choice = "{choice}"  # "single" for mutually exclusive classes\n'
                if template.endswith("_classification")
                else ""
            )
            +
            "\n"
            "[data]\n"
            'root = "data/raw"  # files live here; may be an absolute path\n'
            f'type = "{_scaffold_type(template)}"'
            "  # a registered sample type; see 'strata-catalog types'\n"
            "# Where a corpus arrives if it needs converting first — mail,\n"
            "# video. See 'strata-catalog preparers' and 'prepare'.\n"
            '# source_root = "data/source"\n'
            "\n"
            "[model]\n"
            '# "model.py:MyModel" to use a model carried by this project\n'
            'ref = "multilabel"  # a registered name, or "model.py:MyModel"\n'
            "\n"
            "[model.params]\n"
            "num_epochs = 4\n"
            "batch_size = 16\n"
            "lr = 5e-5\n"
            "\n"
            "# Applied over the above when a round has nothing to continue\n"
            "# from. A cold start on an increment's schedule undertrains,\n"
            "# and the result then reads as a baseline.\n"
            "[model.fresh_params]\n"
            "num_epochs = 8\n"
            "\n"
            "[label_studio]\n"
        )
        return cls.load(root)


def _scaffold_type(template: str) -> str:
    """Which sample type a new project starts with.

    From the template's media, because that is what the person choosing a
    template was saying. Anything more specific — frames, satellite scenes —
    is a change they make deliberately, since it decides how samples group.
    """
    spec = schemas.TEMPLATES.get(template)
    return spec.media.name if spec is not None else "image"


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def list_projects(base: Path | None = None) -> list[Path]:
    """Every project directory under ``projects/`` (or another base)."""
    base = base if base is not None else Path(PROJECTS_DIR)
    if not base.is_dir():
        return []
    return sorted(d for d in base.iterdir() if (d / PROJECT_FILE).exists())


def _resolve_named(path: Path) -> Path:
    """A path to a project directory, or a bare name under ``projects/``."""
    if path.name == PROJECT_FILE:
        path = path.parent
    if (path / PROJECT_FILE).exists():
        return path.resolve()
    named = Path(PROJECTS_DIR) / path
    if (named / PROJECT_FILE).exists():
        return named.resolve()
    # Absolute so paths handed to models and workers don't depend on cwd;
    # load() raises the missing-project.toml error from here
    return path.resolve()


def _resolve_root(path: Path | None) -> Path:
    if path is not None:
        return _resolve_named(path)

    env = os.environ.get(PROJECT_ENV_VAR)
    if env:
        return _resolve_named(Path(env))
    if (Path(".") / PROJECT_FILE).exists():
        return Path(".").resolve()

    # Bare command inside a repo with a projects/ folder: unambiguous only
    # when there is exactly one project
    candidates = list_projects()
    if len(candidates) == 1:
        return candidates[0].resolve()
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise ProjectError(
            f"Several projects found — pass --project NAME (one of: {names}) "
            f"or set ${PROJECT_ENV_VAR}."
        )
    return Path(".").resolve()


def _section(spec: type, data: dict, name: str):
    known = {f.name for f in fields(spec)}
    unknown = set(data) - known
    if unknown:
        raise ProjectError(
            f"Unknown key(s) in [{name}]: {', '.join(sorted(unknown))} "
            f"(known: {', '.join(sorted(known))})"
        )
    return spec(**data)
