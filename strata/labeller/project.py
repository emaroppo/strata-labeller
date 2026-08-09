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
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from pathlib import Path, PurePosixPath

from strata.modelling import ModelError, resolve
from strata.modelling.model import Model

from . import schemas
from .dataset import Sample
from .schemas import LabelSchema

PROJECT_FILE = "project.toml"
PROJECT_ENV_VAR = "AUTO_LABELLER_PROJECT"
# Projects live side by side here, addressable by name: -p cats
PROJECTS_DIR = "projects"

CUSTOM_LABEL_CONFIG = "label_config.xml"

# The baselines have moved twice: out of auto_labeller when the packages were
# namespaced, and out of the labeller when they became strata.modelling's.
# A project.toml written before either still resolves rather than failing
# with an import error that says nothing about what to change.
LEGACY_MODEL_REFS: dict[str, str] = {
    f"{old}:{name}": f"strata.modelling.baselines.{module}:{name}"
    for old, module in (
        ("auto_labeller.models.classifier", "classifier"),
        ("auto_labeller.models.text_classifier", "text_classifier"),
        ("strata.labeller.models.classifier", "classifier"),
        ("strata.labeller.models.text_classifier", "text_classifier"),
    )
    for name in (
        "MulticlassClassifier",
        "MultiLabelClassifier",
        "PresenceClassifier",
        "TextClassifier",
        "TextSpanTagger",
    )
}


class ProjectError(Exception):
    """Raised for a missing, malformed, or inconsistent project."""


@dataclass
class LabelConfigSpec:
    template: str = "image_classification"
    classes: list[str] = field(default_factory=list)
    # Template-specific parameters (e.g. choice="single"); validated against
    # the schema the template selects
    choice: str | None = None
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
    """Which label set in the catalog this job annotates against.

    Where the catalog is lives in config.toml — it describes the machine.
    This says what the job is, so it travels with the project.
    """

    #: Defaults to the project's own name, which is what `to-catalog` writes.
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
    # How the samples relate to each other. "images": independent samples.
    # "frames": video frames, one folder per video — near-duplicate frames
    # must not straddle the train/val split, so whole videos move together.
    kind: str = "images"


DATA_KINDS = ("images", "frames")


@dataclass
class LabelStudioSpec:
    project_id: int | None = None
    # Host directory mounted into the Label Studio container, and the name
    # it has under LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT. Image URLs are
    # built from the sample path relative to this mount.
    local_files_root: str = "data"
    local_files_prefix: str = "images"


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
        if self.data.kind not in DATA_KINDS:
            raise ProjectError(
                f"[data] kind must be one of {', '.join(DATA_KINDS)}, "
                f"got '{self.data.kind}'"
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
            params = {k: v for k, v in {"choice": spec.choice}.items() if v is not None}
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
    def dataset_path(self) -> Path:
        return self.root / "dataset.json"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

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
    def local_files_root(self) -> Path:
        return _resolve(self.root, self.label_studio.local_files_root)

    def sample_file(self, sample_path: str) -> Path:
        """Absolute path of a sample's file (sample paths are data-root relative)."""
        return self.data_dir / sample_path

    def relative_sample_path(self, path: Path) -> str:
        """Inverse of :meth:`sample_file` — an absolute path to a sample path."""
        return str(path.resolve().relative_to(self.data_dir.resolve()))

    @property
    def group_key(self) -> Callable[[Sample], str] | None:
        """What keeps related samples on one side of the train/val split.

        ``None`` for independent samples. For ``kind = "frames"`` it is the
        containing folder, one per video: frames a fraction of a second apart
        are near-duplicates, and letting them straddle the split would score
        the model on images it effectively trained on.
        """
        if self.data.kind == "frames":
            return lambda s: str(PurePosixPath(s.path).parent)
        return None

    # -- Label Studio local-files URL mapping --------------------------

    def _data_rel_to_mount(self) -> PurePosixPath:
        data, mount = self.data_dir.resolve(), self.local_files_root.resolve()
        try:
            rel = data.relative_to(mount)
        except ValueError:
            raise ProjectError(
                f"[data] root ({data}) must live inside "
                f"[label_studio] local_files_root ({mount}), which is the "
                f"directory mounted into the Label Studio container."
            ) from None
        return PurePosixPath(rel.as_posix())

    def mount_relative_path(self, sample_path: str) -> str:
        """Sample path as Label Studio sees it, relative to the mounted root."""
        return str(self._data_rel_to_mount() / sample_path)

    def sample_path_from_mount(self, mount_relative: str) -> str:
        """Inverse of :meth:`mount_relative_path`."""
        prefix = self._data_rel_to_mount()
        rel = PurePosixPath(mount_relative)
        if prefix.parts and rel.parts[: len(prefix.parts)] == prefix.parts:
            rel = PurePosixPath(*rel.parts[len(prefix.parts) :])
        return str(rel)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    @property
    def model_ref(self) -> str:
        """The model reference, with pre-move spellings translated.

        A property rather than a step inside :meth:`load_model`, because a
        training request carries the reference instead of the model and would
        otherwise get the untranslated one — the same rule applied in two
        places is a rule applied in one of them.
        """
        return LEGACY_MODEL_REFS.get(self.model.ref, self.model.ref)

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
            model_cls = resolve(self.model_ref, root=self.root)
        except ModelError as exc:
            raise ProjectError(str(exc)) from exc
        return model_cls(**self.model.params)

    def latest_checkpoint(self) -> Path | None:
        if not self.checkpoints_dir.exists():
            return None
        checkpoints = sorted(self.checkpoints_dir.glob("round_*.pt"))
        return checkpoints[-1] if checkpoints else None

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

        Per instance, and kept in the project's state rather than in
        project.toml, because a project.toml is copied between machines and
        a Label Studio project id means nothing on another install. One
        catalog project can have a queue on a desktop and another on a
        laptop; they reconcile through the annotations, not through the
        task ids.

        Falls back to a project.toml written before this existed. That value
        belongs to whichever instance created it, so a copied project.toml
        carrying one is exactly the confusion this replaces — delete the
        line once the machine that owns it has run init.
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
            'root = "data/raw"  # images live here; may be an absolute path\n'
            'kind = "images"  # "frames" for video frames, one folder per video\n'
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
