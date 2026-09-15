"""The dependency graph, enforced rather than described.

Phase 0 said an import violating the graph should fail the build. It never
did, and the graph has held by discipline — which works until the day
someone reaches for the obvious import and nothing objects.

The rules are not stylistic. The catalog is the durable asset: annotations
outlive the tool that collected them, so it must not import the labelling
tool or an ML framework, or a five-year-old corpus needs both installed to
be read. The training core takes a directory and a manifest, so a model can
be tested against a fixture and run where no database exists — the service
layer is the one place allowed to know a catalog.

Lives here because `labeller` imports more of the others than any package
but `experiment`, so it is the natural place to assert what they may not.
"""

import ast
import importlib
import pathlib
import types

import pytest

#: Which strata packages each may import.
ALLOWED = {
    "labels": set(),
    "common": set(),
    "catalog": {"labels", "common"},
    "modelling": {"labels", "catalog", "common"},
    # The job: the one file the two tools above it both read (docs/adr/0016).
    "project": {"labels", "catalog", "modelling"},
    "labeller": {"labels", "catalog", "modelling", "project"},
    # Sequences the others' stages over a project. Nothing may import it,
    # and it does not import the labeller: the two are peers over the job.
    "experiment": {"labels", "common", "catalog", "modelling", "project"},
}

#: Third-party imports that would undo the point of a package.
FORBIDDEN = {
    "labels": {
        "torch",
        "timm",
        "transformers",
        "sqlalchemy",
        "boto3",
        "fastapi",
        "label_studio_sdk",
    },
    # Thin by design: nothing but what its extras name, and never the
    # frameworks or the value types, or every consumer would carry them.
    "common": {"torch", "timm", "transformers", "boto3", "fastapi", "label_studio_sdk", "pydantic"},
    "catalog": {"torch", "timm", "transformers", "label_studio_sdk"},
    "modelling": {"label_studio_sdk"},
    # The job names no tool and carries no framework.
    "project": {"label_studio_sdk", "torch", "timm", "transformers"},
    "labeller": set(),
    # Nothing in the orchestrator may know Label Studio exists, or carry a
    # framework: the labeller's stages talk to the one and the models to the other.
    "experiment": {"label_studio_sdk", "torch", "timm", "transformers"},
}

#: `modelling` may import `catalog`, but only from its remote layer.
CATALOG_IN_MODELLING = {"service.py", "rounds.py", "checks.py"}


def _imports(path: pathlib.Path):
    for module, _ in _imported(path):
        yield module


def _imported(path: pathlib.Path):
    """Every absolute import in a module: the module named, and the names taken from it."""
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, ()
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module, tuple(alias.name for alias in node.names)


def _exported(package: types.ModuleType) -> set[str]:
    """What a package's ``__init__`` offers by name."""
    declared = getattr(package, "__all__", None)
    if declared is not None:
        return set(declared)
    return {
        name
        for name, value in vars(package).items()
        if not name.startswith("_") and not isinstance(value, types.ModuleType)
    }


def _modules(package: str):
    """Every module in a package as installed, and never an empty list.

    Read from the installed package rather than a sibling directory, so
    the check is the same once the packages are repositories apart. A path
    that resolves to nothing makes all of these pass without reading
    anything, which is how this suite spent its first minutes: green, and
    scanning a directory that does not exist.
    """
    try:
        installed = importlib.import_module(f"strata.{package}")
    except ModuleNotFoundError:
        # Alone, this package has only what it depends on installed; the
        # workspace, where every package is, checks the rest.
        pytest.skip(f"strata.{package} is not installed here")
    roots = [pathlib.Path(p) for p in installed.__path__]
    found = sorted(path for root in roots for path in root.rglob("*.py"))
    assert found, f"no modules found under {roots} — this suite would pass vacuously"
    return found


@pytest.mark.parametrize("package", sorted(ALLOWED))
def test_a_package_imports_only_what_it_may(package):
    offences = []
    for path in _modules(package):
        for name in _imports(path):
            if not name.startswith("strata."):
                continue
            other = name.split(".")[1]
            if other != package and other not in ALLOWED[package]:
                offences.append(f"{path.name} imports strata.{other}")
    assert not offences, f"{package}: " + "; ".join(offences)


@pytest.mark.parametrize("package", sorted(ALLOWED))
def test_a_package_uses_only_what_its_dependencies_promise(package):
    """A module path across packages is a promise the producer declares.

    What a sibling may use is what the producer's ``__init__`` exports,
    plus the modules it lists in ``PUBLIC_MODULES`` (docs/adr/0015). Read
    from the installed producer rather than from a list kept here, so the
    check means the same thing once the packages are repositories apart.
    """
    offences = []
    for path in _modules(package):
        for module, names in _imported(path):
            if not module.startswith("strata."):
                continue
            parts = module.split(".")
            other = parts[1]
            if other == package:
                continue
            producer = importlib.import_module(f"strata.{other}")
            promised = getattr(producer, "PUBLIC_MODULES", frozenset())
            rest = ".".join(parts[2:])
            if rest:
                if rest not in promised:
                    offences.append(
                        f"{path.name} imports strata.{other}.{rest}, which is not promised"
                    )
                continue
            exported = _exported(producer)
            for name in names:
                if name not in exported and name not in promised:
                    offences.append(
                        f"{path.name} takes {name} from strata.{other}, which does not export it"
                    )
    assert not offences, f"{package}: " + "; ".join(offences)


@pytest.mark.parametrize("package", sorted(FORBIDDEN))
def test_a_package_avoids_what_would_undo_it(package):
    offences = []
    for path in _modules(package):
        for name in _imports(path):
            top = name.split(".")[0]
            if top in FORBIDDEN[package]:
                offences.append(f"{path.name} imports {top}")
    assert not offences, f"{package}: " + "; ".join(offences)


def test_only_the_service_layer_of_modelling_knows_a_catalog():
    """The training core takes a directory and a manifest, and nothing else.

    That is what makes a dataset version portable and the core testable
    against a fixture. A catalog import anywhere else undoes both, quietly.
    """
    offenders = {
        path.name
        for path in _modules("modelling")
        if any(name.startswith("strata.catalog") for name in _imports(path))
    }
    assert offenders <= CATALOG_IN_MODELLING, (
        f"catalog imported outside the service layer: "
        f"{', '.join(sorted(offenders - CATALOG_IN_MODELLING))}"
    )
