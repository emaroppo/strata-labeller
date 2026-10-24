"""Rendering the packaged label config templates."""

from importlib import resources

from .base import TemplateSyntax

TEMPLATE_PACKAGE = "strata.labeller.labelstudio.label_configs"


def read_template(name: str) -> str:
    return resources.files(TEMPLATE_PACKAGE).joinpath(f"{name}.xml").read_text(encoding="utf-8")


def render_template(
    name: str,
    classes: list[str],
    label_tag: str,
    indent: str = "      ",
    **params: str,
) -> str:
    """Fill a template's @placeholders, rendering the class list itself.

    Label Studio binds only single-digit hotkeys, so classes past the ninth get none.
    """
    lines = []
    for i, value in enumerate(classes, start=1):
        hotkey = f' hotkey="{i}"' if i <= 9 else ""
        lines.append(f'{indent}<{label_tag} value="{value}"{hotkey}/>')
    template = TemplateSyntax(read_template(name))
    return template.substitute(labels="\n".join(lines), **params).rstrip("\n")
