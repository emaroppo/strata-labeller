"""Editing a Label Studio labeling config without disturbing its layout.

Adding a class means inserting one element into XML that may have been
hand-tuned in the Label Studio UI — two-column layouts, zoom settings,
hotkeys. Regenerating the config from a template would throw that away, so
these helpers do a text-level insert and leave every other byte alone.
"""

import re

# Control tag -> the tag its individual classes use
CONTROL_TAGS = {
    "Choices": "Choice",
    "Labels": "Label",
    "RectangleLabels": "Label",
    "PolygonLabels": "Label",
    "BrushLabels": "Label",
    "KeyPointLabels": "Label",
    "EllipseLabels": "Label",
}


class LabelConfigError(Exception):
    """Raised when a labeling config cannot be read or edited."""


def find_control(xml: str) -> tuple[str, str]:
    """Return the (control tag, class tag) this config uses for its classes."""
    for control, item in CONTROL_TAGS.items():
        if re.search(rf"</{control}>", xml):
            return control, item
    known = ", ".join(CONTROL_TAGS)
    raise LabelConfigError(
        f"No class control found in the labeling config (looked for: {known}). "
        "A self-closing control with no classes cannot be extended automatically."
    )


def get_classes(xml: str) -> list[str]:
    _, item = find_control(xml)
    return re.findall(rf"<{item}\s[^>]*value=\"([^\"]*)\"", xml)


def _next_hotkey(xml: str, item: str) -> str | None:
    """The next free numeric hotkey, or None if this config doesn't use them.

    Only offered when every existing class already has one, so a config that
    deliberately avoids hotkeys keeps avoiding them.
    """
    entries = re.findall(rf"<{item}\s[^>]*>", xml)
    if not entries:
        return None
    hotkeys = [re.search(r"hotkey=\"(\d+)\"", e) for e in entries]
    if not all(hotkeys):
        return None
    used = {int(m.group(1)) for m in hotkeys if m}
    candidate = max(used) + 1
    return str(candidate) if candidate <= 9 else None


def add_class(xml: str, value: str) -> str:
    """Insert one class into the config's control, preserving everything else."""
    if re.search(r"[<>\"&]", value):
        raise LabelConfigError(
            f"Class name {value!r} contains a character that is not valid in XML "
            "(<, >, \" or &)"
        )
    control, item = find_control(xml)
    if value in get_classes(xml):
        raise LabelConfigError(f"Class {value!r} is already in the labeling config")

    hotkey = _next_hotkey(xml, item)
    attrs = f'value="{value}"' + (f' hotkey="{hotkey}"' if hotkey else "")
    element = f"<{item} {attrs}/>"

    # Copy the indentation of the last existing class so the insert looks native
    existing = list(re.finditer(rf"^([ \t]*)<{item}\s[^>]*>[ \t]*$", xml, re.MULTILINE))
    indent = existing[-1].group(1) if existing else "    "

    closing = re.search(rf"^([ \t]*)</{control}>", xml, re.MULTILINE)
    if closing is None:  # pragma: no cover - find_control already matched
        raise LabelConfigError(f"Could not locate </{control}> to insert before")

    return xml[: closing.start()] + f"{indent}{element}\n" + xml[closing.start() :]
