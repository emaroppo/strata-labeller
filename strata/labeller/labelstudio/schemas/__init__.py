"""Label schemas: one per task type, selected by task and media.

A template is ``<media>_<task>``: the task is the project's ``[label_set]``,
the media is its sample type's. A project with a labeling config of its
own (``[label_studio] config``) has the schema read from that XML, so the
control names Label Studio already knows stay authoritative, while the
task and the classes remain the job's.
"""

import re
from dataclasses import dataclass

from .base import LabelSchema, Result, strip_volatile
from .bbox import BBoxSchema
from .classification import ClassificationSchema
from .media import IMAGE, MEDIA, TEXT, Media
from .span import SpanSchema

CUSTOM_TEMPLATE = "custom"


@dataclass(frozen=True)
class TemplateSpec:
    schema: type[LabelSchema]
    media: Media

    @property
    def control_tag(self) -> str:
        return self.schema.control_tag


# The valid media/task combinations, which is not their product: boxes
# only make sense on images, character spans only on text
TEMPLATES: dict[str, TemplateSpec] = {
    "image_classification": TemplateSpec(ClassificationSchema, IMAGE),
    "image_bbox": TemplateSpec(BBoxSchema, IMAGE),
    "text_classification": TemplateSpec(ClassificationSchema, TEXT),
    "text_span": TemplateSpec(SpanSchema, TEXT),
}

# Label Studio control tag -> the schemas that use it. A custom config is
# read by its control, and the media it pairs with comes from the media tag
CONTROL_TAGS: dict[str, type[LabelSchema]] = {
    spec.control_tag: spec.schema for spec in TEMPLATES.values()
}

# Label Studio media tag -> media
MEDIA_TAGS: dict[str, Media] = {"Image": IMAGE, "Text": TEXT}


class SchemaError(Exception):
    """Raised when a schema cannot be built from a project or a config."""


def available_templates() -> list[str]:
    return sorted([*TEMPLATES, CUSTOM_TEMPLATE])


def from_template(template: str, classes: list[str], **params) -> LabelSchema:
    """Build a schema from a template name and the project's parameters."""
    try:
        spec = TEMPLATES[template]
    except KeyError:
        raise SchemaError(
            f"Unknown template '{template}' (available: {', '.join(available_templates())})"
        ) from None
    accepted = spec.schema.__init__.__code__.co_varnames
    unknown = [k for k in params if k not in accepted]
    if unknown:
        raise SchemaError(
            f"Template '{template}' takes no parameter(s): {', '.join(sorted(unknown))}"
        )
    return spec.schema(classes, media=spec.media, **params)


def from_label_config(xml: str) -> LabelSchema:
    """Derive a schema by reading a labeling config.

    The config is authoritative for a custom project: control names and the
    class list come from the XML, because that is what annotations in
    Label Studio will actually reference.
    """
    media = _media_from_config(xml)
    for tag, schema_cls in CONTROL_TAGS.items():
        match = re.search(rf"<{tag}\b([^>]*)>", xml)
        if not match:
            continue
        attrs = match.group(1)
        from_name = _attr(attrs, "name") or "label"
        to_name = _attr(attrs, "toName") or media.data_key
        item = "Choice" if tag == "Choices" else "Label"
        classes = re.findall(rf"<{item}\s[^>]*value=\"([^\"]*)\"", xml)
        params: dict[str, str] = {"from_name": from_name, "to_name": to_name}
        if schema_cls is ClassificationSchema:
            params["choice"] = _attr(attrs, "choice") or "multiple"
        if schema_cls is SpanSchema:
            # What the config permits is what reviewers will produce, so it
            # is what the label set has to accept. Overlap has no attribute
            # to read — Label Studio always allows it — so it stays declared
            # in project.toml.
            params["multi_label"] = _attr(attrs, "choice") == "multiple"
        return schema_cls(classes, media=media, **params)

    known = ", ".join(sorted(CONTROL_TAGS))
    raise SchemaError(
        f"No supported labeling control in the config (looked for: {known}). "
        "Label Studio supports more than strata does; a schema for "
        "this control would have to be added."
    )


def _media_from_config(xml: str) -> Media:
    """Which kind of file a custom config presents, from its media tag."""
    for tag, media in MEDIA_TAGS.items():
        if re.search(rf"<{tag}\b", xml):
            return media
    known = ", ".join(sorted(MEDIA_TAGS))
    raise SchemaError(
        f"No supported media tag in the config (looked for: {known})"
    )


def _attr(attrs: str, name: str) -> str | None:
    match = re.search(rf'{name}="([^"]*)"', attrs)
    return match.group(1) if match else None


__all__ = [
    "IMAGE",
    "MEDIA",
    "TEXT",
    "BBoxSchema",
    "ClassificationSchema",
    "LabelSchema",
    "Media",
    "SpanSchema",
    "Result",
    "SchemaError",
    "available_templates",
    "from_label_config",
    "from_template",
    "strip_volatile",
]
