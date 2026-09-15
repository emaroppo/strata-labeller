"""What a label schema is, and the types every schema speaks.

A schema owns what is specific to a task type at the Label Studio boundary:
the labeling config it generates, and the translation between Label Studio
results and the :mod:`strata.labels` values a catalog stores. Ranking a
prediction for review is :mod:`strata.labeller.review.active_learning`'s.
See ``docs/adr/0013`` and ``docs/adr/0012``.
"""

from string import Template
from typing import Any, ClassVar, Protocol, runtime_checkable

from strata.labels import Boxes, Choices, Spans

from .media import Media

Result = dict[str, Any]

# Fields Label Studio attaches that say nothing about the annotation itself;
# dropped so the same annotation reads the same from two exports
VOLATILE_FIELDS = {"id", "origin", "lead_time", "created_at", "updated_at", "parent_id"}


class TemplateSyntax(Template):
    """``@name`` placeholders, so Label Studio's own ``$image`` survives."""

    delimiter = "@"


@runtime_checkable
class LabelSchema(Protocol):
    """The contract every task type implements."""

    #: Which task this is, as the catalog names it: "classification", "bbox", "span"
    task: ClassVar[str]
    #: What an annotation of this type is. docs/adr/0013
    value_type: ClassVar[type[Choices] | type[Spans] | type[Boxes]]
    #: The Label Studio control this schema annotates with
    control_tag: ClassVar[str]
    media: Media
    classes: list[str]
    from_name: str
    to_name: str

    def __init__(self, classes: list[str], media: Media = ..., **params: Any) -> None: ...

    @property
    def type(self) -> str:
        """ "<media>_<task>", e.g. "text_classification"."""
        ...

    @property
    def data_key(self) -> str:
        """Key under task["data"] Label Studio reads the sample from."""
        ...

    def catalog_schema(self):
        """This schema as the catalog stores it, with Label Studio left behind.

        Media does not survive the crossing. See ``docs/adr/0014``.
        """
        raise NotImplementedError

    def label_config(self) -> str:
        """The Label Studio labeling config for this schema."""
        ...

    def canonicalize(self, results: list[dict]) -> list[Result]:
        """Strip a Label Studio result list down to what is worth storing."""
        ...

    def decode_target(self, results: list[Result]) -> Any:
        """Stored results to whatever the model consumes as a target."""
        ...

    def encode_target(self, target: Any) -> list[Result]:
        """The inverse of :meth:`decode_target`."""
        ...


def strip_volatile(result: dict, keep: tuple[str, ...] = ()) -> Result:
    """One result entry, minus the fields that change on every export."""
    cleaned = {k: v for k, v in result.items() if k not in VOLATILE_FIELDS}
    allowed = {"from_name", "to_name", "type", "value", *keep}
    return {k: v for k, v in cleaned.items() if k in allowed}
