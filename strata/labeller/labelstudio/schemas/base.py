"""What a label schema is, and the types every schema speaks.

A schema owns what is specific to a task type at the Label Studio boundary:
the labeling config it generates, and the translation between Label Studio
results and the :mod:`strata.labels` values a catalog stores. Nothing else
in the labeller knows whether a project labels classes, boxes or spans.

Ranking a prediction for review is not a schema's business: see
:mod:`strata.labeller.review.active_learning`, which reads the confidences a value
carries.
"""

from string import Template
from typing import Any, ClassVar, Protocol, runtime_checkable

Result = dict[str, Any]

# Fields Label Studio attaches that say nothing about the annotation itself;
# dropping them keeps re-exports from churning the dataset file
VOLATILE_FIELDS = {"id", "origin", "lead_time", "created_at", "updated_at", "parent_id"}


class TemplateSyntax(Template):
    """``@name`` placeholders, so Label Studio's own ``$image`` survives."""

    delimiter = "@"


@runtime_checkable
class LabelSchema(Protocol):
    """The contract every task type implements."""

    #: "<media>_<task>", e.g. "text_classification"
    type: str
    #: Key under task["data"] Label Studio reads the sample from
    data_key: str
    #: The Label Studio control this schema annotates with
    control_tag: ClassVar[str]
    classes: list[str]
    from_name: str
    to_name: str

    def catalog_schema(self):
        """This schema as the catalog stores it, with Label Studio left behind.

        Media does not survive the crossing: ``image_classification`` and
        ``text_classification`` were template names, and classifying a
        photograph and classifying a document are the same task. What a
        sample is made of is the catalog's business, on the sample.
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
