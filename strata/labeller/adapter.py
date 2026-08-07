"""The Label Studio boundary.

Everything that knows Label Studio's shape stops here. Past this module a
sample is a catalog row and an annotation is a :mod:`strata.labels` value;
inside it, results carry control names, media keys and percentages.

The conversion is thin because ``schemas/`` already knows the wire format —
it converts between a bare Python target and a result list. What was missing
was the step either side of that: a neutral value in, a neutral value out.

A task's image URL is how a sample is recognised on the way back, so it
addresses the blob by content: the checksum is in the path, and a URL maps
to exactly one sample rather than to whatever string used to match.
"""

from dataclasses import dataclass
from urllib.parse import quote, unquote

from strata.catalog import Catalog, SampleRow
from strata.labels import Choices, ChoicesPrediction

from .schemas import LabelSchema

#: What Label Studio serves local files under.
LOCAL_FILES = "/data/local-files/?d="


class AdapterError(Exception):
    """A task or annotation that cannot be carried across the boundary."""


# ----------------------------------------------------------------------
# Values
# ----------------------------------------------------------------------


def to_results(value: Choices, schema: LabelSchema) -> list[dict]:
    """A neutral value as Label Studio results."""
    return schema.encode_target(list(value.values))


def from_results(results: list[dict], schema: LabelSchema) -> Choices:
    """Label Studio results as a neutral value.

    Note what this does with an empty list: it produces an empty value, not
    nothing. A reviewer who looked and found none of the classes present has
    answered the question, and the catalog stores that as a real annotation.
    """
    return Choices(values=list(schema.decode_target(results)))


def prediction_to_results(prediction: ChoicesPrediction, schema: LabelSchema) -> list[dict]:
    return schema.encode_target(list(prediction.values))


# ----------------------------------------------------------------------
# Addressing
# ----------------------------------------------------------------------


def blob_url(sample: SampleRow, prefix: str) -> str:
    """Where Label Studio fetches a sample's bytes.

    Percent-encoded, because a location that reaches a query string
    unescaped breaks on characters a checksum will never contain but a
    suffix might.
    """
    return f"{LOCAL_FILES}{prefix}/{quote(sample.location.container)}"


def location_from_url(url: str, prefix: str) -> str | None:
    """The blob a task's URL points at, or None if it points elsewhere.

    None is the ordinary answer for a task created before the catalog: its
    URL addresses the old data root, which names no blob.
    """
    if LOCAL_FILES not in url:
        return None
    tail = unquote(url.split(LOCAL_FILES, 1)[1])
    marker = f"{prefix}/"
    if not tail.startswith(marker):
        return None
    return tail[len(marker) :]


# ----------------------------------------------------------------------
# Tasks
# ----------------------------------------------------------------------


@dataclass
class Task:
    """One Label Studio task, built from a catalog sample."""

    sample_id: int
    data: dict
    annotations: list[dict]
    #: Whether anyone has answered, which is not the same as the answer
    #: being non-empty. Without this an annotation of "none of these apply"
    #: is indistinguishable from an unasked task, and a rebuilt project
    #: would put every such sample back in the queue.
    answered: bool = False

    def as_import(self) -> dict:
        payload: dict = {"data": self.data}
        if self.answered:
            payload["annotations"] = [{"result": self.annotations}]
        return payload


def build_tasks(
    samples: list[SampleRow],
    catalog: Catalog,
    label_set_id: int,
    schema: LabelSchema,
    prefix: str,
) -> list[Task]:
    """Tasks for a batch of samples, carrying any annotation they already have.

    Sending the existing annotation matters when a project is rebuilt: Label
    Studio is a view of the catalog rather than a second copy of it, so
    everything already answered should arrive answered.
    """
    tasks = []
    for sample in samples:
        value = catalog.annotation_of(sample.id, label_set_id)
        tasks.append(
            Task(
                sample_id=sample.id,
                data={schema.data_key: blob_url(sample, prefix)},
                annotations=to_results(value, schema) if value is not None else [],
                answered=value is not None,
            )
        )
    return tasks
