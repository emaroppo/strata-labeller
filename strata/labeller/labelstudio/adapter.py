"""The Label Studio boundary.

Everything that knows Label Studio's shape stops here. Past this module a
sample is a catalog row and an annotation is a :mod:`strata.labels` value;
inside it, results carry control names, media keys and percentages, and
``schemas/`` does the conversion. A task's sample URL names the checksum.
See ``docs/adr/0013`` and ``docs/adr/0001``.
"""

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, unquote

from strata.catalog import Catalog, SampleRow, SignedUrls, blob_path, suffix_of
from strata.labels import AnyPrediction, AnyValue

from .schemas import LabelSchema

#: What Label Studio serves local files under.
LOCAL_FILES = "/data/local-files/?d="


# ----------------------------------------------------------------------
# Values
# ----------------------------------------------------------------------


def to_results(value: AnyValue, schema: LabelSchema) -> list[dict]:
    """A neutral value as Label Studio results."""
    return schema.encode_target(list(value.values))


def from_results(results: list[dict], schema: LabelSchema) -> AnyValue:
    """Label Studio results as a neutral value.

    Built through the schema's own value type (``docs/adr/0013``). An empty
    list produces an empty value, not nothing: a reviewer who found none of
    the classes present has answered (``docs/adr/0009``).
    """
    return schema.value_type(values=list(schema.decode_target(results)))


def prediction_to_results(prediction: AnyPrediction, schema: LabelSchema) -> list[dict]:
    """A model's output as Label Studio results.

    Delegated to the schema, which refuses a value of the wrong kind rather
    than encoding something meaningless.
    """
    return schema.encode_target(list(prediction.values))


# ----------------------------------------------------------------------
# Addressing
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Addressing:
    """How a task refers to its sample, in both directions.

    One object so the two directions cannot disagree. Writes the mount
    form or, with ``urls``, the catalog's signed HTTP URL; reads both, so
    old tasks keep resolving until relinked. See ``docs/adr/0013``.
    """

    #: What Label Studio serves the blob mount under. Changing it is a
    #: relink, not a rename. docs/adr/0013
    prefix: str = "blobs"
    #: The catalog's blob server, which signs its own URLs. None means the
    #: mount.
    urls: SignedUrls | None = None

    def url_for(self, sample: SampleRow) -> str:
        """Where Label Studio fetches this sample's bytes, whatever it is."""
        if self.urls is None:
            return blob_url(sample, self.prefix)
        return self.urls.url_for(sample)

    def checksum_from(self, url: str) -> str | None:
        """The sample a task's URL names, whichever form it is in."""
        if LOCAL_FILES in url:
            return checksum_from_url(url, self.prefix)
        return SignedUrls.checksum_from(url)


def _digest_or_none(stem: str) -> str | None:
    if len(stem) != 64 or any(c not in "0123456789abcdef" for c in stem):
        return None
    return stem


def blob_url(sample: SampleRow, prefix: str) -> str:
    """Where Label Studio fetches a sample's bytes.

    Built from the checksum, not from where the sample's bytes currently
    sit (``docs/adr/0001``).

    Percent-encoded, because a path that reaches a query string unescaped
    breaks on characters a checksum will never contain but a suffix might.
    """
    return f"{LOCAL_FILES}{prefix}/{quote(blob_path(sample.checksum, suffix_of(sample)))}"


def checksum_from_url(url: str, prefix: str) -> str | None:
    """The sample a task's URL names, or None if it names none.

    None is the ordinary answer for a task created before the catalog: its
    URL addresses the old data root, whose filenames are not checksums.
    """
    if LOCAL_FILES not in url:
        return None
    tail = unquote(url.split(LOCAL_FILES, 1)[1])
    marker = f"{prefix}/"
    if not tail.startswith(marker):
        return None
    # The fan-out directories carry no information the name does not, so the
    # stem is the whole answer — and checking it looks like a digest is what
    # keeps a path that merely sits under the prefix from being taken for a
    # sample.
    return _digest_or_none(Path(tail[len(marker) :]).stem)


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
    #: being non-empty. docs/adr/0009
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
    addressing: "Addressing",
) -> list[Task]:
    """Tasks for a batch of samples, carrying any annotation they already have.

    Everything already answered arrives answered. See ``docs/adr/0029``.
    """
    tasks = []
    for sample in samples:
        value = catalog.annotations.annotation_of(sample.id, label_set_id)
        tasks.append(
            Task(
                sample_id=sample.id,
                data={schema.data_key: addressing.url_for(sample)},
                annotations=to_results(value, schema) if value is not None else [],
                answered=value is not None,
            )
        )
    return tasks
