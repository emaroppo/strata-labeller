"""The Label Studio boundary.

Past this module a sample is a catalog row and an annotation a :mod:`strata.contracts`
value; inside it, results carry control names, media keys and percentages. A task's URL
names the checksum and its data the type. See ``docs/adr/0013`` and ``docs/adr/0052``.
"""

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, unquote

from strata.catalog import Catalog, SampleRow, SignedUrls, blob_path, suffix_of
from strata.contracts import AnyPrediction, AnyValue

from .schemas import LabelSchema

#: What Label Studio serves local files under.
LOCAL_FILES = "/data/local-files/?d="

#: The sample's type in task["data"]; prefixed so no config's ``$name`` reads it. docs/adr/0052
TYPE_KEY = "strata_type"


def to_results(value: AnyValue, schema: LabelSchema) -> list[dict]:
    """A neutral value as Label Studio results."""
    return schema.encode_target(list(value.values))


def from_results(results: list[dict], schema: LabelSchema) -> AnyValue:
    """Label Studio results as a neutral value.

    An empty list is an empty value, not nothing: finding none present is an answer.
    See ``docs/adr/0009`` and ``docs/adr/0013``.
    """
    return schema.value_type(values=list(schema.decode_target(results)))


def prediction_to_results(prediction: AnyPrediction, schema: LabelSchema) -> list[dict]:
    """A model's output as Label Studio results; the schema refuses a value of the wrong kind."""
    return schema.encode_target(list(prediction.values))


@dataclass(frozen=True)
class Addressing:
    """How a task refers to its sample, in both directions.

    Writes the mount form or, with ``urls``, the catalog's signed URL; reads both, so
    old tasks resolve until relinked. See ``docs/adr/0013``.
    """

    #: What Label Studio serves the blob mount under; changing it is a relink. docs/adr/0013
    prefix: str = "blobs"
    #: The catalog's blob server, which signs its own URLs; None means the mount.
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

    def data_for(self, sample: SampleRow, data_key: str) -> dict:
        """What a task's data says about its sample: where its bytes are, and its type.

        See ``docs/adr/0052``.
        """
        return {data_key: self.url_for(sample), TYPE_KEY: sample.type}

    def sample_from(self, data: dict, data_key: str, catalog: Catalog) -> SampleRow | None:
        """The sample a task's data names, or None if it names none.

        A task of one type never settles for a sample of another; one without a type
        resolves while one live sample has its bytes and raises
        :class:`~strata.catalog.AmbiguousSample` once several do. See ``docs/adr/0052``.
        """
        checksum = self.checksum_from(data.get(data_key, ""))
        if checksum is None:
            return None
        return catalog.samples.by_checksum(checksum, data.get(TYPE_KEY) or None)


def _digest_or_none(stem: str) -> str | None:
    if len(stem) != 64 or any(c not in "0123456789abcdef" for c in stem):
        return None
    return stem


def blob_url(sample: SampleRow, prefix: str) -> str:
    """Where Label Studio fetches a sample's bytes, built from the checksum.

    Percent-encoded: a suffix can hold characters that break inside a query string.
    See ``docs/adr/0001``.
    """
    return f"{LOCAL_FILES}{prefix}/{quote(blob_path(sample.checksum, suffix_of(sample)))}"


def checksum_from_url(url: str, prefix: str) -> str | None:
    """The sample a task's URL names, or None when the URL addresses no blob."""
    if LOCAL_FILES not in url:
        return None
    tail = unquote(url.split(LOCAL_FILES, 1)[1])
    marker = f"{prefix}/"
    if not tail.startswith(marker):
        return None
    # The stem is the whole answer; checking it is a digest keeps a stray path from passing
    return _digest_or_none(Path(tail[len(marker) :]).stem)


@dataclass
class Task:
    """One Label Studio task, built from a catalog sample."""

    sample_id: int
    data: dict
    annotations: list[dict]
    #: Whether anyone has answered, which is not the same as a non-empty answer. docs/adr/0009
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
                data=addressing.data_for(sample, schema.data_key),
                annotations=to_results(value, schema) if value is not None else [],
                answered=value is not None,
            )
        )
    return tasks
