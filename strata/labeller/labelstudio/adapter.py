"""The Label Studio boundary.

Everything that knows Label Studio's shape stops here. Past this module a
sample is a catalog row and an annotation is a :mod:`strata.labels` value;
inside it, results carry control names, media keys and percentages, and
``schemas/`` does the conversion. A task's sample URL names the checksum.
See ``docs/adr/0013`` and ``docs/adr/0001``.
"""

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

from strata.catalog import Catalog, SampleRow, blob_path
from strata.catalog.storage.signing import DEFAULT_TTL, sign, window_expiry
from strata.labels import AnyValue, Choices, Prediction

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


def from_results(results: list[dict], schema: LabelSchema) -> AnyValue:
    """Label Studio results as a neutral value.

    Built through the schema's own value type, so a bbox schema yields boxes
    rather than a Choices holding objects that are not classes. Reading every
    task type back as one of them is how a corpus annotated with boxes came
    back empty.

    Note what this does with an empty list: it produces an empty value, not
    nothing. A reviewer who looked and found none of the classes present has
    answered the question, and the catalog stores that as a real annotation.
    """
    return schema.value_type(values=list(schema.decode_target(results)))


def prediction_to_results(prediction: Prediction, schema: LabelSchema) -> list[dict]:
    """A model's output as Label Studio results.

    Delegated to the schema, so this is only as general as the schemas are —
    and today only classification is implemented, which encodes class names.
    A boxes prediction reaching here would hand it Box objects; the schema
    refuses them rather than encoding something meaningless.
    """
    return schema.encode_target(list(prediction.values))


# ----------------------------------------------------------------------
# Addressing
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Addressing:
    """How a task refers to its sample, in both directions.

    One object so the two directions cannot disagree. Writes the mount
    form or, with ``base_url``, a signed HTTP URL; reads both, so old tasks
    keep resolving until relinked. See ``docs/adr/0013``.
    """

    #: What Label Studio serves the blob mount under. A directory name that
    #: existing tasks point at, so changing it is a relink, not a rename.
    prefix: str = "blobs"
    #: The serving API, e.g. ``http://minipc:8081``. Empty means the mount.
    base_url: str = ""
    #: Signs blob URLs. Required once ``base_url`` is set — a browser
    #: fetching a sample cannot carry a header, so the URL is the credential.
    secret: str = ""
    ttl: int = DEFAULT_TTL

    def __post_init__(self):
        if self.base_url and not self.secret:
            raise AdapterError(
                "Serving blobs over HTTP needs a signing secret, or the URLs "
                "authorise nothing. Set $STRATA_BLOB_SECRET to the same value "
                "the server was started with."
            )

    def url_for(self, sample: SampleRow) -> str:
        """Where Label Studio fetches this sample's bytes, whatever it is."""
        if not self.base_url:
            return blob_url(sample, self.prefix)
        name = blob_path(sample.checksum, _suffix_of(sample)).rsplit("/", 1)[-1]
        expires = window_expiry(self.ttl)
        signature = sign(sample.checksum, self.secret, expires)
        return f"{self.base_url.rstrip('/')}/blob/{name}?exp={expires}&sig={signature}"

    def checksum_from(self, url: str) -> str | None:
        """The sample a task's URL names, whichever form it is in."""
        if LOCAL_FILES in url:
            return checksum_from_url(url, self.prefix)
        path = urlparse(url).path
        if "/blob/" not in path:
            return None
        return _digest_or_none(Path(path.rsplit("/blob/", 1)[1]).stem)


def _suffix_of(sample: SampleRow) -> str:
    return Path((sample.metadata or {}).get("source_path") or "").suffix.lower()


def _digest_or_none(stem: str) -> str | None:
    if len(stem) != 64 or any(c not in "0123456789abcdef" for c in stem):
        return None
    return stem


def blob_url(sample: SampleRow, prefix: str) -> str:
    """Where Label Studio fetches a sample's bytes.

    Built from the checksum, not from where the sample's bytes currently
    sit. The two agree while blobs are files — the local layout *is* the
    checksum — but they part company the moment those files are packed into
    shards, and a task URL outlives that. Addressing a task by location
    would mean every task in Label Studio silently stopped resolving on the
    day the blobs moved.

    Percent-encoded, because a path that reaches a query string unescaped
    breaks on characters a checksum will never contain but a suffix might.
    """
    return f"{LOCAL_FILES}{prefix}/{quote(blob_path(sample.checksum, _suffix_of(sample)))}"


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
    addressing: "Addressing",
) -> list[Task]:
    """Tasks for a batch of samples, carrying any annotation they already have.

    Sending the existing annotation matters when a project is rebuilt: Label
    Studio is a view of the catalog rather than a second copy of it, so
    everything already answered should arrive answered.
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
