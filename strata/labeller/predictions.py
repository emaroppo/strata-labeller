"""Predictions already made, so a push does not make them again.

Ranking a review queue needs a score for every unlabelled sample, not just
the ones about to be shown — least-confident-first cannot pick a top 200
without having looked at all 54,000. That is a full inference pass per
push, and pushing twice from the same checkpoint pays it twice for an
identical answer.

**Nothing here is ever invalidated, and that is a property rather than an
omission.** A prediction is a function of a checkpoint and some bytes.
Checkpoints are immutable — a run writes one and never rewrites it — and
blobs are addressed by content. So an entry keyed on both cannot go stale;
the only reason to drop one is disk.

Keyed on the checksum rather than the sample id, for the same reason task
URLs are: an id belongs to one catalog's numbering, while the bytes are the
thing the model actually saw. A corpus copied between catalogs keeps its
predictions.

This lives in the labeller because active learning does. The catalog stores
what people decided, and a guess a model made is not that — `tables.py` is
explicit that predictions do not belong in the annotation table.
"""

from pathlib import Path

from sqlalchemy import (
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    func,
    select,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from strata.labels import ChoicesPrediction

metadata = MetaData()

prediction = Table(
    "prediction",
    metadata,
    Column("run_id", Integer, primary_key=True),
    Column("checksum", String(64), primary_key=True),
    # The value as written by the model, stored whole rather than split into
    # columns: what a prediction looks like is the label schema's business,
    # and this only has to hand it back unchanged.
    Column("value", Text, nullable=False),
    Column("made_at", Float, nullable=True),
)


class PredictionCache:
    """What a run already said about a sample."""

    def __init__(self, engine):
        self.engine = engine

    @classmethod
    def local(cls, root: Path) -> "PredictionCache":
        """Beside the runs it caches, since it is meaningless without them."""
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        engine = create_engine(f"sqlite:///{root / 'predictions.db'}")
        metadata.create_all(engine)
        return cls(engine)

    def get(self, run_id: int, checksums: list[str]) -> dict[str, ChoicesPrediction]:
        """Whatever of ``checksums`` this run has already answered.

        Read in chunks because SQLite caps how many parameters one statement
        may bind, and a review pool is comfortably past it.
        """
        found: dict[str, ChoicesPrediction] = {}
        if not checksums:
            return found
        with self.engine.connect() as conn:
            for chunk in _chunks(checksums, 500):
                rows = conn.execute(
                    select(prediction.c.checksum, prediction.c.value).where(
                        prediction.c.run_id == run_id,
                        prediction.c.checksum.in_(chunk),
                    )
                )
                for row in rows:
                    found[row.checksum] = ChoicesPrediction.model_validate_json(row.value)
        return found

    def put(self, run_id: int, made: dict[str, ChoicesPrediction], at: float | None = None) -> int:
        """Record what a run said. Rewriting an entry is a no-op by construction."""
        if not made:
            return 0
        rows = [
            {
                "run_id": run_id,
                "checksum": checksum,
                "value": value.model_dump_json(),
                "made_at": at,
            }
            for checksum, value in made.items()
        ]
        with self.engine.begin() as conn:
            for chunk in _chunks(rows, 500):
                # A checkpoint and some bytes give one answer, so a second
                # write of the same key carries the same value. Ignoring the
                # conflict keeps a re-run from failing on work it repeated.
                conn.execute(
                    sqlite_insert(prediction).on_conflict_do_nothing(
                        index_elements=["run_id", "checksum"]
                    ),
                    chunk,
                )
        return len(rows)

    def forget(self, run_id: int) -> None:
        """Drop a run's predictions, for when disk matters more than time."""
        with self.engine.begin() as conn:
            conn.execute(delete(prediction).where(prediction.c.run_id == run_id))

    def counts(self) -> dict[int, int]:
        """How many predictions are held per run."""
        with self.engine.connect() as conn:
            return {
                row[0]: row[1]
                for row in conn.execute(
                    select(prediction.c.run_id, func.count()).group_by(prediction.c.run_id)
                )
            }


def _chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]
