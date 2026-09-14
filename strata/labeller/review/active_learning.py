"""Ordering the review queue so the most informative samples come first.

Each strategy takes a strata.labels prediction and returns a number where
higher means "ask a human sooner". They live here, not with the value
types, because which to use is a choice about review time. See
``docs/adr/0012``.
"""

import math
from collections.abc import Mapping

from strata.labels import Value


def _confidences(prediction: Value) -> list[float]:
    return list(getattr(prediction, "confidences", None) or [])


def least_confident(prediction: Value) -> float:
    """How far the model's best guess is from certain.

    The default, and the one that needs no tuning: a sample nothing scored
    highly is one the model could not commit to. A prediction with nothing
    in it is maximally uncertain — either there was nothing to find or the
    model missed everything, and only a human settles which.
    """
    scores = _confidences(prediction)
    return 1.0 - max(scores) if scores else 1.0


def margin(prediction: Value) -> float:
    """How close the top two guesses are.

    Catches a different failure from least_confident: a model can be
    confident in two classes at once, which is certainty about the wrong
    question. With fewer than two guesses there is no margin to measure, so
    it falls back to being maximally uncertain.
    """
    scores = sorted(_confidences(prediction), reverse=True)
    if len(scores) < 2:
        return 1.0
    return 1.0 - (scores[0] - scores[1])


def entropy(prediction: Value) -> float:
    """How spread the model's belief is across everything it named.

    Uses every score rather than the top one or two, which suits multi-label
    work where several classes are meant to fire at once.
    """
    scores = [s for s in _confidences(prediction) if s > 0]
    return -sum(s * math.log(s + 1e-10) for s in scores) if scores else 1.0


#: A span the model is this sure of is usually right enough to confirm at a
#: glance. Below it, checking costs about what marking from scratch does.
CONFIDENT = 0.9


def density(prediction: Value, threshold: float = CONFIDENT) -> float:
    """How much the model is asking to have *confirmed*: spans over :data:`CONFIDENT`.

    Where a reviewer's hour is worth most while a training set is being
    built, and the wrong strategy once it exists. Counts confident spans,
    not all of them, so a model is not rewarded for guessing more. See
    ``docs/adr/0012``.
    """
    confidences = _confidences(prediction)
    if not confidences:
        # A model may assert something and say nothing about how sure it is.
        # Absent is unknown, not unconfident, so everything it named counts —
        # scoring it zero would hide such a model's output from this ranking
        # entirely.
        return float(len(getattr(prediction, "values", None) or []))
    return float(sum(1 for c in confidences if c >= threshold))


#: How much of a review batch may be documents the model found nothing in.
#: Not zero: a document it missed everything in is worth seeing, and only a
#: reader can tell that from one that is genuinely empty. Not unbounded
#: either, for the reason `rank` describes.
DEFAULT_EMPTY_SHARE = 0.2

#: Selectable by name, so a strategy can be a setting rather than an edit.
STRATEGIES = {
    "least-confident": least_confident,
    "margin": margin,
    "entropy": entropy,
    "density": density,
}


def _asserted_something(prediction: Value) -> bool:
    return bool(getattr(prediction, "values", None))


def rank(samples, scores, strategy=None, empty_share: float = DEFAULT_EMPTY_SHARE):
    """Order a review pool by ``strategy``, dropping the unscored.

    Samples the model found nothing in are drawn as a second pool, in the
    ``empty_share`` proportion, holding at every prefix of the result. A
    sample nobody scored is left out; it is still unlabelled and comes
    back next time. See ``docs/adr/0012``.
    """
    if not 0.0 <= empty_share <= 1.0:
        raise ValueError(f"empty_share is a proportion, got {empty_share}")

    strategy = strategy or least_confident
    scored = [s for s in samples if s.checksum in scores]
    ordered = sorted(scored, key=lambda s: strategy(scores[s.checksum]), reverse=True)

    found = [s for s in ordered if _asserted_something(scores[s.checksum])]
    nothing = [s for s in ordered if not _asserted_something(scores[s.checksum])]
    if not found or not nothing:
        return ordered

    merged, i, j = [], 0, 0
    while i < len(found) or j < len(nothing):
        # Take from the empty pool only while it is under its share of what
        # has been emitted so far, so the proportion holds at every prefix
        # rather than only over the whole list — a caller taking the top N
        # gets the same mix as one taking all of it.
        take_nothing = j < len(nothing) and (
            i >= len(found) or j < empty_share * (len(merged) + 1)
        )
        if take_nothing:
            merged.append(nothing[j])
            j += 1
        else:
            merged.append(found[i])
            i += 1
    return merged


def disagreement(prediction: Value, label: Value) -> float:
    """How far the model's prediction is from a label the corpus arrived with.

    For a spot review of imported labels: imports are trusted and trained
    on, and the ones worth a person's look are those the model disagrees
    with most. One minus the model's mean confidence in the classes the
    label asserts, so a class the model never named counts as zero
    confidence. Classification only; a label with no classes, or a
    prediction with no confidences, is maximally suspect, since nothing
    can vouch for it.
    """
    asserted = list(getattr(label, "values", None) or [])
    named = list(getattr(prediction, "values", None) or [])
    confidences = _confidences(prediction)
    if not asserted or not confidences or len(confidences) != len(named):
        return 1.0
    by_class = dict(zip(named, confidences, strict=True))
    return 1.0 - sum(by_class.get(c, 0.0) for c in asserted) / len(asserted)


def against(labels: Mapping[str, Value]):
    """A ranking strategy over predictions paired with the labels they are checked against.

    ``rank`` scores a prediction on its own; a disagreement needs the label
    too. The pairing is by the prediction object, which the queue holds one
    of per sample, so this binds each to its label before ranking.
    """
    return Against(labels)


class Against:
    """The ranking :func:`against` builds: callable on a prediction, once bound to its label."""

    def __init__(self, labels: Mapping[str, Value]):
        self._labels = labels
        self._paired: dict[int, Value] = {}

    def bind[V: Value](self, checksum: str, prediction: V) -> V:
        self._paired[id(prediction)] = self._labels[checksum]
        return prediction

    def __call__(self, prediction: Value) -> float:
        label = self._paired.get(id(prediction))
        return disagreement(prediction, label) if label is not None else 1.0


def certainty(prediction: Value) -> float:
    """How sure the model was of its best assertion.

    What a reviewer is shown beside a pre-annotation, and the inverse of
    :func:`least_confident` — so the number on a task and the order it
    arrives in cannot tell different stories. Positional confidences make
    the first one meaningless on its own: for multi-label choices it is
    whichever class came first, and for boxes whichever box did.
    """
    scores = _confidences(prediction)
    return max(scores) if scores else 0.0
