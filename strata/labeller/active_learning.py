"""Ordering the review queue so the most informative samples come first.

The strategies live here rather than with the value types, because which one
to use is a choice about how to spend review time and not a fact about what
a prediction is. A prediction carries its confidences; what to make of them
is active learning's business.

Each takes a strata.labels prediction and returns a number where higher
means "ask a human sooner".
"""

import math

from strata.labels import Value

from .schemas import Prediction


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


#: Selectable by name, so a strategy can be a setting rather than an edit.
STRATEGIES = {
    "least-confident": least_confident,
    "margin": margin,
    "entropy": entropy,
}


def rank_by_uncertainty(predictions: list[Prediction]) -> list[Prediction]:
    """Most uncertain first."""
    return sorted(predictions, key=lambda p: p.uncertainty, reverse=True)


def select_for_review(
    predictions: list[Prediction],
    n: int | None = None,
    threshold: float | None = None,
) -> list[Prediction]:
    ranked = rank_by_uncertainty(predictions)
    if threshold is not None:
        ranked = [p for p in ranked if p.score < threshold]
    if n is not None:
        ranked = ranked[:n]
    return ranked
