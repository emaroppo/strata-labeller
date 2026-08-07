"""Ordering the review queue so the most informative samples come first.

What "uncertain" means depends on the task — a classifier's least-confident
choice, a detector's boxes near the threshold — so each schema computes it
and this module only sorts.
"""

from .schemas import Prediction


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
