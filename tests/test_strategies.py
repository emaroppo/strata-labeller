"""Ranking strategies for the review queue.

Each answers "how badly does this want a human", higher meaning sooner. They
disagree deliberately: which one to use is a choice about how to spend
review time, which is why they live here and not with the value types.
"""

import pytest

from strata.labeller.review.active_learning import (
    STRATEGIES,
    entropy,
    least_confident,
    margin,
)
from strata.labels import ChoicesPrediction


def prediction(*confidences: float) -> ChoicesPrediction:
    return ChoicesPrediction(
        values=[f"c{i}" for i in range(len(confidences))], confidences=list(confidences)
    )


# ----------------------------------------------------------------------
# Least confident
# ----------------------------------------------------------------------


def test_a_confident_prediction_scores_low():
    assert least_confident(prediction(0.99)) == pytest.approx(0.01)


def test_an_unsure_prediction_scores_high():
    assert least_confident(prediction(0.4)) == pytest.approx(0.6)


def test_only_the_best_guess_counts():
    assert least_confident(prediction(0.9, 0.1)) == least_confident(prediction(0.9))


def test_a_prediction_of_nothing_is_maximally_uncertain():
    # Either there was nothing to find or the model missed everything, and
    # only a human settles which
    assert least_confident(ChoicesPrediction()) == 1.0


# ----------------------------------------------------------------------
# Margin
# ----------------------------------------------------------------------


def test_two_close_guesses_score_high():
    # Confident in two classes at once is certainty about the wrong question
    assert margin(prediction(0.51, 0.49)) == pytest.approx(0.98)


def test_a_clear_winner_scores_low():
    assert margin(prediction(0.95, 0.05)) == pytest.approx(0.1)


def test_margin_catches_what_least_confident_misses():
    torn = prediction(0.9, 0.88)
    # Confident by one measure, badly split by the other — the case the two
    # strategies exist to tell apart
    assert least_confident(torn) < 0.2
    assert margin(torn) > 0.9


def test_a_single_guess_has_no_margin_to_measure():
    assert margin(prediction(0.9)) == 1.0


def test_a_prediction_of_nothing_has_no_margin_either():
    assert margin(ChoicesPrediction()) == 1.0


# ----------------------------------------------------------------------
# Entropy
# ----------------------------------------------------------------------


def test_spread_belief_scores_higher_than_concentrated():
    assert entropy(prediction(0.34, 0.33, 0.33)) > entropy(prediction(0.98, 0.01, 0.01))


def test_entropy_reads_every_guess():
    # Which is what suits multi-label work, where several classes are meant
    # to fire at once
    assert entropy(prediction(0.5, 0.5)) > entropy(prediction(0.5))


def test_a_prediction_of_nothing_is_maximally_uncertain_by_entropy():
    assert entropy(ChoicesPrediction()) == 1.0


def test_zero_confidences_do_not_blow_up():
    # log(0) is the obvious way to make this crash
    assert entropy(prediction(0.9, 0.0)) == pytest.approx(entropy(prediction(0.9)))


# ----------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", ["least-confident", "margin", "entropy"])
def test_every_strategy_is_selectable_by_name(name):
    # So it can be a setting rather than an edit
    assert STRATEGIES[name](prediction(0.6, 0.4)) >= 0.0


@pytest.mark.parametrize("strategy", [least_confident, margin, entropy])
def test_ranking_puts_the_least_settled_first(strategy):
    confident = prediction(0.99, 0.01)
    torn = prediction(0.51, 0.49)
    assert sorted([confident, torn], key=strategy, reverse=True)[0] is torn


def test_a_prediction_without_confidences_still_ranks():
    # A model may name a class and say nothing about how sure it is; that
    # must not be an exception halfway through sorting a queue
    bare = ChoicesPrediction(values=["cat"])
    assert all(strategy(bare) == 1.0 for strategy in STRATEGIES.values())
