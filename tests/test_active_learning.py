"""Choosing what a human should look at next.

Ordering a review queue is the only thing here, and it has to work for
whatever a model produces — the confidences are positional against the
values, whether those are classes, spans or boxes.
"""

# ----------------------------------------------------------------------
# Ranking a pool
# ----------------------------------------------------------------------


class Sample:
    def __init__(self, sample_id, checksum):
        self.id = sample_id
        self.checksum = checksum


def scored(*confidences):
    from strata.labels import ChoicesPrediction

    return ChoicesPrediction(
        values=["a"] * len(confidences), confidences=list(confidences)
    )


def test_least_confident_comes_first():
    from strata.labeller.active_learning import rank

    pool = [Sample(1, "a" * 64), Sample(2, "b" * 64), Sample(3, "c" * 64)]
    order = rank(
        pool,
        {"a" * 64: scored(0.99), "b" * 64: scored(0.51), "c" * 64: scored(0.80)},
    )
    # What the model committed to least is what a human settles fastest
    assert [s.id for s in order] == [2, 3, 1]


def test_an_unscored_sample_is_left_out():
    from strata.labeller.active_learning import rank

    pool = [Sample(1, "a" * 64), Sample(2, "b" * 64)]
    order = rank(pool, {"a" * 64: scored(0.9)})
    # Its place in a queue claiming to be least-confident-first would be a
    # fiction; it is still unlabelled, so it comes back next time
    assert [s.id for s in order] == [1]


def test_nothing_scored_ranks_nothing():
    from strata.labeller.active_learning import rank

    assert rank([Sample(1, "a" * 64)], {}) == []


def test_certainty_is_the_inverse_of_least_confident():
    from strata.labeller.active_learning import certainty, least_confident

    p = scored(0.3, 0.85, 0.6)
    # The number shown beside a pre-annotation and the order it arrives in
    # have to tell the same story
    assert certainty(p) == 0.85
    assert abs(certainty(p) - (1.0 - least_confident(p))) < 1e-9


def test_certainty_is_not_the_first_confidence():
    from strata.labeller.active_learning import certainty

    # Confidences are positional against values, so the first one is
    # whichever class or box happened to come first — meaningless alone
    assert certainty(scored(0.1, 0.9)) == 0.9


def test_nothing_asserted_is_no_certainty():
    from strata.labeller.active_learning import certainty
    from strata.labels import ChoicesPrediction

    assert certainty(ChoicesPrediction()) == 0.0
