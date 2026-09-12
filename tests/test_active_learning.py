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
    from strata.labeller.review.active_learning import rank

    pool = [Sample(1, "a" * 64), Sample(2, "b" * 64), Sample(3, "c" * 64)]
    order = rank(
        pool,
        {"a" * 64: scored(0.99), "b" * 64: scored(0.51), "c" * 64: scored(0.80)},
    )
    # What the model committed to least is what a human settles fastest
    assert [s.id for s in order] == [2, 3, 1]


def test_an_unscored_sample_is_left_out():
    from strata.labeller.review.active_learning import rank

    pool = [Sample(1, "a" * 64), Sample(2, "b" * 64)]
    order = rank(pool, {"a" * 64: scored(0.9)})
    # Its place in a queue claiming to be least-confident-first would be a
    # fiction; it is still unlabelled, so it comes back next time
    assert [s.id for s in order] == [1]


def test_nothing_scored_ranks_nothing():
    from strata.labeller.review.active_learning import rank

    assert rank([Sample(1, "a" * 64)], {}) == []


def test_certainty_is_the_inverse_of_least_confident():
    from strata.labeller.review.active_learning import certainty, least_confident

    p = scored(0.3, 0.85, 0.6)
    # The number shown beside a pre-annotation and the order it arrives in
    # have to tell the same story
    assert certainty(p) == 0.85
    assert abs(certainty(p) - (1.0 - least_confident(p))) < 1e-9


def test_certainty_is_not_the_first_confidence():
    from strata.labeller.review.active_learning import certainty

    # Confidences are positional against values, so the first one is
    # whichever class or box happened to come first — meaningless alone
    assert certainty(scored(0.1, 0.9)) == 0.9


def test_nothing_asserted_is_no_certainty():
    from strata.labeller.review.active_learning import certainty
    from strata.labels import ChoicesPrediction

    assert certainty(ChoicesPrediction()) == 0.0



# ----------------------------------------------------------------------
# Two questions, not one
# ----------------------------------------------------------------------


def _pool(n_found: int, n_nothing: int):
    """Predictions that assert something, and predictions that assert nothing."""
    from strata.labels import Span, SpansPrediction

    samples, scores = [], {}
    for i in range(n_found):
        c = f"f{i:063x}"
        samples.append(Sample(i, c))
        # Deliberately unsure, but not as "uncertain" as an empty one
        scores[c] = SpansPrediction(
            values=[Span(labels=["PER"], start=0, end=3, text="abc")], confidences=[0.5]
        )
    for i in range(n_nothing):
        c = f"e{i:063x}"
        samples.append(Sample(1000 + i, c))
        scores[c] = SpansPrediction(values=[], confidences=[])
    return samples, scores


def test_empty_predictions_do_not_take_the_whole_queue():
    """The failure this exists for.

    Every strategy scores a prediction with nothing in it at 1.0, so before
    the pools were separated a batch came entirely from them — on one real
    project, fifty documents of a few dozen characters each, while twelve
    thousand with real predictions sat unreachable behind them.
    """
    from strata.labeller.review.active_learning import rank

    samples, scores = _pool(n_found=100, n_nothing=100)
    top = rank(samples, scores)[:50]
    nothing = [s for s in top if not scores[s.checksum].values]
    assert len(nothing) <= 12


def test_the_share_holds_at_every_prefix():
    """A caller taking the top N gets the same mix as one taking all of it."""
    from strata.labeller.review.active_learning import rank

    samples, scores = _pool(n_found=100, n_nothing=100)
    ranked = rank(samples, scores, empty_share=0.2)
    for cut in (5, 10, 25, 50, 100):
        nothing = [s for s in ranked[:cut] if not scores[s.checksum].values]
        assert len(nothing) <= cut * 0.2 + 1


def test_nothing_is_still_reviewed_eventually():
    # Not zero: a document the model missed everything in is worth seeing,
    # and only a reader can tell that from one that is genuinely empty
    from strata.labeller.review.active_learning import rank

    samples, scores = _pool(n_found=100, n_nothing=100)
    ranked = rank(samples, scores)
    assert len(ranked) == 200
    assert any(not scores[s.checksum].values for s in ranked[:20])


def test_a_pool_of_only_empties_is_still_ordered():
    from strata.labeller.review.active_learning import rank

    samples, scores = _pool(n_found=0, n_nothing=5)
    assert len(rank(samples, scores)) == 5


def test_a_pool_with_nothing_empty_is_untouched():
    from strata.labeller.review.active_learning import rank

    samples, scores = _pool(n_found=5, n_nothing=0)
    assert len(rank(samples, scores)) == 5


def test_a_share_outside_a_proportion_is_refused():
    import pytest

    from strata.labeller.review.active_learning import rank

    samples, scores = _pool(n_found=2, n_nothing=2)
    with pytest.raises(ValueError, match="proportion"):
        rank(samples, scores, empty_share=1.5)


def _spans(*confidences):
    from strata.labels import Span, SpansPrediction

    return SpansPrediction(
        values=[Span(labels=["PER"], start=i, end=i + 1) for i in range(len(confidences))],
        confidences=list(confidences),
    )


def test_density_ranks_by_how_much_there_is_to_confirm():
    """The strategy for building a training set rather than refining one.

    Uncertainty avoids dense predictions structurally for spans: a
    document's score is its least certain span, so one carrying fifty of
    them almost surely holds a weak one and never ranks as confident.
    """
    from strata.labeller.review.active_learning import density, rank

    samples = [Sample(i, f"{i:064x}") for i in range(3)]
    scores = {
        samples[0].checksum: _spans(*[0.95]),
        samples[1].checksum: _spans(*[0.95] * 30),
        samples[2].checksum: _spans(*[0.95] * 5),
    }
    assert [s.id for s in rank(samples, scores, density)] == [1, 2, 0]


def test_density_counts_confident_spans_not_every_guess():
    """Counting every span selects the documents the model is most wrong about.

    The first version did, and put forward a document with 400 spans across
    3,896 characters — one per ten — of which 37 cleared 0.9. Deleting a
    wrong span costs what marking a missing one does, so that batch would
    have been slower than a blank page.
    """
    from strata.labeller.review.active_learning import density, rank

    prolific = _spans(*([0.3] * 100))
    careful = _spans(*([0.95] * 10))
    samples = [Sample(0, "a" * 64), Sample(1, "b" * 64)]
    scores = {"a" * 64: prolific, "b" * 64: careful}

    assert density(prolific) == 0.0
    assert density(careful) == 10.0
    assert [s.id for s in rank(samples, scores, density)] == [1, 0]


def test_density_of_an_empty_prediction_is_nothing_to_check():
    from strata.labeller.review.active_learning import density
    from strata.labels import SpansPrediction

    assert density(SpansPrediction(values=[], confidences=[])) == 0.0
