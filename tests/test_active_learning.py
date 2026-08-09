from strata.labeller.active_learning import rank_by_uncertainty, select_for_review
from strata.labeller.schemas import Prediction


def prediction(path: str, score: float, uncertainty: float) -> Prediction:
    return Prediction(path=path, results=[], score=score, uncertainty=uncertainty)


def test_ranking_puts_the_most_uncertain_first():
    predictions = [
        prediction("sure.jpg", 0.9, 0.1),
        prediction("unsure.jpg", 0.4, 0.6),
        prediction("middling.jpg", 0.7, 0.3),
    ]
    ranked = rank_by_uncertainty(predictions)
    assert [p.path for p in ranked] == ["unsure.jpg", "middling.jpg", "sure.jpg"]


def test_ranking_leaves_the_input_alone():
    predictions = [prediction("a.jpg", 0.9, 0.1), prediction("b.jpg", 0.4, 0.6)]
    rank_by_uncertainty(predictions)
    assert [p.path for p in predictions] == ["a.jpg", "b.jpg"]


def test_ranking_an_empty_queue():
    assert rank_by_uncertainty([]) == []


def test_selection_truncates_after_ranking():
    predictions = [
        prediction("sure.jpg", 0.9, 0.1),
        prediction("unsure.jpg", 0.4, 0.6),
        prediction("middling.jpg", 0.7, 0.3),
    ]
    assert [p.path for p in select_for_review(predictions, n=2)] == [
        "unsure.jpg",
        "middling.jpg",
    ]


def test_selection_without_limits_is_just_the_ranking():
    predictions = [prediction("a.jpg", 0.9, 0.1), prediction("b.jpg", 0.4, 0.6)]
    assert select_for_review(predictions) == rank_by_uncertainty(predictions)


def test_the_threshold_filters_on_score_while_the_order_comes_from_uncertainty():
    """The two are independent fields, and only agree for classification.

    A classifier sets uncertainty to 1 - score, but a detector scores its
    weakest box and peaks its uncertainty at the decision threshold, so a
    confident detection can sit at score 0.9 and uncertainty 0.2. Whether
    that is intended is still open (see TODO): this pins what it does.
    """
    predictions = [
        prediction("weak-box.jpg", 0.3, 0.2),
        prediction("borderline.jpg", 0.55, 1.0),
        prediction("confident.jpg", 0.9, 0.2),
    ]
    selected = select_for_review(predictions, threshold=0.6)
    assert [p.path for p in selected] == ["borderline.jpg", "weak-box.jpg"]


def test_the_threshold_and_the_limit_compose():
    predictions = [
        prediction("a.jpg", 0.1, 0.9),
        prediction("b.jpg", 0.2, 0.8),
        prediction("c.jpg", 0.95, 0.05),
    ]
    assert [p.path for p in select_for_review(predictions, n=1, threshold=0.5)] == ["a.jpg"]


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
