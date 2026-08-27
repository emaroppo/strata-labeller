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
    """How much the model is asking to have *confirmed*.

    The others answer "what would teach the model most per document". This
    answers "where is a reviewer's hour worth most", and early on those are
    not the same question.

    Uncertainty sampling avoids dense predictions, and for spans it does so
    structurally: a document scores as its *least* certain span, so anything
    carrying fifty of them almost surely holds a weak one and can never rank
    as confident. Measured on one project, the pool averaged 22 predicted
    spans a document while an uncertainty-ranked batch of sixty averaged
    under one — and reviewing those sparse ones yielded a tenth of the
    training signal per hour that dense ones had.

    **Counting every span was the wrong measure**, and the first version of
    this did. It selected the documents the model was most wrong about: the
    densest was 400 spans across 3,896 characters, one per ten, including a
    hundred and ten phone numbers in a political email and organisations cut
    to "Campaign" and "federal". Deleting a wrong span costs what marking a
    missing one does, so that batch would have been slower than a blank page.

    Confidence separates the two cleanly. In that document only 37 spans
    cleared 0.9; ranking on the confident ones instead puts forward a
    document with 221 spans of which 187 are confident. Same idea, and it
    stops rewarding a model for guessing more.

    Right while a training set is being built, wrong once it exists: a model
    already good at these documents learns nothing from another. Name it
    deliberately, and stop naming it when that turns.
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
    """Order a review pool, least confident first, dropping the unscored.

    Separate from the command because this is where the wiring meets: three
    producers of predictions — a cache, a local handler, a remote host — and
    a ranking that sorts on whatever they agree to hand over. They have to
    agree, and nothing here can tell whether they do.

    A sample nobody scored is left out rather than sorted on a default. Its
    place in a queue that claims to be least-confident-first would be a
    fiction, and the sample is still unlabelled, so it comes back next time.

    **Two questions, not one.** "Correct what I found" and "confirm there is
    nothing here" are different requests, and every strategy above scores a
    prediction with nothing in it at exactly 1.0 — so they arrive as a block
    of ties at the very front and monopolise the queue rather than competing
    for it. Measured on one span project: 684 documents the model found
    nothing in took the whole of a fifty-task batch, while 12,333 documents
    with real predictions to correct were unreachable behind them. Every one
    of the fifty was a message of a few dozen characters that genuinely
    contained nothing.

    So they are drawn as two pools in a declared proportion. Both are still
    ordered by the strategy; what changes is that neither can crowd the
    other out.

    This never bit classification because a classifier cannot return an
    empty prediction — the image baseline falls back to its best guess when
    nothing clears the threshold — so the rule was written for a case that
    could not arise until a task could honestly assert nothing.
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
