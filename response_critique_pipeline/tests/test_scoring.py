import math

from response_critique_pipeline.scoring import (
    Row,
    average_precision,
    grade,
    grade_submission,
    is_malformed,
    row_score,
    sign,
    to_score,
    to_scores,
)


def test_sign():
    assert sign(0.5) == 1
    assert sign(-0.5) == -1
    assert sign(0.0) == 0


def test_average_precision_no_ties_matches_textbook_formula():
    abs_scores = [0.9, 0.8, 0.7, 0.6, 0.5]
    is_true = [1, 0, 1, 0, 0]
    # rank1 hit(1/1), rank3 hit(2/3) -> psum=1+2/3, R=2
    expected = (1 / 1 + 2 / 3) / 2
    assert math.isclose(average_precision(abs_scores, is_true), expected)


def test_average_precision_fully_tied_equals_prevalence():
    abs_scores = [0.5] * 16
    is_true = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ap = average_precision(abs_scores, is_true)
    assert math.isclose(ap, 3 / 16)


def test_average_precision_fully_tied_two_of_two():
    # smallest possible pool: both tied, one relevant
    ap = average_precision([0.5, 0.5], [1, 0])
    assert math.isclose(ap, 0.5)


def test_average_precision_partial_tie_block():
    abs_scores = [0.9, 0.5, 0.5, 0.1]
    is_true = [1, 1, 0, 0]
    expected = (1 / 1 + (2 / 3)) / 2
    assert math.isclose(average_precision(abs_scores, is_true), expected)


def test_average_precision_no_relevant_returns_zero():
    assert average_precision([0.1, 0.2], [0, 0]) == 0.0


def test_row_score_perfect_prediction():
    labels = [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    scores = list(labels)  # exact match: perfect grounding/routing/calibration
    s = row_score(scores, labels)
    assert math.isclose(s, 1.0, rel_tol=1e-9)


def test_row_score_all_zero_prediction_is_neutral_routing():
    labels = [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    scores = [0.0] * 16
    s = row_score(scores, labels)
    # den == 0 -> route == 0.5 exactly; grounding == prevalence (fully tied)
    assert s > 0.0
    assert s < 0.2


def test_grade_floor():
    assert grade([]) == 0.02


def test_is_malformed():
    assert is_malformed(None)
    assert is_malformed([0.1] * 15)
    assert is_malformed([0.1] * 15 + [float("nan")])
    assert is_malformed([1.5] + [0.0] * 15)
    assert not is_malformed([0.0] * 16)


def test_grade_submission_treats_malformed_row_as_zero():
    labels = [1] + [0] * 15
    good = Row(scores=list(labels), labels=labels)
    bad = Row(scores=[0.1] * 15, labels=labels)  # wrong length -> malformed
    g_good_only = grade([good])
    g_with_bad = grade_submission([good, bad])
    # bad row contributes 0, pulling the mean down from the good-only grade
    assert g_with_bad < g_good_only


def test_to_score_and_to_scores():
    assert math.isclose(to_score(0.2, 0.9), 0.7)
    assert to_scores([0.0, 1.0], [1.0, 0.0]) == [1.0, -1.0]
