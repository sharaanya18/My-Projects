import csv
import json

import pytest

from response_critique_pipeline.submission import SubmissionValidationError, write_submission


def test_write_submission_happy_path(tmp_path):
    test_ids = ["item_1", "item_2"]
    predictions = {"item_1": [0.0] * 16, "item_2": [0.1] * 16}
    out = tmp_path / "sub.csv"
    write_submission(test_ids, predictions, str(out))

    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert json.loads(rows[0]["relevance"]) == [0.0] * 16


def test_missing_id_raises():
    with pytest.raises(SubmissionValidationError):
        write_submission(["item_1", "item_2"], {"item_1": [0.0] * 16}, "/dev/null")


def test_extra_unknown_id_raises():
    predictions = {"item_1": [0.0] * 16, "item_ghost": [0.0] * 16}
    with pytest.raises(SubmissionValidationError):
        write_submission(["item_1"], predictions, "/dev/null")


def test_wrong_length_raises():
    with pytest.raises(SubmissionValidationError):
        write_submission(["item_1"], {"item_1": [0.0] * 15}, "/dev/null")


def test_out_of_range_raises():
    scores = [0.0] * 15 + [1.5]
    with pytest.raises(SubmissionValidationError):
        write_submission(["item_1"], {"item_1": scores}, "/dev/null")


def test_non_finite_raises():
    scores = [0.0] * 15 + [float("nan")]
    with pytest.raises(SubmissionValidationError):
        write_submission(["item_1"], {"item_1": scores}, "/dev/null")


def test_duplicate_test_id_raises():
    with pytest.raises(SubmissionValidationError):
        write_submission(
            ["item_1", "item_1"], {"item_1": [0.0] * 16}, "/dev/null"
        )
