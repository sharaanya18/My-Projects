import json

import pandas as pd
import pytest

from response_critique_pipeline import api_client


def _valid_json_array(p_a=0.1, p_b=0.1):
    return json.dumps(
        [{"reasoning": f"r{i}", "p_a": p_a, "p_b": p_b} for i in range(16)]
    )


def test_extract_json_array_plain():
    text = _valid_json_array()
    assert api_client.extract_json_array(text) is not None


def test_extract_json_array_code_fenced():
    text = "```json\n" + _valid_json_array() + "\n```"
    parsed = api_client.extract_json_array(text)
    assert parsed is not None and len(parsed) == 16


def test_extract_json_array_with_prose_prefix():
    text = "Here is my answer:\n" + _valid_json_array() + "\nHope that helps!"
    parsed = api_client.extract_json_array(text)
    assert parsed is not None and len(parsed) == 16


def test_extract_json_array_garbage_returns_none():
    assert api_client.extract_json_array("not json at all") is None


def test_validate_parsed_catches_wrong_length():
    parsed = json.loads(_valid_json_array())[:10]
    assert api_client.validate_parsed(parsed) is not None


def test_validate_parsed_catches_non_numeric():
    parsed = json.loads(_valid_json_array())
    parsed[0]["p_a"] = "high"
    assert api_client.validate_parsed(parsed) is not None


def test_validate_parsed_accepts_good_input():
    parsed = json.loads(_valid_json_array())
    assert api_client.validate_parsed(parsed) is None


def test_call_row_succeeds_first_try():
    calls = []

    def fake_caller(system, user):
        calls.append(user)
        return _valid_json_array(p_a=0.0, p_b=0.9)

    result = api_client.call_row(fake_caller, "row1", "[]", "resp a", "resp b", json.dumps(["c"] * 16))
    assert result.fallback is False
    assert result.attempts == 1
    assert len(calls) == 1
    assert result.result[0]["p_b"] == 0.9


def test_call_row_retries_then_succeeds():
    attempts = {"n": 0}

    def flaky_caller(system, user):
        attempts["n"] += 1
        if attempts["n"] < 3:
            return "not valid json"
        return _valid_json_array()

    result = api_client.call_row(flaky_caller, "row1", "[]", "a", "b", json.dumps(["c"] * 16))
    assert result.fallback is False
    assert result.attempts == 3
    assert attempts["n"] == 3


def test_call_row_falls_back_to_all_zero_after_exhausting_retries():
    def always_broken(system, user):
        return "garbage"

    result = api_client.call_row(always_broken, "row1", "[]", "a", "b", json.dumps(["c"] * 16))
    assert result.fallback is True
    assert result.attempts == 3
    assert len(result.result) == 16
    assert all(e["p_a"] == 0.0 and e["p_b"] == 0.0 for e in result.result)


def test_run_batch_is_resumable_and_skips_cached_rows(tmp_path):
    df = pd.DataFrame(
        [
            {
                "id": "row1",
                "context": "[]",
                "response_a": "a",
                "response_b": "b",
                "candidates": json.dumps(["c"] * 16),
            }
        ]
    )
    call_count = {"n": 0}

    def fake_caller(system, user):
        call_count["n"] += 1
        return _valid_json_array()

    cache_dir = tmp_path / "cache"
    results1 = api_client.run_batch(df, str(cache_dir), caller=fake_caller, max_workers=1)
    assert len(results1) == 1
    assert call_count["n"] == 1

    # second run should hit the cache and not call the API again
    results2 = api_client.run_batch(df, str(cache_dir), caller=fake_caller, max_workers=1)
    assert len(results2) == 1
    assert call_count["n"] == 1  # unchanged: no new API calls


def test_results_to_scores():
    r = api_client.RowResult(
        row_id="row1",
        result=[{"reasoning": "x", "p_a": 0.2, "p_b": 0.9}] * 16,
        attempts=1,
        fallback=False,
    )
    scores = api_client.results_to_scores([r])
    assert scores["row1"]["score"][0] == pytest.approx(0.7)
