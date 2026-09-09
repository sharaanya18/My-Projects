import json

import pandas as pd

from response_critique_pipeline.diagnostics import build_report, flag_lagging_domains, per_domain_metrics


def _row(domain, labels, scores):
    return {"domain": domain, "relevance": json.dumps(labels), "score": scores}


def test_per_domain_metrics_perfect_domain_scores_one():
    labels = [1, -1] + [0] * 14
    rows = [_row("general", labels, list(labels)) for _ in range(5)]
    df = pd.DataFrame(rows)
    metrics = per_domain_metrics(df)
    assert metrics.loc[metrics["domain"] == "general", "row_score_mean"].iloc[0] > 0.99


def test_flag_lagging_domains_detects_gap():
    labels = [1, -1] + [0] * 14
    good_rows = [_row("general", labels, list(labels)) for _ in range(5)]
    bad_rows = [_row("code", labels, [0.0] * 16) for _ in range(5)]
    df = pd.DataFrame(good_rows + bad_rows)
    report = build_report(df, threshold=0.1)
    flagged_domains = {f.domain for f in report["flags"]}
    assert "code" in flagged_domains
    assert "general" not in flagged_domains


def test_no_flags_when_domains_are_close():
    labels = [1, -1] + [0] * 14
    rows = [_row(d, labels, list(labels)) for d in ["general", "code"] for _ in range(5)]
    df = pd.DataFrame(rows)
    metrics = per_domain_metrics(df)
    flags = flag_lagging_domains(metrics, threshold=0.05)
    assert flags == []
