"""CLI orchestrator wiring the pipeline stages together.

Usage:
    python -m response_critique_pipeline.pipeline split --train train.csv --out-dir splits/
    python -m response_critique_pipeline.pipeline call-api --csv splits/dev.csv --cache-dir cache/dev
    python -m response_critique_pipeline.pipeline build-scores --csv splits/calibration.csv --cache-dir cache/calibration --out calibration_scores.csv
    python -m response_critique_pipeline.pipeline calibrate --scores-csv calibration_scores.csv --out-dir calibrators/ --method isotonic
    python -m response_critique_pipeline.pipeline diagnose --scores-csv final_check_scores.csv --calibrators-dir calibrators/ --threshold 0.05
    python -m response_critique_pipeline.pipeline submit --test-csv test.csv --scores-csv test_scores.csv --calibrators-dir calibrators/ --out submission.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import api_client, calibration, diagnostics, splitting, submission


def cmd_split(args):
    df = pd.read_csv(args.train)
    parts = splitting.split_dataframe(df, ratios=tuple(args.ratios), seed=args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        path = out_dir / f"{name}.csv"
        part.to_csv(path, index=False)
        print(f"{name}: {len(part)} rows -> {path}")


def cmd_call_api(args):
    df = pd.read_csv(args.csv)
    results = api_client.run_batch(
        df,
        cache_dir=args.cache_dir,
        model=args.model,
        max_workers=args.max_workers,
        force=args.force,
    )
    n_fallback = sum(1 for r in results if r.fallback)
    print(f"completed {len(results)} rows ({n_fallback} fell back to all-zero) -> cache: {args.cache_dir}")


def cmd_build_scores(args):
    df = pd.read_csv(args.csv)
    results = []
    for row_id in df["id"].astype(str):
        if api_client.is_cached(Path(args.cache_dir), row_id):
            results.append(api_client.load_cached(Path(args.cache_dir), row_id))
    missing = len(df) - len(results)
    if missing:
        print(f"warning: {missing} row(s) not yet in cache {args.cache_dir}; run call-api first", file=sys.stderr)

    scores_by_id = api_client.results_to_scores(results)
    df["id"] = df["id"].astype(str)
    df["score"] = df["id"].map(lambda i: scores_by_id[i]["score"] if i in scores_by_id else None)
    df["fallback"] = df["id"].map(lambda i: scores_by_id[i]["fallback"] if i in scores_by_id else True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows -> {args.out}")


def _parse_score_col(df):
    df = df.copy()
    df["score"] = df["score"].map(lambda s: json.loads(s) if isinstance(s, str) else s)
    return df


def cmd_calibrate(args):
    df = _parse_score_col(pd.read_csv(args.scores_csv))
    pairs = calibration.build_pairs_by_domain(df)
    calibrators = calibration.fit_per_domain_calibrators(pairs, method=args.method)
    calibration.save_calibrators(calibrators, args.out_dir)
    for domain, (scores, labels) in pairs.items():
        print(f"{domain}: fit on {len(scores)} (score,label) pairs")
    print(f"saved calibrators -> {args.out_dir}")


def cmd_diagnose(args):
    df = _parse_score_col(pd.read_csv(args.scores_csv))
    if args.calibrators_dir:
        domains = df["domain"].unique().tolist()
        calibrators = calibration.load_calibrators(args.calibrators_dir, domains)
        df["score"] = df.apply(
            lambda row: calibration.apply_calibration(row["score"], row["domain"], calibrators), axis=1
        )
    report = diagnostics.build_report(df, threshold=args.threshold)
    print(diagnostics.format_report(report))


def cmd_submit(args):
    test_df = pd.read_csv(args.test_csv)
    scores_df = _parse_score_col(pd.read_csv(args.scores_csv))
    scores_df["id"] = scores_df["id"].astype(str)

    if args.calibrators_dir:
        domains = scores_df["domain"].unique().tolist()
        calibrators = calibration.load_calibrators(args.calibrators_dir, domains)
        scores_df["score"] = scores_df.apply(
            lambda row: calibration.apply_calibration(row["score"], row["domain"], calibrators), axis=1
        )

    predictions = dict(zip(scores_df["id"], scores_df["score"]))
    test_ids = test_df["id"].astype(str).tolist()
    submission.write_submission(test_ids, predictions, args.out)
    print(f"wrote validated submission -> {args.out}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("split", help="stratified dev/calibration/final_check split of train.csv")
    p.add_argument("--train", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.15, 0.15])
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("call-api", help="run the resumable batched Anthropic caller over a csv")
    p.add_argument("--csv", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--model", default=api_client.DEFAULT_MODEL)
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_call_api)

    p = sub.add_parser("build-scores", help="merge a cache dir into a compact id,score,domain,relevance csv")
    p.add_argument("--csv", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_build_scores)

    p = sub.add_parser("calibrate", help="fit per-domain calibrators on a calibration-split scores csv")
    p.add_argument("--scores-csv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--method", choices=list(calibration.FITTERS), default="isotonic")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("diagnose", help="per-domain grounding/routing/calibration report on final_check")
    p.add_argument("--scores-csv", required=True)
    p.add_argument("--calibrators-dir", default=None)
    p.add_argument("--threshold", type=float, default=0.05)
    p.set_defaults(func=cmd_diagnose)

    p = sub.add_parser("submit", help="validate and write the final submission csv")
    p.add_argument("--test-csv", required=True)
    p.add_argument("--scores-csv", required=True)
    p.add_argument("--calibrators-dir", default=None)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_submit)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
