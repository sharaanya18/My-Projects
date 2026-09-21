"""Phase 5: automated first pass at "what kind of NLP task is this?".

Heuristics over a dataframe. Returns a RANKED SET of candidate families --
a task may belong to several, and hybrids are normal.

This does not replace the task audit in ../problem_patterns.md. It is the
first ten minutes, not the conclusion.

Usage:
    python task_detector.py train.csv [test.csv]
    # or
    from task_detector import detect; detect(train_df, test_df)
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

TEXTY_MIN_AVG_LEN = 15          # avg chars before a column counts as "text"
ID_UNIQ_RATIO = 0.95
GROUP_MIN_REPEAT = 1.5          # avg rows per distinct value to count as a group
GROUP_MAX_REPEAT = 200.0        # above this it is a low-cardinality target, not a group
GROUP_MIN_DISTINCT = 8          # a group key needs real cardinality


def _group_name_bonus(name: str) -> int:
    low = name.lower()
    if any(k in low for k in ("query", "qid", "group", "session", "doc", "topic")):
        return 2
    if low.endswith("_id") or low.endswith("id"):
        return 1
    return 0


def _is_stringy(s: pd.Series) -> bool:
    """True for object- or string-dtype columns.

    pandas >= 3.0 stores text as `str` dtype, not `object`. Checking
    `dtype == object` alone silently misses every text column there.
    """
    return pd.api.types.is_string_dtype(s) or s.dtype == object


def _is_text_col(s: pd.Series) -> bool:
    if not _is_stringy(s):
        return False
    vals = s.dropna().astype(str)
    if vals.empty:
        return False
    return vals.str.len().mean() >= TEXTY_MIN_AVG_LEN


def _looks_like_id(name: str, s: pd.Series, n: int) -> bool:
    if "id" in name.lower() or name.lower().endswith("_key"):
        return True
    return s.nunique(dropna=True) / max(n, 1) >= ID_UNIQ_RATIO


def profile(df: pd.DataFrame) -> dict:
    n = len(df)
    text_cols, id_cols, group_cols, numeric_cols, cat_cols = [], [], [], [], []

    for c in df.columns:
        s = df[c]
        if _is_text_col(s):
            text_cols.append(c)
            continue
        if _looks_like_id(c, s, n):
            id_cols.append(c)
            continue
        if pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(c)
        else:
            cat_cols.append(c)

    for c in df.columns:
        if c in text_cols:
            continue
        nu = int(df[c].nunique(dropna=True))
        rep = n / nu if nu else 0.0
        # A group key has MANY distinct values, each repeated a FEW times
        # (60 queries x 10 candidates). A 3-class target repeated 100x is a
        # target, not a group -- require real cardinality, not just repetition.
        if (nu >= GROUP_MIN_DISTINCT and nu < n
                and GROUP_MIN_REPEAT <= rep <= GROUP_MAX_REPEAT):
            group_cols.append((c, nu, round(rep, 2)))
    # prefer name-suggestive keys, then higher cardinality
    group_cols.sort(key=lambda x: (-_group_name_bonus(x[0]), -x[1]))

    return {
        "n_rows": n,
        "columns": list(df.columns),
        "text_columns": text_cols,
        "id_columns": id_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": cat_cols,
        "group_candidates": group_cols[:10],
        "text_length_stats": {
            c: {
                "mean_chars": float(df[c].dropna().astype(str).str.len().mean()),
                "p95_chars": float(df[c].dropna().astype(str).str.len().quantile(0.95)),
                "max_chars": int(df[c].dropna().astype(str).str.len().max()),
            }
            for c in text_cols
        },
    }


def _target_shape(df: pd.DataFrame, target: str) -> dict:
    s = df[target].dropna()
    nu = s.nunique()
    out = {"target": target, "n_unique": int(nu)}

    if _is_stringy(s) and s.astype(str).str.contains(r"[,\|\[]").mean() > 0.3:
        out["kind"] = "multilabel"
    elif pd.api.types.is_float_dtype(s) and nu > 20:
        out["kind"] = "continuous"
        out["min"], out["max"] = float(s.min()), float(s.max())
    elif nu == 2:
        out["kind"] = "binary"
    elif nu <= 50:
        out["kind"] = "multiclass"
    else:
        out["kind"] = "high-cardinality (id-like? ranking label? check)"

    if out["kind"] in ("binary", "multiclass"):
        vc = s.value_counts(normalize=True)
        out["class_balance"] = {str(k): round(float(v), 4) for k, v in vc.head(10).items()}
        out["imbalance_ratio"] = round(float(vc.max() / vc.min()), 2)
    return out


def detect(df: pd.DataFrame, test_df: pd.DataFrame | None = None,
           target: str | None = None) -> dict:
    prof = profile(df)
    n_text = len(prof["text_columns"])

    if target is None:
        for cand in ("target", "label", "y", "class", "relevance", "score", "is_duplicate"):
            if cand in df.columns:
                target = cand
                break
        if target is None:
            non_text = [c for c in df.columns
                        if c not in prof["text_columns"] and c not in prof["id_columns"]]
            target = non_text[-1] if non_text else None

    tinfo = _target_shape(df, target) if target and target in df.columns else {}

    # the target is never a group key
    prof["group_candidates"] = [g for g in prof["group_candidates"] if g[0] != target]

    signals, families = [], {}

    def add(family: str, weight: int, why: str):
        families[family] = families.get(family, 0) + weight
        signals.append(f"[{family}] {why}")

    # --- prediction unit from the text columns -------------------------------
    if n_text == 0:
        add("not-text / tabular", 3, "no column looks like free text")
    elif n_text == 1:
        add("classification", 3, f"single text column: {prof['text_columns'][0]}")
    elif n_text == 2:
        add("sentence-pair", 4, f"exactly two text columns: {prof['text_columns']}")
        add("semantic-similarity", 3, "two texts per row -> pair scoring")
        add("reranking", 2, "two texts per row could be (query, candidate)")
    else:
        add("multi-field classification", 2,
            f"{n_text} text columns -- test concatenating them (HF example: 0.596 -> 0.659)")

    # --- group structure -----------------------------------------------------
    if prof["group_candidates"]:
        c, nu, rep = prof["group_candidates"][0]
        if rep >= 2:
            add("ranking / learning-to-rank", 4,
                f"column '{c}' repeats ~{rep} rows/value -> candidate GROUP key")
            add("retrieval", 2, f"'{c}' may be a query id")
            if tinfo.get("kind") in ("binary", "multiclass", "continuous"):
                add("ranking / learning-to-rank", 2,
                    "group structure + a per-row score/label = graded relevance")
            signals.append(f"[validation] group folds on '{c}', NOT random KFold")

    # --- target shape --------------------------------------------------------
    k = tinfo.get("kind")
    if k == "binary":
        add("binary classification", 3, "target has 2 values")
        if prof["group_candidates"]:
            add("ranking / learning-to-rank", 2,
                "binary target + group structure = relevance labels")
        signals.append("[action] tune the decision threshold on OOF -- often the largest gain")
    elif k == "multiclass":
        add("multiclass classification", 3, f"target has {tinfo['n_unique']} values")
    elif k == "multilabel":
        add("multilabel classification", 4, "target values contain separators")
        signals.append("[action] tune ONE threshold PER LABEL on OOF")
    elif k == "continuous":
        add("regression / graded similarity", 3,
            f"continuous target in [{tinfo.get('min')}, {tinfo.get('max')}]")
        if n_text == 2:
            add("semantic-similarity", 3, "two texts + continuous target = STS")
    elif k and k.startswith("high-cardinality"):
        add("retrieval / linking", 2,
            "target has very high cardinality -- is it a document/entity id?")
        signals.append("[warn] do NOT model this as classification over ids "
                       "(see antipatterns.md #8)")

    if tinfo.get("imbalance_ratio", 1) >= 10:
        signals.append(f"[warn] class imbalance {tinfo['imbalance_ratio']}:1 -- "
                       "check the metric, use class_weight, tune thresholds")

    # --- long text -----------------------------------------------------------
    for c, st in prof["text_length_stats"].items():
        if st["p95_chars"] > 2000:
            signals.append(f"[warn] '{c}' p95={int(st['p95_chars'])} chars -- "
                           "chunking/truncation is a real decision here")
            add("document retrieval / long-context", 1, f"'{c}' holds long documents")

    # --- question-ish columns ------------------------------------------------
    lower = [c.lower() for c in df.columns]
    if any("question" in c for c in lower):
        add("question answering", 5, "a column is named like a question")
        if any(x in " ".join(lower) for x in ("context", "passage", "document", "answer")):
            add("extractive QA / retrieval-QA", 3, "question + context/passage columns")
    if any("query" in c for c in lower):
        add("retrieval", 3, "a column is named like a query")
    if any(x in " ".join(lower) for x in ("entity", "mention", "span", "relation", "sense")):
        add("structured prediction / extraction", 3,
            "entity/mention/span/relation/sense columns present")

    # --- train vs test -------------------------------------------------------
    if test_df is not None:
        tprof = profile(test_df)
        missing = set(prof["columns"]) - set(tprof["columns"])
        signals.append(f"[audit] columns in train but not test: {sorted(missing) or 'none'}")
        signals.append(f"[audit] train rows {prof['n_rows']} vs test rows {tprof['n_rows']}")
        signals.append("[action] run validation.adversarial_validation() before modelling")

    ranked = sorted(families.items(), key=lambda kv: -kv[1])
    return {
        "profile": prof,
        "target_info": tinfo,
        "candidate_families": ranked,
        "signals": signals,
        "next_steps": _next_steps(ranked),
    }


def _next_steps(ranked) -> list[str]:
    if not ranked:
        return ["No family inferred -- do the manual audit in problem_patterns.md."]
    book = {
        "ranking / learning-to-rank": "playbooks/ranking.md",
        "retrieval": "playbooks/retrieval.md",
        "document retrieval / long-context": "playbooks/retrieval.md",
        "semantic-similarity": "playbooks/similarity.md",
        "sentence-pair": "playbooks/similarity.md",
        "reranking": "playbooks/ranking.md",
        "binary classification": "playbooks/classification.md",
        "multiclass classification": "playbooks/classification.md",
        "multilabel classification": "playbooks/classification.md",
        "classification": "playbooks/classification.md",
        "multi-field classification": "playbooks/classification.md",
        "question answering": "techniques/question_answering.md",
        "extractive QA / retrieval-QA": "techniques/question_answering.md",
        "structured prediction / extraction": "playbooks/structured_nlp.md",
        "retrieval / linking": "playbooks/structured_nlp.md",
    }
    out, seen = [], set()
    for fam, _ in ranked[:4]:
        pb = book.get(fam)
        if pb and pb not in seen:
            seen.add(pb)
            out.append(f"read {pb}  (matched: {fam})")
    out.append("fill in templates/SHIPD_TASK_ANALYSIS.md before writing model code")
    out.append("build the validation split BEFORE the first model")
    return out


def _report(res: dict) -> str:
    L = ["=" * 72, "TASK DETECTION", "=" * 72]
    p = res["profile"]
    L += [f"rows: {p['n_rows']}",
          f"text columns:     {p['text_columns']}",
          f"id columns:       {p['id_columns']}",
          f"group candidates: {[(c, r) for c, _, r in p['group_candidates'][:5]]}"]
    if res["target_info"]:
        L += ["", f"target: {res['target_info']}"]
    L += ["", "-" * 72, "CANDIDATE FAMILIES (ranked; a task may be several)", "-" * 72]
    for fam, w in res["candidate_families"]:
        L.append(f"  {w:>3}  {fam}")
    L += ["", "-" * 72, "SIGNALS", "-" * 72] + [f"  {s}" for s in res["signals"]]
    L += ["", "-" * 72, "NEXT STEPS", "-" * 72] + [f"  {s}" for s in res["next_steps"]]
    return "\n".join(L)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    train = pd.read_csv(sys.argv[1])
    test = pd.read_csv(sys.argv[2]) if len(sys.argv) > 2 else None
    print(_report(detect(train, test)))
