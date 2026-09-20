"""Validate a submission against the stated contract."""
import sys, re, unicodedata
import pandas as pd

def main(sub_path, test_path, sample_path):
    sub = pd.read_csv(sub_path, dtype=str, keep_default_na=False)
    test = pd.read_csv(test_path, dtype=str, keep_default_na=False)
    samp = pd.read_csv(sample_path, dtype=str, keep_default_na=False)
    fails = []
    if list(sub.columns) != ["id", "witnesses", "question"]:
        fails.append(f"columns are {list(sub.columns)}, expected ['id','witnesses','question']")
    if len(sub) != 190:
        fails.append(f"{len(sub)} rows, expected 190")
    want = set(samp["id"]); got = set(sub["id"])
    if want != got:
        fails.append(f"id mismatch: missing {len(want-got)}, extra {len(got-want)}")
    if sub["id"].duplicated().any():
        fails.append("duplicate ids")
    rc = dict(zip(test["id"], test["record_count"].astype(int)))
    for r in sub.itertuples(index=False):
        toks = r.witnesses.split(" ")
        n = rc.get(r.id)
        if not (2 <= len(toks) <= 4):
            fails.append(f"{r.id}: {len(toks)} witness tokens"); continue
        if len(set(toks)) != len(toks):
            fails.append(f"{r.id}: duplicate witness tokens")
        idx = []
        for t in toks:
            if not re.fullmatch(r"w[0-5]", t):
                fails.append(f"{r.id}: bad token {t!r}"); break
            idx.append(int(t[1:]))
        else:
            if idx != sorted(idx):
                fails.append(f"{r.id}: witnesses not in increasing order")
            if n is not None and max(idx) >= n:
                fails.append(f"{r.id}: token beyond record_count {n}")
            if n is not None and len(idx) != n - 2:
                fails.append(f"{r.id}: {len(idx)} witnesses for record_count {n}")
        if r.witnesses != " ".join(toks) or "  " in r.witnesses:
            fails.append(f"{r.id}: witness separator not a single ASCII space")
        q = r.question
        if not q:
            fails.append(f"{r.id}: empty question")
        if q != q.strip():
            fails.append(f"{r.id}: question has leading/trailing whitespace")
        if len(q) > 512:
            fails.append(f"{r.id}: question {len(q)} chars > 512")
        if any(ord(c) < 32 or ord(c) == 127 for c in q):
            fails.append(f"{r.id}: control character in question")
    print(f"rows={len(sub)}  unique_ids={sub['id'].nunique()}")
    if fails:
        print(f"FAILED ({len(fails)} problems):")
        for f in fails[:20]:
            print("  -", f)
        return 1
    print("SUBMISSION VALID: columns, ids, witness tokens and question field all conform")
    ql = sub["question"].str.split().str.len()
    print(f"question words: mean={ql.mean():.1f} min={ql.min()} max={ql.max()}")
    print(f"witness sizes: {sub['witnesses'].str.split().str.len().value_counts().to_dict()}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:4]))
