def sub_cfg(cfg, which):
    """Shared settings plus the per-model overrides."""
    out = {k: v for k, v in cfg.items() if k not in ("wit", "qst")}
    out.update(cfg[which])
    out.setdefault("gen_temp", 1.0)
    out.setdefault("len_norm", 0.8)
    out.setdefault("mbr_tau", 1.0)
    out.setdefault("witness_only", False)
    return out


def build_vocabs(rows, labels, cfg):
    """Vocabularies are fitted on the training ledgers only."""
    sc, tc = Counter(), Counter()
    for row in rows:
        for a in row["ans"]:
            sc.update(a)
        for e in row["evi"]:
            sc.update(e)
        tc.update(words(labels[row["id"]]["question"]))
    return (Vocab(sc, cfg["src_min_freq"], cfg["src_cap"]),
            Vocab(tc, cfg["tgt_min_freq"], cfg["tgt_cap"]))


def fit(train_ex, cfg, nsrc, ntgt, which, seed):
    """MLM-pretrain the encoder on the supplied text, then train the task model."""
    c = sub_cfg(cfg, which)
    model = Model(c, nsrc, ntgt, seed)
    if c.get("pretrain"):
        pretrain_encoder(model, train_ex, c, seed + 4242)
    return train_model(train_ex, c, nsrc, ntgt, seed, init=model), c


def format_question(text, fallback="what datasets are used?"):
    """Enforce the submission contract for the question field."""
    t = unicodedata.normalize("NFKC", str(text))
    t = "".join(" " if (ord(c) < 32 or ord(c) == 127) else c for c in t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > 512:
        t = t[:512].rstrip()
    return t or fallback


def solve(public_dir, submission_out):
    public_dir = Path(public_dir)
    train = pd.read_csv(public_dir / "train.csv")
    labels_df = pd.read_csv(public_dir / "train_labels.csv")
    test = pd.read_csv(public_dir / "test.csv")
    merged = train.merge(labels_df, on="id", how="inner", validate="one_to_one")
    if len(merged) != len(train):
        raise ValueError("train.csv and train_labels.csv do not join one-to-one")
    labels = {str(r.id): {"witnesses": r.witnesses, "question": r.question}
              for r in labels_df.itertuples(index=False)}

    cfg = dict(CFG)
    train_rows = parse_rows(train)
    test_rows = parse_rows(test)
    # vocabulary and every fitted parameter come from the training ledgers only
    svoc, tvoc = build_vocabs(train_rows, labels, sub_cfg(cfg, "qst"))
    train_ex = build_examples(train_rows, sub_cfg(cfg, "qst"), svoc, tvoc, labels)
    test_ex = build_examples(test_rows, sub_cfg(cfg, "qst"), svoc, tvoc)

    wmodel, wcfg = fit(train_ex, cfg, len(svoc), len(tvoc), "wit", SEED)
    qmodel, qcfg = fit(train_ex, cfg, len(svoc), len(tvoc), "qst", SEED + 101)

    witnesses, questions = {}, {}
    by_r = {}
    for i, e in enumerate(test_ex):
        by_r.setdefault(e["R"], []).append(i)
    for R in sorted(by_r):
        idxs = by_r[R]
        for s in range(0, len(idxs), cfg["infer_batch"]):
            sel = idxs[s:s + cfg["infer_batch"]]
            chosen, qs, _ = predict_two(wmodel, qmodel, [test_ex[i] for i in sel],
                                        wcfg, qcfg, tvoc)
            for k, i in enumerate(sel):
                witnesses[i] = " ".join("w%d" % w for w in sorted(chosen[k]))
                questions[i] = format_question(qs[k])

    sub = pd.DataFrame({
        "id": [e["id"] for e in test_ex],
        "witnesses": [witnesses[i] for i in range(len(test_ex))],
        "question": [questions[i] for i in range(len(test_ex))],
    })
    if len(sub) != len(test) or sub["id"].duplicated().any():
        raise ValueError("submission rows do not match the evaluation ids")
    out = Path(submission_out)
    if out.parent.as_posix() not in ("", "."):
        out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python3 solution.py <public_dir> <submission_out>")
    solve(sys.argv[1], sys.argv[2])
