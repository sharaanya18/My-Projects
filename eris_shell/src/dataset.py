"""Turn raw rows into the two supervision signals the model trains on.

Per training row we derive
  * a PLAN target: the ordered list of command heads, e.g. [find, xargs, grep]
  * one REALIZE target per stage: (stage index, head, stage token sequence),
    each paired with the *entire* input description -- the description is never
    pre-segmented, cross-attention decides which part of it a stage needs.

This is a re-cut of supervision that was already given, not new data: no
(input, output) pair is invented anywhere (Guidebook 4.2.6).

Rows are never dropped or down-weighted for containing a rare adjacency --
rare pairs are the closest proxy in the public data for the held-out-pair
phenomenon the test set is built on.
"""
import csv

from shlexutil import split_top_level_pipes, stage_head
from tokenizer import BOS, EOS, PAD, UNK, tokenize

MAX_SRC = 80
MAX_STAGE_TGT = 48
MAX_STAGES = 3


def read_csv(path):
    csv.field_size_limit(10 ** 7)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_row(output: str):
    """(list of stage strings, list of heads) for one reference command."""
    stages = [s for s in split_top_level_pipes(output) if s.strip()]
    if not stages:
        stages = [output.strip()]
    return stages, [stage_head(s) for s in stages]


def build_examples(rows, with_labels=True):
    """One record per row, carrying source tokens plus (optional) targets."""
    out = []
    for r in rows:
        src = tokenize(r["input"])[:MAX_SRC]
        rec = {"id": r["id"], "src_tok": src, "input": r["input"]}
        if with_labels:
            stages, heads = parse_row(r["output"])
            stages, heads = stages[:MAX_STAGES], heads[:MAX_STAGES]
            rec["output"] = r["output"]
            rec["heads"] = heads
            rec["stage_tok"] = [tokenize(s)[:MAX_STAGE_TGT] for s in stages]
        out.append(rec)
    return out


def encode_source(src_tok, src_vocab, tgt_vocab):
    """Encoder ids plus the pointer bookkeeping for the copy channel.

    `ext_ids[i]` is the id the copy distribution puts mass on when it points at
    source position i: the target-vocab id if that surface form is in the target
    vocab, otherwise a per-example extended id (len(tgt_vocab) + k) that decodes
    back to the literal source string.  This is what lets an unseen filename or
    pattern be reproduced verbatim without ever being in any vocabulary.
    """
    enc_ids = [src_vocab.get(t.lower()) for t in src_tok]
    ext_ids, oov_list, oov_pos = [], [], {}
    for t in src_tok:
        tid = tgt_vocab.stoi.get(t)
        if tid is None:
            if t not in oov_pos:
                oov_pos[t] = len(oov_list)
                oov_list.append(t)
            tid = len(tgt_vocab) + oov_pos[t]
        ext_ids.append(tid)
    return enc_ids, ext_ids, oov_list


def encode_stage_target(stage_tok, tgt_vocab, oov_list):
    """Decoder input ids (in-vocab only) and target ids (extended vocab)."""
    oov_index = {t: i for i, t in enumerate(oov_list)}
    dec_in, tgt = [BOS], []
    for t in stage_tok:
        vid = tgt_vocab.stoi.get(t)
        if vid is not None:
            tgt.append(vid)
        elif t in oov_index:
            tgt.append(len(tgt_vocab) + oov_index[t])
        else:
            tgt.append(UNK)
        dec_in.append(vid if vid is not None else UNK)
    tgt.append(EOS)
    return dec_in, tgt
