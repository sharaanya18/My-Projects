"""Parse the Shipd profile_text into a tabular frame. Shared by every experiment."""
from __future__ import annotations
import pandas as pd, numpy as np
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "raw"

CAT_COLS = ['title_length_band','title_word_band','title_question_band','body_length_band',
            'body_word_band','body_structure','body_question_band','body_reference_band',
            'body_mention_band','backport_term','release_term','test_term','bug_term',
            'docs_term','security_term','base_channel','change_scope']

def parse_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = [dict(t.split('=', 1) for t in s.split() if '=' in t) for s in df.profile_text]
    out = pd.DataFrame(rows)
    out.insert(0, 'id', df.id.values)
    for c in out.columns:
        if c == 'id' or c in CAT_COLS:
            continue
        out[c] = pd.to_numeric(out[c], errors='raise')
    return out

def load(raw=RAW, with_text=False):
    tr = pd.read_csv(raw / "train.csv"); te = pd.read_csv(raw / "test.csv")
    ta = pd.read_csv(raw / "train_targets.csv")
    TR = parse_profile(tr).merge(ta, on='id', how='left')
    TE = parse_profile(te)

    assert TR.target.notna().all()
    TR['target'] = TR.target.astype(int)
    if with_text:
        return TR, TE, tr.profile_text.values, te.profile_text.values
    return TR, TE

NUM_COLS = None
def num_cols(df):
    return [c for c in df.columns if c not in ('id', 'target') and c not in CAT_COLS]
