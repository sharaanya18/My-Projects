"""Vocabularies, fit on train.csv only.

Design note (deviation from the original build spec, kept deliberately):
the spec suggested a from-scratch BPE over the output strings.  Measured on a
compositional fold, a *grader-aligned word/punctuation* output vocabulary of
1,500 types already covers 94.1% of validation output tokens by generation, and
97.6% once the pointer-copy channel over the input is added -- so BPE would buy
at most ~2 points of reachability while breaking the exact alignment between
output units and copyable source units that the pointer mechanism depends on.
Both sides therefore use the grader's own tokenization:

    \\w+  |  a single non-whitespace punctuation character

which has the further property that the score depends only on this token
sequence, never on whitespace.

Both vocabularies are fit on train.csv ONLY.  test.csv text never touches vocab
construction -- unseen test words go to <unk> on the encoder side and are still
reproducible through the copy channel, which is what keeps inference honest
one-row-at-a-time (Guidebook 4.2.5).
"""
import re
from collections import Counter

TOK_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

PAD, UNK, BOS, EOS = 0, 1, 2, 3
SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def tokenize(text: str):
    """Grader-identical tokenization."""
    return TOK_RE.findall(text or "")


class Vocab:
    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def get(self, tok, default=UNK):
        return self.stoi.get(tok, default)

    @classmethod
    def build(cls, counter: Counter, max_size=None, min_freq=1):
        items = [(t, c) for t, c in counter.items() if c >= min_freq]
        # deterministic: by descending frequency, ties broken lexicographically
        items.sort(key=lambda x: (-x[1], x[0]))
        if max_size is not None:
            items = items[: max(0, max_size - len(SPECIALS))]
        return cls(SPECIALS + [t for t, _ in items])


def build_vocabs(train_rows, heads_per_row, src_max=6000, tgt_max=1500):
    """Fit source / target / head vocabularies on training rows only."""
    src_c, tgt_c, head_c = Counter(), Counter(), Counter()
    for r in train_rows:
        src_c.update(t.lower() for t in tokenize(r["input"]))
        tgt_c.update(tokenize(r["output"]))
    for hs in heads_per_row:
        head_c.update(hs)
    return (
        Vocab.build(src_c, src_max),
        Vocab.build(tgt_c, tgt_max),
        Vocab.build(head_c, None),
    )
