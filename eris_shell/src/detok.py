"""Token sequence -> shell string.

The grader scores the token sequence only, so whitespace cannot change the
score.  Detokenization still matters for two things: the submission must pass a
POSIX shlex balance check, and the output has to be readable as a shell command.

Spacing is therefore *fit on the training outputs* rather than hand-written: for
each adjacent token pair we record whether the reference had whitespace between
them, keyed by (prev class, cur class, previous gap, inside-quote, tokens since
last space), with backoff.  This is pure formatting of already-generated
content -- it is re-tokenized and compared afterwards, and falls back to a plain
space join if it would ever change the token sequence, so it provably cannot
affect the score.
"""
import shlex
from collections import Counter, defaultdict

from tokenizer import TOK_RE, tokenize


def _rep(t):
    return "W" if (t[:1].isalnum() or t[:1] == "_") else t


def _state(prev, cur, prev_spaced, in_quote, since):
    return (_rep(prev), _rep(cur), prev_spaced, in_quote, min(since, 3))


class SpacingModel:
    """Deterministic whitespace model fit on training reference strings."""

    def __init__(self):
        self.tables = [defaultdict(Counter) for _ in range(3)]

    @staticmethod
    def _backoff_keys(st):
        p, c, ps, q, si = st
        return [st, (p, c, ps, q), (p, c)]

    def fit(self, outputs):
        for s in outputs:
            sp = [(m.group(0), m.start(), m.end()) for m in TOK_RE.finditer(s)]
            prev_spaced, in_quote, since = True, 0, 0
            for i in range(1, len(sp)):
                gap = s[sp[i - 1][2]: sp[i][1]]
                sep = 1 if (gap != "" and gap.strip() == "") else 0
                p, c = sp[i - 1][0], sp[i][0]
                if p in ("'", '"'):
                    in_quote = 0 if in_quote else 1
                st = _state(p, c, prev_spaced, in_quote, since)
                for t, k in zip(self.tables, self._backoff_keys(st)):
                    t[k][sep] += 1
                prev_spaced = bool(sep)
                since = 0 if sep else since + 1
        return self

    def _space(self, st):
        for t, k in zip(self.tables, self._backoff_keys(st)):
            c = t.get(k)
            if c and sum(c.values()) >= 2:
                return c[1] >= c[0]
        # unseen context: separate only if concatenation would be ambiguous
        return _rep(st[0]) == "W" and _rep(st[1]) == "W"

    def join(self, tokens):
        if not tokens:
            return ""
        out = [tokens[0]]
        prev_spaced, in_quote, since = True, 0, 0
        for i in range(1, len(tokens)):
            p, c = tokens[i - 1], tokens[i]
            if p in ("'", '"'):
                in_quote = 0 if in_quote else 1
            sp = self._space(_state(p, c, prev_spaced, in_quote, since))
            # hard round-trip guarantee: two word tokens must never merge
            if _rep(p) == "W" and _rep(c) == "W":
                sp = True
            out.append((" " if sp else "") + c)
            prev_spaced = sp
            since = 0 if sp else since + 1
        s = "".join(out)
        return s if tokenize(s) == list(tokens) else " ".join(tokens)


def balance_repair(tokens):
    """Drop unmatched quote tokens so the string passes the grader's shlex gate.

    Dropping the dangling quote costs one deletion edit, the same as inserting
    one would, and cannot open a span that swallows the rest of the command.
    """
    toks = list(tokens)
    for q in ("'", '"'):
        idxs = [i for i, t in enumerate(toks) if t == q]
        if len(idxs) % 2 == 1:
            toks.pop(idxs[-1])
    return toks


def finalize(tokens, spacing: SpacingModel, fallback="ls"):
    """Tokens -> a submission-legal command string."""
    toks = [t for t in tokens if t and "\x00" not in t]
    for attempt in range(3):
        if not toks:
            return fallback
        s = spacing.join(toks)
        try:
            shlex.split(s)
        except ValueError:
            toks = balance_repair(toks) if attempt == 0 else [t for t in toks if t not in ("'", '"')]
            continue
        return (s.strip() or fallback)[:4096]
    return fallback
