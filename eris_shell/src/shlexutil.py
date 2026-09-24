"""Quote-aware shell scanning helpers shared by data prep and post-processing.

These are *label-construction* and *validity-check* utilities only: they split a
recorded reference into its pipeline stages so the model has stage-level
supervision, and they check the balance of a string the model has already
generated. They never map a description to a command -- all of that is learned.
"""
import re
import shlex

_WORD = re.compile(r"\S+")


def split_top_level_pipes(cmd: str):
    """Split on `|` that is not inside single/double quotes.

    A naive cmd.split('|') would break on pipes inside quoted regexes such as
    awk '/a|b/', which appear in the training references.
    """
    parts, cur, quote, i = [], [], None, 0
    n = len(cmd)
    while i < n:
        c = cmd[i]
        if quote is not None:
            cur.append(c)
            if c == "\\" and quote == '"' and i + 1 < n:
                cur.append(cmd[i + 1])
                i += 2
                continue
            if c == quote:
                quote = None
        else:
            if c in ("'", '"'):
                quote = c
                cur.append(c)
            elif c == "\\" and i + 1 < n:
                cur.append(c)
                cur.append(cmd[i + 1])
                i += 2
                continue
            elif c == "|":
                parts.append("".join(cur))
                cur = []
            else:
                cur.append(c)
        i += 1
    parts.append("".join(cur))
    return [p.strip() for p in parts]


def stage_head(stage: str) -> str:
    """First shell word of a stage (the command head)."""
    s = stage.strip()
    if not s:
        return ""
    try:
        toks = shlex.split(s)
        if toks:
            return toks[0]
    except ValueError:
        pass
    m = _WORD.search(s)
    return m.group(0) if m else ""


def is_balanced(cmd: str) -> bool:
    """POSIX shlex balance check -- the same lexical gate the grader applies."""
    if not cmd or "\x00" in cmd:
        return False
    try:
        shlex.split(cmd)
        return True
    except ValueError:
        return False
