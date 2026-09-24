from __future__ import annotations

import csv
from dataclasses import dataclass


@dataclass
class Cases:
    case_ids: list[str]
    titles: list[str]
    pools: list[list[str]]  # each len 80
    answers: list[set[str]] | None  # None for test


def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_train(train_csv: str, labels_csv: str) -> Cases:
    rows = load_csv(train_csv)
    labels = {r["case_id"]: set(r["assigned_tags"].split()) for r in load_csv(labels_csv)}
    case_ids = [r["case_id"] for r in rows]
    titles = [r["record_title"] for r in rows]
    pools = [r["candidate_tags"].split() for r in rows]
    answers = [labels[cid] for cid in case_ids]
    return Cases(case_ids, titles, pools, answers)


def load_test(test_csv: str) -> Cases:
    rows = load_csv(test_csv)
    case_ids = [r["case_id"] for r in rows]
    titles = [r["record_title"] for r in rows]
    pools = [r["candidate_tags"].split() for r in rows]
    return Cases(case_ids, titles, pools, None)


def build_tag_vocab(pools_list: list[list[str]]) -> dict[str, int]:
    vocab: dict[str, int] = {}
    for pool in pools_list:
        for t in pool:
            if t not in vocab:
                vocab[t] = len(vocab)
    return vocab
