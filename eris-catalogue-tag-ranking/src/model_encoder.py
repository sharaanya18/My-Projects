"""Fine-tuned multilingual transformer branch.

Encodes the raw (un-normalized) title with an ungated public HF backbone,
mean-pools, and feeds the shared TagRankerHead. The whole encoder is
trainable (real fine-tuning, not frozen embeddings + a tabular head).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Ordered by preference; every one is ungated and needs no HF token.
CANDIDATE_BACKBONES = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "microsoft/Multilingual-MiniLM-L12-H384",
    "distilbert-base-multilingual-cased",
]


def load_backbone(name: str | None = None):
    names = [name] if name else CANDIDATE_BACKBONES
    last_err = None
    for n in names:
        try:
            tok = AutoTokenizer.from_pretrained(n)
            enc = AutoModel.from_pretrained(n)
            print(f"loaded backbone: {n}")
            return n, tok, enc
        except Exception as e:  # noqa: BLE001 - download/availability fallback
            print(f"backbone {n} failed to load ({e}); trying next")
            last_err = e
    raise RuntimeError(f"no backbone could be loaded: {last_err}")


class EncoderTitleEmbedder(nn.Module):
    def __init__(self, encoder: nn.Module, out_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, out_dim))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return self.proj(pooled)


def tokenize_titles(tokenizer, titles: list[str], max_length: int = 48):
    return tokenizer(
        titles, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
