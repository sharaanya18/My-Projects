from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TagRankerHead(nn.Module):
    """Projects an arbitrary feature vector into a shared space with trainable
    per-tag embeddings, and scores a case's own 80-candidate pool only."""

    def __init__(self, in_dim: int, num_tags: int, emb_dim: int = 256, dropout: float = 0.2, input_dropout: float = 0.0):
        super().__init__()
        self.input_dropout = nn.Dropout(input_dropout)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )
        self.tag_emb = nn.Embedding(num_tags, emb_dim)
        nn.init.normal_(self.tag_emb.weight, std=0.02)
        self.log_temp = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(10.0)))))

    def title_repr(self, feat: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(self.input_dropout(feat)), dim=-1)

    def forward(self, feat: torch.Tensor, pool_idx: torch.Tensor) -> torch.Tensor:
        """feat: (B, in_dim); pool_idx: (B, 80) tag-vocab indices for this case's pool.
        Returns logits (B, 80)."""
        title_e = self.title_repr(feat)  # (B, D), unit norm
        pool_e = F.normalize(self.tag_emb(pool_idx), dim=-1)  # (B, 80, D)
        logits = torch.einsum("bd,bnd->bn", title_e, pool_e)
        temp = self.log_temp.exp().clamp(max=100.0)
        return logits * temp


def multi_positive_softmax_loss(logits: torch.Tensor, pos_mask: torch.Tensor) -> torch.Tensor:
    """logits: (B, 80); pos_mask: (B, 80) bool, True at the (up to 3) correct
    positions within this case's pool. Averages -log p(correct) over positives."""
    log_probs = F.log_softmax(logits, dim=-1)
    pos = pos_mask.float()
    per_case = -(log_probs * pos).sum(dim=-1) / pos.sum(dim=-1).clamp(min=1)
    return per_case.mean()
