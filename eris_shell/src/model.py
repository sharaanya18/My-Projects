"""Two-stage plan+realize sequence model with a pointer-generator copy channel.

Why this decomposition, given the split:
  the test set only contains head->head adjacencies that appear in *no* training
  command.  A single seq2seq over the whole pipeline has to emit an unseen
  bigram of heads inside one string it has never produced.  Splitting the job
  makes the hard part small:

    PLAN     input description        -> ordered command heads  (tiny output
                                         space, ~213 types)
    REALIZE  input description + head -> that stage's tokens    (never sees
                                         cross-stage composition at all)

  Realizing head `b` as a second stage is then supported by every training
  command in which `b` appeared anywhere, so a held-out pair (a, b) is
  assembled from two well-supported halves.  The remaining risk is that the
  plan decoder's own bigram prior blocks the unseen pair, which is what the
  prev-head dropout below is for: during training the previously emitted head
  is replaced by a learned <mask> with probability p, forcing the plan decoder
  to justify each head from the description rather than from its predecessor.
  p is selected on the compositional folds, not hand-picked.

All weights are randomly initialised and trained only on the provided
train.csv (Guidebook 5.5).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import BOS, EOS, PAD, UNK


class Encoder(nn.Module):
    """Shared description encoder; both heads of the model read from it."""

    def __init__(self, n_src, d, nhead, nlayers, ff, dropout, max_len):
        super().__init__()
        self.tok = nn.Embedding(n_src, d, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(
            d, nhead, ff, dropout=dropout, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.d = d

    def forward(self, src, src_pad):
        p = torch.arange(src.size(1), device=src.device).unsqueeze(0)
        x = self.drop(self.tok(src) * math.sqrt(self.d) + self.pos(p))
        return self.norm(self.enc(x, src_key_padding_mask=src_pad))


def causal_mask(n, device):
    return torch.triu(torch.full((n, n), float("-inf"), device=device), diagonal=1)


class PlanDecoder(nn.Module):
    """Autoregressive decoder over command heads only."""

    def __init__(self, n_head_vocab, d, nhead, nlayers, ff, dropout, max_stages):
        super().__init__()
        self.emb = nn.Embedding(n_head_vocab, d, padding_idx=PAD)
        self.mask_emb = nn.Parameter(torch.randn(d) * 0.02)   # stands in for a dropped prev head
        self.pos = nn.Embedding(max_stages + 2, d)
        layer = nn.TransformerDecoderLayer(
            d, nhead, ff, dropout=dropout, batch_first=True, norm_first=True
        )
        self.dec = nn.TransformerDecoder(layer, nlayers)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, n_head_vocab)
        self.drop = nn.Dropout(dropout)
        self.d = d

    def forward(self, dec_in, memory, src_pad, prev_drop=0.0):
        B, T = dec_in.shape
        e = self.emb(dec_in) * math.sqrt(self.d)
        if prev_drop > 0.0 and self.training:
            # position 0 is <bos>; only genuine previous heads may be masked
            keep = (torch.rand(B, T, 1, device=dec_in.device) >= prev_drop).float()
            keep[:, 0, :] = 1.0
            e = keep * e + (1.0 - keep) * self.mask_emb.view(1, 1, -1)
        p = torch.arange(T, device=dec_in.device).unsqueeze(0)
        x = self.drop(e + self.pos(p))
        h = self.norm(self.dec(x, memory, tgt_mask=causal_mask(T, dec_in.device),
                               memory_key_padding_mask=src_pad))
        return self.out(h)


class RealizeDecoder(nn.Module):
    """Decodes one pipeline stage, with a pointer-generator copy channel."""

    def __init__(self, n_tgt, n_head_vocab, d, nhead, nlayers, ff, dropout,
                 max_tgt, n_poscls, head_emb):
        super().__init__()
        self.tok = nn.Embedding(n_tgt, d, padding_idx=PAD)
        self.pos = nn.Embedding(max_tgt + 2, d)
        self.head_emb = head_emb            # shared with the plan decoder
        self.poscls = nn.Embedding(n_poscls, d)
        layer = nn.TransformerDecoderLayer(
            d, nhead, ff, dropout=dropout, batch_first=True, norm_first=True
        )
        self.dec = nn.TransformerDecoder(layer, nlayers)
        self.norm = nn.LayerNorm(d)
        self.gen = nn.Linear(d, n_tgt)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.p_gen = nn.Linear(3 * d, 1)
        self.drop = nn.Dropout(dropout)
        self.d = d
        self.n_tgt = n_tgt

    def hidden(self, dec_in, memory, src_pad, head_ids, pos_ids):
        T = dec_in.size(1)
        p = torch.arange(T, device=dec_in.device).unsqueeze(0)
        cond = (self.head_emb(head_ids) + self.poscls(pos_ids)).unsqueeze(1)
        x = self.drop(self.tok(dec_in) * math.sqrt(self.d) + self.pos(p) + cond)
        return self.norm(self.dec(x, memory, tgt_mask=causal_mask(T, dec_in.device),
                                  memory_key_padding_mask=src_pad))

    def dist(self, h, memory, src_pad, src_ext, n_ext):
        """Mixture of the fixed-vocab softmax and the copy distribution.

        Returns log P over [target vocab | per-example source OOVs].
        """
        q = self.q(h)                                   # (M, T, d)
        k = self.k(memory)                              # (M, S, d)
        att = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.d)
        att = att.masked_fill(src_pad.unsqueeze(1), float("-inf"))
        a = torch.softmax(att, dim=-1)                  # (M, T, S)
        ctx = torch.bmm(a, memory)                      # (M, T, d)
        pg = torch.sigmoid(self.p_gen(torch.cat([h, ctx, h * ctx], dim=-1)))  # (M, T, 1)

        pvocab = torch.softmax(self.gen(h), dim=-1)     # (M, T, V)
        M, T = h.size(0), h.size(1)
        full = torch.zeros(M, T, self.n_tgt + n_ext, device=h.device, dtype=pvocab.dtype)
        full[:, :, : self.n_tgt] = pg * pvocab
        idx = src_ext.unsqueeze(1).expand(M, T, src_ext.size(1))
        full.scatter_add_(2, idx, (1.0 - pg) * a)
        return torch.log(full + 1e-10), a


class ShellSynth(nn.Module):
    """Encoder + plan decoder + realize decoder, trained jointly.

    A third, auxiliary head (`bag`) predicts the *unordered set* of command
    heads the description calls for, straight off the pooled encoder.  It
    carries no order or adjacency information at all, which is exactly why it
    helps here: it teaches the encoder to detect "this description needs a
    `sed` and an `xargs`" independently of any head-to-head transition the
    plan decoder may never have seen.
    """

    def __init__(self, n_src, n_tgt, n_head, cfg):
        super().__init__()
        d = cfg["d_model"]
        self.encoder = Encoder(n_src, d, cfg["nhead"], cfg["enc_layers"],
                               cfg["ff"], cfg["dropout"], cfg["max_src"])
        self.plan = PlanDecoder(n_head, d, cfg["nhead"], cfg["plan_layers"],
                                cfg["ff"], cfg["dropout"], cfg["max_stages"])
        self.realize = RealizeDecoder(n_tgt, n_head, d, cfg["nhead"],
                                      cfg["dec_layers"], cfg["ff"], cfg["dropout"],
                                      cfg["max_tgt"], cfg["n_poscls"], self.plan.emb)
        self.bag = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_head))
        self.cfg = cfg
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
