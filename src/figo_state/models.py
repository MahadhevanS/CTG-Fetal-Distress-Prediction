"""
The three detection models, in one file on purpose.

THE HYPOTHESIS UNDER TEST
-------------------------
Detection sits at AUROC 0.7518 against a label whose ceiling is provably 1.0.
Suspicious is 65.8% driven by REPETITIVE DECELERATIONS -- defined as
decelerations accompanying more than 50% of contractions -- so the label
depends on a relation between two channels, not on either channel alone.

Measured (docs/figo_state_gate1_gate2.md section 5.4): giving a plain CNN the
UC channel changes nothing (0.7383 with it, 0.7518 without). The hypothesis
is that this is not because UC lacks information but because a network that
concatenates channels at the input and convolves them jointly has no
mechanism to represent "this dip corresponds to that contraction".

So the question is deliberately NOT "does UC contain information" but:

    does the FHR response CONDITIONED ON uterine contraction contain
    information that neither channel carries alone?

WHY ONE FILE
------------
A and B are the controls for C. If they differed from C in encoder width,
pooling, normalisation or initialisation, a difference in score would be
uninterpretable -- and this repo has already been burned by exactly that
(src/training/protocol.py: "Do not mix recipes", after an architecture sweep
and a ladder reported the same model at 0.7178 and 0.6834). Here the FHR
encoder, the temporal encoder, the pooling and the head are literally the
same classes. The ONLY difference between B and C is whether the two streams
meet through concatenation or through cross-attention.

SIZE
----
~200-500k parameters, per the pre-registered budget. 3,497 epochs from 547
patients does not support more, and this project has already measured what
happens when it tries: Model 9 (KG-MIL) put 2.5M parameters on ~435 bags and
collapsed (docs/model8_crossformer_run_history.md, Part E).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# shared parts -- identical objects in every arm
# --------------------------------------------------------------------------
class SignalEncoder(nn.Module):
    """
    One channel of 4 Hz signal -> a sequence of tokens.

    Total stride 32 takes 2400 samples (10 min at 4 Hz) to 75 tokens of 8 s
    each. 8 s is well below the 15 s minimum duration of any FIGO event, so
    nothing the label depends on is aliased away, and it is fine enough to
    place a deceleration against a contraction while keeping cross-attention
    at 75x75 rather than 2400x2400.
    """

    N_TOKENS = 75

    def __init__(self, in_ch: int = 1, d_model: int = 96, width: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, width, 15, stride=4, padding=7),      # 2400 -> 600
            nn.BatchNorm1d(width), nn.GELU(),
            nn.Conv1d(width, width, 9, stride=4, padding=4),       # 600 -> 150
            nn.BatchNorm1d(width), nn.GELU(),
            nn.Conv1d(width, d_model, 7, stride=2, padding=3),     # 150 -> 75
            nn.BatchNorm1d(d_model), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).transpose(1, 2)          # (B, T, d)


def downsample_mask(mask: torch.Tensor, out_len: int) -> torch.Tensor:
    """
    Sample-rate validity -> token-rate validity.

    A token is valid if ANY of the samples behind it were usable. `max` and
    not `mean` because a token that is half real still carries signal, and
    marking it invalid would throw away more than the dropout did.
    """
    m = mask.float().unsqueeze(1)
    return (F.adaptive_max_pool1d(m, out_len).squeeze(1) > 0.5)


class PositionalEncoding(nn.Module):
    """
    Learned position per token.

    Required for the interaction hypothesis and not optional: whether a
    deceleration begins before, during or after a contraction is what
    separates early from late from variable in the FIGO typing. Without a
    position signal, attention over an unordered set of tokens cannot
    represent that distinction at all.
    """

    def __init__(self, n_tokens: int, d_model: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class AttentionPool(nn.Module):
    """Masked attention pooling: padded/invalid tokens get exactly zero weight."""

    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.Tanh(),
                                   nn.Linear(d_model // 2, 1))

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        s = self.score(x).squeeze(-1)                       # (B, T)
        if mask is not None:
            s = s.masked_fill(~mask, float("-inf"))
            # A row with no valid token would softmax to NaN; fall back to
            # uniform attention over that row rather than propagate it.
            dead = (~mask).all(dim=1)
            if dead.any():
                s = torch.where(dead.unsqueeze(1), torch.zeros_like(s), s)
        w = torch.softmax(s, dim=1).unsqueeze(-1)
        return (x * w).sum(dim=1)


class Head(nn.Module):
    def __init__(self, d_model: int, n_out: int = 1, p_drop: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(p_drop),
                                 nn.Linear(d_model, d_model // 2), nn.GELU(),
                                 nn.Dropout(p_drop),
                                 nn.Linear(d_model // 2, n_out))

    def forward(self, x):
        return self.net(x)


def _temporal_encoder(d_model: int, n_layers: int = 2, n_heads: int = 4):
    layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
        dropout=0.1, batch_first=True, norm_first=True, activation="gelu")
    return nn.TransformerEncoder(layer, n_layers)


# --------------------------------------------------------------------------
# Model A / B -- the controls
# --------------------------------------------------------------------------
class ConcatModel(nn.Module):
    """
    Model A (FHR only) and Model B (FHR + UC concatenated at the input).

    B is the NEGATIVE CONTROL for the whole hypothesis. It receives exactly
    the same information as C -- both channels and both masks -- and differs
    only in that the two streams are stacked and convolved jointly rather
    than allowed to attend to one another. If B matches C, the interaction
    mechanism is not what is doing the work.
    """

    def __init__(self, use_uc: bool, d_model: int = 96, n_tokens: int = 75,
                 n_layers: int = 2):
        super().__init__()
        self.use_uc = use_uc
        in_ch = 4 if use_uc else 2          # (fhr, fhr_valid) [+ (uc, uc_valid)]
        self.enc = SignalEncoder(in_ch, d_model)
        self.pos = PositionalEncoding(n_tokens, d_model)
        self.temporal = _temporal_encoder(d_model, n_layers)
        self.pool = AttentionPool(d_model)
        self.head = Head(d_model)

    def forward(self, fhr, uc, fhr_valid, uc_valid):
        ch = [fhr, fhr_valid.float()]
        if self.use_uc:
            ch += [uc, uc_valid.float()]
        h = self.enc(torch.stack(ch, dim=1))
        m = downsample_mask(fhr_valid, h.size(1))
        h = self.temporal(self.pos(h), src_key_padding_mask=~m)
        return self.head(self.pool(h, m))


# --------------------------------------------------------------------------
# Model C -- the experiment
# --------------------------------------------------------------------------
class CrossAttentionBlock(nn.Module):
    """
    One direction of cross-attention: Q from one stream, K and V from the other.

    For every FHR token, attend over the UC tokens and ask which contractions
    are relevant to it. That is the computation "is this deceleration
    accompanying a contraction, and with what lag", which is exactly what the
    repetitive-deceleration criterion needs and what a joint convolution
    cannot express.
    """

    def __init__(self, d_model: int, n_heads: int = 4, p_drop: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=p_drop,
                                          batch_first=True)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(),
                                nn.Dropout(p_drop), nn.Linear(d_model * 2, d_model))
        self.drop = nn.Dropout(p_drop)

    def forward(self, q, kv, kv_mask=None, need_weights=False):
        a, w = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv),
                         key_padding_mask=(~kv_mask) if kv_mask is not None else None,
                         need_weights=need_weights, average_attn_weights=True)
        # A query row whose keys are ALL masked returns NaN from
        # MultiheadAttention. Zero the update for those rows instead, so a
        # patient with no usable toco degrades to FHR-only rather than
        # poisoning the batch.
        if kv_mask is not None:
            dead = (~kv_mask).all(dim=1)
            if dead.any():
                a = torch.where(dead.view(-1, 1, 1), torch.zeros_like(a), a)
        a = torch.nan_to_num(a)
        q = q + self.drop(a)
        q = q + self.drop(self.ff(self.norm_ff(q)))
        return (q, w) if need_weights else (q, None)


class CrossAttentionModel(nn.Module):
    """
    Model C -- separate encoders, bidirectional cross-attention, then time.

        FHR --> FHR encoder --+--> cross-attn (Q=FHR, K=V=UC) --+
                              |                                  +--> temporal
        UC  --> UC  encoder --+--> cross-attn (Q=UC, K=V=FHR) --+     -> pool
                                                                       -> class

    Both directions are computed and concatenated. FHR-attending-to-UC asks
    "which contraction explains this dip"; UC-attending-to-FHR asks "did this
    contraction produce a response", which is the direction the >50%-of-
    contractions ratio is actually phrased in.

    Attention masks are the USABLE-signal masks, not the raw missingness
    channel, so a dropout region can neither be attended to nor attend.
    """

    def __init__(self, d_model: int = 96, n_tokens: int = 75,
                 n_cross: int = 1, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.fhr_enc = SignalEncoder(2, d_model)
        self.uc_enc = SignalEncoder(2, d_model)
        self.pos_f = PositionalEncoding(n_tokens, d_model)
        self.pos_u = PositionalEncoding(n_tokens, d_model)
        self.f2u = nn.ModuleList([CrossAttentionBlock(d_model, n_heads)
                                  for _ in range(n_cross)])
        self.u2f = nn.ModuleList([CrossAttentionBlock(d_model, n_heads)
                                  for _ in range(n_cross)])
        self.fuse = nn.Linear(d_model * 2, d_model)
        self.temporal = _temporal_encoder(d_model, n_layers, n_heads)
        self.pool = AttentionPool(d_model)
        self.head = Head(d_model)
        self.last_attn = None                # populated when return_attn=True

    def forward(self, fhr, uc, fhr_valid, uc_valid, return_attn: bool = False):
        hf = self.pos_f(self.fhr_enc(torch.stack([fhr, fhr_valid.float()], 1)))
        hu = self.pos_u(self.uc_enc(torch.stack([uc, uc_valid.float()], 1)))
        mf = downsample_mask(fhr_valid, hf.size(1))
        mu = downsample_mask(uc_valid, hu.size(1))

        f, u = hf, hu
        attn = None
        for i, (a, b) in enumerate(zip(self.f2u, self.u2f)):
            last = return_attn and i == len(self.f2u) - 1
            f_new, w = a(f, u, mu, need_weights=last)
            u_new, _ = b(u, f, mf)
            f, u = f_new, u_new
            if last:
                attn = w
        if return_attn:
            # Rows whose keys were all masked return NaN weights even though
            # their contribution to the output was zeroed above. These are
            # only ever read for explainability, so make them explicit zeros.
            self.last_attn = None if attn is None else torch.nan_to_num(attn)

        h = self.fuse(torch.cat([f, u], dim=-1))
        h = self.temporal(h, src_key_padding_mask=~mf)
        return self.head(self.pool(h, mf))


# --------------------------------------------------------------------------
def build(arch: str, **kw) -> nn.Module:
    if arch == "fhr":
        return ConcatModel(use_uc=False, **kw)
    if arch == "concat":
        return ConcatModel(use_uc=True, **kw)
    if arch == "cross":
        return CrossAttentionModel(**kw)
    raise ValueError(f"unknown arch {arch!r}")


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
