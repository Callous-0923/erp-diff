"""
ERPDiff — Gated Differential Cross-Attention Module (inter-branch module).

Part of the ERPDiff framework for P300 detection.
"""

import math

import torch
import torch.nn as nn


class _DiffCrossAttnDirection(nn.Module):
    """Single-direction differential cross-attention with residual connection."""

    def __init__(
        self,
        d_model: int = 32,
        num_heads: int = 4,
        attn_dropout: float = 0.2,
        proj_dropout: float = 0.2,
        linear_bias: bool = False,
        lambda_init: float = 0.8,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, 2 * d_model, bias=linear_bias)
        self.k_proj = nn.Linear(d_model, 2 * d_model, bias=linear_bias)
        self.v_proj = nn.Linear(d_model, 2 * d_model, bias=linear_bias)
        self.o_proj = nn.Linear(2 * d_model, d_model, bias=linear_bias)

        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim))
        self.register_buffer("lambda_init", torch.tensor(float(lambda_init)))

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)
        self.head_norm = nn.LayerNorm(2 * self.head_dim)
        self.last_lambda = None
        self.last_attn1 = None
        self.last_attn2 = None

    def forward(self, query_tokens: torch.Tensor, keyvalue_tokens: torch.Tensor, disable_second_term: bool):
        q_ln = self.ln_q(query_tokens)
        kv_ln = self.ln_kv(keyvalue_tokens)

        bsz, t_steps, _ = q_ln.shape
        q = self.q_proj(q_ln).view(bsz, t_steps, self.num_heads, 2 * self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(kv_ln).view(bsz, t_steps, self.num_heads, 2 * self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(kv_ln).view(bsz, t_steps, self.num_heads, 2 * self.head_dim).permute(0, 2, 1, 3)

        q1, q2 = q.split(self.head_dim, dim=-1)
        k1, k2 = k.split(self.head_dim, dim=-1)

        scores1 = torch.matmul(q1, k1.transpose(-1, -2)) * self.scale
        scores2 = torch.matmul(q2, k2.transpose(-1, -2)) * self.scale
        attn1 = self.attn_dropout(torch.softmax(scores1, dim=-1))
        attn2 = self.attn_dropout(torch.softmax(scores2, dim=-1))

        if not self.training:
            self.last_attn1 = attn1.detach()
            self.last_attn2 = attn2.detach()

        lambda_init = self.lambda_init.to(dtype=query_tokens.dtype, device=query_tokens.device)
        if disable_second_term:
            lambda_scalar = torch.zeros((), dtype=query_tokens.dtype, device=query_tokens.device)
        else:
            dot1 = torch.dot(self.lambda_q1, self.lambda_k1)
            dot2 = torch.dot(self.lambda_q2, self.lambda_k2)
            lambda_scalar = torch.exp(dot1) - torch.exp(dot2) + lambda_init
            lambda_scalar = lambda_scalar.to(dtype=query_tokens.dtype, device=query_tokens.device)
        self.last_lambda = lambda_scalar.detach()

        out = torch.matmul(attn1 - lambda_scalar * attn2, v)
        out = out.reshape(bsz * self.num_heads, t_steps, 2 * self.head_dim)
        out = self.head_norm(out).reshape(bsz, self.num_heads, t_steps, 2 * self.head_dim)
        out = out * (1.0 - lambda_init)

        out = out.permute(0, 2, 1, 3).contiguous().view(bsz, t_steps, 2 * self.d_model)
        out = self.o_proj(out)
        out = self.proj_dropout(out)
        return query_tokens + out


class GatedDiffCrossAttention(nn.Module):
    """
    Training-Phase-Aware Gated Differential Cross-Attention Module.

    Part of the ERPDiff framework.  Performs bidirectional differential
    cross-attention between CLB and RBB branches, with alpha-derived gating
    that follows the cumulative learning schedule.

    Args:
        d_model:          Feature dimension.
        num_heads:        Number of attention heads.
        attn_dropout:     Dropout on attention weights.
        proj_dropout:     Dropout on output projection.
        linear_bias:      Whether linear layers use bias.
        lambda_init:      Initial λ for differential attention.
        use_alpha_gate:   Enable/disable alpha gating (for ablation).
        min_gate:         Minimum gate value to prevent gradient starvation.
    """

    def __init__(
        self,
        d_model: int = 32,
        num_heads: int = 4,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.2,
        linear_bias: bool = False,
        lambda_init: float = 0.8,
        use_alpha_gate: bool = True,
        min_gate: float = 0.1,
    ):
        super().__init__()
        self.use_alpha_gate = use_alpha_gate
        self.min_gate = min_gate

        self.clb_from_rbb = _DiffCrossAttnDirection(
            d_model=d_model, num_heads=num_heads, attn_dropout=attn_dropout,
            proj_dropout=proj_dropout, linear_bias=linear_bias, lambda_init=lambda_init,
        )
        self.rbb_from_clb = _DiffCrossAttnDirection(
            d_model=d_model, num_heads=num_heads, attn_dropout=attn_dropout,
            proj_dropout=proj_dropout, linear_bias=linear_bias, lambda_init=lambda_init,
        )

    def forward(
        self,
        x_clb: torch.Tensor,
        x_rbb: torch.Tensor,
        alpha: float = 0.5,
        disable_second_term: bool = False,
    ):
        clb_cross = self.clb_from_rbb(x_clb, x_rbb, disable_second_term)
        rbb_cross = self.rbb_from_clb(x_rbb, x_clb, disable_second_term)

        if not self.use_alpha_gate:
            return clb_cross, rbb_cross

        clb_delta = clb_cross - x_clb
        rbb_delta = rbb_cross - x_rbb

        gate_clb = max(1.0 - alpha, self.min_gate)
        gate_rbb = max(alpha, self.min_gate)

        x_clb_new = x_clb + gate_clb * clb_delta
        x_rbb_new = x_rbb + gate_rbb * rbb_delta

        return x_clb_new, x_rbb_new
