"""
ERPDiff — ERP-Aware Temporal Differential Attention (intra-branch module).

Part of the ERPDiff framework for P300 detection.
"""

import math

import torch
import torch.nn as nn


TEMPORAL_BIAS_SIGMA_MODES = ("shared", "per_head")


class TemporalDiffAttn(nn.Module):
    """
    ERP-Aware Temporal Differential Attention.

    Core module of the ERPDiff framework's RBB branch.  Applies differential
    attention (attn1 − λ·attn2) over temporal token sequences, with dual-sigma
    Gaussian bias to separate ERP signal from background noise.

    Design rationale:
      - sigma_signal (narrow, init=3.0): biases the signal head toward local
        ERP components (P300 typically spans a few adjacent tokens).
      - sigma_noise  (wide,  init=10.0): lets the noise head capture broad
        background patterns across the entire epoch.
      - The differential mechanism then cancels the shared noise component.

    Args:
        d_model:             Feature dimension (default 32).
        num_heads:           Number of attention heads.
        attn_dropout:        Dropout on attention weights.
        proj_dropout:        Dropout on output projection.
        bias:                Whether linear layers use bias.
        lambda_init:         Initial value for the differential λ scalar.
        use_temporal_bias:   Enable/disable Gaussian temporal bias (for ablation).
    """

    def __init__(
        self,
        d_model: int = 32,
        num_heads: int = 4,
        attn_dropout: float = 0.2,
        proj_dropout: float = 0.2,
        bias: bool = False,
        lambda_init: float = 0.8,
        use_temporal_bias: bool = True,
        temporal_bias_sigma_mode: str = "shared",
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if temporal_bias_sigma_mode not in TEMPORAL_BIAS_SIGMA_MODES:
            raise ValueError(
                f"Unsupported temporal_bias_sigma_mode={temporal_bias_sigma_mode!r}. "
                f"Expected one of {TEMPORAL_BIAS_SIGMA_MODES}."
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.ln = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, 2 * d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, 2 * d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, 2 * d_model, bias=bias)
        self.o_proj = nn.Linear(2 * d_model, d_model, bias=bias)

        # --- Differential λ parameters ---
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim))
        self.register_buffer("lambda_init", torch.tensor(float(lambda_init)))

        # --- ERP-aware dual-sigma Gaussian temporal bias ---
        self.use_temporal_bias = use_temporal_bias
        self.temporal_bias_sigma_mode = temporal_bias_sigma_mode
        if use_temporal_bias:
            if temporal_bias_sigma_mode == "shared":
                self.sigma_signal = nn.Parameter(torch.tensor(3.0))
                self.sigma_noise = nn.Parameter(torch.tensor(10.0))
            else:
                self.sigma_signal = nn.Parameter(torch.full((num_heads,), 3.0))
                self.sigma_noise = nn.Parameter(torch.full((num_heads,), 10.0))

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)
        self.head_norm = nn.LayerNorm(2 * self.head_dim)

        # --- Interpretability cache (populated during eval) ---
        self.last_lambda = None
        self.last_attn1 = None
        self.last_attn2 = None
        self.last_sigma_signal = None
        self.last_sigma_noise = None

    @staticmethod
    def _reshape_sigma_for_checkpoint(loaded_value: torch.Tensor, target_value: torch.Tensor) -> torch.Tensor:
        loaded_flat = loaded_value.reshape(-1)
        target_shape = target_value.shape

        if loaded_value.shape == target_shape:
            return loaded_value
        if target_value.ndim == 0:
            return loaded_flat.mean().to(dtype=target_value.dtype).reshape(())
        if loaded_flat.numel() == 1:
            return loaded_flat.repeat(target_value.numel()).to(dtype=target_value.dtype).view(target_shape)
        if loaded_flat.numel() == target_value.numel():
            return loaded_flat.to(dtype=target_value.dtype).view(target_shape)
        return loaded_value

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if self.use_temporal_bias:
            for sigma_name in ("sigma_signal", "sigma_noise"):
                key = prefix + sigma_name
                if key in state_dict and hasattr(self, sigma_name):
                    state_dict[key] = self._reshape_sigma_for_checkpoint(
                        state_dict[key], getattr(self, sigma_name)
                    )
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def _compute_temporal_bias(self, t_steps: int, dtype: torch.dtype, device: torch.device):
        pos = torch.arange(t_steps, dtype=dtype, device=device)
        rel_dist = pos.unsqueeze(0) - pos.unsqueeze(1)

        sigma_s = self.sigma_signal.clamp(min=1.0)
        sigma_n = self.sigma_noise.clamp(min=1.0)

        if self.temporal_bias_sigma_mode == "shared":
            bias_signal = -0.5 * (rel_dist / sigma_s) ** 2
            bias_noise  = -0.5 * (rel_dist / sigma_n) ** 2
            return bias_signal.unsqueeze(0).unsqueeze(0), bias_noise.unsqueeze(0).unsqueeze(0)

        rel_dist = rel_dist.unsqueeze(0)
        bias_signal = -0.5 * (rel_dist / sigma_s.view(-1, 1, 1)) ** 2
        bias_noise  = -0.5 * (rel_dist / sigma_n.view(-1, 1, 1)) ** 2
        return bias_signal.unsqueeze(0), bias_noise.unsqueeze(0)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens_ln = self.ln(tokens)
        bsz, t_steps, _ = tokens_ln.shape

        q = self.q_proj(tokens_ln).view(bsz, t_steps, self.num_heads, 2 * self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(tokens_ln).view(bsz, t_steps, self.num_heads, 2 * self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(tokens_ln).view(bsz, t_steps, self.num_heads, 2 * self.head_dim).permute(0, 2, 1, 3)

        q1, q2 = q.split(self.head_dim, dim=-1)
        k1, k2 = k.split(self.head_dim, dim=-1)

        scores1 = torch.matmul(q1, k1.transpose(-1, -2)) * self.scale
        scores2 = torch.matmul(q2, k2.transpose(-1, -2)) * self.scale

        if self.use_temporal_bias:
            bias_signal, bias_noise = self._compute_temporal_bias(
                t_steps, tokens.dtype, tokens.device
            )
            scores1 = scores1 + bias_signal
            scores2 = scores2 + bias_noise

        attn1 = self.attn_dropout(torch.softmax(scores1, dim=-1))
        attn2 = self.attn_dropout(torch.softmax(scores2, dim=-1))

        if not self.training:
            self.last_attn1 = attn1.detach()
            self.last_attn2 = attn2.detach()
            if self.use_temporal_bias:
                self.last_sigma_signal = self.sigma_signal.detach().clone()
                self.last_sigma_noise = self.sigma_noise.detach().clone()

        lambda_init = self.lambda_init.to(dtype=tokens.dtype, device=tokens.device)
        dot1 = torch.dot(self.lambda_q1, self.lambda_k1)
        dot2 = torch.dot(self.lambda_q2, self.lambda_k2)
        lambda_scalar = torch.exp(dot1) - torch.exp(dot2) + lambda_init
        lambda_scalar = lambda_scalar.to(dtype=tokens.dtype, device=tokens.device)
        self.last_lambda = lambda_scalar.detach()

        out = torch.matmul(attn1 - lambda_scalar * attn2, v)
        out = out.reshape(bsz * self.num_heads, t_steps, 2 * self.head_dim)
        out = self.head_norm(out).reshape(bsz, self.num_heads, t_steps, 2 * self.head_dim)
        out = out * (1.0 - lambda_init)

        out = out.permute(0, 2, 1, 3).contiguous().view(bsz, t_steps, 2 * self.d_model)
        out = self.o_proj(out)
        out = self.proj_dropout(out)
        return tokens + out


def _format_sigma_debug(sigma: torch.Tensor) -> str:
    sigma_flat = sigma.detach().reshape(-1).tolist()
    if len(sigma_flat) == 1:
        return f"{sigma_flat[0]:.2f}"
    return "[" + ", ".join(f"{value:.2f}" for value in sigma_flat) + "]"


def _sanity_check() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 40, 32)

    for sigma_mode in ("shared", "per_head"):
        attn = TemporalDiffAttn(use_temporal_bias=True, temporal_bias_sigma_mode=sigma_mode)
        y = attn(x)
        assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
        assert not torch.isnan(y).any(), "NaN detected"
        y.mean().backward()
        assert attn.last_lambda is not None and torch.isfinite(attn.last_lambda)
        print(
            f"[OK] with temporal bias ({sigma_mode}): "
            f"sigma_signal={_format_sigma_debug(attn.sigma_signal)}, "
            f"sigma_noise={_format_sigma_debug(attn.sigma_noise)}, "
            f"lambda={attn.last_lambda.item():.4f}"
        )

    attn_no_bias = TemporalDiffAttn(use_temporal_bias=False)
    y2 = attn_no_bias(x)
    assert y2.shape == x.shape
    y2.mean().backward()
    print("[OK] without temporal bias (ablation mode)")


if __name__ == "__main__":
    _sanity_check()
