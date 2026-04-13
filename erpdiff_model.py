"""
ERPDiff — Main model for the fine-tuning stage.

ERPDiff: Enhancing P300 Detection via ERP-Aware Differential Attention
and Gated Branch Fusion.
"""

from typing import Tuple

import torch
import torch.nn as nn

from dcm_diff_cross_attn import GatedDiffCrossAttention
from icnn import ICNN, ICNNStem, ICNNTail
from erpdiff_rbb_model import ICNNStemRBB_Dilated
from temporal_diff_attn import TemporalDiffAttn


def _to_tokens(x: torch.Tensor) -> torch.Tensor:
    return x.squeeze(2).permute(0, 2, 1).contiguous()


def _from_tokens(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 1).contiguous().unsqueeze(2)


def _strip_prefix(state: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


class ERPDiff(nn.Module):
    """
    ERPDiff: ERP-Aware Differential Attention with Gated Branch Fusion.

    Architecture:
        Input → ICNNStem (CLB) ─────────────────→ tokens_clb ─┐
        Input → ICNNStemRBB ─→ TemporalDiffAttn ─→ tokens_rbb ─┤
                                                                ├─ GatedDiffCrossAttn ─→ tails ─→ fused logits

    Forward returns 5 values:
        logits_clb, logits_rbb, fused_logits, tokens_clb, tokens_rbb

    Args:
        in_channels:        Number of EEG channels.
        n_samples:          Number of time samples per trial.
        dropout_p:          Dropout probability.
        n_classes:          Number of output classes (default 2: target/nontarget).
        attn_dropout:       Dropout on attention weights.
        proj_dropout:       Dropout on output projection.
        num_heads:          Number of attention heads.
        enable_dcm:         Enable gated differential cross-attention module.
        lambda_init:        Initial λ for differential attention.
        use_temporal_bias:  Enable dual-sigma Gaussian temporal bias (ablation switch).
        temporal_bias_sigma_mode:
                            Choose shared scalar sigma or per-head sigma vectors.
        use_alpha_gate:     Enable alpha-gated cross-attention (ablation switch).
        min_gate:           Minimum gate value to prevent gradient starvation.
    """

    def __init__(
        self,
        in_channels: int,
        n_samples: int,
        dropout_p: float = 0.2,
        n_classes: int = 2,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.2,
        num_heads: int = 4,
        enable_dcm: bool = True,
        lambda_init: float = 0.8,
        use_temporal_bias: bool = True,
        temporal_bias_sigma_mode: str = "shared",
        use_alpha_gate: bool = True,
        min_gate: float = 0.1,
    ):
        super().__init__()
        self.stem_clb = ICNNStem(in_channels, dropout_p=dropout_p)
        self.stem_rbb = ICNNStemRBB_Dilated(in_channels, dropout_p=dropout_p)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_channels, n_samples)
            feat_t = self.stem_clb(dummy).shape[-1]

        self.tail_clb = ICNNTail(feat_t=feat_t, dropout_p=dropout_p, n_classes=n_classes)
        self.tail_rbb = ICNNTail(feat_t=feat_t, dropout_p=dropout_p, n_classes=n_classes)

        self.rbb_temporal_attn = TemporalDiffAttn(
            d_model=32,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            bias=False,
            lambda_init=lambda_init,
            use_temporal_bias=use_temporal_bias,
            temporal_bias_sigma_mode=temporal_bias_sigma_mode,
        )

        self.dcm = GatedDiffCrossAttention(
            d_model=32,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            linear_bias=False,
            lambda_init=lambda_init,
            use_alpha_gate=use_alpha_gate,
            min_gate=min_gate,
        )
        self.enable_dcm = enable_dcm

    def load_branch_state(self, clb_state: dict, rbb_state: dict) -> None:
        """Load pretrained weights into CLB and RBB branches."""
        self.stem_clb.load_state_dict(clb_state, strict=False)
        self.tail_clb.load_state_dict(clb_state, strict=False)
        stem_rbb_state = _strip_prefix(rbb_state, "stem_rbb.") or _strip_prefix(rbb_state, "stem.") or rbb_state
        attn_state = (
            _strip_prefix(rbb_state, "rbb_temporal_attn.")
            or _strip_prefix(rbb_state, "temporal_attn.")
            or rbb_state
        )
        tail_rbb_state = _strip_prefix(rbb_state, "tail_rbb.") or _strip_prefix(rbb_state, "tail.") or rbb_state
        self.stem_rbb.load_state_dict(stem_rbb_state, strict=False)
        self.rbb_temporal_attn.load_state_dict(attn_state, strict=False)
        self.tail_rbb.load_state_dict(tail_rbb_state, strict=False)

    def forward(
        self, x: torch.Tensor, alpha: float, disable_second_term: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits_clb:    [B, n_classes]
            logits_rbb:    [B, n_classes]
            fused_logits:  [B, n_classes]
            tokens_clb:    [B, T', d_model]  (for complementarity loss)
            tokens_rbb:    [B, T', d_model]  (for complementarity loss)
        """
        feat_clb = self.stem_clb(x)
        feat_rbb = self.stem_rbb(x)

        tokens_clb = _to_tokens(feat_clb)
        tokens_rbb = _to_tokens(feat_rbb)

        tokens_rbb = self.rbb_temporal_attn(tokens_rbb)

        if self.enable_dcm:
            tokens_clb, tokens_rbb = self.dcm(
                tokens_clb, tokens_rbb,
                alpha=alpha,
                disable_second_term=disable_second_term,
            )

        feat_clb = _from_tokens(tokens_clb)
        feat_rbb = _from_tokens(tokens_rbb)

        logits_clb = self.tail_clb(feat_clb)
        logits_rbb = self.tail_rbb(feat_rbb)

        fused_logits = alpha * logits_clb + (1.0 - alpha) * logits_rbb

        return logits_clb, logits_rbb, fused_logits, tokens_clb, tokens_rbb


__all__ = ["ERPDiff"]
