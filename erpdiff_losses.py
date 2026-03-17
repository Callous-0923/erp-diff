"""
ERPDiff — Loss functions.

Contains FocalLoss (for RBB pretraining) and complementarity_loss
(for preventing branch feature collapse during fine-tuning).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logpt = F.log_softmax(logits, dim=1)
        pt = torch.exp(logpt)
        target_one_hot = F.one_hot(target, num_classes=logits.shape[1]).float()
        loss = -self.alpha * target_one_hot * torch.pow(1 - pt, self.gamma) * logpt
        loss = loss.sum(dim=1)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def complementarity_loss(
    tokens_clb: torch.Tensor,
    tokens_rbb: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """
    Margin-based soft decorrelation loss.

    Prevents the two branches from collapsing into identical representations
    after cross-attention, while allowing them to share basic EEG features.

    Only penalizes cosine similarity exceeding the margin.

    Args:
        tokens_clb:  [B, T, d_model]  CLB tokens after gated cross-attention.
        tokens_rbb:  [B, T, d_model]  RBB tokens after gated cross-attention.
        margin:      Similarity threshold (default 0.5).

    Returns:
        Scalar loss (0 when similarity ≤ margin).
    """
    clb_repr = tokens_clb.mean(dim=1)
    rbb_repr = tokens_rbb.mean(dim=1)
    cos_sim = F.cosine_similarity(clb_repr, rbb_repr, dim=-1)
    loss = F.relu(cos_sim - margin)
    return loss.mean()
