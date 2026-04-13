"""
ERPDiff — Loss functions.

Contains FocalLoss (for RBB pretraining) and complementarity_loss
(for preventing branch feature collapse during fine-tuning).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss with per-class alpha weighting (Lin et al., 2017).

    For binary classification with class imbalance:
      - target class (y=1, minority): weighted by alpha
      - non-target class (y=0, majority): weighted by (1 - alpha)

    When alpha > 0.5, the minority (positive) class receives higher weight.
    The (1-pt)^gamma modulating factor down-weights easy examples regardless
    of class.

    Args:
        alpha:     Weighting factor for the positive class. Default 0.75 gives
                   3x weight to minority class vs majority class.
        gamma:     Focusing parameter. Higher values increase focus on hard examples.
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        # Pre-register alpha weights as buffer: moves with model.to(device) automatically.
        # alpha_weights[0] = 1-alpha (majority/negative), alpha_weights[1] = alpha (minority/positive)
        self.register_buffer(
            "alpha_weights",
            torch.tensor([1.0 - alpha, alpha]),
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = logits.shape[1]
        logpt = F.log_softmax(logits, dim=1)           # [B, C]
        pt = torch.exp(logpt)                           # [B, C]
        target_one_hot = F.one_hot(target, num_classes=n_classes).float()  # [B, C]

        # Per-class alpha_t: alpha for positive (y=1), (1-alpha) for negative (y=0)
        # .to() is a no-op if alpha_weights is already on the correct device/dtype
        aw = self.alpha_weights.to(device=logits.device, dtype=logits.dtype)
        alpha_t = (target_one_hot * aw.unsqueeze(0)).sum(dim=1)  # [B]

        # pt for the correct class
        pt_correct = (pt * target_one_hot).sum(dim=1)                      # [B]
        logpt_correct = (logpt * target_one_hot).sum(dim=1)                # [B]

        # Focal loss: -alpha_t * (1 - pt)^gamma * log(pt)
        loss = -alpha_t * torch.pow(1.0 - pt_correct, self.gamma) * logpt_correct  # [B]

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
