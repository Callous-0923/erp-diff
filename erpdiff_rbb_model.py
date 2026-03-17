"""
ERPDiff — RBB branch model (re-balancing branch with dilated Inception + TemporalDiffAttn).
"""

import torch
import torch.nn as nn

import icnn
from icnn import ICNNStem, ICNNTail
from temporal_diff_attn import TemporalDiffAttn


def _same_pad_1xk_dilated(k: int, dilation: int):
    k_eff = dilation * (k - 1) + 1
    if k_eff % 2 == 1:
        left = right = k_eff // 2
    else:
        left = k_eff // 2 - 1
        right = k_eff // 2
    return left, right


class Conv2dSame1xKDilated(nn.Module):
    """Conv2d with manual 'same' padding along time for kernel (1,k) and dilation."""

    def __init__(self, in_ch: int, out_ch: int, k: int, dilation: int, groups: int = 1, bias: bool = False):
        super().__init__()
        pad_l, pad_r = _same_pad_1xk_dilated(k, dilation)
        self.pad = nn.ZeroPad2d((pad_l, pad_r, 0, 0))
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=(1, k), stride=(1, 1), padding=0,
            dilation=(1, dilation), groups=groups, bias=bias,
        )

    def forward(self, x):
        return self.conv(self.pad(x))


class InceptionBlockDilatedC(nn.Module):
    """Inception block with dilated convolution in the C2 branch."""

    def __init__(self, in_ch: int, k_b: int, k_c: int, dil_c: int, pool_k: int):
        super().__init__()
        self.a1 = icnn.ConvBN(nn.Conv2d(in_ch, 8, kernel_size=(1, 1), bias=False), 8, activation=True)
        self.b1 = icnn.ConvBN(nn.Conv2d(in_ch, 4, kernel_size=(1, 1), bias=False), 4, activation=False)
        self.b2 = icnn.ConvBN(icnn.Conv2dSame1xK(4, 8, k=k_b, bias=False), 8, activation=True)
        self.c1 = icnn.ConvBN(nn.Conv2d(in_ch, 4, kernel_size=(1, 1), bias=False), 4, activation=False)
        self.c2 = icnn.ConvBN(
            Conv2dSame1xKDilated(4, 8, k=k_c, dilation=dil_c, bias=False), 8, activation=True
        )
        self.d1 = nn.AvgPool2d(kernel_size=(1, pool_k), stride=(1, 1), padding=(0, pool_k // 2))
        self.d2 = icnn.ConvBN(nn.Conv2d(in_ch, 8, kernel_size=(1, 1), bias=False), 8, activation=False)

    def forward(self, x):
        a = self.a1(x)
        b = self.b2(self.b1(x))
        c = self.c2(self.c1(x))
        d = self.d2(self.d1(x))
        return torch.cat([a, b, c, d], dim=1)


class ICNNStemRBB_Dilated(nn.Module):
    """RBB-specific stem with dilated Inception block."""

    def __init__(self, in_channels: int, dropout_p: float = 0.2):
        super().__init__()
        self.temporal = icnn.ConvBN(icnn.Conv2dSame1xK(1, 8, k=64, bias=False), 8, activation=False)
        self.spatial = icnn.ConvBN(
            nn.Conv2d(8, 32, kernel_size=(in_channels, 1), stride=(1, 1), padding=(0, 0), groups=8, bias=False),
            32, activation=False,
        )
        self.incep1 = InceptionBlockDilatedC(in_ch=32, k_b=16, k_c=16, dil_c=4, pool_k=7)
        self.down1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=(0, 0)),
            nn.Dropout(p=dropout_p),
        )
        self.incep2 = icnn.InceptionBlock(in_ch=32, k_b=8, k_c=16, pool_k=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.incep1(x)
        x = self.down1(x)
        x = self.incep2(x)
        return x


class RBBPretrainICNN(nn.Module):
    """RBB pretraining model: ICNNStemRBB_Dilated → TemporalDiffAttn → ICNNTail."""

    def __init__(
        self,
        in_channels: int,
        n_samples: int,
        dropout_p: float = 0.2,
        n_classes: int = 2,
        use_temporal_bias: bool = True,
    ):
        super().__init__()
        self.stem_rbb = ICNNStemRBB_Dilated(in_channels, dropout_p=dropout_p)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_channels, n_samples)
            feat_t = self.stem_rbb(dummy).shape[-1]
        self.rbb_temporal_attn = TemporalDiffAttn(
            d_model=32, num_heads=4, attn_dropout=0.0, proj_dropout=dropout_p,
            bias=False, lambda_init=0.8, use_temporal_bias=use_temporal_bias,
        )
        self.tail_rbb = ICNNTail(feat_t=feat_t, dropout_p=dropout_p, n_classes=n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem_rbb(x)
        tok = feat.squeeze(2).permute(0, 2, 1).contiguous()
        tok = self.rbb_temporal_attn(tok)
        feat = tok.permute(0, 2, 1).contiguous().unsqueeze(2)
        return self.tail_rbb(feat)


__all__ = ["RBBPretrainICNN", "ICNNStemRBB_Dilated"]
