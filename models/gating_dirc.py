"""Dirichlet-evidence uncertainty gate for DiagMIL."""

import math
import torch
import torch.nn as nn

__all__ = ["DircUncertaintyGate"]


class DircUncertaintyGate(nn.Module):
    """
    Uncertainty-aware gate from Stage1 sigmoid scores (Dirichlet view).

    Input ``p``: [B, C], values in (0, 1).

    Returns ``g``, ``H_bar``, ``m``, ``u``, ``mass`` (see forward).
    """

    def __init__(
        self,
        w_H: float = 3.0,
        w_u: float = 3.0,
        bias: float = 0.0,
    ):
        super().__init__()
        self.w_H = nn.Parameter(torch.tensor(float(w_H), dtype=torch.float32))
        self.w_u = nn.Parameter(torch.tensor(float(w_u), dtype=torch.float32))
        self.w_mass = nn.Parameter(torch.tensor(2.0, dtype=torch.float32))
        self.w_m = nn.Parameter(torch.tensor(2.0, dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(float(bias), dtype=torch.float32))

    def forward(self, p: torch.Tensor):
        assert p.dim() == 2, f"expected [B,C], got {tuple(p.shape)}"
        B, C = p.shape

        p = p.float().clamp(min=1e-6, max=1.0 - 1e-6)

        mass = p.sum(dim=-1)
        mass_feat = torch.log1p(mass).clamp(min=0.0) / math.log1p(float(C))

        e = p / (1.0 - p + 1e-8)
        e = e.clamp(max=1e6)

        alpha = e + 1.0
        S = alpha.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        E_p = alpha / S
        E_log_p = torch.digamma(alpha) - torch.digamma(S)
        H = -(E_p * E_log_p).sum(dim=-1)
        H_bar = (H / math.log(C)).clamp(min=0.0, max=1.0)

        u = (float(C) / S.squeeze(-1)).clamp(min=0.0)
        u_feat = torch.tanh((u - 1.0) / 2.0)

        top2 = torch.topk(E_p, k=2, dim=-1).values
        m = (top2[:, 0] - top2[:, 1]).clamp(0.0, 1.0)

        g_logit = (
            self.w_H * (H_bar - 0.5)
            + self.w_u * u_feat
            + self.w_mass * (-mass_feat)
            + self.w_m * (-m)
            + self.bias
        )

        g = torch.sigmoid(g_logit / 3.0).to(p.dtype)

        return (
            g,
            H_bar.to(p.dtype),
            m.to(p.dtype),
            u.to(p.dtype),
            mass.to(p.dtype),
        )
