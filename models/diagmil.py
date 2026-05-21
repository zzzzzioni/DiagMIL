import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gating_dirc import DircUncertaintyGate
from models.stages import structure_align_loss
from utils.utils import _check_finite


class DiagMIL(nn.Module):
    def __init__(
        self,
        stage1: nn.Module,
        stage2: nn.Module,
        num_classes: int,
        hidden_dim: int = 256,
        lambda_H: float = 0.1,
        lambda_align: float = 0.1,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.stage1 = stage1
        self.stage2 = stage2
        self.num_classes = num_classes

        self.lambda_H = lambda_H
        self.lambda_align = lambda_align
        self.eps = eps

        self.gate = DircUncertaintyGate()

        self.W_s = nn.Linear(hidden_dim, hidden_dim)
        self.W_t = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        graph_low: dict,
        graph_high: dict,
        cross_patch_index: torch.Tensor,
        y: torch.Tensor | None = None,
        use_high: bool = True,
    ):
        """
        Forward pass.

        If ``use_high`` is False, Stage2 and cross-magnification paths are skipped
        and ``q_hat`` equals ``q1`` (cheaper inference).

        Returns:
            Dict with s1, q1, q2, q_hat, g, H_bar, m, u, and optional loss_* keys.
        """

        out1 = self.stage1(graph_low)
        logits1 = out1["logits1"]
        s1 = out1["s1"]
        s_low = out1["s_struct"]

        _check_finite("logits1", logits1)
        _check_finite("s1", s1)
        _check_finite("s_low(s_struct)", s_low)

        g, H_bar, m, u, _ = self.gate(s1)
        g_exp = g.unsqueeze(-1)

        s1_clamped = s1.clamp(min=self.eps, max=1.0 - self.eps)
        q1 = s1_clamped / s1_clamped.sum(dim=-1, keepdim=True).clamp(min=self.eps)

        _check_finite("q1", q1)
        _check_finite("H_bar", H_bar)
        _check_finite("m", m)
        _check_finite("g", g)

        if use_high:
            out2 = self.stage2(graph_low, graph_high, cross_patch_index)
            q2 = out2["q2"]
            logits2 = out2["logits2"]
            s_high = out2["s_struct_high"]
            q2 = q2.clamp(min=self.eps, max=1.0)
            q2 = q2 / q2.sum(dim=-1, keepdim=True).clamp(min=self.eps)
            _check_finite("logits2", logits2)
            _check_finite("q2", q2)
            _check_finite("s_high(s_struct_high)", s_high)
        else:
            q2 = q1.clone()
            logits2 = logits1.clone()
            s_high = s_low

        if use_high:
            q_hat = (1.0 - g_exp) * q1 + g_exp * q2     # [1, C]
        else:
            q_hat = q1
        q_hat = q_hat.clamp(min=self.eps, max=1.0)
        q_hat = q_hat / q_hat.sum(dim=-1, keepdim=True).clamp(min=self.eps)

        loss_total = loss_ce_hat = loss_ce_q2 = loss_H = loss_align = None

        if y is not None:
            if y.dim() > 1:
                y = y.view(-1)
            y = y.long()

            log_q_hat = torch.log(q_hat.clamp(min=self.eps))
            loss_ce_hat = F.nll_loss(log_q_hat, y)

            if use_high:
                loss_ce_q2 = F.cross_entropy(logits2, y, label_smoothing=0.05)
            else:
                loss_ce_q2 = logits2.new_zeros(1).squeeze()

            loss_H = self.lambda_H * H_bar.mean()

            if use_high:
                loss_align_raw = structure_align_loss(
                    s_low=s_low,
                    s_high=s_high,
                    W_s=self.W_s,
                    W_t=self.W_t,
                )
                loss_align = self.lambda_align * loss_align_raw
            else:
                loss_align = logits2.new_zeros(1).squeeze()

            loss_total = loss_ce_hat + loss_ce_q2 + loss_H + loss_align

        return {
            "logits1": logits1,
            "logits2": logits2,
            "s1": s1,
            "q1": q1,
            "q2": q2,
            "q_hat": q_hat,
            "g": g,
            "H_bar": H_bar,
            "m": m,
            "u": u,
            "loss_total": loss_total,
            "loss_ce_hat": loss_ce_hat,
            "loss_ce_q2": loss_ce_q2,
            "loss_H": loss_H,
            "loss_align": loss_align,
        }
