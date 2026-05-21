import torch
import torch.nn as nn
import torch.nn.functional as F


def scatter_add(src, index, dim: int, dim_size: int):
    assert dim == 0
    index = index.long()
    out_shape = (dim_size,) + src.shape[1:]
    out = torch.zeros(out_shape, device=src.device, dtype=src.dtype)
    return out.index_add(0, index, src)

def scatter_softmax(src, index, dim: int, dim_size: int):
    assert dim == 0
    index = index.long()

    src_f = src.float()

    max_per = torch.full((dim_size,), -1e9, device=src.device, dtype=torch.float32)
    max_per.scatter_reduce_(0, index, src_f, reduce="amax", include_self=True)
    max_per = torch.where(torch.isfinite(max_per), max_per, torch.zeros_like(max_per))

    exp = torch.exp(src_f - max_per[index])
    exp = torch.nan_to_num(exp, nan=0.0, posinf=0.0, neginf=0.0)

    denom = torch.zeros((dim_size,), device=src.device, dtype=torch.float32)
    denom.scatter_add_(0, index, exp)

    out = exp / (denom[index] + 1e-12)
    return out.to(src.dtype)


class AttnMP(nn.Module):
    """
    Fast Attention-based Message Passing (AttnMP), loop-free.

    edge_index: [2, E] where
      edge_index[0] = dst (i), edge_index[1] = src (j)
      i receives messages from j

    h_{i}^{(l+1)} = act( sum_{j in N(i)} softmax_j( score_{ij} ) * W h_j )
    score_{ij} = h_i dot h_j
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: nn.Module = nn.ReLU(),
        bias: bool = True,
        residual: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation
        self.residual = bool(residual and (in_dim == out_dim))

        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: [N, in_dim]
        edge_index: [2, E] (dst=row, src=col)
        """
        if edge_index.numel() == 0:
            out = self.lin(x)
            return self.activation(out) if self.activation is not None else out

        row = edge_index[0]  # dst indices, [E]
        col = edge_index[1]  # src indices, [E]

        # Linear projection for messages
        x_proj = self.lin(x)  # [N, out_dim]

        # scores: [E]
        # NOTE: attention score uses ORIGINAL x (in_dim) like your code
        scores = (x[row] * x[col]).sum(dim=-1)

        # Softmax over incoming edges per dst node i (grouped by row)
        attn = scatter_softmax(scores, row, dim=0, dim_size=x.size(0))

        # messages: [E, out_dim]
        messages = attn.unsqueeze(-1) * x_proj[col]

        # aggregate to dst nodes: [N, out_dim]
        out = scatter_add(messages, row, dim=0, dim_size=x.size(0))

        if self.residual:
            out = out + x  # only valid when in_dim == out_dim

        if self.activation is not None:
            out = self.activation(out)

        return out


class AttnPool(nn.Module):
    """
    Fast Attention pooling within each group (loop-free).

    For each group g:
      alpha_i = softmax_i( w^T h_i ) for i in group g
      pooled_g = sum_i alpha_i h_i
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.w_p = nn.Parameter(torch.randn(in_dim))

    def forward(
        self,
        x: torch.Tensor,         # [N, d]
        group_id: torch.Tensor,  # [N] values in {0..K-1}
        num_groups: int = None,
    ) -> torch.Tensor:
        if x.numel() == 0:
            K = int(num_groups) if num_groups is not None else 0
            return x.new_zeros((K, x.size(-1)))

        if num_groups is None:
            num_groups = int(group_id.max().item()) + 1

        # scores: [N]
        scores = (x * self.w_p).sum(dim=-1)

        # alpha: [N] softmax within each group
        alpha = scatter_softmax(scores, group_id, dim=0, dim_size=num_groups)

        # weighted sum per group
        pooled = scatter_add(alpha.unsqueeze(-1) * x, group_id, dim=0, dim_size=num_groups)  # [K, d]
        return pooled
