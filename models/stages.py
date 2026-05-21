import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attn_layers import AttnMP, AttnPool


def build_structure_graph(
    struct_id: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """
    Build structure-level graph E_S from patch-level edges (vectorized).

    For any patch-level edge (i, j) with struct_id[i] != struct_id[j],
    connect the corresponding structures.

    Returns:
        edge_index_struct: [2, E_s] LongTensor (undirected: both directions)
    """
    device = struct_id.device

    if edge_index is None or edge_index.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    # enforce long
    edge_index = edge_index.to(device)
    if edge_index.dtype != torch.long:
        edge_index = edge_index.long()

    src = edge_index[0]  # [E]
    dst = edge_index[1]  # [E]

    # guard index range
    if src.numel() > 0:
        n = struct_id.size(0)
        if src.min() < 0 or dst.min() < 0 or src.max() >= n or dst.max() >= n:
            raise RuntimeError(
                f"[build_structure_graph] edge_index out of range: "
                f"src[{int(src.min())},{int(src.max())}] dst[{int(dst.min())},{int(dst.max())}] vs N={n}"
            )

    si = struct_id[src]  # [E]
    sj = struct_id[dst]  # [E]

    mask = si != sj
    if mask.sum() == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    si = si[mask]
    sj = sj[mask]

    # undirected canonical (min, max)
    u = torch.minimum(si, sj)
    v = torch.maximum(si, sj)
    pairs = torch.stack([u, v], dim=0)          # [2, E']
    pairs = torch.unique(pairs, dim=1)          # [2, E_uniq]

    # expand to both directions
    rev_pairs = pairs[[1, 0], :]
    edge_index_struct = torch.cat([pairs, rev_pairs], dim=1)  # [2, 2*E_uniq]
    return edge_index_struct.to(device)


class Stage1DiagMIL(nn.Module):
    """
    Stage 1: low-res multi-class *sigmoid* scores s1 in [0,1]^C (NOT softmax).
    Output:
      - logits1: [1, C]
      - s1     : [1, C]  (= sigmoid(logits1))
      - s_struct / s_struct_out: structure embeddings
    """

    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_intra_layers: int = 1,
        num_inter_layers: int = 1,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Intra-structure AttnMP (patch graph with MST edges)
        intra_layers = []
        in_dim = feat_dim
        for _ in range(num_intra_layers):
            intra_layers.append(AttnMP(in_dim=in_dim, out_dim=hidden_dim))
            in_dim = hidden_dim
        self.intra_mp = nn.ModuleList(intra_layers)

        # Pool patches -> structures
        self.struct_pool = AttnPool(in_dim)  # in_dim == hidden_dim

        # Inter-structure AttnMP
        inter_layers = []
        s_dim = hidden_dim
        for _ in range(num_inter_layers):
            inter_layers.append(AttnMP(in_dim=s_dim, out_dim=s_dim))
        self.inter_mp = nn.ModuleList(inter_layers)

        # Readout over structures
        self.struct_readout_pool = AttnPool(s_dim)

        # classifier
        self.fc_stage1 = nn.Linear(s_dim, num_classes)

    def forward(self, graph: dict) -> dict:
        x = graph["x"]                         # [N, feat_dim]
        edge_all = graph["edge_index"]         # [2, E_all]
        edge_intra = graph["edge_index_mst"]   # [2, E_intra]
        struct_id = graph["struct_id"]         # [N]
        num_struct = int(graph["num_struct"])

        if num_struct <= 0:
            raise RuntimeError(f"[Stage1] num_struct must be > 0, got {num_struct}")

        # 1) intra MP on MST edges
        h = x
        for layer in self.intra_mp:
            h = layer(h, edge_intra)  # [N, hidden_dim]

        # 2) pool patches -> structures
        s_struct = self.struct_pool(h, struct_id, num_groups=num_struct)  # [K, hidden_dim]

        # 3) build structure graph from patch edges
        edge_struct = build_structure_graph(struct_id=struct_id, edge_index=edge_all)  # [2, E_s]

        # 4) inter MP on structure graph
        s_out = s_struct
        for layer in self.inter_mp:
            if edge_struct.numel() > 0:
                s_out = layer(s_out, edge_struct)
            else:
                empty_edge = torch.zeros((2, 0), dtype=torch.long, device=x.device)
                s_out = layer(s_out, empty_edge)

        # 5) readout -> logits -> sigmoid
        struct_ids_for_readout = torch.zeros(num_struct, dtype=torch.long, device=x.device)
        g_embed = self.struct_readout_pool(s_out, struct_ids_for_readout, num_groups=1)  # [1, hidden_dim]

        logits1 = self.fc_stage1(g_embed)          # [1, C]
        s1 = torch.sigmoid(logits1)                # [1, C]

        return {
            "s1": s1,                  # [1, C] 
            "logits1": logits1,        # [1, C]
            "s_struct": s_struct,      # [K, D] (pre-inter)
            "s_struct_out": s_out,     # [K, D] (post-inter)
            "g_embed": g_embed.squeeze(0),  # [D]
        }


class Stage2DiagMIL(nn.Module):
    """
    Stage 2: high-res multi-class softmax distribution q2.
    Output:
      - logits2: [1, C]
      - q2     : [1, C] (softmax)
      - s_struct_high / s_struct_high_out
    """

    def __init__(
        self,
        feat_dim_low: int,
        feat_dim_high: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_intra_layers: int = 1,
        num_inter_layers: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.proj_low = nn.Linear(feat_dim_low, hidden_dim)
        self.proj_high = nn.Linear(feat_dim_high, hidden_dim)

        self.cross_mp = AttnMP(
            in_dim=hidden_dim,
            out_dim=hidden_dim,
            activation=nn.ReLU(),
            residual=True,
        )

        intra_layers = []
        in_dim = hidden_dim
        for _ in range(num_intra_layers):
            intra_layers.append(AttnMP(in_dim=in_dim, out_dim=hidden_dim))
            in_dim = hidden_dim
        self.intra_mp_high = nn.ModuleList(intra_layers)

        self.struct_pool_high = AttnPool(hidden_dim)

        inter_layers = []
        s_dim = hidden_dim
        for _ in range(num_inter_layers):
            inter_layers.append(AttnMP(in_dim=s_dim, out_dim=s_dim))
        self.inter_mp_high = nn.ModuleList(inter_layers)

        self.struct_readout_pool_high = AttnPool(s_dim)
        self.fc_stage2 = nn.Linear(s_dim, num_classes)

    def forward(self, graph_low: dict, graph_high: dict, cross_patch_index: torch.Tensor = None) -> dict:
        x_low = graph_low["x"]       # [N_low, feat_dim_low]
        x_high = graph_high["x"]     # [N_high, feat_dim_high]

        edge_all_high = graph_high["edge_index"]         # [2, E_high_all]
        edge_intra_high = graph_high["edge_index_mst"]   # [2, E_high_intra]
        struct_id_high = graph_high["struct_id"]         # [N_high]
        num_struct_high = int(graph_high["num_struct"])

        device = x_low.device
        if num_struct_high <= 0:
            raise RuntimeError(f"[Stage2] num_struct_high must be > 0, got {num_struct_high}")

        # 0) project
        h_low = self.proj_low(x_low)     # [N_low, D]
        h_high = self.proj_high(x_high)  # [N_high, D]

        # 1) cross-resolution MP
        if cross_patch_index is not None and cross_patch_index.numel() > 0:
            cross_patch_index = cross_patch_index.to(device)
            if cross_patch_index.dtype != torch.long:
                cross_patch_index = cross_patch_index.long()

            low_idx = cross_patch_index[0]
            high_idx = cross_patch_index[1]

            N_low = h_low.size(0)
            N_high = h_high.size(0)

            # guard
            if low_idx.min() < 0 or high_idx.min() < 0 or low_idx.max() >= N_low or high_idx.max() >= N_high:
                raise RuntimeError(
                    f"[Stage2] cross_patch_index out of range: "
                    f"low[{int(low_idx.min())},{int(low_idx.max())}] vs N_low={N_low}, "
                    f"high[{int(high_idx.min())},{int(high_idx.max())}] vs N_high={N_high}"
                )

            h_combined = torch.cat([h_low, h_high], dim=0)  # [N_low+N_high, D]

            # make undirected cross edges in combined index space
            src1 = low_idx
            dst1 = N_low + high_idx
            src2 = dst1
            dst2 = src1

            src = torch.cat([src1, src2], dim=0)
            dst = torch.cat([dst1, dst2], dim=0)

            # IMPORTANT: edge_index = [src, dst]
            edge_cross_combined = torch.stack([src, dst], dim=0)  # [2, 2E]

            h_combined = self.cross_mp(h_combined, edge_cross_combined)

            h_high_cr = h_combined[N_low:]  # [N_high, D]
        else:
            h_high_cr = h_high

        # 2) intra MP high
        h_h = h_high_cr
        for layer in self.intra_mp_high:
            h_h = layer(h_h, edge_intra_high)

        # 3) pool to structures
        s_struct_high = self.struct_pool_high(h_h, struct_id_high, num_groups=num_struct_high)  # [K, D]

        # 4) structure graph high
        edge_struct_high = build_structure_graph(struct_id=struct_id_high, edge_index=edge_all_high)

        # 5) inter MP high
        s_out_high = s_struct_high
        for layer in self.inter_mp_high:
            if edge_struct_high.numel() > 0:
                s_out_high = layer(s_out_high, edge_struct_high)
            else:
                empty_edge = torch.zeros((2, 0), dtype=torch.long, device=device)
                s_out_high = layer(s_out_high, empty_edge)

        # 6) readout
        struct_ids_for_readout = torch.zeros(num_struct_high, dtype=torch.long, device=device)
        g_embed_high = self.struct_readout_pool_high(s_out_high, struct_ids_for_readout, num_groups=1)  # [1, D]

        # 7) logits + softmax
        logits2 = self.fc_stage2(g_embed_high)            # [1, C]
        q2 = F.softmax(logits2, dim=-1)                   # [1, C]

        return {
            "q2": q2,
            "logits2": logits2,
            "s_struct_high": s_struct_high,
            "s_struct_high_out": s_out_high,
            "g_embed_high": g_embed_high.squeeze(0),
        }


def structure_align_loss(
    s_low: torch.Tensor,
    s_high: torch.Tensor,
    W_s: nn.Linear,
    W_t: nn.Linear,
    T: float = 1.0,
) -> torch.Tensor:
    """
    KL distillation between projected structure embeddings.

    Returns:
        scalar tensor
    """
    device = s_low.device
    if s_low.numel() == 0 or s_high.numel() == 0:
        return torch.tensor(0.0, device=device)

    K = min(s_low.size(0), s_high.size(0))
    if K <= 0:
        return torch.tensor(0.0, device=device)

    s_low = s_low[:K]
    s_high = s_high[:K]

    # project + temperature
    z_t = W_t(s_high) / max(T, 1e-8)
    z_s = W_s(s_low) / max(T, 1e-8)

    # teacher prob / student log prob
    p_t = F.softmax(z_t, dim=-1)
    log_p_s = F.log_softmax(z_s, dim=-1)

    # KL(p_t || p_s)
    log_p_t = torch.log(p_t.clamp(min=1e-8))
    kl = (p_t * (log_p_t - log_p_s)).sum(dim=-1).mean()

    return (T ** 2) * kl
