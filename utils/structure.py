# utils/structure.py

import numpy as np
from typing import Optional
import torch
import networkx as nx
from tqdm import tqdm


def slic_k_partition(
    feats: np.ndarray,
    coords: np.ndarray,
    lam: float = 0.1,
    num_struct: Optional[int] = None,
    N_high: Optional[int] = None,
):
    """SLIC-K structure partitioning."""
    feats = feats.astype(np.float32)
    coords = coords.astype(np.float32)
    N = feats.shape[0]
    if N == 0:
        return np.zeros((0,), dtype=np.int64)

    if N_high is None:
        N_high = N

    if num_struct is None:
        k = max(1, int(lam * N_high))
    else:
        k = max(1, num_struct)

    S = float(np.sqrt(N_high / max(k, 1)))
    rng = np.random.default_rng(seed=0)

    if k >= N:
        return np.arange(N, dtype=np.int64)

    center_idx = rng.choice(N, size=k, replace=False)
    centers_f = feats[center_idx].copy()     # [k, D]
    centers_u = coords[center_idx].copy()    # [k, 2]

    labels = np.zeros(N, dtype=np.int64)

    for it in range(10):

        df = feats[:, None, :] - centers_f[None, :, :]
        df2 = np.sum(df * df, axis=-1)

        du = coords[:, None, :] - centers_u[None, :, :]
        du_norm = np.linalg.norm(du, axis=-1)
        du_term = (du_norm / S) ** 2

        dist = df2 + du_term
        new_labels = np.argmin(dist, axis=1)

        if it > 0 and np.array_equal(new_labels, labels):
            break

        labels = new_labels

        for j in range(k):
            mask = (labels == j)
            if not np.any(mask):
                ridx = rng.integers(low=0, high=N)
                centers_f[j] = feats[ridx]
                centers_u[j] = coords[ridx]
            else:
                centers_f[j] = feats[mask].mean(axis=0)
                centers_u[j] = coords[mask].mean(axis=0)

    return labels.astype(np.int64)


def mst_refine_edges(
    feats: np.ndarray,
    coords: np.ndarray,
    edge_index: np.ndarray,
    struct_id: np.ndarray,
):
    """MST refinement per structure."""
    feats = feats.astype(np.float32)
    coords = coords.astype(np.float32)
    struct_id = struct_id.astype(np.int64)
    N = feats.shape[0]

    src = edge_index[0]
    dst = edge_index[1]

    df = feats[src] - feats[dst]
    df2 = np.sum(df * df, axis=1)

    du = coords[src] - coords[dst]
    du2 = np.sum(du * du, axis=1)

    weights = df2 + du2

    E = edge_index.shape[1]
    mst_mask = np.zeros(E, dtype=bool)

    unique_structs = np.unique(struct_id)

    for sid in unique_structs:

        nodes_j = np.where(struct_id == sid)[0]
        if nodes_j.size <= 1:
            continue

        node_set = set(nodes_j.tolist())
        edge_indices_j = []
        local_edges = []

        for e in range(E):
            i = src[e]
            j = dst[e]
            if (i in node_set) and (j in node_set):
                edge_indices_j.append(e)
                local_edges.append((i, j))

        if len(edge_indices_j) == 0:
            continue

        G = nx.Graph()
        for n in nodes_j:
            G.add_node(int(n))
        for e_idx, (i, j) in zip(edge_indices_j, local_edges):
            w = float(weights[e_idx])
            G.add_edge(int(i), int(j), weight=w)

        T = nx.minimum_spanning_tree(G, weight="weight")

        mst_edges_set = {(u, v) for u, v, _ in T.edges(data=True)}
        mst_edges_set |= {(v, u) for u, v, _ in T.edges(data=True)}

        for e_idx, (i, j) in zip(edge_indices_j, local_edges):
            if (i, j) in mst_edges_set:
                mst_mask[e_idx] = True

    mst_edge_index = edge_index[:, mst_mask]
    return mst_edge_index


def add_structures_and_mst_to_graph(
    graph: dict,
    lam: float = 0.1,
    num_struct: Optional[int] = None,
    N_high: Optional[int] = None,
) -> dict:
    """Add SLIC-K struct_id and per-structure MST edges to the graph dict."""
    x: torch.Tensor = graph["x"]
    coords_t: torch.Tensor = graph["coords"]
    edge_index_t: torch.Tensor = graph["edge_index"]

    feats_np = x.cpu().numpy()
    coords_np = coords_t.cpu().numpy()
    edge_index_np = edge_index_t.cpu().numpy()

    # 1) SLIC-K partition
    struct_id_np = slic_k_partition(
        feats=feats_np,
        coords=coords_np,
        lam=lam,
        num_struct=num_struct,
        N_high=N_high,
    )

    mst_edge_index_np = mst_refine_edges(
        feats=feats_np,
        coords=coords_np,
        edge_index=edge_index_np,
        struct_id=struct_id_np,
    )

    struct_id = torch.from_numpy(struct_id_np).long()
    edge_index_mst = torch.from_numpy(mst_edge_index_np).long()

    graph["struct_id"] = struct_id
    graph["num_struct"] = int(struct_id.max().item() + 1)
    graph["edge_index_mst"] = edge_index_mst
    return graph
