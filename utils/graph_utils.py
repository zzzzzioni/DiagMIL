import numpy as np
from typing import Tuple
from sklearn.neighbors import NearestNeighbors
import torch


def build_knn_edges(coords: np.ndarray, k: int = 8, bidirectional: bool = True) -> np.ndarray:
    """
    coords: [N, 2] (numpy)
    k: number of neighbors per node
    return: edge_index: [2, E] (numpy, (src, dst))
    """
    N = coords.shape[0]
    k_eff = min(k + 1, N)

    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="auto").fit(coords)
    _, indices = nbrs.kneighbors(coords)   # indices: [N, k_eff]

    src_list, dst_list = [], []
    for i in range(N):
        for j in indices[i, 1:]:
            src_list.append(i)
            dst_list.append(j)
            if bidirectional:
                src_list.append(j)
                dst_list.append(i)

    edge_index = np.stack([np.array(src_list), np.array(dst_list)], axis=0)  # [2, E]
    return edge_index


def prepare_graph_inputs_from_batch(batch, k_knn: int = 8) -> dict:
    """Convert a WSIDataset batch (batch_size=1) into a graph dict for the model."""
    feats: torch.Tensor = batch["feats"]      # [N, D]
    coords: torch.Tensor = batch["coords"]    # [N, 2]
    label: torch.Tensor = batch["label"]      # scalar

    if feats.dim() == 3:
        # [B, N, D] -> [N, D]
        if feats.size(0) != 1:
            raise ValueError(f"only batch_size=1 | feats.shape={feats.shape}")
        feats = feats[0]
        coords = coords[0]
        if label.dim() > 0:
            label = label[0]
    elif feats.dim() != 2:
        raise ValueError(f"wrong feats shape: {feats.shape}")

    coords_np = coords.cpu().numpy()
    edge_index_np = build_knn_edges(coords_np, k=k_knn, bidirectional=True)

    edge_index = torch.from_numpy(edge_index_np).long()  # [2, E]

    graph = {
        "x": feats,          # [N, D]
        "coords": coords,    # [N, 2]
        "edge_index": edge_index,
        "y": label,
        "slide_id": batch["slide_id"],
        "class_name": batch["class_name"],
    }
    return graph


def build_cross_resolution_edges(
    coords_low,
    coords_high,
    k: int = 4,
    max_radius: float = None,
    bidirectional: bool = True,
) -> torch.Tensor:
    """
    Build cross-resolution edges between low-res and high-res patches.

    For each low-res patch i, connect to its k nearest high-res neighbors j.

    Args:
        coords_low:  [N_low, 2] (torch.Tensor or np.ndarray)
        coords_high: [N_high, 2] (torch.Tensor or np.ndarray)
        k:          number of high-res neighbors per low-res patch
        max_radius: optional. if set, only keep neighbors with distance <= max_radius
        bidirectional: if True, add edges in both directions (low->high, high->low)

    Returns:
        cross_patch_index: [2, E_cross] torch.LongTensor
            row 0: indices in [0, N_low-1]   (low)
            row 1: indices in [0, N_high-1]  (high)
    """
    # to numpy
    if isinstance(coords_low, torch.Tensor):
        coords_low_np = coords_low.detach().cpu().numpy()
        device = coords_low.device
    else:
        coords_low_np = np.asarray(coords_low, dtype=np.float32)
        device = torch.device("cpu")

    if isinstance(coords_high, torch.Tensor):
        coords_high_np = coords_high.detach().cpu().numpy()
    else:
        coords_high_np = np.asarray(coords_high, dtype=np.float32)

    N_low = coords_low_np.shape[0]
    N_high = coords_high_np.shape[0]
    if N_low == 0 or N_high == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    k_eff = min(k, N_high)

    # Fit k-NN on high-res coords
    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="auto").fit(coords_high_np)
    dists, indices = nbrs.kneighbors(coords_low_np)  # dists: [N_low, k_eff], indices: [N_low, k_eff]

    src_list = []  # low indices
    dst_list = []  # high indices

    for i in range(N_low):
        for kk in range(k_eff):
            j = int(indices[i, kk])
            if max_radius is not None and dists[i, kk] > max_radius:
                continue
            src_list.append(i)
            dst_list.append(j)

    if len(src_list) == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    cross_low = torch.tensor(src_list, dtype=torch.long, device=device)
    cross_high = torch.tensor(dst_list, dtype=torch.long, device=device)

    cross_patch_index = torch.stack([cross_low, cross_high], dim=0)  # [2, E_cross]

    if not bidirectional:
        return cross_patch_index

    return cross_patch_index