"""Build graph_v2 tensors (kNN, MST, cross edges) from patch PKLs."""

import os
import argparse
import pickle
import torch
import numpy as np
from tqdm import tqdm

from data.dataset import WSIDataset, BCNBWSIDataset, BRACSWDataset
from utils.graph_utils import prepare_graph_inputs_from_batch, build_cross_resolution_edges
from utils.structure import add_structures_and_mst_to_graph, mst_refine_edges


def is_bcnb_root(root: str) -> bool:
    return "bcnb" in os.path.abspath(root).lower()


def is_bracs_root(root: str) -> bool:
    return "bracs" in os.path.abspath(root).lower()


def mag_to_resolution(mag: str) -> str:
    """Map x10/x20 mag names to BCNB resolution folder names (1.0/2.0)."""
    m = str(mag).lower().strip()
    if m in ["x10", "10", "1.0", "low"]:
        return "1.0"
    if m in ["x20", "20", "2.0", "high"]:
        return "2.0"
    raise ValueError(f"Unknown mag for BCNB: {mag}")


def load_bcnb_label_map(root: str):
    cand1 = os.path.join(root, "label_map.pkl")
    cand2 = os.path.join(root, "vit", "label_map.pkl")

    if os.path.exists(cand1):
        p = cand1
    elif os.path.exists(cand2):
        p = cand2
    else:
        raise FileNotFoundError(f"BCNB label_map not found: {cand1} or {cand2}")

    with open(p, "rb") as f:
        lm = pickle.load(f)
    return lm, p


def make_dataset(root: str, split: str, mag: str, tile: str, label_map=None):
    """Create WSIDataset, BCNBWSIDataset, or BRACSWDataset based on root path."""
    if is_bcnb_root(root):
        resolution = mag_to_resolution(mag)
        return BCNBWSIDataset(
            root_dir=root,
            split=split,
            resolution=resolution,
            label_map=label_map,
        )
    elif is_bracs_root(root):
        return BRACSWDataset(
            root_dir=root,
            split=split,
            mag=mag,
            tile_size=tile,
            label_map=label_map,
        )
    else:
        return WSIDataset(
            root_dir=root,
            split=split,
            mag=mag,
            tile_size=tile,
            label_map=label_map,
        )


def clamp_k(lam: float, N_high: int, k_min: int = 10, k_max: int = 300) -> int:
    k = int(round(lam * N_high))
    k = max(k_min, min(k, k_max))
    return max(1, k)


def add_structures_with_clamp(
    graph: dict,
    lam: float = 0.03,
    k_min: int = 10,
    k_max: int = 300,
) -> dict:
    N = graph["x"].shape[0]
    k = clamp_k(lam, N, k_min=k_min, k_max=k_max)
    graph_out = add_structures_and_mst_to_graph(
        graph,
        lam=lam,
        num_struct=k,
        N_high=N,
    )
    return graph_out


def compute_centroids(coords_np: np.ndarray, struct_id_np: np.ndarray, num_struct: int) -> np.ndarray:
    K = num_struct
    centroids = np.zeros((K, 2), dtype=np.float32)
    for k in range(K):
        mask = (struct_id_np == k)
        if not np.any(mask):
            centroids[k] = coords_np.mean(axis=0)
        else:
            centroids[k] = coords_np[mask].mean(axis=0)
    return centroids


def assign_high_to_low_struct(coords_high_np: np.ndarray, centroids_low: np.ndarray) -> np.ndarray:
    diff = coords_high_np[:, None, :] - centroids_low[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    struct_id_high = np.argmin(dists, axis=1)
    return struct_id_high.astype(np.int64)


def _extract_slide_id(batch: dict) -> str:
    sid = batch.get("slide_id")
    if isinstance(sid, (list, tuple)):
        sid = sid[0]
    return str(sid)


def _extract_class_name(batch: dict) -> str:
    cn = batch.get("class_name")
    if isinstance(cn, (list, tuple)):
        cn = cn[0]
    return str(cn)


def build_sid_to_index(ds: torch.utils.data.Dataset) -> dict:
    sid2idx = {}
    for i in range(len(ds)):
        b = ds[i]
        sid = _extract_slide_id(b)
        if sid in sid2idx:
            raise RuntimeError(f"Duplicate slide_id in dataset: {sid}")
        sid2idx[sid] = i
    return sid2idx


def process_split_pair_join(
    root: str,
    split: str,
    mag_low: str,
    mag_high: str,
    tile: str,
    out_dir_low: str,
    out_dir_high: str,
    out_dir_cross: str,
    lam: float = 0.03,
    k_min: int = 10,
    k_max: int = 300,
    k_cross: int = 4,
    max_radius=None,
    label_map=None,
    min_patches_low: int = 10,
    min_patches_high: int = 10,
    quiet_skip_log: bool = True,
):
    os.makedirs(out_dir_low, exist_ok=True)
    os.makedirs(out_dir_high, exist_ok=True)
    os.makedirs(out_dir_cross, exist_ok=True)

    ds_low = make_dataset(root=root, split=split, mag=mag_low, tile=tile, label_map=label_map)
    ds_high = make_dataset(root=root, split=split, mag=mag_high, tile=tile, label_map=label_map)

    sid2i_low = build_sid_to_index(ds_low)
    sid2i_high = build_sid_to_index(ds_high)

    sids = sorted(set(sid2i_low.keys()) & set(sid2i_high.keys()))
    missing_low = sorted(set(sid2i_high.keys()) - set(sid2i_low.keys()))
    missing_high = sorted(set(sid2i_low.keys()) - set(sid2i_high.keys()))

    print(f"[{split}] #low={len(ds_low)} #high={len(ds_high)} #paired(intersection)={len(sids)}")
    if missing_low:
        print(f"[{split}] WARNING: present in high but missing in low: {len(missing_low)} (e.g., {missing_low[:3]})")
    if missing_high:
        print(f"[{split}] WARNING: present in low but missing in high: {len(missing_high)} (e.g., {missing_high[:3]})")

    n_total_pairs = len(sids)
    n_skipped = 0
    skip_reasons = {"low_too_small": 0, "high_too_small": 0}
    skipped_sids = []

    k_list = []
    pbar = tqdm(sids, desc=f"[{split} {mag_low}<->{mag_high}] precompute(join)", total=len(sids))

    for sid in pbar:
        b_low = ds_low[sid2i_low[sid]]
        b_high = ds_high[sid2i_high[sid]]

        sid_low = _extract_slide_id(b_low)
        sid_high = _extract_slide_id(b_high)
        if sid_low != sid or sid_high != sid:
            raise RuntimeError(f"[JOIN BUG] expected sid={sid} but got low={sid_low}, high={sid_high}")

        if "feats" not in b_low or "feats" not in b_high:
            raise KeyError(
                f"Dataset batch must contain 'feats'. got keys low={list(b_low.keys())}, high={list(b_high.keys())}"
            )

        n_low = int(b_low["feats"].shape[0])
        n_high = int(b_high["feats"].shape[0])

        if n_low < min_patches_low:
            n_skipped += 1
            skip_reasons["low_too_small"] += 1
            skipped_sids.append((sid, f"low<{min_patches_low}", n_low, n_high))
            if not quiet_skip_log:
                print(f"[{split}] skip sid={sid} reason=low<{min_patches_low} n_low={n_low} n_high={n_high}")
            continue

        if n_high < min_patches_high:
            n_skipped += 1
            skip_reasons["high_too_small"] += 1
            skipped_sids.append((sid, f"high<{min_patches_high}", n_low, n_high))
            if not quiet_skip_log:
                print(f"[{split}] skip sid={sid} reason=high<{min_patches_high} n_low={n_low} n_high={n_high}")
            continue

        graph_low = prepare_graph_inputs_from_batch(b_low, k_knn=8)
        graph_high = prepare_graph_inputs_from_batch(b_high, k_knn=8)

        graph_low = add_structures_with_clamp(
            graph_low, lam=lam, k_min=k_min, k_max=k_max
        )

        struct_id_low = graph_low["struct_id"].cpu().numpy()
        coords_low_np = graph_low["coords"].cpu().numpy()
        K = int(graph_low["num_struct"])
        centroids_low = compute_centroids(coords_low_np, struct_id_low, K)

        feats_high_np = graph_high["x"].cpu().numpy()
        coords_high_np = graph_high["coords"].cpu().numpy()
        edge_high_np = graph_high["edge_index"].cpu().numpy()

        struct_id_high_np = assign_high_to_low_struct(coords_high_np, centroids_low)

        mst_high_np = mst_refine_edges(
            feats=feats_high_np,
            coords=coords_high_np,
            edge_index=edge_high_np,
            struct_id=struct_id_high_np,
        )

        graph_high["struct_id"] = torch.from_numpy(struct_id_high_np).long()
        graph_high["num_struct"] = K
        graph_high["edge_index_mst"] = torch.from_numpy(mst_high_np).long()

        cross_patch_index = build_cross_resolution_edges(
            graph_low["coords"], graph_high["coords"], k=k_cross, max_radius=max_radius
        ).long()

        class_name = _extract_class_name(b_low)
        save_low = {
            "x": graph_low["x"],
            "coords": graph_low["coords"],
            "edge_index_knn": graph_low["edge_index"],
            "edge_index_mst": graph_low["edge_index_mst"],
            "struct_id": graph_low["struct_id"],
            "num_struct": graph_low["num_struct"],
            "y": graph_low["y"],
            "slide_id": sid,
            "class_name": class_name,
            "centroids": torch.from_numpy(centroids_low),
        }
        save_high = {
            "x": graph_high["x"],
            "coords": graph_high["coords"],
            "edge_index_knn": graph_high["edge_index"],
            "edge_index_mst": graph_high["edge_index_mst"],
            "struct_id": graph_high["struct_id"],
            "num_struct": graph_high["num_struct"],
            "y": graph_high["y"],
            "slide_id": sid,
            "class_name": _extract_class_name(b_high),
        }
        save_cross = {
            "slide_id": sid,
            "low_mag": mag_low,
            "high_mag": mag_high,
            "k_cross": int(k_cross),
            "max_radius": max_radius,
            "cross_patch_index": cross_patch_index,
            "N_low": int(graph_low["coords"].shape[0]),
            "N_high": int(graph_high["coords"].shape[0]),
            "E_cross": int(cross_patch_index.shape[1]) if cross_patch_index.numel() > 0 else 0,
        }

        torch.save(save_low, os.path.join(out_dir_low, f"{sid}.pt"))
        torch.save(save_high, os.path.join(out_dir_high, f"{sid}.pt"))
        torch.save(save_cross, os.path.join(out_dir_cross, f"{sid}.pt"))

        k_list.append(K)

    if k_list:
        kt = torch.tensor(k_list, dtype=torch.float32)
        print(f"[{split} {mag_low}] K stats: min={int(kt.min())} max={int(kt.max())} mean={float(kt.mean()):.2f}")

    n_kept = n_total_pairs - n_skipped
    print(
        f"[{split}] kept={n_kept}/{n_total_pairs} "
        f"(skipped={n_skipped} | low<{min_patches_low}: {skip_reasons['low_too_small']}, "
        f"high<{min_patches_high}: {skip_reasons['high_too_small']})"
    )

    return {
        "split": split,
        "total_pairs": n_total_pairs,
        "kept": n_kept,
        "skipped": n_skipped,
        "low_too_small": skip_reasons["low_too_small"],
        "high_too_small": skip_reasons["high_too_small"],
        "skipped_sids": skipped_sids,
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="dataset root. if contains 'bcnb', BCNB branch is used.")
    ap.add_argument("--tile", type=str, default="256")
    ap.add_argument("--splits", type=str, default="train,val,test")
    ap.add_argument("--mag_low", type=str, default="x10")
    ap.add_argument("--mag_high", type=str, default="x20")
    ap.add_argument("--out_root", type=str, default="graph_v2")

    ap.add_argument("--lam", type=float, default=0.03)
    ap.add_argument("--k_min", type=int, default=10)
    ap.add_argument("--k_max", type=int, default=300)

    ap.add_argument("--k_cross", type=int, default=4)
    ap.add_argument("--max_radius", type=float, default=-1.0, help="-1 means None")

    ap.add_argument("--min_patches_low", type=int, default=4, help="min #patches for low (x10/1.0)")
    ap.add_argument("--min_patches_high", type=int, default=4, help="min #patches for high (x20/2.0)")
    return ap.parse_args()


def main():
    args = parse_args()

    root = args.root
    tile = args.tile

    lam = args.lam
    k_min = args.k_min
    k_max = args.k_max

    k_cross = args.k_cross
    max_radius = None if args.max_radius < 0 else args.max_radius

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    mag_low = args.mag_low
    mag_high = args.mag_high

    label_map = None
    if is_bcnb_root(root):
        label_map, label_map_path = load_bcnb_label_map(root)
        print(f"[BCNB] loaded label_map: {len(label_map)} from {label_map_path}")

    cross_dirname = f"cross_{mag_low}_{mag_high}"

    all_stats = []
    all_skipped_sids = []

    for split in splits:
        out_dir_low = os.path.join(args.out_root, split, mag_low)
        out_dir_high = os.path.join(args.out_root, split, mag_high)
        out_dir_cross = os.path.join(args.out_root, split, cross_dirname)

        st = process_split_pair_join(
            root=root,
            split=split,
            mag_low=mag_low,
            mag_high=mag_high,
            tile=tile,
            out_dir_low=out_dir_low,
            out_dir_high=out_dir_high,
            out_dir_cross=out_dir_cross,
            lam=lam,
            k_min=k_min,
            k_max=k_max,
            k_cross=k_cross,
            max_radius=max_radius,
            label_map=label_map,
            min_patches_low=args.min_patches_low,
            min_patches_high=args.min_patches_high,
            quiet_skip_log=True,
        )
        all_stats.append(st)
        all_skipped_sids.extend([(split, *x) for x in st.get("skipped_sids", [])])

    total_pairs = sum(x["total_pairs"] for x in all_stats)
    total_kept = sum(x["kept"] for x in all_stats)
    total_skipped = sum(x["skipped"] for x in all_stats)
    low_small = sum(x["low_too_small"] for x in all_stats)
    high_small = sum(x["high_too_small"] for x in all_stats)

    print(
        f"[ALL] kept={total_kept}/{total_pairs} (skipped={total_skipped} | "
        f"low_too_small={low_small}, high_too_small={high_small})"
    )

    if len(all_skipped_sids) > 0:
        os.makedirs(args.out_root, exist_ok=True)
        out_txt = os.path.join(args.out_root, "skipped_slides.txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            for rec in all_skipped_sids:
                split, sid, reason, n_low, n_high = rec
                f.write(f"{split}\t{sid}\t{reason}\t{n_low}\t{n_high}\n")
        print(f"[ALL] saved skipped list: {out_txt} (n={len(all_skipped_sids)})")


if __name__ == "__main__":
    main()