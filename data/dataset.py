import os
import glob
from typing import List, Dict, Optional, Any, Tuple
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


def load_pkl(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj

def parse_features_and_coords(obj: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    obj: {
      'features': { 'x-y': feature_vec(384,), ... },
      'caption': str,
      'subsite': np.int64,
      ...
    }
    return:
      feats:  [N, D] float32
      coords: [N, 2] float32  (x, y)
    """
    feat_dict = obj["features"]
    coords_list = []
    feats_list = []

    for key, vec in feat_dict.items():
        x_str, y_str = key.split("-")
        x, y = int(x_str), int(y_str)
        coords_list.append([x, y])
        feats_list.append(vec.astype("float32"))

    coords = np.asarray(coords_list, dtype="float32")
    feats = np.stack(feats_list, axis=0).astype("float32")
    return feats, coords


class WSIDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        mag: str = "x5",
        tile_size: str = "256",
        label_map: Optional[Dict[str, int]] = None,
    ):
        """
        WSI patch dataset from per-slide PKL files.

        ``root_dir`` layout: ``{mag}/{tile_size}/{split}/{class_name}/*.pkl``.

        ``label_map``: optional ``{class_name: id}``. If None, classes are ordered
        alphabetically after merging ``TVA`` and ``VA`` into ``TVA+VA``.
        """
        self.root_dir = root_dir
        self.split = split
        self.mag = mag
        self.tile_size = tile_size

        split_dir = os.path.join(root_dir, mag, tile_size, split)
        if not os.path.isdir(split_dir):
            raise ValueError(f"Split dir not found: {split_dir}")

        class_names_raw = [
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        ]
        class_names_raw = sorted(class_names_raw)

        merged_class_names: List[str] = []
        for c in class_names_raw:
            if c in ["TVA", "VA"]:
                if "TVA+VA" not in merged_class_names:
                    merged_class_names.append("TVA+VA")
            else:
                merged_class_names.append(c)

        class_names = merged_class_names

        if label_map is None:
            self.class_to_label = {c: i for i, c in enumerate(class_names)}
        else:
            self.class_to_label = label_map

        self.samples: List[Dict[str, Any]] = []

        for class_name_raw in class_names_raw:
            if class_name_raw in ["TVA", "VA"]:
                merged_name = "TVA+VA"
            else:
                merged_name = class_name_raw

            class_dir = os.path.join(split_dir, class_name_raw)
            pkl_paths = glob.glob(os.path.join(class_dir, "*.pkl"))

            for p in sorted(pkl_paths):
                slide_id = os.path.splitext(os.path.basename(p))[0]
                label = self.class_to_label[merged_name]
                self.samples.append(
                    {
                        "slide_id": slide_id,
                        "pkl_path": p,
                        "class_name": merged_name,
                        "label": label,
                    }
                )

        if len(self.samples) == 0:
            raise ValueError(f"No pkl files found in {split_dir}")

        print(
            f"[WSIDataset] split={split}, mag={mag}, tile={tile_size}, "
            f"num_classes={len(self.class_to_label)}, num_slides={len(self.samples)}"
        )
        print(f"[WSIDataset] class_to_label = {self.class_to_label}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_meta = self.samples[idx]
        obj = load_pkl(sample_meta["pkl_path"])
        feats_np, coords_np = parse_features_and_coords(obj)

        feats = torch.from_numpy(feats_np)
        coords = torch.from_numpy(coords_np)
        label = torch.tensor(sample_meta["label"], dtype=torch.long)

        return {
            "slide_id": sample_meta["slide_id"],
            "class_name": sample_meta["class_name"],
            "label": label,
            "feats": feats,
            "coords": coords,
        }


class GraphWSIDataset(Dataset):
    """Loads one slide graph per ``.pt`` file produced by ``preprocess_graph_v2.py``."""

    def __init__(
        self,
        graph_root: str,
        split: str = "train",
        mag: str = "x10",
        label_map: Optional[Dict[str, int]] = None,
    ):
        _ = label_map
        self.graph_root = graph_root
        self.split = split
        self.mag = mag

        split_dir = os.path.join(graph_root, split, mag)
        if not os.path.isdir(split_dir):
            raise ValueError(f"Graph split dir not found: {split_dir}")

        pt_paths = sorted(glob.glob(os.path.join(split_dir, "*.pt")))
        if len(pt_paths) == 0:
            raise ValueError(f"No .pt files found in {split_dir}")

        self.samples: List[Dict[str, Any]] = []
        class_to_label: Dict[str, int] = {}
        label_set = set()

        for p in pt_paths:
            obj = torch.load(p, map_location="cpu")

            slide_id = obj.get("slide_id")
            class_name = obj.get("class_name")
            y_val = int(obj.get("y"))

            self.samples.append(
                {
                    "path": p,
                    "slide_id": slide_id,
                    "class_name": class_name,
                }
            )

            label_set.add(y_val)
            if class_name is not None:
                if class_name in class_to_label and class_to_label[class_name] != y_val:
                    raise ValueError(
                        f"[GraphWSIDataset] Inconsistent label for class '{class_name}': "
                        f"{class_to_label[class_name]} vs {y_val}"
                    )
                class_to_label[class_name] = y_val

        self.class_to_label = class_to_label
        self.num_classes = len(label_set)

        print(
            f"[GraphWSIDataset] split={split}, mag={mag}, "
            f"num_classes={self.num_classes}, num_slides={len(self.samples)}"
        )
        if self.class_to_label:
            print(f"[GraphWSIDataset] class_to_label (from .pt): {self.class_to_label}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        meta = self.samples[idx]
        obj = torch.load(meta["path"], map_location="cpu")

        y = torch.as_tensor(int(obj["y"]), dtype=torch.long)

        x = obj["x"].float()
        coords = obj["coords"].float()
        edge_index_knn = obj["edge_index_knn"].long()
        edge_index_mst = obj["edge_index_mst"].long()
        struct_id = obj["struct_id"].long()
        num_struct = int(obj["num_struct"])

        return {
            "slide_id": obj["slide_id"],
            "class_name": obj["class_name"],
            "y": y,
            "x": x,
            "coords": coords,
            "edge_index": edge_index_knn,
            "edge_index_mst": edge_index_mst,
            "struct_id": struct_id,
            "num_struct": num_struct,
        }


class CrossGraphDataset(Dataset):
    """Cross-magnification patch edges (one ``.pt`` per slide)."""

    def __init__(
        self,
        graph_root: str,
        split: str = "train",
        cross_dirname: str = "cross_x10_x20",
    ):
        self.graph_root = graph_root
        self.split = split
        self.cross_dirname = cross_dirname

        split_dir = os.path.join(graph_root, split, cross_dirname)
        if not os.path.isdir(split_dir):
            raise ValueError(f"Cross split dir not found: {split_dir}")

        pt_paths = sorted(glob.glob(os.path.join(split_dir, "*.pt")))
        if len(pt_paths) == 0:
            raise ValueError(f"No .pt files found in {split_dir}")

        self.samples: List[Dict[str, Any]] = []

        for p in pt_paths:
            obj = torch.load(p, map_location="cpu")
            slide_id = obj.get("slide_id")

            if slide_id is None:
                slide_id = os.path.splitext(os.path.basename(p))[0]

            if "cross_patch_index" not in obj:
                raise KeyError(f"[CrossGraphDataset] 'cross_patch_index' missing in {p}")

            self.samples.append({"path": p, "slide_id": slide_id})

        print(
            f"[CrossGraphDataset] split={split}, cross={cross_dirname}, "
            f"num_slides={len(self.samples)}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        meta = self.samples[idx]
        obj = torch.load(meta["path"], map_location="cpu")

        slide_id = obj.get("slide_id")
        if slide_id is None:
            slide_id = meta["slide_id"]

        cross_patch_index = obj["cross_patch_index"].long()

        if cross_patch_index.numel() == 0:
            cross_patch_index = torch.zeros((2, 0), dtype=torch.long)

        return {
            "slide_id": slide_id,
            "cross_patch_index": cross_patch_index,
        }
    

class PairedWSIDataset(torch.utils.data.Dataset):
    """Zips low-mag, high-mag, and cross graphs for the same slide index."""

    def __init__(
        self,
        ds_low: GraphWSIDataset,
        ds_high: GraphWSIDataset,
        ds_cross: CrossGraphDataset,
    ):
        assert len(ds_low) == len(ds_high) == len(ds_cross), \
            "Low/High/Cross dataset length mismatch"

        self.ds_low = ds_low
        self.ds_high = ds_high
        self.ds_cross = ds_cross

        for i in range(len(ds_low)):
            sid_l = ds_low.samples[i]["slide_id"]
            sid_h = ds_high.samples[i]["slide_id"]
            sid_c = ds_cross.samples[i]["slide_id"]
            if not (sid_l == sid_h == sid_c):
                raise ValueError(
                    f"[PairedWSIDataset] slide_id mismatch at idx={i}: "
                    f"low={sid_l}, high={sid_h}, cross={sid_c}"
                )

        print(
            f"[PairedWSIDataset] num_slides={len(self.ds_low)} "
            f"(low/high/cross aligned)"
        )

    def __len__(self):
        return len(self.ds_low)

    def __getitem__(self, idx):
        low  = self.ds_low[idx]
        high = self.ds_high[idx]
        cross = self.ds_cross[idx]

        # runtime safety check (cheap)
        sid = low["slide_id"]
        assert high["slide_id"] == sid
        assert cross["slide_id"] == sid

        return {
            "low": low,
            "high": high,
            "cross": cross,
        }


def _parse_coors_list_to_coords(coors_list):
    xs = np.empty(len(coors_list), dtype=np.float32)
    ys = np.empty(len(coors_list), dtype=np.float32)
    for i, s in enumerate(coors_list):
        a, b = s.split("-")
        xs[i] = float(a)
        ys[i] = float(b)
    return torch.from_numpy(np.stack([xs, ys], axis=1))


def parse_bracs_features_and_coords(obj: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse BRACS slide PKL into patch features ``[N, D]`` and coords ``[N, 2]``.

    Expects ``obj['features']`` as a dict ``{'x-y': vector, ...}``, list/array of
    features (with ``coords`` or ``coors`` in ``obj``), or raises if missing.
    """
    if "features" in obj:
        feat_dict = obj["features"]
        if isinstance(feat_dict, dict):
            coords_list = []
            feats_list = []
            for key, vec in feat_dict.items():
                x_str, y_str = key.split("-")
                x, y = int(x_str), int(y_str)
                coords_list.append([x, y])
                feats_list.append(vec.astype("float32"))
            coords = np.asarray(coords_list, dtype="float32")
            feats = np.stack(feats_list, axis=0).astype("float32")
            return feats, coords
        elif isinstance(feat_dict, (list, np.ndarray)):
            feats = np.asarray(feat_dict, dtype=np.float32)
            if "coords" in obj:
                coords = np.asarray(obj["coords"], dtype=np.float32)
            elif "coors" in obj:
                coords = _parse_coors_list_to_coords(obj["coors"]).numpy()
            else:
                import warnings
                warnings.warn(f"BRACS: No coords found, generating dummy coords")
                coords = np.zeros((feats.shape[0], 2), dtype=np.float32)
            return feats, coords

    raise ValueError(
        "BRACS pkl file does not contain 'features' key. "
        "Please ensure features are extracted and stored in the pkl file."
    )


class BRACSWDataset(Dataset):
    """BRACS patches from ``patches/{split}/Type_*/{slide_id}/{resolution}.pkl``."""

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        mag: str = "x10",
        tile_size: str = "256",
        label_map: Optional[Dict[str, int]] = None,
    ):
        self.root_dir = root_dir
        self.split = split
        self.mag = mag
        self.tile_size = tile_size

        mag_lower = mag.lower().strip()
        if mag_lower in ["x10", "10", "1.0", "low"]:
            resolution = "1.0"
        elif mag_lower in ["x20", "20", "2.0", "high"]:
            resolution = "2.0"
        else:
            raise ValueError(f"Unknown mag for BRACS: {mag}")
        
        self.resolution = resolution

        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
            raise ValueError(f"Split dir not found: {split_dir}")

        class_names = [
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d)) and d.startswith("Type_")
        ]
        class_names = sorted(class_names)

        if label_map is None:
            self.class_to_label = {c: i for i, c in enumerate(class_names)}
        else:
            self.class_to_label = label_map

        self.samples: List[Dict[str, Any]] = []
        for class_name in class_names:
            class_dir = os.path.join(split_dir, class_name)
            slide_dirs = [
                d for d in os.listdir(class_dir)
                if os.path.isdir(os.path.join(class_dir, d))
            ]
            
            for slide_id in sorted(slide_dirs):
                pkl_path = os.path.join(class_dir, slide_id, f"{resolution}.pkl")
                if os.path.exists(pkl_path):
                    label = self.class_to_label[class_name]
                    self.samples.append(
                        {
                            "slide_id": slide_id,
                            "pkl_path": pkl_path,
                            "class_name": class_name,
                            "label": label,
                        }
                    )
        
        if len(self.samples) == 0:
            raise ValueError(f"No pkl files found in {split_dir} for resolution {resolution}")
        
        print(
            f"[BRACSWDataset] split={split}, mag={mag} (resolution={resolution}), "
            f"num_classes={len(self.class_to_label)}, num_slides={len(self.samples)}"
        )
        print(f"[BRACSWDataset] class_to_label = {self.class_to_label}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_meta = self.samples[idx]
        obj = load_pkl(sample_meta["pkl_path"])
        
        try:
            feats_np, coords_np = parse_bracs_features_and_coords(obj)
        except ValueError as e:
            raise ValueError(
                f"Failed to parse features from {sample_meta['pkl_path']}: {e}\n"
                f"BRACS pkl files must contain 'features' key with feature vectors."
            )
        
        feats = torch.from_numpy(feats_np)
        coords = torch.from_numpy(coords_np)
        label = torch.tensor(sample_meta["label"], dtype=torch.long)
        
        return {
            "slide_id": sample_meta["slide_id"],
            "class_name": sample_meta["class_name"],
            "label": label,
            "feats": feats,
            "coords": coords,
        }


class BCNBWSIDataset(Dataset):
    """BCNB slide PKLs indexed by a slide-id to label map."""

    def __init__(self, root_dir, split, resolution, label_map):
        self.root_dir = root_dir
        self.split = split
        self.resolution = str(resolution)

        self.dir = os.path.join(root_dir, split, self.resolution)
        if not os.path.isdir(self.dir):
            raise FileNotFoundError(f"dir not found: {self.dir}")

        self.paths = sorted(
            [os.path.join(self.dir, fn) for fn in os.listdir(self.dir) if fn.endswith(".pkl")],
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
        )

        self.label_map = label_map

        example_val = next(iter(self.label_map.values()))
        if isinstance(example_val, str):
            uniq = sorted(set(self.label_map.values()))
            self.class_to_label = {c: i for i, c in enumerate(uniq)}
        else:
            self.class_to_label = None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        sid = os.path.splitext(os.path.basename(p))[0]
        sid_int = int(sid)

        with open(p, "rb") as f:
            obj = pickle.load(f)

        feats = torch.from_numpy(np.asarray(obj["features"], dtype=np.float32))
        coords = _parse_coors_list_to_coords(obj["coors"])

        y_raw = self.label_map.get(sid_int, self.label_map.get(sid))
        if y_raw is None:
            raise KeyError(f"label missing for id={sid_int}")

        if self.class_to_label is not None:
            y = self.class_to_label[y_raw]
            class_name = y_raw
        else:
            y = int(y_raw)
            class_name = str(y)

        return {
            "slide_id": sid,
            "class_name": class_name,
            "label": torch.tensor(y, dtype=torch.long),
            "feats": feats,
            "coords": coords,
        }