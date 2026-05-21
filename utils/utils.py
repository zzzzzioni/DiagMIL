import os
import json
import csv
from datetime import datetime
from typing import Optional
import random
import numpy as np

import torch
from torch.utils.data import DataLoader

from data.dataset import GraphWSIDataset, CrossGraphDataset, PairedWSIDataset


def compute_balanced_accuracy(y_true, y_pred, num_classes: int) -> float:
    """
    Macro-averaged per-class recall. Classes absent from y_true get recall 0.
    Compatible with older sklearn (no ``labels`` kwarg on balanced_accuracy_score).
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.size == 0:
        return float("nan")
    recalls = []
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() == 0:
            recalls.append(0.0)
        else:
            recalls.append(float((y_pred[mask] == c).mean()))
    return float(np.mean(recalls))


def set_seed(seed: int, deterministic: bool = True):
    """
    Call ONCE at the very beginning of the program (before dataset/dataloader/model init).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Deterministic CUDA ops; set CUBLAS_WORKSPACE_CONFIG (e.g. in test.sh).
        torch.use_deterministic_algorithms(True, warn_only=True)

def get_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state

def set_rng_state(state: dict):
    if state is None:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def move_graph_to_device(graph: dict, device: torch.device):
    out = {}
    for k, v in graph.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out


def build_graph_loader(
    graph_root,
    split,
    mag,
    class_to_label=None,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    persistent_workers=False,
):
    ds = GraphWSIDataset(
        graph_root=graph_root,
        split=split,
        mag=mag,
        label_map=class_to_label,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: batch[0],
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )
    return ds, loader



def build_cross_loader(graph_root, split, cross_dirname,
                       batch_size=1, shuffle=False,
                       num_workers=0, persistent_workers=False):
    ds = CrossGraphDataset(
        graph_root=graph_root,
        split=split,
        cross_dirname=cross_dirname,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: batch[0],
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    return ds, loader


def build_paired_loader(
    graph_root,
    split,
    low_mag,
    high_mag,
    cross_dirname,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    persistent_workers=False,
    seed=0,
):
    ds_low = GraphWSIDataset(
        graph_root=graph_root,
        split=split,
        mag=low_mag,
    )

    ds_high = GraphWSIDataset(
        graph_root=graph_root,
        split=split,
        mag=high_mag,
        label_map=ds_low.class_to_label,
    )

    ds_cross = CrossGraphDataset(
        graph_root=graph_root,
        split=split,
        cross_dirname=cross_dirname,
    )

    paired_ds = PairedWSIDataset(ds_low, ds_high, ds_cross)

    generator = torch.Generator().manual_seed(seed) if shuffle else None

    loader = DataLoader(
        paired_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: batch[0],
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        generator=generator,
    )

    return paired_ds, loader





def save_summary_json(summary: dict, run_dir: str, exp_name: str | None):
    exp = exp_name or "diagmil"
    path = os.path.join(run_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Info] Saved summary JSON to {path}")


def make_run_dir(base_dir: str, exp_name: str | None):
    exp = exp_name or "diagmil"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, exp, ts)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, exp, ts


def init_epoch_logger(run_dir: str, exp_name: str | None, args):
    exp = exp_name or "diagmil"

    csv_path = os.path.join(run_dir, "history.csv")
    config_path = os.path.join(run_dir, "config.json")

    fieldnames = [
        "epoch",
        "tr_loss_total", "tr_loss_ce_hat", "tr_loss_ce_q2", "tr_loss_H", "tr_loss_align",
        "tr_acc", "tr_auc",
        "va_loss_total", "va_loss_ce_hat", "va_loss_ce_q2", "va_loss_H", "va_loss_align",
        "va_acc", "va_auc",
        "lr",
        "num_train", "num_val",
    ]

    f = open(csv_path, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    f.flush()

    with open(config_path, "w") as cf:
        json.dump(vars(args), cf, indent=2)

    print(f"[Info] Run dir:        {run_dir}")
    print(f"[Info] Epoch log CSV: {csv_path}")
    print(f"[Info] Config JSON:    {config_path}")
    return csv_path, writer, f



def _prepare_graph_for_model(graph: dict) -> dict:
    if "edge_index" not in graph and "edge_index_knn" in graph:
        graph["edge_index"] = graph["edge_index_knn"]
    return graph


def _check_finite(name: str, t):
    if not isinstance(t, torch.Tensor):
        return True
    ok = torch.isfinite(t).all().item()
    if not ok:
        print(f"[NaN/Inf DETECTED] {name}:",
              "dtype", t.dtype,
              "shape", tuple(t.shape),
              "min", float(torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0).min().item()),
              "max", float(torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0).max().item()),
              "any_nan", bool(torch.isnan(t).any().item()),
              "any_inf", bool(torch.isinf(t).any().item()))
    return ok

