import os
import argparse
import csv
from datetime import datetime
import numpy as np
from tqdm import tqdm

import torch
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from models.stages import Stage1DiagMIL, Stage2DiagMIL
from models.diagmil import DiagMIL

from utils.utils import (
    move_graph_to_device,
    build_graph_loader,
    build_cross_loader,
    save_summary_json,
    _prepare_graph_for_model,
    set_seed,
    compute_balanced_accuracy,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser("DiagMIL Eval")

    # data
    p.add_argument("--graph_root", type=str, default="./graph_v2")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--low_mag", type=str, default="x10")
    p.add_argument("--high_mag", type=str, default="x20")
    p.add_argument("--num_classes", type=int, required=True)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--lambda_H", type=float, default=0.1)
    p.add_argument("--lambda_align", type=float, default=0.1)

    # ckpt + output
    p.add_argument("--ckpt_path", type=str, required=True, help="path to checkpoints/best.pt or last.pt")
    p.add_argument("--out_dir", type=str, default=None, help="output directory (default: sibling of ckpt)")
    p.add_argument("--exp_name", type=str, default="eval")

    # repro
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")

    return p.parse_args()


def build_model(args, feat_dim_low, feat_dim_high):
    hidden_dim = 256

    stage1 = Stage1DiagMIL(
        feat_dim=feat_dim_low,
        num_classes=args.num_classes,
        hidden_dim=hidden_dim,
    )
    stage2 = Stage2DiagMIL(
        feat_dim_low=feat_dim_low,
        feat_dim_high=feat_dim_high,
        num_classes=args.num_classes,
        hidden_dim=hidden_dim,
    )
    model = DiagMIL(
        stage1=stage1,
        stage2=stage2,
        num_classes=args.num_classes,
        hidden_dim=hidden_dim,
        lambda_H=args.lambda_H,
        lambda_align=args.lambda_align,
    )
    return model.to(DEVICE)


def load_model_from_ckpt(ckpt_path, model):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    # train_new.py save_checkpoint format: {"model": state_dict, ...}
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    return ckpt


@torch.no_grad()
def evaluate_diagmil(model, loader_low, loader_high, loader_cross, args, run_dir: str, class_to_label: dict):
    model.eval()

    y_true, y_pred = [], []
    probs_list = []
    total_loss = 0.0
    n_samples = 0

    label_to_class = {v: k for k, v in class_to_label.items()}
    class_cols = [f"score_{label_to_class.get(i, str(i))}" for i in range(args.num_classes)]

    rows = []

    for batch_low, batch_high, batch_cross in tqdm(
        zip(loader_low, loader_high, loader_cross),
        total=min(len(loader_low), len(loader_high), len(loader_cross)),
        desc=f"[Eval:{args.split}]",
    ):
        graph_low = _prepare_graph_for_model(batch_low)
        graph_high = _prepare_graph_for_model(batch_high)

        sid_l = graph_low.get("slide_id", None)
        sid_h = graph_high.get("slide_id", None)
        sid_c = batch_cross.get("slide_id", None)

        if sid_l is not None and sid_h is not None and sid_l != sid_h:
            raise RuntimeError(f"[PAIR MISMATCH] low({sid_l}) != high({sid_h})")
        if sid_l is not None and sid_c is not None and sid_l != sid_c:
            raise RuntimeError(f"[PAIR MISMATCH] low({sid_l}) != cross({sid_c})")

        cross_index = batch_cross["cross_patch_index"]

        graph_low = move_graph_to_device(graph_low, DEVICE)
        graph_high = move_graph_to_device(graph_high, DEVICE)
        cross_index = cross_index.to(DEVICE)
        y = graph_low["y"].view(1).to(DEVICE)

        out = model(graph_low=graph_low, graph_high=graph_high, cross_patch_index=cross_index, y=y)

        loss = out["loss_total"]
        probs = out["q_hat"]

        total_loss += float(loss.item())
        n_samples += 1

        p = probs.squeeze(0).detach().cpu().numpy()
        pred = int(np.argmax(p))
        label = int(y.item())

        y_true.append(label)
        y_pred.append(pred)
        probs_list.append(p)

        row = {
            "slide_id": sid_l,
            "gt_label": label,
            "gt_class": label_to_class.get(label, str(label)),
            "pred_label": pred,
            "pred_class": label_to_class.get(pred, str(pred)),
            "correct": int(pred == label),
            "loss": float(loss.item()),
        }
        for i in range(args.num_classes):
            row[class_cols[i]] = float(p[i])
        rows.append(row)

    os.makedirs(run_dir, exist_ok=True)
    out_csv = os.path.join(run_dir, f"{args.split}_per_slide_scores.csv")
    fieldnames = ["slide_id","gt_label","gt_class","pred_label","pred_class","correct","loss"] + class_cols

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[Info] Saved per-slide scores CSV: {out_csv}")

    avg_loss = total_loss / max(n_samples, 1)
    probs_arr = np.vstack(probs_list) if len(probs_list) else np.zeros((0, args.num_classes), dtype=np.float32)

    # AUC
    try:
        if len(np.unique(np.array(y_true))) < 2:
            auc = float("nan")
        elif args.num_classes == 2:
            auc = float(roc_auc_score(np.array(y_true), probs_arr[:, 1]))
        else:
            auc = float(roc_auc_score(np.array(y_true), probs_arr, multi_class="ovr", average="macro"))
    except Exception:
        auc = float("nan")

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = compute_balanced_accuracy(y_true, y_pred, args.num_classes)
    print(
        f"\n[Eval:{args.split}] avg_loss={avg_loss:.4f} acc={acc:.4f} "
        f"bal_acc={bal_acc:.4f} auc={auc:.4f}"
    )
    print("\nClassification report:")
    print(classification_report(y_true, y_pred))

    return {
        "loss": float(avg_loss),
        "acc": float(acc),
        "bal_acc": float(bal_acc),
        "auc": float(auc),
        "csv": out_csv,
    }


def main():
    args = parse_args()
    assert args.batch_size == 1, "this eval script assumes batch_size=1 (like your trainer)"

    set_seed(args.seed, deterministic=args.deterministic)

    # loaders: we want consistent label mapping => build from TRAIN split first
    ds_low_train, _ = build_graph_loader(
        args.graph_root, "train", args.low_mag,
        batch_size=1, shuffle=False, num_workers=args.num_workers
    )
    class_to_label = ds_low_train.class_to_label

    ds_low, loader_low = build_graph_loader(
        args.graph_root, args.split, args.low_mag,
        class_to_label=class_to_label,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )
    ds_high, loader_high = build_graph_loader(
        args.graph_root, args.split, args.high_mag,
        class_to_label=class_to_label,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    cross_dirname = f"cross_{args.low_mag}_{args.high_mag}"
    ds_cross, loader_cross = build_cross_loader(
        args.graph_root, args.split, cross_dirname=cross_dirname,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
    )

    # infer feat dims
    sample_graph_low = _prepare_graph_for_model(next(iter(loader_low)))
    sample_graph_high = _prepare_graph_for_model(next(iter(loader_high)))
    feat_dim_low = sample_graph_low["x"].shape[-1]
    feat_dim_high = sample_graph_high["x"].shape[-1]

    model = build_model(args, feat_dim_low, feat_dim_high)
    ckpt = load_model_from_ckpt(args.ckpt_path, model)

    # output dir
    if args.out_dir is None:
        # default: <ckpt_dir>/../eval_<timestamp>
        base = os.path.dirname(os.path.dirname(args.ckpt_path))  # .../checkpoints/ -> run_dir
        args.out_dir = os.path.join(base, f"{args.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(args.out_dir, exist_ok=True)

    metrics = evaluate_diagmil(
        model, loader_low, loader_high, loader_cross,
        args=args, run_dir=args.out_dir, class_to_label=class_to_label
    )

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "split": args.split,
        "ckpt_path": args.ckpt_path,
        "config": vars(args),
        "feat_dim_low": int(feat_dim_low),
        "feat_dim_high": int(feat_dim_high),
        "metrics": metrics,
        "ckpt_meta": {
            k: ckpt.get(k)
            for k in ["epoch", "best_val_loss", "best_val_acc", "best_val_auc", "best_val", "best_epoch"]
            if isinstance(ckpt, dict) and k in ckpt
        },
    }
    save_summary_json(summary, args.out_dir, args.exp_name)
    print(f"[Info] Saved summary.json in {args.out_dir}")


if __name__ == "__main__":
    main()
