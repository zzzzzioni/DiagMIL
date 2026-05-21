import os
import argparse
from tqdm import tqdm
import csv
from datetime import datetime
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from models.stages import Stage1DiagMIL, Stage2DiagMIL
from models.diagmil import DiagMIL

from utils.utils import (
    move_graph_to_device,
    save_summary_json,
    make_run_dir,
    init_epoch_logger,
    _prepare_graph_for_model,
    _check_finite,
    set_seed,
    get_rng_state,
    set_rng_state,
    build_paired_loader,
)
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser("DiagMIL Trainer (resume-ready)")
    seed_default = int(os.environ.get("SEED", "0"))

    parser.add_argument("--low_mag", type=str, default="x10")
    parser.add_argument("--high_mag", type=str, default="x20")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--test_split", type=str, default="test")

    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--lambda_H", type=float, default=0.1)
    parser.add_argument("--lambda_align", type=float, default=0.1)

    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--graph_root", type=str, default="./graph_v2")

    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--stop_on_nan", action="store_true")
    parser.add_argument("--eps_prob", type=float, default=1e-8)
    parser.add_argument("--patience", type=int, default=30,
                        help="early-stop patience epochs (default=30)")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--accum_steps", type=int, default=8,
                        help="gradient accumulation steps (effective batch = batch_size * accum_steps)")
    parser.add_argument("--seed", type=int, default=seed_default)
    parser.add_argument("--deterministic", action="store_true", help="cudnn deterministic on")
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoints/last.pt")
    parser.add_argument("--save_every", type=int, default=1, help="save last.pt every N epochs (default=1)")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers (0 for reproducible runs)")

    parser.add_argument(
        "--test_ckpt",
        type=str,
        default="best_auc",
        choices=["best_auc", "best_loss", "last"],
        help="checkpoint used for final test eval (default: best_auc, val macro AUC max)",
    )

    return parser.parse_args()


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


def save_checkpoint(
    path, model, optimizer, epoch, best_val, best_epoch, args,
    val_acc=None, val_auc=None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "best_val": float(best_val),
        "best_epoch": int(best_epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "rng": get_rng_state(),
        "args": vars(args),
    }
    if val_acc is not None:
        ckpt["val_acc"] = float(val_acc)
    if val_auc is not None:
        ckpt["val_auc"] = float(val_auc)
    torch.save(ckpt, path)

def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

    try:
        set_rng_state(ckpt.get("rng"))
    except Exception as e:
        print("[Resume] set_rng_state failed -> skip:", repr(e))

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_val = float(ckpt.get("best_val", float("inf")))
    best_epoch = int(ckpt.get("best_epoch", -1))
    return start_epoch, best_val, best_epoch


def run_epoch(
    model,
    loader,
    optimizer,
    desc,
    args,
    train=True,
):
    if train:
        model.train()
        context = torch.enable_grad()
        assert optimizer is not None
    else:
        model.eval()
        context = torch.no_grad()

    sums = {
        "loss_total": 0.0,
        "loss_ce_hat": 0.0,
        "loss_ce_q2": 0.0,
        "loss_H": 0.0,
        "loss_align": 0.0,
        "acc": 0.0,
        "count": 0,
    }

    all_y_true = []
    all_y_prob = []
    all_y_pred = []

    def _safe_item(x):
        if x is None:
            return 0.0
        return float(x.detach().item())

    with context:
        for batch in tqdm(
            loader,
            total=len(loader),
            desc=desc,
        ):
            graph_low  = _prepare_graph_for_model(batch["low"])
            graph_high = _prepare_graph_for_model(batch["high"])
            batch_cross = batch["cross"]
            cross_index = batch_cross["cross_patch_index"]

            sid_l = graph_low.get("slide_id", None)
            sid_h = graph_high.get("slide_id", None)
            sid_c = batch_cross.get("slide_id", None)

            if sid_l is None or sid_h is None or sid_c is None:
                raise RuntimeError(f"No slide id!")
            if sid_l is not None and sid_h is not None and sid_l != sid_h:
                raise RuntimeError(f"[PAIR MISMATCH] low({sid_l}) != high({sid_h})")
            if sid_l is not None and sid_c is not None and sid_l != sid_c:
                raise RuntimeError(f"[PAIR MISMATCH] low({sid_l}) != cross({sid_c})")

            y_l = int(graph_low["y"])
            y_h = int(graph_high["y"])
            assert y_l == y_h, f"Y MISMATCH: slide={sid_l} low_y={y_l} high_y={y_h}"

            graph_low = move_graph_to_device(graph_low, DEVICE)
            graph_high = move_graph_to_device(graph_high, DEVICE)
            cross_index = cross_index.to(DEVICE)
            y = graph_low["y"].view(1).to(DEVICE)

            if train:
                if (sums["count"] % args.accum_steps) == 0:
                    optimizer.zero_grad(set_to_none=True)

            out = model(
                graph_low=graph_low,
                graph_high=graph_high,
                cross_patch_index=cross_index,
                y=y,
            )

            loss = out["loss_total"]
            probs = out["q_hat"]

            ok = True
            ok &= _check_finite("loss", loss)
            ok &= _check_finite("probs", probs)
            ok &= _check_finite("q1", out.get("q1"))
            ok &= _check_finite("q2", out.get("q2"))
            ok &= _check_finite("g", out.get("g"))
            if not ok:
                _check_finite("x_low", graph_low["x"])
                _check_finite("x_high", graph_high["x"])
                if isinstance(cross_index, torch.Tensor) and cross_index.numel() > 0:
                    print("[cross_index] shape:", tuple(cross_index.shape),
                          "min:", int(cross_index.min().item()),
                          "max:", int(cross_index.max().item()),
                          "N_low:", int(graph_low["x"].shape[0]),
                          "N_high:", int(graph_high["x"].shape[0]))
                print("[bad sample] slide_id:", sid_l)
                if args.stop_on_nan:
                    raise RuntimeError("Stop: NaN/Inf detected")
                else:
                    continue

            if train:
                loss_scaled = loss / float(args.accum_steps)
                loss_scaled.backward()

                if ((sums["count"] + 1) % args.accum_steps) == 0:
                    if args.grad_clip is not None and args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                    optimizer.step()

            raw_probs_for_auc = probs.detach()
            if args.eps_prob is not None and args.eps_prob > 0:
                probs = probs.clamp(min=args.eps_prob, max=1.0)
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)

            pred = probs.argmax(dim=-1)
            acc = (pred == y).float().item()

            sums["loss_total"] += float(loss.detach().item())
            sums["loss_ce_hat"] += _safe_item(out.get("loss_ce_hat"))
            sums["loss_ce_q2"] += _safe_item(out.get("loss_ce_q2"))
            sums["loss_H"] += _safe_item(out.get("loss_H"))
            sums["loss_align"] += _safe_item(out.get("loss_align"))
            sums["acc"] += float(acc)
            sums["count"] += 1

            all_y_true.append(int(y.item()))
            all_y_prob.append(raw_probs_for_auc.cpu().numpy()[0])
            all_y_pred.append(int(pred.item()))

    if train and (sums["count"] % args.accum_steps) != 0:
        if args.grad_clip is not None and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()

    n = max(sums["count"], 1)
    avg = {k: (v / n) for k, v in sums.items() if k != "count"}
    avg["count"] = sums["count"]

    y_true_np = np.array(all_y_true, dtype=np.int64)
    y_prob_np = np.array(all_y_prob, dtype=np.float32)
    y_pred_np = np.array(all_y_pred, dtype=np.int64)

    auc_macro = float("nan")
    try:
        if y_prob_np.shape[0] == 0 or len(np.unique(y_true_np)) < 2:
            pass
        elif args.num_classes == 2:
            auc_macro = float(roc_auc_score(y_true_np, y_prob_np[:, 1]))
        else:
            auc_macro = float(roc_auc_score(
                y_true_np, y_prob_np,
                multi_class="ovr", average="macro",
                labels=list(range(args.num_classes)),
            ))
    except Exception as e:
        print("[AUC ERROR]", repr(e))

    auc = float(auc_macro)

    return avg, auc


@torch.no_grad()
def evaluate_diagmil(model, loader, args, run_dir: str, class_to_label: dict):
    model.eval()
    y_true = []
    y_pred = []

    probs_list = []

    total_loss = 0.0
    n_samples = 0
    g_sum = 0.0

    label_to_class = {v: k for k, v in class_to_label.items()}
    rows = []
    class_cols = [f"score_{label_to_class.get(i, str(i))}" for i in range(args.num_classes)]


    for batch in tqdm(
        loader,
        total=len(loader),
        desc="[Test Eval]",
    ):
        graph_low  = _prepare_graph_for_model(batch["low"])
        graph_high = _prepare_graph_for_model(batch["high"])
        batch_cross = batch["cross"]

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

        if not _check_finite("test_loss", loss) or not _check_finite("test_probs", probs):
            print("[bad test sample] slide_id:", sid_l)
            continue

        total_loss += float(loss.item())
        n_samples += 1

        if args.eps_prob is not None and args.eps_prob > 0:
            probs = probs.clamp(min=args.eps_prob, max=1.0)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)

        p = probs.squeeze(0).detach().cpu().numpy()
        pred = int(np.argmax(p))
        label = int(y.item())

        g_val = out["g"].squeeze().detach().cpu().item()
        g_val = round(float(g_val), 3)
        g_sum += g_val

        H_bar_val = out["H_bar"].squeeze().detach().cpu().item()
        H_bar_val = round(float(H_bar_val), 3)

        m_val = out["m"].squeeze().detach().cpu().item()
        m_val = round(float(m_val), 3)

        u_val = float(out["u"].squeeze().detach().cpu().item())
        u_val = round(float(u_val), 3)


        q1_probs = F.softmax(out["logits1"], dim=-1).squeeze(0)
        q2_probs = F.softmax(out["logits2"], dim=-1).squeeze(0)

        q1_pred = int(torch.argmax(q1_probs).item())
        q2_pred = int(torch.argmax(q2_probs).item())


        y_true.append(label)
        y_pred.append(pred)
        probs_list.append(p)

        row = {
            "slide_id": sid_l,
            "gt_class": label_to_class.get(label, str(label)),
            "pred_class": label_to_class.get(pred, str(pred)),
            "pred_q1": label_to_class.get(q1_pred, str(q1_pred)),
            "pred_q2": label_to_class.get(q2_pred, str(q2_pred)),
            "g": g_val,
            "H_bar": H_bar_val,
            "m": m_val,
            "u": u_val,
            "correct": int(pred == label),
        }
        for i in range(args.num_classes):
            row[class_cols[i]] = round(float(p[i]), 3)
        rows.append(row)

    os.makedirs(run_dir, exist_ok=True)
    out_csv = os.path.join(run_dir, "test_per_slide_scores.csv")
    fieldnames = [
        "slide_id",
        "gt_class",
        "pred_class",
        "pred_q1",
        "pred_q2",
        "g",
        "H_bar",
        "m",
        "u",
        "correct",
    ] + class_cols
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[Info] Saved per-slide test scores CSV: {out_csv}")

    avg_loss = total_loss / max(n_samples, 1)
    avg_g = g_sum / max(n_samples, 1)
    probs_arr = np.vstack(probs_list) if len(probs_list) else np.zeros((0, args.num_classes))

    # AUC
    try:
        if len(np.unique(np.array(y_true))) < 2:
            auc = float("nan")
        elif args.num_classes == 2:
            auc = roc_auc_score(np.array(y_true), probs_arr[:, 1])
        else:
            auc = roc_auc_score(np.array(y_true), probs_arr, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    acc = accuracy_score(y_true, y_pred)
    print(
        f"\n[Test] avg_loss={avg_loss:.4f} acc={acc:.4f} auc={auc:.4f}"
    )
    print("\nClassification report:")
    print(classification_report(y_true, y_pred))
    rep = classification_report(y_true, y_pred)

    return {
        "loss": float(avg_loss),
        "acc": float(acc),
        "auc": float(auc),
        "g": float(avg_g),
        "report": ["Classification report:"] + rep.splitlines(),
    }


def main():
    args = parse_args()
    assert args.batch_size == 1

    set_seed(args.seed, deterministic=args.deterministic)

    graph_root = args.graph_root

    cross_dirname = f"cross_{args.low_mag}_{args.high_mag}"

    paired_ds_train, loader_train = build_paired_loader(
        graph_root=graph_root,
        split=args.train_split,
        low_mag=args.low_mag,
        high_mag=args.high_mag,
        cross_dirname=cross_dirname,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    paired_ds_val, loader_val = build_paired_loader(
        graph_root=graph_root,
        split=args.val_split,
        low_mag=args.low_mag,
        high_mag=args.high_mag,
        cross_dirname=cross_dirname,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    paired_ds_test, loader_test = build_paired_loader(
        graph_root=graph_root,
        split=args.test_split,
        low_mag=args.low_mag,
        high_mag=args.high_mag,
        cross_dirname=cross_dirname,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    from collections import Counter

    def get_label_value(x):
        if torch.is_tensor(x):
            if x.numel() == 1:
                return int(x.item())
            return tuple(x.detach().cpu().tolist())
        return int(x)

    cnt = Counter()
    n = 0

    for batch in loader_train:
        low = batch["low"]
        y = low.get("y", low.get("label"))
        yv = get_label_value(y)
        cnt[yv] += 1
        n += 1

    print(f"[train] num slides seen: {n}")
    print("[train] label counts:", cnt)

    label_to_class = {v: k for k, v in paired_ds_train.ds_low.class_to_label.items()}
    print("[train] label->class:", label_to_class)

    for lab, c in sorted(cnt.items()):
        name = label_to_class.get(lab, "UNKNOWN")
        print(f"  label {lab:>2} ({name}): {c}")


    batch = next(iter(loader_train))
    sample_graph_low  = _prepare_graph_for_model(batch["low"])
    sample_graph_high = _prepare_graph_for_model(batch["high"])

    feat_dim_low  = sample_graph_low["x"].shape[-1]
    feat_dim_high = sample_graph_high["x"].shape[-1]
    print("feat_dim_low:", feat_dim_low, "feat_dim_high:", feat_dim_high)

    model = build_model(args, feat_dim_low, feat_dim_high)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    bn = [n for n, m in model.named_modules() if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d))]
    print("BN layers:", bn)

    base_results = "results"
    run_dir, exp, ts = make_run_dir(base_results, args.exp_name)

    csv_path, epoch_writer, epoch_f = init_epoch_logger(run_dir, args.exp_name, args)

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    last_ckpt_path = os.path.join(ckpt_dir, "last.pt")
    best_loss_ckpt_path = os.path.join(ckpt_dir, "best.pt")
    best_auc_ckpt_path = os.path.join(ckpt_dir, "best_auc.pt")

    # resume
    start_epoch = 1
    best_val_loss = float("inf")
    best_epoch_loss = -1
    best_val_auc = -1.0
    best_epoch_auc = -1
    if args.resume is not None:
        print(f"[Resume] Loading: {args.resume}")
        start_epoch, best_val_loss, best_epoch_loss = load_checkpoint(
            args.resume, model, optimizer=optimizer, map_location=DEVICE
        )
        print(
            f"[Resume] start_epoch={start_epoch} "
            f"best_val_loss={best_val_loss:.6f} best_epoch_loss={best_epoch_loss}"
        )

    # early stop: val loss; default test ckpt: val macro AUC (best_auc.pt)
    best_es_loss = float("inf")
    bad = 0
    patience = args.patience
    min_delta = 0.0

    for epoch in range(start_epoch, args.epochs + 1):
        tr_avg, tr_auc = run_epoch(
            model, loader_train,
            optimizer=optimizer, desc=f"[Train Epoch {epoch}]", args=args, train=True,
        )

        va_avg, va_auc = run_epoch(
            model, loader_val,
            optimizer=None, desc=f"[Val   Epoch {epoch}]", args=args, train=False,
        )

        print(
            f"[Epoch {epoch}] "
            f"Train: loss={tr_avg['loss_total']:.3f} acc={tr_avg['acc']:.3f} auc={tr_auc:.3f} | "
            f"Val: loss={va_avg['loss_total']:.3f} acc={va_avg['acc']:.3f} auc={va_auc:.3f}"
        )

        def r3(x):
            try:
                return round(float(x), 3)
            except Exception:
                return x

        epoch_writer.writerow({
            "epoch": epoch,
            "tr_loss_total": r3(tr_avg["loss_total"]),
            "tr_loss_ce_hat": r3(tr_avg["loss_ce_hat"]),
            "tr_loss_ce_q2": r3(tr_avg["loss_ce_q2"]),
            "tr_loss_H": r3(tr_avg["loss_H"]),
            "tr_loss_align": r3(tr_avg["loss_align"]),
            "tr_acc": r3(tr_avg["acc"]),
            "tr_auc": r3(tr_auc),
            "va_loss_total": r3(va_avg["loss_total"]),
            "va_loss_ce_hat": r3(va_avg["loss_ce_hat"]),
            "va_loss_ce_q2": r3(va_avg["loss_ce_q2"]),
            "va_loss_H": r3(va_avg["loss_H"]),
            "va_loss_align": r3(va_avg["loss_align"]),
            "va_acc": r3(va_avg["acc"]),
            "va_auc": r3(va_auc),
            "lr": r3(optimizer.param_groups[0]["lr"]),
            "num_train": len(loader_train),
            "num_val": len(loader_val),
        })
        epoch_f.flush()

        # save last.pt (always or every N)
        cur_acc = float(va_avg["acc"])
        cur_auc = float(va_auc)
        if args.save_every > 0 and (epoch % args.save_every == 0):
            save_checkpoint(
                last_ckpt_path, model, optimizer, epoch,
                best_val_loss, best_epoch_loss, args,
                val_acc=cur_acc, val_auc=cur_auc,
            )
            print(f"   → Saved last checkpoint: {last_ckpt_path}")

        if va_avg["loss_total"] < best_val_loss:
            best_val_loss = float(va_avg["loss_total"])
            best_epoch_loss = epoch
            save_checkpoint(
                best_loss_ckpt_path, model, optimizer, epoch,
                best_val_loss, best_epoch_loss, args,
                val_acc=cur_acc, val_auc=cur_auc,
            )
            print(f"   → New best val_loss checkpoint (epoch={epoch}, loss={best_val_loss:.4f})")

        if not np.isnan(cur_auc) and cur_auc > best_val_auc:
            best_val_auc = cur_auc
            best_epoch_auc = epoch
            save_checkpoint(
                best_auc_ckpt_path, model, optimizer, epoch,
                best_val_loss, best_epoch_auc, args,
                val_acc=cur_acc, val_auc=cur_auc,
            )
            print(
                f"   → New best val_auc checkpoint "
                f"(epoch={epoch}, auc={best_val_auc:.4f})"
            )

        cur_loss = float(va_avg["loss_total"])
        if cur_loss < best_es_loss - min_delta:
            best_es_loss = cur_loss
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            print(f"[EarlyStop] stop at epoch {epoch} (best val_loss={best_es_loss:.4f})")
            break

    epoch_f.close()
    print(f"[Info] Closed epoch log file: {csv_path}")

    test_ckpt_map = {
        "best_auc": best_auc_ckpt_path,
        "best_loss": best_loss_ckpt_path,
        "last": last_ckpt_path,
    }
    test_ckpt_path = test_ckpt_map[args.test_ckpt]
    if not os.path.isfile(test_ckpt_path):
        fallback = best_loss_ckpt_path if os.path.isfile(best_loss_ckpt_path) else last_ckpt_path
        print(f"[Warn] {test_ckpt_path} not found; fallback to {fallback}")
        test_ckpt_path = fallback

    print(f"\n[Info] Loading checkpoint for test: {test_ckpt_path} (criterion={args.test_ckpt})")
    load_checkpoint(test_ckpt_path, model, optimizer=None, map_location=DEVICE)

    test_metrics = evaluate_diagmil(
        model,
        loader_test,
        args,
        run_dir=run_dir,
        class_to_label=loader_train.dataset.ds_low.class_to_label,
    )

    test_metrics_clean = {}
    for k, v in test_metrics.items():
        if isinstance(v, (int, float, np.floating)):
            test_metrics_clean[k] = round(float(v), 3)
        else:
            test_metrics_clean[k] = v

    summary = {
        "exp_name": args.exp_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": vars(args),
        "feat_dim_low": int(feat_dim_low),
        "feat_dim_high": int(feat_dim_high),
        "best_val_loss": float(best_val_loss),
        "best_epoch_loss": int(best_epoch_loss),
        "best_val_auc": float(best_val_auc),
        "best_epoch_auc": int(best_epoch_auc),
        "test_ckpt": str(args.test_ckpt),
        "test_ckpt_path": test_ckpt_path,
        "test_metrics": test_metrics_clean,
    }
    save_summary_json(summary, run_dir, args.exp_name)

    log_model_info(model, loader_test, run_dir, args)


def log_model_info(model, test_loader, run_dir, args):
    """Save model size, inference time, GPU memory to model_info.txt"""
    import time as _time
    import glob as _glob

    model.eval()
    lines = []
    lines.append("=" * 50)
    lines.append("DiagMIL Model Info")
    lines.append("=" * 50)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines.append(f"\n[Model Size]")
    lines.append(f"  Total params:      {n_params:,}")
    lines.append(f"  Trainable params:  {n_trainable:,}")
    lines.append(f"  Params (MB):       {n_params * 4 / 1024 / 1024:.4f}  (float32)")

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    ckpts = _glob.glob(os.path.join(ckpt_dir, "*.pt"))
    if ckpts:
        ckpt_path = max(ckpts, key=os.path.getmtime)
        fsize_mb = os.path.getsize(ckpt_path) / 1024 / 1024
        lines.append(f"  Saved ckpt (MB):   {fsize_mb:.4f}")

    if torch.cuda.is_available() and test_loader is not None:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        n_measure = min(50, len(test_loader))

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= 3:
                    break
                graph_low = _prepare_graph_for_model(batch["low"])
                graph_high = _prepare_graph_for_model(batch["high"])
                batch_cross = batch["cross"]
                cross_index = batch_cross["cross_patch_index"]
                graph_low = move_graph_to_device(graph_low, DEVICE)
                graph_high = move_graph_to_device(graph_high, DEVICE)
                cross_index = cross_index.to(DEVICE)
                _ = model(graph_low, graph_high, cross_index)
        torch.cuda.synchronize()

        t0 = _time.perf_counter()
        count = 0
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if count >= n_measure:
                    break
                graph_low = _prepare_graph_for_model(batch["low"])
                graph_high = _prepare_graph_for_model(batch["high"])
                batch_cross = batch["cross"]
                cross_index = batch_cross["cross_patch_index"]
                graph_low = move_graph_to_device(graph_low, DEVICE)
                graph_high = move_graph_to_device(graph_high, DEVICE)
                cross_index = cross_index.to(DEVICE)
                _ = model(graph_low, graph_high, cross_index)
                count += 1
        torch.cuda.synchronize()
        elapsed = _time.perf_counter() - t0
        gpu_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

        lines.append(f"\n[Inference (GPU)]")
        lines.append(f"  Samples measured:  {count}")
        lines.append(f"  Total time (s):    {elapsed:.4f}")
        lines.append(f"  Time per slide (ms): {elapsed / max(1, count) * 1000:.2f}")
        lines.append(f"  GPU memory (MB):   {gpu_mb:.2f}")
    else:
        lines.append(f"\n[Inference] CPU mode - GPU stats skipped")

    lines.append("\n" + "=" * 50)
    out_path = os.path.join(run_dir, "model_info.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Model info saved to {out_path}")


if __name__ == "__main__":
    main()
