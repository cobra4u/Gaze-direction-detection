# ~/gazemain/main.py
import os
import math
import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from loader import (
    create_xgaze_h5_dataloaders,
    create_xgaze_h5_dataloaders_with_val,
)
from models.multitask_model import MultiTaskGazeNet, angular_cosine_loss, angular_error_deg


def parse_args():
    p = argparse.ArgumentParser("Gaze training")
    # Data
    p.add_argument("--data.root_dir", type=str, required=True, help="Path to xgaze_224")
    p.add_argument("--val.subjects_file", type=str, default="", help="Path to val_subjects.txt (optional)")
    p.add_argument("--train.batch_size", type=int, default=128)
    p.add_argument("--train.num_workers", type=int, default=8)
    # Limits for quick sanity/dry runs
    p.add_argument("--max.train.steps", type=int, default=0, help="Max training steps per epoch (0 = no limit)")
    p.add_argument("--max.val.batches", type=int, default=0, help="Max eval batches for val (0 = no limit)")
    p.add_argument("--max.test.batches", type=int, default=0, help="Max eval batches for test (0 = no limit)")
    # Model/optim
    p.add_argument("--model.backbone", type=str, default="resnet50")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--outdir", type=str, default="checkpoints")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", help="Enable mixed precision")
    p.add_argument("--lambda.head", type=float, default=0.3, help="Weight for head-pose auxiliary loss")

    # Optional (parked for future demographic extensions; not used in this baseline run)
    p.add_argument("--tasks.use_demographics", action="store_true", help="Enable demographic heads; requires labels in batch")
    p.add_argument("--tasks.use_film", action="store_true", help="Condition gaze features on demographics (FiLM)")
    p.add_argument("--lambda.age", type=float, default=0.2)
    p.add_argument("--lambda.gender", type=float, default=0.2)
    p.add_argument("--lambda.eth", type=float, default=0.2)

    return p.parse_args()


def set_seed(seed: int):
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def save_checkpoint(state: dict, outpath: str):
    Path(os.path.dirname(outpath)).mkdir(parents=True, exist_ok=True)
    torch.save(state, outpath)


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    use_amp,
    max_steps: int = 0,
    lambda_head: float = 0.3,
):
    model.train()
    running_loss = 0.0
    running_ang = 0.0
    n_seen = 0

    for step, batch in enumerate(tqdm(loader, desc="Train", leave=False), start=1):
        face = batch["face"].to(device, non_blocking=True)
        gaze = batch["gaze"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(face)
            loss = angular_cosine_loss(out["gaze"], gaze)

            if "head_pose" in batch:
                hp = batch["head_pose"].to(device, non_blocking=True)
                loss = loss + lambda_head * angular_cosine_loss(out["head_pose"], hp)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            ang = angular_error_deg(out["gaze"], gaze).mean().item()
            bs = face.size(0)
            running_loss += loss.item() * bs
            running_ang += ang * bs
            n_seen += bs

        if max_steps > 0 and step >= max_steps:
            break

    return running_loss / max(n_seen, 1), running_ang / max(n_seen, 1)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, title: str = "Eval", max_batches: int = 0):
    if loader is None:
        return float("nan")
    model.eval()
    running_ang = 0.0
    n = 0
    for bi, batch in enumerate(tqdm(loader, desc=title, leave=False), start=1):
        face = batch["face"].to(device, non_blocking=True)
        gaze = batch["gaze"].to(device, non_blocking=True)
        pred = model(face)["gaze"]
        ang = angular_error_deg(pred, gaze).mean().item()
        bs = face.size(0)
        running_ang += ang * bs
        n += bs
        if max_batches > 0 and bi >= max_batches:
            break
    return running_ang / max(n, 1)


@torch.no_grad()
def evaluate_groupwise(model: torch.nn.Module, loader: DataLoader, device: torch.device, group_key: str = "gender", max_batches: int = 0):
    if loader is None:
        return {}
    model.eval()
    sums, counts = {}, {}
    for bi, batch in enumerate(tqdm(loader, desc=f"Eval by {group_key}", leave=False), start=1):
        if group_key not in batch:
            break  # no labels available in this loader
        face = batch["face"].to(device, non_blocking=True)
        gaze = batch["gaze"].to(device, non_blocking=True)
        pred = model(face)["gaze"]
        ang = angular_error_deg(pred, gaze).cpu().numpy().tolist()
        groups = batch[group_key].cpu().numpy().tolist()
        for g, a in zip(groups, ang):
            sums[g] = sums.get(g, 0.0) + a
            counts[g] = counts.get(g, 0) + 1
        if max_batches > 0 and bi >= max_batches:
            break
    return {k: (sums[k] / max(1, counts[k])) for k in sums}


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Data
    val_file = args.__dict__.get("val.subjects_file", "")
    if val_file:
        loaders = create_xgaze_h5_dataloaders_with_val(
            root_dir=args.__dict__["data.root_dir"],
            val_subjects_file=val_file,
            batch_size_train=args.__dict__["train.batch_size"],
            batch_size_eval=args.__dict__["train.batch_size"] * 2,
            num_workers=args.__dict__["train.num_workers"],
            distributed=False,
        )
        val_loader = loaders["val"]
    else:
        loaders = create_xgaze_h5_dataloaders(
            root_dir=args.__dict__["data.root_dir"],
            batch_size_train=args.__dict__["train.batch_size"],
            batch_size_eval=args.__dict__["train.batch_size"] * 2,
            num_workers=args.__dict__["train.num_workers"],
            distributed=False,
        )
        val_loader = None

    train_loader = loaders["train"]
    test_loader = loaders.get("test", None)  # may be None if test lacks gaze

    # Model (baseline multi-task gaze + headpose)
    model = MultiTaskGazeNet(
        backbone_name=args.__dict__["model.backbone"],
        pretrained=True
    ).to(device)

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # AMP scaler
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Limits
    max_train_steps = int(args.__dict__.get("max.train.steps", 0))
    max_val_batches = int(args.__dict__.get("max.val.batches", 0))
    max_test_batches = int(args.__dict__.get("max.test.batches", 0))

    # Train
    best_metric = math.inf  # val if available, else test
    outdir = os.path.join(os.getcwd(), args.outdir)
    Path(outdir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        tr_loss, tr_ang = train_one_epoch(
            model, train_loader, optimizer, scaler, device, use_amp,
            max_steps=max_train_steps,
            lambda_head=args.__dict__.get("lambda.head", 0.3),
        )

        va_ang = evaluate(model, val_loader, device, title="Val", max_batches=max_val_batches) if val_loader is not None else math.inf
        te_ang = evaluate(model, test_loader, device, title="Test", max_batches=max_test_batches) if test_loader is not None else float("nan")
        scheduler.step()

        test_str = f" | Test: {te_ang:.2f} deg" if test_loader is not None else " | Test: n/a"
        if val_loader is not None:
            crit = va_ang
            print(f"Train: loss={tr_loss:.4f}, MAE={tr_ang:.2f} deg | Val: {va_ang:.2f} deg{test_str}")
        else:
            crit = te_ang
            print(f"Train: loss={tr_loss:.4f}, MAE={tr_ang:.2f} deg{test_str}")

        # Save checkpoint
        ckpt_path = os.path.join(outdir, f"gaze_epoch{epoch:03d}.pth")
        save_checkpoint(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_mae": None if val_loader is None else va_ang,
                "test_mae": None if test_loader is None else te_ang,
            },
            ckpt_path,
        )

        if crit < best_metric:
            best_metric = crit
            best_path = os.path.join(outdir, "gaze_best.pth")
            save_checkpoint({"model": model.state_dict(), "metric": best_metric}, best_path)
            print(f"New best model (criterion={'Val' if val_loader is not None else 'Test'}): {best_metric:.2f} deg (saved to {best_path})")

    print("Done.")


if __name__ == "__main__":
    main()
