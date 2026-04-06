import argparse
import csv
import os
import random
from pathlib import Path
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from dataset_overlapping_stage3 import (
    Stage3OverlappingMultiTaskDataset,
    get_stage3_overlapping_multitask_transforms,
)

DATA_ROOT = None
OUT_DIR = None

IMG_SIZE = None
BATCH_SIZE = None
EPOCHS = None
WARMUP_EPOCHS = None
LR = None
WEIGHT_DECAY = None
PATIENCE = None
PRETRAINED_WEIGHT_PATH = None
MIN_DELTA = None
NUM_WORKERS = None
SEED = None

LAMBDA_FG = None
LAMBDA_OVERLAP = None
LAMBDA_BOUNDARY = None
LAMBDA_CONSISTENCY = None

FG_THRESHOLD = None
OVERLAP_THRESHOLD = None
BOUNDARY_THRESHOLD = None
BOUNDARY_POS_WEIGHT = None

RESUME_TRAINING = False
RESUME_CHECKPOINT = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def update_globals_from_config(config_path):
    global DATA_ROOT, OUT_DIR, IMG_SIZE, BATCH_SIZE, EPOCHS, WARMUP_EPOCHS
    global LR, WEIGHT_DECAY, PATIENCE, PRETRAINED_WEIGHT_PATH, MIN_DELTA, NUM_WORKERS, SEED
    global LAMBDA_FG, LAMBDA_OVERLAP, LAMBDA_BOUNDARY, LAMBDA_CONSISTENCY
    global FG_THRESHOLD, OVERLAP_THRESHOLD, BOUNDARY_THRESHOLD, BOUNDARY_POS_WEIGHT
    global RESUME_CHECKPOINT

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    DATA_ROOT = cfg['training']['data_root']
    OUT_DIR = Path(cfg['training']['save_dir'])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_SIZE = cfg['model']['input_size']
    BATCH_SIZE = cfg['training']['batch_size']
    EPOCHS = cfg['training']['epochs']
    WARMUP_EPOCHS = cfg['training'].get('warmup_epochs', 5)
    LR = float(cfg['training']['lr'])
    WEIGHT_DECAY = float(cfg['training'].get('weight_decay', 1e-4))
    PATIENCE = int(cfg['training'].get('patience', 12))
    PRETRAINED_WEIGHT_PATH = cfg['model'].get('pretrained_weights', None)
    MIN_DELTA = float(cfg['training'].get('min_delta', 1e-4))
    NUM_WORKERS = int(cfg['training'].get('num_workers', 0))
    SEED = int(cfg['training'].get('seed', 42))

    LAMBDA_FG = float(cfg['loss'].get('lambda_fg', 1.0))
    LAMBDA_OVERLAP = float(cfg['loss'].get('lambda_overlap', 1.2))
    LAMBDA_BOUNDARY = float(cfg['loss'].get('lambda_boundary', 1.0))
    LAMBDA_CONSISTENCY = float(cfg['loss'].get('lambda_consistency', 0.2))

    FG_THRESHOLD = float(cfg['loss'].get('fg_threshold', 0.5))
    OVERLAP_THRESHOLD = float(cfg['loss'].get('overlap_threshold', 0.45))
    BOUNDARY_THRESHOLD = float(cfg['loss'].get('boundary_threshold', 0.35))
    BOUNDARY_POS_WEIGHT = float(cfg['loss'].get('boundary_pos_weight', 3.0))
    RESUME_CHECKPOINT = OUT_DIR / "stage3_overlapping_multitask_last_checkpoint.pth"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        improved = score > self.best_score + self.min_delta if self.mode == "max" else score < self.best_score - self.min_delta
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).reshape(-1)
        targets = targets.reshape(-1)
        inter = (probs * targets).sum()
        den = probs.sum() + targets.sum()
        dice = (2.0 * inter + self.eps) / (den + self.eps)
        return 1.0 - dice


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.4, beta=0.6, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).reshape(-1)
        targets = targets.reshape(-1)

        tp = (probs * targets).sum()
        fp = (probs * (1.0 - targets)).sum()
        fn = ((1.0 - probs) * targets).sum()

        score = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return 1.0 - score


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)


class BCETverskyLoss(nn.Module):
    def __init__(self, bce_weight=0.3, tversky_weight=0.7, alpha=0.4, beta=0.6, pos_weight=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)
        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight

    def forward(self, logits, targets):
        return self.bce_weight * self.bce(logits, targets) + self.tversky_weight * self.tversky(logits, targets)


class MultiHeadOverlapBoundaryNet(nn.Module):
    def __init__(self, pretrained_weights=None):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name="tu-poolformer_m36",
            encoder_weights="imagenet" if pretrained_weights is None else None,
            in_channels=3,
            classes=3,
            activation=None,
        )
        if pretrained_weights is not None:
            print(f"Loading custom poolformer weights from {pretrained_weights}")
            state_dict = torch.load(pretrained_weights, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            self.unet.encoder.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        logits = self.unet(x)
        foreground_logits = logits[:, 0:1, :, :]
        overlap_logits = logits[:, 1:2, :, :]
        boundary_logits = logits[:, 2:3, :, :]
        return foreground_logits, overlap_logits, boundary_logits


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    best_score,
    best_epoch,
    early_stopper,
):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_score": best_score,
        "best_epoch": best_epoch,
        "early_stopping": {
            "best_score": early_stopper.best_score,
            "counter": early_stopper.counter,
            "should_stop": early_stopper.should_stop,
            "patience": early_stopper.patience,
            "min_delta": early_stopper.min_delta,
            "mode": early_stopper.mode,
        },
    }, path)


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    early_stopper=None,
    device="cpu",
):
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    if early_stopper is not None and "early_stopping" in checkpoint:
        es = checkpoint["early_stopping"]
        early_stopper.best_score = es.get("best_score", None)
        early_stopper.counter = es.get("counter", 0)
        early_stopper.should_stop = es.get("should_stop", False)
        early_stopper.patience = es.get("patience", early_stopper.patience)
        early_stopper.min_delta = es.get("min_delta", early_stopper.min_delta)
        early_stopper.mode = es.get("mode", early_stopper.mode)

    start_epoch = checkpoint.get("epoch", 0)
    best_score = checkpoint.get("best_score", -1.0)
    best_epoch = checkpoint.get("best_epoch", -1)

    return start_epoch, best_score, best_epoch


def compute_confusion_stats(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).long().reshape(-1)
    targs = targets.long().reshape(-1)

    tp = int(((preds == 1) & (targs == 1)).sum().item())
    tn = int(((preds == 0) & (targs == 0)).sum().item())
    fp = int(((preds == 1) & (targs == 0)).sum().item())
    fn = int(((preds == 0) & (targs == 1)).sum().item())

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def stats_to_metrics(stats, eps=1e-7):
    tp = stats["tp"]
    fp = stats["fp"]
    fn = stats["fn"]

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2.0 * precision * recall) / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "dice": float(dice),
        "tn": int(stats["tn"]),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def consistency_loss(foreground_logits, overlap_logits, boundary_logits):
    fg_probs = torch.sigmoid(foreground_logits)
    ov_probs = torch.sigmoid(overlap_logits)
    bd_probs = torch.sigmoid(boundary_logits)

    overlap_inside_fg = torch.relu(ov_probs - fg_probs).mean()
    boundary_inside_fg = torch.relu(bd_probs - fg_probs).mean()
    return overlap_inside_fg + boundary_inside_fg


def train_one_epoch(model, loader, criterion_fg, criterion_overlap, criterion_boundary, optimizer, scaler):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc="Training", leave=False)

    for images, foreground_t, overlap_t, boundary_t in pbar:
        images = images.to(DEVICE, non_blocking=True)
        foreground_t = foreground_t.to(DEVICE, non_blocking=True).unsqueeze(1)
        overlap_t = overlap_t.to(DEVICE, non_blocking=True).unsqueeze(1)
        boundary_t = boundary_t.to(DEVICE, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            foreground_logits, overlap_logits, boundary_logits = model(images)
            loss_fg = criterion_fg(foreground_logits, foreground_t)
            loss_overlap = criterion_overlap(overlap_logits, overlap_t)
            loss_boundary = criterion_boundary(boundary_logits, boundary_t)
            loss_consistency = consistency_loss(foreground_logits, overlap_logits, boundary_logits)
            loss = (
                LAMBDA_FG * loss_fg
                + LAMBDA_OVERLAP * loss_overlap
                + LAMBDA_BOUNDARY * loss_boundary
                + LAMBDA_CONSISTENCY * loss_consistency
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            fg=f"{loss_fg.item():.4f}",
            ov=f"{loss_overlap.item():.4f}",
            bd=f"{loss_boundary.item():.4f}",
        )

    return running_loss / max(len(loader), 1)


@torch.no_grad()
def validate_one_epoch(model, loader, criterion_fg, criterion_overlap, criterion_boundary):
    model.eval()
    running_loss = 0.0

    total_fg = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    total_ov = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    total_bd = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    pbar = tqdm(loader, desc="Validation", leave=False)
    for images, foreground_t, overlap_t, boundary_t in pbar:
        images = images.to(DEVICE, non_blocking=True)
        foreground_t = foreground_t.to(DEVICE, non_blocking=True).unsqueeze(1)
        overlap_t = overlap_t.to(DEVICE, non_blocking=True).unsqueeze(1)
        boundary_t = boundary_t.to(DEVICE, non_blocking=True).unsqueeze(1)

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            foreground_logits, overlap_logits, boundary_logits = model(images)
            loss_fg = criterion_fg(foreground_logits, foreground_t)
            loss_overlap = criterion_overlap(overlap_logits, overlap_t)
            loss_boundary = criterion_boundary(boundary_logits, boundary_t)
            loss_consistency = consistency_loss(foreground_logits, overlap_logits, boundary_logits)
            loss = (
                LAMBDA_FG * loss_fg
                + LAMBDA_OVERLAP * loss_overlap
                + LAMBDA_BOUNDARY * loss_boundary
                + LAMBDA_CONSISTENCY * loss_consistency
            )

        running_loss += loss.item()

        fg_stats = compute_confusion_stats(foreground_logits, foreground_t, FG_THRESHOLD)
        ov_stats = compute_confusion_stats(overlap_logits, overlap_t, OVERLAP_THRESHOLD)
        bd_stats = compute_confusion_stats(boundary_logits, boundary_t, BOUNDARY_THRESHOLD)

        for key in ["tp", "tn", "fp", "fn"]:
            total_fg[key] += fg_stats[key]
            total_ov[key] += ov_stats[key]
            total_bd[key] += bd_stats[key]

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / max(len(loader), 1)
    fg_metrics = stats_to_metrics(total_fg)
    ov_metrics = stats_to_metrics(total_ov)
    bd_metrics = stats_to_metrics(total_bd)
    return avg_loss, fg_metrics, ov_metrics, bd_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Train Stage 3 Overlapping')
    parser.add_argument('--config', default='configs/stage3_sep.yaml', help='path to config file')
    return parser.parse_args()


def main():
    set_seed(SEED)

    print("=" * 60)
    print("STAGE 3 OVERLAPPING MULTI-TASK TRAINING - PoolFormer-M36 U-Net")
    print("=" * 60)
    print(f"Device        : {DEVICE}")
    print(f"AMP Enabled   : {USE_AMP}")
    print(f"Image Size    : {IMG_SIZE}")
    print(f"Batch Size    : {BATCH_SIZE}")
    print(f"Epochs        : {EPOCHS}")
    print(f"Learning Rate : {LR}")
    print(f"Data Root     : {DATA_ROOT}")
    print(f"Output Dir    : {OUT_DIR.resolve()}")
    print(f"Resume        : {RESUME_TRAINING}")
    print("=" * 60)

    train_tf, val_tf = get_stage3_overlapping_multitask_transforms(IMG_SIZE)
    train_dataset = Stage3OverlappingMultiTaskDataset(DATA_ROOT, split="train", transform=train_tf)
    val_dataset = Stage3OverlappingMultiTaskDataset(DATA_ROOT, split="val", transform=val_tf)

    print(f"Train samples : {len(train_dataset)}")
    print(f"Val samples   : {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = MultiHeadOverlapBoundaryNet(pretrained_weights=PRETRAINED_WEIGHT_PATH).to(DEVICE)

    boundary_pos_weight = torch.tensor([BOUNDARY_POS_WEIGHT], device=DEVICE)
    criterion_fg = BCEDiceLoss(0.5, 0.5)
    criterion_overlap = BCETverskyLoss(bce_weight=0.3, tversky_weight=0.7, alpha=0.35, beta=0.65)
    criterion_boundary = BCEDiceLoss(0.6, 0.4, pos_weight=boundary_pos_weight)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS])
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    early_stopper = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA, mode="max")

    log_path = OUT_DIR / "train_log_stage3_overlapping_multitask.csv"
    last_ckpt_path = OUT_DIR / "stage3_overlapping_multitask_last_checkpoint.pth"
    best_ckpt_path = OUT_DIR / "stage3_overlapping_multitask_best_checkpoint.pth"
    last_weights_path = OUT_DIR / "stage3_overlapping_multitask_last.pth"
    best_weights_path = OUT_DIR / "stage3_overlapping_multitask_best.pth"

    best_score = -1.0
    best_epoch = -1
    start_epoch = 0

    if RESUME_TRAINING and RESUME_CHECKPOINT.exists():
        start_epoch, best_score, best_epoch = load_checkpoint(
            RESUME_CHECKPOINT,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            early_stopper=early_stopper,
            device=DEVICE,
        )
        print(f"✅ Resumed from checkpoint: {RESUME_CHECKPOINT}")
        print(f"Start epoch    : {start_epoch}")
        print(f"Best epoch     : {best_epoch}")
        print(f"Best score     : {best_score:.6f}")
    else:
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "lr", "train_loss", "val_loss", "score",
                "fg_iou", "fg_dice", "fg_precision", "fg_recall", "fg_f1",
                "ov_iou", "ov_dice", "ov_precision", "ov_recall", "ov_f1",
                "bd_iou", "bd_dice", "bd_precision", "bd_recall", "bd_f1"
            ])

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{EPOCHS} =====")
        train_loss = train_one_epoch(model, train_loader, criterion_fg, criterion_overlap, criterion_boundary, optimizer, scaler)
        val_loss, fg_metrics, ov_metrics, bd_metrics = validate_one_epoch(
            model, val_loader, criterion_fg, criterion_overlap, criterion_boundary
        )
        current_lr = optimizer.param_groups[0]["lr"]

        combined_score = 0.2 * fg_metrics["iou"] + 0.5 * ov_metrics["iou"] + 0.3 * bd_metrics["iou"]
        scheduler.step()

        print(
            f"LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"FG IoU: {fg_metrics['iou']:.4f} | OV IoU: {ov_metrics['iou']:.4f} | BD IoU: {bd_metrics['iou']:.4f}"
        )

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, current_lr, train_loss, val_loss, combined_score,
                fg_metrics["iou"], fg_metrics["dice"], fg_metrics["precision"], fg_metrics["recall"], fg_metrics["f1"],
                ov_metrics["iou"], ov_metrics["dice"], ov_metrics["precision"], ov_metrics["recall"], ov_metrics["f1"],
                bd_metrics["iou"], bd_metrics["dice"], bd_metrics["precision"], bd_metrics["recall"], bd_metrics["f1"],
            ])

        save_checkpoint(
            path=last_ckpt_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_score=best_score,
            best_epoch=best_epoch,
            early_stopper=early_stopper,
        )
        torch.save(model.state_dict(), last_weights_path)

        if combined_score > best_score + MIN_DELTA:
            best_score = combined_score
            best_epoch = epoch

            save_checkpoint(
                path=best_ckpt_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_score=best_score,
                best_epoch=best_epoch,
                early_stopper=early_stopper,
            )
            torch.save(model.state_dict(), best_weights_path)

            with open(OUT_DIR / "best_metrics_stage3_overlapping_multitask.txt", "w", encoding="utf-8") as f:
                f.write(f"best_epoch={best_epoch}\n")
                f.write(f"best_score={best_score:.6f}\n")
                f.write(f"train_loss={train_loss:.6f}\n")
                f.write(f"val_loss={val_loss:.6f}\n")
                for prefix, met in [("fg", fg_metrics), ("ov", ov_metrics), ("bd", bd_metrics)]:
                    for k, v in met.items():
                        f.write(f"{prefix}_{k}={v}\n")

            print(f"✅ Saved best model/checkpoint at epoch {epoch}")

        if early_stopper.step(combined_score):
            print(f"⏹ Early stopping triggered at epoch {epoch}")
            break

    print("\n" + "=" * 60)
    print("🏁 Stage 3 overlapping multitask training finished")
    print(f"Best epoch     : {best_epoch}")
    print(f"Best score     : {best_score:.6f}")
    print(f"Saved in       : {OUT_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()

    # Load YAML Config similarly to train_cluster.py
    project_root = os.getcwd()
    config_path = os.path.join(project_root, args.config)

    update_globals_from_config(config_path)

    main()
