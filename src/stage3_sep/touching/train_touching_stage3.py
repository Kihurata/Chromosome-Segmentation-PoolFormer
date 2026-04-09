import argparse
import csv
import os
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from dataset_touching_stage3 import Stage3TouchingDataset, get_stage3_transforms

# Global variables (will be updated from config)
DATA_ROOT = ""
OUT_DIR = Path("experiments/stage3_sep")
IMG_SIZE = 384
BATCH_SIZE = 8
EPOCHS = 100
WARMUP_EPOCHS = 5
LR = 1.0e-4
WEIGHT_DECAY = 1.0e-4
PATIENCE = 15
PRETRAINED_WEIGHT_PATH = None
MIN_DELTA = 1.0e-4
NUM_WORKERS = 1
SEED = 42
RESUME_TRAINING = False

LAMBDA_FG = 1.0
LAMBDA_MARKER = 1.5
FG_THRESHOLD = 0.5
MARKER_THRESHOLD = 0.4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

def update_globals_from_config(config_path):
    global DATA_ROOT, OUT_DIR, IMG_SIZE, BATCH_SIZE, EPOCHS, WARMUP_EPOCHS
    global LR, WEIGHT_DECAY, PATIENCE, PRETRAINED_WEIGHT_PATH, MIN_DELTA, NUM_WORKERS, SEED
    global LAMBDA_FG, LAMBDA_MARKER, FG_THRESHOLD, MARKER_THRESHOLD, RESUME_TRAINING

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    DATA_ROOT = cfg['training']['data_root']
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    base_save_dir = Path(cfg['training']['save_dir'])
    OUT_DIR = base_save_dir / f"{timestamp}_poolformer_touching"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    shutil.copy(config_path, OUT_DIR / "config.yaml")

    IMG_SIZE = cfg['model']['input_size']
    BATCH_SIZE = cfg['training']['batch_size']
    EPOCHS = cfg['training']['epochs']
    WARMUP_EPOCHS = cfg['training'].get('warmup_epochs', 5)
    LR = float(cfg['training']['lr'])
    WEIGHT_DECAY = float(cfg['training'].get('weight_decay', 1e-4))
    PATIENCE = int(cfg['training'].get('patience', 15))
    PRETRAINED_WEIGHT_PATH = cfg['model'].get('pretrained_weights', None)
    MIN_DELTA = float(cfg['training'].get('min_delta', 1e-4))
    NUM_WORKERS = int(cfg['training'].get('num_workers', 1))
    SEED = int(cfg['training'].get('seed', 42))
    RESUME_TRAINING = cfg['training'].get('resume', False)

    LAMBDA_FG = float(cfg['loss'].get('lambda_fg', 1.0))
    LAMBDA_MARKER = float(cfg['loss'].get('lambda_marker', 1.5))
    FG_THRESHOLD = float(cfg['loss'].get('fg_threshold', 0.5))
    MARKER_THRESHOLD = float(cfg['loss'].get('marker_threshold', 0.4))

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class EarlyStopping:
    def __init__(self, patience=12, min_delta=0.0, mode="max"):
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
        probs = torch.sigmoid(logits)
        num = (probs * targets).sum(dim=(2, 3))
        den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2.0 * num + self.eps) / (den + self.eps)
        return 1.0 - dice.mean()

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probas = torch.sigmoid(logits)
        p_t = probas * targets + (1 - probas) * (1 - targets)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce_loss
        return loss.mean()

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)

class PoolFormerTouchingNet(nn.Module):
    def __init__(self, pretrained_weights=None):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name="tu-poolformer_m36",
            encoder_weights="imagenet" if pretrained_weights is None else None,
            in_channels=3,
            classes=2,
            activation=None,
        )
        if pretrained_weights is not None:
            print(f"Loading custom poolformer weights from {pretrained_weights}")
            state_dict = torch.load(pretrained_weights, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            # Filter encoder weights
            encoder_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")}
            self.unet.encoder.load_state_dict(encoder_dict, strict=False)

    def forward(self, x):
        logits = self.unet(x)
        fg_logits = logits[:, 0:1, :, :]
        mk_logits = logits[:, 1:2, :, :]
        return fg_logits, mk_logits

def compute_metrics(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    
    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()
    
    iou = tp / max(tp + fp + fn, 1)
    dice = (2 * tp) / max(2 * tp + fp + fn, 1)
    return {"iou": iou, "dice": dice, "tp": tp, "fp": fp, "fn": fn}

def train_one_epoch(model, loader, crit_fg, crit_mk, optimizer, scaler):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc="Training", leave=False)

    for images, foregrounds, markers in pbar:
        images = images.to(DEVICE, non_blocking=True)
        foregrounds = foregrounds.to(DEVICE, non_blocking=True).unsqueeze(1)
        markers = markers.to(DEVICE, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            fg_logits, mk_logits = model(images)
            loss_fg = crit_fg(fg_logits, foregrounds)
            loss_mk = crit_mk(mk_logits, markers)
            loss = LAMBDA_FG * loss_fg + LAMBDA_MARKER * loss_mk

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(len(loader), 1)

@torch.no_grad()
def validate_one_epoch(model, loader, crit_fg, crit_mk):
    model.eval()
    running_loss = 0.0
    total_fg = {"iou": 0, "dice": 0, "tp": 0, "fp": 0, "fn": 0}
    total_mk = {"iou": 0, "dice": 0, "tp": 0, "fp": 0, "fn": 0}
    
    n_samples = 0
    for images, foregrounds, markers in tqdm(loader, desc="Validation", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        foregrounds = foregrounds.to(DEVICE, non_blocking=True).unsqueeze(1)
        markers = markers.to(DEVICE, non_blocking=True).unsqueeze(1)

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            fg_logits, mk_logits = model(images)
            loss_fg = crit_fg(fg_logits, foregrounds)
            loss_mk = crit_mk(mk_logits, markers)
            loss = LAMBDA_FG * loss_fg + LAMBDA_MARKER * loss_mk

        running_loss += loss.item()
        
        m_fg = compute_metrics(fg_logits, foregrounds, FG_THRESHOLD)
        m_mk = compute_metrics(mk_logits, markers, MARKER_THRESHOLD)
        
        for k in total_fg: total_fg[k] += m_fg[k]
        for k in total_mk: total_mk[k] += m_mk[k]
        n_samples += 1

    avg_loss = running_loss / max(len(loader), 1)
    metrics = {
        "fg_iou": total_fg["iou"] / n_samples,
        "fg_dice": total_fg["dice"] / n_samples,
        "mk_iou": total_mk["iou"] / n_samples,
        "mk_dice": total_mk["dice"] / n_samples,
    }
    return avg_loss, metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Train Stage 3 Touching PoolFormer')
    parser.add_argument('--config', default='configs/stage3_touching.yaml', help='path to config file')
    return parser.parse_args()

def main():
    args = parse_args()
    update_globals_from_config(args.config)
    set_seed(SEED)

    print("=" * 60)
    print("STAGE 3 TOUCHING TRAINING - PoolFormer-M36 Multi-Mask")
    print("=" * 60)
    print(f"Device        : {DEVICE}")
    print(f"Output Dir    : {OUT_DIR.resolve()}")
    print("-" * 60)

    train_tf, val_tf = get_stage3_transforms(IMG_SIZE)
    train_dataset = Stage3TouchingDataset(DATA_ROOT, split="train", transform=train_tf)
    val_dataset = Stage3TouchingDataset(DATA_ROOT, split="val", transform=val_tf)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = PoolFormerTouchingNet(pretrained_weights=PRETRAINED_WEIGHT_PATH).to(DEVICE)
    
    crit_fg = BCEDiceLoss(0.5, 0.5)
    crit_mk = BinaryFocalLoss(alpha=0.5, gamma=2.0)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS])
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    early_stopper = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA, mode="max")

    log_path = OUT_DIR / "train_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "train_loss", "val_loss", "fg_iou", "mk_iou", "mk_dice"])

    best_iou = -1.0
    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{EPOCHS} =====")
        train_loss = train_one_epoch(model, train_loader, crit_fg, crit_mk, optimizer, scaler)
        val_loss, metrics = validate_one_epoch(model, val_loader, crit_fg, crit_mk)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        print(f"LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"FG IoU: {metrics['fg_iou']:.4f} | MK IoU: {metrics['mk_iou']:.4f} | MK Dice: {metrics['mk_dice']:.4f}")

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, current_lr, train_loss, val_loss, metrics['fg_iou'], metrics['mk_iou'], metrics['mk_dice']])

        torch.save(model.state_dict(), OUT_DIR / "last.pth")
        if metrics['mk_iou'] > best_iou:
            best_iou = metrics['mk_iou']
            torch.save(model.state_dict(), OUT_DIR / "best.pth")
            print(f"⭐ New best MK IoU: {best_iou:.4f}")

        if early_stopper.step(metrics['mk_iou']):
            print("Early stopping triggered")
            break

if __name__ == "__main__":
    main()
