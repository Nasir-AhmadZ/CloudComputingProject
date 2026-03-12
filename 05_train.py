"""
05_train.py
===========
Training loop for the TradingLSTM classifier.

Features
--------
* Class-weighted CrossEntropyLoss to combat label imbalance
* ReduceLROnPlateau scheduler (monitors val macro-F1)
* Early stopping on val macro-F1 with configurable patience
* Best-model checkpointing (saves the epoch with highest val F1)
* Per-epoch metrics printed to stdout + saved to CSV

Usage:
    python 05_train.py

Outputs:
    checkpoints/best_model.pt      — best checkpoint
    checkpoints/last_model.pt      — final epoch checkpoint
    logs/training_log.csv          — epoch-by-epoch metrics
"""

import os
import time
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

from dataset   import TradingDataset, compute_class_weights
from model     import TradingLSTM
from torch.utils.data import Subset
from collections import Counter

# Rename imports to match file names without the numeric prefix
# (add the directory to sys.path so we can import cleanly)
import sys
sys.path.insert(0, str(Path(__file__).parent))


# ── Configuration ─────────────────────────────────────────────────────────────
CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR        = Path("logs")

# Data
NUM_TICKERS = 2  # Limit to N tickers for faster training (set to None for all)

# Model
HIDDEN_SIZE = 64       # Reduced from 128 for faster training
NUM_LAYERS  = 1        # Reduced from 2 for faster training
DROPOUT     = 0.2      # Reduced dropout
EMBED_DIM   = 8        # Reduced from 16

# Training
BATCH_SIZE    = 2048       # Increased for faster throughput
MAX_EPOCHS    = 20
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
PATIENCE      = 7          # early-stopping patience (epochs)
MIN_LR        = 1e-6
LR_FACTOR     = 0.5
LR_PATIENCE   = 3

SEQ_LEN       = 60
NUM_WORKERS   = 0          # Set to 0 on Windows to avoid hanging
SUBSAMPLE_FRAC = 0.1       # Use 10% of data per epoch (10x speedup)
VAL_SUBSAMPLE_FRAC = 0.2   # Use 20% of validation data for faster validation


LABEL_NAMES = {0: "Buy", 1: "Hold", 2: "Sell"}


# ── Utilities ─────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    print(f"Using device: {dev}")
    return dev


def run_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  nn.Module,
    optimizer:  torch.optim.Optimizer | None,
    device:     torch.device,
    train:      bool = True,
    epoch:      int = 0,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    One forward (+ optional backward) pass over the dataset.
    Returns (mean_loss, macro_f1, all_preds, all_targets).
    """
    model.train() if train else model.eval()

    total_loss = 0.0
    all_preds  = []
    all_targets= []
    n_batches  = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    phase = "Train" if train else "Val"
    
    with ctx:
        pbar = tqdm(loader, desc=f"Epoch {epoch} [{phase}]", leave=False)
        for x_batch, t_batch, y_batch in pbar:
            x_batch = x_batch.to(device)
            t_batch = t_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch, t_batch)
            loss   = criterion(logits, y_batch)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y_batch.cpu().numpy())
            n_batches += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    mean_loss   = total_loss / max(n_batches, 1)
    macro_f1    = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return mean_loss, macro_f1, all_preds, all_targets


# ── Main training routine ─────────────────────────────────────────────────────

def train():
    # Optimize PyTorch for CPU performance
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    device = get_device()

    # ── Datasets & loaders ───────────────────────────────────────────────────
    print("Loading datasets...")
    train_ds = TradingDataset("train", seq_len=SEQ_LEN, num_tickers=NUM_TICKERS)
    val_ds   = TradingDataset("val",   seq_len=SEQ_LEN, num_tickers=NUM_TICKERS)

    class_weights = compute_class_weights(train_ds).to(device)
    print(f"Class weights: Buy={class_weights[0]:.3f}, "
          f"Hold={class_weights[1]:.3f}, Sell={class_weights[2]:.3f}")

    # Stratified subsampling to maintain class balance
    labels = np.array([train_ds.windows[i][2] for i in range(len(train_ds))])
    train_indices = []
    
    for class_label in [0, 1, 2]:
        class_indices = np.where(labels == class_label)[0]
        n_samples = int(len(class_indices) * SUBSAMPLE_FRAC)
        sampled = np.random.choice(class_indices, n_samples, replace=False)
        train_indices.extend(sampled)
    
    train_indices = np.array(train_indices)
    np.random.shuffle(train_indices)
    train_subset = Subset(train_ds, train_indices)
    
    subset_labels = labels[train_indices]
    subset_dist = Counter(subset_labels)
    print(f"Using {len(train_subset):,} / {len(train_ds):,} training samples ({SUBSAMPLE_FRAC:.0%})")
    print(f"Stratified distribution: Buy={subset_dist[0]:,}, Hold={subset_dist[1]:,}, Sell={subset_dist[2]:,}")
    
    # Subsample validation data for faster validation
    val_labels = np.array([val_ds.windows[i][2] for i in range(len(val_ds))])
    val_indices = []
    for class_label in [0, 1, 2]:
        class_indices = np.where(val_labels == class_label)[0]
        n_samples = int(len(class_indices) * VAL_SUBSAMPLE_FRAC)
        sampled = np.random.choice(class_indices, n_samples, replace=False)
        val_indices.extend(sampled)
    val_indices = np.array(val_indices)
    np.random.shuffle(val_indices)
    val_subset = Subset(val_ds, val_indices)
    print(f"Using {len(val_subset):,} / {len(val_ds):,} validation samples ({VAL_SUBSAMPLE_FRAC:.0%})")
    
    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = TradingLSTM(
        num_tickers = train_ds.num_tickers,
        hidden_size = HIDDEN_SIZE,
        num_layers  = NUM_LAYERS,
        dropout     = DROPOUT,
        embed_dim   = EMBED_DIM,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_FACTOR,
        patience=LR_PATIENCE, min_lr=MIN_LR,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_f1    = -np.inf
    epochs_no_impr = 0
    log_rows       = []

    print(f"\nStarting training (max {MAX_EPOCHS} epochs, "
          f"early-stop patience={PATIENCE})...\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()

        tr_loss, tr_f1, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True, epoch=epoch
        )
        va_loss, va_f1, va_preds, va_targets = run_epoch(
            model, val_loader, criterion, None, device, train=False, epoch=epoch
        )

        scheduler.step(va_f1)
        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        improved = va_f1 > best_val_f1
        flag     = " ✓" if improved else ""

        print(
            f"Epoch {epoch:>3}/{MAX_EPOCHS} | "
            f"tr_loss={tr_loss:.4f}  tr_f1={tr_f1:.4f} | "
            f"va_loss={va_loss:.4f}  va_f1={va_f1:.4f}{flag} | "
            f"lr={lr_now:.2e} | {elapsed:.1f}s"
        )

        log_rows.append({
            "epoch": epoch, "tr_loss": tr_loss, "tr_f1": tr_f1,
            "va_loss": va_loss, "va_f1": va_f1, "lr": lr_now,
        })

        if improved:
            best_val_f1     = va_f1
            epochs_no_impr  = 0
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "opt_state":   optimizer.state_dict(),
                    "val_f1":      va_f1,
                    "ticker2idx":  train_ds.ticker2idx,
                    "config": {
                        "num_tickers": train_ds.num_tickers,
                        "hidden_size": HIDDEN_SIZE,
                        "num_layers":  NUM_LAYERS,
                        "dropout":     DROPOUT,
                        "embed_dim":   EMBED_DIM,
                        "seq_len":     SEQ_LEN,
                    },
                },
                CHECKPOINT_DIR / "best_model.pt",
            )
            # Print a full classification report at the best epoch
            print("\n  Best epoch so far — classification report (val):")
            print(
                classification_report(
                    va_targets, va_preds,
                    target_names=["Buy", "Hold", "Sell"],
                    zero_division=0,
                )
            )
        else:
            epochs_no_impr += 1
            if epochs_no_impr >= PATIENCE:
                print(f"\nEarly stopping — no val_f1 improvement for "
                      f"{PATIENCE} consecutive epochs.")
                break

    # Save last checkpoint
    torch.save(model.state_dict(), CHECKPOINT_DIR / "last_model.pt")

    # Save training log
    log_path = LOG_DIR / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\nTraining log saved → {log_path}")
    print(f"Best checkpoint  saved → {CHECKPOINT_DIR / 'best_model.pt'}")
    print(f"Best val macro-F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    train()
