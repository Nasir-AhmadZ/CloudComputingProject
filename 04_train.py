"""
04_train.py
===========
Defines the LSTM classifier and trains it with early stopping, learning-rate
scheduling, gradient clipping, and checkpointing.

Model architecture
──────────────────
  Input  : (batch, seq_len=60, features=1)  normalised log returns
  Layer 1: Stacked LSTM  (num_layers, hidden_size, dropout between layers)
  Layer 2: LayerNorm on last hidden state
  Layer 3: Dropout
  Layer 4: Linear → 3 logits (Hold / Buy / Sell)

Training features
─────────────────
  • Weighted CrossEntropyLoss  (class weights from training set)
  • WeightedRandomSampler      (oversample minority classes in batches)
  • AdamW optimiser with cosine-annealing LR schedule
  • Gradient clipping (max norm 1.0)
  • Early stopping on validation F1-macro (patience configurable)
  • Saves best checkpoint to checkpoints/best_model.pt
  • Logs per-epoch metrics to checkpoints/training_log.csv

Run
───
  python 04_train.py

Adjust the CONFIG dict below to tune hyperparameters.
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import our dataset builder
import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset import build_loaders   # noqa: E402  (local import)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {
    # Data
    "seq_len":      60,
    "batch_size":   512,
    "num_workers":  0,  # must be 0 on Windows
    "max_files":    None,   # set e.g. 10 to use only 10 tickers (debugging)

    # Model
    "input_size":   1,
    "hidden_size":  128,
    "num_layers":   2,
    "dropout":      0.3,
    "num_classes":  3,

    # Training
    "epochs":       50,
    "lr":           3e-4,
    "weight_decay": 1e-4,
    "grad_clip":    1.0,
    "patience":     7,       # early stopping patience (epochs)

    # Output
    "ckpt_dir":     "checkpoints",
}

LABEL_NAMES = ["Hold", "Buy", "Sell"]


# ── Model ─────────────────────────────────────────────────────────────────────

class TradingLSTM(nn.Module):
    """
    Stacked bidirectional-optional LSTM for 3-class trade signal classification.

    Parameters
    ----------
    input_size  : number of features per time step (1 for norm_ret only)
    hidden_size : LSTM hidden units
    num_layers  : number of stacked LSTM layers
    dropout     : dropout probability between LSTM layers (and before head)
    num_classes : output classes (3: Hold/Buy/Sell)
    """

    def __init__(
        self,
        input_size:  int = 1,
        hidden_size: int = 128,
        num_layers:  int = 2,
        dropout:     float = 0.3,
        num_classes: int = 3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size   = input_size,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = dropout if num_layers > 1 else 0.0,
            bidirectional = False,   # causal: no future data
        )
        self.norm    = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(hidden_size, num_classes)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 (helps with long sequences)
                n = param.shape[0]
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.0)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, input_size)
        returns logits : (batch, num_classes)
        """
        out, _ = self.lstm(x)          # (batch, seq_len, hidden_size)
        last    = out[:, -1, :]        # take the last time step
        last    = self.norm(last)
        last    = self.dropout(last)
        logits  = self.head(last)      # (batch, num_classes)
        return logits


# ── Training utilities ────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    log.info(f"Using device: {dev}")
    return dev


def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Returns (loss, f1_macro, all_preds, all_labels).
    """
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)
            total_loss += loss.item() * len(y)
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    avg_loss   = total_loss / len(all_labels)
    f1         = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, f1, all_preds, all_labels


def train_one_epoch(
    model:     nn.Module,
    loader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device:    torch.device,
    grad_clip: float,
) -> tuple[float, float]:
    """Returns (avg_loss, f1_macro) for the epoch."""
    model.train()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimiser.zero_grad(set_to_none=True)
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimiser.step()

        total_loss += loss.item() * len(y)
        preds = logits.detach().argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(y.cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    avg_loss   = total_loss / len(all_labels)
    f1         = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, f1


# ── Main training loop ────────────────────────────────────────────────────────

def main(cfg: dict = CONFIG):
    ckpt_dir = Path(cfg["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    # ── Data ──────────────────────────────────────────────────────────────────
    log.info("Building data loaders …")
    train_loader, val_loader, test_loader, class_weights = build_loaders(
        seq_len     = cfg["seq_len"],
        batch_size  = cfg["batch_size"],
        num_workers = cfg["num_workers"],
        use_sampler = True,
        max_files   = cfg["max_files"],
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TradingLSTM(
        input_size  = cfg["input_size"],
        hidden_size = cfg["hidden_size"],
        num_layers  = cfg["num_layers"],
        dropout     = cfg["dropout"],
        num_classes = cfg["num_classes"],
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters: {num_params:,}")

    # ── Loss, optimiser, scheduler ────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(
        weight = class_weights.to(device)
    )
    optimiser = AdamW(
        model.parameters(),
        lr           = cfg["lr"],
        weight_decay = cfg["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimiser,
        T_max  = cfg["epochs"],
        eta_min = cfg["lr"] * 0.01,
    )

    # ── Training log ──────────────────────────────────────────────────────────
    log_path = ckpt_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "train_f1", "val_loss", "val_f1", "lr", "elapsed_s"]
        )

    best_val_f1  = -1.0
    patience_cnt = 0
    best_epoch   = 0

    log.info("Starting training …\n")
    t0 = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        t_ep = time.time()

        tr_loss, tr_f1 = train_one_epoch(
            model, train_loader, criterion, optimiser, device, cfg["grad_clip"]
        )
        va_loss, va_f1, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t_ep
        lr_now  = optimiser.param_groups[0]["lr"]

        log.info(
            f"Epoch {epoch:3d}/{cfg['epochs']} | "
            f"train loss {tr_loss:.4f} f1 {tr_f1:.4f} | "
            f"val loss {va_loss:.4f} f1 {va_f1:.4f} | "
            f"lr {lr_now:.2e} | {elapsed:.1f}s"
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{tr_loss:.6f}", f"{tr_f1:.6f}",
                 f"{va_loss:.6f}", f"{va_f1:.6f}",
                 f"{lr_now:.6e}", f"{elapsed:.1f}"]
            )

        # Early stopping & checkpointing
        if va_f1 > best_val_f1:
            best_val_f1  = va_f1
            best_epoch   = epoch
            patience_cnt = 0
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "val_f1":      va_f1,
                    "config":      cfg,
                },
                ckpt_dir / "best_model.pt",
            )
            log.info(f"  ✓ New best val F1: {va_f1:.4f} — checkpoint saved.")
        else:
            patience_cnt += 1
            if patience_cnt >= cfg["patience"]:
                log.info(
                    f"Early stopping at epoch {epoch} "
                    f"(best val F1 {best_val_f1:.4f} at epoch {best_epoch})"
                )
                break

    total_time = time.time() - t0
    log.info(f"\nTraining complete in {total_time/60:.1f} min.")
    log.info(f"Best checkpoint: epoch {best_epoch}, val F1 = {best_val_f1:.4f}")
    log.info("Run 05_evaluate.py to assess test-set performance.")


if __name__ == "__main__":
    main()
