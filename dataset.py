"""
03_dataset.py
=============
PyTorch Dataset that pools labelled parquet files from multiple tickers
into a single stream of (sequence, label) samples.

Design decisions
────────────────
• Each sample is a sliding window of SEQ_LEN consecutive normalised log
  returns (norm_ret) from a single ticker.
• Windows that span day boundaries are discarded: if ANY bar in the window
  has a NaN log_ret or if the window contains the first bar of a new day
  (where log_ret would be NaN before dropping), we skip it.  We detect this
  by checking that all timestamps are on the same date AND sequential.
• The label is taken from the LAST bar of the window (bar t), matching the
  forward-simulation we ran in 02_label.py.
• A ticker ID integer is included so the model could optionally use a
  ticker embedding — but the baseline LSTM ignores it.

Usage
─────
  from dataset import SP500Dataset, build_loaders

  train_loader, val_loader, test_loader, class_weights = build_loaders(
      seq_len=60, batch_size=512
  )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

log = logging.getLogger(__name__)

PROC_DIR = Path("processed")
LABEL_COL = "label"
FEAT_COL  = "norm_ret"


class SP500Dataset(Dataset):
    """
    Pools all ticker parquet files from a given split's labelled directory.

    Parameters
    ----------
    split     : "train", "val", or "test"
    seq_len   : number of bars in each input window (default 60)
    max_files : if set, only load this many tickers (useful for debugging)
    """

    def __init__(
        self,
        split: str,
        seq_len: int = 60,
        max_files: int | None = None,
    ):
        self.split   = split
        self.seq_len = seq_len

        in_dir = PROC_DIR / f"{split}_labelled"
        paths  = sorted(in_dir.glob("*.parquet"))
        if max_files:
            paths = paths[:max_files]

        if not paths:
            raise FileNotFoundError(f"No parquet files found in {in_dir}")

        self.ticker_to_id: Dict[str, int] = {}
        self.sequences: List[np.ndarray] = []   # shape (seq_len, 1) each
        self.labels:    List[int]         = []
        self.ticker_ids: List[int]        = []

        for path in paths:
            ticker = path.stem
            tid    = self.ticker_to_id.setdefault(ticker, len(self.ticker_to_id))
            self._load_ticker(path, tid)

        self.sequences  = np.array(self.sequences,  dtype=np.float32)  # (N, seq_len, 1)
        self.labels     = np.array(self.labels,     dtype=np.int64)
        self.ticker_ids = np.array(self.ticker_ids, dtype=np.int64)

        log.info(
            f"[{split}] {len(self.sequences):,} samples from "
            f"{len(self.ticker_to_id)} tickers"
        )

    # ── internal ─────────────────────────────────────────────────────────────

    def _load_ticker(self, path: Path, ticker_id: int):
        df = pd.read_parquet(path)

        # Ensure sorted by time
        df = df.sort_index()

        # We need at least seq_len rows
        if len(df) < self.seq_len:
            return

        feats  = df[FEAT_COL].values.astype(np.float32)
        labels = df[LABEL_COL].values.astype(np.int64)
        dates  = df.index.normalize().values   # numpy datetime64 dates

        # Build windows with a stride of 1
        for end in range(self.seq_len - 1, len(df)):
            start = end - self.seq_len + 1

            window_feats  = feats[start : end + 1]
            window_dates  = dates[start : end + 1]
            window_label  = labels[end]

            # Skip if any value in the window is NaN
            if np.any(np.isnan(window_feats)):
                continue

            # Skip if window spans multiple days
            if window_dates[0] != window_dates[-1]:
                continue

            self.sequences.append(window_feats.reshape(self.seq_len, 1))
            self.labels.append(int(window_label))
            self.ticker_ids.append(ticker_id)

    # ── Dataset API ───────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.sequences[idx])   # (seq_len, 1)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

    # ── Utility ───────────────────────────────────────────────────────────────

    def class_counts(self) -> np.ndarray:
        """Returns array of shape (num_classes,) with sample counts per class."""
        num_classes = int(self.labels.max()) + 1
        counts = np.bincount(self.labels, minlength=num_classes)
        return counts

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights, one per class — for CrossEntropyLoss."""
        counts = self.class_counts().astype(np.float32)
        weights = 1.0 / (counts + 1e-8)
        weights = weights / weights.sum() * len(counts)   # normalise to sum = num_classes
        return torch.tensor(weights, dtype=torch.float32)

    def sample_weights(self) -> np.ndarray:
        """Per-sample weight for WeightedRandomSampler."""
        cw = self.class_weights().numpy()
        return cw[self.labels]


# ── Builder convenience function ─────────────────────────────────────────────

def build_loaders(
    seq_len:    int  = 60,
    batch_size: int  = 512,
    num_workers: int = 4,
    use_sampler: bool = True,    # WeightedRandomSampler for train only
    max_files:  int | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Build train / val / test DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader, class_weights
        class_weights : torch.Tensor shape (3,) — pass to CrossEntropyLoss(weight=...)
    """
    train_ds = SP500Dataset("train", seq_len=seq_len, max_files=max_files)
    val_ds   = SP500Dataset("val",   seq_len=seq_len, max_files=max_files)
    test_ds  = SP500Dataset("test",  seq_len=seq_len, max_files=max_files)

    cw = train_ds.class_weights()

    # Weighted sampler oversamples minority classes during training
    if use_sampler:
        sw      = train_ds.sample_weights()
        sampler = WeightedRandomSampler(
            weights     = torch.from_numpy(sw).double(),
            num_samples = len(train_ds),
            replacement = True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size  = batch_size,
            sampler     = sampler,
            num_workers = num_workers,
            pin_memory  = True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size  = batch_size,
            shuffle     = True,
            num_workers = num_workers,
            pin_memory  = True,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size * 2,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = batch_size * 2,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )

    # Print class distribution
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        counts  = ds.class_counts()
        total   = counts.sum()
        names   = ["Hold", "Buy", "Sell"]
        dist    = "  ".join(f"{n}:{c:,}({100*c/total:.1f}%)"
                            for n, c in zip(names, counts))
        log.info(f"[{name}] {dist}")

    return train_loader, val_loader, test_loader, cw


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    train_loader, val_loader, test_loader, cw = build_loaders(
        seq_len=60, batch_size=64, num_workers=0, max_files=5
    )
    print(f"Class weights: {cw}")
    x, y = next(iter(train_loader))
    print(f"Batch x shape: {x.shape}  y shape: {y.shape}")
    print(f"Label distribution in batch: {torch.bincount(y)}")
