"""
03_dataset.py
=============
PyTorch Dataset that serves fixed-length rolling windows of normalised
log returns alongside their labels.

Key design decisions
--------------------
* Windows never span overnight gaps — a window is only valid if all bars
  belong to the same contiguous intra-day block.  We detect day boundaries
  by checking whether consecutive timestamps are more than 1 minute apart
  (with a small tolerance for data irregularities).
* Ticker identity is encoded as an integer index, which the LSTM model can
  optionally embed.
* The dataset is built per-split; call TradingDataset("train") etc.

Usage (standalone test):
    python 03_dataset.py
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle
import random


# ── Configuration ─────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("processed")
SEQ_LEN       = 60    # input window length (minutes)
MAX_GAP_MINUTES = 2   # max gap before treating as day boundary
NUM_TICKERS_SAMPLE = 2  # Use only 2 tickers for faster training


class TradingDataset(Dataset):
    """
    Parameters
    ----------
    split : "train" | "val" | "test"
    seq_len : int
    feature_col : str
    num_tickers : int or None
        If specified, randomly sample this many tickers
    """

    def __init__(self, split: str, seq_len: int = SEQ_LEN,
                 feature_col: str = "norm_ret", num_tickers: int = None):
        self.seq_len     = seq_len
        self.feature_col = feature_col

        path = PROCESSED_DIR / split / "data_labelled.parquet"
        cache_suffix = f"_{num_tickers}tickers" if num_tickers else ""
        cache_path = PROCESSED_DIR / split / f"windows_cache_{seq_len}{cache_suffix}.pkl"
        
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run 01_data_pipeline.py and "
                "02_label_generator.py first."
            )

        # Try to load from cache
        if cache_path.exists():
            print(f"Loading cached windows from {cache_path}...")
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            self.windows = cache["windows"]
            self.ticker2idx = cache["ticker2idx"]
            self.num_tickers = cache["num_tickers"]
        else:
            # Build windows from scratch
            df_tickers = pd.read_parquet(path, columns=["ticker"])
            tickers = sorted(df_tickers["ticker"].unique())
            del df_tickers
            
            # Sample tickers if requested
            if num_tickers is not None:
                random.seed(42)
                tickers = random.sample(tickers, min(num_tickers, len(tickers)))
            
            self.ticker2idx  = {t: i for i, t in enumerate(tickers)}
            self.num_tickers = len(tickers)
            self.windows = []
            
            print(f"Building windows for {len(tickers)} tickers: {tickers}...")
            for i, ticker in enumerate(tickers, 1):
                df = pd.read_parquet(path, filters=[("ticker", "==", ticker)])
                df.sort_index(inplace=True)
                self._build_windows_for_ticker(df, ticker)
                del df
                if i % 10 == 0:
                    print(f"  Processed {i}/{len(tickers)} tickers, {len(self.windows):,} windows")
            
            # Save cache
            print(f"Saving cache to {cache_path}...")
            with open(cache_path, "wb") as f:
                pickle.dump({
                    "windows": self.windows,
                    "ticker2idx": self.ticker2idx,
                    "num_tickers": self.num_tickers
                }, f)

        print(
            f"[{split}] {len(self.windows):,} windows | "
            f"{self.num_tickers} tickers | seq_len={seq_len}"
        )

    def _build_windows_for_ticker(self, df: pd.DataFrame, ticker: str) -> None:
        t_idx = self.ticker2idx[ticker]
        
        features = df[self.feature_col].values.astype(np.float32)
        labels   = df["label"].values.astype(np.int64)
        times    = df.index

        # Pre-compute gaps once
        gaps = np.diff(times.view('int64')) / (60 * 1e9)  # Convert to minutes
        gaps = np.concatenate([[0], gaps])  # Prepend 0 for first element
        
        # Find day boundaries (gaps > MAX_GAP_MINUTES)
        is_boundary = gaps > MAX_GAP_MINUTES
        
        n = len(df)
        for end in range(self.seq_len, n):
            start = end - self.seq_len
            # Check if any boundary exists in the window
            if is_boundary[start + 1: end + 1].any():
                continue

            x = features[start:end]
            y = labels[end]
            self.windows.append((x, t_idx, y))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        x, t_idx, y = self.windows[idx]
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
        t_tensor = torch.tensor(t_idx, dtype=torch.long)
        y_tensor = torch.tensor(y, dtype=torch.long)
        return x_tensor, t_tensor, y_tensor


def compute_class_weights(dataset: TradingDataset) -> torch.Tensor:
    """
    Inverse-frequency class weights for CrossEntropyLoss.
    Only call this on the training set.
    """
    labels      = np.array([w[2] for w in dataset.windows])
    num_classes = 3
    counts      = np.bincount(labels, minlength=num_classes).astype(float)
    weights     = counts.sum() / (num_classes * np.where(counts == 0, 1, counts))
    return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        try:
            ds = TradingDataset(split)
            dl = DataLoader(ds, batch_size=256, shuffle=(split == "train"))
            xb, tb, yb = next(iter(dl))
            print(f"  x: {xb.shape}, t: {tb.shape}, y: {yb.shape}")
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
