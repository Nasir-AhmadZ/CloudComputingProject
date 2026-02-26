"""
01_preprocess.py
================
Loads all 1-minute OHLCV CSV files from rawSP500data/, computes log returns
from Close prices, fits a normaliser on the training period ONLY, applies it
to val/test, and saves three directories of per-ticker Parquet files.

Directory layout produced
-------------------------
processed/
  train/   AAPL.parquet, MSFT.parquet, ...
  val/     AAPL.parquet, MSFT.parquet, ...
  test/    AAPL.parquet, MSFT.parquet, ...
  normaliser.json   (train mean & std of log-returns, per ticker)

Columns in each parquet
-----------------------
  datetime  : index (tz-naive, market hours only)
  log_ret   : raw log return
  norm_ret  : normalised log return  (log_ret - mean) / std

Split boundaries (inclusive)
-----------------------------
  train : 2020-12-28 -> 2022-12-31
  val   : 2023-01-01 -> 2023-12-31
  test  : 2024-01-01 -> 2025-12-23

Adjust TRAIN_END / VAL_END below if you prefer different boundaries.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Configuration ────────────────────────────────────────────────────────────
RAW_DIR   = Path("rawSP500data")
OUT_DIR   = Path("processed")
TRAIN_DIR = OUT_DIR / "train"
VAL_DIR   = OUT_DIR / "val"
TEST_DIR  = OUT_DIR / "test"

# Split boundaries (end of period, inclusive date)
TRAIN_END = pd.Timestamp("2022-12-31")
VAL_END   = pd.Timestamp("2023-12-31")
# test is everything after VAL_END

# US market hours - stored as time objects for fast masking
MARKET_OPEN  = pd.Timestamp("09:30").time()
MARKET_CLOSE = pd.Timestamp("16:00").time()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_ticker(path: Path) -> Optional[pd.DataFrame]:
    """
    Read CSV, parse datetime index, keep Close only, filter to market hours.

    Handles CSVs where:
      - The date column may be named 'Date', 'Datetime', 'date', etc.
      - There may be extra columns like 'Ticker', 'Open', 'High', 'Low', 'Volume'
        (all ignored).
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        log.warning(f"Could not read {path.name}: {e}")
        return None

    # Find the date column
    date_col = None
    for candidate in ("Date", "Datetime", "date", "datetime", "timestamp", "Time"):
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        log.warning(f"{path.name}: cannot find a date column - skipping.")
        return None

    # Find the close column
    close_col = None
    for candidate in ("Close", "close", "CLOSE"):
        if candidate in df.columns:
            close_col = candidate
            break
    if close_col is None:
        log.warning(f"{path.name}: cannot find a Close column - skipping.")
        return None

    # Parse and set datetime index
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df.index.name = "datetime"

    # Remove timezone if present (keep everything tz-naive)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Keep only Close column
    df = df[[close_col]].rename(columns={close_col: "Close"}).copy()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    # Sort chronologically
    df = df.sort_index()

    # Filter to market hours
    times = df.index.time
    mask  = (times >= MARKET_OPEN) & (times <= MARKET_CLOSE)
    df    = df.loc[mask].copy()

    # Drop zero or missing prices
    df = df[df["Close"] > 0].dropna()

    if len(df) < 100:
        log.warning(
            f"{path.name}: only {len(df)} valid rows after filtering - skipping."
        )
        return None

    return df


def compute_log_returns(close: pd.Series) -> pd.Series:
    """
    Log return for bar t = ln(Close_t / Close_{t-1}).

    The first bar of each trading day is set to NaN to avoid including the
    overnight gap as a return.

    Uses pd.Series.shift (position-based) rather than DatetimeIndex.shift
    (frequency-based) to avoid NullFrequencyError on irregular timestamps.
    """
    lr = np.log(close / close.shift(1))

    # Wrap dates in a plain Series so .shift() is position-based (no freq needed)
    dates_s    = pd.Series(close.index.normalize(), index=close.index)
    day_starts = dates_s != dates_s.shift(1)   # True at first bar of each day
    lr[day_starts] = np.nan

    return lr


def split_and_normalise(
    df: pd.DataFrame,
) -> Tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[dict],
]:
    """
    Split into train / val / test and normalise log_ret.

    Normalisation statistics (mean, std) are fitted on TRAINING data only
    and applied to val/test - no leakage.
    """
    train = df[df.index <= TRAIN_END].copy()
    val   = df[(df.index > TRAIN_END) & (df.index <= VAL_END)].copy()
    test  = df[df.index > VAL_END].copy()

    # Fit on training log returns only
    train_rets = train["log_ret"].dropna()
    if len(train_rets) < 2:
        return None, None, None, None

    mu  = float(train_rets.mean())
    std = float(train_rets.std(ddof=1))
    if std == 0.0:
        return None, None, None, None

    stats = {"mean": mu, "std": std}

    for part in (train, val, test):
        part["norm_ret"] = (part["log_ret"] - mu) / std

    return train, val, test, stats


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    for d in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        d.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(RAW_DIR.glob("*_1min.csv"))
    if not csv_files:
        log.error(f"No CSV files found in {RAW_DIR}/ - check that RAW_DIR is correct.")
        return

    log.info(f"Found {len(csv_files)} ticker files.")

    normaliser: dict = {}
    skipped:    list = []

    for path in tqdm(csv_files, desc="Processing tickers"):
        ticker = path.stem.replace("_1min", "")

        df = load_ticker(path)
        if df is None:
            skipped.append(ticker)
            continue

        df["log_ret"] = compute_log_returns(df["Close"])
        df = df.drop(columns=["Close"])   # only log returns needed downstream

        train, val, test, stats = split_and_normalise(df)
        if train is None:
            log.warning(f"{ticker}: not enough training data - skipping.")
            skipped.append(ticker)
            continue

        # Save each split, dropping NaN rows (day-boundary bars)
        train.dropna(subset=["log_ret"]).to_parquet(TRAIN_DIR / f"{ticker}.parquet")
        val.dropna(subset=["log_ret"]).to_parquet(VAL_DIR     / f"{ticker}.parquet")
        test.dropna(subset=["log_ret"]).to_parquet(TEST_DIR   / f"{ticker}.parquet")

        normaliser[ticker] = stats

    # Persist normaliser so downstream scripts can inspect it
    with open(OUT_DIR / "normaliser.json", "w") as f:
        json.dump(normaliser, f, indent=2)

    log.info(f"Done. Processed {len(normaliser)} tickers, skipped {len(skipped)}.")
    if skipped:
        log.info(f"Skipped tickers: {skipped}")

    # Sanity check on one ticker
    if normaliser:
        example = next(iter(normaliser.keys()))
        ex_df   = pd.read_parquet(TRAIN_DIR / f"{example}.parquet")
        log.info(f"\nSanity check - {example} (train split):")
        log.info(f"  Shape      : {ex_df.shape}")
        log.info(f"  Columns    : {list(ex_df.columns)}")
        log.info(f"  Date range : {ex_df.index.min()} -> {ex_df.index.max()}")
        log.info(
            f"  Normaliser : mean={normaliser[example]['mean']:.6f}, "
            f"std={normaliser[example]['std']:.6f}"
        )
        log.info(
            f"  norm_ret   : mean={ex_df['norm_ret'].mean():.4f}, "
            f"std={ex_df['norm_ret'].std():.4f}  (should be ~0 and ~1)"
        )


if __name__ == "__main__":
    main()
