"""
01_data_pipeline.py
====================
Loads raw 1-minute OHLCV CSVs from rawSP500data/, filters to market hours,
computes log returns from Close prices, normalizes using a trailing rolling
z-score (no global leakage), then splits into train / val / test sets and
saves them as parquet files.

Directory structure produced:
    processed/
        train/  -> 2020-12-28 .. 2022-12-31
        val/    -> 2023-01-01 .. 2023-12-31
        test/   -> 2024-01-01 .. 2025-12-23

Usage:
    python 01_data_pipeline.py
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────
RAW_DATA_DIR   = Path("rawSP500data")
PROCESSED_DIR  = Path("processed")
MARKET_OPEN    = "09:30"
MARKET_CLOSE   = "16:00"

# Limit number of tickers for faster processing (set to None for all)
MAX_TICKERS = 2

# Chronological split boundaries (inclusive)
TRAIN_END = "2022-12-31"
VAL_START = "2023-01-01"
VAL_END   = "2023-12-31"
TEST_START= "2024-01-01"

# Trailing window (in bars) for rolling z-score normalisation of log returns.
# 390 bars = 1 full trading day of 1-minute data.
NORM_WINDOW = 390

# Minimum bars a ticker must have after cleaning to be included.
MIN_BARS = NORM_WINDOW * 10


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_ticker(filepath: Path) -> pd.DataFrame | None:
    """
    Load a single ticker CSV. Returns a DataFrame with a DatetimeIndex and
    a 'Close' column, restricted to market hours only (no pre/after market,
    no cross-day contamination within a single session).
    """
    ticker = filepath.stem.split("_")[0]
    try:
        df = pd.read_csv(
            filepath,
            parse_dates=["Date"],
            index_col="Date",
        )
    except Exception as e:
        print(f"  [WARN] Could not read {filepath.name}: {e}")
        return None

    if "Close" not in df.columns:
        print(f"  [WARN] No 'Close' column in {filepath.name}")
        return None

    df = df[["Close"]].copy()
    df.index = pd.to_datetime(df.index)

    # Filter to market hours only
    df = df.between_time(MARKET_OPEN, MARKET_CLOSE)

    # Drop NaN / zero / negative closes
    df = df[df["Close"].notna() & (df["Close"] > 0)]

    # Drop duplicate timestamps
    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True)

    if len(df) < MIN_BARS:
        print(f"  [SKIP] {ticker}: only {len(df)} bars after cleaning")
        return None

    df["ticker"] = ticker
    return df


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns within each trading day so we never compute a
    return spanning the overnight gap.  The first bar of each day gets NaN
    and is subsequently dropped.
    """
    df = df.copy()
    df["date"] = df.index.date

    # Group by ticker + date to compute intra-day log returns only
    df["log_ret"] = df.groupby(["ticker", "date"])["Close"].transform(
        lambda s: np.log(s / s.shift(1))
    )

    df.drop(columns=["date", "Close"], inplace=True)
    df.dropna(subset=["log_ret"], inplace=True)
    return df


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    Trailing rolling z-score.  Uses only past data (min_periods=window).
    Returns NaN for the first `window` observations — these are dropped later.
    """
    roll  = series.rolling(window=window, min_periods=window)
    mu    = roll.mean()
    sigma = roll.std(ddof=1)
    return (series - mu) / sigma.replace(0, np.nan)


def normalise_ticker(df_ticker: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Apply trailing rolling z-score to log_ret for a single ticker's data.
    The rolling statistics are computed strictly on past data so there is
    no lookahead.  The first `window` rows (which would have NaN z-scores)
    are dropped.
    """
    df_ticker = df_ticker.copy().sort_index()
    df_ticker["norm_ret"] = rolling_zscore(df_ticker["log_ret"], window)
    df_ticker.dropna(subset=["norm_ret"], inplace=True)
    return df_ticker


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    csv_files = sorted(RAW_DATA_DIR.glob("*_1min.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{RAW_DATA_DIR}'. "
            "Make sure your data files follow the pattern TICKER_1min.csv"
        )

    # Limit tickers for faster processing
    if MAX_TICKERS is not None:
        csv_files = csv_files[:MAX_TICKERS]

    print(f"Processing {len(csv_files)} ticker files.")
    print("Loading and cleaning raw data...")

    frames = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(load_ticker, fp): fp for fp in csv_files}
        for future in as_completed(futures):
            df = future.result()
            if df is not None:
                frames.append(df)

    if not frames:
        raise RuntimeError("No valid ticker data loaded — check your CSV files.")

    print(f"Loaded {len(frames)} tickers successfully.")

    # ── Combine all tickers ───────────────────────────────────────────────────
    raw = pd.concat(frames, axis=0)
    raw.sort_index(inplace=True)

    print(f"Total bars (all tickers): {len(raw):,}")
    print(f"Date range: {raw.index.min()} → {raw.index.max()}")

    # ── Compute log returns (intra-day only) ──────────────────────────────────
    print("Computing intra-day log returns...")
    ret_df = compute_log_returns(raw)
    print(f"Bars after log-return computation: {len(ret_df):,}")

    # ── Trailing rolling z-score normalisation per ticker ────────────────────
    print(f"Normalising with trailing rolling z-score (window={NORM_WINDOW})...")
    normalised_parts = []
    
    tickers_list = list(ret_df.groupby("ticker"))
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(normalise_ticker, grp, NORM_WINDOW): ticker 
                   for ticker, grp in tickers_list}
        for future in as_completed(futures):
            normed = future.result()
            normalised_parts.append(normed)

    full = pd.concat(normalised_parts, axis=0).sort_index()
    print(f"Bars after normalisation: {len(full):,}")

    # ── Chronological split ───────────────────────────────────────────────────
    train_df = full.loc[:TRAIN_END]
    val_df   = full.loc[VAL_START:VAL_END]
    test_df  = full.loc[TEST_START:]

    print(f"\nSplit summary:")
    print(f"  Train : {train_df.index.min().date()} → {train_df.index.max().date()}  "
          f"({len(train_df):,} bars, {train_df['ticker'].nunique()} tickers)")
    print(f"  Val   : {val_df.index.min().date()} → {val_df.index.max().date()}    "
          f"({len(val_df):,} bars, {val_df['ticker'].nunique()} tickers)")
    print(f"  Test  : {test_df.index.min().date()} → {test_df.index.max().date()}  "
          f"({len(test_df):,} bars, {test_df['ticker'].nunique()} tickers)")

    # ── Save as parquet ───────────────────────────────────────────────────────
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_dir = PROCESSED_DIR / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "data.parquet"
        split_df.to_parquet(out_path)
        print(f"  Saved {split_name} → {out_path}")

    # Also save a combined parquet for label generation convenience
    full_path = PROCESSED_DIR / "full.parquet"
    full.to_parquet(full_path)
    print(f"  Saved combined → {full_path}")

    print("\nData pipeline complete.")


if __name__ == "__main__":
    main()
