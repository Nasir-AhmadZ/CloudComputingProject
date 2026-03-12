"""
02_label_generator.py
======================
Reads the normalised parquet files produced by 01_data_pipeline.py and
attaches Buy / Hold / Sell labels using a volatility-scaled TP / SL scheme.

Labelling logic
---------------
For each bar t (using the raw log_ret column, NOT the normalised column):

1. Compute a trailing 60-bar rolling std of log returns, using only data
   strictly before bar t  (via .shift(1) before rolling — no lookahead).

2. Set:
       TP threshold = vol_t * TP_MULT
       SL threshold = vol_t * SL_MULT

3. Look ahead up to HORIZON bars (30 minutes).  Accumulate the cumulative
   log return from t+1 onward.  The label is determined by whichever
   boundary is hit first:
       cum_ret >= +TP  →  Buy
       cum_ret <= -SL  →  Sell
       neither hit     →  Hold

   Note: SL is always positive; we compare cum_ret ≤ −SL_threshold.

4. Bars where volatility cannot be computed (not enough history) get
   label NaN and are dropped.

Outputs
-------
    processed/train/data_labelled.parquet
    processed/val/data_labelled.parquet
    processed/test/data_labelled.parquet

Label encoding: Buy=0, Hold=1, Sell=2

Usage:
    python 02_label_generator.py
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("processed")

VOL_WINDOW = 60      # bars for trailing volatility estimate
TP_MULT    = 1.5     # take-profit = vol * TP_MULT
SL_MULT    = 1.5     # stop-loss   = vol * SL_MULT  (symmetric; tune as needed)
HORIZON    = 30      # forward-looking bars (minutes)

LABEL_MAP  = {"Buy": 0, "Hold": 1, "Sell": 2}
SPLITS     = ["train", "val", "test"]


# ── Labelling ─────────────────────────────────────────────────────────────────

def compute_trailing_vol(log_ret: pd.Series, window: int) -> pd.Series:
    """
    Trailing volatility: rolling std over `window` bars, shifted by 1 so the
    current bar's return is NOT included in its own volatility estimate.
    min_periods=window ensures we don't use partial windows.
    """
    return (
        log_ret
        .shift(1)                                  # exclude current bar
        .rolling(window=window, min_periods=window)
        .std(ddof=1)
    )


def label_single_ticker(df: pd.DataFrame) -> pd.Series:
    """
    Given a single ticker's DataFrame (sorted by time, intra-day only),
    return a Series of labels (int) aligned to df's index.
    Vectorized for speed.
    """
    df = df.sort_index().copy()
    log_ret = df["log_ret"].values
    n       = len(log_ret)

    vol   = compute_trailing_vol(df["log_ret"], VOL_WINDOW).values
    tp_th = vol * TP_MULT
    sl_th = vol * SL_MULT

    labels = np.ones(n, dtype=np.int8)  # default: Hold
    labels[np.isnan(vol) | (vol == 0)] = -1  # mark invalid

    # Precompute cumulative returns for all horizons
    for i in range(n - 1):
        if labels[i] == -1:
            continue
        
        end = min(i + HORIZON, n - 1)
        cum_rets = np.cumsum(log_ret[i+1:end+1])
        
        # Check TP first
        tp_hit = np.where(cum_rets >= tp_th[i])[0]
        sl_hit = np.where(cum_rets <= -sl_th[i])[0]
        
        if len(tp_hit) > 0 and (len(sl_hit) == 0 or tp_hit[0] < sl_hit[0]):
            labels[i] = 0  # Buy
        elif len(sl_hit) > 0:
            labels[i] = 2  # Sell

    labels[labels == -1] = np.nan
    return pd.Series(labels, index=df.index, dtype="Int64")


def label_split(split: str) -> None:
    in_path  = PROCESSED_DIR / split / "data.parquet"
    out_path = PROCESSED_DIR / split / "data_labelled.parquet"

    if not in_path.exists():
        print(f"  [SKIP] {in_path} not found — run 01_data_pipeline.py first")
        return

    print(f"\nLabelling {split} split...")
    df = pd.read_parquet(in_path)

    label_parts = []
    tickers = df["ticker"].unique()
    ticker_dfs = [(ticker, df[df["ticker"] == ticker].copy()) for ticker in tickers]

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(label_single_ticker, sub): ticker 
                   for ticker, sub in ticker_dfs}
        for future in as_completed(futures):
            lbl = future.result()
            label_parts.append(lbl)

    all_labels = pd.concat(label_parts).rename("label")
    df = df.join(all_labels, how="left")

    before = len(df)
    df.dropna(subset=["label"], inplace=True)
    df["label"] = df["label"].astype(int)
    after  = len(df)

    print(f"  Bars before/after dropping unlabelled: {before:,} → {after:,}")

    counts = Counter(df["label"])
    total  = sum(counts.values())
    label_names = {0: "Buy", 1: "Hold", 2: "Sell"}
    print("  Label distribution:")
    for k in sorted(counts):
        pct = 100 * counts[k] / total
        print(f"    {label_names[k]:4s} ({k}): {counts[k]:>8,}  ({pct:.1f}%)")

    df.to_parquet(out_path)
    print(f"  Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    for split in SPLITS:
        label_split(split)

    print("\nLabel generation complete.")
    print("\nTip: if Hold >> Buy/Sell, increase TP_MULT/SL_MULT slightly,")
    print("or reduce VOL_WINDOW to make thresholds more responsive.")


if __name__ == "__main__":
    main()
