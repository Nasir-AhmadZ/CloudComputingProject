"""
02_label.py
===========
Reads the processed (normalised) parquet files and attaches Buy / Hold / Sell
labels using a volatility-adaptive TP/SL scheme.

Labelling logic
───────────────
For each bar t we:

  1. Compute a trailing 60-bar rolling std of log_ret using only bars BEFORE t
     (shift(1) ensures bar t itself is not included in its own estimate).

  2. Multiply by VOL_MULT to get a symmetric threshold (same value used for
     both TP and SL in price-return space).

  3. Walk forward up to HORIZON bars (bar t+1 … t+HORIZON).
     At each step accumulate the cumulative log return from bar t.
     • If cum_log_ret >=  threshold → BUY  label (price rose by TP first)
     • If cum_log_ret <= -threshold → SELL label (price fell by SL first)
     • If neither triggered within HORIZON bars → HOLD

  4. Bars where the rolling std cannot be computed (first 60 bars of a
     session or where vol is zero) are dropped.

Note: "Buy" here means "at bar t the price subsequently rose by ≥ threshold
within 30 minutes, before it fell by ≥ threshold" — i.e. a long entry signal.
"Sell" is the mirror. This framing makes BUY and SELL symmetric.

Parameters (edit freely)
─────────────────────────
  VOL_WINDOW = 60    rolling window for volatility estimate (bars = minutes)
  VOL_MULT   = 1.5   TP/SL = VOL_MULT * rolling_std
  HORIZON    = 30    forward look window (bars = minutes)

Outputs
───────
  processed/train_labelled/  AAPL.parquet, ...
  processed/val_labelled/    AAPL.parquet, ...
  processed/test_labelled/   AAPL.parquet, ...

Each parquet keeps columns:
  log_ret, norm_ret, vol_est, threshold, label  (0=Hold, 1=Buy, 2=Sell)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Configuration ────────────────────────────────────────────────────────────
PROC_DIR  = Path("processed")
SPLITS    = ["train", "val", "test"]

VOL_WINDOW = 60     # bars for rolling volatility
VOL_MULT   = 3.0    # multiplier: TP = SL = VOL_MULT * σ
HORIZON    = 30     # forward bars to simulate

LABEL_MAP = {0: "Hold", 1: "Buy", 2: "Sell"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Core labelling ───────────────────────────────────────────────────────────

def compute_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    df must have columns: log_ret, norm_ret
    index: datetime (market hours only, no cross-day NaN handling needed
           because 01_preprocess.py already set cross-day returns to NaN
           and dropped them).

    Returns df with added columns: vol_est, threshold, label
    """
    df = df.copy()

    # ── 1. Trailing volatility estimate ─────────────────────────────────────
    # shift(1): window ends at t-1, so bar t is NOT included in its own vol.
    # min_periods ensures we only produce estimates with enough history.
    df["vol_est"] = (
        df["log_ret"]
        .shift(1)
        .rolling(window=VOL_WINDOW, min_periods=VOL_WINDOW)
        .std(ddof=1)
    )
    df["threshold"] = VOL_MULT * df["vol_est"]

    # ── 2. Forward simulation ────────────────────────────────────────────────
    log_rets = df["log_ret"].values
    thresholds = df["threshold"].values
    n = len(df)
    labels = np.full(n, -1, dtype=np.int8)   # -1 = will be masked later

    for i in range(n):
        thr = thresholds[i]
        if np.isnan(thr) or thr <= 0:
            continue   # not enough history — will be dropped

        cum_ret = 0.0
        label   = 0   # default Hold

        end = min(i + HORIZON, n - 1)
        for j in range(i + 1, end + 1):
            r = log_rets[j]
            if np.isnan(r):
                # Hit a day boundary inside the horizon — stop here
                break
            cum_ret += r
            if cum_ret >= thr:
                label = 1   # Buy
                break
            if cum_ret <= -thr:
                label = 2   # Sell
                break

        labels[i] = label

    df["label"] = labels

    # Drop bars where vol_est was undefined or label couldn't be set
    df = df[df["label"] >= 0].copy()
    df["label"] = df["label"].astype(np.int8)

    return df


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    all_counts = {split: {0: 0, 1: 0, 2: 0} for split in SPLITS}

    for split in SPLITS:
        in_dir  = PROC_DIR / split
        out_dir = PROC_DIR / f"{split}_labelled"
        out_dir.mkdir(parents=True, exist_ok=True)

        parquets = sorted(in_dir.glob("*.parquet"))
        if not parquets:
            log.warning(f"No parquet files found in {in_dir}")
            continue

        log.info(f"\n── {split.upper()} ({'─'*40})")

        for path in tqdm(parquets, desc=f"Labelling {split}"):
            ticker = path.stem
            df = pd.read_parquet(path)

            if len(df) < VOL_WINDOW + HORIZON + 10:
                log.warning(f"  {ticker}: too short ({len(df)} rows) – skipping.")
                continue

            df_labelled = compute_labels(df)

            if len(df_labelled) == 0:
                log.warning(f"  {ticker}: no valid labels produced – skipping.")
                continue

            df_labelled.to_parquet(out_dir / f"{ticker}.parquet")

            # Accumulate counts
            vc = df_labelled["label"].value_counts()
            for k in (0, 1, 2):
                all_counts[split][k] += int(vc.get(k, 0))

    # ── Report class distribution ────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  CLASS DISTRIBUTION SUMMARY")
    print("═" * 60)
    for split in SPLITS:
        counts = all_counts[split]
        total  = sum(counts.values()) or 1
        print(f"\n  {split.upper()}")
        for k, name in LABEL_MAP.items():
            c = counts[k]
            print(f"    {name:4s} ({k}): {c:>10,d}  ({100*c/total:5.1f}%)")
    print("═" * 60)

    print("""
Interpretation guide
────────────────────
• If Hold >> Buy ≈ Sell, your VOL_MULT is too high → lower it.
• If Buy  ≈ Sell ≈ Hold (≈33% each), the labelling is well-balanced.
• If Buy  >> Sell or Sell >> Buy, check for a systematic drift/bias.

Typical good target: Hold 50-60%, Buy 20-25%, Sell 20-25%.
If Hold > 70%, try reducing VOL_MULT (e.g. 1.0 or 1.2).
""")


if __name__ == "__main__":
    main()
