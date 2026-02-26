"""
05_evaluate.py
==============
Loads the best checkpoint and runs full evaluation on the test set.

Outputs
───────
  1. Classification report  (precision / recall / F1 per class + macro)
  2. Confusion matrix        (printed and saved as PNG)
  3. Simulated P&L           (simple, cost-aware backtest)

P&L simulation assumptions
───────────────────────────
  • BUY  signal  → enter long  at next bar's open (approximated as current close)
  • SELL signal  → enter short at next bar's open
  • HOLD signal  → no position
  • Each trade is held for at most HORIZON bars, then closed at market.
  • One-way transaction cost: COST_BPS basis points per trade (each side).
  • Position sizing: fixed fractional (1 unit per signal regardless of capital).
  • P&L = sum of per-trade log returns (long: +ret, short: -ret), minus 2×cost.
  • No compounding — returns are additive in log-return space.

Note: this is an optimistic simulation (no slippage beyond fixed cost, perfect
fill assumed). Treat it as a signal-quality indicator, not a live-trading result.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset import build_loaders, SP500Dataset, PROC_DIR, FEAT_COL, LABEL_COL
from train import TradingLSTM, get_device, CONFIG

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
CKPT_PATH  = Path("checkpoints/best_model.pt")
OUT_DIR    = Path("evaluation")
LABEL_NAMES = ["Hold", "Buy", "Sell"]

# P&L simulation
COST_BPS   = 3.0    # one-way transaction cost in basis points (3 bps = 0.03%)
HORIZON    = 30     # bars to hold trade (must match 02_label.py)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path: Path, device: torch.device) -> tuple[TradingLSTM, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt["config"]
    model = TradingLSTM(
        input_size  = cfg["input_size"],
        hidden_size = cfg["hidden_size"],
        num_layers  = cfg["num_layers"],
        dropout     = cfg["dropout"],
        num_classes = cfg["num_classes"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log.info(f"Loaded checkpoint from epoch {ckpt['epoch']}, val F1={ckpt['val_f1']:.4f}")
    return model, cfg


# ── Evaluation helpers ────────────────────────────────────────────────────────

def run_inference(
    model: nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (all_preds, all_probs, all_labels)."""
    all_preds  = []
    all_probs  = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x      = x.to(device)
            logits = model(x)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = probs.argmax(axis=1)
            all_preds.append(preds)
            all_probs.append(probs)
            all_labels.append(y.numpy())
    return (
        np.concatenate(all_preds),
        np.concatenate(all_probs),
        np.concatenate(all_labels),
    )


def plot_confusion_matrix(labels, preds, out_path: Path):
    cm  = confusion_matrix(labels, preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format=".2f")
    ax.set_title("Normalised Confusion Matrix — Test Set")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"Confusion matrix saved to {out_path}")


# ── P&L simulation ────────────────────────────────────────────────────────────

def simulate_pnl(
    split:    str = "test",
    seq_len:  int = 60,
    preds:    np.ndarray | None = None,
    cost_bps: float = COST_BPS,
    horizon:  int   = HORIZON,
) -> pd.DataFrame:
    """
    Reconstruct a simple P&L by replaying predictions against the raw
    normalised log-return series for each ticker in the test split.

    Because the Dataset merges tickers without preserving the bar-level
    mapping, we re-read each ticker's parquet directly and regenerate
    predictions bar-by-bar (model not needed here — we use stored labels
    from the labelled parquets as a proxy, since we want realised returns
    NOT the label).

    Strategy:
      • BUY  (pred=1): long  position, realised ret = sum of next 'horizon' log_rets
      • SELL (pred=2): short position, realised ret = -sum of next 'horizon' log_rets
      • HOLD (pred=0): skip

    Returns a DataFrame with columns: ticker, trade_ret, direction
    """
    COST = cost_bps * 1e-4   # convert bps to fraction

    in_dir   = PROC_DIR / f"{split}_labelled"
    records  = []
    pred_idx = 0   # pointer into flattened preds array (same order as Dataset)

    for path in sorted(in_dir.glob("*.parquet")):
        ticker = path.stem
        df = pd.read_parquet(path).sort_index()

        if len(df) < seq_len:
            continue

        log_rets = df[FEAT_COL].values   # normalised returns (used as proxy)
        n        = len(df)

        for end in range(seq_len - 1, n):
            start = end - seq_len + 1

            # Check same-day and no NaN (mirrors Dataset._load_ticker)
            window = log_rets[start : end + 1]
            dates  = df.index.normalize().values[start : end + 1]
            if np.any(np.isnan(window)) or dates[0] != dates[-1]:
                continue

            if preds is None or pred_idx >= len(preds):
                pred_idx += 1
                continue

            pred = int(preds[pred_idx])
            pred_idx += 1

            if pred == 0:
                continue  # Hold — no trade

            # Realised return: sum of log_ret for bars end+1 … end+horizon
            future = log_rets[end + 1 : end + 1 + horizon]
            future = future[~np.isnan(future)]
            if len(future) == 0:
                continue

            cum_ret = float(future.sum())
            trade_ret = cum_ret if pred == 1 else -cum_ret
            trade_ret -= 2 * COST   # round-trip cost

            records.append({
                "ticker":     ticker,
                "trade_ret":  trade_ret,
                "direction":  "Buy" if pred == 1 else "Sell",
            })

    df_trades = pd.DataFrame(records)
    return df_trades


def print_pnl_summary(df_trades: pd.DataFrame):
    if df_trades.empty:
        log.warning("No trades generated.")
        return

    total      = df_trades["trade_ret"].sum()
    n_trades   = len(df_trades)
    win_rate   = (df_trades["trade_ret"] > 0).mean()
    avg_ret    = df_trades["trade_ret"].mean()
    sharpe_raw = (df_trades["trade_ret"].mean()
                  / (df_trades["trade_ret"].std() + 1e-10)
                  * np.sqrt(252 * 390))  # annualised (390 bars/day)

    print("\n" + "═" * 55)
    print("  P&L SIMULATION SUMMARY (cost-aware, no compounding)")
    print("═" * 55)
    print(f"  Total trades       : {n_trades:>10,}")
    print(f"  Win rate           : {win_rate:>10.1%}")
    print(f"  Avg trade log-ret  : {avg_ret:>10.5f}")
    print(f"  Total log-ret sum  : {total:>10.4f}")
    print(f"  Annualised Sharpe* : {sharpe_raw:>10.2f}")
    print("  * raw estimate — treat with caution")
    by_dir = df_trades.groupby("direction")["trade_ret"].agg(["count", "mean", "sum"])
    print(f"\n{by_dir.to_string()}")
    print("═" * 55)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()

    # Load model
    model, cfg = load_model(CKPT_PATH, device)

    # Build test loader (no sampler needed for evaluation)
    log.info("Building test data loader …")
    _, _, test_loader, _ = build_loaders(
        seq_len     = cfg["seq_len"],
        batch_size  = cfg["batch_size"],
        num_workers = cfg.get("num_workers", 4),
        use_sampler = False,
        max_files   = cfg.get("max_files"),
    )

    # Inference
    log.info("Running inference on test set …")
    preds, probs, labels = run_inference(model, test_loader, device)

    # ── Classification report ──────────────────────────────────────────────
    print("\n" + "═" * 55)
    print("  CLASSIFICATION REPORT — TEST SET")
    print("═" * 55)
    print(classification_report(labels, preds, target_names=LABEL_NAMES, digits=4))

    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    log.info(f"Macro F1 (test): {macro_f1:.4f}")

    # ── Confusion matrix ───────────────────────────────────────────────────
    plot_confusion_matrix(labels, preds, OUT_DIR / "confusion_matrix.png")

    # ── Confidence histogram ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
    for i, name in enumerate(LABEL_NAMES):
        axes[i].hist(probs[:, i], bins=50, color=["steelblue", "green", "crimson"][i],
                     alpha=0.8, edgecolor="white", linewidth=0.3)
        axes[i].set_title(f"P({name})")
        axes[i].set_xlabel("Predicted probability")
    axes[0].set_ylabel("Count")
    fig.suptitle("Model Confidence Distribution — Test Set")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confidence_histograms.png", dpi=150)
    plt.close(fig)
    log.info(f"Confidence histograms saved to {OUT_DIR / 'confidence_histograms.png'}")

    # ── P&L simulation ─────────────────────────────────────────────────────
    log.info("Running P&L simulation …")
    df_trades = simulate_pnl(
        split    = "test",
        seq_len  = cfg["seq_len"],
        preds    = preds,
        cost_bps = COST_BPS,
        horizon  = HORIZON,
    )
    print_pnl_summary(df_trades)

    if not df_trades.empty:
        df_trades.to_csv(OUT_DIR / "trades.csv", index=False)
        log.info(f"Trade log saved to {OUT_DIR / 'trades.csv'}")

        # Cumulative P&L chart
        cumret = df_trades["trade_ret"].cumsum()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(cumret.values, linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Cumulative log-return")
        ax.set_title("Cumulative P&L — Test Set (normalised log-ret, cost-deducted)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "cumulative_pnl.png", dpi=150)
        plt.close(fig)
        log.info(f"Cumulative P&L chart saved to {OUT_DIR / 'cumulative_pnl.png'}")

    log.info("Evaluation complete.")


if __name__ == "__main__":
    main()
