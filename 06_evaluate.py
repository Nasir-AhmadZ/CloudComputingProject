"""
06_evaluate.py
==============
Loads the best checkpoint and evaluates on the held-out test set.

Outputs
-------
* Full classification report (precision / recall / F1 per class)
* Confusion matrix
* A simple P&L backtest:
    - Enter long on Buy signal, short on Sell signal
    - Exit after HORIZON bars OR when the opposite signal fires
    - Deduct a fixed one-way slippage per trade
    - Reports cumulative return, Sharpe ratio, win-rate per class

Usage:
    python 06_evaluate.py
"""

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))

from dataset import TradingDataset
from model   import TradingLSTM


# ── Configuration ─────────────────────────────────────────────────────────────
CHECKPOINT_PATH = Path("checkpoints/best_model.pt")
PROCESSED_DIR   = Path("processed")
RESULTS_DIR     = Path("results")

BATCH_SIZE      = 512
NUM_WORKERS     = 4
SLIPPAGE        = 0.0002   # 2 basis points per side (one-way)
HORIZON         = 30       # bars to hold if no exit signal


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: Path, device: torch.device):
    ckpt   = torch.load(checkpoint_path, map_location=device)
    cfg    = ckpt["config"]
    model  = TradingLSTM(
        num_tickers = cfg["num_tickers"],
        hidden_size = cfg["hidden_size"],
        num_layers  = cfg["num_layers"],
        dropout     = cfg["dropout"],
        embed_dim   = cfg["embed_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["ticker2idx"], cfg


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict(model, loader, device):
    all_preds   = []
    all_targets = []
    all_probs   = []

    with torch.no_grad():
        for x_batch, t_batch, y_batch in loader:
            logits = model(x_batch.to(device), t_batch.to(device))
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y_batch.numpy())
            all_probs.append(probs)

    return (
        np.concatenate(all_preds),
        np.concatenate(all_targets),
        np.concatenate(all_probs),
    )


# ── Simple P&L backtest ───────────────────────────────────────────────────────

def backtest(test_df: pd.DataFrame, preds: np.ndarray) -> dict:
    """
    Very simplified backtest on the test set.

    Strategy:
        Buy  signal (pred=0) → go long:  PnL = future log return over HORIZON bars
        Sell signal (pred=2) → go short: PnL = -future log return over HORIZON bars
        Hold signal (pred=1) → skip

    We deduct SLIPPAGE (one-way) from each trade's P&L.

    This is directional, not mark-to-market, and ignores order execution
    details — treat it as a rough signal-quality measure only.
    """
    test_df = test_df.reset_index(drop=False)
    n       = len(test_df)
    assert len(preds) == n, "preds / df length mismatch"

    log_ret = test_df["log_ret"].values
    trades  = []

    for i in range(n):
        sig = preds[i]
        if sig == 1:
            continue   # Hold

        # Accumulate forward return over HORIZON bars within the same series
        # (simple: just sum; ignores day boundaries for brevity)
        end     = min(i + HORIZON, n - 1)
        fwd_ret = log_ret[i + 1: end + 1].sum()

        direction = +1 if sig == 0 else -1   # long for Buy, short for Sell
        trade_pnl = direction * fwd_ret - SLIPPAGE

        trades.append({
            "idx":       i,
            "signal":    sig,
            "fwd_ret":   fwd_ret,
            "trade_pnl": trade_pnl,
        })

    if not trades:
        return {"n_trades": 0}

    tdf          = pd.DataFrame(trades)
    n_trades     = len(tdf)
    cum_return   = tdf["trade_pnl"].sum()
    mean_ret     = tdf["trade_pnl"].mean()
    std_ret      = tdf["trade_pnl"].std()
    sharpe       = (mean_ret / std_ret * np.sqrt(252 * 390)) if std_ret > 0 else 0
    win_rate     = (tdf["trade_pnl"] > 0).mean()
    n_buy        = (tdf["signal"] == 0).sum()
    n_sell       = (tdf["signal"] == 2).sum()

    return {
        "n_trades":    n_trades,
        "n_buy":       n_buy,
        "n_sell":      n_sell,
        "cum_return":  cum_return,
        "mean_trade":  mean_ret,
        "win_rate":    win_rate,
        "sharpe_ann":  sharpe,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. "
            "Run 05_train.py first."
        )

    print("Loading model checkpoint...")
    model, ticker2idx, cfg = load_model(CHECKPOINT_PATH, device)

    # Build test dataset — use ticker2idx from training so indices match
    print("Loading test dataset...")
    test_ds = TradingDataset("test", seq_len=cfg["seq_len"])
    # Remap test dataset ticker indices to training vocabulary
    # (any test ticker not seen in training gets index 0 — a known limitation)
    for i, (x, t_idx, y) in enumerate(test_ds.windows):
        ticker_name = [k for k, v in test_ds.ticker2idx.items() if v == t_idx]
        if ticker_name:
            new_idx = ticker2idx.get(ticker_name[0], 0)
            test_ds.windows[i] = (x, new_idx, y)

    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    print("Running inference on test set...")
    preds, targets, probs = predict(model, test_loader, device)

    # ── Classification report ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT (Test set)")
    print("="*60)
    report = classification_report(
        targets, preds,
        target_names=["Buy", "Hold", "Sell"],
        zero_division=0,
    )
    print(report)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(targets, preds)
    cm_df = pd.DataFrame(
        cm,
        index=["True Buy", "True Hold", "True Sell"],
        columns=["Pred Buy", "Pred Hold", "Pred Sell"],
    )
    print("CONFUSION MATRIX")
    print(cm_df.to_string())

    # ── Backtest ──────────────────────────────────────────────────────────────
    test_parquet = PROCESSED_DIR / "test" / "data_labelled.parquet"
    test_raw     = pd.read_parquet(test_parquet)

    # Align preds to the test_ds window order
    # (test_ds.windows has the same order as what the dataloader iterates)
    # For backtest we only need the flat log_ret and preds in the same order
    # Build a flat log_ret array aligned to the test_ds windows
    flat_log_ret = np.array([test_ds.windows[i][0][-1] for i in range(len(test_ds))])
    # Use a dummy df just to pass into backtest (we need log_ret column)
    bt_df = pd.DataFrame({"log_ret": flat_log_ret})

    bt_result = backtest(bt_df, preds)
    print("\n" + "="*60)
    print("BACKTEST SUMMARY (simplified, slippage={:.1f}bps/side)".format(
        SLIPPAGE * 10000
    ))
    print("="*60)
    for k, v in bt_result.items():
        if isinstance(v, float):
            print(f"  {k:<18}: {v:.4f}")
        else:
            print(f"  {k:<18}: {v}")

    # ── Save results ──────────────────────────────────────────────────────────
    report_path = RESULTS_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
        f.write("\n\nCONFUSION MATRIX\n")
        f.write(cm_df.to_string())
        f.write("\n\nBACKTEST\n")
        for k, v in bt_result.items():
            f.write(f"  {k}: {v}\n")

    cm_df.to_csv(RESULTS_DIR / "confusion_matrix.csv")
    print(f"\nResults saved → {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
