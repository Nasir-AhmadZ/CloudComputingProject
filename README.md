# S&P 500 1-Minute LSTM Day Trading Classifier

End-to-end pipeline: raw OHLCV CSVs → log returns → labels → LSTM training → evaluation.

---

## Directory structure expected

```
project/
├── rawSP500data/
│   ├── AAPL_1min.csv
│   ├── MSFT_1min.csv
│   └── ...
├── 01_preprocess.py
├── 02_label.py
├── 03_dataset.py
├── 04_train.py
├── 05_evaluate.py
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

GPU support (recommended for training speed):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Pipeline — run in order

### Step 1 — Preprocess

```bash
python 01_preprocess.py
```

**What it does:**
- Reads every `*_1min.csv` from `rawSP500data/`
- Keeps only `Close` prices during market hours (09:30–16:00)
- Computes per-bar log returns; sets cross-day returns to NaN
- Fits a normaliser (mean & std of log-returns) on **training data only**
- Applies the normaliser to val/test (no leakage)
- Saves `processed/train/`, `processed/val/`, `processed/test/` as per-ticker Parquet files
- Saves `processed/normaliser.json`

**Split dates:**
| Split | Dates |
|-------|-------|
| Train | 2020-12-28 → 2022-12-31 |
| Val   | 2023-01-01 → 2023-12-31 |
| Test  | 2024-01-01 → 2025-12-23 |

---

### Step 2 — Label

```bash
python 02_label.py
```

**What it does:**
- For every bar `t`, computes trailing 60-bar rolling std of log-returns
  using only bars **before** `t` (`.shift(1)` — no leakage)
- Sets TP = SL = `VOL_MULT × σ` (default `VOL_MULT = 1.5`)
- Simulates each bar forward up to 30 bars:
  - Cumulative log-return hits +threshold → **Buy (1)**
  - Cumulative log-return hits −threshold → **Sell (2)**
  - Neither triggered → **Hold (0)**
- Prints class distribution for each split

**Tuning imbalance:**
- Hold > 70% → lower `VOL_MULT` (try 1.0 or 1.2)
- Buy/Sell severely asymmetric → check for systematic trend bias in your data

---

### Step 3 — Train

```bash
python 04_train.py
```

**What it does:**
- Builds pooled Dataset across all tickers (windows that span day boundaries are discarded)
- Uses `WeightedRandomSampler` to oversample minority classes in each batch
- Also passes class weights to `CrossEntropyLoss`
- Trains a stacked 2-layer LSTM (128 hidden units) with:
  - AdamW optimiser, cosine LR annealing
  - Gradient clipping (max norm 1.0)
  - LayerNorm + Dropout on the final hidden state
  - Early stopping on validation macro-F1 (patience=7)
- Saves best checkpoint to `checkpoints/best_model.pt`
- Logs per-epoch metrics to `checkpoints/training_log.csv`

**Key hyperparameters (edit `CONFIG` in `04_train.py`):**

| Parameter | Default | Notes |
|-----------|---------|-------|
| `seq_len` | 60 | Input window (bars) |
| `hidden_size` | 128 | LSTM units |
| `num_layers` | 2 | Stacked LSTM layers |
| `dropout` | 0.3 | |
| `batch_size` | 512 | Increase if GPU memory allows |
| `lr` | 3e-4 | Starting LR |
| `patience` | 7 | Early-stopping epochs |
| `max_files` | None | Set e.g. `10` for a quick smoke-test |

---

### Step 4 — Evaluate

```bash
python 05_evaluate.py
```

**What it does:**
- Loads best checkpoint, runs inference on test set
- Prints per-class precision / recall / F1 + macro averages
- Saves `evaluation/confusion_matrix.png`
- Saves `evaluation/confidence_histograms.png`
- Runs a simple cost-aware P&L simulation (3 bps one-way cost)
- Saves `evaluation/trades.csv` and `evaluation/cumulative_pnl.png`

---

## Model architecture

```
Input (batch, 60, 1)           ← normalised log-returns
    │
    ▼
LSTM × 2 layers (hidden=128)   ← causal, unidirectional
    │
    ▼
Last hidden state (batch, 128)
    │
    ▼
LayerNorm → Dropout(0.3)
    │
    ▼
Linear(128 → 3)
    │
    ▼
Logits → softmax → {Hold, Buy, Sell}
```

**Why unidirectional?** The model must only see past data. A bidirectional LSTM
would leak future information at inference time.

---

## Important caveats

1. **Transaction costs matter.** Even 3 bps per side can wipe out a weak edge at
   1-minute frequency. Measure your actual broker costs.

2. **The P&L simulation is optimistic.** It assumes perfect fills at the bar's
   price, no market impact, and no partial fills.

3. **Walk-forward validation** is more realistic than a single static test set.
   Consider re-running training on a rolling basis (e.g. retrain every quarter).

4. **Signal-to-noise at 1-minute resolution is low.** A macro F1 of 0.38–0.45
   may still imply a real edge; 0.33 is random. The P&L simulation is the more
   meaningful indicator.

5. **No risk management.** The simulation has no position sizing, no drawdown
   limits, and no portfolio-level constraints. These are critical for live trading.
# CloudComputingProject
