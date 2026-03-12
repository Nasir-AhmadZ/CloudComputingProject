# S&P500 Intraday LSTM Trading Classifier

## Overview
A complete pipeline for training a 3-class (Buy / Hold / Sell) LSTM classifier
on 1-minute S&P500 intraday data using volatility-scaled labelling and
rolling z-score normalisation with strict no-lookahead guarantees.

---

## Directory structure

```
lstm_trading/
├── rawSP500data/          ← Place your TICKER_1min.csv files here
│
├── 01_data_pipeline.py    ← Load, clean, compute log returns, normalise, split
├── 02_label_generator.py  ← Volatility-scaled Buy/Hold/Sell labelling
├── 03_dataset.py          ← PyTorch Dataset (rolling windows, no day-crossing)
├── 04_model.py            ← TradingLSTM architecture
├── 05_train.py            ← Training loop (class weights, early stopping)
├── 06_evaluate.py         ← Test-set eval + simplified P&L backtest
│
├── dataset.py             ← Import shim (used by train/evaluate)
├── model.py               ← Import shim (used by train/evaluate)
├── requirements.txt
│
├── processed/             ← Created by 01 and 02
│   ├── train/data.parquet
│   ├── train/data_labelled.parquet
│   ├── val/data.parquet
│   ├── val/data_labelled.parquet
│   ├── test/data.parquet
│   ├── test/data_labelled.parquet
│   └── full.parquet
│
├── checkpoints/           ← Created by 05_train.py
│   ├── best_model.pt
│   └── last_model.pt
│
├── logs/                  ← Created by 05_train.py
│   └── training_log.csv
│
└── results/               ← Created by 06_evaluate.py
    ├── classification_report.txt
    └── confusion_matrix.csv
```

---

## Setup

```bash
pip install -r requirements.txt
```

For GPU training install PyTorch with CUDA from https://pytorch.org/get-started/locally/

---

## Run order

```bash
# 1. Build normalised parquet splits
python 01_data_pipeline.py

# 2. Attach volatility-scaled labels
python 02_label_generator.py

# 3. (optional) Smoke-test the dataset
python 03_dataset.py

# 4. (optional) Smoke-test the model
python 04_model.py

# 5. Train
python 05_train.py

# 6. Evaluate on test set
python 06_evaluate.py
```

---

## Key design decisions

### No-lookahead normalisation
Log returns are z-scored using a **trailing** rolling window (390 bars = 1 day).
The rolling mean and std are computed on data strictly before each bar.

### No-lookahead volatility for labels
Volatility for TP/SL thresholds is computed with `.shift(1)` applied **before**
the rolling window, so the current bar's return never contributes to its own
threshold.

### No overnight contamination
Log returns are computed **within each trading day only** — the first bar of
each day gets NaN. Windows that cross a day boundary (gap > 2 minutes) are
discarded by the Dataset.

### Class imbalance
`CrossEntropyLoss(weight=class_weights)` uses inverse-frequency weights
computed on the training set only. Macro-F1 is the primary evaluation metric.

### Ticker identity
Each ticker gets a learned 16-dimensional embedding, concatenated to the LSTM's
final hidden state before the classifier head.

---

## Tuning guide

| Parameter      | Location            | Effect                                              |
|---------------|---------------------|-----------------------------------------------------|
| TP_MULT/SL_MULT | 02_label_generator | Higher → fewer Buy/Sell labels; lower → more noise  |
| VOL_WINDOW    | 02_label_generator  | Shorter → more responsive; longer → smoother        |
| HORIZON       | 02_label_generator  | Shorter → more frequent signals                     |
| HIDDEN_SIZE   | 05_train.py         | Larger → more capacity; watch overfitting            |
| DROPOUT       | 05_train.py         | Higher → more regularisation                         |
| NORM_WINDOW   | 01_data_pipeline.py | Longer → smoother normalisation                      |

---

## Important caveats

1. **Survivorship bias**: if your tickers are current S&P500 constituents,
   delisted stocks are absent — this biases toward winners.
2. **Transaction costs**: the backtest uses a fixed 2bps slippage. Real costs
   (spread, market impact) are higher at 1-minute frequency.
3. **Regime risk**: 2020-2025 includes COVID, meme stocks, and rate hike cycles
   — the model may not generalise beyond this window.
4. **Signal-to-noise**: 1-minute returns are very noisy; macro-F1 > 0.38
   (random baseline ~0.33) should be considered meaningful but not conclusive.
