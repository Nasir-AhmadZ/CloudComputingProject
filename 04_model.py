"""
04_model.py
===========
LSTM classifier for intraday trading signals.

Architecture
------------
Input          : (batch, seq_len, 1)  — normalised log returns
Ticker embed   : (batch, embed_dim)   — optional learned per-ticker identity
LSTM stack     : n_layers of LSTM with dropout between layers
Classifier head: Linear → 3 logits (Buy, Hold, Sell)

The ticker embedding is concatenated to the LSTM's final hidden state
before the classifier head, so the model can adjust its decision boundary
per ticker without changing the sequence encoding path.
"""

import torch
import torch.nn as nn


class TradingLSTM(nn.Module):
    """
    Parameters
    ----------
    num_tickers   : int   — vocabulary size for the ticker embedding
    input_size    : int   — feature dimension per bar (1 for univariate)
    hidden_size   : int   — LSTM hidden units
    num_layers    : int   — stacked LSTM layers
    dropout       : float — dropout between LSTM layers (and before head)
    embed_dim     : int   — ticker embedding dimension (0 = no embedding)
    num_classes   : int   — 3 (Buy / Hold / Sell)
    """

    def __init__(
        self,
        num_tickers: int,
        input_size:  int   = 1,
        hidden_size: int   = 128,
        num_layers:  int   = 2,
        dropout:     float = 0.3,
        embed_dim:   int   = 16,
        num_classes: int   = 3,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.embed_dim   = embed_dim

        # Ticker embedding (optional — set embed_dim=0 to disable)
        if embed_dim > 0:
            self.ticker_embed = nn.Embedding(num_tickers, embed_dim)
        else:
            self.ticker_embed = None

        # LSTM
        # dropout only applied between layers (not after the last layer)
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        # Classifier head
        head_input_size = hidden_size + (embed_dim if embed_dim > 0 else 0)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_input_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier / orthogonal init for stable training."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Set forget-gate bias to 1 to encourage remembering early
                n = param.size(0)
                param.data[n // 4: n // 2].fill_(1.0)

        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        x: torch.Tensor,          # (batch, seq_len, input_size)
        ticker_idx: torch.Tensor,  # (batch,)
    ) -> torch.Tensor:             # (batch, num_classes)

        # LSTM — we only need the final hidden state
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers, batch, hidden_size) — take the last layer
        h_last = h_n[-1]           # (batch, hidden_size)

        if self.ticker_embed is not None:
            t_emb = self.ticker_embed(ticker_idx)   # (batch, embed_dim)
            h_last = torch.cat([h_last, t_emb], dim=-1)

        logits = self.head(h_last)   # (batch, num_classes)
        return logits


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, T, F = 8, 60, 1
    model = TradingLSTM(num_tickers=500)
    x     = torch.randn(B, T, F)
    tix   = torch.randint(0, 500, (B,))
    out   = model(x, tix)
    print(f"Output shape: {out.shape}")   # expect (8, 3)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
