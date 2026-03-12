"""
model.py
========
Import shim for 04_model.py
"""
from importlib import import_module

_module = import_module("04_model")
TradingLSTM = _module.TradingLSTM

__all__ = ["TradingLSTM"]
