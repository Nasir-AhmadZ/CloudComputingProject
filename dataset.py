"""
dataset.py
==========
Import shim for 03_dataset.py
"""
from importlib import import_module

_module = import_module("03_dataset")
TradingDataset = _module.TradingDataset
compute_class_weights = _module.compute_class_weights

__all__ = ["TradingDataset", "compute_class_weights"]
