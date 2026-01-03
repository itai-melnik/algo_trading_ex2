"""
Strategy Engine for algorithmic trading.

This module contains pure Python trading strategies that generate
deterministic signals based on price data. No LLM calls are made here.
"""

from .signals import Signal
from .base_strategy import BaseStrategy
from .sma_crossover import SMACrossoverStrategy
from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy
from .volatility_filter import VolatilityFilter
from .ensemble import EnsembleStrategy
from .nn_predictor import NNPredictorStrategy

__all__ = [
    'Signal',
    'BaseStrategy',
    'SMACrossoverStrategy',
    'MomentumStrategy',
    'MeanReversionStrategy',
    'VolatilityFilter',
    'EnsembleStrategy',
    'NNPredictorStrategy',
]
