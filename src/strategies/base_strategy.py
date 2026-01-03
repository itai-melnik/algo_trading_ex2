"""
Base Strategy abstract class.
"""
from abc import ABC, abstractmethod
import pandas as pd
from .signals import Signal


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    Subclasses must implement:
        - name: Property returning the strategy name
        - generate_signal: Method that analyzes price data and returns a Signal
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this strategy."""
        pass
    
    @abstractmethod
    def generate_signal(self, symbol: str, price_history: pd.DataFrame) -> Signal:
        """
        Generate a trading signal based on price history.
        
        Args:
            symbol: The stock ticker symbol
            price_history: DataFrame with columns: open, high, low, close, volume
                          Index should be datetime or date strings
        
        Returns:
            Signal object with action (BUY/SELL/HOLD), confidence, and reason
        """
        pass

