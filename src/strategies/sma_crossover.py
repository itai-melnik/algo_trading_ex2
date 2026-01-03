"""
Simple Moving Average (SMA) Crossover Strategy.

Generates BUY signals when fast SMA crosses above slow SMA,
and SELL signals when fast SMA crosses below slow SMA.
"""
import pandas as pd
from .base_strategy import BaseStrategy
from .signals import Signal


class SMACrossoverStrategy(BaseStrategy):
    """
    SMA Crossover strategy implementation.
    
    Args:
        fast_period: Number of days for the fast (short-term) SMA
        slow_period: Number of days for the slow (long-term) SMA
    """
    
    def __init__(self, fast_period: int = 10, slow_period: int = 20):
        if fast_period >= slow_period:
            raise ValueError("fast_period must be less than slow_period")
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    @property
    def name(self) -> str:
        return f"SMA_Crossover_{self.fast_period}_{self.slow_period}"
    
    def generate_signal(self, symbol: str, price_history: pd.DataFrame) -> Signal:
        """
        Generate signal based on SMA crossover.
        
        Args:
            symbol: Stock ticker
            price_history: DataFrame with 'close' column
            
        Returns:
            Signal with BUY/SELL/HOLD action
        """
        # Check for sufficient data
        if len(price_history) < self.slow_period:
            return Signal(
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reason=f"Insufficient data: need {self.slow_period} days, have {len(price_history)}"
            )
        
        closes = price_history['close'].values
        
        # Calculate SMAs
        fast_sma = self._calculate_sma(closes, self.fast_period)
        slow_sma = self._calculate_sma(closes, self.slow_period)
        
        # Get current and previous values for crossover detection
        current_fast = fast_sma[-1]
        current_slow = slow_sma[-1]
        
        # Calculate the difference as a percentage of price
        current_price = closes[-1]
        diff_pct = (current_fast - current_slow) / current_price
        
        # Determine signal based on SMA relationship
        if current_fast > current_slow:
            # Fast SMA is above slow SMA - bullish
            confidence = min(abs(diff_pct) * 10, 1.0)  # Scale confidence
            return Signal(
                symbol=symbol,
                action="BUY",
                confidence=confidence,
                strategy_name=self.name,
                reason=f"Fast SMA ({self.fast_period}d: {current_fast:.2f}) > Slow SMA ({self.slow_period}d: {current_slow:.2f})"
            )
        elif current_fast < current_slow:
            # Fast SMA is below slow SMA - bearish
            confidence = min(abs(diff_pct) * 10, 1.0)
            return Signal(
                symbol=symbol,
                action="SELL",
                confidence=confidence,
                strategy_name=self.name,
                reason=f"Fast SMA ({self.fast_period}d: {current_fast:.2f}) < Slow SMA ({self.slow_period}d: {current_slow:.2f})"
            )
        else:
            # SMAs are equal - no signal
            return Signal(
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reason=f"Fast SMA equals Slow SMA at {current_fast:.2f}"
            )
    
    def _calculate_sma(self, prices: list, period: int) -> list:
        """Calculate Simple Moving Average."""
        sma = []
        for i in range(len(prices)):
            if i < period - 1:
                sma.append(prices[i])  # Not enough data yet
            else:
                window = prices[i - period + 1:i + 1]
                sma.append(sum(window) / period)
        return sma

