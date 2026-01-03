"""
Volatility Filter.

Uses Average True Range (ATR) to determine if market conditions
are suitable for trading. High volatility may warrant avoiding trades.
"""
import pandas as pd
from typing import Tuple


class VolatilityFilter:
    """
    Volatility filter using ATR (Average True Range).
    
    Args:
        atr_period: Number of days for ATR calculation
        max_atr_percent: Maximum ATR as percentage of price to allow trading
    """
    
    def __init__(self, atr_period: int = 14, max_atr_percent: float = 5.0):
        if atr_period < 1:
            raise ValueError("atr_period must be at least 1")
        if max_atr_percent <= 0:
            raise ValueError("max_atr_percent must be positive")
        
        self.atr_period = atr_period
        self.max_atr_percent = max_atr_percent
    
    def check_volatility(self, symbol: str, price_history: pd.DataFrame) -> Tuple[bool, float]:
        """
        Check if volatility is within acceptable bounds.
        
        Args:
            symbol: Stock ticker
            price_history: DataFrame with 'high', 'low', 'close' columns
            
        Returns:
            Tuple of (can_trade: bool, atr_percent: float)
        """
        if len(price_history) < self.atr_period + 1:
            # Not enough data - assume safe to trade
            return True, 0.0
        
        # Calculate True Range for each day
        highs = price_history['high'].values
        lows = price_history['low'].values
        closes = price_history['close'].values
        
        true_ranges = []
        for i in range(1, len(price_history)):
            high_low = highs[i] - lows[i]
            high_prev_close = abs(highs[i] - closes[i - 1])
            low_prev_close = abs(lows[i] - closes[i - 1])
            tr = max(high_low, high_prev_close, low_prev_close)
            true_ranges.append(tr)
        
        # Calculate ATR (average of last N true ranges)
        recent_tr = true_ranges[-self.atr_period:]
        atr = sum(recent_tr) / len(recent_tr)
        
        # Calculate ATR as percentage of current price
        current_price = closes[-1]
        atr_percent = (atr / current_price) * 100
        
        # Check if within threshold (convert to Python bool for JSON serialization)
        can_trade = bool(atr_percent <= self.max_atr_percent)
        
        return can_trade, float(atr_percent)
    
    def get_atr_details(self, symbol: str, price_history: pd.DataFrame) -> dict:
        """
        Get detailed ATR information.
        
        Returns:
            Dictionary with ATR metrics
        """
        can_trade, atr_percent = self.check_volatility(symbol, price_history)
        
        return {
            "symbol": symbol,
            "atr_percent": float(round(atr_percent, 2)),
            "max_allowed_percent": float(self.max_atr_percent),
            "can_trade": bool(can_trade),
            "reason": "Low volatility - safe to trade" if can_trade else f"High volatility ({atr_percent:.1f}%) exceeds threshold ({self.max_atr_percent}%)"
        }

