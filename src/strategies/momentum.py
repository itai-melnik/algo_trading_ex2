"""
Momentum Strategy.

Generates BUY signals when price is significantly higher than N days ago,
and SELL signals when price is significantly lower than N days ago.
"""
import pandas as pd
from .base_strategy import BaseStrategy
from .signals import Signal


class MomentumStrategy(BaseStrategy):
    """
    Momentum strategy implementation.
    
    Args:
        lookback_days: Number of days to look back for comparison
        threshold: Minimum price change percentage to trigger a signal (e.g., 0.05 for 5%)
    """
    
    def __init__(self, lookback_days: int = 10, threshold: float = 0.05):
        if lookback_days < 1:
            raise ValueError("lookback_days must be at least 1")
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        
        self.lookback_days = lookback_days
        self.threshold = threshold
    
    @property
    def name(self) -> str:
        return f"Momentum_{self.lookback_days}d_{int(self.threshold * 100)}pct"
    
    def generate_signal(self, symbol: str, price_history: pd.DataFrame) -> Signal:
        """
        Generate signal based on price momentum.
        
        Args:
            symbol: Stock ticker
            price_history: DataFrame with 'close' column
            
        Returns:
            Signal with BUY/SELL/HOLD action
        """
        # Check for sufficient data
        if len(price_history) <= self.lookback_days:
            return Signal(
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reason=f"Insufficient data: need {self.lookback_days + 1} days, have {len(price_history)}"
            )
        
        closes = price_history['close'].values
        
        # Get current price and price from N days ago
        current_price = closes[-1]
        past_price = closes[-(self.lookback_days + 1)]
        
        # Calculate price change percentage
        price_change_pct = (current_price - past_price) / past_price
        
        # Determine signal based on momentum
        if price_change_pct > self.threshold:
            # Strong upward momentum - BUY signal
            confidence = min(abs(price_change_pct) / (self.threshold * 2), 1.0)
            return Signal(
                symbol=symbol,
                action="BUY",
                confidence=confidence,
                strategy_name=self.name,
                reason=f"Price up {price_change_pct:.1%} over {self.lookback_days} days (threshold: {self.threshold:.1%})"
            )
        elif price_change_pct < -self.threshold:
            # Strong downward momentum - SELL signal
            confidence = min(abs(price_change_pct) / (self.threshold * 2), 1.0)
            return Signal(
                symbol=symbol,
                action="SELL",
                confidence=confidence,
                strategy_name=self.name,
                reason=f"Price down {abs(price_change_pct):.1%} over {self.lookback_days} days (threshold: {self.threshold:.1%})"
            )
        else:
            # Price change within threshold - no clear momentum
            return Signal(
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reason=f"Price change {price_change_pct:.1%} within threshold ({self.threshold:.1%})"
            )

