"""
Mean Reversion Strategy (Bollinger Bands style).

Generates BUY signals when price is below lower Bollinger Band,
and SELL signals when price is above upper Bollinger Band.
"""
import pandas as pd
import math
from .base_strategy import BaseStrategy
from .signals import Signal


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion strategy using Bollinger Bands.
    
    Args:
        period: Number of days for the moving average
        num_std: Number of standard deviations for the bands
    """
    
    def __init__(self, period: int = 20, num_std: float = 2.0):
        if period < 2:
            raise ValueError("period must be at least 2")
        if num_std <= 0:
            raise ValueError("num_std must be positive")
        
        self.period = period
        self.num_std = num_std
    
    @property
    def name(self) -> str:
        return f"MeanReversion_BB{self.period}_{self.num_std}std"
    
    def generate_signal(self, symbol: str, price_history: pd.DataFrame) -> Signal:
        """
        Generate signal based on Bollinger Bands.
        
        Args:
            symbol: Stock ticker
            price_history: DataFrame with 'close' column
            
        Returns:
            Signal with BUY/SELL/HOLD action
        """
        # Check for sufficient data
        if len(price_history) < self.period:
            return Signal(
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reason=f"Insufficient data: need {self.period} days, have {len(price_history)}"
            )
        
        closes = price_history['close'].values
        
        # Get the last 'period' prices for calculation
        recent_prices = closes[-self.period:]
        current_price = closes[-1]
        
        # Calculate moving average and standard deviation
        sma = sum(recent_prices) / self.period
        variance = sum((p - sma) ** 2 for p in recent_prices) / self.period
        std_dev = math.sqrt(variance)
        
        # Calculate Bollinger Bands
        upper_band = sma + (self.num_std * std_dev)
        lower_band = sma - (self.num_std * std_dev)
        
        # Handle edge case where std_dev is 0 (all prices identical)
        if std_dev == 0:
            return Signal(
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reason=f"No price variation in last {self.period} days"
            )
        
        # Calculate how far price is from the bands (in std devs)
        z_score = (current_price - sma) / std_dev
        
        # Determine signal based on band position
        if current_price < lower_band:
            # Price below lower band - oversold, BUY signal
            distance = abs(z_score) - self.num_std
            confidence = min(distance / self.num_std + 0.5, 1.0)
            return Signal(
                symbol=symbol,
                action="BUY",
                confidence=confidence,
                strategy_name=self.name,
                reason=f"Price ${current_price:.2f} below lower band ${lower_band:.2f} (z-score: {z_score:.2f})"
            )
        elif current_price > upper_band:
            # Price above upper band - overbought, SELL signal
            distance = abs(z_score) - self.num_std
            confidence = min(distance / self.num_std + 0.5, 1.0)
            return Signal(
                symbol=symbol,
                action="SELL",
                confidence=confidence,
                strategy_name=self.name,
                reason=f"Price ${current_price:.2f} above upper band ${upper_band:.2f} (z-score: {z_score:.2f})"
            )
        else:
            # Price within bands - HOLD
            return Signal(
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reason=f"Price ${current_price:.2f} within bands [${lower_band:.2f}, ${upper_band:.2f}]"
            )

