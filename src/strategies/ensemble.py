"""
Ensemble Strategy.

Combines multiple strategies with weighted voting to produce
a final signal.
"""
import pandas as pd
from typing import List, Tuple
from .base_strategy import BaseStrategy
from .signals import Signal
from .volatility_filter import VolatilityFilter


class EnsembleStrategy(BaseStrategy):
    """
    Ensemble strategy that combines multiple strategies with weighted voting.
    
    Args:
        strategies: List of (strategy, weight) tuples
        volatility_filter: Optional VolatilityFilter to suppress signals in high volatility
        min_confidence: Minimum confidence threshold to generate a non-HOLD signal
    """
    
    def __init__(
        self, 
        strategies: List[Tuple[BaseStrategy, float]],
        volatility_filter: VolatilityFilter = None,
        min_confidence: float = 0.3
    ):
        if not strategies:
            raise ValueError("At least one strategy is required")
        
        self.strategies = strategies
        self.volatility_filter = volatility_filter
        self.min_confidence = min_confidence
    
    @property
    def name(self) -> str:
        strategy_names = [s.name for s, _ in self.strategies]
        return f"Ensemble[{','.join(strategy_names)}]"
    
    def get_all_signals(self, symbol: str, price_history: pd.DataFrame) -> List[Signal]:
        """
        Get individual signals from all strategies.
        
        Args:
            symbol: Stock ticker
            price_history: DataFrame with OHLCV data
            
        Returns:
            List of Signal objects from each strategy
        """
        signals = []
        for strategy, _ in self.strategies:
            try:
                signal = strategy.generate_signal(symbol, price_history)
                signals.append(signal)
            except Exception as e:
                # If a strategy fails, add a HOLD signal
                signals.append(Signal(
                    symbol=symbol,
                    action="HOLD",
                    confidence=0.0,
                    strategy_name=strategy.name,
                    reason=f"Error: {str(e)}"
                ))
        return signals
    
    def generate_signal(self, symbol: str, price_history: pd.DataFrame) -> Signal:
        """
        Generate a combined signal from all strategies.
        
        Uses weighted voting to determine the final action.
        
        Args:
            symbol: Stock ticker
            price_history: DataFrame with OHLCV data
            
        Returns:
            Combined Signal with weighted confidence
        """
        # Check volatility filter first
        if self.volatility_filter:
            can_trade, atr_pct = self.volatility_filter.check_volatility(symbol, price_history)
            if not can_trade:
                return Signal(
                    symbol=symbol,
                    action="HOLD",
                    confidence=0.0,
                    strategy_name=self.name,
                    reason=f"Volatility too high (ATR: {atr_pct:.1f}%) - suppressing signals"
                )
        
        # Collect all signals
        all_signals = self.get_all_signals(symbol, price_history)
        
        # Calculate weighted scores for each action
        scores = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
        total_weight = 0.0
        reasons = []
        
        for (strategy, weight), signal in zip(self.strategies, all_signals):
            # Weight the confidence by the strategy weight
            weighted_confidence = signal.confidence * weight
            scores[signal.action] += weighted_confidence
            total_weight += weight
            
            if signal.action != "HOLD":
                reasons.append(f"{strategy.name}: {signal.action} ({signal.confidence:.0%})")
        
        # Normalize scores
        if total_weight > 0:
            for action in scores:
                scores[action] /= total_weight
        
        # Determine winning action
        best_action = max(scores, key=scores.get)
        best_confidence = scores[best_action]
        
        # Check if confidence meets minimum threshold
        if best_action != "HOLD" and best_confidence < self.min_confidence:
            return Signal(
                symbol=symbol,
                action="HOLD",
                confidence=best_confidence,
                strategy_name=self.name,
                reason=f"Confidence {best_confidence:.0%} below threshold {self.min_confidence:.0%}"
            )
        
        # Build reason string
        if reasons:
            combined_reason = "; ".join(reasons)
        else:
            combined_reason = "All strategies indicate HOLD"
        
        return Signal(
            symbol=symbol,
            action=best_action,
            confidence=best_confidence,
            strategy_name=self.name,
            reason=combined_reason
        )
    
    def get_detailed_analysis(self, symbol: str, price_history: pd.DataFrame) -> dict:
        """
        Get detailed analysis from all strategies.
        
        Returns:
            Dictionary with individual signals and combined result
        """
        all_signals = self.get_all_signals(symbol, price_history)
        combined = self.generate_signal(symbol, price_history)
        
        volatility_info = None
        if self.volatility_filter:
            volatility_info = self.volatility_filter.get_atr_details(symbol, price_history)
        
        return {
            "symbol": symbol,
            "individual_signals": [s.to_dict() for s in all_signals],
            "combined_signal": combined.to_dict(),
            "volatility": volatility_info
        }

