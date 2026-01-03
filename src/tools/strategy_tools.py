"""
Strategy Tools for CrewAI agents.

These tools wrap the StrategyEngine to provide LLM-callable interfaces
that return structured signals. All heavy computation is done in the
strategy layer, not by the LLM.
"""
import json
import pandas as pd
from typing import Any
from datetime import datetime, timedelta
from crewai.tools import BaseTool
from pydantic import Field

from src.data.interface import MarketDataProvider
from src.strategies import (
    EnsembleStrategy,
    SMACrossoverStrategy,
    MomentumStrategy,
    MeanReversionStrategy,
    VolatilityFilter,
)


class GenerateSignalsTool(BaseTool):
    """
    Tool that generates trading signals using the ensemble strategy.
    
    This tool does ALL the computation - the LLM just interprets the result.
    """
    name: str = "Generate Trading Signals"
    description: str = """
    Analyzes a stock and returns a trading signal (BUY/SELL/HOLD) with confidence.
    Input: SYMBOL (e.g., 'PLTR')
    Output: JSON with action, confidence, and detailed reasoning from multiple strategies.
    """
    
    provider: MarketDataProvider = Field(exclude=True)
    exchange: Any = Field(default=None, exclude=True)
    lookback_days: int = Field(default=30, exclude=True)
    
    # Pre-configured ensemble strategy
    _ensemble: EnsembleStrategy = None
    
    def __init__(self, **data):
        super().__init__(**data)
        # Build the ensemble with default strategies
        self._ensemble = EnsembleStrategy(
            strategies=[
                (SMACrossoverStrategy(fast_period=5, slow_period=20), 1.0),
                (MomentumStrategy(lookback_days=10, threshold=0.03), 1.0),
                (MeanReversionStrategy(period=20, num_std=2.0), 0.8),
            ],
            volatility_filter=VolatilityFilter(atr_period=14, max_atr_percent=8.0),
            min_confidence=0.2
        )
    
    def _run(self, symbol: str) -> str:
        try:
            symbol = symbol.strip().upper()
            
            # Get current date from exchange or use today
            if self.exchange:
                current_date = getattr(self.exchange, 'current_date', None)
                if current_date and current_date != "N/A":
                    end_date = current_date
                else:
                    end_date = datetime.now().strftime("%Y-%m-%d")
            else:
                end_date = datetime.now().strftime("%Y-%m-%d")
            
            # Calculate start date
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            start_dt = end_dt - timedelta(days=self.lookback_days + 10)  # Extra buffer
            start_date = start_dt.strftime("%Y-%m-%d")
            
            # Fetch price history
            price_data = self.provider.get_price_history(symbol, start_date, end_date)
            
            if not price_data:
                return json.dumps({
                    "symbol": symbol,
                    "action": "HOLD",
                    "confidence": 0.0,
                    "reason": "No price data available"
                })
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(price_data, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Generate detailed analysis
            analysis = self._ensemble.get_detailed_analysis(symbol, df)
            
            return json.dumps(analysis, indent=2)
            
        except Exception as e:
            return json.dumps({
                "symbol": symbol,
                "action": "HOLD",
                "confidence": 0.0,
                "reason": f"Error generating signals: {str(e)}"
            })


class RiskFilterTool(BaseTool):
    """
    Tool that applies risk management rules to a proposed trade.
    """
    name: str = "Risk Filter"
    description: str = """
    Checks if a proposed trade meets risk management rules.
    Input: 'SYMBOL|ACTION|CONFIDENCE' (e.g., 'PLTR|BUY|0.75')
    Output: JSON with approval status and position constraints.
    
    Risk Rules:
    - Never invest more than 20% of portfolio in a single position
    - Keep at least 10% of portfolio in cash
    - Require minimum 30% confidence for any trade
    """
    
    provider: MarketDataProvider = Field(exclude=True)
    exchange: Any = Field(default=None, exclude=True)
    
    # Risk parameters
    max_position_pct: float = Field(default=0.20, exclude=True)
    min_cash_reserve_pct: float = Field(default=0.10, exclude=True)
    min_confidence: float = Field(default=0.30, exclude=True)
    
    def _run(self, trade_input: str) -> str:
        try:
            # Parse input
            parts = [p.strip() for p in trade_input.split('|')]
            if len(parts) != 3:
                return json.dumps({
                    "approved": False,
                    "reason": "Invalid input format. Use 'SYMBOL|ACTION|CONFIDENCE'"
                })
            
            symbol, action, confidence_str = parts
            symbol = symbol.upper()
            action = action.upper()
            
            try:
                confidence = float(confidence_str)
            except ValueError:
                return json.dumps({
                    "approved": False,
                    "reason": f"Invalid confidence value: {confidence_str}"
                })
            
            # Get account info
            if self.exchange:
                cash = self.exchange.cash
                portfolio_value = self.exchange.get_total_portfolio_value(
                    getattr(self.exchange, 'current_date', datetime.now().strftime("%Y-%m-%d"))
                )
                current_holdings = self.exchange.holdings.get(symbol, 0)
            else:
                cash = self.provider.get_account_balance()
                portfolio_value = cash  # Simplified
                current_holdings = 0
            
            # Get current price
            current_price = self.provider.get_latest_price(symbol)
            
            # Apply risk rules
            issues = []
            
            # Rule 1: Minimum confidence
            if confidence < self.min_confidence:
                issues.append(f"Confidence {confidence:.0%} below minimum {self.min_confidence:.0%}")
            
            # Rule 2: Cash reserve
            min_cash = portfolio_value * self.min_cash_reserve_pct
            available_cash = cash - min_cash
            if available_cash <= 0:
                issues.append(f"Cash ${cash:.2f} at or below minimum reserve ${min_cash:.2f}")
                available_cash = 0
            
            # Rule 3: Max position size
            max_position_value = portfolio_value * self.max_position_pct
            current_position_value = current_holdings * current_price
            
            if action == "BUY":
                remaining_allocation = max_position_value - current_position_value
                max_buy_value = min(available_cash, remaining_allocation)
                max_shares = int(max_buy_value / current_price) if current_price > 0 else 0
                
                if max_shares <= 0:
                    if remaining_allocation <= 0:
                        issues.append(f"Position already at max allocation (${current_position_value:.2f} of ${max_position_value:.2f})")
                    else:
                        issues.append("Insufficient available cash after reserves")
            elif action == "SELL":
                max_shares = current_holdings
                if max_shares <= 0:
                    issues.append(f"No {symbol} shares to sell")
            else:
                max_shares = 0
            
            # Build response
            approved = len(issues) == 0 and max_shares > 0
            
            return json.dumps({
                "approved": approved,
                "symbol": symbol,
                "action": action,
                "max_shares": max_shares,
                "current_price": current_price,
                "max_trade_value": max_shares * current_price,
                "portfolio_value": portfolio_value,
                "available_cash": available_cash,
                "current_position_shares": current_holdings,
                "issues": issues if issues else None,
                "reason": "Trade approved within risk limits" if approved else "; ".join(issues)
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                "approved": False,
                "reason": f"Error in risk check: {str(e)}"
            })


class PositionSizeTool(BaseTool):
    """
    Tool that calculates optimal position size based on risk parameters.
    """
    name: str = "Calculate Position Size"
    description: str = """
    Calculates the optimal number of shares to trade based on confidence and risk.
    Input: 'SYMBOL|ACTION|CONFIDENCE|MAX_SHARES' (e.g., 'PLTR|BUY|0.75|100')
    Output: Recommended quantity and trade details.
    
    Uses confidence to scale position:
    - High confidence (>80%): Full max allocation
    - Medium confidence (50-80%): 50-80% of max
    - Low confidence (30-50%): 30-50% of max
    """
    
    provider: MarketDataProvider = Field(exclude=True)
    
    def _run(self, size_input: str) -> str:
        try:
            # Parse input
            parts = [p.strip() for p in size_input.split('|')]
            if len(parts) != 4:
                return json.dumps({
                    "error": "Invalid input. Use 'SYMBOL|ACTION|CONFIDENCE|MAX_SHARES'"
                })
            
            symbol, action, confidence_str, max_shares_str = parts
            symbol = symbol.upper()
            action = action.upper()
            
            try:
                confidence = float(confidence_str)
                max_shares = int(max_shares_str)
            except ValueError:
                return json.dumps({
                    "error": "Invalid confidence or max_shares value"
                })
            
            # Scale position by confidence
            if confidence >= 0.8:
                position_scale = 1.0
            elif confidence >= 0.5:
                position_scale = 0.5 + (confidence - 0.5) * 1.0  # 0.5 to 0.8
            elif confidence >= 0.3:
                position_scale = 0.3 + (confidence - 0.3) * 1.0  # 0.3 to 0.5
            else:
                position_scale = 0.0
            
            recommended_shares = int(max_shares * position_scale)
            
            # Get current price
            current_price = self.provider.get_latest_price(symbol)
            trade_value = recommended_shares * current_price
            
            return json.dumps({
                "symbol": symbol,
                "action": action,
                "recommended_shares": recommended_shares,
                "max_shares": max_shares,
                "position_scale": f"{position_scale:.0%}",
                "confidence": f"{confidence:.0%}",
                "current_price": current_price,
                "estimated_trade_value": trade_value,
                "reason": f"Scaling to {position_scale:.0%} of max based on {confidence:.0%} confidence"
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                "error": f"Error calculating position size: {str(e)}"
            })

