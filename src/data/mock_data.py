from .interface import MarketDataProvider
from datetime import datetime, timedelta
import random
import math


class MockDataProvider(MarketDataProvider):
    """
    A mock data provider for backtesting with zero external dependencies.
    Generates synthetic but realistic price movements.
    """
    
    # Base prices for symbols (can be extended)
    BASE_PRICES = {
        "PLTR": 25.0,
        "NFLX": 400.0,
        "PLTK": 15.0,
        "AAPL": 175.0,
        "GOOGL": 140.0,
    }
    DEFAULT_PRICE = 100.0
    
    def __init__(self, exchange=None, seed=42):
        """
        :param exchange: VirtualExchange instance for order routing
        :param seed: Random seed for reproducible price generation
        """
        self.exchange = exchange
        self.seed = seed
        random.seed(seed)
        
        # Cache generated prices for consistency within a backtest run
        self._price_cache = {}

    def set_exchange(self, exchange):
        """Helper to link exchange after initialization"""
        self.exchange = exchange

    def _generate_price_for_date(self, symbol: str, date_str: str) -> dict:
        """
        Generate synthetic OHLCV data for a symbol on a given date.
        Uses a deterministic random walk based on symbol + date for reproducibility.
        """
        cache_key = f"{symbol}_{date_str}"
        if cache_key in self._price_cache:
            return self._price_cache[cache_key]
        
        # Get base price
        base = self.BASE_PRICES.get(symbol, self.DEFAULT_PRICE)
        
        # Create deterministic seed from symbol + date
        date_seed = hash(f"{symbol}_{date_str}_{self.seed}") % (2**32)
        rng = random.Random(date_seed)
        
        # Calculate days since a reference date for trending behavior
        ref_date = datetime(2023, 1, 1)
        current = datetime.strptime(date_str, "%Y-%m-%d")
        days_elapsed = (current - ref_date).days
        
        # Add a slow trend + daily volatility
        trend = math.sin(days_elapsed / 30) * 0.1  # ~10% oscillation over ~90 days
        daily_change = rng.gauss(0, 0.02)  # 2% daily volatility
        
        multiplier = 1 + trend + daily_change
        close_price = round(base * multiplier, 2)
        
        # Generate OHLC around close
        volatility = close_price * 0.015  # 1.5% intraday range
        open_price = round(close_price + rng.uniform(-volatility, volatility), 2)
        high_price = round(max(open_price, close_price) + rng.uniform(0, volatility), 2)
        low_price = round(min(open_price, close_price) - rng.uniform(0, volatility), 2)
        volume = int(rng.uniform(500000, 2000000))
        
        result = {
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        }
        
        self._price_cache[cache_key] = result
        return result

    def get_price_history(self, symbol: str, start_date: str, end_date: str) -> dict:
        """
        Returns historical price data for a date range.
        Generates synthetic data for each trading day.
        """
        result = {}
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        current = start
        while current <= end:
            # Skip weekends
            if current.weekday() < 5:
                date_str = current.strftime("%Y-%m-%d")
                result[date_str] = self._generate_price_for_date(symbol, date_str)
            current += timedelta(days=1)
        
        return result

    def get_latest_price(self, symbol: str) -> float:
        """Returns the price for the current simulation date."""
        if self.exchange:
            current_date = getattr(self.exchange, 'current_date', None)
            if current_date and current_date != "N/A":
                data = self._generate_price_for_date(symbol, current_date)
                return data['close']
        
        # Fallback to base price
        return self.BASE_PRICES.get(symbol, self.DEFAULT_PRICE)

    def get_account_balance(self) -> float:
        """Returns the actual cash balance from VirtualExchange."""
        if self.exchange:
            return self.exchange.cash
        return 100000.0  # Fallback for testing without exchange

    def execute_order(self, symbol: str, side: str, qty: int) -> str:
        if self.exchange:
            current_sim_date = getattr(self.exchange, 'current_date', 'UNKNOWN_DATE')
            
            self.exchange.execute_trade(
                action=side.upper(), 
                symbol=symbol, 
                quantity=qty, 
                current_date=current_sim_date
            )
            return f"MOCK_FILLED_{symbol}_{qty}"
        else:
            return "ERROR: No Exchange Connected"
