import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from .interface import MarketDataProvider
from .caching import disk_cache
from src.utils.rate_limiter import RateLimiter
from loguru import logger

load_dotenv()


class AlpacaDataProvider(MarketDataProvider):
    """
    Real data provider for Alpaca paper/live trading.
    All orders go directly to Alpaca - no VirtualExchange needed.
    Alpaca paper trading already simulates the exchange for you.
    """
    
    def __init__(self, rate_limiter: RateLimiter):
        self.api = tradeapi.REST(
            os.getenv("ALPACA_KEY"),
            os.getenv("ALPACA_SECRET"),
            base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        )
        self.rate_limiter = rate_limiter

    @disk_cache
    def get_price_history(self, symbol: str, start_date: str, end_date: str):
        """Fetch historical OHLCV data from Alpaca."""
        self.rate_limiter.wait_for_token()
        
        logger.info(f"Fetching REAL data for {symbol}...")
        bars = self.api.get_bars(
            symbol,
            tradeapi.TimeFrame.Day,
            start=start_date,
            end=end_date
        ).df

        if bars.empty:
            return {}
        
        bars.index = bars.index.strftime('%Y-%m-%d')
        return bars.to_dict(orient='index')

    def get_latest_price(self, symbol: str) -> float:
        """Get real-time price from Alpaca."""
        self.rate_limiter.wait_for_token()
        trade = self.api.get_latest_trade(symbol)
        return float(trade.price)

    def get_account_balance(self) -> float:
        """Get actual cash balance from Alpaca account."""
        self.rate_limiter.wait_for_token()
        account = self.api.get_account()
        return float(account.cash)

    def execute_order(self, symbol: str, side: str, qty: int) -> str:
        """
        Execute order directly on Alpaca (paper or live).
        Alpaca handles all the exchange simulation for paper trading.
        """
        self.rate_limiter.wait_for_token()
        logger.info(f"[ALPACA] Sending {side} order for {qty} {symbol}...")
        
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side.lower(),  # Alpaca expects lowercase
                type='market',
                time_in_force='gtc'
            )
            logger.info(f"[ALPACA] Order submitted: {order.id}")
            return str(order.id)
        except Exception as e:
            logger.error(f"[ALPACA] Order failed: {e}")
            return f"Error: {str(e)}"
