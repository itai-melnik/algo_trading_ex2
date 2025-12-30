import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from .interface import MarketDataProvider
from .caching import disk_cache
from src.utils.rate_limiter import RateLimiter
from loguru import logger

load_dotenv()

class AlpacaDataProvider(MarketDataProvider):
    def __init__(self, rate_limiter: RateLimiter):
        self.api = tradeapi.REST(
            os.getenv("ALPACA_KEY"),
            os.getenv("ALPACA_SECRET"),
            base_url="https://paper-api.alpaca.markets"
        )
        #Dependency injection for the rate limiter
        self.rate_limiter = rate_limiter

    #Cache history
    @disk_cache
    def get_price_history(self, symbol: str, start_date: str, end_date: str):
        # 1. Check Rate Limit
        self.rate_limiter.wait_for_token()
        
        # 2. Call API
        logger.info(f"Fetching REAL data for {symbol}...") # Debug log to prove caching works
        barset = self.api.get_bars(
            symbol, 
            tradeapi.TimeFrame.Day, 
            start=start_date, 
            end=end_date
        ).df
        
        # 3. Return as simple dict or keep as DF
        return barset.to_dict()

    # DO NOT cache real-time price unless necessary for short windows
    def get_latest_price(self, symbol: str) -> float:
        self.rate_limiter.wait_for_token()
        trade = self.api.get_latest_trade(symbol)
        return float(trade.price)

    def get_account_balance(self) -> float:
        self.rate_limiter.wait_for_token()
        account = self.api.get_account()
        return float(account.cash)