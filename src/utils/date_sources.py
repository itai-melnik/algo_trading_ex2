"""
Date Sources for Trading Loop

Provides iterators that yield dates for both backtest and live trading modes.
This enables dependency injection of the "time source" into the trading loop.
"""
from datetime import datetime, timedelta
import time
from typing import Generator
from zoneinfo import ZoneInfo

from .market_hours import is_trading_day, is_market_open, seconds_until_market_open

ET = ZoneInfo("America/New_York")


def historical_dates(start_date: str, end_date: str) -> Generator[str, None, None]:
    """
    Yields each trading day in the specified date range.
    
    Used for backtesting - iterates quickly without waiting.
    
    :param start_date: Start date in YYYY-MM-DD format
    :param end_date: End date in YYYY-MM-DD format
    :yields: Date strings in YYYY-MM-DD format for each trading day
    
    Example:
        for date in historical_dates("2023-11-01", "2023-11-07"):
            print(date)  # "2023-11-01", "2023-11-02", "2023-11-03", ...
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    current = start
    while current <= end:
        if is_trading_day(current):
            yield current.strftime("%Y-%m-%d")
        current += timedelta(days=1)


def live_dates(interval_seconds: int = 300, max_iterations: int = None) -> Generator[str, None, None]:
    """
    Yields current datetime at specified intervals during market hours.
    
    Used for live trading - waits between iterations and respects market hours.
    
    :param interval_seconds: Seconds between each trading cycle (default: 300 = 5 min)
    :param max_iterations: Optional limit on iterations (for testing)
    :yields: Current datetime string in YYYY-MM-DD HH:MM format
    
    Behavior:
    - If market is open: yields current time, then sleeps for interval
    - If market is closed: waits until market opens, then continues
    """
    from loguru import logger
    
    iterations = 0
    
    while max_iterations is None or iterations < max_iterations:
        now = datetime.now(ET)
        
        if is_market_open(now):
            # Market is open - yield current time
            yield now.strftime("%Y-%m-%d %H:%M")
            iterations += 1
            
            # Sleep until next cycle
            time.sleep(interval_seconds)
        else:
            # Market is closed - wait for it to open
            wait_seconds = seconds_until_market_open(now)
            
            if wait_seconds > 0:
                hours = wait_seconds // 3600
                minutes = (wait_seconds % 3600) // 60
                logger.info(f"Market closed. Waiting {hours}h {minutes}m until market opens...")
                
                # Sleep in chunks to allow for graceful shutdown
                sleep_chunk = 60  # Check every minute
                while wait_seconds > 0:
                    time.sleep(min(sleep_chunk, wait_seconds))
                    wait_seconds -= sleep_chunk


def simulated_live_dates(
    start_date: str, 
    end_date: str, 
    interval_minutes: int = 30
) -> Generator[str, None, None]:
    """
    Simulates intraday trading by yielding multiple times per day.
    
    Useful for testing intraday strategies on historical data.
    
    :param start_date: Start date in YYYY-MM-DD format
    :param end_date: End date in YYYY-MM-DD format
    :param interval_minutes: Minutes between each simulated check (default: 30)
    :yields: Datetime strings in YYYY-MM-DD HH:MM format
    """
    from .market_hours import MARKET_OPEN, MARKET_CLOSE
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    current_day = start
    while current_day <= end:
        if is_trading_day(current_day):
            # Start at market open
            current_time = current_day.replace(
                hour=MARKET_OPEN.hour, 
                minute=MARKET_OPEN.minute
            )
            market_close_dt = current_day.replace(
                hour=MARKET_CLOSE.hour, 
                minute=MARKET_CLOSE.minute
            )
            
            while current_time < market_close_dt:
                yield current_time.strftime("%Y-%m-%d %H:%M")
                current_time += timedelta(minutes=interval_minutes)
        
        current_day += timedelta(days=1)

