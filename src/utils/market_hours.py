"""
Market Hours Utilities

Helpers for checking if US stock market is open.
Market hours: 9:30 AM - 4:00 PM Eastern Time, Monday-Friday
"""
from datetime import datetime, time
from zoneinfo import ZoneInfo

# US Eastern timezone
ET = ZoneInfo("America/New_York")

# Market hours (Eastern Time)
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)


def is_trading_day(date: datetime) -> bool:
    """
    Check if a given date is a trading day (weekday).
    
    Note: This does not account for market holidays.
    For production, consider using a holiday calendar library.
    
    :param date: datetime object to check
    :return: True if Monday-Friday, False otherwise
    """
    return date.weekday() < 5  # 0=Monday, 4=Friday


def is_market_open(dt: datetime = None) -> bool:
    """
    Check if US stock market is currently open.
    
    :param dt: datetime to check (defaults to current time)
    :return: True if market is open, False otherwise
    """
    if dt is None:
        dt = datetime.now(ET)
    else:
        # Ensure we're working in Eastern time
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ET)
        else:
            dt = dt.astimezone(ET)
    
    # Check if it's a weekday
    if not is_trading_day(dt):
        return False
    
    # Check if within market hours
    current_time = dt.time()
    return MARKET_OPEN <= current_time < MARKET_CLOSE


def get_next_market_open(dt: datetime = None) -> datetime:
    """
    Get the next market open time.
    
    :param dt: Starting datetime (defaults to now)
    :return: datetime of next market open
    """
    from datetime import timedelta
    
    if dt is None:
        dt = datetime.now(ET)
    else:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ET)
        else:
            dt = dt.astimezone(ET)
    
    # If market is currently open, return current open time today
    if is_market_open(dt):
        return dt.replace(hour=MARKET_OPEN.hour, minute=MARKET_OPEN.minute, second=0, microsecond=0)
    
    # Check if we're before market open today
    if is_trading_day(dt) and dt.time() < MARKET_OPEN:
        return dt.replace(hour=MARKET_OPEN.hour, minute=MARKET_OPEN.minute, second=0, microsecond=0)
    
    # Find next trading day
    next_day = dt + timedelta(days=1)
    while not is_trading_day(next_day):
        next_day += timedelta(days=1)
    
    return next_day.replace(hour=MARKET_OPEN.hour, minute=MARKET_OPEN.minute, second=0, microsecond=0)


def seconds_until_market_open(dt: datetime = None) -> int:
    """
    Get seconds until market opens.
    
    :param dt: Starting datetime (defaults to now)
    :return: Seconds until market open (0 if market is open)
    """
    if dt is None:
        dt = datetime.now(ET)
    
    if is_market_open(dt):
        return 0
    
    next_open = get_next_market_open(dt)
    delta = next_open - dt
    return int(delta.total_seconds())

