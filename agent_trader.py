#!/usr/bin/env python3
"""
Agent Trader - Unified Trading Script

Runs AI trading agents in two modes:
- LIVE: Real-time paper trading with Alpaca
- BACKTEST: Historical simulation with mock or real data

Usage:
    # Live paper trading (runs until Ctrl+C)
    python agent_trader.py --live --interval 300

    # Backtest on historical week with mock data
    python agent_trader.py --backtest --start 2023-11-01 --end 2023-11-07

    # Backtest with real Alpaca historical data
    python agent_trader.py --backtest --start 2023-11-01 --end 2023-11-07 --real-data
"""
import argparse
import signal
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables
script_dir = Path(__file__).parent
load_dotenv(dotenv_path=script_dir / '.env')

# Configure loguru
LOG_DIR = script_dir / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Remove default handler and add custom ones
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO"
)
logger.add(
    LOG_DIR / "agent_trader_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    level="DEBUG",
    rotation="1 day",
    retention="30 days"
)

# Global flag for graceful shutdown
running = True


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global running
    logger.warning("Shutdown signal received. Finishing current cycle...")
    running = False


def run_backtest_mode(args):
    """
    Run backtest with historical dates.
    
    Uses MockDataProvider + VirtualExchange for local simulation,
    or AlpacaDataProvider + VirtualExchange for real historical data.
    """
    from src.crew import TradingCrew
    from src.simulation.virtual_exchange import VirtualExchange
    from src.utils.date_sources import historical_dates
    
    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    logger.info("=" * 60)
    logger.info("BACKTEST MODE")
    logger.info(f"Period: {args.start} to {args.end}")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Data Source: {'Alpaca (Real Historical)' if args.real_data else 'Mock (Synthetic)'}")
    logger.info(f"Initial Cash: ${args.cash:,.2f}")
    logger.info("=" * 60)
    
    # Setup provider based on --real-data flag
    if args.real_data:
        from src.data.real_data import AlpacaDataProvider
        from src.utils.rate_limiter import RateLimiter
        
        limiter = RateLimiter(max_calls=45, period_seconds=60)
        provider = AlpacaDataProvider(rate_limiter=limiter)
        logger.info("Using Alpaca for historical price data")
    else:
        from src.data.mock_data import MockDataProvider
        
        provider = MockDataProvider(seed=args.seed)
        logger.info(f"Using Mock data with seed={args.seed}")
    
    # Setup VirtualExchange for simulated trading
    exchange = VirtualExchange(initial_cash=args.cash, provider=provider)
    
    # Link exchange to provider (for MockDataProvider)
    if hasattr(provider, 'set_exchange'):
        provider.set_exchange(exchange)
    
    # Run through historical dates
    cycle_count = 0
    for current_date in historical_dates(args.start, args.end):
        if not running:
            break
            
        cycle_count += 1
        exchange.update_date(current_date)
        
        logger.info(f"--- Trading Cycle #{cycle_count}: {current_date} ---")
        
        # Log prices for the day
        for sym in symbols:
            try:
                price_data = provider.get_price_history(sym, current_date, current_date)
                if current_date in price_data:
                    p = price_data[current_date]
                    logger.debug(f"{sym}: O=${p['open']:.2f} H=${p['high']:.2f} L=${p['low']:.2f} C=${p['close']:.2f}")
            except Exception as e:
                logger.warning(f"Could not get price for {sym}: {e}")
        
        # Build and run the crew
        try:
            trading_bot = TradingCrew(provider=provider, exchange=exchange)
            crew = trading_bot.build_crew(current_date=current_date, stock_selection=symbols)
            crew.kickoff()
        except Exception as e:
            logger.error(f"Crew execution failed: {e}")
        
        # Log portfolio status
        portfolio_value = exchange.get_total_portfolio_value(current_date)
        logger.info(f"Portfolio Value: ${portfolio_value:,.2f} | Cash: ${exchange.cash:,.2f} | Holdings: {exchange.holdings}")
    
    # Final summary
    _print_summary(exchange, args.cash, args.end, "BACKTEST")


def run_live_mode(args):
    """
    Run live paper trading with Alpaca.
    
    Continuously runs until Ctrl+C, executing trades at specified intervals.
    Only trades during market hours.
    """
    from src.crew import TradingCrew
    from src.data.real_data import AlpacaDataProvider
    from src.utils.rate_limiter import RateLimiter
    from src.utils.date_sources import live_dates
    
    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    logger.info("=" * 60)
    logger.info("LIVE PAPER TRADING MODE")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Interval: {args.interval} seconds")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)
    
    # Setup Alpaca provider (no VirtualExchange - Alpaca handles portfolio)
    limiter = RateLimiter(max_calls=45, period_seconds=60)
    provider = AlpacaDataProvider(rate_limiter=limiter)
    
    # Get initial balance from Alpaca
    try:
        initial_balance = provider.get_account_balance()
        logger.info(f"Alpaca Account Balance: ${initial_balance:,.2f}")
    except Exception as e:
        logger.error(f"Could not connect to Alpaca: {e}")
        logger.error("Check your ALPACA_KEY and ALPACA_SECRET in .env")
        return
    
    # Run trading loop
    cycle_count = 0
    for current_datetime in live_dates(interval_seconds=args.interval):
        if not running:
            break
            
        cycle_count += 1
        current_date = current_datetime.split()[0]  # Extract date part
        
        logger.info(f"--- Trading Cycle #{cycle_count}: {current_datetime} ---")
        
        # Build and run the crew
        try:
            trading_bot = TradingCrew(provider=provider, exchange=None)
            crew = trading_bot.build_crew(current_date=current_date, stock_selection=symbols)
            crew.kickoff()
        except Exception as e:
            logger.error(f"Crew execution failed: {e}")
        
        # Log portfolio status from Alpaca
        try:
            balance = provider.get_account_balance()
            logger.info(f"Alpaca Account Balance: ${balance:,.2f}")
        except Exception as e:
            logger.warning(f"Could not fetch account balance: {e}")
    
    # Final summary
    logger.info("=" * 60)
    logger.info("LIVE TRADING SESSION ENDED")
    logger.info(f"Total Cycles Executed: {cycle_count}")
    try:
        final_balance = provider.get_account_balance()
        logger.info(f"Final Account Balance: ${final_balance:,.2f}")
    except Exception:
        pass
    logger.info("=" * 60)


def _print_summary(exchange, initial_cash: float, end_date: str, mode: str):
    """Print final trading session summary."""
    final_value = exchange.get_total_portfolio_value(end_date)
    profit_loss = final_value - initial_cash
    pct_return = (profit_loss / initial_cash) * 100
    
    logger.info("=" * 60)
    logger.info(f"{mode} COMPLETE")
    logger.info(f"Final Portfolio Value: ${final_value:,.2f}")
    logger.info(f"Cash Remaining: ${exchange.cash:,.2f}")
    logger.info(f"Holdings: {exchange.holdings}")
    logger.info(f"P&L: ${profit_loss:+,.2f} ({pct_return:+.2f}%)")
    logger.info(f"Total Transactions: {len(exchange.transaction_log)}")
    logger.info("=" * 60)
    
    if exchange.transaction_log:
        logger.info("Transaction Log:")
        for tx in exchange.transaction_log:
            logger.info(f"  {tx['date']} | {tx['action']:4} | {tx['qty']:4} x {tx['symbol']:5} @ ${tx['price']:.2f}")


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Agent Trader - AI-powered trading with CrewAI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent_trader.py --live --interval 300
  python agent_trader.py --backtest --start 2023-11-01 --end 2023-11-07
  python agent_trader.py --backtest --start 2023-11-01 --end 2023-11-07 --real-data
        """
    )
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--live', action='store_true', help='Run live paper trading')
    mode_group.add_argument('--backtest', action='store_true', help='Run historical backtest')
    
    # Common arguments
    parser.add_argument('--symbols', type=str, default='PLTR,NFLX',
                        help='Comma-separated list of symbols (default: PLTR,NFLX)')
    
    # Live mode arguments
    parser.add_argument('--interval', type=int, default=300,
                        help='Seconds between trading cycles in live mode (default: 300)')
    
    # Backtest mode arguments
    parser.add_argument('--start', type=str, help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='Backtest end date (YYYY-MM-DD)')
    parser.add_argument('--real-data', action='store_true',
                        help='Use real Alpaca historical data instead of mock')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for mock data generation (default: 42)')
    parser.add_argument('--cash', type=float, default=100000.0,
                        help='Initial cash for backtest (default: 100000)')
    
    args = parser.parse_args()
    
    # Validate backtest arguments
    if args.backtest:
        if not args.start or not args.end:
            parser.error("--backtest requires --start and --end dates")
        try:
            datetime.strptime(args.start, "%Y-%m-%d")
            datetime.strptime(args.end, "%Y-%m-%d")
        except ValueError:
            parser.error("Dates must be in YYYY-MM-DD format")
    
    # Setup signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run appropriate mode
    if args.live:
        run_live_mode(args)
    else:
        run_backtest_mode(args)


if __name__ == "__main__":
    main()

