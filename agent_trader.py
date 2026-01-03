#!/usr/bin/env python3
"""
Agent Trader - Unified Trading Script

Runs trading in three modes:
- FAST BACKTEST (default): Pure Python strategy engine, no LLM calls
- CREW BACKTEST: CrewAI with LLM orchestration
- LIVE: Real-time paper trading with Alpaca

Usage:
    # Fast backtest (default, no LLM - very fast!)
    python agent_trader.py --backtest --start 2023-11-01 --end 2023-11-07

    # Fast backtest with real Alpaca historical data
    python agent_trader.py --backtest --start 2023-11-01 --end 2023-11-07 --real-data

    # Crew backtest with LLM orchestration (slower)
    python agent_trader.py --backtest --start 2023-11-01 --end 2023-11-07 --crew

    # Live paper trading (runs until Ctrl+C)
    python agent_trader.py --live --interval 300
"""
import argparse
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
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

# Risk Management Parameters
MAX_POSITION_PCT = 0.20      # Max 20% of portfolio in one position
MIN_CASH_RESERVE_PCT = 0.10  # Keep 10% cash reserve
MIN_CONFIDENCE = 0.30        # Minimum confidence to trade


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global running
    logger.warning("Shutdown signal received. Finishing current cycle...")
    running = False


# ==============================================================================
# HELPER FUNCTIONS FOR FAST BACKTEST
# ==============================================================================

def get_price_df(provider, symbol: str, end_date: str, lookback_days: int = 30) -> pd.DataFrame:
    """Fetch price history and convert to DataFrame."""
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=lookback_days + 10)
    start_date = start_dt.strftime("%Y-%m-%d")
    
    price_data = provider.get_price_history(symbol, start_date, end_date)
    
    if not price_data:
        return pd.DataFrame()
    
    df = pd.DataFrame.from_dict(price_data, orient='index')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def apply_risk_rules(signal, exchange, provider, current_date: str) -> dict:
    """
    Apply risk management rules to a signal.
    
    Returns:
        dict with 'approved', 'max_shares', and 'reason'
    """
    symbol = signal.symbol
    action = signal.action
    confidence = signal.confidence
    
    # Get portfolio state
    cash = exchange.cash
    portfolio_value = exchange.get_total_portfolio_value(current_date)
    current_holdings = exchange.holdings.get(symbol, 0)
    current_price = provider.get_latest_price(symbol)
    
    # Rule 1: Minimum confidence
    if confidence < MIN_CONFIDENCE:
        return {
            'approved': False,
            'max_shares': 0,
            'reason': f"Confidence {confidence:.0%} below minimum {MIN_CONFIDENCE:.0%}"
        }
    
    # Rule 2: Cash reserve
    min_cash = portfolio_value * MIN_CASH_RESERVE_PCT
    available_cash = cash - min_cash
    if available_cash <= 0:
        return {
            'approved': False,
            'max_shares': 0,
            'reason': f"Cash ${cash:.2f} at minimum reserve"
        }
    
    # Rule 3: Max position size
    max_position_value = portfolio_value * MAX_POSITION_PCT
    current_position_value = current_holdings * current_price
    
    if action == "BUY":
        remaining_allocation = max_position_value - current_position_value
        max_buy_value = min(available_cash, remaining_allocation)
        max_shares = int(max_buy_value / current_price) if current_price > 0 else 0
        
        if max_shares <= 0:
            return {
                'approved': False,
                'max_shares': 0,
                'reason': "Position at max allocation or insufficient cash"
            }
    elif action == "SELL":
        max_shares = current_holdings
        if max_shares <= 0:
            return {
                'approved': False,
                'max_shares': 0,
                'reason': f"No {symbol} shares to sell"
            }
    else:
        return {
            'approved': False,
            'max_shares': 0,
            'reason': "HOLD signal - no trade needed"
        }
    
    return {
        'approved': True,
        'max_shares': max_shares,
        'reason': "Trade approved within risk limits"
    }


def calculate_position_size(confidence: float, max_shares: int) -> int:
    """Scale position size based on confidence."""
    if confidence >= 0.8:
        scale = 1.0
    elif confidence >= 0.5:
        scale = 0.5 + (confidence - 0.5) * 1.0
    elif confidence >= 0.3:
        scale = 0.3 + (confidence - 0.3) * 1.0
    else:
        scale = 0.0
    
    return int(max_shares * scale)


# ==============================================================================
# FAST BACKTEST MODE: Pure Python Strategy Engine (No LLM)
# ==============================================================================

def run_fast_backtest_mode(args):
    """
    Run backtest using pure Python strategy engine - no LLM calls.
    
    This is 10-100x faster than crew mode since all computation
    is done in Python without any LLM API calls.
    """
    from src.simulation.virtual_exchange import VirtualExchange
    from src.utils.date_sources import historical_dates
    from src.strategies import (
        EnsembleStrategy,
        SMACrossoverStrategy,
        MomentumStrategy,
        MeanReversionStrategy,
        VolatilityFilter,
    )
    
    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    logger.info("=" * 60)
    logger.info("FAST BACKTEST MODE (No LLM)")
    logger.info("=" * 60)
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
    
    # Build ensemble strategy
    ensemble = EnsembleStrategy(
        strategies=[
            (SMACrossoverStrategy(fast_period=5, slow_period=20), 1.0),
            (MomentumStrategy(lookback_days=10, threshold=0.03), 1.0),
            (MeanReversionStrategy(period=20, num_std=2.0), 0.8),
        ],
        volatility_filter=VolatilityFilter(atr_period=14, max_atr_percent=8.0),
        min_confidence=0.2
    )
    
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
                    logger.debug(f"{sym}: ${p['close']:.2f}")
            except Exception as e:
                logger.warning(f"Could not get price for {sym}: {e}")
        
        # Generate signals for all symbols
        best_signal = None
        for symbol in symbols:
            df = get_price_df(provider, symbol, current_date)
            if df.empty:
                continue
            
            signal = ensemble.generate_signal(symbol, df)
            
            if signal.action != "HOLD":
                if best_signal is None or signal.confidence > best_signal.confidence:
                    best_signal = signal
        
        # Process best signal
        if best_signal and best_signal.action != "HOLD":
            logger.info(f"Signal: {best_signal.action} {best_signal.symbol} ({best_signal.confidence:.0%})")
            
            # Apply risk rules
            risk_result = apply_risk_rules(best_signal, exchange, provider, current_date)
            
            if risk_result['approved']:
                # Calculate position size
                shares = calculate_position_size(best_signal.confidence, risk_result['max_shares'])
                
                if shares > 0:
                    # Execute trade
                    provider.execute_order(
                        best_signal.symbol,
                        best_signal.action,
                        shares
                    )
                    logger.info(f"Executed: {best_signal.action} {shares} {best_signal.symbol}")
                else:
                    logger.info(f"Holding: Position size = 0")
            else:
                logger.info(f"Holding: {risk_result['reason']}")
        else:
            logger.info("Holding: No actionable signals")
        
        # Log portfolio status
        portfolio_value = exchange.get_total_portfolio_value(current_date)
        logger.info(f"Portfolio: ${portfolio_value:,.2f} | Cash: ${exchange.cash:,.2f} | Holdings: {exchange.holdings}")
    
    # Final summary
    _print_summary(exchange, args.cash, args.end, "FAST BACKTEST")
    return exchange


# ==============================================================================
# CREW BACKTEST MODE: CrewAI with LLM Orchestration
# ==============================================================================

def run_crew_backtest_mode(args):
    """
    Run backtest with CrewAI and LLM orchestration.
    
    Uses MockDataProvider + VirtualExchange for local simulation,
    or AlpacaDataProvider + VirtualExchange for real historical data.
    """
    from src.crew import TradingCrew
    from src.simulation.virtual_exchange import VirtualExchange
    from src.utils.date_sources import historical_dates
    
    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    logger.info("=" * 60)
    logger.info("CREW BACKTEST MODE (With LLM)")
    logger.info("=" * 60)
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
    _print_summary(exchange, args.cash, args.end, "CREW BACKTEST")
    return exchange


# ==============================================================================
# LIVE MODE: Real-time Paper Trading with Alpaca
# ==============================================================================

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


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

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


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Agent Trader - AI-powered trading with strategy engine and CrewAI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast backtest (no LLM, default)
  python agent_trader.py --backtest --start 2023-11-01 --end 2023-11-07

  # Fast backtest with real Alpaca data
  python agent_trader.py --backtest --start 2023-11-01 --end 2023-11-07 --real-data

  # Crew backtest with LLM orchestration
  python agent_trader.py --backtest --start 2023-11-01 --end 2023-11-07 --crew

  # Live paper trading
  python agent_trader.py --live --interval 300
        """
    )
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--live', action='store_true', help='Run live paper trading')
    mode_group.add_argument('--backtest', action='store_true', help='Run historical backtest')
    
    # Common arguments
    parser.add_argument('--symbols', type=str, default='PLTR,NFLX,PLTK',
                        help='Comma-separated list of symbols (default: PLTR,NFLX,PLTK)')
    
    # Live mode arguments
    parser.add_argument('--interval', type=int, default=300,
                        help='Seconds between trading cycles in live mode (default: 300)')
    
    # Backtest mode arguments
    parser.add_argument('--start', type=str, help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='Backtest end date (YYYY-MM-DD)')
    parser.add_argument('--real-data', action='store_true',
                        help='Use real Alpaca historical data instead of mock')
    parser.add_argument('--crew', action='store_true',
                        help='Use CrewAI with LLM orchestration (slower but with reasoning)')
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
    elif args.backtest:
        if args.crew:
            run_crew_backtest_mode(args)
        else:
            run_fast_backtest_mode(args)


if __name__ == "__main__":
    main()
