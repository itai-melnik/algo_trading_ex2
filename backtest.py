"""
Local Backtesting Script - Pure Mock Data (No Alpaca API Required)

Uses MockDataProvider + VirtualExchange for complete local simulation.
For paper trading with real data, use a separate script with AlpacaDataProvider.
"""
from datetime import timedelta, datetime
from src.crew import TradingCrew
from src.simulation.virtual_exchange import VirtualExchange
from src.data.mock_data import MockDataProvider


# 1. Configuration
SYMBOLS = ['PLTR', 'NFLX', 'PLTK']
START_DATE = "2023-11-01"
END_DATE = "2023-11-07"
INITIAL_CASH = 100000
RANDOM_SEED = 42  # For reproducible price generation


# 2. Setup Infrastructure (Pure Mock - No Alpaca!)
mock_provider = MockDataProvider(seed=RANDOM_SEED)
exchange = VirtualExchange(initial_cash=INITIAL_CASH, provider=mock_provider)
mock_provider.set_exchange(exchange)


# Helper to generate dates
def daterange(start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    for n in range(int((end - start).days) + 1):
        yield start + timedelta(n)


# 3. The Loop
print(f"🚀 Starting MOCK Backtest from {START_DATE} to {END_DATE}")
print(f"💰 Initial Cash: ${INITIAL_CASH:,.2f}")
print(f"🎲 Random Seed: {RANDOM_SEED} (for reproducibility)")
print("=" * 60)

for single_date in daterange(START_DATE, END_DATE):
    current_date_str = single_date.strftime("%Y-%m-%d")
    
    # Update the simulation clock
    exchange.update_date(current_date_str)
    
    # Skip weekends
    if single_date.weekday() > 4: 
        print(f"⏭️  Skipping Weekend: {current_date_str}")
        continue

    print(f"\n--- 📅 PROCESSING DATE: {current_date_str} ---")
    
    # Show mock prices for the day
    for sym in SYMBOLS:
        price_data = mock_provider.get_price_history(sym, current_date_str, current_date_str)
        if current_date_str in price_data:
            p = price_data[current_date_str]
            print(f"   {sym}: O=${p['open']:.2f} H=${p['high']:.2f} L=${p['low']:.2f} C=${p['close']:.2f}")
    
    # A. Build the Crew for this specific day
    trading_bot = TradingCrew(provider=mock_provider, exchange=exchange) 
    crew = trading_bot.build_crew(current_date=current_date_str, stock_selection=SYMBOLS)

    # B. Kickoff
    crew.kickoff() 

    # Daily Summary
    total_val = exchange.get_total_portfolio_value(current_date_str)
    print(f"📊 EOD Portfolio Value: ${total_val:,.2f}")


# 4. Final Summary
print("\n" + "=" * 60)
print("✅ Backtest Complete.")

# Get final portfolio value
final_value = exchange.get_total_portfolio_value(END_DATE)
profit_loss = final_value - INITIAL_CASH
pct_return = (profit_loss / INITIAL_CASH) * 100

print(f"📈 Final Portfolio Value: ${final_value:,.2f}")
print(f"💵 Cash Remaining: ${exchange.cash:,.2f}")
print(f"📦 Holdings: {exchange.holdings}")
print(f"💹 P&L: ${profit_loss:+,.2f} ({pct_return:+.2f}%)")
print(f"📝 Total Transactions: {len(exchange.transaction_log)}")

if exchange.transaction_log:
    print("\n📋 Transaction Log:")
    for tx in exchange.transaction_log:
        print(f"   {tx['date']} | {tx['action']:4} | {tx['qty']:4} x {tx['symbol']:5} @ ${tx['price']:.2f}")
