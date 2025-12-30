import json
import re
from datetime import timedelta, datetime
from src.crew import TradingCrew
from src.data.real_data import AlpacaDataProvider
from src.utils.rate_limiter import RateLimiter
from src.simulation.virtual_exchange import VirtualExchange

# 1. Configuration
SYMBOLS = ['PLTR', 'NFLX', 'PLTK']
START_DATE = "2023-11-01" # Pick a Monday
END_DATE = "2023-11-07"   # A short 1-week test

# 2. Setup Infrastructure
# We use REAL data provider because we want real historical prices, 
# but we wrap it in our VirtualExchange so we don't spend real money.
limiter = RateLimiter(max_calls=45, period_seconds=60) # Careful with Alpaca limits
data_provider = AlpacaDataProvider(rate_limiter=limiter)
exchange = VirtualExchange(initial_cash=100000, provider=data_provider)

# Helper to generate dates
def daterange(start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    for n in range(int((end - start).days) + 1):
        yield start + timedelta(n)

# 3. The Loop
print(f"Starting Backtest from {START_DATE} to {END_DATE}")

for single_date in daterange(START_DATE, END_DATE):
    current_date_str = single_date.strftime("%Y-%m-%d")
    
    # Skip weekends (simplified logic)
    if single_date.weekday() > 4: 
        print(f"Skipping Weekend: {current_date_str}")
        continue

    print(f"\n--- PROCESSING DATE: {current_date_str} ---")
    
    # A. Build the Crew for this specific day
    # We pass the exchange to the crew so tools can see *simulated* balance
    # NOTE: You might need to update your AccountBalanceTool to look at 
    # 'exchange.cash' instead of 'provider.get_balance' during backtests.
    trading_bot = TradingCrew(provider=data_provider) 
    crew = trading_bot.build_crew(current_date=current_date_str, stock_selection=SYMBOLS)

    # B. Kickoff
    result = crew.kickoff()

    # C. Parse Result (The messy part!)
    # The Head Trader returns text. We need to extract the JSON.
    try:
        print(f"Raw Output: {result}")
        
        # Regex to find JSON block if the LLM adds extra text
        match = re.search(r'\{.*\}', str(result), re.DOTALL)
        if match:
            clean_json = match.group(0)
            decision = json.loads(clean_json)
            
            action = decision.get("action")
            ticker = decision.get("ticker")
            qty = int(decision.get("quantity", 0))
            
            if action in ["BUY", "SELL"] and qty > 0:
                exchange.execute_trade(action, ticker, qty, current_date_str)
            else:
                print("Decision was HOLD or Invalid.")
        else:
            print("Could not parse JSON from agent output.")

    except Exception as e:
        print(f"Error executing trade: {e}")

    # D. Daily Summary
    total_val = exchange.get_total_portfolio_value(current_date_str)
    print(f"EOD Portfolio Value: ${total_val:,.2f}")

print("\nBacktest Complete.")
print("Transactions:", exchange.transaction_log)