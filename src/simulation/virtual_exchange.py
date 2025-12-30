import pandas as pd


class VirtualExchange:
    def __init__(self, initial_cash=100000.0, provider=None):
        """
        :param provider: The DataProvider (Mock or Real) to fetch prices for Mark-to-Market calculation.
        """
        self.cash = initial_cash
        self.holdings = {} # e.g., {'PLTR': 50, 'NFLX': 10}
        self.provider = provider
        self.transaction_log = []

    def execute_trade(self, action: str, symbol: str, quantity: int, current_date: str):
        # 1. Get the price at that specific date
        # Note: In a real backtest, you'd use the 'Open' or 'Close' of that date
        price_data = self.provider.get_price_history(symbol, current_date, current_date)
        
        # Safe extraction of price from the data structure
        try:
            # Assuming data structure matches Alpaca's barset dictionary
            # Adjust this line based on your exact data shape from Phase 2
            current_price = list(price_data.values())[0][0]['c'] 
        except (IndexError, KeyError):
            print(f"❌ ERROR: No price data found for {symbol} on {current_date}")
            return

        cost = current_price * quantity

        if action.upper() == "BUY":
            if self.cash >= cost:
                self.cash -= cost
                self.holdings[symbol] = self.holdings.get(symbol, 0) + quantity
                self.log_trade(current_date, "BUY", symbol, quantity, current_price)
                print(f"✅ BOUGHT {quantity} {symbol} @ ${current_price}")
            else:
                print(f"❌ FAIL: Insufficient funds to buy {quantity} {symbol}")

        elif action.upper() == "SELL":
            current_shares = self.holdings.get(symbol, 0)
            if current_shares >= quantity:
                self.cash += cost
                self.holdings[symbol] = current_shares - quantity
                self.log_trade(current_date, "SELL", symbol, quantity, current_price)
                print(f"✅ SOLD {quantity} {symbol} @ ${current_price}")
            else:
                print(f"❌ FAIL: Not enough shares to sell {quantity} {symbol}")

    def get_total_portfolio_value(self, current_date: str) -> float:
        """Calculates Cash + Market Value of all stocks"""
        equity = 0.0
        for symbol, qty in self.holdings.items():
            if qty > 0:
                price_data = self.provider.get_price_history(symbol, current_date, current_date)
                try:
                    price = list(price_data.values())[0][0]['c']
                    equity += price * qty
                except:
                    pass # Skip if no price found (or use yesterday's price)
        return self.cash + equity

    def log_trade(self, date, action, symbol, qty, price):
        self.transaction_log.append({
            "date": date, "action": action, "symbol": symbol, "qty": qty, "price": price
        })