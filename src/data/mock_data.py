from .interface import MarketDataProvider

class MockDataProvider(MarketDataProvider):
    def __init__(self, exchange=None):
        self.exchange = exchange  

    def set_exchange(self, exchange):
        """Helper to link exchange after initialization"""
        self.exchange = exchange

    def get_price_history(self, symbol: str, start_date: str, end_date: str):
        # 1. Determine the price based on symbol
        price = 100.0
        if symbol == "PLTR":
            price = 25.0
        elif symbol == "NFLX":
            price = 400.0
        
        # 2. Return the format expected by VirtualExchange
        return {
            start_date: {
                'close': price,
                'open': price,
                'high': price,
                'low': price,
                'volume': 1000
            }
        }

    def get_latest_price(self, symbol: str) -> float:
        # Return a fixed price to make logic testing predictable
        if symbol == "PLTR": return 25.0
        if symbol == "NFLX": return 400.0
        return 100.0

    def get_account_balance(self) -> float:
        return 100000.0

    def execute_order(self, symbol: str, side: str, qty: int) -> str:
        if self.exchange:
            # Route the order to the Virtual Exchange logic
            # use a dummy date or handle date tracking in the exchange
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