from .interface import MarketDataProvider

class MockDataProvider(MarketDataProvider):
    def get_price_history(self, symbol: str, start_date: str, end_date: str):
        # Return static fake data
        return {
            "2023-01-01": {"close": 150.0},
            "2023-01-02": {"close": 155.0}
        }

    def get_latest_price(self, symbol: str) -> float:
        # Return a fixed price to make logic testing predictable
        if symbol == "PLTR": return 25.0
        if symbol == "NFLX": return 400.0
        return 100.0

    def get_account_balance(self) -> float:
        return 100000.0

    def execute_order(self, symbol: str, side: str, qty: int) -> str:
        return f"MOCK_ORDER_ID_123_FOR_{symbol}_{side}_{qty}"