import pytest
from src.simulation.virtual_exchange import VirtualExchange
from src.data.interface import MarketDataProvider


class MockProviderForTesting(MarketDataProvider):
    """Mock provider that returns data in the format VirtualExchange expects"""
    def __init__(self, price_data=None):
        # price_data format: {symbol: {date: price}}
        self.price_data = price_data or {}
    
    def get_price_history(self, symbol: str, start_date: str, end_date: str):
        # Return format: {date: {'close': price, 'open': price, ...}} to match VirtualExchange expectations
        if symbol in self.price_data and start_date in self.price_data[symbol]:
            price = self.price_data[symbol][start_date]
            return {
                start_date: {
                    'close': price,
                    'open': price,
                    'high': price,
                    'low': price,
                    'volume': 1000
                }
            }
        # Return empty dict if symbol/date not found - this will cause execute_trade to fail gracefully
        return {}
    
    def get_latest_price(self, symbol: str) -> float:
        return 100.0
    
    def get_account_balance(self) -> float:
        return 100000.0
    
    def execute_order(self, symbol: str, side: str, qty: int) -> str:
        return f"MOCK_ORDER_{symbol}_{qty}"


class TestVirtualExchange:
    """Test suite for VirtualExchange"""
    
    def test_initialization(self):
        """Test that VirtualExchange initializes with correct default values"""
        exchange = VirtualExchange(initial_cash=50000.0)
        assert exchange.cash == 50000.0
        assert exchange.holdings == {}
        assert exchange.transaction_log == []
        assert exchange.current_date == "N/A"
    
    def test_update_date(self):
        """Test date update functionality"""
        exchange = VirtualExchange()
        exchange.update_date("2023-01-15")
        assert exchange.current_date == "2023-01-15"
    
    def test_execute_buy_success(self):
        """Test successful buy order execution"""
        provider = MockProviderForTesting({"PLTR": {"2023-01-01": 25.0}})
        exchange = VirtualExchange(initial_cash=10000.0, provider=provider)
        exchange.update_date("2023-01-01")
        
        initial_cash = exchange.cash
        exchange.execute_trade("BUY", "PLTR", 10, "2023-01-01")
        
        assert exchange.cash == initial_cash - (25.0 * 10)
        assert exchange.holdings["PLTR"] == 10
        assert len(exchange.transaction_log) == 1
        assert exchange.transaction_log[0]["action"] == "BUY"
        assert exchange.transaction_log[0]["symbol"] == "PLTR"
        assert exchange.transaction_log[0]["qty"] == 10
        assert exchange.transaction_log[0]["price"] == 25.0
    
    def test_execute_buy_insufficient_funds(self):
        """Test buy order fails when insufficient funds"""
        provider = MockProviderForTesting({"PLTR": {"2023-01-01": 25.0}})
        exchange = VirtualExchange(initial_cash=100.0, provider=provider)
        exchange.update_date("2023-01-01")
        
        initial_cash = exchange.cash
        initial_holdings = exchange.holdings.copy()
        
        # Try to buy more than we can afford
        exchange.execute_trade("BUY", "PLTR", 10, "2023-01-01")  # Cost: 250.0, but only have 100.0
        
        # Should not execute
        assert exchange.cash == initial_cash
        assert exchange.holdings == initial_holdings
        assert len(exchange.transaction_log) == 0
    
    def test_execute_sell_success(self):
        """Test successful sell order execution"""
        provider = MockProviderForTesting({"PLTR": {"2023-01-01": 25.0, "2023-01-02": 30.0}})
        exchange = VirtualExchange(initial_cash=10000.0, provider=provider)
        exchange.update_date("2023-01-01")
        
        # First buy some shares
        exchange.execute_trade("BUY", "PLTR", 20, "2023-01-01")
        initial_cash = exchange.cash
        initial_holdings = exchange.holdings["PLTR"]
        
        # Then sell some
        exchange.execute_trade("SELL", "PLTR", 10, "2023-01-02")
        
        assert exchange.cash == initial_cash + (30.0 * 10)
        assert exchange.holdings["PLTR"] == initial_holdings - 10
        assert len(exchange.transaction_log) == 2
        assert exchange.transaction_log[1]["action"] == "SELL"
        assert exchange.transaction_log[1]["price"] == 30.0
    
    def test_execute_sell_insufficient_shares(self):
        """Test sell order fails when insufficient shares"""
        provider = MockProviderForTesting({"PLTR": {"2023-01-01": 25.0}})
        exchange = VirtualExchange(initial_cash=10000.0, provider=provider)
        exchange.update_date("2023-01-01")
        
        # Buy 5 shares
        exchange.execute_trade("BUY", "PLTR", 5, "2023-01-01")
        initial_cash = exchange.cash
        initial_holdings = exchange.holdings["PLTR"]
        
        # Try to sell more than we have
        exchange.execute_trade("SELL", "PLTR", 10, "2023-01-01")
        
        # Should not execute
        assert exchange.cash == initial_cash
        assert exchange.holdings["PLTR"] == initial_holdings
        assert len(exchange.transaction_log) == 1  # Only the buy
    
    def test_execute_sell_without_holdings(self):
        """Test sell order fails when no holdings exist"""
        provider = MockProviderForTesting({"PLTR": {"2023-01-01": 25.0}})
        exchange = VirtualExchange(initial_cash=10000.0, provider=provider)
        exchange.update_date("2023-01-01")
        
        initial_cash = exchange.cash
        
        # Try to sell without owning any shares
        exchange.execute_trade("SELL", "PLTR", 10, "2023-01-01")
        
        assert exchange.cash == initial_cash
        assert "PLTR" not in exchange.holdings or exchange.holdings.get("PLTR", 0) == 0
        assert len(exchange.transaction_log) == 0
    
    def test_execute_trade_case_insensitive(self):
        """Test that action is case-insensitive"""
        provider = MockProviderForTesting({"PLTR": {"2023-01-01": 25.0}})
        exchange = VirtualExchange(initial_cash=10000.0, provider=provider)
        exchange.update_date("2023-01-01")
        
        # Test lowercase
        exchange.execute_trade("buy", "PLTR", 10, "2023-01-01")
        assert exchange.holdings["PLTR"] == 10
        
        # Test mixed case
        exchange.execute_trade("SeLL", "PLTR", 5, "2023-01-01")
        assert exchange.holdings["PLTR"] == 5
    
    def test_execute_trade_missing_price_data(self):
        """Test that trade fails gracefully when price data is missing"""
        provider = MockProviderForTesting()  # No price data configured
        exchange = VirtualExchange(initial_cash=10000.0, provider=provider)
        exchange.update_date("2023-01-01")
        
        initial_cash = exchange.cash
        
        # Try to execute trade for symbol with no price data
        # This should fail silently (returns early)
        exchange.execute_trade("BUY", "UNKNOWN", 10, "2023-01-01")
        
        # Should not execute
        assert exchange.cash == initial_cash
        assert len(exchange.transaction_log) == 0
    
    def test_get_total_portfolio_value(self):
        """Test portfolio valuation calculation"""
        provider = MockProviderForTesting({
            "PLTR": {"2023-01-01": 25.0, "2023-01-02": 30.0},
            "NFLX": {"2023-01-01": 400.0, "2023-01-02": 410.0}
        })
        exchange = VirtualExchange(initial_cash=10000.0, provider=provider)
        exchange.update_date("2023-01-01")
        
        # Buy some shares
        exchange.execute_trade("BUY", "PLTR", 20, "2023-01-01")  # Cost: 500
        exchange.execute_trade("BUY", "NFLX", 5, "2023-01-01")   # Cost: 2000
        
        # Check portfolio value on a different date (different prices)
        portfolio_value = exchange.get_total_portfolio_value("2023-01-02")
        
        # Cash: 10000 - 500 - 2000 = 7500
        # PLTR value: 20 * 30 = 600
        # NFLX value: 5 * 410 = 2050
        # Total: 7500 + 600 + 2050 = 10150
        expected_value = 7500.0 + (20 * 30.0) + (5 * 410.0)
        assert portfolio_value == expected_value
    
    def test_get_total_portfolio_value_cash_only(self):
        """Test portfolio valuation with no holdings"""
        provider = MockProviderForTesting()
        exchange = VirtualExchange(initial_cash=50000.0, provider=provider)
        
        portfolio_value = exchange.get_total_portfolio_value("2023-01-01")
        assert portfolio_value == 50000.0
    
    def test_get_total_portfolio_value_with_zero_holdings(self):
        """Test portfolio valuation when holdings exist but are zero"""
        provider = MockProviderForTesting({"PLTR": {"2023-01-01": 25.0}})
        exchange = VirtualExchange(initial_cash=10000.0, provider=provider)
        exchange.update_date("2023-01-01")
        
        # Buy and then sell all
        exchange.execute_trade("BUY", "PLTR", 10, "2023-01-01")
        exchange.execute_trade("SELL", "PLTR", 10, "2023-01-01")
        
        portfolio_value = exchange.get_total_portfolio_value("2023-01-01")
        # Should only count cash, not zero holdings
        assert portfolio_value == exchange.cash
    
    def test_log_trade(self):
        """Test transaction logging"""
        exchange = VirtualExchange()
        
        exchange.log_trade("2023-01-01", "BUY", "PLTR", 10, 25.0)
        
        assert len(exchange.transaction_log) == 1
        log_entry = exchange.transaction_log[0]
        assert log_entry["date"] == "2023-01-01"
        assert log_entry["action"] == "BUY"
        assert log_entry["symbol"] == "PLTR"
        assert log_entry["qty"] == 10
        assert log_entry["price"] == 25.0
    
    def test_multiple_trades_accumulate_holdings(self):
        """Test that multiple buy orders accumulate holdings correctly"""
        provider = MockProviderForTesting({"PLTR": {"2023-01-01": 25.0}})
        exchange = VirtualExchange(initial_cash=10000.0, provider=provider)
        exchange.update_date("2023-01-01")
        
        exchange.execute_trade("BUY", "PLTR", 10, "2023-01-01")
        exchange.execute_trade("BUY", "PLTR", 5, "2023-01-01")
        exchange.execute_trade("BUY", "PLTR", 3, "2023-01-01")
        
        assert exchange.holdings["PLTR"] == 18
        assert len(exchange.transaction_log) == 3
    
    def test_portfolio_value_with_missing_price_skips(self):
        """Test that missing price data for a holding is skipped gracefully"""
        provider = MockProviderForTesting({
            "PLTR": {"2023-01-01": 25.0},
            # NFLX has no price data
        })
        exchange = VirtualExchange(initial_cash=10000.0, provider=provider)
        exchange.update_date("2023-01-01")
        
        # Manually add holdings for NFLX (simulating a scenario)
        exchange.holdings["NFLX"] = 10
        
        # Should not crash, should just skip NFLX and return cash + PLTR value
        portfolio_value = exchange.get_total_portfolio_value("2023-01-01")
        # Should at least equal cash (NFLX skipped)
        assert portfolio_value >= exchange.cash

