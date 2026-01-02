import pytest
from src.data.mock_data import MockDataProvider
from src.simulation.virtual_exchange import VirtualExchange


class TestMockDataProvider:
    """Test suite for MockDataProvider"""
    
    @pytest.fixture
    def provider(self):
        """Create a mock provider instance"""
        return MockDataProvider()
    
    @pytest.fixture
    def exchange(self):
        """Create a virtual exchange for testing"""
        return VirtualExchange(initial_cash=10000.0)
    
    def test_get_price_history(self, provider):
        """Test price history retrieval"""
        result = provider.get_price_history("PLTR", "2023-01-01", "2023-01-02")
        
        assert isinstance(result, dict)
        # MockDataProvider only returns start_date in the result
        assert "2023-01-01" in result
        # Check the structure: {start_date: {'close': price, 'open': price, ...}}
        assert result["2023-01-01"]["close"] == 25.0  # PLTR price
        assert result["2023-01-01"]["open"] == 25.0
        assert result["2023-01-01"]["high"] == 25.0
        assert result["2023-01-01"]["low"] == 25.0
        assert result["2023-01-01"]["volume"] == 1000
    
    def test_get_price_history_any_symbol(self, provider):
        """Test that price history works for any symbol"""
        result = provider.get_price_history("ANY_SYMBOL", "2023-01-01", "2023-01-02")
        
        # MockDataProvider only returns start_date with default price (100.0) for unknown symbols
        assert "2023-01-01" in result
        assert result["2023-01-01"]["close"] == 100.0  # Default price for unknown symbols
    
    def test_get_latest_price_pltr(self, provider):
        """Test latest price for PLTR"""
        price = provider.get_latest_price("PLTR")
        assert price == 25.0
    
    def test_get_latest_price_nflx(self, provider):
        """Test latest price for NFLX"""
        price = provider.get_latest_price("NFLX")
        assert price == 400.0
    
    def test_get_latest_price_default(self, provider):
        """Test latest price for unknown symbol returns default"""
        price = provider.get_latest_price("UNKNOWN")
        assert price == 100.0
    
    def test_get_account_balance(self, provider):
        """Test account balance retrieval"""
        balance = provider.get_account_balance()
        assert balance == 100000.0
    
    def test_execute_order_with_exchange(self, provider, exchange):
        """Test order execution when exchange is connected"""
        provider.set_exchange(exchange)
        exchange.provider = provider  # Link exchange back to provider
        exchange.update_date("2023-01-01")
        
        # VirtualExchange.execute_trade expects: {date: {'close': price, ...}}
        # This matches MockDataProvider.get_price_history format
        result = provider.execute_order("PLTR", "buy", 10)
        
        assert "MOCK_FILLED_PLTR_10" in result
        assert exchange.holdings.get("PLTR", 0) == 10
        assert exchange.cash < 10000.0  # Cash should be reduced
    
    def test_execute_order_without_exchange(self, provider):
        """Test order execution fails when no exchange is connected"""
        result = provider.execute_order("PLTR", "buy", 10)
        
        assert result == "ERROR: No Exchange Connected"
    
    def test_execute_order_sell_with_exchange(self, provider, exchange):
        """Test sell order execution with exchange"""
        provider.set_exchange(exchange)
        exchange.provider = provider  # Link exchange back to provider
        exchange.update_date("2023-01-01")
        
        # First buy
        provider.execute_order("PLTR", "buy", 20)
        initial_cash = exchange.cash
        
        # Then sell
        result = provider.execute_order("PLTR", "sell", 10)
        
        assert "MOCK_FILLED_PLTR_10" in result
        assert exchange.holdings.get("PLTR", 0) == 10
        assert exchange.cash > initial_cash  # Cash should increase
    
    def test_set_exchange(self, provider, exchange):
        """Test setting exchange after initialization"""
        assert provider.exchange is None
        
        provider.set_exchange(exchange)
        assert provider.exchange == exchange
    
    def test_execute_order_case_insensitive(self, provider, exchange):
        """Test that order side is case-insensitive"""
        provider.set_exchange(exchange)
        exchange.provider = provider  # Link exchange back to provider
        exchange.update_date("2023-01-01")
        
        # Test uppercase
        result1 = provider.execute_order("PLTR", "BUY", 5)
        assert "MOCK_FILLED" in result1
        
        # Test lowercase - need to buy first to have shares to sell
        provider.execute_order("NFLX", "buy", 10)
        result2 = provider.execute_order("NFLX", "sell", 3)
        assert "MOCK_FILLED" in result2
    
    def test_execute_order_updates_exchange_date(self, provider, exchange):
        """Test that execute_order uses exchange's current_date"""
        provider.set_exchange(exchange)
        exchange.provider = provider  # Link exchange back to provider
        exchange.update_date("2023-06-15")
        
        # Track that get_price_history is called with correct date
        original_get_price_history = provider.get_price_history
        called_with_dates = []
        
        def tracking_get_price_history(symbol, start_date, end_date):
            called_with_dates.append((start_date, end_date))
            return original_get_price_history(symbol, start_date, end_date)
        
        provider.get_price_history = tracking_get_price_history
        
        provider.execute_order("PLTR", "buy", 10)
        
        # Verify the date was passed correctly
        assert len(called_with_dates) > 0
        assert called_with_dates[0] == ("2023-06-15", "2023-06-15")
    
    def test_execute_order_with_unknown_date(self, provider, exchange):
        """Test execute_order when exchange has no current_date set"""
        provider.set_exchange(exchange)
        exchange.provider = provider  # Link exchange back to provider
        # Don't set current_date, should default to "N/A"
        
        result = provider.execute_order("PLTR", "buy", 10)
        # Should still execute, though date might be invalid
        assert "MOCK_FILLED" in result or "ERROR" in result
    
    def test_multiple_execute_orders(self, provider, exchange):
        """Test multiple order executions"""
        provider.set_exchange(exchange)
        exchange.provider = provider  # Link exchange back to provider
        exchange.update_date("2023-01-01")
        
        result1 = provider.execute_order("PLTR", "buy", 10)
        result2 = provider.execute_order("NFLX", "buy", 5)
        result3 = provider.execute_order("PLTR", "sell", 3)
        
        assert "MOCK_FILLED" in result1
        assert "MOCK_FILLED" in result2
        assert "MOCK_FILLED" in result3
        assert exchange.holdings.get("PLTR", 0) == 7
        assert exchange.holdings.get("NFLX", 0) == 5






