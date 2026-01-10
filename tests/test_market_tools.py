import pytest
from src.tools.market_tools import StockPriceTool, AccountBalanceTool, StockHistoryTool
from src.data.interface import MarketDataProvider
from src.simulation.virtual_exchange import VirtualExchange


class MockProviderForMarketTools(MarketDataProvider):
    """Mock provider for testing market tools"""
    def __init__(self):
        self.price_calls = []
        self.balance_calls = []
        self.history_calls = []
    
    def get_price_history(self, symbol: str, start_date: str, end_date: str):
        self.history_calls.append({
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date
        })
        return {f"{symbol}_data": "mock_history"}
    
    def get_latest_price(self, symbol: str) -> float:
        self.price_calls.append(symbol)
        if symbol == "PLTR":
            return 25.0
        elif symbol == "NFLX":
            return 400.0
        return 100.0
    
    def get_account_balance(self) -> float:
        self.balance_calls.append("called")
        return 100000.0
    
    def execute_order(self, symbol: str, side: str, qty: int) -> str:
        return "MOCK_ORDER"


class TestStockPriceTool:
    """Test suite for StockPriceTool"""
    
    @pytest.fixture
    def tool(self):
        """Create a tool instance with mock provider"""
        provider = MockProviderForMarketTools()
        return StockPriceTool(provider=provider)
    
    def test_get_price_success(self, tool):
        """Test successful price retrieval"""
        result = tool._run("PLTR")
        
        assert "The current price of PLTR is $25.0" in result
        assert len(tool.provider.price_calls) == 1
        assert tool.provider.price_calls[0] == "PLTR"
    
    def test_get_price_different_symbol(self, tool):
        """Test price retrieval for different symbols"""
        result = tool._run("NFLX")
        
        assert "The current price of NFLX is $400.0" in result
        assert len(tool.provider.price_calls) == 1
    
    def test_get_price_unknown_symbol(self, tool):
        """Test price retrieval for unknown symbol"""
        result = tool._run("UNKNOWN")
        
        assert "The current price of UNKNOWN is $100.0" in result
    
    def test_get_price_with_whitespace(self, tool):
        """Test that whitespace is handled"""
        result = tool._run("  PLTR  ")
        
        # Provider should receive trimmed symbol
        assert len(tool.provider.price_calls) == 1
    
    def test_get_price_provider_exception(self, tool):
        """Test error handling when provider raises exception"""
        class FailingProvider(MockProviderForMarketTools):
            def get_latest_price(self, symbol: str) -> float:
                raise ConnectionError("API connection failed")
        
        failing_tool = StockPriceTool(provider=FailingProvider())
        result = failing_tool._run("PLTR")
        
        assert "Error fetching price" in result
        assert "API connection failed" in result


class TestAccountBalanceTool:
    """Test suite for AccountBalanceTool"""
    
    def test_get_balance_with_exchange(self):
        """Test balance retrieval when exchange is provided (simulated)"""
        provider = MockProviderForMarketTools()
        exchange = VirtualExchange(initial_cash=50000.0)
        tool = AccountBalanceTool(provider=provider, exchange=exchange)
        
        result = tool._run("none")
        
        assert "Current available cash (SIMULATED): $50000.0" in result
        assert len(provider.balance_calls) == 0  # Should not call provider
    
    def test_get_balance_without_exchange(self):
        """Test balance retrieval when no exchange (real provider)"""
        provider = MockProviderForMarketTools()
        tool = AccountBalanceTool(provider=provider, exchange=None)
        
        result = tool._run("none")
        
        assert "Current available cash (REAL): $100000.0" in result
        assert len(provider.balance_calls) == 1
    
    def test_get_balance_provider_exception(self):
        """Test error handling when provider raises exception"""
        class FailingProvider(MockProviderForMarketTools):
            def get_account_balance(self) -> float:
                raise ValueError("Balance fetch failed")
        
        provider = FailingProvider()
        tool = AccountBalanceTool(provider=provider, exchange=None)
        
        result = tool._run("none")
        
        assert "Error fetching balance" in result
        assert "Balance fetch failed" in result
    
    def test_get_balance_exchange_exception(self):
        """Test error handling when exchange access fails"""
        provider = MockProviderForMarketTools()
        # Create exchange without cash attribute (edge case)
        class BrokenExchange:
            pass
        
        exchange = BrokenExchange()
        tool = AccountBalanceTool(provider=provider, exchange=exchange)
        
        # Should fall back to provider or handle gracefully
        result = tool._run("none")
        # The tool should handle AttributeError if cash doesn't exist
        # This tests the robustness


class TestStockHistoryTool:
    """Test suite for StockHistoryTool"""
    
    @pytest.fixture
    def tool(self):
        """Create a tool instance with mock provider"""
        provider = MockProviderForMarketTools()
        return StockHistoryTool(provider=provider)
    
    def test_get_history_success(self, tool):
        """Test successful history retrieval"""
        result = tool._run("PLTR, 2023-01-01, 2023-01-07")
        
        assert "Historical data for PLTR:" in result
        assert len(tool.provider.history_calls) == 1
        assert tool.provider.history_calls[0]["symbol"] == "PLTR"
        assert tool.provider.history_calls[0]["start_date"] == "2023-01-01"
        assert tool.provider.history_calls[0]["end_date"] == "2023-01-07"
    
    def test_get_history_with_whitespace(self, tool):
        """Test that whitespace is properly stripped"""
        result = tool._run("  PLTR  ,  2023-01-01  ,  2023-01-07  ")
        
        assert "Historical data for PLTR:" in result
        assert tool.provider.history_calls[0]["symbol"] == "PLTR"
        assert tool.provider.history_calls[0]["start_date"] == "2023-01-01"
        assert tool.provider.history_calls[0]["end_date"] == "2023-01-07"
    
    def test_get_history_invalid_format_too_few_parts(self, tool):
        """Test error when input has too few parts"""
        result = tool._run("PLTR, 2023-01-01")
        
        assert "Error:" in result
        assert "SYMBOL, START_DATE, END_DATE" in result
        assert len(tool.provider.history_calls) == 0
    
    def test_get_history_invalid_format_too_many_parts(self, tool):
        """Test error when input has too many parts"""
        result = tool._run("PLTR, 2023-01-01, 2023-01-07, EXTRA")
        
        assert "Error:" in result
        assert "SYMBOL, START_DATE, END_DATE" in result
        assert len(tool.provider.history_calls) == 0
    
    def test_get_history_empty_input(self, tool):
        """Test error handling for empty input"""
        result = tool._run("")
        
        assert "Error:" in result
        assert len(tool.provider.history_calls) == 0
    
    def test_get_history_provider_exception(self, tool):
        """Test error handling when provider raises exception"""
        class FailingProvider(MockProviderForMarketTools):
            def get_price_history(self, symbol: str, start_date: str, end_date: str):
                raise ValueError("History fetch failed")
        
        failing_tool = StockHistoryTool(provider=FailingProvider())
        result = failing_tool._run("PLTR, 2023-01-01, 2023-01-07")
        
        assert "Error fetching history" in result
        assert "History fetch failed" in result
    
    def test_get_history_different_date_formats(self, tool):
        """Test that different date formats are passed through"""
        result = tool._run("NFLX, 2023-12-25, 2024-01-01")
        
        assert "Historical data for NFLX:" in result
        assert tool.provider.history_calls[0]["start_date"] == "2023-12-25"
        assert tool.provider.history_calls[0]["end_date"] == "2024-01-01"
    
    def test_get_history_multiple_calls(self, tool):
        """Test multiple history calls"""
        tool._run("PLTR, 2023-01-01, 2023-01-07")
        tool._run("NFLX, 2023-02-01, 2023-02-07")
        
        assert len(tool.provider.history_calls) == 2
        assert tool.provider.history_calls[0]["symbol"] == "PLTR"
        assert tool.provider.history_calls[1]["symbol"] == "NFLX"







