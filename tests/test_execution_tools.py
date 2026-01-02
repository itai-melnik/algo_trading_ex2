import pytest
from src.tools.execution_tools import ExecuteTradeTool
from src.data.interface import MarketDataProvider


class MockProviderForExecution(MarketDataProvider):
    """Mock provider for testing ExecuteTradeTool"""
    def __init__(self):
        self.executed_orders = []
    
    def get_price_history(self, symbol: str, start_date: str, end_date: str):
        return {}
    
    def get_latest_price(self, symbol: str) -> float:
        return 100.0
    
    def get_account_balance(self) -> float:
        return 100000.0
    
    def execute_order(self, symbol: str, side: str, qty: int) -> str:
        self.executed_orders.append({
            "symbol": symbol,
            "side": side,
            "qty": qty
        })
        return f"ORDER_{symbol}_{qty}_{side}"


class TestExecuteTradeTool:
    """Test suite for ExecuteTradeTool"""
    
    @pytest.fixture
    def tool(self):
        """Create a tool instance with mock provider"""
        provider = MockProviderForExecution()
        return ExecuteTradeTool(provider=provider)
    
    def test_valid_buy_order(self, tool):
        """Test successful buy order parsing and execution"""
        result = tool._run("BUY|PLTR|10")
        
        assert "Order executed successfully" in result
        assert "ORDER_PLTR_10_buy" in result
        assert len(tool.provider.executed_orders) == 1
        assert tool.provider.executed_orders[0]["symbol"] == "PLTR"
        assert tool.provider.executed_orders[0]["side"] == "buy"
        assert tool.provider.executed_orders[0]["qty"] == 10
    
    def test_valid_sell_order(self, tool):
        """Test successful sell order parsing and execution"""
        result = tool._run("SELL|NFLX|5")
        
        assert "Order executed successfully" in result
        assert "ORDER_NFLX_5_sell" in result
        assert len(tool.provider.executed_orders) == 1
        assert tool.provider.executed_orders[0]["symbol"] == "NFLX"
        assert tool.provider.executed_orders[0]["side"] == "sell"
        assert tool.provider.executed_orders[0]["qty"] == 5
    
    def test_order_with_whitespace(self, tool):
        """Test that whitespace is properly stripped"""
        result = tool._run("  BUY  |  PLTR  |  10  ")
        
        assert "Order executed successfully" in result
        assert len(tool.provider.executed_orders) == 1
        assert tool.provider.executed_orders[0]["symbol"] == "PLTR"
        assert tool.provider.executed_orders[0]["qty"] == 10
    
    def test_order_case_insensitive(self, tool):
        """Test that side and symbol are converted to uppercase"""
        result = tool._run("buy|pltr|10")
        
        assert "Order executed successfully" in result
        assert len(tool.provider.executed_orders) == 1
        assert tool.provider.executed_orders[0]["side"] == "buy"
    
    def test_invalid_format_too_few_parts(self, tool):
        """Test error when input has too few parts"""
        result = tool._run("BUY|PLTR")
        
        assert "Error:" in result
        assert "SIDE|TICKER|QTY" in result
        assert len(tool.provider.executed_orders) == 0
    
    def test_invalid_format_too_many_parts(self, tool):
        """Test error when input has too many parts"""
        result = tool._run("BUY|PLTR|10|EXTRA")
        
        assert "Error:" in result
        assert "SIDE|TICKER|QTY" in result
        assert len(tool.provider.executed_orders) == 0
    
    def test_invalid_side(self, tool):
        """Test error when side is not BUY or SELL"""
        result = tool._run("HOLD|PLTR|10")
        
        assert "Error:" in result
        assert "Side must be BUY or SELL" in result
        assert len(tool.provider.executed_orders) == 0
    
    def test_invalid_quantity_not_integer(self, tool):
        """Test error when quantity is not an integer"""
        result = tool._run("BUY|PLTR|10.5")
        
        assert "Error:" in result
        assert "Quantity must be an integer" in result
        assert len(tool.provider.executed_orders) == 0
    
    def test_invalid_quantity_non_numeric(self, tool):
        """Test error when quantity is not numeric"""
        result = tool._run("BUY|PLTR|abc")
        
        assert "Error:" in result
        assert "Quantity must be an integer" in result
        assert len(tool.provider.executed_orders) == 0
    
    def test_invalid_quantity_empty(self, tool):
        """Test error when quantity is empty"""
        result = tool._run("BUY|PLTR|")
        
        assert "Error:" in result
        assert "Quantity must be an integer" in result
        assert len(tool.provider.executed_orders) == 0
    
    def test_zero_quantity(self, tool):
        """Test that zero quantity is accepted (edge case)"""
        result = tool._run("BUY|PLTR|0")
        
        # Should execute (though not very useful)
        assert "Order executed successfully" in result
        assert len(tool.provider.executed_orders) == 1
        assert tool.provider.executed_orders[0]["qty"] == 0
    
    def test_negative_quantity(self, tool):
        """Test that negative quantity is accepted (parsed as int)"""
        result = tool._run("BUY|PLTR|-5")
        
        # Should execute (though semantically wrong)
        assert "Order executed successfully" in result
        assert len(tool.provider.executed_orders) == 1
        assert tool.provider.executed_orders[0]["qty"] == -5
    
    def test_provider_exception_handling(self, tool):
        """Test that provider exceptions are caught and returned as error message"""
        class FailingProvider(MockProviderForExecution):
            def execute_order(self, symbol: str, side: str, qty: int) -> str:
                raise ValueError("Provider error occurred")
        
        failing_tool = ExecuteTradeTool(provider=FailingProvider())
        result = failing_tool._run("BUY|PLTR|10")
        
        assert "Execution Failed" in result
        assert "Provider error occurred" in result
    
    def test_empty_input(self, tool):
        """Test error handling for empty input"""
        result = tool._run("")
        
        assert "Error:" in result
        assert len(tool.provider.executed_orders) == 0
    
    def test_multiple_valid_orders(self, tool):
        """Test multiple valid orders can be executed"""
        tool._run("BUY|PLTR|10")
        tool._run("SELL|NFLX|5")
        tool._run("BUY|AAPL|3")
        
        assert len(tool.provider.executed_orders) == 3
        assert tool.provider.executed_orders[0]["symbol"] == "PLTR"
        assert tool.provider.executed_orders[1]["symbol"] == "NFLX"
        assert tool.provider.executed_orders[2]["symbol"] == "AAPL"






