from crewai.tools import tool
from typing import Any
from src.data.interface import MarketDataProvider
from crewai.tools import BaseTool
from pydantic import Field

class MarketTools:
    def __init__(self, provider: MarketDataProvider):
        self.provider = provider


class StockPriceTool(BaseTool):
    name: str = "Get Stock Price"
    description: str = "Useful to get the current price of a stock. Input is the ticker symbol."
    provider: MarketDataProvider = Field(exclude=True) # Exclude from AI input schema

    def _run(self, symbol: str) -> str:
        try:
            price = self.provider.get_latest_price(symbol)
            return f"The current price of {symbol} is ${price}"
        except Exception as e:
            return f"Error fetching price for {symbol}: {str(e)}"

class AccountBalanceTool(BaseTool):
    name: str = "Get Account Balance"
    description: str = "Useful to check how much cash is available to trade."
    provider: MarketDataProvider = Field(exclude=True)
    exchange: Any = Field(default=None, exclude=True)

    def _run(self, dummy_arg: str = "none") -> str:
        try:
            if self.exchange:
                return f"Current available cash (SIMULATED): ${self.exchange.cash}"
            else:
                bal = self.provider.get_account_balance()
                return f"Current available cash (REAL): ${bal}"
        except Exception as e:
            return f"Error fetching balance: {str(e)}"

class StockHistoryTool(BaseTool):
    name: str = "Get Stock History"
    description: str = "Useful to get historical pricing data for analysis. Input format: 'SYMBOL, START_DATE, END_DATE' (e.g., 'PLTR, 2023-01-01, 2023-01-07')"
    provider: MarketDataProvider = Field(exclude=True)

    def _run(self, arguments: str) -> str:
        try:
            # Parse the comma-separated string
            parts = [x.strip() for x in arguments.split(',')]
            if len(parts) != 3:
                return "Error: Input must be 'SYMBOL, START_DATE, END_DATE'"
            
            symbol, start, end = parts
            data = self.provider.get_price_history(symbol, start, end)
            return f"Historical data for {symbol}: {str(data)}"
        except Exception as e:
            return f"Error fetching history: {str(e)}"