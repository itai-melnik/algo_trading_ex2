from abc import ABC, abstractmethod
from typing import Dict, Any

class MarketDataProvider(ABC):
    
    @abstractmethod
    def get_price_history(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Returns historical price data (Open, High, Low, Close, Volume)."""
        pass

    @abstractmethod
    def get_latest_price(self, symbol: str) -> float:
        """Returns the current real-time price."""
        pass

    @abstractmethod
    def get_account_balance(self) -> float:
        """Returns current cash balance."""
        pass