from crewai_tools import BaseTool
from pydantic import Field
from src.data.interface import MarketDataProvider

class ExecuteTradeTool(BaseTool):
    name: str = "Execute Trade"
    description: str = "Use this to BUY or SELL. Input format: 'SIDE|TICKER|QTY' (e.g., 'BUY|PLTR|10'). Returns Order ID."
    provider: MarketDataProvider = Field(exclude=True)

    def _run(self, order_string: str) -> str:
        try:
            # 1. Parse Input (We enforce a strict format to help the LLM)
            parts = order_string.split('|')
            if len(parts) != 3:
                return "Error: Input must be 'SIDE|TICKER|QTY' (e.g., 'SELL|NFLX|5')"
            
            side, symbol, qty_str = [p.strip().upper() for p in parts]
            
            # 2. Safety Checks
            if side not in ['BUY', 'SELL']:
                return "Error: Side must be BUY or SELL"
            
            try:
                qty = int(qty_str)
            except ValueError:
                return "Error: Quantity must be an integer"

            # 3. Execute via Provider
            result = self.provider.execute_order(symbol, side.lower(), qty)
            return f"Order executed successfully. ID: {result}"

        except Exception as e:
            return f"Execution Failed: {str(e)}"