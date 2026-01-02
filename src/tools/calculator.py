import re
from langchain.tools import tool

# Pre-compiled regex patterns for better performance
_DANGEROUS_PATTERN = re.compile(
    r'import\s+|__\w+__|exec\s*\(|eval\s*\(|open\s*\(|'
    r'file\s*\(|input\s*\(|raw_input\s*\(|compile\s*\(|'
    r'reload\s*\(|__import__\s*\(',
    re.IGNORECASE
)
_SAFE_CHARS_PATTERN = re.compile(r'^[0-9+\-*/().\s]+$')


class CalculatorTools:
    """Tools for performing mathematical calculations safely."""
    
    @tool("Calculate General Math")
    def calculate_math_expression(expression: str):
        """
        Safely evaluate a mathematical expression.
        
        Args:
            expression: A string containing a mathematical expression
            
        Returns:
            The result of the calculation, or an error message string
        """
        # Block dangerous code patterns
        if _DANGEROUS_PATTERN.search(expression):
            return "Error: Invalid input"
        
        # Only allow safe characters: numbers, operators, parentheses, spaces, decimal points
        if not _SAFE_CHARS_PATTERN.match(expression):
            return "Error: Invalid input"
        
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            
            if not isinstance(result, (int, float)):
                return "Error: Invalid input"
            
            # Handle infinity/NaN from operations like 1.0/0.0
            if result != result or result == float('inf') or result == float('-inf'):
                return "Error: Division by zero"
            
            return result
        except ZeroDivisionError:
            return "Error: Division by zero"
        except Exception:
            return "Error: Invalid input"
    
    @tool("Calculate Position Size")
    def calculate_position_size(account_balance: float, risk_percentage: float, 
                                entry_price: float, stop_loss: float) -> int:
        """
        Calculate the position size based on account balance, risk percentage, entry price, and stop loss.
        
        Args:
            account_balance: Total account balance
            risk_percentage: Risk percentage per trade (e.g., 0.01 for 1%)
            entry_price: Entry price per share
            stop_loss: Stop loss price per share
            
        Returns:
            Number of shares to buy (floored to integer)
        """
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            return 0
        return int((account_balance * risk_percentage) / risk_per_share)

    @tool("Calculate Reward to Risk Ratio")
    def calculate_reward_risk_ratio(entry_price: float, stop_loss: float,
                                    profit_target: float) -> float:
        """
        Calculate the reward to risk ratio.
        
        Args:
            entry_price: Entry price per share
            stop_loss: Stop loss price per share
            profit_target: Profit target price per share
            
        Returns:
            Reward to risk ratio rounded to 1 decimal place
        """
        risk = entry_price - stop_loss
        if risk == 0:
            return 0.0
        return round((profit_target - entry_price) / risk, 1)
