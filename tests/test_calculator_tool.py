import pytest
from src.tools.calculator import CalculatorTools

def test_calculate_simple_math():
    calc = CalculatorTools()
    assert calc.calculate_math_expression("100 + 2") == 102
    assert calc.calculate_math_expression("100 - 2") == 98
    assert calc.calculate_math_expression("100 * 2") == 200
    assert calc.calculate_math_expression("(10 + 5) * 2") == 30
    

def test_calculate_position_size():
    calc = CalculatorTools()
    # Scenario:
    # Account = $100,000
    # Risk per trade = 1% ($1,000 risk)
    # Entry Price = $150
    # Stop Loss = $140 ($10 risk per share)
    # Result should be: $1,000 total risk / $10 risk per share = 100 shares
    shares = calc.calculate_position_size(100000, 0.01, 150, 140)
    assert shares == 100

def test_calculate_position_size_rounding():
    calc = CalculatorTools()
    # Test that it floors the result (you can't buy 10.7 shares easily)
    # Risk $1000. Risk per share $13. 1000/13 = 76.92
    shares = calc.calculate_position_size(100000, 0.01, 113, 100)
    assert shares == 76

def test_calculate_reward_risk_ratio():
    calc = CalculatorTools()
    # Entry Price $100. stop loss $90. profit target $120. Reward Risk Ratio = (120 - 100) / (100 - 90) = 2 
    assert calc.calculate_reward_risk_ratio(100, 90, 120) == 2
    assert calc.calculate_reward_risk_ratio(100, 93, 118) == 2.6 #round from 2.57 to 2.6
    assert calc.calculate_reward_risk_ratio(100, 89, 118) == 1.6 #round from 1.64 to 1.6


def test_block_dangerous_code():
    calc = CalculatorTools()
    # Ensure the agent can't run import os; os.system('rm -rf')
    result = calc.calculate_math_expression("import os")
    assert "Error" in result


def test_calculate_math_expression_with_errors():
    calc = CalculatorTools()
    assert calc.calculate_math_expression("100 / 0") == "Error: Division by zero"
    assert calc.calculate_math_expression("100 / 'a'") == "Error: Invalid input"
    assert calc.calculate_math_expression("100 / [1, 2, 3]") == "Error: Invalid input"
    assert calc.calculate_math_expression("100 / {1: 'a', 2: 'b'}") == "Error: Invalid input"
    assert calc.calculate_math_expression("100 / (1, 2, 3)") == "Error: Invalid input"
    assert calc.calculate_math_expression("100 / (1, 2, 3)") == "Error: Invalid input"