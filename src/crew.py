"""
Refactored Trading Crew with Functional Agent Roles.

Agents are now lightweight orchestrators - all heavy computation
is done in the strategy engine via tools.
"""
from crewai import Agent, Task, Crew, Process
from textwrap import dedent
from langchain_openai import ChatOpenAI

from src.tools.market_tools import StockPriceTool, AccountBalanceTool
from src.tools.strategy_tools import GenerateSignalsTool, RiskFilterTool, PositionSizeTool
from src.tools.execution_tools import ExecuteTradeTool

# Use GPT-4o-mini for faster, cheaper execution
# The LLM now only orchestrates - doesn't compute
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class TradingCrew:
    """
    Functional Trading Crew with 4 specialized agents:
    1. Signal Generator - Analyzes stocks using strategy engine
    2. Risk Filter - Applies risk management rules
    3. Position Sizer - Calculates optimal position size
    4. Executor - Executes approved trades
    
    All heavy computation is done in tools, not by the LLM.
    """
    
    def __init__(self, provider, exchange=None):
        """
        Initialize the trading crew with data provider and optional exchange.
        
        Args:
            provider: An instance of AlpacaDataProvider (Real) or MockDataProvider.
            exchange: VirtualExchange for backtesting, or None for live trading.
        """
        self.provider = provider
        self.exchange = exchange
        
        # Strategy tools - these do all the computation
        self.signal_tool = GenerateSignalsTool(
            provider=provider, 
            exchange=exchange,
            lookback_days=30
        )
        self.risk_tool = RiskFilterTool(
            provider=provider,
            exchange=exchange
        )
        self.position_tool = PositionSizeTool(provider=provider)
        self.execution_tool = ExecuteTradeTool(provider=provider, exchange=exchange)
        
        # Supporting tools
        self.price_tool = StockPriceTool(provider=provider)
        self.balance_tool = AccountBalanceTool(provider=provider, exchange=exchange)

    def build_crew(self, current_date: str, stock_selection: list):
        """
        Builds the crew with functional agent roles.
        
        Args:
            current_date: The 'simulated' or 'real' date for the trading session.
            stock_selection: List of tickers to analyze (e.g., ['PLTR', 'NFLX']).
        
        Returns:
            Crew object ready to kickoff.
        """
        stocks_str = ", ".join(stock_selection)
        
        # --- 1. FUNCTIONAL AGENTS ---
        
        signal_generator = Agent(
            role='Signal Generator',
            goal='Generate trading signals for each stock using the strategy engine.',
            backstory=dedent(f"""
                You are the Signal Generator. Date: {current_date}.
                
                YOUR ONLY JOB:
                1. Call the 'Generate Trading Signals' tool for EACH stock: {stocks_str}
                2. Parse the JSON response
                3. Report which stock has the STRONGEST signal (highest confidence)
                
                You do NOT compute anything yourself - the tool does all analysis.
                Just call it and interpret the results.
            """),
            tools=[self.signal_tool],
            llm=llm,
            verbose=True,
            allow_delegation=False
        )

        risk_filter = Agent(
            role='Risk Filter',
            goal='Validate trades against risk management rules.',
            backstory=dedent(f"""
                You are the Risk Filter. Date: {current_date}.
                
                YOUR ONLY JOB:
                1. Take the best signal from the Signal Generator
                2. Call the 'Risk Filter' tool with: SYMBOL|ACTION|CONFIDENCE
                3. If approved, report the max_shares allowed
                4. If rejected, explain why
                
                The tool enforces all risk rules - you just call it and report.
            """),
            tools=[self.risk_tool, self.balance_tool],
            llm=llm,
            verbose=True,
            allow_delegation=False
        )

        position_sizer = Agent(
            role='Position Sizer',
            goal='Calculate the optimal position size for approved trades.',
            backstory=dedent(f"""
                You are the Position Sizer. Date: {current_date}.
                
                YOUR ONLY JOB:
                1. Take the approved trade from Risk Filter (with max_shares)
                2. Call 'Calculate Position Size' with: SYMBOL|ACTION|CONFIDENCE|MAX_SHARES
                3. Report the recommended_shares to trade
                
                The tool scales position by confidence - you just call it.
            """),
            tools=[self.position_tool, self.price_tool],
            llm=llm,
            verbose=True,
            allow_delegation=False
        )

        executor = Agent(
            role='Trade Executor',
            goal='Execute approved trades with the calculated position size.',
            backstory=dedent(f"""
                You are the Trade Executor. Date: {current_date}.
                
                YOUR ONLY JOB:
                1. Take the final trade decision (SYMBOL, ACTION, SHARES)
                2. If shares > 0, call 'Execute Trade' with: ACTION|SYMBOL|SHARES
                3. Verify the result shows "Order executed successfully"
                4. If no trade needed, report "Holding cash - no action taken"
                
                You are the final step. Execute and confirm.
            """),
            tools=[self.execution_tool],
            llm=llm,
            verbose=True,
            allow_delegation=False
        )

        # --- 2. SEQUENTIAL TASKS ---
        
        signal_task = Task(
            description=dedent(f"""
                Generate trading signals for these stocks: {stocks_str}
                
                Steps:
                1. Call 'Generate Trading Signals' tool for EACH stock
                2. Compare the combined_signal from each response
                3. Select the best opportunity (highest confidence, not HOLD)
                4. If all signals are HOLD, report that clearly
                
                Output format:
                BEST_SIGNAL: SYMBOL|ACTION|CONFIDENCE
                REASON: <why this is the best opportunity>
            """),
            agent=signal_generator,
            expected_output="The best trading signal in format: SYMBOL|ACTION|CONFIDENCE"
        )

        risk_task = Task(
            description=dedent("""
                Validate the best signal against risk rules.
                
                Steps:
                1. Parse the BEST_SIGNAL from previous task (SYMBOL|ACTION|CONFIDENCE)
                2. Call 'Risk Filter' tool with that input
                3. Report the result
                
                Output format:
                APPROVED: YES/NO
                MAX_SHARES: <number if approved>
                REASON: <approval or rejection reason>
            """),
            agent=risk_filter,
            expected_output="Risk assessment with approval status and max shares",
            context=[signal_task]
        )

        sizing_task = Task(
            description=dedent("""
                Calculate optimal position size for approved trades.
                
                Steps:
                1. If previous task shows APPROVED: NO, output "NO_TRADE"
                2. If approved, call 'Calculate Position Size' with: SYMBOL|ACTION|CONFIDENCE|MAX_SHARES
                3. Report the recommended quantity
                
                Output format:
                TRADE: SYMBOL|ACTION|RECOMMENDED_SHARES
                or
                NO_TRADE: <reason>
            """),
            agent=position_sizer,
            expected_output="Final trade decision with quantity",
            context=[signal_task, risk_task]
        )

        execute_task = Task(
            description=dedent("""
                Execute the final trade decision.
                
                Steps:
                1. If previous task shows NO_TRADE, report "Holding cash"
                2. If there's a trade (SYMBOL|ACTION|SHARES), call 'Execute Trade'
                3. Verify execution success
                4. Report final status
                
                Output must include:
                - ACTION TAKEN: <BUY/SELL X shares of SYMBOL> or <Holding cash>
                - EXECUTION STATUS: <Success/No trade needed>
            """),
            agent=executor,
            expected_output="Execution confirmation or hold status",
            context=[sizing_task]
        )

        # --- 3. ASSEMBLE CREW ---
        
        crew = Crew(
            agents=[signal_generator, risk_filter, position_sizer, executor],
            tasks=[signal_task, risk_task, sizing_task, execute_task],
            process=Process.sequential,
            verbose=True
        )

        return crew


# Convenience function to create a default ensemble crew
def create_trading_crew(provider, exchange=None):
    """
    Factory function to create a TradingCrew.
    
    Args:
        provider: Data provider instance
        exchange: Optional VirtualExchange for backtesting
        
    Returns:
        TradingCrew instance
    """
    return TradingCrew(provider=provider, exchange=exchange)
