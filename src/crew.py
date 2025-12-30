from crewai import Agent, Task, Crew, Process
from langchain.tools import tool
from textwrap import dedent
from src.tools.market_tools import StockPriceTool, StockHistoryTool, AccountBalanceTool
from src.tools.calculator import CalculatorTools
from src.tools.execution_tools import ExecuteTradeTool

class TradingCrew:
    def __init__(self, provider, exchange=None):
        """
        :param provider: An instance of AlpacaDataProvider (Real) or MockDataProvider.
        """
        self.provider = provider
        self.exchange = exchange # VirtualExchange or alpaca exchange if None
        
        # Instantiate Tools with the specific provider
        self.price_tool = StockPriceTool(provider=provider)
        self.history_tool = StockHistoryTool(provider=provider)
        self.balance_tool = AccountBalanceTool(provider=provider, exchange=exchange)
        self.calc_tool = CalculatorTools().calculate
        self.execution_tool = ExecuteTradeTool(provider=provider)

    def build_crew(self, current_date: str, stock_selection: list):
        """
        Builds the agents and tasks.
        :param current_date: The 'simulated' or 'real' date the agents should think it is.
        :param stock_selection: List of tickers to analyze (e.g., ['PLTR', 'NFLX']).
        """
        
        # --- 1. THE AGENTS ---
        
        researcher = Agent(
            role='Financial Researcher',
            goal='Uncover significant market news and trends.',
            backstory=dedent(f"""
                You are an expert financial researcher. 
                The current date is {current_date}. 
                You act as the eyes and ears of the trading desk.
                Your job is to find news or macro-economic factors that could affect: {stock_selection}.
                You strictly NEVER invent news. If you don't know, say you don't know.
            """),
            tools=[], # TODO: Add a NewsTool (SerpApi)
            verbose=True,
            allow_delegation=False
        )

        analyst = Agent(
            role='Technical Analyst',
            goal='Analyze price movements and predict future trends.',
            backstory=dedent(f"""
                You are a veteran technical analyst.
                The current date is {current_date}.
                You look at charts and historical data for {stock_selection}.
                You use mathematical indicators (SMA, RSI, etc.) to identify entry and exit points.
                You are cautious and precise.
            """),
            tools=[self.price_tool, self.history_tool, self.calc_tool],
            verbose=True,
            allow_delegation=False
        )

        risk_manager = Agent(
            role='Risk Manager',
            goal='Protect the portfolio capital.',
            backstory=dedent(f"""
                You are the "Adult in the Room". The current date is {current_date}.
                Your only job is to ensure the team doesn't blow up the account.
                You check the cash balance and enforce position sizing rules.
                Rules:
                1. Never put more than 20% of cash into a single trade.
                2. Always keep 10% of the portfolio in cash.
            """),
            tools=[self.balance_tool, self.calc_tool],
            verbose=True,
            allow_delegation=False
        )

        head_trader = Agent(
            role='Head Trader',
            goal='Listen to the team and EXECUTE trades directly.',
            backstory=dedent(f"""
                You are the Head Trader. The current date is {current_date}.
                You have the authority to buy and sell.
                
                PROCESS:
                1. Review analysis from Researcher and Analyst.
                2. Check with Risk Manager for MAX position size (Crucial!).
                3. If the signal is strong and Risk Manager approves:
                   USE the 'Execute Trade' tool immediately.
                4. If no trade is needed, just say "Holding cash."
            """),
            tools=[self.execution_tool], # Give them the button
            verbose=True,
            allow_delegation=True
        )

        # --- 2. THE TASKS ---
        
        # Task 1: Research
        research_task = Task(
            description=f"Find news and sentiment for {stock_selection} for date {current_date}.",
            agent=researcher,
            expected_output="A summary of market sentiment (Bullish/Bearish/Neutral) with sources."
        )

        # Task 2: Analysis
        analysis_task = Task(
            description=f"Retrieve price history for {stock_selection} ending on {current_date}. Calculate trends.",
            agent=analyst,
            expected_output="Technical analysis report with support/resistance levels."
        )

        # Task 3: Risk Assessment
        risk_task = Task(
            description=f"Check current cash balance. Calculate maximum safe position size for {stock_selection}.",
            agent=risk_manager,
            expected_output="Max shares allowable to buy for each stock based on current capital."
        )

        # Task 4: Execution
        trade_task = Task(
            description="Synthesize team inputs. If a trade is viable, EXECUTE it using the tool. Do not ask for permission.",
            agent=head_trader,
            expected_output="Confirmation that the trade was executed or a reason for holding."
        )

        # --- 3. THE CREW ---
        
        crew = Crew(
            agents=[researcher, analyst, risk_manager, head_trader],
            tasks=[research_task, analysis_task, risk_task, trade_task],
            process=Process.sequential, # TODO: check if hierarchical is better if you want the boss to manage them
            verbose=True
        )

        return crew