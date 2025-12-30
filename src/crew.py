from crewai import Agent, Task, Crew, Process
from langchain.tools import tool
from textwrap import dedent
from src.tools.market_tools import StockPriceTool, StockHistoryTool, AccountBalanceTool
from src.tools.calculator import CalculatorTools

class TradingCrew:
    def __init__(self, provider):
        """
        :param provider: An instance of AlpacaDataProvider (Real) or MockDataProvider.
        """
        self.provider = provider
        
        # Instantiate Tools with the specific provider
        self.price_tool = StockPriceTool(provider=provider)
        self.history_tool = StockHistoryTool(provider=provider)
        self.balance_tool = AccountBalanceTool(provider=provider)
        self.calc_tool = CalculatorTools().calculate

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
            goal='Execute the best trade decision based on team input.',
            backstory=dedent(f"""
                You are the Head Trader. The current date is {current_date}.
                You listen to the Researcher (News), Analyst (Price), and Risk Manager (Safety).
                You synthesize their inputs into a FINAL decision.
                Your output must be a clear JSON-like instruction:
                {{ "action": "BUY/SELL/HOLD", "ticker": "XYZ", "quantity": 10, "reason": "..." }}
            """),
            verbose=True,
            allow_delegation=True # The boss can ask questions to others if needed
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
            description="Review reports from Researcher, Analyst, and Risk Manager. Decide what to trade.",
            agent=head_trader,
            expected_output="Final JSON execution plan.",
            context=[research_task, analysis_task, risk_task] # This passes previous outputs to the boss
        )

        # --- 3. THE CREW ---
        
        crew = Crew(
            agents=[researcher, analyst, risk_manager, head_trader],
            tasks=[research_task, analysis_task, risk_task, trade_task],
            process=Process.sequential, # TODO: check if hierarchical is better if you want the boss to manage them
            verbose=True
        )

        return crew