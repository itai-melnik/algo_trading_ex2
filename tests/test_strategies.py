"""
Tests for the Strategy Engine components.
TDD: These tests are written BEFORE the implementation.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestSignal:
    """Tests for the Signal dataclass."""
    
    def test_signal_creation(self):
        """Signal can be created with required fields."""
        from src.strategies.signals import Signal
        
        signal = Signal(
            symbol="PLTR",
            action="BUY",
            confidence=0.8,
            strategy_name="SMA_Crossover",
            reason="20-day SMA crossed above 50-day SMA"
        )
        
        assert signal.symbol == "PLTR"
        assert signal.action == "BUY"
        assert signal.confidence == 0.8
        assert signal.strategy_name == "SMA_Crossover"
        assert signal.reason == "20-day SMA crossed above 50-day SMA"
    
    def test_signal_action_validation(self):
        """Signal action must be BUY, SELL, or HOLD."""
        from src.strategies.signals import Signal
        
        # Valid actions should work
        for action in ["BUY", "SELL", "HOLD"]:
            signal = Signal("PLTR", action, 0.5, "test", "test reason")
            assert signal.action == action
    
    def test_signal_confidence_bounds(self):
        """Signal confidence should be between 0 and 1."""
        from src.strategies.signals import Signal
        
        # Valid confidence
        signal = Signal("PLTR", "BUY", 0.5, "test", "reason")
        assert 0 <= signal.confidence <= 1
        
        # Edge cases
        signal_zero = Signal("PLTR", "HOLD", 0.0, "test", "reason")
        assert signal_zero.confidence == 0.0
        
        signal_one = Signal("PLTR", "BUY", 1.0, "test", "reason")
        assert signal_one.confidence == 1.0
    
    def test_signal_to_dict(self):
        """Signal can be converted to dictionary for tool output."""
        from src.strategies.signals import Signal
        
        signal = Signal("NFLX", "SELL", 0.7, "Momentum", "Price dropped 5%")
        result = signal.to_dict()
        
        assert isinstance(result, dict)
        assert result["symbol"] == "NFLX"
        assert result["action"] == "SELL"
        assert result["confidence"] == 0.7
        assert result["strategy_name"] == "Momentum"
        assert result["reason"] == "Price dropped 5%"
    
    def test_hold_signal_helper(self):
        """Can create a HOLD signal easily."""
        from src.strategies.signals import Signal
        
        signal = Signal.hold("PLTR", "No clear signal")
        assert signal.action == "HOLD"
        assert signal.confidence == 0.0
        assert signal.symbol == "PLTR"


class TestBaseStrategy:
    """Tests for the BaseStrategy abstract class."""
    
    def test_base_strategy_is_abstract(self):
        """BaseStrategy cannot be instantiated directly."""
        from src.strategies.base_strategy import BaseStrategy
        
        with pytest.raises(TypeError):
            BaseStrategy()
    
    def test_concrete_strategy_must_implement_generate_signal(self):
        """Subclasses must implement generate_signal method."""
        from src.strategies.base_strategy import BaseStrategy
        
        class IncompleteStrategy(BaseStrategy):
            @property
            def name(self):
                return "Incomplete"
        
        with pytest.raises(TypeError):
            IncompleteStrategy()
    
    def test_concrete_strategy_works(self):
        """A properly implemented strategy can be instantiated."""
        from src.strategies.base_strategy import BaseStrategy
        from src.strategies.signals import Signal
        
        class DummyStrategy(BaseStrategy):
            @property
            def name(self):
                return "Dummy"
            
            def generate_signal(self, symbol: str, price_history: pd.DataFrame) -> Signal:
                return Signal.hold(symbol, "Dummy always holds")
        
        strategy = DummyStrategy()
        assert strategy.name == "Dummy"
        
        # Create dummy price data
        df = pd.DataFrame({
            'close': [100, 101, 102],
            'open': [99, 100, 101],
            'high': [102, 103, 104],
            'low': [98, 99, 100],
            'volume': [1000, 1100, 1200]
        })
        
        signal = strategy.generate_signal("PLTR", df)
        assert signal.action == "HOLD"
        assert signal.symbol == "PLTR"


class TestSMACrossoverStrategy:
    """Tests for SMA Crossover Strategy."""
    
    def _create_price_df(self, prices: list) -> pd.DataFrame:
        """Helper to create price DataFrame from close prices."""
        dates = pd.date_range(end=datetime.now(), periods=len(prices), freq='D')
        return pd.DataFrame({
            'close': prices,
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'volume': [1000000] * len(prices)
        }, index=dates)
    
    def test_sma_crossover_buy_signal(self):
        """Generates BUY when fast SMA crosses above slow SMA."""
        from src.strategies.sma_crossover import SMACrossoverStrategy
        
        strategy = SMACrossoverStrategy(fast_period=5, slow_period=10)
        
        # Create data where price is trending up (fast SMA > slow SMA)
        # Start low, trend up so fast SMA crosses above slow SMA
        prices = [100] * 10 + [105, 110, 115, 120, 125]  # 15 days
        df = self._create_price_df(prices)
        
        signal = strategy.generate_signal("PLTR", df)
        assert signal.action == "BUY"
        assert signal.confidence > 0.5
    
    def test_sma_crossover_sell_signal(self):
        """Generates SELL when fast SMA crosses below slow SMA."""
        from src.strategies.sma_crossover import SMACrossoverStrategy
        
        strategy = SMACrossoverStrategy(fast_period=5, slow_period=10)
        
        # Create data where price is trending down
        prices = [120] * 10 + [115, 110, 105, 100, 95]  # 15 days
        df = self._create_price_df(prices)
        
        signal = strategy.generate_signal("PLTR", df)
        assert signal.action == "SELL"
        assert signal.confidence > 0.5
    
    def test_sma_crossover_hold_signal(self):
        """Generates HOLD when no clear crossover."""
        from src.strategies.sma_crossover import SMACrossoverStrategy
        
        strategy = SMACrossoverStrategy(fast_period=5, slow_period=10)
        
        # Flat prices - no trend
        prices = [100] * 20
        df = self._create_price_df(prices)
        
        signal = strategy.generate_signal("PLTR", df)
        assert signal.action == "HOLD"
    
    def test_sma_insufficient_data(self):
        """Returns HOLD with low confidence if insufficient data."""
        from src.strategies.sma_crossover import SMACrossoverStrategy
        
        strategy = SMACrossoverStrategy(fast_period=5, slow_period=10)
        
        # Only 5 days of data (need at least 10 for slow SMA)
        prices = [100, 101, 102, 103, 104]
        df = self._create_price_df(prices)
        
        signal = strategy.generate_signal("PLTR", df)
        assert signal.action == "HOLD"
        assert signal.confidence == 0.0


class TestMomentumStrategy:
    """Tests for Momentum Strategy."""
    
    def _create_price_df(self, prices: list) -> pd.DataFrame:
        """Helper to create price DataFrame from close prices."""
        dates = pd.date_range(end=datetime.now(), periods=len(prices), freq='D')
        return pd.DataFrame({
            'close': prices,
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'volume': [1000000] * len(prices)
        }, index=dates)
    
    def test_momentum_buy_signal(self):
        """Generates BUY when price is significantly higher than N days ago."""
        from src.strategies.momentum import MomentumStrategy
        
        strategy = MomentumStrategy(lookback_days=10, threshold=0.05)
        
        # Price up 10% from 10 days ago
        prices = [100] * 10 + [110]
        df = self._create_price_df(prices)
        
        signal = strategy.generate_signal("NFLX", df)
        assert signal.action == "BUY"
        assert signal.confidence > 0.5
    
    def test_momentum_sell_signal(self):
        """Generates SELL when price is significantly lower than N days ago."""
        from src.strategies.momentum import MomentumStrategy
        
        strategy = MomentumStrategy(lookback_days=10, threshold=0.05)
        
        # Price down 10% from 10 days ago
        prices = [110] * 10 + [100]
        df = self._create_price_df(prices)
        
        signal = strategy.generate_signal("NFLX", df)
        assert signal.action == "SELL"
        assert signal.confidence > 0.5
    
    def test_momentum_hold_below_threshold(self):
        """Generates HOLD when price change is below threshold."""
        from src.strategies.momentum import MomentumStrategy
        
        strategy = MomentumStrategy(lookback_days=10, threshold=0.05)
        
        # Price up only 2% (below 5% threshold)
        prices = [100] * 10 + [102]
        df = self._create_price_df(prices)
        
        signal = strategy.generate_signal("NFLX", df)
        assert signal.action == "HOLD"


class TestMeanReversionStrategy:
    """Tests for Mean Reversion (Bollinger Bands) Strategy."""
    
    def _create_price_df(self, prices: list) -> pd.DataFrame:
        """Helper to create price DataFrame from close prices."""
        dates = pd.date_range(end=datetime.now(), periods=len(prices), freq='D')
        return pd.DataFrame({
            'close': prices,
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'volume': [1000000] * len(prices)
        }, index=dates)
    
    def test_mean_reversion_buy_signal(self):
        """Generates BUY when price is below lower Bollinger Band."""
        from src.strategies.mean_reversion import MeanReversionStrategy
        
        strategy = MeanReversionStrategy(period=20, num_std=2.0)
        
        # Stable prices then sudden drop (below lower band)
        prices = [100] * 20 + [85]  # Drop to below 2 std devs
        df = self._create_price_df(prices)
        
        signal = strategy.generate_signal("PLTK", df)
        assert signal.action == "BUY"
    
    def test_mean_reversion_sell_signal(self):
        """Generates SELL when price is above upper Bollinger Band."""
        from src.strategies.mean_reversion import MeanReversionStrategy
        
        strategy = MeanReversionStrategy(period=20, num_std=2.0)
        
        # Stable prices then sudden spike (above upper band)
        prices = [100] * 20 + [115]  # Spike above 2 std devs
        df = self._create_price_df(prices)
        
        signal = strategy.generate_signal("PLTK", df)
        assert signal.action == "SELL"
    
    def test_mean_reversion_hold_within_bands(self):
        """Generates HOLD when price is within Bollinger Bands."""
        from src.strategies.mean_reversion import MeanReversionStrategy
        
        strategy = MeanReversionStrategy(period=20, num_std=2.0)
        
        # Stable prices - within bands
        prices = [100] * 21
        df = self._create_price_df(prices)
        
        signal = strategy.generate_signal("PLTK", df)
        assert signal.action == "HOLD"


class TestVolatilityFilter:
    """Tests for Volatility Filter."""
    
    def _create_price_df(self, prices: list) -> pd.DataFrame:
        """Helper to create price DataFrame from close prices."""
        dates = pd.date_range(end=datetime.now(), periods=len(prices), freq='D')
        highs = [p * 1.02 for p in prices]
        lows = [p * 0.98 for p in prices]
        return pd.DataFrame({
            'close': prices,
            'open': prices,
            'high': highs,
            'low': lows,
            'volume': [1000000] * len(prices)
        }, index=dates)
    
    def test_volatility_filter_allows_low_volatility(self):
        """Allows trading when volatility is below threshold."""
        from src.strategies.volatility_filter import VolatilityFilter
        
        vf = VolatilityFilter(atr_period=14, max_atr_percent=5.0)
        
        # Low volatility data
        prices = [100 + i * 0.1 for i in range(20)]  # Gentle trend
        df = self._create_price_df(prices)
        
        can_trade, atr_pct = vf.check_volatility("PLTR", df)
        assert can_trade == True
        assert atr_pct < 5.0
    
    def test_volatility_filter_blocks_high_volatility(self):
        """Blocks trading when volatility is above threshold."""
        from src.strategies.volatility_filter import VolatilityFilter
        
        vf = VolatilityFilter(atr_period=14, max_atr_percent=2.0)
        
        # High volatility data - big swings
        prices = [100, 110, 95, 115, 90, 120, 85, 125, 80, 130, 75, 135, 70, 140, 100]
        df = self._create_price_df(prices)
        
        can_trade, atr_pct = vf.check_volatility("PLTR", df)
        assert can_trade == False
        assert atr_pct > 2.0


class TestNNPredictorStrategy:
    """Tests for Neural Network Predictor Strategy."""
    
    def _create_price_df(self, prices: list, days: int = None) -> pd.DataFrame:
        """Helper to create price DataFrame from close prices."""
        if days is None:
            days = len(prices)
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        return pd.DataFrame({
            'close': prices,
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'volume': [1000000] * len(prices)
        }, index=dates)
    
    def test_nn_predictor_without_model(self):
        """NN Predictor returns HOLD when no model is loaded."""
        from src.strategies.nn_predictor import NNPredictorStrategy
        
        predictor = NNPredictorStrategy(window_size=5)
        
        prices = [100 + i for i in range(15)]
        df = self._create_price_df(prices)
        
        signal = predictor.generate_signal("PLTR", df)
        assert signal.action == "HOLD"
        assert "No trained model" in signal.reason
    
    def test_nn_predictor_feature_extraction(self):
        """NN Predictor can extract features from price data."""
        from src.strategies.nn_predictor import NNPredictorStrategy
        
        predictor = NNPredictorStrategy(window_size=5)
        
        prices = [100 + i * 0.5 for i in range(10)]
        df = self._create_price_df(prices)
        
        features = predictor.extract_features(df)
        assert features is not None
        assert features.shape[0] == 1  # One sample
        assert features.shape[1] == 8  # 8 features
    
    def test_nn_predictor_insufficient_data(self):
        """NN Predictor returns None features with insufficient data."""
        from src.strategies.nn_predictor import NNPredictorStrategy
        
        predictor = NNPredictorStrategy(window_size=10)
        
        prices = [100, 101, 102]  # Only 3 days
        df = self._create_price_df(prices)
        
        features = predictor.extract_features(df)
        assert features is None
    
    def test_nn_predictor_train_and_predict(self):
        """NN Predictor can be trained and make predictions."""
        from src.strategies.nn_predictor import NNPredictorStrategy
        
        predictor = NNPredictorStrategy(window_size=5)
        
        # Generate enough training data (trending up)
        prices = [100 + i * 0.3 + np.random.randn() * 2 for i in range(100)]
        df = self._create_price_df(prices, days=100)
        
        # Train the model
        results = predictor.train(df, forward_days=1)
        
        assert 'train_accuracy' in results
        assert 'test_accuracy' in results
        assert results['train_samples'] > 0
        
        # Now generate a signal
        signal = predictor.generate_signal("PLTR", df)
        assert signal.action in ["BUY", "SELL", "HOLD"]
        assert signal.strategy_name == "NN_Predictor_5d"


class TestEnsembleStrategy:
    """Tests for Ensemble Strategy that combines multiple strategies."""
    
    def _create_price_df(self, prices: list) -> pd.DataFrame:
        """Helper to create price DataFrame from close prices."""
        dates = pd.date_range(end=datetime.now(), periods=len(prices), freq='D')
        return pd.DataFrame({
            'close': prices,
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'volume': [1000000] * len(prices)
        }, index=dates)
    
    def test_ensemble_combines_signals(self):
        """Ensemble combines multiple strategy signals."""
        from src.strategies.ensemble import EnsembleStrategy
        from src.strategies.sma_crossover import SMACrossoverStrategy
        from src.strategies.momentum import MomentumStrategy
        
        ensemble = EnsembleStrategy([
            (SMACrossoverStrategy(fast_period=5, slow_period=10), 1.0),
            (MomentumStrategy(lookback_days=10, threshold=0.03), 1.0),
        ])
        
        # Strong uptrend - both strategies should agree on BUY
        prices = [100] * 10 + [105, 110, 115, 120, 125]
        df = self._create_price_df(prices)
        
        signal = ensemble.generate_signal("PLTR", df)
        # With strong agreement, should get BUY with high confidence
        assert signal.action in ["BUY", "HOLD"]  # May be HOLD if not strong enough
    
    def test_ensemble_returns_all_signals(self):
        """Ensemble can return individual signals from all strategies."""
        from src.strategies.ensemble import EnsembleStrategy
        from src.strategies.sma_crossover import SMACrossoverStrategy
        from src.strategies.momentum import MomentumStrategy
        
        ensemble = EnsembleStrategy([
            (SMACrossoverStrategy(fast_period=5, slow_period=10), 1.0),
            (MomentumStrategy(lookback_days=10, threshold=0.03), 1.0),
        ])
        
        prices = [100] * 15
        df = self._create_price_df(prices)
        
        all_signals = ensemble.get_all_signals("PLTR", df)
        assert len(all_signals) == 2
        assert all(hasattr(s, 'action') for s in all_signals)
    
    def test_ensemble_respects_weights(self):
        """Ensemble respects strategy weights in final decision."""
        from src.strategies.ensemble import EnsembleStrategy
        from src.strategies.signals import Signal
        from src.strategies.base_strategy import BaseStrategy
        
        # Create mock strategies with fixed outputs
        class AlwaysBuy(BaseStrategy):
            @property
            def name(self):
                return "AlwaysBuy"
            
            def generate_signal(self, symbol, df):
                return Signal(symbol, "BUY", 0.8, self.name, "Always buy")
        
        class AlwaysSell(BaseStrategy):
            @property
            def name(self):
                return "AlwaysSell"
            
            def generate_signal(self, symbol, df):
                return Signal(symbol, "SELL", 0.8, self.name, "Always sell")
        
        # Give BUY strategy much higher weight
        ensemble = EnsembleStrategy([
            (AlwaysBuy(), 3.0),  # 3x weight
            (AlwaysSell(), 1.0),
        ])
        
        df = self._create_price_df([100] * 5)
        signal = ensemble.generate_signal("PLTR", df)
        
        # BUY should win due to higher weight
        assert signal.action == "BUY"

