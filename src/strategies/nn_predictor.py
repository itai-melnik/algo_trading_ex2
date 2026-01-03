"""
Neural Network Price Predictor Strategy.

Uses a simple MLP (Multi-Layer Perceptron) to predict price direction
based on recent price features.
"""
import os
import pickle
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from .base_strategy import BaseStrategy
from .signals import Signal


class NNPredictorStrategy(BaseStrategy):
    """
    Neural Network strategy that predicts price direction.
    
    Uses a trained MLP model to predict whether the price will go up or down.
    The model is trained on features extracted from price history.
    
    Args:
        model_path: Path to saved model file (.pkl)
        window_size: Number of days to use for feature extraction
        confidence_threshold: Minimum prediction probability to generate signal
    """
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        window_size: int = 10,
        confidence_threshold: float = 0.6
    ):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.scaler = None
        
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
    
    @property
    def name(self) -> str:
        return f"NN_Predictor_{self.window_size}d"
    
    def _load_model(self, model_path: str):
        """Load trained model and scaler from file."""
        try:
            with open(model_path, 'rb') as f:
                saved = pickle.load(f)
                self.model = saved.get('model')
                self.scaler = saved.get('scaler')
        except Exception as e:
            print(f"Warning: Could not load NN model from {model_path}: {e}")
            self.model = None
            self.scaler = None
    
    def save_model(self, model_path: str):
        """Save trained model and scaler to file."""
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler
            }, f)
    
    def extract_features(self, price_history: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Extract features from price history for prediction.
        
        Features:
        - Returns over various periods (1, 3, 5, 10 days)
        - Volatility (rolling std)
        - Price relative to recent high/low
        - Volume changes
        """
        if len(price_history) < self.window_size + 1:
            return None
        
        closes = price_history['close'].values
        volumes = price_history['volume'].values if 'volume' in price_history else np.ones(len(closes))
        highs = price_history['high'].values
        lows = price_history['low'].values
        
        # Use last window_size days
        recent_closes = closes[-(self.window_size + 1):]
        recent_volumes = volumes[-(self.window_size + 1):]
        recent_highs = highs[-(self.window_size + 1):]
        recent_lows = lows[-(self.window_size + 1):]
        
        features = []
        
        # Returns at various lookbacks
        for lookback in [1, 3, 5, min(10, self.window_size)]:
            if len(recent_closes) > lookback:
                ret = (recent_closes[-1] - recent_closes[-lookback-1]) / recent_closes[-lookback-1]
                features.append(ret)
            else:
                features.append(0.0)
        
        # Rolling volatility (normalized)
        if len(recent_closes) >= 5:
            returns = np.diff(recent_closes) / recent_closes[:-1]
            volatility = np.std(returns[-5:]) if len(returns) >= 5 else 0
            features.append(volatility)
        else:
            features.append(0.0)
        
        # Price relative to recent high/low
        recent_high = np.max(recent_highs)
        recent_low = np.min(recent_lows)
        price_range = recent_high - recent_low
        if price_range > 0:
            position = (recent_closes[-1] - recent_low) / price_range
        else:
            position = 0.5
        features.append(position)
        
        # Volume change (normalized)
        if len(recent_volumes) >= 2 and recent_volumes[-2] > 0:
            vol_change = (recent_volumes[-1] - recent_volumes[-2]) / recent_volumes[-2]
            features.append(np.clip(vol_change, -1, 1))
        else:
            features.append(0.0)
        
        # Average volume ratio
        avg_vol = np.mean(recent_volumes[:-1])
        if avg_vol > 0:
            vol_ratio = recent_volumes[-1] / avg_vol
            features.append(np.clip(vol_ratio - 1, -1, 1))
        else:
            features.append(0.0)
        
        return np.array(features).reshape(1, -1)
    
    def generate_signal(self, symbol: str, price_history: pd.DataFrame) -> Signal:
        """
        Generate signal based on NN prediction.
        
        Args:
            symbol: Stock ticker
            price_history: DataFrame with OHLCV data
            
        Returns:
            Signal with BUY/SELL/HOLD action based on prediction
        """
        if self.model is None:
            return Signal(
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reason="No trained model available"
            )
        
        # Extract features
        features = self.extract_features(price_history)
        if features is None:
            return Signal(
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reason=f"Insufficient data (need {self.window_size + 1} days)"
            )
        
        try:
            # Scale features if scaler available
            if self.scaler is not None:
                features = self.scaler.transform(features)
            
            # Get prediction probabilities
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features)[0]
                # Assume: class 0 = DOWN, class 1 = UP
                up_prob = proba[1] if len(proba) > 1 else proba[0]
                down_prob = proba[0] if len(proba) > 1 else 1 - proba[0]
            else:
                # If no probability, just get prediction
                pred = self.model.predict(features)[0]
                up_prob = 1.0 if pred == 1 else 0.0
                down_prob = 1.0 if pred == 0 else 0.0
            
            # Generate signal based on prediction
            if up_prob >= self.confidence_threshold:
                return Signal(
                    symbol=symbol,
                    action="BUY",
                    confidence=float(up_prob),
                    strategy_name=self.name,
                    reason=f"NN predicts UP with {up_prob:.0%} probability"
                )
            elif down_prob >= self.confidence_threshold:
                return Signal(
                    symbol=symbol,
                    action="SELL",
                    confidence=float(down_prob),
                    strategy_name=self.name,
                    reason=f"NN predicts DOWN with {down_prob:.0%} probability"
                )
            else:
                return Signal(
                    symbol=symbol,
                    action="HOLD",
                    confidence=float(max(up_prob, down_prob)),
                    strategy_name=self.name,
                    reason=f"NN prediction below threshold (UP: {up_prob:.0%}, DOWN: {down_prob:.0%})"
                )
        
        except Exception as e:
            return Signal(
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reason=f"Prediction error: {str(e)}"
            )
    
    def train(self, price_data: pd.DataFrame, forward_days: int = 1):
        """
        Train the NN model on historical price data.
        
        Args:
            price_data: DataFrame with OHLCV columns
            forward_days: Number of days ahead to predict
        """
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        
        # Prepare training data
        X = []
        y = []
        
        for i in range(self.window_size, len(price_data) - forward_days):
            window = price_data.iloc[i - self.window_size:i + 1]
            features = self.extract_features(window)
            
            if features is not None:
                # Target: 1 if price goes up, 0 if down
                current_price = price_data['close'].iloc[i]
                future_price = price_data['close'].iloc[i + forward_days]
                target = 1 if future_price > current_price else 0
                
                X.append(features.flatten())
                y.append(target)
        
        if len(X) < 50:
            raise ValueError(f"Not enough training samples: {len(X)}")
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train MLP
        self.model = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_acc = self.model.score(X_train_scaled, y_train)
        test_acc = self.model.score(X_test_scaled, y_test)
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }

