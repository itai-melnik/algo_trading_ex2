#!/usr/bin/env python3
"""
Neural Network Training Script.

Trains the price direction predictor using generated mock data
or real historical data.

Usage:
    python scripts/train_nn.py              # Train on mock data
    python scripts/train_nn.py --symbol PLTR --days 365  # Train on specific data
"""
import os
import sys
import argparse
from datetime import datetime, timedelta
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategies.nn_predictor import NNPredictorStrategy
from src.data.mock_data import MockDataProvider


def generate_training_data(
    symbols: list,
    days: int = 365,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic training data using MockDataProvider.
    
    Returns combined price data for all symbols.
    """
    provider = MockDataProvider(seed=seed)
    
    end_date = datetime(2023, 12, 31)
    start_date = end_date - timedelta(days=days)
    
    all_data = []
    
    for symbol in symbols:
        price_data = provider.get_price_history(
            symbol,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        
        if price_data:
            df = pd.DataFrame.from_dict(price_data, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df['symbol'] = symbol
            all_data.append(df)
    
    if all_data:
        combined = pd.concat(all_data)
        combined = combined.sort_index()
        return combined
    
    return pd.DataFrame()


def train_model(
    symbols: list = ['PLTR', 'NFLX', 'PLTK'],
    days: int = 365,
    window_size: int = 10,
    output_path: str = 'models/price_predictor.pkl'
):
    """
    Train the NN predictor model.
    
    Args:
        symbols: List of stock symbols to train on
        days: Number of days of historical data
        window_size: Feature window size
        output_path: Where to save the trained model
    """
    print("=" * 60)
    print("🧠 Neural Network Training")
    print("=" * 60)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Training days: {days}")
    print(f"Window size: {window_size}")
    print(f"Output path: {output_path}")
    print("=" * 60)
    
    # Generate training data
    print("\n📊 Generating training data...")
    training_data = generate_training_data(symbols, days)
    print(f"   Total samples: {len(training_data)}")
    
    if training_data.empty:
        print("❌ No training data generated!")
        return
    
    # Train on each symbol separately and aggregate
    all_results = []
    
    for symbol in symbols:
        symbol_data = training_data[training_data['symbol'] == symbol].copy()
        if len(symbol_data) < window_size + 10:
            print(f"   ⚠️ Skipping {symbol}: insufficient data")
            continue
        
        print(f"\n🔧 Training on {symbol} ({len(symbol_data)} samples)...")
        
        # Create and train predictor
        predictor = NNPredictorStrategy(window_size=window_size)
        
        try:
            results = predictor.train(symbol_data, forward_days=1)
            all_results.append({
                'symbol': symbol,
                **results
            })
            print(f"   ✅ Train accuracy: {results['train_accuracy']:.1%}")
            print(f"   ✅ Test accuracy: {results['test_accuracy']:.1%}")
        except Exception as e:
            print(f"   ❌ Training failed: {e}")
    
    # Train final model on all data
    print("\n🔧 Training final model on combined data...")
    
    # Remove symbol column for training
    combined_training = training_data.copy()
    if 'symbol' in combined_training.columns:
        combined_training = combined_training.drop(columns=['symbol'])
    
    final_predictor = NNPredictorStrategy(window_size=window_size)
    
    try:
        final_results = final_predictor.train(combined_training, forward_days=1)
        print(f"   ✅ Train accuracy: {final_results['train_accuracy']:.1%}")
        print(f"   ✅ Test accuracy: {final_results['test_accuracy']:.1%}")
        
        # Save model
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_predictor.save_model(output_path)
        print(f"\n💾 Model saved to: {output_path}")
        
    except Exception as e:
        print(f"   ❌ Final training failed: {e}")
        return
    
    # Summary
    print("\n" + "=" * 60)
    print("📈 Training Summary")
    print("=" * 60)
    
    for result in all_results:
        print(f"   {result['symbol']}: "
              f"Train {result['train_accuracy']:.1%}, "
              f"Test {result['test_accuracy']:.1%}")
    
    print(f"\n   Combined: "
          f"Train {final_results['train_accuracy']:.1%}, "
          f"Test {final_results['test_accuracy']:.1%}")
    
    print("\n✅ Training complete!")
    
    return final_predictor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NN price predictor")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=['PLTR', 'NFLX', 'PLTK'],
        help="Symbols to train on"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Days of training data (default: 365)"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Feature window size (default: 10)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/price_predictor.pkl",
        help="Output model path"
    )
    
    args = parser.parse_args()
    
    train_model(
        symbols=args.symbols,
        days=args.days,
        window_size=args.window,
        output_path=args.output
    )

