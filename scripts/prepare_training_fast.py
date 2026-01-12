"""
Fast Training Data Preparation for Vast.ai H100
================================================
Vectorized feature computation for maximum speed.

Usage:
    python scripts/prepare_training_fast.py --symbols EURUSD,GBPUSD,USDJPY
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_dukascopy_data(symbol: str) -> pd.DataFrame:
    """Load all available Dukascopy tick data for symbol."""
    data_dir = Path("data_cleaned/dukascopy_local")
    files = sorted(data_dir.glob(f"{symbol}_*.csv"))

    if not files:
        logger.warning(f"No Dukascopy data found for {symbol}")
        return pd.DataFrame()

    frames = []
    for f in files:
        df = pd.read_csv(f)
        frames.append(df)
        logger.info(f"Loaded {len(df)} ticks from {f.name}")

    data = pd.concat(frames, ignore_index=True)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['mid'] = (data['bid'] + data['ask']) / 2
    data['spread'] = data['ask'] - data['bid']
    data['volume'] = data.get('bid_volume', 1) + data.get('ask_volume', 1)

    logger.info(f"Total {len(data)} ticks for {symbol}")
    return data


def compute_fast_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features using vectorized operations."""
    logger.info("Computing vectorized features...")

    mid = df['mid'].values
    bid = df['bid'].values
    ask = df['ask'].values
    spread = df['spread'].values
    volume = df['volume'].values

    # Returns at different lags
    for lag in [1, 5, 10, 20, 50, 100]:
        df[f'ret_{lag}'] = pd.Series(mid).pct_change(lag).values * 10000  # bps

    # Volatility (rolling std)
    for window in [10, 20, 50, 100, 200]:
        df[f'vol_{window}'] = pd.Series(mid).pct_change().rolling(window).std().values * 10000

    # Z-scores
    for window in [20, 50, 100]:
        rolling_mean = pd.Series(mid).rolling(window).mean()
        rolling_std = pd.Series(mid).rolling(window).std()
        df[f'zscore_{window}'] = ((mid - rolling_mean) / (rolling_std + 1e-10)).values

    # Price range position
    for window in [20, 50, 100]:
        rolling_max = pd.Series(mid).rolling(window).max()
        rolling_min = pd.Series(mid).rolling(window).min()
        df[f'range_pos_{window}'] = ((mid - rolling_min) / (rolling_max - rolling_min + 1e-10)).values

    # Momentum indicators
    for window in [5, 10, 20, 50]:
        df[f'roc_{window}'] = pd.Series(mid).pct_change(window).values * 10000

    # Moving averages
    for window in [5, 10, 20, 50, 100, 200]:
        df[f'ma_{window}'] = pd.Series(mid).rolling(window).mean().values
        df[f'ma_ratio_{window}'] = (mid / df[f'ma_{window}'].values - 1) * 10000

    # EMA
    for span in [5, 10, 20, 50]:
        df[f'ema_{span}'] = pd.Series(mid).ewm(span=span).mean().values
        df[f'ema_ratio_{span}'] = (mid / df[f'ema_{span}'].values - 1) * 10000

    # Spread features
    df['spread_bps'] = spread / mid * 10000
    df['spread_ma_20'] = pd.Series(spread).rolling(20).mean().values
    df['spread_ratio'] = spread / (df['spread_ma_20'] + 1e-10)

    # Volume features
    df['volume_ma_20'] = pd.Series(volume).rolling(20).mean().values
    df['volume_ratio'] = volume / (df['volume_ma_20'] + 1e-10)

    # Bid-Ask imbalance
    if 'bid_volume' in df.columns and 'ask_volume' in df.columns:
        bid_vol = df['bid_volume'].values
        ask_vol = df['ask_volume'].values
        df['imbalance'] = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-10)

    # RSI
    for window in [14, 28]:
        delta = pd.Series(mid).diff()
        gain = delta.where(delta > 0, 0).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / (loss + 1e-10)
        df[f'rsi_{window}'] = (100 - 100 / (1 + rs)).values

    # MACD
    ema_12 = pd.Series(mid).ewm(span=12).mean()
    ema_26 = pd.Series(mid).ewm(span=26).mean()
    df['macd'] = (ema_12 - ema_26).values
    df['macd_signal'] = pd.Series(df['macd']).ewm(span=9).mean().values
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Bollinger Bands
    for window in [20, 50]:
        ma = pd.Series(mid).rolling(window).mean()
        std = pd.Series(mid).rolling(window).std()
        df[f'bb_upper_{window}'] = (ma + 2 * std).values
        df[f'bb_lower_{window}'] = (ma - 2 * std).values
        df[f'bb_pct_{window}'] = ((mid - df[f'bb_lower_{window}']) /
                                  (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}'] + 1e-10)).values

    # ATR proxy (using high/low approximation from tick data)
    high = pd.Series(mid).rolling(20).max()
    low = pd.Series(mid).rolling(20).min()
    df['atr_20'] = (high - low).values
    df['atr_ratio'] = df['atr_20'] / mid

    # Trend strength
    for window in [20, 50, 100]:
        ma = pd.Series(mid).rolling(window).mean()
        df[f'trend_{window}'] = ((mid - ma) / (ma * 0.0001 + 1e-10)).values  # in pips

    # Acceleration
    ret_1 = pd.Series(mid).pct_change()
    df['acceleration'] = ret_1.diff().values * 10000

    # Skewness and Kurtosis
    for window in [50, 100]:
        df[f'skew_{window}'] = pd.Series(mid).pct_change().rolling(window).skew().values
        df[f'kurt_{window}'] = pd.Series(mid).pct_change().rolling(window).kurt().values

    # Time features
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'])
        df['hour'] = ts.dt.hour
        df['minute'] = ts.dt.minute
        df['dayofweek'] = ts.dt.dayofweek
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    logger.info(f"Generated {len([c for c in df.columns if c not in ['timestamp', 'bid', 'ask', 'mid', 'spread', 'volume', 'bid_volume', 'ask_volume', 'pair']])} features")
    return df


def create_targets(df: pd.DataFrame, horizons: list = None) -> pd.DataFrame:
    """Create prediction targets."""
    if horizons is None:
        horizons = [1, 5, 10, 20, 50, 100]

    mid = df['mid']

    for h in horizons:
        future_mid = mid.shift(-h)
        ret = (np.log(future_mid) - np.log(mid)) * 10000  # bps

        df[f'target_return_{h}'] = ret
        df[f'target_direction_{h}'] = (ret > 0).astype(int)
        df[f'target_up_{h}'] = (ret > 0.5).astype(int)  # > 0.5 bps
        df[f'target_down_{h}'] = (ret < -0.5).astype(int)  # < -0.5 bps

    return df


def create_training_package(symbols: list, output_dir: Path):
    """Create complete training package."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {symbol}")
        logger.info(f"{'='*50}")

        # Load data
        df = load_dukascopy_data(symbol)
        if df.empty:
            logger.warning(f"No data for {symbol}, skipping")
            continue

        # Compute features
        df = compute_fast_features(df)

        # Create targets
        df = create_targets(df)

        # Drop rows with NaN
        initial_len = len(df)
        df = df.dropna()
        logger.info(f"Dropped {initial_len - len(df)} rows with NaN")

        # Split train/val/test (70/15/15)
        n = len(df)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)

        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]

        logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

        # Save as parquet
        symbol_dir = output_dir / symbol
        symbol_dir.mkdir(exist_ok=True)

        train.to_parquet(symbol_dir / "train.parquet", index=False)
        val.to_parquet(symbol_dir / "val.parquet", index=False)
        test.to_parquet(symbol_dir / "test.parquet", index=False)

        # Save metadata
        feature_cols = [c for c in df.columns if not c.startswith('target_') and
                       c not in ['timestamp', 'bid', 'ask', 'mid', 'spread', 'volume',
                                'bid_volume', 'ask_volume', 'pair']]
        target_cols = [c for c in df.columns if c.startswith('target_')]

        metadata = {
            'symbol': symbol,
            'n_samples': {'train': len(train), 'val': len(val), 'test': len(test)},
            'n_features': len(feature_cols),
            'features': feature_cols,
            'targets': target_cols,
            'created': datetime.now().isoformat()
        }

        with open(symbol_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved {symbol} to {symbol_dir}")

    # Create Vast.ai training script
    create_vastai_script(output_dir, symbols)

    logger.info(f"\n{'='*50}")
    logger.info(f"Training package ready at {output_dir}")
    logger.info(f"{'='*50}")


def create_vastai_script(output_dir: Path, symbols: list):
    """Create training script for Vast.ai H100."""

    train_script = '''#!/usr/bin/env python3
"""
HFT Model Training for Vast.ai H100
====================================
Run: python train_models.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle
from datetime import datetime

# GPU-accelerated libraries
try:
    import xgboost as xgb
    import lightgbm as lgb
    import catboost as cb
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("Warning: GPU libraries not found, using CPU")

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


def load_data(symbol_dir):
    """Load train/val/test data."""
    train = pd.read_parquet(symbol_dir / "train.parquet")
    val = pd.read_parquet(symbol_dir / "val.parquet")
    test = pd.read_parquet(symbol_dir / "test.parquet")

    with open(symbol_dir / "metadata.json") as f:
        metadata = json.load(f)

    return train, val, test, metadata


def train_xgboost(X_train, y_train, X_val, y_val, target_name):
    """Train XGBoost with GPU."""
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'gpu_hist' if HAS_GPU else 'hist',
        'device': 'cuda' if HAS_GPU else 'cpu',
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=100
    )

    return model


def train_lightgbm(X_train, y_train, X_val, y_val, target_name):
    """Train LightGBM with GPU."""
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.1,
        'num_leaves': 31,
        'device': 'gpu' if HAS_GPU else 'cpu',
        'verbose': -1
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50)]
    )

    return model


def train_catboost(X_train, y_train, X_val, y_val, target_name):
    """Train CatBoost with GPU."""
    model = cb.CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        loss_function='Logloss',
        eval_metric='AUC',
        task_type='GPU' if HAS_GPU else 'CPU',
        early_stopping_rounds=50,
        verbose=100
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val))

    return model


def train_symbol(symbol_dir, output_dir):
    """Train all models for a symbol."""
    symbol = symbol_dir.name
    print(f"\\n{'='*50}")
    print(f"Training {symbol}")
    print(f"{'='*50}")

    train, val, test, metadata = load_data(symbol_dir)

    features = metadata['features']
    targets = [t for t in metadata['targets'] if 'direction' in t]

    X_train = train[features].values
    X_val = val[features].values
    X_test = test[features].values

    results = {}
    models = {}

    for target in targets[:3]:  # Train on top 3 horizons
        print(f"\\nTarget: {target}")

        y_train = train[target].values
        y_val = val[target].values
        y_test = test[target].values

        # Train ensemble
        xgb_model = train_xgboost(X_train, y_train, X_val, y_val, target)
        lgb_model = train_lightgbm(X_train, y_train, X_val, y_val, target)
        cb_model = train_catboost(X_train, y_train, X_val, y_val, target)

        # Ensemble predictions
        xgb_pred = xgb_model.predict(xgb.DMatrix(X_test))
        lgb_pred = lgb_model.predict(X_test)
        cb_pred = cb_model.predict_proba(X_test)[:, 1]

        ensemble_pred = (xgb_pred + lgb_pred + cb_pred) / 3
        ensemble_class = (ensemble_pred > 0.5).astype(int)

        # Metrics
        acc = accuracy_score(y_test, ensemble_class)
        auc = roc_auc_score(y_test, ensemble_pred)
        f1 = f1_score(y_test, ensemble_class)

        print(f"  Accuracy: {acc:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  F1: {f1:.4f}")

        results[target] = {'accuracy': acc, 'auc': auc, 'f1': f1}
        models[target] = {
            'xgboost': xgb_model,
            'lightgbm': lgb_model,
            'catboost': cb_model,
            'features': features
        }

    # Save models
    model_file = output_dir / f"{symbol}_models.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(models, f)

    # Save results
    results_file = output_dir / f"{symbol}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\\nSaved models to {model_file}")
    return results


def main():
    """Train all symbols."""
    data_dir = Path(".")
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)

    symbols = [d.name for d in data_dir.iterdir()
               if d.is_dir() and (d / "train.parquet").exists()]

    print(f"Found symbols: {symbols}")

    all_results = {}
    for symbol_dir in [data_dir / s for s in symbols]:
        results = train_symbol(symbol_dir, output_dir)
        all_results[symbol_dir.name] = results

    # Summary
    print(f"\\n{'='*50}")
    print("TRAINING COMPLETE")
    print(f"{'='*50}")

    for symbol, results in all_results.items():
        print(f"\\n{symbol}:")
        for target, metrics in results.items():
            print(f"  {target}: ACC={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}")

    # Package models
    import shutil
    shutil.make_archive('hft_models', 'gztar', output_dir)
    print(f"\\nModels packaged to hft_models.tar.gz")


if __name__ == "__main__":
    main()
'''

    with open(output_dir / "train_models.py", 'w') as f:
        f.write(train_script)

    # Requirements
    requirements = """numpy>=1.21.0
pandas>=1.3.0
pyarrow>=8.0.0
xgboost>=1.6.0
lightgbm>=3.3.0
catboost>=1.0.0
scikit-learn>=1.0.0
"""

    with open(output_dir / "requirements.txt", 'w') as f:
        f.write(requirements)

    # Run script
    run_script = """#!/bin/bash
# Vast.ai H100 Training Script
# Upload this package and run: ./run_training.sh

pip install -r requirements.txt
python train_models.py

echo "Training complete! Download hft_models.tar.gz"
"""

    with open(output_dir / "run_training.sh", 'w') as f:
        f.write(run_script)

    logger.info(f"Created Vast.ai training scripts")


def main():
    parser = argparse.ArgumentParser(description="Prepare HFT training data")
    parser.add_argument("--symbols", default="EURUSD,GBPUSD,USDJPY", help="Comma-separated symbols")
    parser.add_argument("--output", default="training_package", help="Output directory")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    output_dir = Path(args.output)

    create_training_package(symbols, output_dir)


if __name__ == "__main__":
    main()
