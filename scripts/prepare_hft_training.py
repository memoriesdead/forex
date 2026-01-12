"""
HFT Training Data Preparation for Vast.ai
==========================================
Packages historical data with features for H100 GPU training.

Workflow:
1. Load historical tick data from all sources
2. Generate 100+ features per tick
3. Create target labels (direction, returns)
4. Split train/val/test
5. Package as .parquet for fast loading
6. Generate training script

Usage:
    python scripts/prepare_hft_training.py --symbols EURUSD,GBPUSD --days 30
    # Upload to Vast.ai and run train_hft_models.py
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import logging
import json
import pickle
import gzip
import shutil
from typing import Dict, List, Tuple, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HFTTrainingPackager:
    """
    Packages HFT data for Vast.ai GPU training.
    """

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("training_package")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Lazy load components
        self._data_loader = None
        self._feature_engine = None

    def _get_data_loader(self):
        """Lazy load data loader."""
        if self._data_loader is None:
            from core.hft_data_loader import UnifiedDataLoader
            self._data_loader = UnifiedDataLoader()
        return self._data_loader

    def _get_feature_engine(self):
        """Lazy load feature engine."""
        if self._feature_engine is None:
            from core.hft_feature_engine import HFTFeatureEngine
            self._feature_engine = HFTFeatureEngine()
        return self._feature_engine

    def load_historical_data(self, symbol: str, days: int = 30,
                             source: str = 'truefx') -> pd.DataFrame:
        """Load historical tick data."""
        logger.info(f"Loading {days} days of {symbol} data from {source}")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        loader = self._get_data_loader()

        try:
            data = loader.load_historical(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                source=source
            )
            logger.info(f"Loaded {len(data)} ticks for {symbol}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load from {source}: {e}")
            # Try alternative source
            if source == 'truefx':
                return self.load_historical_data(symbol, days, 'oracle')
            else:
                # Generate synthetic data for testing
                return self._generate_synthetic_data(symbol, days)

    def _generate_synthetic_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Generate synthetic data for testing pipeline."""
        logger.warning(f"Generating synthetic data for {symbol} (testing only)")

        np.random.seed(hash(symbol) % 2**32)

        # Approximate ticks per day (forex = ~100k ticks/day)
        ticks_per_day = 100000
        n_ticks = days * ticks_per_day

        # Reduce for testing
        n_ticks = min(n_ticks, 1000000)

        # Base price for symbol
        base_prices = {
            'EURUSD': 1.1000,
            'GBPUSD': 1.2500,
            'USDJPY': 150.00,
            'USDCHF': 0.8800,
            'AUDUSD': 0.6500,
            'NZDUSD': 0.6000,
            'USDCAD': 1.3500,
        }
        base = base_prices.get(symbol, 1.0)

        # Generate realistic tick data
        returns = np.random.randn(n_ticks) * 0.0001  # 1 pip std
        prices = base * np.exp(np.cumsum(returns))

        # Add spread
        spread = base * 0.00005  # 0.5 pip spread
        bids = prices - spread / 2
        asks = prices + spread / 2

        # Generate timestamps (approximately uniform during market hours)
        timestamps = pd.date_range(
            end=datetime.now(),
            periods=n_ticks,
            freq='100ms'
        )

        # Volume (exponential distribution)
        volumes = np.random.exponential(100000, n_ticks)

        data = pd.DataFrame({
            'timestamp': timestamps,
            'bid': bids,
            'ask': asks,
            'volume': volumes
        })

        logger.info(f"Generated {len(data)} synthetic ticks for {symbol}")
        return data

    def generate_features(self, data: pd.DataFrame,
                          symbol: str) -> pd.DataFrame:
        """Generate all HFT features."""
        logger.info(f"Generating features for {len(data)} ticks")

        engine = self._get_feature_engine()
        featured_data = engine.batch_process(data, symbol)

        # Count features
        feature_cols = [c for c in featured_data.columns
                       if c not in ['timestamp', 'bid', 'ask', 'volume']]
        logger.info(f"Generated {len(feature_cols)} features")

        return featured_data

    def create_labels(self, data: pd.DataFrame,
                      horizons: List[int] = None) -> pd.DataFrame:
        """
        Create target labels for prediction.

        Labels:
        - direction_N: 1 if price up, -1 if down after N ticks
        - return_N: Log return after N ticks (in bps)
        - vol_N: Realized volatility over next N ticks
        """
        if horizons is None:
            horizons = [1, 5, 10, 20, 50, 100]

        logger.info(f"Creating labels for horizons: {horizons}")

        # Get mid price
        if 'mid_price' in data.columns:
            mid = data['mid_price']
        elif 'bid' in data.columns and 'ask' in data.columns:
            mid = (data['bid'] + data['ask']) / 2
        else:
            mid = data['close']

        result = data.copy()

        for h in horizons:
            # Future return
            future_price = mid.shift(-h)
            returns = (np.log(future_price) - np.log(mid)) * 10000  # bps

            result[f'target_return_{h}'] = returns

            # Direction (1 = up, 0 = down)
            result[f'target_direction_{h}'] = (returns > 0).astype(int)

            # Binary classification with threshold
            threshold = 0.5  # 0.5 bps = 0.5 pips for EURUSD
            result[f'target_up_{h}'] = (returns > threshold).astype(int)
            result[f'target_down_{h}'] = (returns < -threshold).astype(int)

        # Realized volatility targets
        for h in [10, 20, 50]:
            if h < len(mid):
                # Forward-looking volatility
                vol = mid.rolling(h).std().shift(-h) * 10000
                result[f'target_vol_{h}'] = vol

        # Drop rows with NaN targets
        target_cols = [c for c in result.columns if c.startswith('target_')]
        n_before = len(result)
        result = result.dropna(subset=target_cols)
        logger.info(f"Dropped {n_before - len(result)} rows with NaN targets")

        return result

    def split_data(self, data: pd.DataFrame,
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test sets.
        Uses time-based split (no shuffling for time series).
        """
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train = data.iloc[:train_end].copy()
        val = data.iloc[train_end:val_end].copy()
        test = data.iloc[val_end:].copy()

        logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

        return train, val, test

    def package_symbol(self, symbol: str, days: int = 30,
                       source: str = 'truefx') -> Dict[str, Path]:
        """
        Package all data for a single symbol.

        Returns dict of output file paths.
        """
        logger.info(f"Packaging {symbol}...")

        # 1. Load data
        data = self.load_historical_data(symbol, days, source)

        # 2. Generate features
        featured = self.generate_features(data, symbol)

        # 3. Create labels
        labeled = self.create_labels(featured)

        # 4. Split data
        train, val, test = self.split_data(labeled)

        # 5. Save as parquet (compressed, fast loading)
        symbol_dir = self.output_dir / symbol
        symbol_dir.mkdir(exist_ok=True)

        paths = {}

        for name, df in [('train', train), ('val', val), ('test', test)]:
            path = symbol_dir / f"{name}.parquet"
            df.to_parquet(path, compression='snappy')
            paths[name] = path
            logger.info(f"Saved {name}: {path} ({len(df)} rows)")

        # Save feature names
        feature_cols = [c for c in train.columns
                       if not c.startswith('target_')
                       and c not in ['timestamp', 'bid', 'ask', 'volume']]
        target_cols = [c for c in train.columns if c.startswith('target_')]

        metadata = {
            'symbol': symbol,
            'n_features': len(feature_cols),
            'n_targets': len(target_cols),
            'feature_names': feature_cols,
            'target_names': target_cols,
            'train_size': len(train),
            'val_size': len(val),
            'test_size': len(test),
            'created_at': datetime.now().isoformat()
        }

        meta_path = symbol_dir / 'metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        paths['metadata'] = meta_path

        return paths

    def package_all(self, symbols: List[str], days: int = 30,
                    source: str = 'truefx') -> Dict[str, Dict[str, Path]]:
        """Package data for all symbols."""
        all_paths = {}

        for symbol in symbols:
            try:
                paths = self.package_symbol(symbol, days, source)
                all_paths[symbol] = paths
            except Exception as e:
                logger.error(f"Failed to package {symbol}: {e}")

        # Create master metadata
        master_meta = {
            'symbols': symbols,
            'days': days,
            'source': source,
            'created_at': datetime.now().isoformat(),
            'package_contents': {s: {k: str(v) for k, v in p.items()}
                                for s, p in all_paths.items()}
        }

        master_path = self.output_dir / 'package_metadata.json'
        with open(master_path, 'w') as f:
            json.dump(master_meta, f, indent=2)

        return all_paths

    def create_training_script(self) -> Path:
        """Create training script for Vast.ai."""
        script_content = '''#!/usr/bin/env python3
"""
HFT Model Training for Vast.ai H100
===================================
Trains ensemble of models on packaged data.

Models:
- XGBoost
- LightGBM
- CatBoost
- Linear (baseline)

Usage:
    python train_hft_models.py --symbol EURUSD --target target_direction_10
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(symbol: str, data_dir: Path = Path(".")):
    """Load packaged data."""
    symbol_dir = data_dir / symbol

    train = pd.read_parquet(symbol_dir / "train.parquet")
    val = pd.read_parquet(symbol_dir / "val.parquet")
    test = pd.read_parquet(symbol_dir / "test.parquet")

    with open(symbol_dir / "metadata.json") as f:
        metadata = json.load(f)

    return train, val, test, metadata


def prepare_features(df, feature_names, target_name):
    """Extract features and target."""
    X = df[feature_names].fillna(0)
    y = df[target_name]
    return X, y


def train_xgboost(X_train, y_train, X_val, y_val, is_classifier=True):
    """Train XGBoost model."""
    import xgboost as xgb

    if is_classifier:
        model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='auc',
            early_stopping_rounds=50,
            tree_method='gpu_hist',  # GPU acceleration
            gpu_id=0
        )
    else:
        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            early_stopping_rounds=50,
            tree_method='gpu_hist',
            gpu_id=0
        )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
    return model


def train_lightgbm(X_train, y_train, X_val, y_val, is_classifier=True):
    """Train LightGBM model."""
    import lightgbm as lgb

    if is_classifier:
        model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary',
            metric='auc',
            device='gpu',  # GPU acceleration
            gpu_platform_id=0,
            gpu_device_id=0
        )
    else:
        model = lgb.LGBMRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            device='gpu',
            gpu_platform_id=0,
            gpu_device_id=0
        )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50)])
    return model


def train_catboost(X_train, y_train, X_val, y_val, is_classifier=True):
    """Train CatBoost model."""
    from catboost import CatBoostClassifier, CatBoostRegressor

    if is_classifier:
        model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            loss_function='Logloss',
            eval_metric='AUC',
            early_stopping_rounds=50,
            task_type='GPU',  # GPU acceleration
            devices='0'
        )
    else:
        model = CatBoostRegressor(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            early_stopping_rounds=50,
            task_type='GPU',
            devices='0'
        )

    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=100)
    return model


def evaluate_model(model, X_test, y_test, is_classifier=True):
    """Evaluate model performance."""
    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    if is_classifier:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_proba),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred)
        }
    else:
        y_pred = model.predict(X_test)
        return {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default='EURUSD')
    parser.add_argument('--target', type=str, default='target_direction_10')
    parser.add_argument('--data-dir', type=str, default='.')
    parser.add_argument('--output-dir', type=str, default='models')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load data
    logger.info(f"Loading data for {args.symbol}...")
    train, val, test, metadata = load_data(args.symbol, data_dir)

    feature_names = metadata['feature_names']
    target = args.target
    is_classifier = 'direction' in target or 'up' in target or 'down' in target

    logger.info(f"Features: {len(feature_names)}, Target: {target}, Classifier: {is_classifier}")

    # Prepare data
    X_train, y_train = prepare_features(train, feature_names, target)
    X_val, y_val = prepare_features(val, feature_names, target)
    X_test, y_test = prepare_features(test, feature_names, target)

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    results = {}
    models = {}

    # Train XGBoost
    logger.info("Training XGBoost...")
    try:
        xgb_model = train_xgboost(X_train, y_train, X_val, y_val, is_classifier)
        xgb_metrics = evaluate_model(xgb_model, X_test, y_test, is_classifier)
        results['xgboost'] = xgb_metrics
        models['xgboost'] = xgb_model
        logger.info(f"XGBoost: {xgb_metrics}")
    except Exception as e:
        logger.error(f"XGBoost failed: {e}")

    # Train LightGBM
    logger.info("Training LightGBM...")
    try:
        lgb_model = train_lightgbm(X_train, y_train, X_val, y_val, is_classifier)
        lgb_metrics = evaluate_model(lgb_model, X_test, y_test, is_classifier)
        results['lightgbm'] = lgb_metrics
        models['lightgbm'] = lgb_model
        logger.info(f"LightGBM: {lgb_metrics}")
    except Exception as e:
        logger.error(f"LightGBM failed: {e}")

    # Train CatBoost
    logger.info("Training CatBoost...")
    try:
        cat_model = train_catboost(X_train, y_train, X_val, y_val, is_classifier)
        cat_metrics = evaluate_model(cat_model, X_test, y_test, is_classifier)
        results['catboost'] = cat_metrics
        models['catboost'] = cat_model
        logger.info(f"CatBoost: {cat_metrics}")
    except Exception as e:
        logger.error(f"CatBoost failed: {e}")

    # Save models
    model_path = output_dir / f"{args.symbol}_{target}_models.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'models': models,
            'feature_names': feature_names,
            'target': target,
            'is_classifier': is_classifier,
            'results': results,
            'trained_at': datetime.now().isoformat()
        }, f)
    logger.info(f"Saved models to {model_path}")

    # Save results
    results_path = output_dir / f"{args.symbol}_{target}_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'symbol': args.symbol,
            'target': target,
            'results': results,
            'best_model': max(results, key=lambda k: results[k].get('auc', results[k].get('r2', 0))),
            'trained_at': datetime.now().isoformat()
        }, f, indent=2)
    logger.info(f"Saved results to {results_path}")

    # Print summary
    print("\\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    for model_name, metrics in results.items():
        print(f"\\n{model_name.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    best = max(results, key=lambda k: results[k].get('auc', results[k].get('r2', 0)))
    print(f"\\nBest model: {best}")


if __name__ == '__main__':
    main()
'''
        script_path = self.output_dir / 'train_hft_models.py'
        with open(script_path, 'w') as f:
            f.write(script_content)

        logger.info(f"Created training script: {script_path}")
        return script_path

    def create_requirements(self) -> Path:
        """Create requirements.txt for Vast.ai."""
        requirements = """# HFT Training Requirements
numpy>=1.24.0
pandas>=2.0.0
pyarrow>=12.0.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0
optuna>=3.3.0
"""
        req_path = self.output_dir / 'requirements.txt'
        with open(req_path, 'w') as f:
            f.write(requirements)

        logger.info(f"Created requirements: {req_path}")
        return req_path

    def create_vastai_script(self) -> Path:
        """Create Vast.ai setup and run script."""
        script = """#!/bin/bash
# Vast.ai H100 Training Script
# ============================
# 1. Rent H100 on Vast.ai (~$2.50/hr)
# 2. Upload training_package folder
# 3. Run this script

set -e

echo "=== HFT Training on H100 ==="
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Training models..."
SYMBOLS="EURUSD GBPUSD USDJPY USDCHF AUDUSD"
TARGETS="target_direction_10 target_direction_20 target_return_10"

for symbol in $SYMBOLS; do
    for target in $TARGETS; do
        echo "Training $symbol - $target"
        python train_hft_models.py --symbol $symbol --target $target --output-dir models
    done
done

echo "=== Training Complete ==="
echo "Models saved to ./models/"
ls -la models/

# Compress models for download
tar -czvf hft_models.tar.gz models/
echo "Download: hft_models.tar.gz"
"""
        script_path = self.output_dir / 'run_training.sh'
        with open(script_path, 'w') as f:
            f.write(script)

        logger.info(f"Created Vast.ai script: {script_path}")
        return script_path

    def create_upload_instructions(self) -> Path:
        """Create upload instructions."""
        instructions = """# Vast.ai Training Instructions

## 1. Rent H100 Instance

1. Go to https://vast.ai/
2. Search for H100 GPU (~$2.50/hr)
3. Select "PyTorch" template
4. Launch instance

## 2. Upload Training Package

```bash
# Get instance IP from Vast.ai dashboard
VAST_IP="your_instance_ip"

# Upload package
scp -r training_package/* root@$VAST_IP:/workspace/
```

## 3. Run Training

```bash
# SSH to instance
ssh root@$VAST_IP

# Run training
cd /workspace
chmod +x run_training.sh
./run_training.sh
```

## 4. Download Models

```bash
# Download trained models
scp root@$VAST_IP:/workspace/hft_models.tar.gz .
tar -xzvf hft_models.tar.gz

# Move to forex project
mv models/* C:/Users/kevin/forex/models/hft_ensemble/
```

## 5. Destroy Instance

Don't forget to destroy the instance to stop billing!

## Expected Training Time

- EURUSD + 3 targets: ~5-10 min per target
- 5 symbols x 3 targets = 15 training runs
- Total: ~1-2 hours
- Cost: ~$3-5

## Expected Results

- Accuracy: >55% (random = 50%)
- AUC: >0.55
- Sharpe: >1.5 (when combined with execution)
"""
        path = self.output_dir / 'TRAINING_INSTRUCTIONS.md'
        with open(path, 'w') as f:
            f.write(instructions)

        logger.info(f"Created instructions: {path}")
        return path


def main():
    parser = argparse.ArgumentParser(description='Prepare HFT training data for Vast.ai')
    parser.add_argument('--symbols', type=str, default='EURUSD,GBPUSD,USDJPY',
                        help='Comma-separated list of symbols')
    parser.add_argument('--days', type=int, default=30,
                        help='Days of historical data')
    parser.add_argument('--source', type=str, default='truefx',
                        choices=['truefx', 'oracle', 'local'],
                        help='Data source')
    parser.add_argument('--output', type=str, default='training_package',
                        help='Output directory')
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(',')]

    logger.info(f"Preparing training data for: {symbols}")
    logger.info(f"Days: {args.days}, Source: {args.source}")

    packager = HFTTrainingPackager(Path(args.output))

    # Package all symbols
    paths = packager.package_all(symbols, args.days, args.source)

    # Create training script
    packager.create_training_script()
    packager.create_requirements()
    packager.create_vastai_script()
    packager.create_upload_instructions()

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING PACKAGE READY")
    print("=" * 60)
    print(f"Output: {packager.output_dir}")
    print(f"Symbols: {symbols}")

    for symbol, symbol_paths in paths.items():
        train_size = pd.read_parquet(symbol_paths['train']).shape[0]
        print(f"  {symbol}: {train_size:,} training samples")

    print("\nNext steps:")
    print("1. Rent H100 on Vast.ai (~$2.50/hr)")
    print("2. scp -r training_package/* root@<VAST_IP>:/workspace/")
    print("3. ssh root@<VAST_IP>")
    print("4. ./run_training.sh")
    print("5. Download hft_models.tar.gz")


if __name__ == '__main__':
    main()
