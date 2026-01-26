#!/usr/bin/env python3
"""
HFT Model Training - MAXIMUM GPU Optimized for RTX 5080 16GB
=============================================================
System: AMD Ryzen 9 9900X (12c/24t) + RTX 5080 16GB + 32GB DDR5-4800

Usage:
    # Train with base features (~315 features)
    python train_models.py

    # Train with ALL 806+ features (mega arsenal)
    python train_models.py --features all

    # MAXIMUM TRAINING (1350+ features, max GPU settings)
    python train_models.py --features ultra --max-gpu

    # Train specific symbols with max settings
    python train_models.py --symbols EURUSD,GBPUSD,USDJPY --features ultra --max-gpu

    # Parallel training (4 pairs at once)
    python train_models.py --features ultra --max-gpu --parallel 4 --workers 12

    # Dry run (test feature generation only)
    python train_models.py --features ultra --dry-run

Target: 89% accuracy with 1350+ features + max GPU settings + LLM validation
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle
from datetime import datetime
import sys
import time
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
import queue

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import GPU config FIRST (sets environment variables)
try:
    from core.ml.gpu_config import (
        get_xgb_gpu_params,
        get_lgb_gpu_params,
        get_catboost_gpu_params,
        optimize_for_training,
        print_system_info
    )
    HAS_GPU_CONFIG = True
except ImportError as e:
    HAS_GPU_CONFIG = False
    print(f"Warning: gpu_config not found ({e}), using default params")

# Import MAX GPU config for --max-gpu mode
try:
    from core.ml.max_gpu_config import (
        get_max_xgb_params,
        get_max_lgb_params,
        get_max_catboost_params,
        maximize_system,
        configure_max_gpu,
        print_gpu_status
    )
    HAS_MAX_GPU_CONFIG = True
except ImportError as e:
    HAS_MAX_GPU_CONFIG = False
    print(f"Warning: max_gpu_config not found ({e})")

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


def regenerate_features_mega(df: pd.DataFrame, verbose: bool = False) -> tuple:
    """
    Regenerate features using MegaFeatureGenerator (806+ features).

    Args:
        df: DataFrame with OHLCV data
        verbose: Print progress

    Returns:
        Tuple of (feature_df, feature_names)
    """
    from core.features.mega_generator import MegaFeatureGenerator

    generator = MegaFeatureGenerator(
        enable_alpha101=True,
        enable_alpha158=True,
        enable_alpha360=True,
        enable_barra=True,
        enable_us_academic=True,
        enable_mlfinlab=True,
        enable_timeseries=True,
        enable_renaissance=True,
        alpha360_lookback=20,  # Compact version
        verbose=verbose
    )

    features = generator.generate_all(df)

    # Get expected counts for logging
    counts = generator.get_feature_counts()
    if verbose:
        print(f"Feature counts: {counts}")

    return features, list(features.columns)


def regenerate_features_ultra(df: pd.DataFrame, verbose: bool = False) -> tuple:
    """
    ULTRA feature generation - ALL 1350+ features for MAXIMUM accuracy.

    Includes:
    - Alpha101 (62)
    - Alpha158 (179)
    - Alpha191 Guotai Junan (191)
    - Alpha360 (120 with lookback=20)
    - Barra CNE6 (46)
    - Chinese HFT (50+)
    - Chinese HFT Additions (25+)
    - Chinese Gold Standard (35+)
    - US Academic (50)
    - MLFinLab (17)
    - Time-Series-Library (41)
    - Renaissance (51)
    - India Quant (25)
    - Japan Quant (30)
    - Europe Quant (15)
    - Emerging Markets (20)
    - Universal Math (30)
    - RL Algorithms (35)
    - MOE Trading (20)
    - GNN Temporal (15)
    - Korea Quant (20)
    - Asia FX Spread (15)
    - MARL Trading (15)

    Total: ~1350+ features

    Args:
        df: DataFrame with OHLCV data
        verbose: Print progress

    Returns:
        Tuple of (feature_df, feature_names)
    """
    from core.features.mega_generator import MegaFeatureGenerator

    # Enable ALL feature generators for maximum accuracy
    generator = MegaFeatureGenerator(
        # Chinese Quant Core
        enable_alpha101=True,
        enable_alpha158=True,
        enable_alpha360=True,
        enable_barra=True,
        enable_alpha191=True,
        enable_chinese_hft=True,
        enable_chinese_additions=True,
        enable_chinese_gold=True,
        # US Academic
        enable_us_academic=True,
        enable_mlfinlab=True,
        enable_timeseries=True,
        enable_renaissance=True,
        # Global Expansion
        enable_india=True,
        enable_japan=True,
        enable_europe=True,
        enable_emerging=True,
        enable_universal_math=True,
        # Reinforcement Learning
        enable_rl=True,
        # Eastern Asia Gold Standard
        enable_moe=True,
        enable_gnn=True,
        enable_korea=True,
        enable_asia_fx=True,
        enable_marl=True,
        # Settings
        alpha360_lookback=20,  # 20 for 120 features (compact but effective)
        verbose=verbose
    )

    features = generator.generate_all(df)

    # Get expected counts for logging
    counts = generator.get_feature_counts()
    if verbose:
        print(f"\n{'='*60}")
        print("ULTRA FEATURE GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total features generated: {len(features.columns)}")
        print("\nFeature breakdown:")
        for name, count in counts.items():
            if name != 'total':
                print(f"  - {name}: {count}")
        print(f"{'='*60}\n")

    return features, list(features.columns)


def prepare_data_with_mega_features(symbol_dir, verbose=False):
    """
    Prepare data with 806+ mega features.

    Returns train/val/test splits with regenerated features.
    """
    # Load original data
    train, val, test, metadata = load_data(symbol_dir)

    # Keep original targets
    original_targets = metadata['targets']
    target_cols = [c for c in train.columns if c in original_targets]

    print(f"  Regenerating features with MegaFeatureGenerator...")

    # Regenerate features for each split
    # First, identify OHLCV columns needed
    ohlcv_cols = []
    for col in ['open', 'high', 'low', 'close', 'volume', 'vwap', 'price', 'bid', 'ask', 'returns']:
        if col in train.columns:
            ohlcv_cols.append(col)

    # If we don't have OHLCV, map from tick data columns
    if 'close' not in train.columns:
        # For tick data, use mid price as close
        if 'mid' in train.columns:
            train['close'] = train['mid']
            val['close'] = val['mid']
            test['close'] = test['mid']
            print(f"  Using 'mid' as close price")
        elif 'bid' in train.columns and 'ask' in train.columns:
            train['close'] = (train['bid'] + train['ask']) / 2
            val['close'] = (val['bid'] + val['ask']) / 2
            test['close'] = (test['bid'] + test['ask']) / 2
            print(f"  Using (bid+ask)/2 as close price")
        else:
            # Fallback: try to find price-like column
            close_candidates = [c for c in train.columns if 'close' in c.lower() or 'price' in c.lower()]
            if close_candidates:
                train['close'] = train[close_candidates[0]]
                val['close'] = val[close_candidates[0]]
                test['close'] = test[close_candidates[0]]
                print(f"  Using '{close_candidates[0]}' as close price")
            else:
                raise ValueError("No price column found - cannot generate mega features")

    # Generate mega features
    train_features, feature_names = regenerate_features_mega(train, verbose=verbose)
    val_features, _ = regenerate_features_mega(val, verbose=False)
    test_features, _ = regenerate_features_mega(test, verbose=False)

    # Filter to only numeric columns (exclude timestamps, strings, etc.)
    numeric_cols = train_features.select_dtypes(include=[np.number]).columns.tolist()
    train_features = train_features[numeric_cols]
    val_features = val_features[numeric_cols]
    test_features = test_features[numeric_cols]
    feature_names = numeric_cols

    print(f"  Filtered to {len(feature_names)} numeric features")

    # Combine features with targets
    train_final = pd.concat([train_features, train[target_cols]], axis=1)
    val_final = pd.concat([val_features, val[target_cols]], axis=1)
    test_final = pd.concat([test_features, test[target_cols]], axis=1)

    # Update metadata
    new_metadata = metadata.copy()
    new_metadata['features'] = feature_names
    new_metadata['feature_count'] = len(feature_names)
    new_metadata['feature_generator'] = 'MegaFeatureGenerator'

    return train_final, val_final, test_final, new_metadata


def prepare_data_with_ultra_features(symbol_dir, verbose=False):
    """
    Prepare data with 1350+ ULTRA features for MAXIMUM accuracy.

    Returns train/val/test splits with regenerated features.
    """
    # Load original data
    train, val, test, metadata = load_data(symbol_dir)

    # Keep original targets
    original_targets = metadata['targets']
    target_cols = [c for c in train.columns if c in original_targets]

    print(f"  Regenerating features with ULTRA MegaFeatureGenerator (1350+ features)...")

    # Regenerate features for each split
    # If we don't have OHLCV, map from tick data columns
    if 'close' not in train.columns:
        # For tick data, use mid price as close
        if 'mid' in train.columns:
            train['close'] = train['mid']
            val['close'] = val['mid']
            test['close'] = test['mid']
            print(f"  Using 'mid' as close price")
        elif 'bid' in train.columns and 'ask' in train.columns:
            train['close'] = (train['bid'] + train['ask']) / 2
            val['close'] = (val['bid'] + val['ask']) / 2
            test['close'] = (test['bid'] + test['ask']) / 2
            print(f"  Using (bid+ask)/2 as close price")
        else:
            # Fallback: try to find price-like column
            close_candidates = [c for c in train.columns if 'close' in c.lower() or 'price' in c.lower()]
            if close_candidates:
                train['close'] = train[close_candidates[0]]
                val['close'] = val[close_candidates[0]]
                test['close'] = test[close_candidates[0]]
                print(f"  Using '{close_candidates[0]}' as close price")
            else:
                raise ValueError("No price column found - cannot generate ultra features")

    # Generate ULTRA features (1350+)
    print(f"  Generating ULTRA features for train set...")
    train_features, feature_names = regenerate_features_ultra(train, verbose=verbose)
    print(f"  Generating ULTRA features for validation set...")
    val_features, _ = regenerate_features_ultra(val, verbose=False)
    print(f"  Generating ULTRA features for test set...")
    test_features, _ = regenerate_features_ultra(test, verbose=False)

    # Filter to only numeric columns (exclude timestamps, strings, etc.)
    numeric_cols = train_features.select_dtypes(include=[np.number]).columns.tolist()
    train_features = train_features[numeric_cols]
    val_features = val_features[numeric_cols]
    test_features = test_features[numeric_cols]
    feature_names = numeric_cols

    print(f"  Generated {len(feature_names)} ULTRA features!")

    # Combine features with targets
    train_final = pd.concat([train_features, train[target_cols]], axis=1)
    val_final = pd.concat([val_features, val[target_cols]], axis=1)
    test_final = pd.concat([test_features, test[target_cols]], axis=1)

    # Update metadata
    new_metadata = metadata.copy()
    new_metadata['features'] = feature_names
    new_metadata['feature_count'] = len(feature_names)
    new_metadata['feature_generator'] = 'MegaFeatureGenerator_ULTRA'

    return train_final, val_final, test_final, new_metadata


def train_xgboost(X_train, y_train, X_val, y_val, target_name, max_samples=100000, use_max_gpu=False):
    """Train XGBoost with GPU acceleration - MAX PERFORMANCE for RTX 5080 16GB."""
    # Use more samples for max GPU mode
    if use_max_gpu:
        max_samples = 200000  # Can handle more with max settings

    # Subsample if too large for GPU memory
    if len(X_train) > max_samples:
        print(f"  Subsampling from {len(X_train)} to {max_samples} for GPU memory")
        idx = np.random.choice(len(X_train), max_samples, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

    # Use MAX GPU params if requested
    if use_max_gpu and HAS_MAX_GPU_CONFIG:
        print(f"  [MAX GPU] Using maximum XGBoost settings (depth=16, bins=1024, estimators=3000)")
        params = get_max_xgb_params()
        num_boost_round = params.pop('n_estimators', 3000)
        early_stopping = params.pop('early_stopping_rounds', 100)
    else:
        # Standard GPU params
        params = {
            'tree_method': 'hist',  # XGBoost 2.0+
            'device': 'cuda',
            'max_depth': 12,  # Deep trees for 16GB
            'max_bin': 256,  # More bins
            'learning_rate': 0.03,
            'subsample': 0.85,
            'colsample_bytree': 0.7,
            'min_child_weight': 10,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
        }
        num_boost_round = 1500
        early_stopping = 50

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dval, 'val')],
        early_stopping_rounds=early_stopping,
        verbose_eval=100
    )

    return model


def train_lightgbm(X_train, y_train, X_val, y_val, target_name, max_samples=100000, use_max_gpu=False):
    """Train LightGBM with GPU acceleration - MAX PERFORMANCE for RTX 5080 16GB."""
    # Use more samples for max GPU mode
    if use_max_gpu:
        max_samples = 200000

    # Subsample if too large for GPU memory
    if len(X_train) > max_samples:
        print(f"  Subsampling from {len(X_train)} to {max_samples} for GPU memory")
        idx = np.random.choice(len(X_train), max_samples, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

    # Use MAX GPU params if requested
    if use_max_gpu and HAS_MAX_GPU_CONFIG:
        print(f"  [MAX GPU] Using maximum LightGBM settings (depth=16, leaves=4095, estimators=3000)")
        params = get_max_lgb_params()
        num_boost_round = params.pop('n_estimators', 3000)
        early_stopping = params.pop('early_stopping_rounds', 100)
    else:
        # Standard GPU params
        params = {
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'max_depth': 10,  # Stable depth
            'num_leaves': 255,  # Stable leaves
            'learning_rate': 0.03,
            'feature_fraction': 0.75,
            'bagging_fraction': 0.85,
            'bagging_freq': 3,
            'min_data_in_leaf': 50,  # More data per leaf
            'lambda_l1': 0.1,
            'lambda_l2': 1.0,
            'verbose': -1,
            'objective': 'binary',
            'metric': 'auc',
            'force_col_wise': True,  # Better GPU performance
            'gpu_use_dp': False,  # Single precision for speed
        }
        num_boost_round = 1500
        early_stopping = 50

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(early_stopping), lgb.log_evaluation(100)]
    )

    return model


def train_catboost(X_train, y_train, X_val, y_val, target_name, max_samples=100000, use_max_gpu=False):
    """Train CatBoost with GPU acceleration - MAX PERFORMANCE for RTX 5080 16GB."""
    # Use more samples for max GPU mode
    if use_max_gpu:
        max_samples = 200000

    # Subsample if too large for GPU memory
    if len(X_train) > max_samples:
        print(f"  Subsampling from {len(X_train)} to {max_samples} for GPU memory")
        idx = np.random.choice(len(X_train), max_samples, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

    # Use MAX GPU params if requested
    if use_max_gpu and HAS_MAX_GPU_CONFIG:
        print(f"  [MAX GPU] Using maximum CatBoost settings (depth=12, border=254, iterations=3000)")
        cb_params = get_max_catboost_params()
        model = cb.CatBoostClassifier(**cb_params)
    else:
        # Standard GPU params
        model = cb.CatBoostClassifier(
            task_type='GPU',
            devices='0',
            depth=10,  # Max GPU depth for CatBoost
            border_count=128,  # More splits
            iterations=1500,  # More iterations
            learning_rate=0.03,
            l2_leaf_reg=3.0,
            early_stopping_rounds=50,
            verbose=100,
            boosting_type='Plain',  # Faster on GPU
            loss_function='Logloss',
            eval_metric='AUC',
            thread_count=12,  # Use all CPU cores for data prep
        )

    model.fit(X_train, y_train, eval_set=(X_val, y_val))

    return model


def train_symbol(symbol_dir, output_dir, feature_mode='base', verbose=False, use_max_gpu=False):
    """Train all models for a symbol."""
    symbol = symbol_dir.name
    mode_desc = f"features: {feature_mode}" + (" + MAX GPU" if use_max_gpu else "")
    print(f"\n{'='*60}")
    print(f"Training {symbol} ({mode_desc})")
    print(f"{'='*60}")

    # Load data based on feature mode
    if feature_mode == 'ultra':
        train, val, test, metadata = prepare_data_with_ultra_features(symbol_dir, verbose)
        print(f"  Using {len(metadata['features'])} ULTRA features (MAXIMUM)")
    elif feature_mode == 'all':
        train, val, test, metadata = prepare_data_with_mega_features(symbol_dir, verbose)
        print(f"  Using {len(metadata['features'])} mega features")
    else:
        train, val, test, metadata = load_data(symbol_dir)
        print(f"  Using {len(metadata['features'])} base features")

    features = metadata['features']
    targets = [t for t in metadata['targets'] if 'direction' in t]

    X_train = train[features].values
    X_val = val[features].values
    X_test = test[features].values

    results = {}
    models = {}

    for target in targets[:3]:  # Train on top 3 horizons
        print(f"\nTarget: {target}")

        y_train = train[target].values
        y_val = val[target].values
        y_test = test[target].values

        # Train ensemble (pass use_max_gpu to all models)
        xgb_model = train_xgboost(X_train, y_train, X_val, y_val, target, use_max_gpu=use_max_gpu)
        lgb_model = train_lightgbm(X_train, y_train, X_val, y_val, target, use_max_gpu=use_max_gpu)
        cb_model = train_catboost(X_train, y_train, X_val, y_val, target, use_max_gpu=use_max_gpu)

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
            'features': features,
            'feature_mode': feature_mode,
            'feature_count': len(features)
        }

    # Save models
    model_file = output_dir / f"{symbol}_models.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(models, f)

    # Save results
    results_file = output_dir / f"{symbol}_results.json"
    results_to_save = results.copy()
    results_to_save['_meta'] = {
        'feature_mode': feature_mode,
        'feature_count': len(features),
        'timestamp': datetime.now().isoformat()
    }
    with open(results_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)

    print(f"\nSaved models to {model_file}")
    return results


def dry_run(symbol_dir, feature_mode='all', verbose=True):
    """Test feature generation without training."""
    symbol = symbol_dir.name
    print(f"\n{'='*60}")
    print(f"DRY RUN: Testing feature generation for {symbol}")
    print(f"Feature mode: {feature_mode}")
    print(f"{'='*60}")

    train, val, test, metadata = load_data(symbol_dir)
    print(f"  Original features: {len(metadata['features'])}")
    print(f"  Train samples: {len(train)}")

    if feature_mode in ['all', 'ultra']:
        from core.features.mega_generator import MegaFeatureGenerator

        if feature_mode == 'ultra':
            print(f"\n  Generating ULTRA features (1350+)...")
            # Enable ALL generators for ultra mode
            generator = MegaFeatureGenerator(
                enable_alpha101=True,
                enable_alpha158=True,
                enable_alpha360=True,
                enable_barra=True,
                enable_alpha191=True,
                enable_chinese_hft=True,
                enable_chinese_additions=True,
                enable_chinese_gold=True,
                enable_us_academic=True,
                enable_mlfinlab=True,
                enable_timeseries=True,
                enable_renaissance=True,
                enable_india=True,
                enable_japan=True,
                enable_europe=True,
                enable_emerging=True,
                enable_universal_math=True,
                enable_rl=True,
                enable_moe=True,
                enable_gnn=True,
                enable_korea=True,
                enable_asia_fx=True,
                enable_marl=True,
                alpha360_lookback=20,
                verbose=True
            )
        else:
            print(f"\n  Generating mega features (806+)...")
            generator = MegaFeatureGenerator(verbose=True)

        # Test on a small sample first
        sample_size = min(1000, len(train))
        sample = train.head(sample_size).copy()

        # Ensure we have close column
        if 'close' not in sample.columns:
            close_candidates = [c for c in sample.columns if 'close' in c.lower() or 'price' in c.lower()]
            if close_candidates:
                sample['close'] = sample[close_candidates[0]]
            else:
                first_feature = metadata['features'][0]
                sample['close'] = sample[first_feature]

        start_time = time.time()
        features = generator.generate_all(sample)
        elapsed = time.time() - start_time

        print(f"\n  Generated {len(features.columns)} features in {elapsed:.2f}s")
        print(f"  Sample shape: {features.shape}")
        print(f"  Memory usage: {features.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        # Show feature breakdown
        counts = generator.get_feature_counts()
        print(f"\n  Feature breakdown:")
        for name, count in counts.items():
            print(f"    - {name}: {count}")

        # Check for NaN/inf
        nan_count = features.isna().sum().sum()
        # Convert to float to check for inf (handles object dtypes)
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        inf_count = np.isinf(features[numeric_cols].values).sum() if len(numeric_cols) > 0 else 0
        print(f"\n  NaN values: {nan_count}")
        print(f"  Inf values: {inf_count}")

        # Estimate full dataset memory
        full_features_mb = (len(features.columns) * len(train) * 8) / 1024**2
        print(f"\n  Estimated full dataset memory: {full_features_mb:.1f} MB")
    else:
        print(f"\n  Using base features ({len(metadata['features'])} features)")
        print(f"  To test feature generation, use --features all or --features ultra")

    print(f"\n  DRY RUN COMPLETE")
    return True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='HFT Model Training - MAXIMUM GPU Optimized for RTX 5080 16GB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard training with base features
  python train_models.py

  # MAXIMUM training (1350+ features + max GPU settings) - RECOMMENDED
  python train_models.py --features ultra --max-gpu

  # Parallel training (4 pairs at once)
  python train_models.py --features ultra --max-gpu --parallel 4 --workers 12

  # Dry run to test feature generation
  python train_models.py --features ultra --dry-run

Target: 89% accuracy with 1350+ features + max GPU + LLM validation
        """
    )
    parser.add_argument(
        '--features',
        choices=['base', 'all', 'ultra'],
        default='base',
        help='Feature set: base (~315), all (806+ mega), ultra (1350+ MAXIMUM)'
    )
    parser.add_argument(
        '--max-gpu',
        action='store_true',
        help='Use MAXIMUM GPU settings (depth=16, bins=1024, estimators=3000)'
    )
    parser.add_argument(
        '--symbols',
        type=str,
        default=None,
        help='Comma-separated symbols to train (default: all found)'
    )
    parser.add_argument(
        '--parallel',
        type=int,
        default=1,
        help='Number of pairs to train in parallel (default: 1, max: 4)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=12,
        help='Number of CPU workers for feature generation (default: 12)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Test feature generation without training'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Output directory for models (default: models)'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show training status and exit'
    )
    return parser.parse_args()


def show_training_status(data_dir: Path, output_dir: Path):
    """Show current training status."""
    print("\n" + "="*70)
    print("TRAINING STATUS")
    print("="*70)

    # Find all symbols
    all_symbols = [d.name for d in data_dir.iterdir()
                   if d.is_dir() and (d / "train.parquet").exists()]

    # Check which are trained
    trained = []
    untrained = []
    results_summary = []

    for symbol in sorted(all_symbols):
        results_file = output_dir / f"{symbol}_results.json"
        if results_file.exists():
            trained.append(symbol)
            with open(results_file) as f:
                data = json.load(f)
            # Get accuracy for target_direction_10
            acc = data.get('target_direction_10', {}).get('accuracy', 0)
            fc = data.get('_meta', {}).get('feature_count', 0)
            results_summary.append((symbol, acc, fc))
        else:
            untrained.append(symbol)

    print(f"\nTotal pairs: {len(all_symbols)}")
    print(f"Trained: {len(trained)}")
    print(f"Untrained: {len(untrained)}")

    if results_summary:
        print(f"\nAccuracy Summary (target_direction_10):")
        avg_acc = sum(r[1] for r in results_summary) / len(results_summary)
        print(f"  Average: {avg_acc*100:.2f}%")
        print(f"\nTop 10 performers:")
        for sym, acc, fc in sorted(results_summary, key=lambda x: -x[1])[:10]:
            print(f"  {sym}: {acc*100:.2f}% ({fc} features)")

    if untrained:
        print(f"\nUntrained pairs: {', '.join(untrained[:10])}")
        if len(untrained) > 10:
            print(f"  ... and {len(untrained) - 10} more")

    print("="*70)


def main():
    """Train all symbols with MAXIMUM performance."""
    args = parse_args()

    # Determine data and output directories
    data_dir = Path(".")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Status mode
    if args.status:
        show_training_status(data_dir, output_dir)
        return

    # Print header
    print("\n" + "="*70)
    print("HFT MODEL TRAINING - RTX 5080 16GB + RYZEN 9 9900X")
    print("="*70)

    # Maximize system if --max-gpu
    if args.max_gpu:
        if HAS_MAX_GPU_CONFIG:
            print("\n[MAX GPU MODE ENABLED]")
            maximize_system()
        else:
            print("\nWarning: --max-gpu specified but max_gpu_config not found")
            print("Using standard GPU settings instead")

    # Print system info
    if HAS_GPU_CONFIG and not args.dry_run:
        print_system_info()
        if not args.max_gpu:
            optimize_for_training()

    total_start = time.time()

    # Find symbols
    all_symbols = [d.name for d in data_dir.iterdir()
                   if d.is_dir() and (d / "train.parquet").exists()]

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
        # Validate
        for s in symbols:
            if s not in all_symbols:
                print(f"Warning: Symbol {s} not found in data directory")
        symbols = [s for s in symbols if s in all_symbols]
    else:
        symbols = all_symbols

    print(f"\nTraining symbols: {len(symbols)} pairs")
    print(f"Feature mode: {args.features}")
    print(f"Max GPU: {args.max_gpu}")
    print(f"Parallel: {args.parallel}")

    # Dry run mode
    if args.dry_run:
        for symbol in symbols:
            symbol_dir = data_dir / symbol
            dry_run(symbol_dir, args.features, args.verbose)
        return

    # Training mode
    all_results = {}

    if args.parallel > 1:
        # Parallel training mode
        print(f"\n[PARALLEL MODE] Training {args.parallel} pairs concurrently")
        print(f"[PARALLEL MODE] Feature workers: {args.workers}")

        # Use ThreadPoolExecutor for parallel symbol training
        # (GPU operations are serialized anyway by CUDA)
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {}
            for symbol in symbols:
                symbol_dir = data_dir / symbol
                future = executor.submit(
                    train_symbol,
                    symbol_dir,
                    output_dir,
                    feature_mode=args.features,
                    verbose=args.verbose,
                    use_max_gpu=args.max_gpu
                )
                futures[future] = symbol

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    results = future.result()
                    all_results[symbol] = results
                    print(f"\n[COMPLETE] {symbol}")
                except Exception as e:
                    print(f"\n[ERROR] {symbol}: {e}")
    else:
        # Sequential training
        for symbol in symbols:
            symbol_dir = data_dir / symbol
            symbol_start = time.time()
            results = train_symbol(
                symbol_dir,
                output_dir,
                feature_mode=args.features,
                verbose=args.verbose,
                use_max_gpu=args.max_gpu
            )
            all_results[symbol] = results
            symbol_time = time.time() - symbol_start
            print(f"\n{symbol} training time: {symbol_time/60:.1f} minutes")

    total_time = time.time() - total_start

    # Summary
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Feature mode: {args.features}")
    print(f"Max GPU: {args.max_gpu}")
    print(f"Total training time: {total_time/60:.1f} minutes")
    print(f"Pairs trained: {len(all_results)}")

    # Calculate average accuracy
    accuracies = []
    for symbol, results in all_results.items():
        print(f"\n{symbol}:")
        for target, metrics in results.items():
            if not target.startswith('_'):  # Skip metadata
                print(f"  {target}: ACC={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}")
                if target == 'target_direction_10':
                    accuracies.append(metrics['accuracy'])

    if accuracies:
        avg_acc = sum(accuracies) / len(accuracies)
        print(f"\n{'='*70}")
        print(f"AVERAGE ACCURACY (target_direction_10): {avg_acc*100:.2f}%")
        print(f"Target: 80%+ (with LLM validation: 89%+)")
        print(f"{'='*70}")

    # Package models
    import shutil
    archive_name = f'hft_models_{args.features}'
    if args.max_gpu:
        archive_name += '_max'
    shutil.make_archive(archive_name, 'gztar', output_dir)
    print(f"\nModels packaged to {archive_name}.tar.gz")

    # Print GPU status if available
    if HAS_MAX_GPU_CONFIG and args.max_gpu:
        print_gpu_status()


if __name__ == "__main__":
    main()
