#!/usr/bin/env python3
"""
MAX PERFORMANCE Training Script
===============================
Fully utilizes RTX 5080 (16GB) + Ryzen 9 9900X (24 threads)

Target: 1083 features, all data, maximum GPU throughput
"""

import os
import sys
import time
import pickle
import warnings
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Tuple, Any

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set thread counts BEFORE imports
N_CORES = 12  # Physical cores
N_THREADS = 24  # Logical threads
os.environ['OMP_NUM_THREADS'] = str(N_CORES)
os.environ['MKL_NUM_THREADS'] = str(N_CORES)
os.environ['OPENBLAS_NUM_THREADS'] = str(N_CORES)
os.environ['NUMEXPR_NUM_THREADS'] = str(N_THREADS)
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async GPU

import numpy as np
import pandas as pd
import psutil
import torch

# Verify GPU
if not torch.cuda.is_available():
    print("ERROR: CUDA not available!")
    sys.exit(1)

import functools
print = functools.partial(print, flush=True)

print("=" * 70)
print("MAX PERFORMANCE TRAINING - RTX 5080 + RYZEN 9 9900X")
print("=" * 70)
print(f"\nGPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"CPU: {N_CORES} cores / {N_THREADS} threads @ {psutil.cpu_freq().current:.0f} MHz")
print(f"RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
print("=" * 70)

# Configure PyTorch for max performance
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

# Set process priority to HIGH
try:
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00008000)
    print("Process priority: HIGH")
except:
    pass

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import ML libraries
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import train_test_split

# Import feature generator
from core.features.mega_generator import MegaFeatureGenerator

# =============================================================================
# MAX GPU CONFIG - Tuned for RTX 5080 16GB VRAM
# =============================================================================

XGB_GPU_PARAMS = {
    'tree_method': 'hist',  # XGBoost 2.0+
    'device': 'cuda',
    'max_depth': 14,  # Deep trees for 16GB
    'max_bin': 512,  # More bins for GPU
    'n_estimators': 1500,  # More trees
    'learning_rate': 0.05,
    'subsample': 0.9,
    'colsample_bytree': 0.8,
    'min_child_weight': 10,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'verbosity': 1,
    'n_jobs': -1,
    'random_state': 42,
}

LGB_GPU_PARAMS = {
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'max_depth': 14,
    'num_leaves': 1024,  # 2^14 leaves
    'n_estimators': 1500,
    'learning_rate': 0.05,
    'subsample': 0.9,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': 1,
    'n_jobs': -1,
    'random_state': 42,
    'force_col_wise': True,  # Better GPU performance
}

CB_GPU_PARAMS = {
    'task_type': 'GPU',
    'devices': '0',
    'depth': 12,  # CatBoost max GPU depth
    'iterations': 1500,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3.0,
    'border_count': 128,  # More splits for GPU
    'boosting_type': 'Plain',  # Faster on GPU
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'verbose': 100,
    'random_seed': 42,
    'thread_count': N_CORES,
}


def load_data(symbol: str) -> pd.DataFrame:
    """Load training data for symbol."""
    train_path = Path(f"training_package/{symbol}/train.parquet")
    if not train_path.exists():
        raise FileNotFoundError(f"No training data for {symbol}")

    df = pd.read_parquet(train_path)
    print(f"  Loaded {len(df):,} rows from {train_path}")
    return df


def generate_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Generate mega features using all CPU cores."""
    print("  Generating 1083 mega features...")

    gen = MegaFeatureGenerator()

    # Generate features
    features_df = gen.generate_all(df)

    # Get numeric columns only
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove target columns
    feature_cols = [c for c in numeric_cols if not c.startswith('target_')]

    print(f"  Generated {len(feature_cols)} features")
    return features_df, feature_cols


def prepare_data(df: pd.DataFrame, feature_cols: List[str], target: str) -> Tuple:
    """Prepare train/val/test splits."""
    # Drop rows with NaN in target
    df = df.dropna(subset=[target])

    # Create feature matrix
    X = df[feature_cols].fillna(0).values.astype(np.float32)
    y = df[target].values.astype(np.float32)

    # Convert direction to binary (0/1)
    if 'direction' in target:
        y = (y > 0).astype(np.float32)

    # Split: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, shuffle=False  # Time-series: no shuffle
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, shuffle=False
    )

    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_xgboost(X_train, X_val, y_train, y_val) -> Tuple[Any, float]:
    """Train XGBoost with GPU acceleration."""
    print("\n  [XGBoost] Training with GPU...")
    start = time.time()

    model = xgb.XGBClassifier(**XGB_GPU_PARAMS)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    # Get best score
    best_auc = model.best_score
    elapsed = time.time() - start

    print(f"  [XGBoost] AUC: {best_auc:.5f} | Time: {elapsed:.1f}s")
    return model, best_auc


def train_lightgbm(X_train, X_val, y_train, y_val) -> Tuple[Any, float]:
    """Train LightGBM with GPU acceleration."""
    print("\n  [LightGBM] Training with GPU...")
    start = time.time()

    model = lgb.LGBMClassifier(**LGB_GPU_PARAMS)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )

    # Get best score
    best_auc = model.best_score_['valid_0']['auc']
    elapsed = time.time() - start

    print(f"  [LightGBM] AUC: {best_auc:.5f} | Time: {elapsed:.1f}s")
    return model, best_auc


def train_catboost(X_train, X_val, y_train, y_val) -> Tuple[Any, float]:
    """Train CatBoost with GPU acceleration."""
    print("\n  [CatBoost] Training with GPU...")
    start = time.time()

    model = cb.CatBoostClassifier(**CB_GPU_PARAMS)

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=100
    )

    # Get best score
    best_auc = model.get_best_score()['validation']['AUC']
    elapsed = time.time() - start

    print(f"  [CatBoost] AUC: {best_auc:.5f} | Time: {elapsed:.1f}s")
    return model, best_auc


def train_symbol(symbol: str, output_dir: Path) -> Dict:
    """Train all models for a symbol."""
    print(f"\n{'='*70}")
    print(f"TRAINING {symbol}")
    print(f"{'='*70}")

    total_start = time.time()

    # Load data
    df = load_data(symbol)

    # Generate features
    features_df, feature_cols = generate_features(df)

    # Target: direction 10 ticks ahead (more realistic than 1 tick)
    target = 'target_direction_10'

    # Prepare data (use ALL data, no subsampling)
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(
        features_df, feature_cols, target
    )

    # Train models
    results = {}
    models = {}

    # XGBoost
    try:
        xgb_model, xgb_auc = train_xgboost(X_train, X_val, y_train, y_val)
        models['xgboost'] = xgb_model
        results['xgboost_auc'] = xgb_auc
    except Exception as e:
        print(f"  [XGBoost] ERROR: {e}")

    # LightGBM
    try:
        lgb_model, lgb_auc = train_lightgbm(X_train, X_val, y_train, y_val)
        models['lightgbm'] = lgb_model
        results['lightgbm_auc'] = lgb_auc
    except Exception as e:
        print(f"  [LightGBM] ERROR: {e}")

    # CatBoost
    try:
        cb_model, cb_auc = train_catboost(X_train, X_val, y_train, y_val)
        models['catboost'] = cb_model
        results['catboost_auc'] = cb_auc
    except Exception as e:
        print(f"  [CatBoost] ERROR: {e}")

    # Ensemble test accuracy
    if len(models) > 0:
        print(f"\n  [Ensemble] Testing on holdout set...")

        # Get predictions from each model
        preds = []
        for name, model in models.items():
            try:
                pred = model.predict_proba(X_test)[:, 1]
                preds.append(pred)
            except:
                pass

        if preds:
            # Average predictions
            ensemble_pred = np.mean(preds, axis=0)
            ensemble_binary = (ensemble_pred > 0.5).astype(int)

            # Calculate accuracy
            accuracy = (ensemble_binary == y_test).mean()
            results['ensemble_accuracy'] = accuracy
            print(f"  [Ensemble] Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Save models
    output_file = output_dir / f"{symbol}_models.pkl"
    save_data = {
        'target_direction_10': {
            **models,
            'features': feature_cols,
            'accuracy': results.get('ensemble_accuracy', 0),
        }
    }

    with open(output_file, 'wb') as f:
        pickle.dump(save_data, f)

    print(f"\n  Saved to {output_file}")
    print(f"  Features: {len(feature_cols)}")

    total_time = time.time() - total_start
    print(f"\n  Total time for {symbol}: {total_time:.1f}s ({total_time/60:.1f}m)")

    results['total_time'] = total_time
    return results


def main():
    """Main training loop."""
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
    output_dir = Path("models/production")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining {len(symbols)} symbols: {symbols}")
    print(f"Output: {output_dir}")

    # Track overall results
    all_results = {}
    total_start = time.time()

    for symbol in symbols:
        try:
            results = train_symbol(symbol, output_dir)
            all_results[symbol] = results
        except Exception as e:
            print(f"\nERROR training {symbol}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    total_time = time.time() - total_start

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 70)

    for symbol, results in all_results.items():
        acc = results.get('ensemble_accuracy', 0) * 100
        xgb = results.get('xgboost_auc', 0)
        lgb = results.get('lightgbm_auc', 0)
        cb = results.get('catboost_auc', 0)
        t = results.get('total_time', 0)

        print(f"\n{symbol}:")
        print(f"  Ensemble Accuracy: {acc:.2f}%")
        print(f"  XGBoost AUC: {xgb:.5f}")
        print(f"  LightGBM AUC: {lgb:.5f}")
        print(f"  CatBoost AUC: {cb:.5f}")
        print(f"  Time: {t:.1f}s")

    print(f"\nTotal training time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"\nModels saved to: {output_dir}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
