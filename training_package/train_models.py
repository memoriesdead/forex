#!/usr/bin/env python3
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
    print(f"\n{'='*50}")
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
        print(f"\nTarget: {target}")

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

    print(f"\nSaved models to {model_file}")
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
    print(f"\n{'='*50}")
    print("TRAINING COMPLETE")
    print(f"{'='*50}")

    for symbol, results in all_results.items():
        print(f"\n{symbol}:")
        for target, metrics in results.items():
            print(f"  {target}: ACC={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}")

    # Package models
    import shutil
    shutil.make_archive('hft_models', 'gztar', output_dir)
    print(f"\nModels packaged to hft_models.tar.gz")


if __name__ == "__main__":
    main()
