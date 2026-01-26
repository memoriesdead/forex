#!/usr/bin/env python3
"""
Quick Full Arsenal Training - Optimized for Speed
Uses first 50k rows per symbol to get quick validation results.
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'core/_experimental')

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import pickle
import json
import time
warnings.filterwarnings('ignore')

from core.features.alpha101 import Alpha101Complete
from core.features.renaissance import RenaissanceSignalGenerator
from core.ml.gpu_config import get_xgb_gpu_params, get_lgb_gpu_params, get_catboost_gpu_params, configure_gpu, print_system_info

import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split


def generate_full_features(df):
    """Generate all Alpha101 + Renaissance features."""
    # Prepare OHLCV
    df = df.copy()
    df['mid'] = (df['bid'] + df['ask']) / 2
    df['price'] = df['mid']
    df['open'] = df['mid']
    df['high'] = df['mid']
    df['low'] = df['mid']
    df['close'] = df['mid']
    df['volume'] = 1
    df['vwap'] = df['mid']
    df['returns'] = df['close'].pct_change()

    # Generate Alpha101
    print("    Generating Alpha101 (62 factors)...", flush=True)
    alpha101 = Alpha101Complete()
    alpha_result = alpha101.generate_all_alphas(df)
    alpha_cols = [c for c in alpha_result.columns if c.startswith('alpha')]

    # Generate Renaissance
    print("    Generating Renaissance (51 signals)...", flush=True)
    renaissance = RenaissanceSignalGenerator()
    ren_result = renaissance.generate_all_signals(df)
    signal_cols = [c for c in ren_result.columns if c.startswith('signal_')]

    # Combine
    features = pd.concat([alpha_result[alpha_cols], ren_result[signal_cols]], axis=1)
    features = features.fillna(0).replace([np.inf, -np.inf], 0)

    return features, df


def train_symbol(symbol, max_rows=50000):
    """Train a single symbol with full arsenal features."""
    print(f'\n{"="*60}', flush=True)
    print(f'TRAINING {symbol} - FULL ARSENAL', flush=True)
    print(f'{"="*60}', flush=True)

    start_time = time.time()

    # Load data
    df = pd.read_parquet(f'training_package/{symbol}/train.parquet')
    original_rows = len(df)

    # Limit rows for speed
    if len(df) > max_rows:
        df = df.head(max_rows)

    print(f'Using {len(df):,} rows (of {original_rows:,})', flush=True)

    # Generate features
    print('Generating features...', flush=True)
    features, df = generate_full_features(df)
    print(f'  Total features: {len(features.columns)}', flush=True)

    # Create target
    target = (df['mid'].shift(-1) > df['mid']).astype(int)

    # Combine and clean
    data = pd.concat([features, target.rename('target')], axis=1).dropna()
    print(f'Final dataset: {len(data):,} rows', flush=True)

    # Split
    X = data.drop('target', axis=1).values
    y = data['target'].values
    feature_names = list(data.drop('target', axis=1).columns)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f'Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}', flush=True)

    # Train XGBoost
    print('Training XGBoost (GPU)...', flush=True)
    xgb_params = get_xgb_gpu_params()
    xgb_params.update({'objective': 'binary:logistic', 'eval_metric': 'auc'})

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=500,
        evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=100)

    xgb_pred = xgb_model.predict(dtest)

    # Train LightGBM
    print('Training LightGBM (GPU)...', flush=True)
    lgb_params = get_lgb_gpu_params()
    lgb_params.update({'objective': 'binary', 'metric': 'auc', 'verbosity': -1})

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    lgb_model = lgb.train(lgb_params, train_data, num_boost_round=500,
        valid_sets=[val_data], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])

    lgb_pred = lgb_model.predict(X_test)

    # Train CatBoost
    print('Training CatBoost (GPU)...', flush=True)
    cb_params = get_catboost_gpu_params()
    cb_params.update({'loss_function': 'Logloss', 'eval_metric': 'AUC',
                     'verbose': 100, 'iterations': 500, 'early_stopping_rounds': 50})

    cb_model = cb.CatBoostClassifier(**cb_params)
    cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=100)

    cb_pred = cb_model.predict_proba(X_test)[:, 1]

    # Ensemble
    ensemble_pred = (xgb_pred + lgb_pred + cb_pred) / 3

    # Metrics
    acc = accuracy_score(y_test, (ensemble_pred > 0.5).astype(int))
    auc = roc_auc_score(y_test, ensemble_pred)
    f1 = f1_score(y_test, (ensemble_pred > 0.5).astype(int))

    elapsed = time.time() - start_time

    print(f'\n{symbol} RESULTS:', flush=True)
    print(f'  Accuracy: {acc:.4f} ({acc*100:.2f}%)', flush=True)
    print(f'  AUC: {auc:.4f}', flush=True)
    print(f'  F1: {f1:.4f}', flush=True)
    print(f'  Time: {elapsed:.1f}s', flush=True)

    # Save models
    output_dir = Path('models/full_arsenal')
    output_dir.mkdir(exist_ok=True, parents=True)

    models = {
        'xgboost': xgb_model,
        'lightgbm': lgb_model,
        'catboost': cb_model,
        'features': feature_names
    }

    with open(output_dir / f'{symbol}_models.pkl', 'wb') as f:
        pickle.dump(models, f)

    results = {
        'accuracy': float(acc),
        'auc': float(auc),
        'f1': float(f1),
        'num_features': len(feature_names),
        'num_train': len(X_train),
        'num_test': len(X_test),
        'training_time_seconds': elapsed
    }

    with open(output_dir / f'{symbol}_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return acc, auc, f1


def main():
    """Train all symbols with full arsenal."""
    print_system_info()
    configure_gpu()

    total_start = time.time()

    symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
    all_results = {}

    for symbol in symbols:
        try:
            acc, auc, f1 = train_symbol(symbol, max_rows=50000)
            all_results[symbol] = {'accuracy': acc, 'auc': auc, 'f1': f1}
        except Exception as e:
            print(f'Error training {symbol}: {e}', flush=True)
            import traceback
            traceback.print_exc()

    total_time = time.time() - total_start

    # Summary
    print(f'\n{"="*60}', flush=True)
    print('FULL ARSENAL TRAINING COMPLETE', flush=True)
    print(f'{"="*60}', flush=True)
    print(f'Total time: {total_time/60:.1f} minutes', flush=True)

    print('\n{:^10} {:^12} {:^12} {:^12}'.format('Symbol', 'Accuracy', 'AUC', 'F1'), flush=True)
    print('-'*50, flush=True)
    for symbol, r in all_results.items():
        print('{:^10} {:^12.2%} {:^12.4f} {:^12.4f}'.format(symbol, r['accuracy'], r['auc'], r['f1']), flush=True)

    # Comparison
    print('\n' + '='*60, flush=True)
    print('COMPARISON: Basic Features vs Full Arsenal', flush=True)
    print('='*60, flush=True)
    basic = {'EURUSD': 0.6296, 'GBPUSD': 0.5981, 'USDJPY': 0.5801}
    print('{:^10} {:^12} {:^12} {:^12}'.format('Symbol', 'Basic', 'Full', 'Gain'), flush=True)
    print('-'*50, flush=True)
    for symbol in symbols:
        if symbol in all_results:
            b = basic[symbol]
            f = all_results[symbol]['accuracy']
            print('{:^10} {:^12.2%} {:^12.2%} {:^+12.2f}%'.format(symbol, b, f, (f-b)*100), flush=True)

    # Trading implications
    print('\n' + '='*60, flush=True)
    print('TRADING IMPLICATIONS', flush=True)
    print('='*60, flush=True)
    for symbol, r in all_results.items():
        acc = r['accuracy']
        edge = (acc - 0.5) * 100
        print(f'{symbol}:', flush=True)
        print(f'  Win Rate: {acc*100:.1f}%', flush=True)
        print(f'  Edge: {edge:.1f}% over random', flush=True)
        # Simplified P&L estimate (10 trades/day at $100 per trade)
        daily_trades = 10
        trade_size = 100
        expected_daily = trade_size * daily_trades * (acc - 0.5)
        print(f'  Expected daily P&L (10 trades @ $100): ${expected_daily:.2f}', flush=True)


if __name__ == '__main__':
    main()
