#!/usr/bin/env python3
"""
Train with COMPLETE Chinese Quant Arsenal
==========================================
Full feature set from all Chinese quant alpha libraries:

| Library    | Source                    | Factors |
|------------|---------------------------|---------|
| Alpha101   | WorldQuant (2015)         | 62      |
| Alpha158   | Microsoft Qlib            | 179     |
| Alpha360   | Microsoft Qlib (compact)  | 120     |
| Barra CNE6 | MSCI (adapted)            | 46      |
| Renaissance| Proprietary signals       | 51      |

TOTAL: 458 features (using compact Alpha360)
FULL: 698 features (using full Alpha360)

GPU: RTX 5080 (16GB VRAM) - Max Utilization
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import pickle
import json
import time
import gc
warnings.filterwarnings('ignore')

from core.features.alpha101 import Alpha101Complete
from core.features.alpha158 import Alpha158
from core.features.alpha360 import Alpha360Compact  # Using compact for speed
from core.features.barra_cne6 import BarraCNE6Forex
from core.features.renaissance import RenaissanceSignalGenerator
from core.ml.gpu_config import get_xgb_gpu_params, get_lgb_gpu_params, get_catboost_gpu_params, configure_gpu, print_system_info

import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split


def generate_all_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Generate ALL Chinese quant alpha features."""
    df = df.copy()

    # Prepare OHLCV from tick data
    df['mid'] = (df['bid'] + df['ask']) / 2
    df['price'] = df['mid']
    df['open'] = df['mid']
    df['high'] = df['mid']
    df['low'] = df['mid']
    df['close'] = df['mid']
    df['volume'] = 1
    df['vwap'] = df['mid']
    df['returns'] = df['close'].pct_change()

    all_features = []

    # 1. Alpha101 (62 factors)
    if verbose:
        print("    Generating Alpha101 (62 factors)...", flush=True)
    try:
        alpha101 = Alpha101Complete()
        result = alpha101.generate_all_alphas(df)
        alpha_cols = [c for c in result.columns if c.startswith('alpha')]
        all_features.append(result[alpha_cols])
        if verbose:
            print(f"      Generated: {len(alpha_cols)}", flush=True)
    except Exception as e:
        if verbose:
            print(f"      Alpha101 error: {e}", flush=True)

    # 2. Alpha158 (179 factors)
    if verbose:
        print("    Generating Alpha158 (179 factors)...", flush=True)
    try:
        alpha158 = Alpha158()
        result = alpha158.generate_all(df)
        all_features.append(result)
        if verbose:
            print(f"      Generated: {len(result.columns)}", flush=True)
    except Exception as e:
        if verbose:
            print(f"      Alpha158 error: {e}", flush=True)

    # 3. Alpha360 Compact (120 factors) - using compact for memory efficiency
    if verbose:
        print("    Generating Alpha360 Compact (120 factors)...", flush=True)
    try:
        alpha360 = Alpha360Compact(lookback=20)
        result = alpha360.generate_all(df)
        all_features.append(result)
        if verbose:
            print(f"      Generated: {len(result.columns)}", flush=True)
    except Exception as e:
        if verbose:
            print(f"      Alpha360 error: {e}", flush=True)

    # 4. Barra CNE6 (46 factors)
    if verbose:
        print("    Generating Barra CNE6 (46 factors)...", flush=True)
    try:
        barra = BarraCNE6Forex()
        result = barra.generate_all(df)
        all_features.append(result)
        if verbose:
            print(f"      Generated: {len(result.columns)}", flush=True)
    except Exception as e:
        if verbose:
            print(f"      Barra CNE6 error: {e}", flush=True)

    # 5. Renaissance (51 signals)
    if verbose:
        print("    Generating Renaissance (51 signals)...", flush=True)
    try:
        renaissance = RenaissanceSignalGenerator()
        result = renaissance.generate_all_signals(df)
        signal_cols = [c for c in result.columns if c.startswith('signal_')]
        all_features.append(result[signal_cols])
        if verbose:
            print(f"      Generated: {len(signal_cols)}", flush=True)
    except Exception as e:
        if verbose:
            print(f"      Renaissance error: {e}", flush=True)

    # Combine all features
    features = pd.concat(all_features, axis=1)
    features = features.fillna(0).replace([np.inf, -np.inf], 0)

    if verbose:
        print(f"    TOTAL FEATURES: {len(features.columns)}", flush=True)

    return features, df


def train_symbol(symbol: str, max_rows: int = 100000) -> dict:
    """Train a single symbol with full Chinese quant arsenal."""
    print(f'\n{"="*60}', flush=True)
    print(f'TRAINING {symbol} - CHINESE QUANT ARSENAL', flush=True)
    print(f'{"="*60}', flush=True)

    start_time = time.time()

    # Load data
    df = pd.read_parquet(f'training_package/{symbol}/train.parquet')
    original_rows = len(df)

    # Limit rows for memory management
    if len(df) > max_rows:
        df = df.head(max_rows)

    print(f'Using {len(df):,} rows (of {original_rows:,})', flush=True)

    # Generate features
    print('Generating features...', flush=True)
    features, df = generate_all_features(df)
    print(f'Total features: {len(features.columns)}', flush=True)

    # Create target: next tick direction
    target = (df['mid'].shift(-1) > df['mid']).astype(int)

    # Combine and clean
    data = pd.concat([features, target.rename('target')], axis=1)

    # Drop rows with any NaN in target (last row)
    data = data.dropna(subset=['target'])

    # Drop warmup period (first 100 rows may have NaN features)
    data = data.iloc[100:]

    print(f'Final dataset: {len(data):,} rows x {len(data.columns)} cols', flush=True)

    # Split data
    X = data.drop('target', axis=1).values
    y = data['target'].values
    feature_names = list(data.drop('target', axis=1).columns)

    # Time-series split (no shuffling for time series data)
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]

    print(f'Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}', flush=True)

    # Free memory
    del data, features, df
    gc.collect()

    # =========================================================================
    # TRAIN XGBOOST (GPU)
    # =========================================================================
    print('\nTraining XGBoost (GPU)...', flush=True)
    xgb_params = get_xgb_gpu_params()
    xgb_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 10,
        'learning_rate': 0.05,
    })

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    xgb_model = xgb.train(
        xgb_params, dtrain,
        num_boost_round=500,
        evals=[(dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=100
    )

    xgb_pred = xgb_model.predict(dtest)
    xgb_acc = accuracy_score(y_test, (xgb_pred > 0.5).astype(int))
    print(f'  XGBoost Accuracy: {xgb_acc:.4f}', flush=True)

    # =========================================================================
    # TRAIN LIGHTGBM (GPU)
    # =========================================================================
    print('\nTraining LightGBM (GPU)...', flush=True)
    lgb_params = get_lgb_gpu_params()
    lgb_params.update({
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'max_depth': 10,
        'learning_rate': 0.05,
        'num_leaves': 255,
    })

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    lgb_model = lgb.train(
        lgb_params, train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )

    lgb_pred = lgb_model.predict(X_test)
    lgb_acc = accuracy_score(y_test, (lgb_pred > 0.5).astype(int))
    print(f'  LightGBM Accuracy: {lgb_acc:.4f}', flush=True)

    # =========================================================================
    # TRAIN CATBOOST (GPU)
    # =========================================================================
    print('\nTraining CatBoost (GPU)...', flush=True)
    cb_params = get_catboost_gpu_params()
    cb_params.update({
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'verbose': 100,
        'iterations': 500,
        'early_stopping_rounds': 50,
        'depth': 8,
        'learning_rate': 0.05,
    })

    cb_model = cb.CatBoostClassifier(**cb_params)
    cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=100)

    cb_pred = cb_model.predict_proba(X_test)[:, 1]
    cb_acc = accuracy_score(y_test, (cb_pred > 0.5).astype(int))
    print(f'  CatBoost Accuracy: {cb_acc:.4f}', flush=True)

    # =========================================================================
    # ENSEMBLE
    # =========================================================================
    print('\nCalculating Ensemble...', flush=True)
    ensemble_pred = (xgb_pred + lgb_pred + cb_pred) / 3

    acc = accuracy_score(y_test, (ensemble_pred > 0.5).astype(int))
    auc = roc_auc_score(y_test, ensemble_pred)
    f1 = f1_score(y_test, (ensemble_pred > 0.5).astype(int))

    elapsed = time.time() - start_time

    print(f'\n{symbol} FINAL RESULTS:', flush=True)
    print(f'  Individual:', flush=True)
    print(f'    XGBoost:  {xgb_acc:.4f} ({xgb_acc*100:.2f}%)', flush=True)
    print(f'    LightGBM: {lgb_acc:.4f} ({lgb_acc*100:.2f}%)', flush=True)
    print(f'    CatBoost: {cb_acc:.4f} ({cb_acc*100:.2f}%)', flush=True)
    print(f'  Ensemble:', flush=True)
    print(f'    Accuracy: {acc:.4f} ({acc*100:.2f}%)', flush=True)
    print(f'    AUC:      {auc:.4f}', flush=True)
    print(f'    F1:       {f1:.4f}', flush=True)
    print(f'  Time: {elapsed:.1f}s', flush=True)

    # Save models
    output_dir = Path('models/chinese_quant_arsenal')
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
        'symbol': symbol,
        'accuracy': float(acc),
        'auc': float(auc),
        'f1': float(f1),
        'xgb_accuracy': float(xgb_acc),
        'lgb_accuracy': float(lgb_acc),
        'cb_accuracy': float(cb_acc),
        'num_features': len(feature_names),
        'num_train': int(len(X_train)),
        'num_val': int(len(X_val)),
        'num_test': int(len(X_test)),
        'training_time_seconds': float(elapsed)
    }

    with open(output_dir / f'{symbol}_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def main():
    """Train all symbols with Chinese quant arsenal."""
    print("\n" + "="*60)
    print("CHINESE QUANT ARSENAL TRAINING")
    print("="*60)
    print("Feature Libraries:")
    print("  - Alpha101 (WorldQuant): 62 factors")
    print("  - Alpha158 (Microsoft Qlib): 179 factors")
    print("  - Alpha360 Compact (Qlib): 120 factors")
    print("  - Barra CNE6 (MSCI): 46 factors")
    print("  - Renaissance (Proprietary): 51 signals")
    print("  TOTAL: ~458 features")
    print("="*60 + "\n")

    print_system_info()
    configure_gpu()

    total_start = time.time()

    # Symbols to train
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
    all_results = {}

    for symbol in symbols:
        try:
            results = train_symbol(symbol, max_rows=100000)
            all_results[symbol] = results
        except Exception as e:
            print(f'Error training {symbol}: {e}', flush=True)
            import traceback
            traceback.print_exc()

    total_time = time.time() - total_start

    # Summary
    print(f'\n{"="*60}', flush=True)
    print('CHINESE QUANT ARSENAL - TRAINING COMPLETE', flush=True)
    print(f'{"="*60}', flush=True)
    print(f'Total time: {total_time/60:.1f} minutes', flush=True)

    print('\n{:^10} {:^12} {:^12} {:^12} {:^10}'.format(
        'Symbol', 'Accuracy', 'AUC', 'F1', 'Features'), flush=True)
    print('-'*60, flush=True)

    for symbol, r in all_results.items():
        print('{:^10} {:^12.2%} {:^12.4f} {:^12.4f} {:^10}'.format(
            symbol, r['accuracy'], r['auc'], r['f1'], r['num_features']), flush=True)

    # Comparison with previous results
    print('\n' + '='*60, flush=True)
    print('COMPARISON: Previous vs Chinese Quant Arsenal', flush=True)
    print('='*60, flush=True)

    previous = {
        'EURUSD': {'basic': 0.6296, 'full_arsenal': 0.7039},
        'GBPUSD': {'basic': 0.5981, 'full_arsenal': 0.0},
        'USDJPY': {'basic': 0.5801, 'full_arsenal': 0.0}
    }

    print('{:^10} {:^12} {:^12} {:^12} {:^12}'.format(
        'Symbol', 'Basic', 'Prev Best', 'Chinese', 'Gain'), flush=True)
    print('-'*60, flush=True)

    for symbol in symbols:
        if symbol in all_results:
            basic = previous[symbol]['basic']
            prev_best = max(previous[symbol]['full_arsenal'], basic)
            current = all_results[symbol]['accuracy']
            gain = (current - prev_best) * 100
            print('{:^10} {:^12.2%} {:^12.2%} {:^12.2%} {:^+12.2f}%'.format(
                symbol, basic, prev_best, current, gain), flush=True)

    # Trading implications
    print('\n' + '='*60, flush=True)
    print('TRADING IMPLICATIONS', flush=True)
    print('='*60, flush=True)

    for symbol, r in all_results.items():
        acc = r['accuracy']
        edge = (acc - 0.5) * 100
        print(f'{symbol}:', flush=True)
        print(f'  Win Rate: {acc*100:.1f}%', flush=True)
        print(f'  Edge over Random: {edge:.1f}%', flush=True)

        # Kelly criterion
        p = acc
        q = 1 - p
        b = 1  # Assume 1:1 risk/reward
        kelly = (p * b - q) / b
        kelly_pct = kelly * 100

        print(f'  Kelly Optimal Bet: {kelly_pct:.1f}%', flush=True)
        print(f'  Fractional Kelly (25%): {kelly_pct * 0.25:.1f}%', flush=True)

    print('\n' + '='*60, flush=True)
    print('Models saved to: models/chinese_quant_arsenal/', flush=True)
    print('='*60, flush=True)


if __name__ == '__main__':
    main()
