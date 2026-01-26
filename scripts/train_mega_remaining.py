"""Train remaining 13 pairs with MEGA features (575+)."""
import os
import sys
import json
import pickle
import warnings
import threading
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Pairs needing retraining (currently <55%)
PAIRS = ['AUDCAD', 'AUDCHF', 'AUDNZD', 'AUDSGD', 'CADJPY', 'CHFJPY',
         'EURCAD', 'EURDKK', 'EURHUF', 'EURNOK', 'EURNZD', 'EURPLN', 'EURSEK']

BASE = Path(r'C:\Users\kevin\forex\forex')
TRAIN_PKG = BASE / 'training_package'
MODELS = BASE / 'models' / 'production'
MODELS.mkdir(parents=True, exist_ok=True)

# GPU lock
gpu_lock = threading.Lock()

def load_data(symbol):
    """Load parquet data."""
    import pandas as pd
    path = TRAIN_PKG / symbol
    train = pd.read_parquet(path / 'train.parquet')
    val = pd.read_parquet(path / 'val.parquet')
    test = pd.read_parquet(path / 'test.parquet')
    return train, val, test

def generate_mega_features(df, verbose=True):
    """Generate MEGA features (575+) using MegaFeatureGenerator."""
    import pandas as pd
    import numpy as np

    try:
        from core.features.mega_generator import MegaFeatureGenerator

        # Prepare OHLCV format
        ohlcv = df.copy()

        # Ensure required columns
        if 'bid' in df.columns and 'ask' in df.columns:
            ohlcv['close'] = (df['bid'] + df['ask']) / 2
            if 'open' not in ohlcv.columns:
                ohlcv['open'] = ohlcv['close'].shift(1).fillna(ohlcv['close'])
            if 'high' not in ohlcv.columns:
                ohlcv['high'] = ohlcv[['bid', 'ask']].max(axis=1)
            if 'low' not in ohlcv.columns:
                ohlcv['low'] = ohlcv[['bid', 'ask']].min(axis=1)
            if 'volume' not in ohlcv.columns:
                ohlcv['volume'] = 1  # Tick data, use 1

        # Init generator with subset of features for speed
        generator = MegaFeatureGenerator(
            enable_alpha101=True,
            enable_alpha158=True,
            enable_alpha360=False,  # Skip for speed
            enable_barra=True,
            enable_us_academic=True,
            enable_mlfinlab=False,  # Skip for speed
            enable_timeseries=False,  # Skip for speed
            enable_renaissance=True,
            alpha360_lookback=10,
            verbose=verbose
        )

        features = generator.generate_all(ohlcv)

        if verbose:
            counts = generator.get_feature_counts()
            print(f"  Feature counts: {counts}")
            print(f"  Total: {len(features.columns)}")

        return features

    except Exception as e:
        print(f"  MegaFeatureGenerator error: {e}")
        print("  Falling back to basic features...")
        return generate_basic_features(df)

def generate_basic_features(df):
    """Fallback basic feature generation."""
    import pandas as pd
    import numpy as np

    features = pd.DataFrame(index=df.index)
    mid = (df['bid'] + df['ask']) / 2 if 'bid' in df.columns else df['close']

    # Returns
    for h in [1, 2, 5, 10, 20, 50, 100]:
        features[f'ret_{h}'] = mid.pct_change(h)

    # MAs
    for w in [5, 10, 20, 50, 100, 200]:
        ma = mid.rolling(w, min_periods=1).mean()
        features[f'ma_ratio_{w}'] = mid / ma

    # Vol
    for w in [5, 10, 20, 50]:
        features[f'vol_{w}'] = mid.pct_change().rolling(w, min_periods=1).std()

    # RSI
    for w in [7, 14, 21]:
        delta = mid.diff()
        gain = delta.where(delta > 0, 0).rolling(w, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(w, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        features[f'rsi_{w}'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = mid.ewm(span=12).mean()
    ema26 = mid.ewm(span=26).mean()
    features['macd'] = ema12 - ema26
    features['macd_signal'] = features['macd'].ewm(span=9).mean()

    # BB
    for w in [10, 20]:
        ma = mid.rolling(w, min_periods=1).mean()
        std = mid.rolling(w, min_periods=1).std()
        features[f'bb_pos_{w}'] = (mid - (ma - 2*std)) / (4*std + 1e-10)

    features = features.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
    return features

def prepare_features(train, val, test, verbose=True):
    """Generate features and prepare X/y splits."""
    import pandas as pd
    import numpy as np

    if verbose:
        print("  Generating features for train...")
    train_feat = generate_mega_features(train, verbose=False)

    if verbose:
        print("  Generating features for val...")
    val_feat = generate_mega_features(val, verbose=False)

    if verbose:
        print("  Generating features for test...")
    test_feat = generate_mega_features(test, verbose=False)

    # Align columns
    common_cols = list(set(train_feat.columns) & set(val_feat.columns) & set(test_feat.columns))
    train_feat = train_feat[common_cols]
    val_feat = val_feat[common_cols]
    test_feat = test_feat[common_cols]

    # Create target - 10 tick direction
    mid_train = (train['bid'] + train['ask']) / 2 if 'bid' in train.columns else train['close']
    mid_val = (val['bid'] + val['ask']) / 2 if 'bid' in val.columns else val['close']
    mid_test = (test['bid'] + test['ask']) / 2 if 'bid' in test.columns else test['close']

    y_train = (mid_train.shift(-10) > mid_train).astype(int).reindex(train_feat.index)
    y_val = (mid_val.shift(-10) > mid_val).astype(int).reindex(val_feat.index)
    y_test = (mid_test.shift(-10) > mid_test).astype(int).reindex(test_feat.index)

    # Drop NaN
    mask_train = train_feat.notna().all(axis=1) & y_train.notna()
    mask_val = val_feat.notna().all(axis=1) & y_val.notna()
    mask_test = test_feat.notna().all(axis=1) & y_test.notna()

    X_train = train_feat[mask_train].values
    y_train = y_train[mask_train].values
    X_val = val_feat[mask_val].values
    y_val = y_val[mask_val].values
    X_test = test_feat[mask_test].values
    y_test = y_test[mask_test].values

    return X_train, y_train, X_val, y_val, X_test, y_test, len(common_cols)

def train_models(X_train, y_train, X_val, y_val):
    """Train XGB + LGB + CB ensemble on GPU."""
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    import numpy as np

    models = {}

    # XGBoost
    xgb_params = {
        'tree_method': 'hist', 'device': 'cuda',
        'max_depth': 10, 'max_bin': 256, 'learning_rate': 0.03,
        'subsample': 0.8, 'colsample_bytree': 0.6,
        'objective': 'binary:logistic', 'eval_metric': 'auc',
        'n_estimators': 600, 'early_stopping_rounds': 50
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    models['xgb'] = xgb.train(xgb_params, dtrain, num_boost_round=600,
                               evals=[(dval, 'val')], verbose_eval=False)

    # LightGBM
    lgb_params = {
        'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0,
        'max_depth': 10, 'num_leaves': 255, 'learning_rate': 0.03,
        'feature_fraction': 0.6, 'bagging_fraction': 0.8, 'bagging_freq': 5,
        'verbose': -1, 'objective': 'binary', 'metric': 'auc'
    }
    dtrain_lgb = lgb.Dataset(X_train, label=y_train)
    dval_lgb = lgb.Dataset(X_val, label=y_val, reference=dtrain_lgb)
    models['lgb'] = lgb.train(lgb_params, dtrain_lgb, num_boost_round=600,
                               valid_sets=[dval_lgb], callbacks=[lgb.early_stopping(50)])

    # CatBoost
    cb_params = {
        'task_type': 'GPU', 'devices': '0', 'depth': 8, 'iterations': 500,
        'learning_rate': 0.03, 'early_stopping_rounds': 30, 'verbose': False,
        'loss_function': 'Logloss'
    }
    models['cb'] = CatBoostClassifier(**cb_params)
    models['cb'].fit(X_train, y_train, eval_set=(X_val, y_val))

    # Ensemble predictions
    xgb_pred = models['xgb'].predict(dval)
    lgb_pred = models['lgb'].predict(X_val)
    cb_pred = models['cb'].predict_proba(X_val)[:, 1]

    ensemble_pred = (xgb_pred + lgb_pred + cb_pred) / 3
    ensemble_class = (ensemble_pred > 0.5).astype(int)

    metrics = {
        'accuracy': accuracy_score(y_val, ensemble_class),
        'auc': roc_auc_score(y_val, ensemble_pred),
        'f1': f1_score(y_val, ensemble_class)
    }

    return models, metrics

def process_pair(symbol):
    """Full pipeline for one pair."""
    try:
        print(f'\n[{symbol}] Loading data...')
        train, val, test = load_data(symbol)
        print(f'[{symbol}] Train: {len(train)}, Val: {len(val)}, Test: {len(test)}')

        print(f'[{symbol}] Generating MEGA features...')
        X_train, y_train, X_val, y_val, X_test, y_test, n_features = prepare_features(
            train, val, test, verbose=True
        )
        print(f'[{symbol}] Features: {n_features}, Train samples: {len(X_train)}')

        print(f'[{symbol}] Training models on GPU...')
        with gpu_lock:
            models, metrics = train_models(X_train, y_train, X_val, y_val)

        # Save
        model_path = MODELS / f'{symbol}_models.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(models, f)

        results = {
            'target_direction_10': metrics,
            '_meta': {
                'feature_mode': 'mega',
                'feature_count': n_features,
                'timestamp': datetime.now().isoformat()
            }
        }
        results_path = MODELS / f'{symbol}_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f'[{symbol}] DONE: {metrics["accuracy"]*100:.1f}% acc, {metrics["auc"]:.3f} AUC')
        return symbol, metrics['accuracy']

    except Exception as e:
        print(f'[{symbol}] ERROR: {e}')
        import traceback
        traceback.print_exc()
        return symbol, 0

def main():
    print('=' * 60)
    print('MEGA FEATURE TRAINING - 13 REMAINING PAIRS')
    print('=' * 60)

    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,temperature.gpu',
                            '--format=csv,noheader'], capture_output=True, text=True)
    print(f'GPU: {result.stdout.strip()}')
    print('=' * 60)

    results = []
    for i, pair in enumerate(PAIRS):
        print(f'\n[{i+1}/{len(PAIRS)}] Processing {pair}...')
        sym, acc = process_pair(pair)
        results.append((sym, acc))

        # Progress
        done = len(results)
        good = sum(1 for _, a in results if a > 0.55)
        avg = sum(a for _, a in results if a > 0) / max(1, sum(1 for _, a in results if a > 0))
        print(f'\n--- Progress: {done}/{len(PAIRS)} | >=55%: {good} | Avg: {avg*100:.1f}% ---')

    print('\n' + '=' * 60)
    print('FINAL RESULTS:')
    print('=' * 60)
    for sym, acc in sorted(results, key=lambda x: -x[1]):
        status = '✓' if acc >= 0.55 else '✗'
        print(f'  {status} {sym}: {acc*100:.1f}%')

if __name__ == '__main__':
    main()
