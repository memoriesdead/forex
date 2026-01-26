"""Train remaining 13 pairs that need feature generation."""
import os
import sys
import json
import pickle
import warnings
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# Pairs needing training
PAIRS = ['AUDCAD', 'AUDCHF', 'AUDNZD', 'AUDSGD', 'CADJPY', 'CHFJPY',
         'EURCAD', 'EURDKK', 'EURHUF', 'EURNOK', 'EURNZD', 'EURPLN', 'EURSEK']

BASE = Path(r'C:\Users\kevin\forex\forex')
TRAIN_PKG = BASE / 'training_package'
MODELS = BASE / 'models' / 'production'
MODELS.mkdir(parents=True, exist_ok=True)

# GPU lock for serialized training
gpu_lock = threading.Lock()

def load_data(symbol):
    """Load parquet data for symbol."""
    import pandas as pd
    path = TRAIN_PKG / symbol
    train = pd.read_parquet(path / 'train.parquet')
    val = pd.read_parquet(path / 'val.parquet')
    test = pd.read_parquet(path / 'test.parquet')
    return train, val, test

def generate_features(df):
    """Generate minimal but effective features."""
    import numpy as np
    import pandas as pd

    features = pd.DataFrame(index=df.index)

    # Price columns
    mid = (df['bid'] + df['ask']) / 2 if 'bid' in df.columns else df['close']
    spread = df['ask'] - df['bid'] if 'bid' in df.columns else pd.Series(0, index=df.index)

    # Returns at multiple horizons
    for h in [1, 2, 5, 10, 20, 50]:
        features[f'ret_{h}'] = mid.pct_change(h)
        features[f'ret_lag_{h}'] = mid.pct_change(h).shift(h)

    # Moving averages
    for w in [5, 10, 20, 50, 100]:
        ma = mid.rolling(w).mean()
        features[f'ma_{w}'] = ma
        features[f'ma_ratio_{w}'] = mid / ma
        features[f'ma_dist_{w}'] = (mid - ma) / mid

    # Volatility
    for w in [5, 10, 20, 50]:
        features[f'vol_{w}'] = mid.pct_change().rolling(w).std()
        features[f'vol_ratio_{w}'] = features[f'vol_{w}'] / features[f'vol_{w}'].shift(w)

    # RSI
    for w in [7, 14, 21]:
        delta = mid.diff()
        gain = delta.where(delta > 0, 0).rolling(w).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(w).mean()
        rs = gain / (loss + 1e-10)
        features[f'rsi_{w}'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = mid.ewm(span=12).mean()
    ema26 = mid.ewm(span=26).mean()
    features['macd'] = ema12 - ema26
    features['macd_signal'] = features['macd'].ewm(span=9).mean()
    features['macd_hist'] = features['macd'] - features['macd_signal']

    # Bollinger Bands
    for w in [10, 20]:
        ma = mid.rolling(w).mean()
        std = mid.rolling(w).std()
        features[f'bb_upper_{w}'] = (mid - (ma + 2*std)) / mid
        features[f'bb_lower_{w}'] = (mid - (ma - 2*std)) / mid
        features[f'bb_width_{w}'] = (4 * std) / mid
        features[f'bb_pos_{w}'] = (mid - (ma - 2*std)) / (4*std + 1e-10)

    # Momentum
    for w in [5, 10, 20]:
        features[f'mom_{w}'] = mid - mid.shift(w)
        features[f'mom_acc_{w}'] = features[f'mom_{w}'] - features[f'mom_{w}'].shift(w)

    # Rate of change
    for w in [5, 10, 20]:
        features[f'roc_{w}'] = (mid / mid.shift(w) - 1) * 100

    # Spread features
    features['spread'] = spread
    features['spread_ma'] = spread.rolling(20).mean()
    features['spread_ratio'] = spread / (features['spread_ma'] + 1e-10)

    # Hour/minute features (if timestamp available)
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'])
        features['hour'] = ts.dt.hour
        features['minute'] = ts.dt.minute
        features['day_of_week'] = ts.dt.dayofweek
        # Cyclical encoding
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)

    # Cross-features
    features['ret_vol_ratio'] = features['ret_1'].abs() / (features['vol_5'] + 1e-10)
    features['trend_strength'] = features['ma_ratio_5'] - features['ma_ratio_50']

    # Target - direction 10 ticks ahead
    features['target'] = (mid.shift(-10) > mid).astype(int)

    # Drop NaN
    features = features.replace([np.inf, -np.inf], np.nan).dropna()

    return features

def train_models(X_train, y_train, X_val, y_val):
    """Train XGB, LGB, CB ensemble."""
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    import numpy as np

    models = {}

    # XGBoost
    xgb_params = {
        'tree_method': 'hist', 'device': 'cuda',
        'max_depth': 8, 'max_bin': 128, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.7,
        'objective': 'binary:logistic', 'eval_metric': 'auc',
        'nthread': 6, 'early_stopping_rounds': 30
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    models['xgb'] = xgb.train(xgb_params, dtrain, num_boost_round=400,
                               evals=[(dval, 'val')], verbose_eval=False)

    # LightGBM
    lgb_params = {
        'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0,
        'max_depth': 8, 'num_leaves': 127, 'learning_rate': 0.05,
        'feature_fraction': 0.7, 'bagging_fraction': 0.8, 'bagging_freq': 5,
        'verbose': -1, 'objective': 'binary', 'metric': 'auc', 'n_jobs': 6
    }
    dtrain_lgb = lgb.Dataset(X_train, label=y_train)
    dval_lgb = lgb.Dataset(X_val, label=y_val, reference=dtrain_lgb)
    models['lgb'] = lgb.train(lgb_params, dtrain_lgb, num_boost_round=400,
                               valid_sets=[dval_lgb], callbacks=[lgb.early_stopping(30)])

    # CatBoost
    cb_params = {
        'task_type': 'GPU', 'devices': '0', 'depth': 6, 'iterations': 300,
        'learning_rate': 0.05, 'early_stopping_rounds': 20, 'verbose': False,
        'loss_function': 'Logloss', 'thread_count': 6
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
    """Generate features and train for one pair."""
    try:
        print(f'[{symbol}] Loading data...')
        train, val, test = load_data(symbol)

        print(f'[{symbol}] Generating features...')
        train_feat = generate_features(train)
        val_feat = generate_features(val)
        test_feat = generate_features(test)

        # Split X/y
        target_col = 'target'
        X_train = train_feat.drop(columns=[target_col])
        y_train = train_feat[target_col]
        X_val = val_feat.drop(columns=[target_col])
        y_val = val_feat[target_col]
        X_test = test_feat.drop(columns=[target_col])
        y_test = test_feat[target_col]

        print(f'[{symbol}] Training ({len(X_train)} samples, {len(X_train.columns)} features)...')

        # GPU training with lock
        with gpu_lock:
            models, metrics = train_models(X_train, y_train, X_val, y_val)

        # Save models
        model_path = MODELS / f'{symbol}_models.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(models, f)

        # Save results
        results = {
            'target_direction_10': metrics,
            '_meta': {
                'feature_count': len(X_train.columns),
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
    print(f'Training {len(PAIRS)} pairs...')
    print('=' * 50)

    results = []
    for pair in PAIRS:
        sym, acc = process_pair(pair)
        results.append((sym, acc))

        # Progress summary
        done = len(results)
        avg_acc = sum(a for _, a in results if a > 0) / max(1, sum(1 for _, a in results if a > 0))
        print(f'\nProgress: {done}/{len(PAIRS)} | Avg: {avg_acc*100:.1f}%\n')

    print('=' * 50)
    print('FINAL RESULTS:')
    for sym, acc in sorted(results, key=lambda x: -x[1]):
        print(f'  {sym}: {acc*100:.1f}%')

if __name__ == '__main__':
    main()
