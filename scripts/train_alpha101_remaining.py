"""Train remaining 13 pairs with Alpha101 features (proven 72%+ accuracy)."""
import os
import sys
import json
import pickle
import warnings
import threading
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

# Pairs to retrain
PAIRS = ['AUDCAD', 'AUDCHF', 'AUDNZD', 'AUDSGD', 'CADJPY', 'CHFJPY',
         'EURCAD', 'EURDKK', 'EURHUF', 'EURNOK', 'EURNZD', 'EURPLN', 'EURSEK']

BASE = Path(r'C:\Users\kevin\forex\forex')
TRAIN_PKG = BASE / 'training_package'
MODELS = BASE / 'models' / 'production'
gpu_lock = threading.Lock()

# Alpha101 formulas adapted for forex
def alpha001(close, returns):
    """Signed power of returns - reversal signal"""
    return np.sign(returns) * np.sqrt(np.abs(returns))

def alpha002(open_, close, volume):
    """-1 * correlation(rank(delta(log(volume), 2)), rank((close - open) / open), 6)"""
    delta_log_vol = np.log(volume + 1).diff(2)
    co_ratio = (close - open_) / (open_ + 1e-10)
    return -delta_log_vol.rolling(6).corr(co_ratio)

def alpha003(open_, volume):
    """-1 * correlation(rank(open), rank(volume), 10)"""
    return -open_.rolling(10).corr(volume)

def alpha004(low):
    """-1 * ts_rank(rank(low), 9)"""
    return -low.rolling(9).apply(lambda x: pd.Series(x).rank().iloc[-1] / len(x), raw=False)

def alpha005(open_, vwap, close):
    """rank(open - mean(vwap, 10)) * rank(close - vwap)^(-1)"""
    vwap_ma = vwap.rolling(10).mean()
    return (open_ - vwap_ma) * (1 / ((close - vwap).abs() + 1e-10))

def alpha006(open_, volume):
    """-1 * correlation(open, volume, 10)"""
    return -open_.rolling(10).corr(volume)

def alpha007(close, volume):
    """ts_rank(volume,5) if close < close[-7] else -ts_rank(abs(delta_close),60)"""
    delta_close = close.diff(7)
    vol_rank = volume.rolling(5).apply(lambda x: pd.Series(x).rank().iloc[-1] / len(x), raw=False)
    close_rank = close.diff().abs().rolling(60, min_periods=1).apply(
        lambda x: pd.Series(x).rank().iloc[-1] / len(x), raw=False)
    return np.where(delta_close < 0, vol_rank, -close_rank)

def alpha008(open_, returns):
    """-1 * rank(sum(open*returns - delay(open, 5)*returns[-5], 5))"""
    delayed = open_.shift(5) * returns.shift(5)
    current = open_ * returns
    return -(current - delayed).rolling(5).sum()

def alpha009(close):
    """ts_min(close, 5) if close > delay(close, 1) else ts_max(close, 5)"""
    close_prev = close.shift(1)
    return np.where(close > close_prev,
                    close.rolling(5).min(),
                    close.rolling(5).max())

def alpha010(close):
    """rank(close - ts_max(close, 5)) if close > delay(close, 1) else rank(close - ts_min(close, 5))"""
    close_prev = close.shift(1)
    max5 = close.rolling(5).max()
    min5 = close.rolling(5).min()
    return np.where(close > close_prev, close - max5, close - min5)

def generate_alpha101_features(df):
    """Generate Alpha101 features."""
    features = pd.DataFrame(index=df.index)

    # Prepare price data
    if 'bid' in df.columns and 'ask' in df.columns:
        close = (df['bid'] + df['ask']) / 2
        open_ = close.shift(1).fillna(close)
        high = df[['bid', 'ask']].max(axis=1)
        low = df[['bid', 'ask']].min(axis=1)
        spread = df['ask'] - df['bid']
    else:
        close = df['close']
        open_ = df.get('open', close.shift(1).fillna(close))
        high = df.get('high', close)
        low = df.get('low', close)
        spread = pd.Series(0.0001, index=df.index)

    volume = df.get('volume', pd.Series(1, index=df.index))
    vwap = (high + low + close) / 3

    # Returns at multiple horizons
    returns = close.pct_change()

    # Alpha101 formulas
    features['alpha001'] = alpha001(close, returns)
    features['alpha002'] = alpha002(open_, close, volume)
    features['alpha003'] = alpha003(open_, volume)
    features['alpha004'] = alpha004(low)
    features['alpha005'] = alpha005(open_, vwap, close)
    features['alpha006'] = alpha006(open_, volume)
    features['alpha007'] = alpha007(close, volume)
    features['alpha008'] = alpha008(open_, returns)
    features['alpha009'] = alpha009(close)
    features['alpha010'] = alpha010(close)

    # Technical indicators
    for w in [5, 10, 20, 50, 100, 200]:
        ma = close.rolling(w, min_periods=1).mean()
        features[f'ma_ratio_{w}'] = close / ma
        features[f'ma_dist_{w}'] = (close - ma) / close
        features[f'ma_slope_{w}'] = ma.diff(5) / ma

    # EMAs
    for w in [5, 10, 20, 50]:
        ema = close.ewm(span=w).mean()
        features[f'ema_ratio_{w}'] = close / ema

    # Returns at multiple horizons
    for h in [1, 2, 5, 10, 20, 50, 100]:
        features[f'ret_{h}'] = close.pct_change(h)
        features[f'ret_lag_{h}'] = close.pct_change(h).shift(h)

    # Volatility
    for w in [5, 10, 20, 50, 100]:
        features[f'vol_{w}'] = returns.rolling(w, min_periods=1).std()
        features[f'vol_ratio_{w}'] = features[f'vol_{w}'] / features[f'vol_{w}'].shift(w).fillna(features[f'vol_{w}'])

    # RSI
    for w in [7, 14, 21, 50]:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(w, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(w, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        features[f'rsi_{w}'] = 100 - (100 / (1 + rs))

    # MACD
    for slow, fast in [(26, 12), (50, 20)]:
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=9).mean()
        features[f'macd_{fast}_{slow}'] = macd
        features[f'macd_signal_{fast}_{slow}'] = signal
        features[f'macd_hist_{fast}_{slow}'] = macd - signal

    # Bollinger Bands
    for w in [10, 20, 50]:
        ma = close.rolling(w, min_periods=1).mean()
        std = close.rolling(w, min_periods=1).std()
        features[f'bb_upper_{w}'] = (close - (ma + 2*std)) / close
        features[f'bb_lower_{w}'] = (close - (ma - 2*std)) / close
        features[f'bb_width_{w}'] = (4 * std) / close
        features[f'bb_pos_{w}'] = (close - (ma - 2*std)) / (4*std + 1e-10)

    # Momentum
    for w in [5, 10, 20, 50]:
        features[f'mom_{w}'] = close - close.shift(w)
        features[f'mom_ratio_{w}'] = close / close.shift(w)
        features[f'roc_{w}'] = (close / close.shift(w) - 1) * 100

    # Acceleration
    for w in [5, 10, 20]:
        mom = close - close.shift(w)
        features[f'acc_{w}'] = mom - mom.shift(w)

    # High-Low range
    for w in [5, 10, 20, 50]:
        features[f'range_{w}'] = (high.rolling(w).max() - low.rolling(w).min()) / close
        features[f'atr_{w}'] = pd.concat([high - low, (high - close.shift(1)).abs(),
                                          (low - close.shift(1)).abs()], axis=1).max(axis=1).rolling(w).mean() / close

    # Spread features
    features['spread'] = spread
    features['spread_ma'] = spread.rolling(20, min_periods=1).mean()
    features['spread_ratio'] = spread / (features['spread_ma'] + 1e-10)
    features['spread_vol'] = spread.rolling(20, min_periods=1).std()

    # Price position in range
    for w in [5, 10, 20, 50, 100]:
        max_p = close.rolling(w, min_periods=1).max()
        min_p = close.rolling(w, min_periods=1).min()
        features[f'price_pos_{w}'] = (close - min_p) / (max_p - min_p + 1e-10)

    # Volume features
    features['vol_ma20'] = volume.rolling(20, min_periods=1).mean()
    features['vol_ratio'] = volume / (features['vol_ma20'] + 1e-10)

    # Cross-period correlations
    for w1, w2 in [(5, 20), (10, 50), (20, 100)]:
        features[f'corr_{w1}_{w2}'] = returns.rolling(w1).corr(returns.rolling(w2).std())

    # Trend strength
    for w in [10, 20, 50]:
        adx_period = w
        plus_dm = (high - high.shift(1)).clip(lower=0)
        minus_dm = (low.shift(1) - low).clip(lower=0)
        tr = pd.concat([high - low, (high - close.shift(1)).abs(),
                        (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(adx_period, min_periods=1).mean()
        plus_di = 100 * (plus_dm.rolling(adx_period, min_periods=1).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(adx_period, min_periods=1).mean() / (atr + 1e-10))
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        features[f'adx_{w}'] = dx.rolling(adx_period, min_periods=1).mean()
        features[f'plus_di_{w}'] = plus_di
        features[f'minus_di_{w}'] = minus_di

    # Clean up
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.ffill().bfill().fillna(0)

    return features

def train_models(X_train, y_train, X_val, y_val):
    """Train ensemble on GPU."""
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

    models = {}

    # XGBoost - deeper trees
    xgb_params = {
        'tree_method': 'hist', 'device': 'cuda',
        'max_depth': 12, 'max_bin': 256, 'learning_rate': 0.02,
        'subsample': 0.8, 'colsample_bytree': 0.5,
        'reg_alpha': 0.1, 'reg_lambda': 1.0,
        'objective': 'binary:logistic', 'eval_metric': 'auc',
        'early_stopping_rounds': 80
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    models['xgb'] = xgb.train(xgb_params, dtrain, num_boost_round=1000,
                               evals=[(dval, 'val')], verbose_eval=False)

    # LightGBM
    lgb_params = {
        'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0,
        'max_depth': 12, 'num_leaves': 511, 'learning_rate': 0.02,
        'feature_fraction': 0.5, 'bagging_fraction': 0.8, 'bagging_freq': 5,
        'lambda_l1': 0.1, 'lambda_l2': 1.0,
        'verbose': -1, 'objective': 'binary', 'metric': 'auc'
    }
    dtrain_lgb = lgb.Dataset(X_train, label=y_train)
    dval_lgb = lgb.Dataset(X_val, label=y_val, reference=dtrain_lgb)
    models['lgb'] = lgb.train(lgb_params, dtrain_lgb, num_boost_round=1000,
                               valid_sets=[dval_lgb], callbacks=[lgb.early_stopping(80)])

    # CatBoost
    cb_params = {
        'task_type': 'GPU', 'devices': '0', 'depth': 10, 'iterations': 800,
        'learning_rate': 0.02, 'early_stopping_rounds': 50, 'verbose': False,
        'loss_function': 'Logloss', 'l2_leaf_reg': 3.0
    }
    models['cb'] = CatBoostClassifier(**cb_params)
    models['cb'].fit(X_train, y_train, eval_set=(X_val, y_val))

    # Ensemble
    xgb_pred = models['xgb'].predict(dval)
    lgb_pred = models['lgb'].predict(X_val)
    cb_pred = models['cb'].predict_proba(X_val)[:, 1]

    ensemble_pred = (xgb_pred + lgb_pred + cb_pred) / 3
    ensemble_class = (ensemble_pred > 0.5).astype(int)

    return models, {
        'accuracy': accuracy_score(y_val, ensemble_class),
        'auc': roc_auc_score(y_val, ensemble_pred),
        'f1': f1_score(y_val, ensemble_class)
    }

def process_pair(symbol):
    """Full pipeline for one pair."""
    try:
        print(f'[{symbol}] Loading data...')
        path = TRAIN_PKG / symbol
        train = pd.read_parquet(path / 'train.parquet')
        val = pd.read_parquet(path / 'val.parquet')
        print(f'[{symbol}] Train: {len(train)}, Val: {len(val)}')

        print(f'[{symbol}] Generating Alpha101 features...')
        train_feat = generate_alpha101_features(train)
        val_feat = generate_alpha101_features(val)

        # Align columns
        common = list(set(train_feat.columns) & set(val_feat.columns))
        train_feat = train_feat[common]
        val_feat = val_feat[common]

        # Target
        mid_train = (train['bid'] + train['ask']) / 2 if 'bid' in train.columns else train['close']
        mid_val = (val['bid'] + val['ask']) / 2 if 'bid' in val.columns else val['close']

        y_train = (mid_train.shift(-10) > mid_train).astype(int).reindex(train_feat.index).dropna()
        y_val = (mid_val.shift(-10) > mid_val).astype(int).reindex(val_feat.index).dropna()

        # Align
        idx_train = train_feat.index.intersection(y_train.index)
        idx_val = val_feat.index.intersection(y_val.index)

        X_train = train_feat.loc[idx_train].values
        y_train = y_train.loc[idx_train].values
        X_val = val_feat.loc[idx_val].values
        y_val = y_val.loc[idx_val].values

        print(f'[{symbol}] Training ({len(X_train)} samples, {len(common)} features)...')

        with gpu_lock:
            models, metrics = train_models(X_train, y_train, X_val, y_val)

        # Save
        with open(MODELS / f'{symbol}_models.pkl', 'wb') as f:
            pickle.dump(models, f)

        results = {
            'target_direction_10': metrics,
            '_meta': {
                'feature_mode': 'alpha101',
                'feature_count': len(common),
                'timestamp': datetime.now().isoformat()
            }
        }
        with open(MODELS / f'{symbol}_results.json', 'w') as f:
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
    print('ALPHA101 TRAINING - 13 REMAINING PAIRS')
    print('=' * 60)

    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,temperature.gpu',
                            '--format=csv,noheader'], capture_output=True, text=True)
    print(f'GPU: {result.stdout.strip()}')

    results = []
    for i, pair in enumerate(PAIRS):
        print(f'\n[{i+1}/{len(PAIRS)}] Processing {pair}...')
        sym, acc = process_pair(pair)
        results.append((sym, acc))

        # Progress
        good = sum(1 for _, a in results if a >= 0.55)
        avg = sum(a for _, a in results if a > 0) / max(1, sum(1 for _, a in results if a > 0))
        print(f'--- Progress: {len(results)}/{len(PAIRS)} | >=55%: {good} | Avg: {avg*100:.1f}% ---')

    print('\n' + '=' * 60)
    print('FINAL:')
    for sym, acc in sorted(results, key=lambda x: -x[1]):
        print(f'  {sym}: {acc*100:.1f}%')

if __name__ == '__main__':
    main()
