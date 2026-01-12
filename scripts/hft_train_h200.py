#!/usr/bin/env python3
"""
HFT Forex Training Script - H200 GPU
Uses Renaissance Technologies weak signals methodology
Goal: Train ensemble for $100 -> highest ROI
"""

import os
import glob
import time
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

print("="*70)
print("HFT FOREX TRAINING - H200 GPU")
print("Renaissance Technologies Weak Signals + ML Ensemble")
print("="*70)
print(f"Start time: {datetime.now()}")
start_time = time.time()

# ========== DATA LOADING ==========
print("\n[1/6] Loading data...")

def load_dukascopy_data(data_dir):
    """Load all Dukascopy historical data"""
    all_data = []
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

    for f in sorted(csv_files):
        pair = os.path.basename(f).split("_")[0]
        try:
            df = pd.read_csv(f)
            df["pair"] = pair
            all_data.append(df)
        except Exception as e:
            print(f"  Error loading {f}: {e}")

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print(f"  Loaded {len(csv_files)} files, {len(combined):,} rows")
        return combined
    return pd.DataFrame()

def load_live_data(data_dir):
    """Load live tick data"""
    all_data = []
    for date_dir in glob.glob(os.path.join(data_dir, "*")):
        if os.path.isdir(date_dir):
            for f in glob.glob(os.path.join(date_dir, "*.csv")):
                pair = os.path.basename(f).split("_")[0]
                try:
                    df = pd.read_csv(f)
                    df["pair"] = pair
                    all_data.append(df)
                except:
                    pass

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print(f"  Loaded live data: {len(combined):,} rows")
        return combined
    return pd.DataFrame()

# Load data
hist_data = load_dukascopy_data("/root/data/dukascopy_local")
live_data = load_live_data("/root/data/live")

if hist_data.empty:
    print("ERROR: No historical data found!")
    exit(1)

# ========== RENAISSANCE SIGNALS ==========
print("\n[2/6] Generating Renaissance weak signals...")

def generate_renaissance_signals(df):
    """Generate 50+ weak signals like Renaissance Technologies"""

    # Ensure we have required columns
    if "bid" in df.columns and "ask" in df.columns:
        df["mid"] = (df["bid"] + df["ask"]) / 2
        df["spread"] = df["ask"] - df["bid"]
    elif "close" in df.columns:
        df["mid"] = df["close"]
        df["spread"] = 0
    else:
        # Try to create mid from available columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df["mid"] = df[numeric_cols[0]]
            df["spread"] = 0
        else:
            print("ERROR: No price columns found")
            return df

    # TREND SIGNALS (10)
    for period in [5, 10, 20, 50]:
        df[f"sma_{period}"] = df["mid"].rolling(period).mean()
        df[f"ema_{period}"] = df["mid"].ewm(span=period).mean()

    df["trend_sma_5_20"] = (df["sma_5"] > df["sma_20"]).astype(int)
    df["trend_sma_10_50"] = (df["sma_10"] > df["sma_50"]).astype(int)
    df["trend_slope_20"] = df["sma_20"].diff(5) / df["sma_20"].shift(5)

    # MOMENTUM SIGNALS (10)
    for period in [5, 10, 20]:
        df[f"roc_{period}"] = df["mid"].pct_change(period) * 100

    df["momentum_roc_divergence"] = df["roc_5"] - df["roc_20"]
    df["momentum_acceleration"] = df["roc_5"].diff(3)

    # MACD
    ema12 = df["mid"].ewm(span=12).mean()
    ema26 = df["mid"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # MEAN REVERSION SIGNALS (10)
    for period in [10, 20, 50]:
        mean = df["mid"].rolling(period).mean()
        std = df["mid"].rolling(period).std()
        df[f"zscore_{period}"] = (df["mid"] - mean) / (std + 1e-10)

    # Bollinger Bands
    df["bb_upper"] = df["sma_20"] + 2 * df["mid"].rolling(20).std()
    df["bb_lower"] = df["sma_20"] - 2 * df["mid"].rolling(20).std()
    df["bb_pct"] = (df["mid"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)

    # RSI
    delta = df["mid"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi_signal"] = np.where(df["rsi"] < 30, 1, np.where(df["rsi"] > 70, -1, 0))

    # VOLATILITY SIGNALS (10)
    df["atr_14"] = df["mid"].diff().abs().rolling(14).mean()
    df["vol_20"] = df["mid"].rolling(20).std()
    df["vol_ratio"] = df["vol_20"] / df["vol_20"].rolling(50).mean()
    df["vol_percentile"] = df["vol_20"].rolling(100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])

    # Keltner Channel
    df["keltner_upper"] = df["ema_20"] + 2 * df["atr_14"]
    df["keltner_lower"] = df["ema_20"] - 2 * df["atr_14"]
    df["keltner_pct"] = (df["mid"] - df["keltner_lower"]) / (df["keltner_upper"] - df["keltner_lower"] + 1e-10)

    # MICROSTRUCTURE SIGNALS (10)
    df["spread_pct"] = df["spread"] / df["mid"]
    df["spread_zscore"] = (df["spread"] - df["spread"].rolling(50).mean()) / (df["spread"].rolling(50).std() + 1e-10)
    df["price_change"] = df["mid"].diff()
    df["price_change_sign"] = np.sign(df["price_change"])
    df["tick_reversal"] = (df["price_change_sign"] != df["price_change_sign"].shift(1)).astype(int)

    # Order flow proxy
    df["volume_proxy"] = df["spread"].rolling(10).mean() * df["atr_14"]
    df["imbalance"] = df["price_change"].rolling(10).sum()

    # ADDITIONAL WEAK SIGNALS (10+)
    df["high_low_ratio"] = df["mid"].rolling(20).max() / (df["mid"].rolling(20).min() + 1e-10)
    df["price_position"] = (df["mid"] - df["mid"].rolling(50).min()) / (df["mid"].rolling(50).max() - df["mid"].rolling(50).min() + 1e-10)
    df["momentum_quality"] = df["roc_10"] * (1 / (df["vol_20"] + 1e-10))

    # Correlation signals
    df["autocorr_10"] = df["mid"].rolling(50).apply(lambda x: pd.Series(x).autocorr(10) if len(x) >= 50 else 0)
    df["returns"] = df["mid"].pct_change()

    print(f"  Generated {len([c for c in df.columns if c not in ['bid', 'ask', 'mid', 'spread', 'pair', 'timestamp']])} signals")
    return df

# Apply signals to data
hist_data = generate_renaissance_signals(hist_data)

# ========== FEATURE ENGINEERING ==========
print("\n[3/6] Engineering features for HFT...")

# Target: next tick direction
hist_data["target"] = (hist_data["mid"].shift(-1) > hist_data["mid"]).astype(int)

# Drop NaN rows
hist_data = hist_data.dropna()

# Select features
exclude_cols = ["bid", "ask", "mid", "spread", "pair", "timestamp", "target", "returns", "price_change"]
feature_cols = [c for c in hist_data.columns if c not in exclude_cols and hist_data[c].dtype in [np.float64, np.int64, np.float32, np.int32]]

print(f"  Features: {len(feature_cols)}")
print(f"  Training samples: {len(hist_data):,}")

X = hist_data[feature_cols].values
y = hist_data["target"].values

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ========== MODEL TRAINING ==========
print("\n[4/6] Training ensemble models on H200...")

# Time series split for validation
tscv = TimeSeriesSplit(n_splits=3)
train_idx, val_idx = list(tscv.split(X))[-1]
X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

print(f"  Train: {len(X_train):,}, Validation: {len(X_val):,}")

models = {}
predictions = {}

# XGBoost (GPU accelerated)
print("\n  Training XGBoost (GPU)...")
t0 = time.time()
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    tree_method="hist",
    device="cuda",
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
models["xgboost"] = xgb_model
predictions["xgboost"] = xgb_model.predict(X_val)
xgb_time = time.time() - t0
xgb_acc = accuracy_score(y_val, predictions["xgboost"])
print(f"    XGBoost: {xgb_acc:.4f} accuracy ({xgb_time:.1f}s)")

# LightGBM (GPU accelerated)
print("\n  Training LightGBM (GPU)...")
t0 = time.time()
lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    device="gpu",
    random_state=42,
    verbose=-1
)
lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
models["lightgbm"] = lgb_model
predictions["lightgbm"] = lgb_model.predict(X_val)
lgb_time = time.time() - t0
lgb_acc = accuracy_score(y_val, predictions["lightgbm"])
print(f"    LightGBM: {lgb_acc:.4f} accuracy ({lgb_time:.1f}s)")

# CatBoost (GPU accelerated)
print("\n  Training CatBoost (GPU)...")
t0 = time.time()
cb_model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    task_type="GPU",
    random_seed=42,
    verbose=False
)
cb_model.fit(X_train, y_train, eval_set=(X_val, y_val))
models["catboost"] = cb_model
predictions["catboost"] = cb_model.predict(X_val)
cb_time = time.time() - t0
cb_acc = accuracy_score(y_val, predictions["catboost"])
print(f"    CatBoost: {cb_acc:.4f} accuracy ({cb_time:.1f}s)")

# ========== ENSEMBLE ==========
print("\n[5/6] Creating ensemble...")

# Weighted voting (weights based on validation accuracy)
weights = {
    "xgboost": xgb_acc,
    "lightgbm": lgb_acc,
    "catboost": cb_acc
}
total_weight = sum(weights.values())

# Ensemble prediction
ensemble_probs = np.zeros(len(X_val))
for name, preds in predictions.items():
    ensemble_probs += preds * (weights[name] / total_weight)

ensemble_preds = (ensemble_probs >= 0.5).astype(int)
ensemble_acc = accuracy_score(y_val, ensemble_preds)

print(f"\n  Ensemble Accuracy: {ensemble_acc:.4f}")
print(f"  Precision: {precision_score(y_val, ensemble_preds):.4f}")
print(f"  Recall: {recall_score(y_val, ensemble_preds):.4f}")

# ========== SAVE MODELS ==========
print("\n[6/6] Saving models...")

os.makedirs("/root/models", exist_ok=True)

# Save all models
for name, model in models.items():
    path = f"/root/models/hft_{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved: {path}")

# Save scaler
with open("/root/models/hft_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save feature list
with open("/root/models/hft_features.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

# Save weights
with open("/root/models/hft_weights.pkl", "wb") as f:
    pickle.dump(weights, f)

# ========== SUMMARY ==========
total_time = time.time() - start_time
print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print(f"Total time: {total_time:.1f}s")
print(f"\nModel Performance:")
print(f"  XGBoost:   {xgb_acc:.4f}")
print(f"  LightGBM:  {lgb_acc:.4f}")
print(f"  CatBoost:  {cb_acc:.4f}")
print(f"  Ensemble:  {ensemble_acc:.4f}")
print(f"\nModels saved to: /root/models/")
print(f"  - hft_xgboost.pkl")
print(f"  - hft_lightgbm.pkl")
print(f"  - hft_catboost.pkl")
print(f"  - hft_scaler.pkl")
print(f"  - hft_features.pkl")
print(f"  - hft_weights.pkl")
print("\nReady for live trading deployment!")
