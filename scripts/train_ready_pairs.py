#!/usr/bin/env python3
"""
Train pairs that already have features cached.
Skips feature generation, goes straight to GPU training.
"""

import os
import sys
import gc
import time
import json
import pickle
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# CONFIGURATION
# ============================================================================

TRAINING_DIR = PROJECT_ROOT / 'training_package'
OUTPUT_DIR = PROJECT_ROOT / 'models' / 'production'
FEATURES_DIR = TRAINING_DIR / 'features_cache'
MIN_ACCURACY = 0.55

# GPU Settings - optimized for RTX 5080 16GB
XGB_PARAMS = {
    'tree_method': 'hist',
    'device': 'cuda',
    'max_depth': 10,
    'max_bin': 256,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
}

LGB_PARAMS = {
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'max_depth': 10,
    'num_leaves': 255,
    'learning_rate': 0.03,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'objective': 'binary',
    'metric': 'auc',
    'n_jobs': 8,
}

CB_PARAMS = {
    'task_type': 'GPU',
    'devices': '0',
    'depth': 8,
    'iterations': 500,
    'learning_rate': 0.03,
    'early_stopping_rounds': 30,
    'verbose': False,
    'loss_function': 'Logloss',
    'thread_count': 8,
}


def get_gpu_temp():
    """Get GPU temperature."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        return int(result.stdout.strip())
    except:
        return 0


def configure_gpu():
    """Configure GPU for max performance."""
    print("Configuring GPU...")
    os.environ['OMP_NUM_THREADS'] = '12'
    os.environ['MKL_NUM_THREADS'] = '12'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

    try:
        import torch
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except:
        pass

    print(f"GPU Temperature: {get_gpu_temp()}°C")
    print()


def get_pairs_needing_training():
    """Get pairs that need training (have features, low accuracy)."""
    pairs_to_train = []

    # Check each feature file
    for f in FEATURES_DIR.glob('*_features.pkl'):
        if 'mega' in f.name:
            continue

        pair = f.stem.replace('_features', '')
        results_path = OUTPUT_DIR / f"{pair}_results.json"

        needs_training = True

        if results_path.exists():
            try:
                with open(results_path) as fp:
                    data = json.load(fp)

                features = data.get('_meta', {}).get('feature_count', 0)
                if features > 0:
                    acc = 0
                    for t in ['target_direction_10', 'target_direction_5', 'target_direction_1']:
                        if t in data and 'accuracy' in data[t]:
                            acc = data[t]['accuracy']
                            break

                    if acc >= MIN_ACCURACY:
                        needs_training = False
            except:
                pass

        if needs_training:
            pairs_to_train.append((pair, str(f)))

    return pairs_to_train


def train_pair(pair: str, features_path: str) -> dict:
    """Train a single pair with XGB+LGB+CB ensemble."""
    import xgboost as xgb
    import lightgbm as lgb
    import catboost as cb
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

    print(f"[{pair}] Loading features...")

    with open(features_path, 'rb') as f:
        data = pickle.load(f)

    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    feature_names = data['feature_names']
    target_cols = data['target_cols']

    print(f"[{pair}] Training {len(X_train)} samples, {len(feature_names)} features")

    results = {}
    models = {}

    for target in target_cols[:3]:
        print(f"[{pair}] Training {target}...")

        y_tr = y_train[target]
        y_v = y_val[target]
        y_te = y_test[target]

        # XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_v)

        xgb_model = xgb.train(
            XGB_PARAMS, dtrain,
            num_boost_round=700,
            evals=[(dval, 'val')],
            early_stopping_rounds=40,
            verbose_eval=False
        )

        # LightGBM
        train_data = lgb.Dataset(X_train, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_v, reference=train_data)

        lgb_model = lgb.train(
            LGB_PARAMS, train_data,
            num_boost_round=700,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(40, verbose=False)]
        )

        # CatBoost
        cb_model = cb.CatBoostClassifier(**CB_PARAMS)
        cb_model.fit(X_train, y_tr, eval_set=(X_val, y_v))

        # Ensemble predictions
        dtest = xgb.DMatrix(X_test)
        xgb_pred = xgb_model.predict(dtest)
        lgb_pred = lgb_model.predict(X_test)
        cb_pred = cb_model.predict_proba(X_test)[:, 1]

        ensemble_pred = (xgb_pred + lgb_pred + cb_pred) / 3
        ensemble_class = (ensemble_pred > 0.5).astype(int)

        acc = accuracy_score(y_te, ensemble_class)
        auc = roc_auc_score(y_te, ensemble_pred)
        f1 = f1_score(y_te, ensemble_class)

        print(f"[{pair}] {target}: ACC={acc:.4f}, AUC={auc:.4f}, F1={f1:.4f}")

        results[target] = {'accuracy': float(acc), 'auc': float(auc), 'f1': float(f1)}
        models[target] = {'xgb': xgb_model, 'lgb': lgb_model, 'cb': cb_model}

        del dtrain, dval, dtest

    # Cleanup GPU memory
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass

    # Save models
    model_path = OUTPUT_DIR / f"{pair}_models.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(models, f)

    # Save results
    results['_meta'] = {
        'feature_mode': 'mega',
        'feature_count': len(feature_names),
        'timestamp': datetime.now().isoformat()
    }

    results_path = OUTPUT_DIR / f"{pair}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[{pair}] COMPLETE - Saved to {model_path}")

    return results


def main():
    configure_gpu()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pairs = get_pairs_needing_training()

    print(f"{'='*60}")
    print(f"TRAINING {len(pairs)} PAIRS")
    print(f"{'='*60}")

    if not pairs:
        print("All pairs already trained with good accuracy!")
        return

    for pair, _ in pairs:
        print(f"  - {pair}")
    print()

    start_time = time.time()
    results = {}

    for i, (pair, features_path) in enumerate(pairs):
        print(f"\n[{i+1}/{len(pairs)}] Training {pair}")
        print(f"GPU Temp: {get_gpu_temp()}°C")

        # Safety check
        temp = get_gpu_temp()
        if temp > 80:
            print(f"WARNING: GPU hot ({temp}°C) - waiting 30s to cool")
            time.sleep(30)

        try:
            result = train_pair(pair, features_path)
            results[pair] = result

            elapsed = time.time() - start_time
            rate = (i + 1) / (elapsed / 60)
            remaining = (len(pairs) - i - 1) / rate if rate > 0 else 0

            print(f"\nProgress: {i+1}/{len(pairs)} ({rate:.1f} pairs/min, ETA: {remaining:.0f} min)")

        except Exception as e:
            print(f"[{pair}] ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    total_time = time.time() - start_time

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Pairs trained: {len(results)}")

    if results:
        accs = []
        for pair, res in results.items():
            for target, metrics in res.items():
                if not target.startswith('_') and 'accuracy' in metrics:
                    accs.append(metrics['accuracy'])

        if accs:
            print(f"Average accuracy: {np.mean(accs)*100:.2f}%")


if __name__ == '__main__':
    main()
