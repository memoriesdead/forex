#!/usr/bin/env python3
"""
Train ALL Forex Pairs with 575+ Mega Features
==============================================
Genesis to Now: 2003-2026 (22+ years of data)
75 pairs from Hostinger/Dukascopy data

Usage:
    # Train all pairs (full run)
    python scripts/train_all_mega.py

    # Train specific tier
    python scripts/train_all_mega.py --tier majors

    # Train specific pairs
    python scripts/train_all_mega.py --symbols EURUSD,GBPUSD,USDJPY

    # Prepare data only (no training)
    python scripts/train_all_mega.py --prepare-only

    # Resume from specific pair
    python scripts/train_all_mega.py --resume-from GBPUSD
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
import gc

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Symbol tiers
SYMBOL_TIERS = {
    'majors': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD'],
    'crosses': ['EURJPY', 'GBPJPY', 'EURGBP', 'EURCHF', 'AUDJPY', 'EURAUD', 'GBPAUD'],
    'eur_crosses': ['EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 'EURJPY', 'EURNOK', 'EURNZD', 'EURPLN', 'EURSEK'],
    'gbp_crosses': ['GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPJPY', 'GBPNOK', 'GBPNZD', 'GBPSEK', 'GBPSGD'],
    'jpy_crosses': ['AUDJPY', 'CADJPY', 'CHFJPY', 'EURJPY', 'GBPJPY', 'NZDJPY', 'SGDJPY', 'USDJPY'],
}


def load_pair_data(pair: str, data_dir: Path, max_files: int = None, sample_rate: int = 10) -> pd.DataFrame:
    """
    Load and concatenate all parquet files for a pair.

    Args:
        pair: Currency pair symbol
        data_dir: Path to hostinger data directory
        max_files: Maximum number of files to load (None = all)
        sample_rate: Sample every Nth tick to reduce size

    Returns:
        DataFrame with tick data
    """
    pair_dir = data_dir / pair
    if not pair_dir.exists():
        raise ValueError(f"No data found for {pair}")

    files = sorted(pair_dir.glob('*.parquet'))
    if max_files:
        files = files[-max_files:]  # Most recent files

    print(f"  Loading {len(files)} files for {pair}...")

    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            if sample_rate > 1:
                df = df.iloc[::sample_rate]  # Sample every Nth row
            dfs.append(df)
        except Exception as e:
            print(f"    Warning: Failed to load {f.name}: {e}")

    if not dfs:
        raise ValueError(f"No valid data loaded for {pair}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(combined):,} ticks for {pair}")

    return combined


def create_targets(df: pd.DataFrame, horizons: list = [1, 5, 10, 20, 50]) -> pd.DataFrame:
    """Create target variables for different horizons."""
    targets = pd.DataFrame(index=df.index)

    mid = df['mid']

    for h in horizons:
        # Future return
        future_ret = mid.shift(-h) / mid - 1
        targets[f'target_return_{h}'] = future_ret

        # Direction (binary)
        targets[f'target_direction_{h}'] = (future_ret > 0).astype(int)

    return targets


def prepare_training_data(
    pair: str,
    data_dir: Path,
    output_dir: Path,
    max_files: int = 500,  # ~2 years of daily data
    sample_rate: int = 100,  # Sample every 100th tick
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> bool:
    """
    Prepare training data for a single pair.

    Returns True if successful, False otherwise.
    """
    try:
        print(f"\nPreparing {pair}...")

        # Load data
        df = load_pair_data(pair, data_dir, max_files=max_files, sample_rate=sample_rate)

        if len(df) < 10000:
            print(f"  Skipping {pair}: Not enough data ({len(df)} rows)")
            return False

        # Ensure required columns
        if 'mid' not in df.columns:
            if 'bid' in df.columns and 'ask' in df.columns:
                df['mid'] = (df['bid'] + df['ask']) / 2
            else:
                print(f"  Skipping {pair}: No price data")
                return False

        # Set close for feature generation
        df['close'] = df['mid']

        # Create volume if missing
        if 'volume' not in df.columns:
            if 'bid_volume' in df.columns and 'ask_volume' in df.columns:
                df['volume'] = df['bid_volume'] + df['ask_volume']
            else:
                df['volume'] = 1

        # Create basic features (fast version for data prep)
        print(f"  Creating base features...")
        df['ret_1'] = df['mid'].pct_change()
        df['ret_5'] = df['mid'].pct_change(5)
        df['ret_10'] = df['mid'].pct_change(10)
        df['ret_20'] = df['mid'].pct_change(20)
        df['vol_20'] = df['ret_1'].rolling(20).std()
        df['vol_50'] = df['ret_1'].rolling(50).std()

        # Create targets
        print(f"  Creating targets...")
        targets = create_targets(df)
        df = pd.concat([df, targets], axis=1)

        # Drop NaN rows
        df = df.dropna()

        if len(df) < 5000:
            print(f"  Skipping {pair}: Not enough valid data ({len(df)} rows)")
            return False

        # Split data
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        print(f"  Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

        # Save
        pair_output_dir = output_dir / pair
        pair_output_dir.mkdir(parents=True, exist_ok=True)

        train_df.to_parquet(pair_output_dir / 'train.parquet')
        val_df.to_parquet(pair_output_dir / 'val.parquet')
        test_df.to_parquet(pair_output_dir / 'test.parquet')

        # Feature and target columns
        base_features = ['ret_1', 'ret_5', 'ret_10', 'ret_20', 'vol_20', 'vol_50']
        target_cols = [c for c in df.columns if c.startswith('target_')]

        # Metadata
        metadata = {
            'pair': pair,
            'features': base_features,
            'targets': target_cols,
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'prepared_at': datetime.now().isoformat(),
            'source_files': max_files,
            'sample_rate': sample_rate,
        }

        with open(pair_output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  Saved to {pair_output_dir}")

        # Clean up memory
        del df, train_df, val_df, test_df
        gc.collect()

        return True

    except Exception as e:
        print(f"  Error preparing {pair}: {e}")
        return False


def train_pair_mega(
    pair: str,
    training_dir: Path,
    output_dir: Path,
    max_samples: int = 50000
) -> dict:
    """
    Train a single pair with mega features.

    Returns dict with results or None if failed.
    """
    try:
        pair_dir = training_dir / pair
        if not (pair_dir / 'train.parquet').exists():
            print(f"  No training data for {pair}")
            return None

        print(f"\n{'='*60}")
        print(f"Training {pair} with Mega Features")
        print(f"{'='*60}")

        # Load data
        train = pd.read_parquet(pair_dir / 'train.parquet')
        val = pd.read_parquet(pair_dir / 'val.parquet')
        test = pd.read_parquet(pair_dir / 'test.parquet')

        with open(pair_dir / 'metadata.json') as f:
            metadata = json.load(f)

        # Set close from mid
        for df in [train, val, test]:
            if 'close' not in df.columns and 'mid' in df.columns:
                df['close'] = df['mid']

        # Generate mega features
        print(f"  Generating mega features...")
        from core.features.mega_generator import MegaFeatureGenerator

        generator = MegaFeatureGenerator(verbose=False)

        train_features = generator.generate_all(train)
        val_features = generator.generate_all(val)
        test_features = generator.generate_all(test)

        # Filter to numeric only
        numeric_cols = train_features.select_dtypes(include=[np.number]).columns.tolist()
        train_features = train_features[numeric_cols]
        val_features = val_features[numeric_cols]
        test_features = test_features[numeric_cols]

        feature_names = numeric_cols
        print(f"  Generated {len(feature_names)} features")

        # Get targets
        target_cols = [c for c in train.columns if c.startswith('target_direction')]

        X_train = train_features.values
        X_val = val_features.values
        X_test = test_features.values

        results = {}

        # Import training functions
        import xgboost as xgb
        import lightgbm as lgb
        import catboost as cb
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

        for target in target_cols[:3]:  # Top 3 horizons
            print(f"\n  Target: {target}")

            y_train = train[target].values
            y_val = val[target].values
            y_test = test[target].values

            # Subsample if needed
            if len(X_train) > max_samples:
                idx = np.random.choice(len(X_train), max_samples, replace=False)
                X_train_sub = X_train[idx]
                y_train_sub = y_train[idx]
            else:
                X_train_sub = X_train
                y_train_sub = y_train

            # Train XGBoost
            xgb_params = {
                'tree_method': 'hist',
                'device': 'cuda',
                'max_depth': 8,
                'max_bin': 128,
                'learning_rate': 0.05,
                'subsample': 0.7,
                'colsample_bytree': 0.6,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
            }

            dtrain = xgb.DMatrix(X_train_sub, label=y_train_sub)
            dval = xgb.DMatrix(X_val, label=y_val)

            xgb_model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=500,
                evals=[(dval, 'val')],
                early_stopping_rounds=30,
                verbose_eval=False
            )

            # Train LightGBM
            lgb_params = {
                'device': 'gpu',
                'max_depth': 8,
                'num_leaves': 127,
                'feature_fraction': 0.6,
                'bagging_fraction': 0.7,
                'bagging_freq': 5,
                'verbose': -1,
                'objective': 'binary',
                'metric': 'auc',
            }

            train_data = lgb.Dataset(X_train_sub, label=y_train_sub)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            lgb_model = lgb.train(
                lgb_params,
                train_data,
                num_boost_round=500,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(30, verbose=False)]
            )

            # Train CatBoost
            cb_model = cb.CatBoostClassifier(
                task_type='GPU',
                devices='0',
                depth=6,
                iterations=300,
                early_stopping_rounds=20,
                verbose=False,
                loss_function='Logloss',
            )
            cb_model.fit(X_train_sub, y_train_sub, eval_set=(X_val, y_val))

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

            print(f"    Accuracy: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")

            results[target] = {
                'accuracy': float(acc),
                'auc': float(auc),
                'f1': float(f1)
            }

        # Save models and results
        model_file = output_dir / f"{pair}_models.pkl"
        results_file = output_dir / f"{pair}_results.json"

        # Save results
        results['_meta'] = {
            'feature_mode': 'mega',
            'feature_count': len(feature_names),
            'timestamp': datetime.now().isoformat()
        }

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"  Saved results to {results_file}")

        # Clean up
        del train, val, test, train_features, val_features, test_features
        gc.collect()

        return results

    except Exception as e:
        print(f"  Error training {pair}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Train all forex pairs with mega features')
    parser.add_argument('--tier', choices=list(SYMBOL_TIERS.keys()), help='Train specific tier')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols')
    parser.add_argument('--prepare-only', action='store_true', help='Only prepare data')
    parser.add_argument('--train-only', action='store_true', help='Only train (data must exist)')
    parser.add_argument('--resume-from', type=str, help='Resume from specific symbol')
    parser.add_argument('--max-files', type=int, default=500, help='Max files per pair')
    parser.add_argument('--sample-rate', type=int, default=100, help='Sample every Nth tick')
    args = parser.parse_args()

    # Paths
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / 'data' / 'hostinger'
    training_dir = project_dir / 'training_package'
    output_dir = project_dir / 'models' / 'production'

    training_dir.mkdir(exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of pairs to process
    if args.symbols:
        pairs = [s.strip().upper() for s in args.symbols.split(',')]
    elif args.tier:
        pairs = SYMBOL_TIERS[args.tier]
    else:
        # All available pairs
        pairs = sorted([d.name for d in data_dir.iterdir()
                       if d.is_dir() and d.name != 'old' and list(d.glob('*.parquet'))])

    print(f"Processing {len(pairs)} pairs")
    print(f"Data dir: {data_dir}")
    print(f"Training dir: {training_dir}")
    print(f"Output dir: {output_dir}")

    # Handle resume
    if args.resume_from:
        try:
            start_idx = pairs.index(args.resume_from.upper())
            pairs = pairs[start_idx:]
            print(f"Resuming from {args.resume_from}, {len(pairs)} pairs remaining")
        except ValueError:
            print(f"Warning: {args.resume_from} not found in pairs list")

    total_start = time.time()
    results_summary = {}

    for i, pair in enumerate(pairs):
        pair_start = time.time()
        print(f"\n[{i+1}/{len(pairs)}] Processing {pair}")

        # Prepare data
        if not args.train_only:
            success = prepare_training_data(
                pair, data_dir, training_dir,
                max_files=args.max_files,
                sample_rate=args.sample_rate
            )
            if not success:
                continue

        # Train
        if not args.prepare_only:
            result = train_pair_mega(pair, training_dir, output_dir)
            if result:
                results_summary[pair] = result

        pair_time = time.time() - pair_start
        print(f"  {pair} completed in {pair_time/60:.1f} minutes")

    total_time = time.time() - total_start

    # Summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Pairs processed: {len(results_summary)}")

    if results_summary:
        print("\nResults Summary:")
        for pair, results in results_summary.items():
            for target, metrics in results.items():
                if not target.startswith('_'):
                    print(f"  {pair} {target}: ACC={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}")

    # Save summary
    summary_file = output_dir / 'training_summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'pairs_trained': list(results_summary.keys()),
            'total_time_minutes': total_time / 60,
            'results': results_summary
        }, f, indent=2)

    print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    main()
